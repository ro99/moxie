// Routed Gemma operations for the selected dense graph image.
// Every FP32 arithmetic boundary is explicit so nvcc cannot contract a
// reduction into an FMA or change its declared rounding order.

static __device__ __forceinline__ float moxie_dense_route_logit_v1(
    const __nv_bfloat16* x, const __nv_bfloat16* proj,
    const __nv_bfloat16* gain, unsigned long long hidden,
    unsigned long long expert, float eps, float input_scale) {
    float sum = 0.0F;
    for (unsigned long long k = 0; k < hidden; ++k) {
        const float value = __bfloat162float(x[k]);
        sum = __fadd_rn(sum, __fmul_rn(value, value));
    }
    const float mean = __fdiv_rn(sum, static_cast<float>(hidden));
    const float denom = __fsqrt_rn(__fadd_rn(mean, eps));

    float logit = 0.0F;
    for (unsigned long long k = 0; k < hidden; ++k) {
        const float value = __bfloat162float(x[k]);
        const float normalized = __bfloat162float(
            __float2bfloat16_rn(__fdiv_rn(value, denom)));
        const float gained = __bfloat162float(
            __float2bfloat16_rn(__fmul_rn(normalized, __bfloat162float(gain[k]))));
        const float scaled = __bfloat162float(
            __float2bfloat16_rn(__fmul_rn(gained, input_scale)));
        logit = __fadd_rn(
            logit,
            __fmul_rn(scaled, __bfloat162float(proj[expert * hidden + k])));
    }
    return __bfloat162float(__float2bfloat16_rn(logit));
}

static __device__ __forceinline__ float moxie_dense_route_exp_v1(
    float logit, float max_logit) {
    // The subtraction itself is FP32; exp is deliberately evaluated in FP64
    // and narrowed once, matching task 0059's softmax precedent.
    const float delta = __fadd_rn(logit, -max_logit);
    return static_cast<float>(exp(static_cast<double>(delta)));
}

extern "C" __global__ void moxie_dense_route_v1(
    const __nv_bfloat16* x, const __nv_bfloat16* proj,
    const __nv_bfloat16* gain, const __nv_bfloat16* per_expert,
    unsigned int* ids, float* coefficients,
    unsigned long long rows, unsigned long long hidden,
    unsigned long long experts, unsigned long long top_k,
    float eps, float input_scale) {
    const unsigned long long row =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows || hidden == 0 || experts == 0 || top_k == 0 || top_k > experts) return;

    const __nv_bfloat16* x_row = x + row * hidden;
    unsigned int* row_ids = ids + row * top_k;
    float* row_coefficients = coefficients + row * top_k;

    // ponytail: recompute logits for each selection pass, O(top_k * experts *
    // hidden) per row; an admitted rows-by-experts workspace is the M6 upgrade.
    for (unsigned long long pick = 0; pick < top_k; ++pick) {
        float max_logit = -INFINITY;
        for (unsigned long long expert = 0; expert < experts; ++expert) {
            max_logit = fmaxf(
                max_logit,
                moxie_dense_route_logit_v1(
                    x_row, proj, gain, hidden, expert, eps, input_scale));
        }
        float total = 0.0F;
        for (unsigned long long expert = 0; expert < experts; ++expert) {
            const float logit = moxie_dense_route_logit_v1(
                x_row, proj, gain, hidden, expert, eps, input_scale);
            total = __fadd_rn(total, moxie_dense_route_exp_v1(logit, max_logit));
        }

        unsigned int best_expert = 0xffffffffU;
        float best_probability = -INFINITY;
        for (unsigned long long expert = 0; expert < experts; ++expert) {
            bool already_picked = false;
            for (unsigned long long previous = 0; previous < pick; ++previous) {
                already_picked |= row_ids[previous] == expert;
            }
            if (already_picked) continue;
            const float logit = moxie_dense_route_logit_v1(
                x_row, proj, gain, hidden, expert, eps, input_scale);
            const float probability = __fdiv_rn(
                moxie_dense_route_exp_v1(logit, max_logit), total);
            if (best_expert == 0xffffffffU || probability > best_probability
                || (probability == best_probability && expert < best_expert)) {
                best_expert = static_cast<unsigned int>(expert);
                best_probability = probability;
            }
        }
        row_ids[pick] = best_expert;
    }

    float max_logit = -INFINITY;
    for (unsigned long long expert = 0; expert < experts; ++expert) {
        max_logit = fmaxf(
            max_logit,
            moxie_dense_route_logit_v1(
                x_row, proj, gain, hidden, expert, eps, input_scale));
    }
    float total = 0.0F;
    for (unsigned long long expert = 0; expert < experts; ++expert) {
        const float logit = moxie_dense_route_logit_v1(
            x_row, proj, gain, hidden, expert, eps, input_scale);
        total = __fadd_rn(total, moxie_dense_route_exp_v1(logit, max_logit));
    }
    float mass = 0.0F;
    for (unsigned long long pick = 0; pick < top_k; ++pick) {
        const float logit = moxie_dense_route_logit_v1(
            x_row, proj, gain, hidden, row_ids[pick], eps, input_scale);
        const float probability = __fdiv_rn(
            moxie_dense_route_exp_v1(logit, max_logit), total);
        mass = __fadd_rn(mass, probability);
    }
    for (unsigned long long pick = 0; pick < top_k; ++pick) {
        const unsigned int expert = row_ids[pick];
        const float logit = moxie_dense_route_logit_v1(
            x_row, proj, gain, hidden, expert, eps, input_scale);
        const float probability = __fdiv_rn(
            moxie_dense_route_exp_v1(logit, max_logit), total);
        row_coefficients[pick] = __fmul_rn(
            __fdiv_rn(probability, mass), __bfloat162float(per_expert[expert]));
    }
}

extern "C" __global__ void moxie_dense_expert_project_gelu_v1(
    const __nv_bfloat16* x, const unsigned int* ids,
    const __nv_bfloat16* gate_up, float* activated,
    unsigned long long assignments, unsigned long long top_k,
    unsigned long long hidden, unsigned long long intermediate) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= assignments * intermediate || top_k == 0) return;
    const unsigned long long slot = index / intermediate;
    const unsigned long long lane = index % intermediate;
    const __nv_bfloat16* x_row = x + (slot / top_k) * hidden;
    const __nv_bfloat16* expert_gate_up =
        gate_up + static_cast<unsigned long long>(ids[slot]) * 2 * intermediate * hidden;
    float gate = 0.0F;
    float up = 0.0F;
    moxie_expert_lanes_v1(
        x_row, expert_gate_up, hidden, intermediate, lane, &gate, &up);
    const double gated = static_cast<double>(__bfloat162float(
        __float2bfloat16_rn(moxie_gelu_tanh_v1(gate))));
    const float h = static_cast<float>(__dmul_rn(gated, static_cast<double>(up)));
    activated[index] = __bfloat162float(__float2bfloat16_rn(h));
}

extern "C" __global__ void moxie_dense_expert_down_v1(
    const float* activated, const unsigned int* ids,
    const __nv_bfloat16* down, __nv_bfloat16* slots,
    unsigned long long assignments, unsigned long long hidden,
    unsigned long long intermediate) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= assignments * hidden) return;
    const unsigned long long slot = index / hidden;
    const unsigned long long component = index % hidden;
    const float* h = activated + slot * intermediate;
    const __nv_bfloat16* row =
        down + static_cast<unsigned long long>(ids[slot]) * hidden * intermediate
        + component * intermediate;
    float acc = 0.0F;
    for (unsigned long long i = 0; i < intermediate; ++i) {
        acc = __fadd_rn(acc, __fmul_rn(h[i], __bfloat162float(row[i])));
    }
    slots[slot * hidden + component] = __float2bfloat16_rn(acc);
}

extern "C" __global__ void moxie_dense_combine_v1(
    const unsigned int* ids, const float* coefficients,
    const __nv_bfloat16* slots, __nv_bfloat16* out,
    unsigned long long rows, unsigned long long top_k,
    unsigned long long hidden, float output_scale) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * hidden || top_k == 0) return;
    const unsigned long long row = index / hidden;
    const unsigned long long component = index % hidden;
    const unsigned int* row_ids = ids + row * top_k;
    const float* row_coefficients = coefficients + row * top_k;
    float acc = 0.0F;
    bool has_previous = false;
    unsigned int previous_id = 0;
    unsigned long long previous_slot = 0;
    for (unsigned long long visit = 0; visit < top_k; ++visit) {
        bool found = false;
        unsigned int best_id = 0xffffffffU;
        unsigned long long best_slot = 0;
        for (unsigned long long slot = 0; slot < top_k; ++slot) {
            const unsigned int expert = row_ids[slot];
            const bool after_previous = !has_previous || expert > previous_id
                || (expert == previous_id && slot > previous_slot);
            if (after_previous
                && (!found || expert < best_id
                    || (expert == best_id && slot < best_slot))) {
                found = true;
                best_id = expert;
                best_slot = slot;
            }
        }
        const unsigned long long slot_index = row * top_k + best_slot;
        acc = __fadd_rn(
            acc,
            __fmul_rn(
                row_coefficients[best_slot],
                __bfloat162float(slots[slot_index * hidden + component])));
        previous_id = best_id;
        previous_slot = best_slot;
        has_previous = true;
    }
    if (output_scale != 1.0F) acc = __fmul_rn(acc, output_scale);
    out[index] = __float2bfloat16_rn(acc);
}
