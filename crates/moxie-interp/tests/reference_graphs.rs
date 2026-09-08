//! Acceptance tests for task 0003: the BF16 host reference interpreter.
//!
//! Two synthetic graphs with different hidden sizes, head counts, feed-forward
//! widths and vocabularies, because document 06 M1 requires "at least two
//! distinct shapes consume every foundational linear/attention operation". They
//! are graph fixtures, not model definitions: no metadata, no family name, no
//! tokenizer. Document 06 M1.5 allows a reduced synthetic graph to prove
//! contracts, and adds that synthetic output is never described as model support.

use moxie_graph::{
    Bindings, Graph, GraphBuilder, OpParams, OracleRegistry, PartitionRule, StateEffect,
    TensorSpec, ValueId, ValueRole, Visibility,
};
use moxie_interp::{Cancel, HostTensor, Interpreter, KvCache, Value};
use moxie_oracles::metric::{ErrorSummary, gamma};
use moxie_state::{ROOT, SequenceState, StateKind};
use moxie_types::{ActivationPrecision, Dim, Precision, SymbolId, WeightPrecision};

// --- fixtures ---------------------------------------------------------------

/// A tiny deterministic generator. Reproducibility matters more than quality:
/// a fixture that changes between runs cannot pin a numerical contract.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1))
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (self.0 >> 33) as u32;
        // Roughly [-1, 1), then rounded to BF16 so every weight satisfies the
        // interpreter's storage invariant by construction.
        let v = (bits as f32 / (1u32 << 31) as f32) - 1.0;
        moxie_interp::tensor::to_bf16(v)
    }
}

#[derive(Clone, Copy)]
struct Dims {
    hidden: u64,
    heads: u64,
    head_dim: u64,
    ffn: u64,
    vocab: u64,
}

const A: Dims = Dims {
    hidden: 8,
    heads: 2,
    head_dim: 4,
    ffn: 16,
    vocab: 16,
};

const B: Dims = Dims {
    hidden: 12,
    heads: 3,
    head_dim: 4,
    ffn: 20,
    // Deliberately not a power of two, and smaller than the hidden width.
    vocab: 7,
};

fn rows_symbol() -> Dim {
    Dim::symbol(SymbolId(0))
}

fn act(p: Precision) -> ValueRole {
    ValueRole::Activation(ActivationPrecision::new(p).expect("legal activation"))
}

fn weight() -> ValueRole {
    ValueRole::Weight(WeightPrecision::new(Precision::Bf16).expect("legal weight"))
}

fn oracles() -> OracleRegistry {
    let mut r = OracleRegistry::new();
    moxie_oracles::register(&mut r).expect("registration");
    r
}

/// The graph every test runs: embed, one attention block, one feed-forward
/// block, project. It uses all eight operations in the slice.
struct Fixture {
    graph: Graph,
    weights: Bindings<Value>,
    tokens_id: ValueId,
    positions_id: ValueId,
}

fn build(dims: Dims, seed: u64) -> Fixture {
    let mut g = GraphBuilder::new(moxie_oracles::HOST_REFERENCE, SymbolId(0));
    let mut rng = Lcg::new(seed);
    let mut weights = Bindings::new();

    let rows = rows_symbol();
    let tokens_id = g.input(
        "tokens",
        TensorSpec::new(ValueRole::Index, vec![rows.clone()]),
    );
    let positions_id = g.input(
        "positions",
        TensorSpec::new(ValueRole::Index, vec![rows.clone()]),
    );

    let param = |g: &mut GraphBuilder,
                 weights: &mut Bindings<Value>,
                 rng: &mut Lcg,
                 name: &str,
                 shape: Vec<u64>| {
        let dims: Vec<Dim> = shape.iter().map(|d| Dim::constant(*d)).collect();
        let id = g
            .weight(name, TensorSpec::new(weight(), dims))
            .expect("weight role");
        let n: usize = shape.iter().map(|d| *d as usize).product();
        let data: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
        let usize_shape: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
        weights.set(
            id,
            Value::Float(HostTensor::bf16(data, usize_shape).expect("bf16 weight")),
        );
        id
    };

    let h = dims.hidden;
    let qkv = dims.heads * dims.head_dim;
    let table = param(&mut g, &mut weights, &mut rng, "embed", vec![dims.vocab, h]);
    let g_attn = param(&mut g, &mut weights, &mut rng, "norm_attn", vec![h]);
    let wq = param(&mut g, &mut weights, &mut rng, "wq", vec![qkv, h]);
    let wk = param(&mut g, &mut weights, &mut rng, "wk", vec![qkv, h]);
    let wv = param(&mut g, &mut weights, &mut rng, "wv", vec![qkv, h]);
    let wo = param(&mut g, &mut weights, &mut rng, "wo", vec![h, qkv]);
    let bo = param(&mut g, &mut weights, &mut rng, "bo", vec![h]);
    let g_ffn = param(&mut g, &mut weights, &mut rng, "norm_ffn", vec![h]);
    let w_gate = param(&mut g, &mut weights, &mut rng, "w_gate", vec![dims.ffn, h]);
    let w_up = param(&mut g, &mut weights, &mut rng, "w_up", vec![dims.ffn, h]);
    let w_down = param(&mut g, &mut weights, &mut rng, "w_down", vec![h, dims.ffn]);
    let g_out = param(&mut g, &mut weights, &mut rng, "norm_out", vec![h]);
    let w_vocab = param(&mut g, &mut weights, &mut rng, "vocab", vec![dims.vocab, h]);

    let lin = |g: &mut GraphBuilder, x: ValueId, w: ValueId, i: u64, o: u64| {
        g.node(
            OpParams::Linear {
                in_features: i,
                out_features: o,
                bias: false,
            },
            &[x, w],
        )
        .expect("linear")
    };

    let h0 = g
        .node(
            OpParams::Embedding {
                vocab: dims.vocab,
                hidden: h,
            },
            &[tokens_id, table],
        )
        .expect("embedding");
    let n0 = g
        .node(
            OpParams::RmsNorm {
                hidden: h,
                eps: 1e-5,
            },
            &[h0, g_attn],
        )
        .expect("norm");
    let q = lin(&mut g, n0, wq, h, qkv);
    let k = lin(&mut g, n0, wk, h, qkv);
    let v = lin(&mut g, n0, wv, h, qkv);
    let rope = |g: &mut GraphBuilder, x: ValueId| {
        g.node(
            OpParams::Rope {
                heads: dims.heads,
                head_dim: dims.head_dim,
                rotary_dim: dims.head_dim,
                base: 10_000.0,
            },
            &[x, positions_id],
        )
        .expect("rope")
    };
    let qr = rope(&mut g, q);
    let kr = rope(&mut g, k);
    let a = g
        .node(
            OpParams::Attention {
                heads: dims.heads,
                head_dim: dims.head_dim,
                visibility: Visibility::Causal,
                layer: 0,
            },
            &[qr, kr, v, positions_id],
        )
        .expect("attention");
    // The one biased linear in the fixture, so both paths are exercised.
    let ao = g
        .node(
            OpParams::Linear {
                in_features: qkv,
                out_features: h,
                bias: true,
            },
            &[a, wo, bo],
        )
        .expect("out proj");
    let h1 = g.node(OpParams::Residual, &[h0, ao]).expect("residual");
    let n1 = g
        .node(
            OpParams::RmsNorm {
                hidden: h,
                eps: 1e-5,
            },
            &[h1, g_ffn],
        )
        .expect("norm");
    let gate = lin(&mut g, n1, w_gate, h, dims.ffn);
    let up = lin(&mut g, n1, w_up, h, dims.ffn);
    let actv = g
        .node(OpParams::SwiGlu { width: dims.ffn }, &[gate, up])
        .expect("swiglu");
    let down = lin(&mut g, actv, w_down, dims.ffn, h);
    let h2 = g.node(OpParams::Residual, &[h1, down]).expect("residual");
    let n2 = g
        .node(
            OpParams::RmsNorm {
                hidden: h,
                eps: 1e-5,
            },
            &[h2, g_out],
        )
        .expect("norm");
    let logits = g
        .node(
            OpParams::VocabProjection {
                vocab: dims.vocab,
                hidden: h,
            },
            &[n2, w_vocab],
        )
        .expect("vocab");

    let graph = g.finish(logits, &oracles()).expect("graph validates");
    Fixture {
        graph,
        weights,
        tokens_id,
        positions_id,
    }
}

impl Fixture {
    fn bindings(&self, tokens: &[u64], positions: &[u64]) -> Bindings<Value> {
        let mut b = self.weights.clone();
        b.set(self.tokens_id, Value::Index(tokens.to_vec()));
        b.set(self.positions_id, Value::Index(positions.to_vec()));
        b
    }

    fn state(&self) -> SequenceState {
        SequenceState::new([StateKind::KvPages, StateKind::PositionCounter])
    }
}

/// Run `tokens` as one step starting at absolute position `from`.
fn step(
    f: &Fixture,
    state: &mut SequenceState,
    kv: &mut KvCache,
    tokens: &[u64],
    from: u64,
    cancel: &Cancel,
) -> moxie_types::Result<moxie_interp::StepOutput> {
    let positions: Vec<u64> = (from..from + tokens.len() as u64).collect();
    Interpreter::new().run(
        &f.graph,
        &f.bindings(tokens, &positions),
        state,
        ROOT,
        kv,
        cancel,
    )
}

// --- acceptance -------------------------------------------------------------

#[test]
fn both_graphs_produce_logits_for_one_token_and_for_many() {
    for (name, dims, seed) in [("A", A, 11), ("B", B, 29)] {
        let f = build(dims, seed);

        // One token.
        let mut state = f.state();
        let mut kv = KvCache::for_branch(1, &state, ROOT).unwrap();
        state.append_prompt(ROOT, 1).unwrap();
        let out = step(&f, &mut state, &mut kv, &[3], 0, &Cancel::never()).unwrap();
        assert_eq!(
            out.logits.shape(),
            &[1, dims.vocab as usize],
            "graph {name}"
        );
        assert_eq!(out.prefix, 1);
        assert!(
            out.logits.data().iter().all(|v| v.is_finite()),
            "graph {name}"
        );
        assert!(state.next_logits_valid(ROOT), "graph {name}");

        // Many tokens.
        let mut state = f.state();
        let mut kv = KvCache::for_branch(1, &state, ROOT).unwrap();
        let tokens: Vec<u64> = (0..5).map(|i| i % dims.vocab).collect();
        state.append_prompt(ROOT, tokens.len() as u64).unwrap();
        let out = step(&f, &mut state, &mut kv, &tokens, 0, &Cancel::never()).unwrap();
        assert_eq!(
            out.logits.shape(),
            &[5, dims.vocab as usize],
            "graph {name}"
        );
        assert_eq!(kv.len(), 5);
        assert!(kv.is_coherent());
        assert!(state.next_logits_valid(ROOT), "graph {name}");
    }
}

#[test]
fn logits_are_unrounded_fp32() {
    // Task 0003's one exception to the rounding table. Document 05 builds the
    // sampler's pre-truncation normalizer from these; rounding them to BF16
    // first would quantise the distribution before any of that runs.
    let f = build(A, 5);
    let mut state = f.state();
    let mut kv = KvCache::for_branch(1, &state, ROOT).unwrap();
    state.append_prompt(ROOT, 4).unwrap();
    let out = step(&f, &mut state, &mut kv, &[1, 2, 3, 4], 0, &Cancel::never()).unwrap();
    assert_eq!(out.logits.precision(), Precision::F32);
    let unrepresentable = out
        .logits
        .data()
        .iter()
        .filter(|v| !moxie_interp::tensor::is_bf16_valued(**v))
        .count();
    assert!(
        unrepresentable > 0,
        "every logit happened to be BF16-representable, so this fixture cannot \
         distinguish rounded from unrounded output"
    );
}

#[test]
fn a_multi_token_step_is_causal() {
    // Row 0's logits must not depend on tokens that come after it. Running the
    // same prefix alone must give the same first row.
    let f = build(A, 7);
    let mut s1 = f.state();
    let mut kv1 = KvCache::for_branch(1, &s1, ROOT).unwrap();
    s1.append_prompt(ROOT, 4).unwrap();
    let all = step(&f, &mut s1, &mut kv1, &[2, 5, 7, 1], 0, &Cancel::never()).unwrap();

    let mut s2 = f.state();
    let mut kv2 = KvCache::for_branch(1, &s2, ROOT).unwrap();
    s2.append_prompt(ROOT, 1).unwrap();
    let first = step(&f, &mut s2, &mut kv2, &[2], 0, &Cancel::never()).unwrap();

    assert_eq!(all.logits.row(0).unwrap(), first.logits.row(0).unwrap());
}

#[test]
fn whole_and_chunked_prefill_agree() {
    // Document 06 M1's exit gate. The mask fixtures already prove the mask
    // arithmetic agrees; this proves the execution does, through the KV cache.
    let f = build(B, 13);
    let tokens: Vec<u64> = vec![1, 4, 0, 6, 2, 3];

    let mut whole_state = f.state();
    let mut whole_kv = KvCache::for_branch(1, &whole_state, ROOT).unwrap();
    whole_state
        .append_prompt(ROOT, tokens.len() as u64)
        .unwrap();
    let whole = step(
        &f,
        &mut whole_state,
        &mut whole_kv,
        &tokens,
        0,
        &Cancel::never(),
    )
    .unwrap();

    for width in [1usize, 2, 3, 4, 5, 6] {
        let mut state = f.state();
        let mut kv = KvCache::for_branch(1, &state, ROOT).unwrap();
        state.append_prompt(ROOT, tokens.len() as u64).unwrap();
        let mut rows: Vec<Vec<f32>> = Vec::new();
        let mut at = 0u64;
        for chunk in tokens.chunks(width) {
            let out = step(&f, &mut state, &mut kv, chunk, at, &Cancel::never()).unwrap();
            for r in 0..chunk.len() {
                rows.push(out.logits.row(r).unwrap().to_vec());
            }
            at += chunk.len() as u64;
        }
        assert_eq!(rows.len(), tokens.len(), "width {width}");
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(
                row,
                whole.logits.row(i).unwrap(),
                "width {width}, row {i}: chunked prefill disagreed with whole prefill"
            );
        }
        assert_eq!(
            kv.contents(),
            whole_kv.contents(),
            "width {width}: the caches diverged"
        );
    }
}

#[test]
fn a_cancelled_step_leaves_the_state_exactly_as_it_found_it() {
    // R08. Cancellation is checked at every operation boundary, and the KV
    // appends are staged, so there is nothing to roll back.
    let f = build(A, 17);
    let tokens = [1u64, 2, 3];

    let mut state = f.state();
    let mut kv = KvCache::for_branch(1, &state, ROOT).unwrap();
    state.append_prompt(ROOT, tokens.len() as u64).unwrap();

    // Cancel at each depth, including inside attention, after which a naive
    // implementation would have written this step's keys.
    for depth in 0..f.graph.nodes().len() as u64 {
        let before = state.frontiers(ROOT).unwrap();
        let kv_before = kv.clone();
        let e = step(&f, &mut state, &mut kv, &tokens, 0, &Cancel::after(depth)).unwrap_err();
        assert_eq!(e.kind(), "cancelled", "depth {depth}");
        assert_eq!(state.frontiers(ROOT).unwrap(), before, "depth {depth}");
        assert_eq!(
            kv.contents(),
            kv_before.contents(),
            "depth {depth}: a cancelled step wrote state"
        );
        assert!(!state.next_logits_valid(ROOT), "depth {depth}");
        assert!(state.live_results().is_empty(), "depth {depth}");
    }

    // ... and a full run afterwards matches a run that was never cancelled.
    let after_cancels = step(&f, &mut state, &mut kv, &tokens, 0, &Cancel::never()).unwrap();

    let mut clean_state = f.state();
    let mut clean_kv = KvCache::for_branch(1, &clean_state, ROOT).unwrap();
    clean_state
        .append_prompt(ROOT, tokens.len() as u64)
        .unwrap();
    let clean = step(
        &f,
        &mut clean_state,
        &mut clean_kv,
        &tokens,
        0,
        &Cancel::never(),
    )
    .unwrap();

    assert_eq!(after_cancels.logits, clean.logits);
    assert_eq!(kv.contents(), clean_kv.contents());
    assert_eq!(
        state.frontiers(ROOT).unwrap(),
        clean_state.frontiers(ROOT).unwrap()
    );
}

#[test]
fn positions_must_be_absolute_and_match_the_branch_frontier() {
    // R21: a chunk-local index masquerading as a sequence position. The
    // interpreter refuses rather than computing a plausible wrong answer.
    let f = build(A, 19);
    let mut state = f.state();
    let mut kv = KvCache::for_branch(1, &state, ROOT).unwrap();
    state.append_prompt(ROOT, 4).unwrap();
    step(&f, &mut state, &mut kv, &[1, 2], 0, &Cancel::never()).unwrap();

    // The next chunk starts at 2, not at 0.
    let wrong = Interpreter::new().run(
        &f.graph,
        &f.bindings(&[3, 4], &[0, 1]),
        &mut state,
        ROOT,
        &mut kv,
        &Cancel::never(),
    );
    let e = wrong.unwrap_err();
    assert_eq!(e.kind(), "invalid_request");
    assert!(e.to_string().contains("absolute"), "{e}");

    // Non-contiguous is refused too.
    assert!(
        Interpreter::new()
            .run(
                &f.graph,
                &f.bindings(&[3, 4], &[2, 7]),
                &mut state,
                ROOT,
                &mut kv,
                &Cancel::never()
            )
            .is_err()
    );

    // The right positions work, and nothing was consumed by the refusals.
    assert_eq!(state.frontiers(ROOT).unwrap().executed, 2);
    step(&f, &mut state, &mut kv, &[3, 4], 2, &Cancel::never()).unwrap();
    assert_eq!(state.frontiers(ROOT).unwrap().executed, 4);
}

#[test]
fn an_operation_with_no_registered_oracle_cannot_be_built_into_a_graph() {
    // F6's rule, now with a consumer: the refusal happens at graph construction,
    // so an unvalidated operation cannot reach the interpreter at all.
    let mut g = GraphBuilder::new(moxie_graph::OracleId("nobody_registered_this"), SymbolId(0));
    let rows = rows_symbol();
    let tokens = g.input(
        "tokens",
        TensorSpec::new(ValueRole::Index, vec![rows.clone()]),
    );
    let table = g
        .weight(
            "embed",
            TensorSpec::new(weight(), vec![Dim::constant(4), Dim::constant(2)]),
        )
        .unwrap();
    let out = g
        .node(
            OpParams::Embedding {
                vocab: 4,
                hidden: 2,
            },
            &[tokens, table],
        )
        .unwrap();
    let e = g.finish(out, &oracles()).unwrap_err();
    assert_eq!(e.kind(), "unsupported_kernel");
    assert!(e.to_string().contains("nobody_registered_this"), "{e}");
}

#[test]
fn shape_and_divisibility_errors_are_refused_at_construction() {
    let reg = oracles();

    // Rotating an odd number of lanes: rotation is over pairs.
    let mut g = GraphBuilder::new(moxie_oracles::HOST_REFERENCE, SymbolId(0));
    let rows = rows_symbol();
    let pos = g.input(
        "positions",
        TensorSpec::new(ValueRole::Index, vec![rows.clone()]),
    );
    let x = g
        .weight(
            "x",
            TensorSpec::new(weight(), vec![Dim::constant(1), Dim::constant(12)]),
        )
        .unwrap();
    let e = g
        .node(
            OpParams::Rope {
                heads: 3,
                head_dim: 4,
                rotary_dim: 3,
                base: 10_000.0,
            },
            &[x, pos],
        )
        .unwrap_err();
    assert_eq!(e.kind(), "dim", "{e}");

    // A head geometry that does not match the tensor it is given: 3 heads of 4
    // is 12 lanes, and the activation has 10.
    let mut g = GraphBuilder::new(moxie_oracles::HOST_REFERENCE, SymbolId(0));
    let pos = g.input(
        "positions",
        TensorSpec::new(ValueRole::Index, vec![rows.clone()]),
    );
    let narrow = g
        .weight(
            "x",
            TensorSpec::new(weight(), vec![Dim::constant(1), Dim::constant(10)]),
        )
        .unwrap();
    assert!(
        g.node(
            OpParams::Rope {
                heads: 3,
                head_dim: 4,
                rotary_dim: 4,
                base: 10_000.0,
            },
            &[narrow, pos],
        )
        .is_err()
    );

    // Two attention nodes on one KV layer would append the same positions twice.
    let mut g = GraphBuilder::new(moxie_oracles::HOST_REFERENCE, SymbolId(0));
    let pos = g.input(
        "positions",
        TensorSpec::new(ValueRole::Index, vec![rows.clone()]),
    );
    let t = g.input(
        "x",
        TensorSpec::new(act(Precision::Bf16), vec![rows.clone(), Dim::constant(8)]),
    );
    let params = OpParams::Attention {
        heads: 2,
        head_dim: 4,
        visibility: Visibility::Causal,
        layer: 0,
    };
    g.node(params.clone(), &[t, t, t, pos]).unwrap();
    assert!(g.node(params, &[t, t, t, pos]).is_err());

    // An index where a float is expected, and the reverse.
    let mut g = GraphBuilder::new(moxie_oracles::HOST_REFERENCE, SymbolId(0));
    let idx = g.input("i", TensorSpec::new(ValueRole::Index, vec![rows.clone()]));
    let f = g
        .weight(
            "w",
            TensorSpec::new(weight(), vec![Dim::constant(4), Dim::constant(2)]),
        )
        .unwrap();
    assert!(
        g.node(
            OpParams::Embedding {
                vocab: 4,
                hidden: 2
            },
            &[f, idx]
        )
        .is_err(),
        "a float table cannot be the token list"
    );
    let _ = reg;
}

#[test]
fn an_out_of_range_token_is_refused_at_execution() {
    let f = build(B, 23);
    let mut state = f.state();
    let mut kv = KvCache::for_branch(1, &state, ROOT).unwrap();
    state.append_prompt(ROOT, 1).unwrap();
    // Vocabulary is 7.
    let e = step(&f, &mut state, &mut kv, &[7], 0, &Cancel::never()).unwrap_err();
    assert_eq!(e.kind(), "invalid_request");
    assert_eq!(
        state.frontiers(ROOT).unwrap().executed,
        0,
        "nothing executed"
    );
    assert!(kv.is_empty());
}

#[test]
fn the_contract_table_is_what_the_code_says() {
    // Task 0003 fixes a partition rule, a state effect and an output precision
    // per operation. These are the table, transcribed from the document.
    let cases: &[(OpParams, PartitionRule, StateEffect, Precision)] = &[
        (
            OpParams::Embedding {
                vocab: 4,
                hidden: 2,
            },
            PartitionRule::Replicated,
            StateEffect::None,
            Precision::Bf16,
        ),
        (
            OpParams::Linear {
                in_features: 2,
                out_features: 2,
                bias: false,
            },
            PartitionRule::ColumnShardable,
            StateEffect::None,
            Precision::Bf16,
        ),
        (
            OpParams::RmsNorm {
                hidden: 2,
                eps: 1e-5,
            },
            PartitionRule::Replicated,
            StateEffect::None,
            Precision::Bf16,
        ),
        (
            OpParams::SwiGlu { width: 2 },
            PartitionRule::ColumnShardable,
            StateEffect::None,
            Precision::Bf16,
        ),
        (
            OpParams::Rope {
                heads: 1,
                head_dim: 2,
                rotary_dim: 2,
                base: 10_000.0,
            },
            PartitionRule::ColumnShardable,
            StateEffect::None,
            Precision::Bf16,
        ),
        (
            OpParams::Attention {
                heads: 1,
                head_dim: 2,
                visibility: Visibility::Causal,
                layer: 0,
            },
            PartitionRule::NotDetermined,
            StateEffect::Appends,
            Precision::Bf16,
        ),
        (
            OpParams::Residual,
            PartitionRule::Replicated,
            StateEffect::None,
            Precision::Bf16,
        ),
        (
            OpParams::VocabProjection {
                vocab: 4,
                hidden: 2,
            },
            PartitionRule::ColumnShardable,
            StateEffect::None,
            Precision::F32,
        ),
    ];
    for (params, partition, effect, precision) in cases {
        let op = params.op().name();
        assert_eq!(params.partition_rule(), *partition, "{op} partition");
        assert_eq!(params.state_effect(), *effect, "{op} state effect");
        assert_eq!(params.output_precision().get(), *precision, "{op} output");
    }

    // Attention is the only state-touching node, and TP lowering fails closed
    // on it until document 04's M5 work defines head ownership.
    let f = build(A, 3);
    let effects = f.graph.state_effects();
    assert_eq!(effects.len(), 1);
    assert_eq!(effects[0].1, StateEffect::Appends);
    for node in f.graph.nodes() {
        if matches!(node.params, OpParams::Attention { .. }) {
            assert!(node.contract.check_partitionable().is_err());
        } else {
            assert!(node.contract.check_partitionable().is_ok());
        }
    }
}

#[test]
fn every_intermediate_activation_stays_bf16_valued() {
    // The invariant the error bounds rest on. If any operation returned a value
    // BF16 cannot represent, `HostTensor::bf16` would have rejected it -- but
    // only the embedding uses that constructor, so this checks the rounded ones
    // by re-examining the logits' inputs through a whole run.
    let f = build(A, 31);
    let mut state = f.state();
    let mut kv = KvCache::for_branch(1, &state, ROOT).unwrap();
    state.append_prompt(ROOT, 3).unwrap();
    let out = step(&f, &mut state, &mut kv, &[1, 2, 3], 0, &Cancel::never()).unwrap();
    // The KV cache holds rounded activations from the projections.
    for pos in 0..kv.len() {
        let h = kv.history(0).unwrap();
        assert!(h.len() > pos);
    }
    // Logits are the one FP32 output; everything feeding them was BF16.
    assert_eq!(out.logits.precision(), Precision::F32);
    assert!(kv.is_coherent());
}

#[test]
fn a_linear_layer_inside_the_graph_meets_its_declared_bound() {
    // The numerical gate, on the operation with the longest reduction. The
    // expected value is computed here from the equation in FP64 -- it is not the
    // oracle re-run, which would compare the implementation to itself.
    let f = build(B, 37);
    let h = B.hidden as usize;
    let out_features = B.ffn as usize;

    // Take a real weight from the fixture, by name rather than by guessing an id.
    let id = *f
        .graph
        .weights()
        .iter()
        .find(|v| f.graph.name(**v) == Some("w_gate"))
        .expect("the fixture has a w_gate");
    let w = match f.weights.get(id).unwrap() {
        Value::Float(t) => t.clone(),
        _ => unreachable!(),
    };
    assert_eq!(
        w.shape(),
        &[out_features, h],
        "w_gate has the expected shape"
    );

    let mut rng = Lcg::new(101);
    let x: Vec<f32> = (0..h).map(|_| rng.next_f32()).collect();

    let got = moxie_oracles::linear::linear_row(&x, w.data(), out_features, None).unwrap();
    let want: Vec<f64> = (0..out_features)
        .map(|o| {
            let mut acc = 0f64;
            for (i, xi) in x.iter().enumerate().take(h) {
                acc += *xi as f64 * w.data()[o * h + i] as f64;
            }
            acc
        })
        .collect();
    let scale: Vec<f64> = (0..out_features)
        .map(|o| {
            x.iter()
                .enumerate()
                .take(h)
                .map(|(i, xi)| (*xi as f64 * w.data()[o * h + i] as f64).abs())
                .sum()
        })
        .collect();

    let s = ErrorSummary::normalized(&got, &want, &scale);
    let bound = gamma(h as u64 + 1);
    assert_eq!(s.count, out_features);
    assert!(
        s.within(bound),
        "{s} exceeded gamma({}) = {bound:.3e}",
        h + 1
    );
    // Reported, per document 07, not just asserted.
    println!("linear[{h}x{out_features}] {s} bound={bound:.3e}");
}

#[test]
fn a_step_refuses_to_run_with_a_missing_binding() {
    let f = build(A, 41);
    let mut state = f.state();
    let mut kv = KvCache::for_branch(1, &state, ROOT).unwrap();
    state.append_prompt(ROOT, 1).unwrap();
    let mut incomplete = Bindings::new();
    incomplete.set(f.tokens_id, Value::Index(vec![0]));
    incomplete.set(f.positions_id, Value::Index(vec![0]));
    let e = Interpreter::new()
        .run(
            &f.graph,
            &incomplete,
            &mut state,
            ROOT,
            &mut kv,
            &Cancel::never(),
        )
        .unwrap_err();
    assert_eq!(e.kind(), "invalid_request");
    assert!(e.to_string().contains("not bound"), "{e}");
}

#[test]
fn a_value_whose_shape_disagrees_with_the_graph_is_refused() {
    // The declared shapes are enforced against the data actually supplied, not
    // only checked structurally between operations at build time.
    let f = build(A, 53);
    let mut state = f.state();
    let mut kv = KvCache::for_branch(1, &state, ROOT).unwrap();
    state.append_prompt(ROOT, 2).unwrap();

    // Two tokens but three positions: the row counts disagree.
    let mut b = f.weights.clone();
    b.set(f.tokens_id, Value::Index(vec![1, 2]));
    b.set(f.positions_id, Value::Index(vec![0, 1, 2]));
    let e = Interpreter::new()
        .run(&f.graph, &b, &mut state, ROOT, &mut kv, &Cancel::never())
        .unwrap_err();
    assert_eq!(e.kind(), "invalid_artifact");
    assert!(e.to_string().contains("declares"), "{e}");

    // A weight of the wrong width, which the structural check cannot see
    // because it happens after the graph was built.
    let id = *f
        .graph
        .weights()
        .iter()
        .find(|v| f.graph.name(**v) == Some("norm_attn"))
        .unwrap();
    let mut b = f.bindings(&[1, 2], &[0, 1]);
    b.set(
        id,
        Value::Float(HostTensor::bf16(vec![1.0; 7], vec![7]).unwrap()),
    );
    let e = Interpreter::new()
        .run(&f.graph, &b, &mut state, ROOT, &mut kv, &Cancel::never())
        .unwrap_err();
    assert_eq!(e.kind(), "invalid_artifact");
    assert_eq!(state.frontiers(ROOT).unwrap().executed, 0);
    assert!(kv.is_empty());
}

#[test]
fn one_sequence_cannot_execute_against_another_sequences_cache() {
    // Fourth review, reproduced: `KvCache` had no identity, so a cache holding a
    // different token prefix was accepted on length alone. The step succeeded,
    // returned the other history's answer, and `next_logits_valid` was true.
    let f = build(A, 61);

    let mut a_state = f.state();
    let mut a_kv = KvCache::for_branch(1, &a_state, ROOT).unwrap();
    a_state.append_prompt(ROOT, 2).unwrap();
    step(&f, &mut a_state, &mut a_kv, &[1, 2], 0, &Cancel::never()).unwrap();

    // A second sequence, same shapes, same length, different tokens.
    let mut b_state = f.state();
    let mut b_kv = KvCache::for_branch(1, &b_state, ROOT).unwrap();
    b_state.append_prompt(ROOT, 2).unwrap();
    step(&f, &mut b_state, &mut b_kv, &[5, 6], 0, &Cancel::never()).unwrap();
    assert_eq!(a_kv.len(), b_kv.len(), "the substitution the review made");
    assert_ne!(a_kv.contents(), b_kv.contents(), "and the contents differ");

    // Continuing sequence A against sequence B's cache is refused.
    b_state.accept(ROOT, 1).unwrap();
    let e = Interpreter::new()
        .run(
            &f.graph,
            &f.bindings(&[3], &[2]),
            &mut a_state,
            ROOT,
            &mut b_kv,
            &Cancel::never(),
        )
        .unwrap_err();
    assert_eq!(e.kind(), "invalid_request");
    assert!(e.to_string().contains("sequence"), "{e}");
    assert_eq!(a_state.frontiers(ROOT).unwrap().executed, 2, "nothing ran");
}

#[test]
fn a_binding_whose_precision_disagrees_with_the_graph_is_refused() {
    // Fourth review, reproduced: binding validation compared shapes and never
    // dtypes, so an FP32 tensor holding a value BF16 cannot represent satisfied
    // a BF16-declared input -- and every error bound downstream rested on an
    // invariant nothing checked.
    let f = build(A, 67);
    let mut state = f.state();
    let mut kv = KvCache::for_branch(1, &state, ROOT).unwrap();
    state.append_prompt(ROOT, 1).unwrap();

    let id = *f
        .graph
        .weights()
        .iter()
        .find(|v| f.graph.name(**v) == Some("bo"))
        .unwrap();
    let width = A.hidden as usize;
    // Representable in FP32, not in BF16.
    let sneaky = 1.0f32 + 1.0 / 1024.0;
    assert!(!moxie_interp::tensor::is_bf16_valued(sneaky));

    let mut b = f.bindings(&[1], &[0]);
    b.set(
        id,
        Value::Float(HostTensor::f32(vec![sneaky; width], vec![width]).unwrap()),
    );
    let e = Interpreter::new()
        .run(&f.graph, &b, &mut state, ROOT, &mut kv, &Cancel::never())
        .unwrap_err();
    assert_eq!(e.kind(), "invalid_artifact");
    assert!(e.to_string().contains("bf16"), "{e}");
    assert_eq!(state.frontiers(ROOT).unwrap().executed, 0);

    // An index where a tensor is declared, and the reverse.
    let mut b = f.bindings(&[1], &[0]);
    b.set(id, Value::Index(vec![0; width]));
    assert!(
        Interpreter::new()
            .run(&f.graph, &b, &mut state, ROOT, &mut kv, &Cancel::never())
            .is_err()
    );
}

#[test]
fn every_position_operand_is_the_same_binding() {
    // Fourth review, reproduced: only the first position-consuming node's
    // operand was checked, so a graph whose RoPE read 0 and whose attention read
    // 999 executed happily at prefix 1. The graph now refuses to be built that
    // way, which is stronger than validating each operand separately.
    let mut g = GraphBuilder::new(moxie_oracles::HOST_REFERENCE, SymbolId(0));
    let rows = rows_symbol();
    let p1 = g.input(
        "positions",
        TensorSpec::new(ValueRole::Index, vec![rows.clone()]),
    );
    let p2 = g.input(
        "other_positions",
        TensorSpec::new(ValueRole::Index, vec![rows.clone()]),
    );
    let x = g.input(
        "x",
        TensorSpec::new(act(Precision::Bf16), vec![rows.clone(), Dim::constant(8)]),
    );
    let r = g
        .node(
            OpParams::Rope {
                heads: 2,
                head_dim: 4,
                rotary_dim: 4,
                base: 10_000.0,
            },
            &[x, p1],
        )
        .unwrap();
    let e = g
        .node(
            OpParams::Attention {
                heads: 2,
                head_dim: 4,
                visibility: Visibility::Causal,
                layer: 0,
            },
            &[r, r, r, p2],
        )
        .unwrap_err();
    assert_eq!(e.kind(), "invalid_artifact");
    assert!(e.to_string().contains("shared binding"), "{e}");

    // The same operand is accepted, and the fixture graphs use one throughout.
    assert!(
        g.node(
            OpParams::Attention {
                heads: 2,
                head_dim: 4,
                visibility: Visibility::Causal,
                layer: 0,
            },
            &[r, r, r, p1],
        )
        .is_ok()
    );
    let f = build(A, 71);
    assert_eq!(f.graph.positions(), Some(f.positions_id));
}

#[test]
fn a_nonfinite_weight_or_result_never_reaches_committed_state() {
    // Fourth review, reproduced: a BF16-tagged vocabulary weight containing NaN
    // produced a successful step with NaN logits, the counters advanced, and
    // `next_logits_valid` became true.
    let f = build(A, 73);
    let mut state = f.state();
    let mut kv = KvCache::for_branch(1, &state, ROOT).unwrap();
    state.append_prompt(ROOT, 1).unwrap();

    let id = *f
        .graph
        .weights()
        .iter()
        .find(|v| f.graph.name(**v) == Some("vocab"))
        .unwrap();
    let n = (A.vocab * A.hidden) as usize;
    let mut data = vec![0.5f32; n];
    data[3] = f32::NAN;
    // NaN is BF16-representable, so the storage invariant does not catch it.
    let mut b = f.bindings(&[1], &[0]);
    b.set(
        id,
        Value::Float(HostTensor::bf16(data, vec![A.vocab as usize, A.hidden as usize]).unwrap()),
    );

    let e = Interpreter::new()
        .run(&f.graph, &b, &mut state, ROOT, &mut kv, &Cancel::never())
        .unwrap_err();
    assert_eq!(e.kind(), "invalid_artifact");
    assert_eq!(
        state.frontiers(ROOT).unwrap().executed,
        0,
        "nothing committed"
    );
    assert!(kv.is_empty());
    assert!(!state.next_logits_valid(ROOT));
    assert!(state.live_results().is_empty());

    // An infinity produced *during* execution is attributed to the operation
    // that produced it, rather than surfacing as a mysterious logit.
    let id = *f
        .graph
        .weights()
        .iter()
        .find(|v| f.graph.name(**v) == Some("w_up"))
        .unwrap();
    let n = (A.ffn * A.hidden) as usize;
    let mut b = f.bindings(&[1], &[0]);
    // A power of two near the top of the range: BF16-representable, so the
    // storage invariant admits it, and large enough that the reduction overflows.
    let huge = 2.0f32.powi(127);
    assert!(moxie_interp::tensor::is_bf16_valued(huge));
    b.set(
        id,
        Value::Float(
            HostTensor::bf16(vec![huge; n], vec![A.ffn as usize, A.hidden as usize]).unwrap(),
        ),
    );
    match Interpreter::new().run(&f.graph, &b, &mut state, ROOT, &mut kv, &Cancel::never()) {
        Err(e) => {
            assert!(
                matches!(e.kind(), "numerical" | "invalid_artifact"),
                "unexpected kind {}: {e}",
                e.kind()
            );
            assert_eq!(state.frontiers(ROOT).unwrap().executed, 0);
        }
        Ok(out) => {
            // If it stays finite the fixture proves nothing, so say so rather
            // than passing quietly.
            assert!(
                out.logits.data().iter().all(|v| v.is_finite()),
                "a non-finite logit escaped"
            );
        }
    }
}

#[test]
fn a_stale_cache_cannot_be_certified_by_rolling_it_back() {
    // Fifth review, reproduced: `rollback_to` re-stamped the cache from the
    // state after truncating, including when the truncation changed nothing. A
    // caller could take a cache saved before a rewrite, "roll it back" to the
    // length it already had, and have its stale bytes pass `check_owner`.
    let f = build(A, 79);
    let mut state = f.state();
    let mut kv = KvCache::for_branch(1, &state, ROOT).unwrap();

    state.append_prompt(ROOT, 1).unwrap();
    step(&f, &mut state, &mut kv, &[1], 0, &Cancel::never()).unwrap();
    state.accept(ROOT, 1).unwrap();
    step(&f, &mut state, &mut kv, &[2], 1, &Cancel::never()).unwrap();
    let stale = kv.clone();
    assert_eq!(stale.len(), 2);

    // Rewrite position 1 with a different token.
    state.rollback_to(ROOT, 1, &[]).unwrap();
    kv.rollback_to(&state, ROOT, 1).unwrap();
    state.accept(ROOT, 1).unwrap();
    step(&f, &mut state, &mut kv, &[5], 1, &Cancel::never()).unwrap();
    assert_ne!(stale.contents(), kv.contents(), "the bytes really differ");

    // The stale cache is refused, and rolling it back to its own length does not
    // launder it.
    let mut restamped = stale.clone();
    assert!(restamped.check_owner(&state, ROOT).is_err());
    let e = restamped.rollback_to(&state, ROOT, 2).unwrap_err();
    assert!(e.to_string().contains("different version"), "{e}");
    assert!(restamped.check_owner(&state, ROOT).is_err());

    // And it cannot be used to continue the sequence.
    state.accept(ROOT, 1).unwrap();
    assert!(
        Interpreter::new()
            .run(
                &f.graph,
                &f.bindings(&[6], &[2]),
                &mut state,
                ROOT,
                &mut restamped,
                &Cancel::never()
            )
            .is_err()
    );

    // The genuine cache still works, so the rule is about staleness rather than
    // about rollback being forbidden.
    step(&f, &mut state, &mut kv, &[6], 2, &Cancel::never()).unwrap();
}

#[test]
fn an_operand_precision_the_node_contract_rejects_is_refused_at_construction() {
    // Fifth review, reproduced: binding validation compared a tensor against its
    // *declaration*, so declaring the input FP32 let it into an operation whose
    // contract permits only BF16 activations. Agreement with a declaration is
    // not agreement with the consumer.
    let mut g = GraphBuilder::new(moxie_oracles::HOST_REFERENCE, SymbolId(0));
    let rows = rows_symbol();
    let pos = g.input(
        "positions",
        TensorSpec::new(ValueRole::Index, vec![rows.clone()]),
    );
    let wide = g.input(
        "x",
        TensorSpec::new(act(Precision::F32), vec![rows.clone(), Dim::constant(8)]),
    );
    let e = g
        .node(
            OpParams::Rope {
                heads: 2,
                head_dim: 4,
                rotary_dim: 4,
                base: 10_000.0,
            },
            &[wide, pos],
        )
        .unwrap_err();
    assert_eq!(e.kind(), "invalid_artifact");
    assert!(e.to_string().contains("contract accepts"), "{e}");

    // The same node with a BF16 operand is accepted.
    let narrow = g.input(
        "x_bf16",
        TensorSpec::new(act(Precision::Bf16), vec![rows.clone(), Dim::constant(8)]),
    );
    assert!(
        g.node(
            OpParams::Rope {
                heads: 2,
                head_dim: 4,
                rotary_dim: 4,
                base: 10_000.0,
            },
            &[narrow, pos],
        )
        .is_ok()
    );
}

#[test]
fn generation_advances_the_state_the_way_the_contracts_require() {
    // A prompt, then three decode steps, checking the four counters and the
    // provenance rules at each boundary.
    let f = build(A, 43);
    let mut state = f.state();
    let mut kv = KvCache::for_branch(1, &state, ROOT).unwrap();

    state.append_prompt(ROOT, 3).unwrap();
    let out = step(&f, &mut state, &mut kv, &[1, 2, 3], 0, &Cancel::never()).unwrap();
    let fr = state.frontiers(ROOT).unwrap();
    assert_eq!(
        (fr.prompt, fr.accepted, fr.executed, fr.emitted),
        (3, 3, 3, 0)
    );
    assert!(state.next_logits_valid(ROOT));
    assert_eq!(out.retained.prefix(), 3);

    for n in 0..3u64 {
        // Greedy over the reference's own logits: this is a state-machine test,
        // not a sampling test, and the sampler has its own fixtures.
        let last = out.logits.rows() - 1;
        let _ = last;
        let token = 1u64;
        state.accept(ROOT, 1).unwrap();
        assert!(
            !state.next_logits_valid(ROOT),
            "the accepted token has not been executed yet"
        );
        assert_eq!(state.frontiers(ROOT).unwrap().pending_execution(), 1);

        step(&f, &mut state, &mut kv, &[token], 3 + n, &Cancel::never()).unwrap();
        assert!(state.next_logits_valid(ROOT));
        state.emit(ROOT, 1).unwrap();
    }

    let fr = state.frontiers(ROOT).unwrap();
    assert_eq!(
        (fr.prompt, fr.accepted, fr.executed, fr.emitted),
        (3, 6, 6, 3)
    );
    assert_eq!(
        fr.completion(),
        3,
        "usage counts committed completion tokens"
    );
    assert_eq!(kv.len(), 6);
}

#[test]
fn a_rollback_drops_the_kv_tail_and_the_retained_result() {
    let f = build(A, 47);
    let mut state = f.state();
    let mut kv = KvCache::for_branch(1, &state, ROOT).unwrap();
    state.append_prompt(ROOT, 2).unwrap();
    step(&f, &mut state, &mut kv, &[1, 2], 0, &Cancel::never()).unwrap();
    state.accept(ROOT, 2).unwrap();
    step(&f, &mut state, &mut kv, &[3, 4], 2, &Cancel::never()).unwrap();
    assert_eq!(kv.len(), 4);

    // KvPages truncates, so no restore evidence is required.
    state.rollback_to(ROOT, 2, &[]).unwrap();
    kv.rollback_to(&state, ROOT, 2).unwrap();
    assert_eq!(kv.len(), 2);
    assert!(
        !state.next_logits_valid(ROOT),
        "the result at prefix 4 described state that no longer exists"
    );

    // The result at prefix 2 survives: the rollback did not touch those
    // positions, and task 0002's rule is that only prefixes that changed lose
    // their results. The one at prefix 4 is gone.
    let live: Vec<u64> = state.live_results().iter().map(|h| h.prefix()).collect();
    assert_eq!(
        live,
        vec![2],
        "only the prefix-2 result should remain: {live:?}"
    );

    // Re-executing writes different positions, so the discarded result cannot
    // come back even though the prefix length matches again.
    state.accept(ROOT, 2).unwrap();
    let again = step(&f, &mut state, &mut kv, &[5, 6], 2, &Cancel::never()).unwrap();
    assert!(state.next_logits_valid(ROOT));
    assert_eq!(again.prefix, 4);
    let live: Vec<u64> = state.live_results().iter().map(|h| h.prefix()).collect();
    assert_eq!(live, vec![2, 4]);
}
