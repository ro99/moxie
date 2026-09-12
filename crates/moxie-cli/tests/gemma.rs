//! The reduced Gemma-4-like graph, end to end through the shared service.
//!
//! Nothing here executes a checkpoint. The weights are the composition root's
//! synthetic pattern and the tokens are not model output; see
//! `docs/models/gemma4.md` for what the real artifact would additionally need.
//!
//! The load-bearing tests are the ones that assert a **difference**: each new
//! operation parameter, set to the value the non-Gemma synthetic graph uses,
//! must change the logits. A parameter that can be swapped without changing
//! anything is decoration, and decoration is how a family's mathematics gets
//! quietly replaced by a standard transformer's.

use moxie_cli::{Options, Shape, gemma, render};
use moxie_engine::{
    Cancel, GenerationEvent as Event, GenerationRequest as Request, Value,
    service::{GenerationService, StartError},
};
use moxie_graph::{Bindings, Graph, OpParams, OracleRegistry, RopeLayout, ValueId};
use moxie_interp::{Interpreter, KvCache, paged::PagedExecution};
use moxie_memory::{CapacitySnapshot, Ledger};
use moxie_models::gemma4::{Fraction, Gemma4Text, TextConfig, embedding_scale};
use moxie_state::{KvGeometry, LayerKv, PagedSequence, ROOT, Retention, SequenceState, StateKind};
use moxie_types::{Precision, Scope, SymbolId};

/// The pages a reduced Gemma configuration needs, transcribed from the
/// configuration rather than obtained from the engine.
///
/// Deliberately a second implementation: the engine derives the same geometry
/// from the composed graph's attention nodes, and a test that called into the
/// engine to build what it then checks the engine against would be checking
/// nothing. The two agree only if both read the configuration the same way.
fn paged_geometry(
    config: &TextConfig,
    page_tokens: usize,
    max_tokens: usize,
    tentative_rows: usize,
) -> KvGeometry {
    KvGeometry {
        layers: (0..config.layers)
            .map(|layer| {
                let g = config.layer_geometry(layer);
                LayerKv {
                    // The pages store key/value heads, which under
                    // grouped-query attention is fewer than the query heads.
                    kv_heads: g.kv_heads as usize,
                    key_dim: g.head_dim as usize,
                    value_dim: g.head_dim as usize,
                    retention: match g.window {
                        None => Retention::All,
                        Some(window) => Retention::Window {
                            window: window as usize,
                        },
                    },
                }
            })
            .collect(),
        precision: Precision::Bf16,
        page_tokens,
        max_tokens,
        tentative_rows,
    }
}

fn ledger() -> Ledger {
    Ledger::new([CapacitySnapshot::new(Scope::Host, 1 << 30, 1).unwrap()]).unwrap()
}

fn drain(service: &mut GenerationService<'_, '_>) -> Vec<u32> {
    let mut tokens = Vec::new();
    let mut terminal = 0;
    while let Some(event) = service.next_event(&Cancel::never()) {
        match event {
            Event::Token { id, .. } => tokens.push(id),
            Event::Finished { usage } => {
                terminal += 1;
                assert_eq!(usage.completion_tokens, tokens.len());
            }
            Event::Failed { error, .. } => panic!("{error}"),
            Event::Cancelled { .. } => panic!("unexpected cancellation"),
            _ => {}
        }
    }
    assert_eq!(terminal, 1, "exactly one terminal event");
    assert!(service.is_idle());
    assert_eq!(service.charged_bytes(), 0);
    tokens
}

/// Run one forward pass over the whole prompt and return its logits.
///
/// The dense reference path, which shares its arithmetic with the paged one.
fn dense_logits(
    graph: &Graph,
    weights: &Bindings<Value>,
    tokens_id: ValueId,
    positions_id: ValueId,
    prompt: &[u64],
    layers: usize,
) -> Vec<f32> {
    let mut dense = SequenceState::new([StateKind::KvPages]);
    let mut cache = KvCache::for_branch(layers, &dense, ROOT).unwrap();
    dense.append_prompt(ROOT, prompt.len() as u64).unwrap();
    let mut bindings = weights.clone();
    bindings.set(tokens_id, Value::Index(prompt.to_vec()));
    bindings.set(
        positions_id,
        Value::Index((0..prompt.len() as u64).collect()),
    );
    Interpreter::new()
        .run(
            graph,
            &bindings,
            &mut dense,
            ROOT,
            &mut cache,
            &Cancel::never(),
        )
        .unwrap()
        .logits
        .data()
        .to_vec()
}

fn logits_for(config: TextConfig, prompt: &[u64]) -> Vec<f32> {
    let layers = config.layers as usize;
    let f = gemma::build_with_config(config).unwrap();
    dense_logits(&f.graph, &f.weights, f.tokens, f.positions, prompt, layers)
}

#[test]
fn both_reduced_geometries_generate_through_the_shared_service() {
    for shape in [gemma::Shape::A, gemma::Shape::B] {
        let config = shape.config();
        let fixture = gemma::build(shape).unwrap();
        let prompt: Vec<u32> = (0..17).map(|i| i % config.vocab as u32).collect();
        for temperature in [0.0, 0.25, 1.0] {
            let mut owner = ledger();
            let mut service = GenerationService::new(&mut owner, fixture.program());
            let whole = Request {
                prompt: &prompt,
                max_new_tokens: 4,
                prefill_chunk: prompt.len(),
                temperature,
                seed: 7,
            };
            service.start(whole).unwrap();
            let reference = drain(&mut service);
            assert_eq!(reference.len(), 4);
            // Whole, chunked, tail-producing and one-row-at-a-time prefill all
            // reach the same committed sequence. The window is 3 and 5 in the
            // two geometries, so several chunk sizes cross it.
            for chunk in [1, 2, 3, 5, 8, 17, 64] {
                service
                    .start(Request {
                        prefill_chunk: chunk,
                        ..whole
                    })
                    .unwrap();
                assert_eq!(
                    reference,
                    drain(&mut service),
                    "{shape:?} chunk {chunk} T {temperature}"
                );
            }
        }
    }
}

#[test]
fn paged_and_dense_logits_are_bit_identical_across_pages_and_decode() {
    for shape in [gemma::Shape::A, gemma::Shape::B] {
        let config = shape.config();
        let layers = config.layers as usize;
        let vocab = config.vocab;
        let fixture = gemma::build(shape).unwrap();
        let rows = 19usize;
        for chunk in [1, 3, 7, 19] {
            let mut owner = ledger();
            let geometry = paged_geometry(&config, 7, 32, 19);
            let mut pages =
                PagedSequence::with_sampling(&mut owner, geometry.clone(), vocab as usize, 10, 42)
                    .unwrap();
            let mut dense = SequenceState::new([StateKind::KvPages]);
            let mut cache = KvCache::for_branch(layers, &dense, ROOT).unwrap();
            pages.append_prompt(rows as u64).unwrap();
            let execution = PagedExecution::bind(
                &fixture.graph,
                &fixture.weights,
                fixture.tokens,
                fixture.positions,
                &mut pages,
            )
            .unwrap();
            dense.append_prompt(ROOT, rows as u64).unwrap();
            for start in (0..rows).step_by(chunk) {
                let end = (start + chunk).min(rows);
                let ids: Vec<u64> = (start..end).map(|i| i as u64 % vocab).collect();
                let positions: Vec<u64> = (start as u64..end as u64).collect();
                let mut bindings = fixture.weights.clone();
                bindings.set(fixture.tokens, Value::Index(ids.clone()));
                bindings.set(fixture.positions, Value::Index(positions.clone()));
                let reference = Interpreter::new()
                    .run(
                        &fixture.graph,
                        &bindings,
                        &mut dense,
                        ROOT,
                        &mut cache,
                        &Cancel::never(),
                    )
                    .unwrap();
                pages.clear_logits().unwrap();
                let txn = pages.begin().unwrap();
                let actual = execution
                    .run(&mut pages, txn, &ids, &positions, &Cancel::never())
                    .unwrap();
                assert_eq!(
                    reference.logits.data(),
                    actual.logits().data(),
                    "{shape:?} chunk {chunk} rows {start}..{end}"
                );
                pages.commit_prefix(txn, 0).unwrap();
            }
        }
    }
}

/// Task 0017's central gate: **reclaiming changes no output bit**.
///
/// The dense `KvCache` reference retains every row and masks; the paged store
/// physically reclaims what a sliding layer's window cannot see. If the two
/// agree bit for bit, the reclamation threw away exactly what was unreachable
/// and nothing else. A tolerance would not be a weaker version of this claim,
/// it would be a different and much weaker one, so the comparison is on raw
/// FP32 bits.
///
/// The geometry is chosen so reclamation actually happens -- the test asserts
/// that it did, because a windowed layer that never reached its capacity would
/// make this pass while proving nothing.
#[test]
fn reclaiming_a_sliding_layers_window_changes_no_logit_bit() {
    for shape in [gemma::Shape::A, gemma::Shape::B] {
        let config = shape.config();
        let layers = config.layers as usize;
        let vocab = config.vocab;
        let fixture = gemma::build(shape).unwrap();
        let rows = 23usize;
        // Small pages and small headroom, so a sliding layer's ring wraps
        // several times inside the sequence. Page sizes on either side of the
        // window put the wrap before, on and after a page edge.
        for (page_tokens, chunk) in [(2usize, 1usize), (2, 2), (3, 3), (4, 2), (5, 5)] {
            let mut owner = ledger();
            let geometry = paged_geometry(&config, page_tokens, 32, chunk);
            let mut pages = PagedSequence::new(&mut owner, geometry).unwrap();
            let mut dense = SequenceState::new([StateKind::KvPages]);
            let mut cache = KvCache::for_branch(layers, &dense, ROOT).unwrap();
            pages.append_prompt(rows as u64).unwrap();
            dense.append_prompt(ROOT, rows as u64).unwrap();
            let execution = PagedExecution::bind(
                &fixture.graph,
                &fixture.weights,
                fixture.tokens,
                fixture.positions,
                &mut pages,
            )
            .unwrap();
            for start in (0..rows).step_by(chunk) {
                let end = (start + chunk).min(rows);
                let ids: Vec<u64> = (start..end).map(|i| i as u64 % vocab).collect();
                let positions: Vec<u64> = (start as u64..end as u64).collect();
                let mut bindings = fixture.weights.clone();
                bindings.set(fixture.tokens, Value::Index(ids.clone()));
                bindings.set(fixture.positions, Value::Index(positions.clone()));
                let reference = Interpreter::new()
                    .run(
                        &fixture.graph,
                        &bindings,
                        &mut dense,
                        ROOT,
                        &mut cache,
                        &Cancel::never(),
                    )
                    .unwrap();
                pages.clear_logits().unwrap();
                let txn = pages.begin().unwrap();
                let actual = execution
                    .run(&mut pages, txn, &ids, &positions, &Cancel::never())
                    .unwrap();
                assert_eq!(
                    reference
                        .logits
                        .data()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    actual
                        .logits()
                        .data()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    "{shape:?} pages {page_tokens} chunk {chunk} rows {start}..{end}"
                );
                pages.commit_prefix(txn, 0).unwrap();
            }
            // Reclamation happened on every sliding layer, and on no global
            // one: otherwise the comparison above compared two full histories.
            let mut reclaimed = 0;
            for layer in 0..layers {
                let start = pages.retained_range(layer).unwrap().start;
                if config.layer_geometry(layer as u32).window.is_some() {
                    assert!(
                        start > 0,
                        "{shape:?} pages {page_tokens} chunk {chunk}: sliding layer \
                         {layer} never reclaimed, so this proved nothing"
                    );
                    reclaimed += 1;
                } else {
                    assert_eq!(start, 0, "a global layer must keep everything");
                }
            }
            assert!(reclaimed > 0);
            // And the reclaimed rows are refused rather than served stale.
            let sliding = (0..layers)
                .find(|l| config.layer_geometry(*l as u32).window.is_some())
                .unwrap();
            assert!(matches!(
                pages.row(sliding, 0),
                Err(moxie_types::Error::Reclaimed { .. })
            ));
            pages.close(&mut owner).unwrap();
        }
    }
}

#[test]
fn the_logit_softcap_bounds_every_logit() {
    // The cap is the only thing standing between the projection's raw output
    // and the sampler, so a graph that dropped it would be visible here.
    for shape in [gemma::Shape::A, gemma::Shape::B] {
        let config = shape.config();
        let cap = config.final_logit_softcap;
        let prompt: Vec<u64> = (0..9).map(|i| i % config.vocab).collect();
        let logits = logits_for(config, &prompt);
        assert!(!logits.is_empty());
        for v in &logits {
            assert!(
                v.is_finite() && v.abs() <= cap,
                "{shape:?}: logit {v} escaped the cap {cap}"
            );
        }
    }
}

/// Each new parameter, set to the value the non-Gemma synthetic graph uses,
/// must change the result.
///
/// This is the acceptance condition the contract states in those words. It is
/// the test that would fail if a parameter were accepted, stored, threaded
/// through every signature and then ignored.
#[test]
fn every_gemma_parameter_is_load_bearing() {
    let prompt: Vec<u64> = (0..11).map(|i| i % 11).collect();
    let base = gemma::Shape::A.config();
    let reference = logits_for(base.clone(), &prompt);

    // 1. The embedding output scale. The synthetic graph uses 1.0.
    let mut c = base.clone();
    c.embedding_scale = 1.0;
    assert_ne!(reference, logits_for(c, &prompt), "embedding scale");

    // 2. The logit softcap, relaxed far past the logits' range so that only
    // the cap's own rounding could still bend them.
    let mut c = base.clone();
    c.final_logit_softcap = 1.0e30;
    assert_ne!(reference, logits_for(c, &prompt), "logit softcap");

    // 3. The per-layer MLP residual scalars, set to the synthetic graph's 1.0.
    let mut c = base.clone();
    c.layer_scalars = vec![1.0; base.layers as usize];
    assert_ne!(reference, logits_for(c, &prompt), "residual scalars");

    // 4. The sliding window, widened past the prompt so every sliding layer
    // becomes causal. The mask is the parameter here.
    let mut c = base.clone();
    c.sliding_window = 4096;
    assert_ne!(reference, logits_for(c, &prompt), "sliding window");

    // 5. The partial rotary fraction on global layers.
    let mut c = base.clone();
    c.global_partial_rotary = Fraction::WHOLE;
    assert_ne!(reference, logits_for(c, &prompt), "partial rotary");

    // 6. The per-layer-type RoPE base.
    let mut c = base.clone();
    c.global_rope_theta = c.sliding_rope_theta;
    assert_ne!(reference, logits_for(c, &prompt), "global rope theta");

    // 7. The grouped-query head ratio: four query heads over four key/value
    // heads instead of two. The graph is rebuilt, so this also proves the
    // narrower stored rows were not incidental.
    let mut c = base.clone();
    c.local_kv_heads = 4;
    assert_ne!(
        reference,
        logits_for(c, &prompt),
        "sliding kv head grouping"
    );

    // 8. And the same on the global layers, which now have their own count.
    // A configuration where the two layer types agreed would pass 7 and miss
    // this entirely.
    let mut c = base.clone();
    c.global_kv_heads = 2;
    assert_ne!(reference, logits_for(c, &prompt), "global kv head grouping");

    // 9. The global head dimension, independent of the sliding one.
    let mut c = base.clone();
    c.global_head_dim = 16;
    assert_ne!(reference, logits_for(c, &prompt), "global head dimension");
}

/// The parameters that cannot be reached through `TextConfig`, because the model
/// module fixes them. Rebuilding the graph with the conventional value proves
/// each is not decoration either.
///
/// Independent review found this list short of what the test's name claimed: it
/// covered the score scale and the RoPE layout but not the inverse-frequency
/// denominator, the activation, or an absent softcap — and the earlier softcap
/// substitution used a very large cap, which the contract explicitly
/// distinguishes from no cap. All of those are here now. The norm grouping,
/// whose substitution also changes a gain's width, has its own test below.
#[test]
fn the_parameters_fixed_by_the_model_module_are_load_bearing() {
    let prompt: Vec<u64> = (0..11).map(|i| i % 11).collect();
    let base = gemma::Shape::A.config();
    let reference = logits_for(base.clone(), &prompt);
    let layers = base.layers as usize;

    for swap in [
        Swap::ConventionalScale,
        Swap::InterleavedRope,
        Swap::RotaryWidthDenominator,
        Swap::SwiGlu,
        Swap::NoSoftcap,
    ] {
        let f = gemma::build_with_config(base.clone()).unwrap();
        let rebuilt = rebuild_with(&f.graph, swap);
        // The rebuild preserves value identity, so the original bindings still
        // name the same weights -- which is what makes this a comparison of one
        // parameter rather than of two unrelated graphs.
        let got = dense_logits(&rebuilt, &f.weights, f.tokens, f.positions, &prompt, layers);
        assert_ne!(reference, got, "{swap:?}");
    }
}

#[derive(Debug, Clone, Copy)]
enum Swap {
    /// `1/sqrt(head_dim)` where Gemma uses exactly 1.0.
    ConventionalScale,
    /// Adjacent-pair rotation where Gemma pairs halves.
    InterleavedRope,
    /// The rotated width as the inverse-frequency denominator, where Gemma
    /// divides by the whole head even on a partially rotated one. Only the
    /// global layers rotate partially, so only they change.
    RotaryWidthDenominator,
    /// SwiGLU where Gemma uses GeGLU. Same shapes, different gate transform.
    SwiGlu,
    /// No logit cap at all, which is not the same as a very large one: the
    /// large cap still rounds four times and still costs a tanh.
    NoSoftcap,
}

/// Rebuild a graph, transforming each node's parameters.
///
/// A test-only transformation. It visits values in `ValueId` order and re-adds
/// each one the way the original builder did, so the rebuilt graph has the same
/// identities as the original and the original's weight bindings still apply.
/// Every node goes back through `GraphBuilder`, so the rebuilt graph is
/// validated rather than assembled.
fn rebuild(graph: &Graph, mut transform: impl FnMut(&OpParams) -> OpParams) -> Graph {
    let mut oracles = OracleRegistry::new();
    moxie_oracles::register(&mut oracles).unwrap();
    let mut builder = moxie_graph::GraphBuilder::new(moxie_oracles::HOST_REFERENCE, SymbolId(0));
    let mut last = None;
    for index in 0..graph.value_count() {
        let id = ValueId(index as u32);
        let spec = graph.spec(id).unwrap().clone();
        let name = graph.name(id).unwrap_or("value").to_string();
        if graph.inputs().contains(&id) {
            assert_eq!(builder.input(&name, spec), id);
        } else if graph.weights().contains(&id) {
            assert_eq!(builder.weight(&name, spec).unwrap(), id);
        } else {
            let node = graph
                .nodes()
                .iter()
                .find(|n| n.output == id)
                .expect("every value is an input, a weight or a node output");
            let out = builder.node(transform(&node.params), &node.inputs).unwrap();
            assert_eq!(out, id, "rebuilding must preserve value identity");
            last = Some(out);
        }
    }
    builder.finish(last.unwrap(), &oracles).unwrap()
}

fn rebuild_with(graph: &Graph, swap: Swap) -> Graph {
    rebuild(graph, |params| match (swap, params.clone()) {
        (
            Swap::ConventionalScale,
            OpParams::Attention {
                heads,
                kv_heads,
                head_dim,
                visibility,
                layer,
                ..
            },
        ) => OpParams::Attention {
            heads,
            kv_heads,
            head_dim,
            scale: moxie_graph::reciprocal_sqrt_scale(head_dim),
            visibility,
            layer,
        },
        (
            Swap::InterleavedRope,
            OpParams::Rope {
                heads,
                head_dim,
                rotary_dim,
                frequency_dim,
                base,
                ..
            },
        ) => OpParams::Rope {
            heads,
            head_dim,
            rotary_dim,
            frequency_dim,
            base,
            layout: RopeLayout::Interleaved,
        },
        (
            Swap::RotaryWidthDenominator,
            OpParams::Rope {
                heads,
                head_dim,
                rotary_dim,
                base,
                layout,
                ..
            },
        ) => OpParams::Rope {
            heads,
            head_dim,
            rotary_dim,
            // Divide by the rotated width instead of the whole head. Only the
            // global layers rotate partially, so only they change.
            frequency_dim: rotary_dim,
            base,
            layout,
        },
        (Swap::SwiGlu, OpParams::GeGlu { width }) => OpParams::SwiGlu { width },
        (Swap::NoSoftcap, OpParams::VocabProjection { vocab, hidden, .. }) => {
            OpParams::VocabProjection {
                vocab,
                hidden,
                softcap: None,
            }
        }
        (_, other) => other,
    })
}

/// Per-head normalization against whole-row normalization, with everything else
/// held identical.
///
/// This one cannot be a plain parameter swap: setting `group` to 1 also changes
/// how wide the gain must be, so a rebuilt Gemma graph would differ in two
/// places at once. The comparison instead uses an **all-ones** gain at both
/// widths, which makes the gain's values identical under either grouping and
/// leaves the reduction as the only difference.
#[test]
fn per_head_normalization_is_load_bearing() {
    let grouped = grouped_norm_logits(4);
    let whole_row = grouped_norm_logits(1);
    assert_ne!(
        grouped, whole_row,
        "normalizing per head and normalizing the whole row must differ"
    );
    // Two groups differ from four as well, so the parameter is a count rather
    // than a grouped/ungrouped flag.
    assert_ne!(grouped_norm_logits(2), grouped);
    assert_ne!(grouped_norm_logits(2), whole_row);
}

/// A graph of embedding, one RMSNorm over `group` groups with a unit gain, one
/// single-head attention (the interpreter needs a state-touching node) and a
/// vocabulary projection. Small on purpose: the norm is the only thing that
/// varies, and the embedding rows carry lanes of very different magnitude so a
/// whole-row reduction visibly borrows one group's scale for the other.
fn grouped_norm_logits(group: u64) -> Vec<f32> {
    use moxie_graph::{GraphBuilder, IndexEncoding, TensorSpec, ValueRole};
    use moxie_types::{Dim, WeightPrecision};

    let (hidden, vocab) = (8u64, 5u64);
    let lanes = (hidden / group) as usize;
    let mut oracles = OracleRegistry::new();
    moxie_oracles::register(&mut oracles).unwrap();
    let mut g = GraphBuilder::new(moxie_oracles::HOST_REFERENCE, SymbolId(0));
    let index = TensorSpec::new(
        ValueRole::Index(IndexEncoding::U64),
        vec![Dim::symbol(SymbolId(0))],
    );
    let tokens = g.input("tokens", index.clone());
    let positions = g.input("absolute positions", index);
    let mut weights = Bindings::new();
    let bf16 = |data: Vec<f32>, shape: Vec<usize>| {
        Value::Float(moxie_engine::HostTensor::bf16(data, shape).unwrap())
    };

    let table = g
        .weight(
            "embedding",
            TensorSpec::new(
                ValueRole::Weight(WeightPrecision::new(Precision::Bf16).unwrap()),
                vec![Dim::constant(vocab), Dim::constant(hidden)],
            ),
        )
        .unwrap();
    // Two halves of very different magnitude, varying within each half and
    // between tokens. Without the intra-group variation every row would
    // normalize to the same vector and the comparison would be vacuous --
    // which is how the first version of this fixture managed to produce
    // identical logits under both groupings.
    let rows: Vec<f32> = (0..vocab * hidden)
        .map(|i| {
            let magnitude = if i % hidden < hidden / 2 { 0.0625 } else { 8.0 };
            magnitude * ((i % 5) + 1) as f32
        })
        .collect();
    weights.set(table, bf16(rows, vec![vocab as usize, hidden as usize]));

    let gain = g
        .weight(
            "unit gain",
            TensorSpec::new(
                ValueRole::Weight(WeightPrecision::new(Precision::Bf16).unwrap()),
                vec![Dim::constant(hidden / group)],
            ),
        )
        .unwrap();
    weights.set(gain, bf16(vec![1.0; lanes], vec![lanes]));

    let embedded = g
        .node(
            OpParams::Embedding {
                vocab,
                hidden,
                scale: 1.0,
            },
            &[tokens, table],
        )
        .unwrap();
    let normed = g
        .node(
            OpParams::RmsNorm {
                hidden,
                group,
                eps: 1e-6,
            },
            &[embedded, gain],
        )
        .unwrap();
    // One head as wide as the row, reading the normalized value as its own
    // query, key and value. It carries the norm's difference into the logits
    // without introducing a weight that could carry one of its own.
    let attended = g
        .node(
            OpParams::Attention {
                heads: 1,
                kv_heads: 1,
                head_dim: hidden,
                scale: 1.0,
                visibility: moxie_graph::Visibility::Causal,
                layer: 0,
            },
            &[normed, normed, normed, positions],
        )
        .unwrap();
    let logits = g
        .node(
            OpParams::VocabProjection {
                vocab,
                hidden,
                softcap: None,
            },
            &[attended, table],
        )
        .unwrap();
    let graph = g.finish(logits, &oracles).unwrap();

    let prompt: Vec<u64> = (0..vocab).collect();
    let mut dense = SequenceState::new([StateKind::KvPages]);
    dense.append_prompt(ROOT, prompt.len() as u64).unwrap();
    let mut cache = KvCache::for_branch(1, &dense, ROOT).unwrap();
    let mut bindings = weights.clone();
    bindings.set(tokens, Value::Index(prompt.clone()));
    bindings.set(positions, Value::Index((0..prompt.len() as u64).collect()));
    Interpreter::new()
        .run(
            &graph,
            &bindings,
            &mut dense,
            ROOT,
            &mut cache,
            &Cancel::never(),
        )
        .unwrap()
        .logits
        .data()
        .to_vec()
}

#[test]
fn the_global_layers_carry_no_value_projection_and_k_still_differs_from_v() {
    // `attention_k_eq_v` is graph composition, not a runtime branch: the global
    // layer has one projection feeding two normalizations. If V were taken
    // after the key norm and rotation, K and V would be identical tensors and
    // the layer would be a different function.
    let config = gemma::Shape::A.config();
    let model = Gemma4Text::reduced(config.clone(), "synthetic").unwrap();
    let mut oracles = OracleRegistry::new();
    moxie_oracles::register(&mut oracles).unwrap();
    let composed = model.compose(&oracles, SymbolId(0)).unwrap();

    // Layer 5 is global under a stride of six; layers 0..4 are not.
    assert!(model.global_layer(5));
    let v_roles: Vec<u32> = composed
        .weights
        .iter()
        .filter(|b| b.role.name == "v_proj")
        .filter_map(|b| b.role.layer)
        .collect();
    assert_eq!(v_roles, vec![0, 1, 2, 3, 4]);

    // The global attention node's key and value operands are different values:
    // the key went through its own norm and a rotation, the value did not.
    let attention: Vec<_> = composed
        .graph
        .nodes()
        .iter()
        .filter(|n| matches!(n.params, OpParams::Attention { layer: 5, .. }))
        .collect();
    assert_eq!(attention.len(), 1);
    let node = attention[0];
    assert_ne!(
        node.inputs[1], node.inputs[2],
        "K and V must not be the same graph value after the layer"
    );
}

#[test]
fn a_sliding_layer_and_a_global_layer_declare_different_visibility() {
    let config = gemma::Shape::A.config();
    let window = config.sliding_window;
    let model = Gemma4Text::reduced(config, "synthetic").unwrap();
    let mut oracles = OracleRegistry::new();
    moxie_oracles::register(&mut oracles).unwrap();
    let composed = model.compose(&oracles, SymbolId(0)).unwrap();
    let mut seen = Vec::new();
    for node in composed.graph.nodes() {
        if let OpParams::Attention {
            layer, visibility, ..
        } = node.params
        {
            seen.push((layer, visibility));
        }
    }
    seen.sort_by_key(|(l, _)| *l);
    assert_eq!(seen.len(), 6);
    for (layer, visibility) in &seen {
        let want = if *layer == 5 {
            moxie_graph::Visibility::Causal
        } else {
            moxie_graph::Visibility::SlidingWindow { window }
        };
        assert_eq!(*visibility, want, "layer {layer}");
    }
}

#[test]
fn the_reduced_embedding_scale_matches_the_format_crate() {
    // The model module rounds `sqrt(hidden)` to BF16 itself, because a model
    // crate may not depend on `moxie-format`. That is only acceptable while the
    // two agree, so this checks the sizes the fixtures use and the artifact's.
    for hidden in [12u64, 24, 64, 128, 1024, 4096, 5376, 21504] {
        let got = embedding_scale(hidden);
        let exact = (hidden as f64).sqrt() as f32;
        let want =
            moxie_format::bf16::bf16_bits_to_f32(moxie_format::bf16::f32_to_bf16_bits(exact));
        assert_eq!(got.to_bits(), want.to_bits(), "hidden {hidden}");
    }
}

#[test]
fn the_cli_selects_the_reduced_shapes_and_agrees_with_the_service() {
    for (flag, shape) in [("gemma-a", Shape::GemmaA), ("gemma-b", Shape::GemmaB)] {
        let args: Vec<String> = [
            "diagnostic",
            "--shape",
            flag,
            "--prompt",
            "0,1,2,3,4",
            "--max-new",
            "3",
            "--chunk",
            "2",
            "--temperature",
            "0.5",
            "--seed",
            "9",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let options = Options::parse(&args).unwrap();
        assert_eq!(options.shape, shape);
        assert_eq!(options.shape.name(), flag);
        // The surface says what the graph is not, before it says anything else.
        let reduction = options.shape.reduction();
        for expected in ["synthetic-bf16-weights", "text-only"] {
            assert!(reduction.contains(expected), "{reduction}");
        }
        // And the two task 0017 removed are gone, not merely unmentioned: a
        // disclosure that still named them would be claiming a limit the store
        // no longer has.
        for gone in ["uniform-kv-geometry", "no-window-eviction"] {
            assert!(!reduction.contains(gone), "{reduction}");
        }

        let fixture = options.shape.build().unwrap();
        let mut owner = ledger();
        let mut service = GenerationService::new(&mut owner, fixture.program());
        service.start(options.request()).unwrap();
        let mut rendered = Vec::new();
        assert!(render(&mut service, &Cancel::never(), &mut rendered).unwrap());
        let printed: Vec<u32> = String::from_utf8(rendered)
            .unwrap()
            .lines()
            .filter_map(|l| l.strip_prefix("event=token id="))
            .filter_map(|l| l.split_whitespace().next())
            .map(|id| id.parse().unwrap())
            .collect();

        let again = options.shape.build().unwrap();
        let mut owner = ledger();
        let mut service = GenerationService::new(&mut owner, again.program());
        service.start(options.request()).unwrap();
        assert_eq!(printed, drain(&mut service), "{flag}");
        assert_eq!(printed.len(), 3);
    }
    // The non-Gemma shapes carry no reduction disclosure, because they make no
    // model-shaped claim to qualify.
    assert!(Shape::SyntheticA.reduction().is_empty());
    assert!(Shape::SyntheticB.reduction().is_empty());
}

#[test]
fn cancellation_releases_everything_and_a_second_generation_starts_empty() {
    let fixture = gemma::build(gemma::Shape::A).unwrap();
    let prompt: Vec<u32> = (0..9).collect();
    let request = Request {
        prompt: &prompt,
        max_new_tokens: 3,
        prefill_chunk: 2,
        temperature: 0.0,
        seed: 1,
    };
    let mut owner = ledger();
    let mut service = GenerationService::new(&mut owner, fixture.program());

    // A reference run first, so the restart can be compared against it.
    service.start(request).unwrap();
    let reference = drain(&mut service);

    let mut cancelled_at_least_once = false;
    for boundary in 0..48u64 {
        service.start(request).unwrap();
        let cancel = Cancel::after(boundary);
        let mut committed = Vec::new();
        let mut terminal = 0;
        while let Some(event) = service.next_event(&cancel) {
            match event {
                Event::Token { id, .. } => committed.push(id),
                Event::Cancelled { usage } => {
                    terminal += 1;
                    cancelled_at_least_once = true;
                    assert_eq!(usage.completion_tokens, committed.len());
                }
                Event::Finished { usage } => {
                    terminal += 1;
                    assert_eq!(usage.completion_tokens, committed.len());
                }
                Event::Failed { error, .. } => panic!("{error}"),
                _ => {}
            }
        }
        assert_eq!(terminal, 1, "boundary {boundary}");
        assert!(service.is_idle(), "boundary {boundary}");
        assert_eq!(service.charged_bytes(), 0, "boundary {boundary}");
        assert!(
            committed.len() <= reference.len() && reference.starts_with(&committed),
            "boundary {boundary}: {committed:?} is not a prefix of {reference:?}"
        );
        // And the next generation is unaffected by the one that stopped.
        service.start(request).unwrap();
        assert_eq!(drain(&mut service), reference, "restart after {boundary}");
    }
    assert!(cancelled_at_least_once, "no boundary actually cancelled");
}

#[test]
fn one_generation_at_a_time_and_invalid_requests_are_refused() {
    let fixture = gemma::build(gemma::Shape::B).unwrap();
    let mut owner = ledger();
    let mut service = GenerationService::new(&mut owner, fixture.program());
    let good = Request {
        prompt: &[0, 1, 2],
        max_new_tokens: 1,
        prefill_chunk: 2,
        temperature: 0.0,
        seed: 0,
    };
    service.start(good).unwrap();
    assert!(matches!(service.start(good), Err(StartError::Busy)));
    drain(&mut service);

    // A token outside the reduced vocabulary of 7.
    assert!(matches!(
        service.start(Request {
            prompt: &[0, 99],
            ..good
        }),
        Err(StartError::Rejected(_))
    ));
    assert!(service.is_idle());
    assert_eq!(service.charged_bytes(), 0);
}

#[test]
fn an_inadmissible_budget_leaves_no_charge() {
    let fixture = gemma::build(gemma::Shape::A).unwrap();
    let mut owner = Ledger::new([CapacitySnapshot::new(Scope::Host, 1 << 10, 1).unwrap()]).unwrap();
    let mut service = GenerationService::new(&mut owner, fixture.program());
    let started = service.start(Request {
        prompt: &[0, 1, 2],
        max_new_tokens: 1,
        prefill_chunk: 2,
        temperature: 0.0,
        seed: 0,
    });
    assert!(started.is_err(), "a 1 KiB budget must not admit this graph");
    assert!(service.is_idle());
    assert_eq!(service.charged_bytes(), 0);
}

#[test]
fn a_graph_whose_layers_disagree_on_geometry_is_paged_at_each_layers_own_width() {
    // Task 0016 refused this outright and recorded uniform geometry as a
    // declared reduction. Task 0017 admits it: each layer is paged at its own
    // width, which is what the artifact's 16x256 sliding layers beside its
    // 4x512 global ones require.
    //
    // The check is on the stored rows, not on the sampled token. A greedy
    // argmax can coincide between two different graphs, and asserting it
    // differs would be a flaky test of nothing; the row widths are the thing
    // the admission actually decides.
    let head_dim = 4usize;
    let (uniform, mixed) = two_layer_graphs();
    let mut reserved = Vec::new();
    let mut logits = Vec::new();
    for (graph, second_kv_heads) in [(uniform, 2usize), (mixed, 1usize)] {
        // Through the service, so admission is the thing being exercised.
        let mut owner = ledger();
        let mut service = GenerationService::new(&mut owner, graph.program());
        service
            .start(Request {
                prompt: &[0, 1, 2],
                max_new_tokens: 1,
                prefill_chunk: 2,
                temperature: 0.0,
                seed: 0,
            })
            .unwrap();
        let Some(Event::Admitted { reserved_bytes, .. }) = service.next_event(&Cancel::never())
        else {
            panic!("the mixed graph must be admitted, not refused");
        };
        reserved.push(reserved_bytes);
        drain(&mut service);
        assert!(service.is_idle());
        assert_eq!(service.charged_bytes(), 0);

        // And directly, to read the stored row widths back.
        let mut owner = ledger();
        let geometry = KvGeometry {
            layers: vec![
                LayerKv {
                    kv_heads: 2,
                    key_dim: head_dim,
                    value_dim: head_dim,
                    retention: Retention::All,
                },
                LayerKv {
                    kv_heads: second_kv_heads,
                    key_dim: head_dim,
                    value_dim: head_dim,
                    retention: Retention::All,
                },
            ],
            precision: Precision::Bf16,
            page_tokens: 2,
            max_tokens: 8,
            tentative_rows: 3,
        };
        let mut pages = PagedSequence::new(&mut owner, geometry).unwrap();
        pages.append_prompt(3).unwrap();
        let execution = PagedExecution::bind(
            &graph.graph,
            &graph.weights,
            graph.tokens,
            graph.positions,
            &mut pages,
        )
        .unwrap();
        let txn = pages.begin().unwrap();
        let output = execution
            .run(&mut pages, txn, &[0, 1, 2], &[0, 1, 2], &Cancel::never())
            .unwrap();
        logits.push(output.logits().data().to_vec());
        pages.commit_prefix(txn, 0).unwrap();
        for position in 0..3 {
            // Two bytes per BF16 element, `kv_heads * head_dim` elements, and
            // the two layers disagree about how many that is.
            assert_eq!(pages.row(0, position).unwrap().key.len(), 2 * head_dim * 2);
            assert_eq!(
                pages.row(1, position).unwrap().key.len(),
                second_kv_heads * head_dim * 2
            );
        }
        pages.close(&mut owner).unwrap();
    }
    // The narrower second layer is admitted for fewer bytes, and reaches the
    // arithmetic: same weights, different stored width, different logits.
    assert!(
        reserved[1] < reserved[0],
        "a narrower second layer must be admitted for less: {reserved:?}"
    );
    assert_ne!(logits[0], logits[1]);
}

/// Two minimal two-layer graphs: one with a single key/value geometry, one
/// whose second layer halves its key/value heads.
///
/// Written directly rather than rewritten from the Gemma graph, because the
/// point is the engine's admission rule and a small graph makes the difference
/// between the two the only thing under test.
fn two_layer_graphs() -> (moxie_cli::fixture::Fixture, moxie_cli::fixture::Fixture) {
    (build_two_layer(2), build_two_layer(1))
}

fn build_two_layer(second_layer_kv_heads: u64) -> moxie_cli::fixture::Fixture {
    use moxie_graph::{GraphBuilder, IndexEncoding, TensorSpec, ValueRole, Visibility};
    use moxie_types::{Dim, WeightPrecision};

    let (heads, head_dim, hidden, vocab) = (2u64, 4u64, 8u64, 5u64);
    let mut oracles = OracleRegistry::new();
    moxie_oracles::register(&mut oracles).unwrap();
    let mut g = GraphBuilder::new(moxie_oracles::HOST_REFERENCE, SymbolId(0));
    let index = TensorSpec::new(
        ValueRole::Index(IndexEncoding::U64),
        vec![Dim::symbol(SymbolId(0))],
    );
    let tokens = g.input("tokens", index.clone());
    let positions = g.input("absolute positions", index);
    let mut weights = Bindings::new();
    let parameter = |g: &mut GraphBuilder,
                     weights: &mut Bindings<Value>,
                     name: &str,
                     rows: u64,
                     cols: u64|
     -> ValueId {
        let id = g
            .weight(
                name,
                TensorSpec::new(
                    ValueRole::Weight(WeightPrecision::new(Precision::Bf16).unwrap()),
                    vec![Dim::constant(rows), Dim::constant(cols)],
                ),
            )
            .unwrap();
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 7 % 23) as f32 - 11.0) / 32.0)
            .collect();
        weights.set(
            id,
            Value::Float(
                moxie_engine::HostTensor::bf16(data, vec![rows as usize, cols as usize]).unwrap(),
            ),
        );
        id
    };
    let table = parameter(&mut g, &mut weights, "embedding", vocab, hidden);
    let mut stream = g
        .node(
            OpParams::Embedding {
                vocab,
                hidden,
                scale: 1.0,
            },
            &[tokens, table],
        )
        .unwrap();
    for layer in 0..2u32 {
        let kv_heads = if layer == 1 { second_layer_kv_heads } else { 2 };
        let wq = parameter(&mut g, &mut weights, "q", heads * head_dim, hidden);
        let wk = parameter(&mut g, &mut weights, "k", kv_heads * head_dim, hidden);
        let wv = parameter(&mut g, &mut weights, "v", kv_heads * head_dim, hidden);
        let wo = parameter(&mut g, &mut weights, "o", hidden, heads * head_dim);
        let lin = |g: &mut GraphBuilder, x: ValueId, w: ValueId, i: u64, o: u64| {
            g.node(
                OpParams::Linear {
                    in_features: i,
                    out_features: o,
                    bias: false,
                },
                &[x, w],
            )
            .unwrap()
        };
        let q = lin(&mut g, stream, wq, hidden, heads * head_dim);
        let k = lin(&mut g, stream, wk, hidden, kv_heads * head_dim);
        let v = lin(&mut g, stream, wv, hidden, kv_heads * head_dim);
        let a = g
            .node(
                OpParams::Attention {
                    heads,
                    kv_heads,
                    head_dim,
                    scale: 1.0,
                    visibility: Visibility::Causal,
                    layer,
                },
                &[q, k, v, positions],
            )
            .unwrap();
        let projected = lin(&mut g, a, wo, heads * head_dim, hidden);
        stream = g
            .node(OpParams::Residual { scale: 1.0 }, &[stream, projected])
            .unwrap();
    }
    let logits = g
        .node(
            OpParams::VocabProjection {
                vocab,
                hidden,
                softcap: None,
            },
            &[stream, table],
        )
        .unwrap();
    moxie_cli::fixture::Fixture {
        graph: g.finish(logits, &oracles).unwrap(),
        weights,
        tokens,
        positions,
    }
}

#[test]
fn the_artifact_geometry_is_recorded_and_not_runnable_here() {
    use moxie_models::gemma4::ARTIFACT;
    // The numbers `docs/models/gemma4.md` records, asserted so the record and
    // the code cannot drift apart silently.
    assert_eq!(ARTIFACT.hidden, 5376);
    assert_eq!(ARTIFACT.layers, 60);
    assert_eq!(ARTIFACT.heads, 32);
    assert_eq!(
        (ARTIFACT.local_kv_heads, ARTIFACT.local_head_dim),
        (16, 256)
    );
    assert_eq!(
        (ARTIFACT.global_kv_heads, ARTIFACT.global_head_dim),
        (4, 512)
    );
    assert_eq!(ARTIFACT.intermediate, 21504);
    assert_eq!(ARTIFACT.vocab, 262_144);
    assert_eq!(ARTIFACT.sliding_window, 1024);
    assert_eq!(ARTIFACT.final_logit_softcap, 30.0);
    assert_eq!(ARTIFACT.max_trained_position, 262_144);
    assert_eq!((0..60).filter(|l| ARTIFACT.global_layer(*l)).count(), 10);

    // And it is not a `TextConfig`: no uniform configuration describes layers
    // whose key/value heads and head dimensions both differ.
    let reduced = gemma::Shape::A.config();
    assert!(reduced.hidden < ARTIFACT.hidden);
    assert!(reduced.max_trained_position <= 256);
}
