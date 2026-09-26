//! Full-size Gemma 4 text graphs composed directly from the checkpoints
//! named in document 06's M6 exit route item 2 (ADR 0038: run source
//! checkpoints directly). Ignored by default: these need
//! `/fast/models/cyankiwi/gemma-4-31B-it-AWQ-8bit` and
//! `/fast/models/google/gemma-4-26B-A4B-it` present on this machine.
//!
//! Nothing here loads a weight or runs a token: `from_checkpoint` composes
//! a graph and a role -> source-tensor map, and this file checks that
//! composition against the checkpoint's own declared numbers and index --
//! a second, independent reading of the same files, so the two only agree
//! if both read the checkpoint the same way.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use moxie_cli::gemma::from_checkpoint;
use moxie_format::checkpoint_config;
use moxie_models::gemma4::{ARTIFACT, ARTIFACT_A4B, ArtifactGeometry, TextConfig};
use moxie_storage::Shard;
use moxie_types::{Dim, SymbolId};

const DENSE: &str = "/fast/models/cyankiwi/gemma-4-31B-it-AWQ-8bit";
const MOE: &str = "/fast/models/google/gemma-4-26B-A4B-it";

fn weight_map(dir: &Path) -> BTreeMap<String, String> {
    let text = std::fs::read_to_string(dir.join("model.safetensors.index.json"))
        .expect("the checkpoint's safetensors index");
    checkpoint_config::parse_index(&text).expect("a valid safetensors index")
}

fn open_shard<'a>(dir: &Path, shards: &'a mut BTreeMap<String, Shard>, file: &str) -> &'a Shard {
    if !shards.contains_key(file) {
        shards.insert(
            file.to_string(),
            Shard::open(&dir.join(file)).unwrap_or_else(|e| panic!("open {file}: {e}")),
        );
    }
    &shards[file]
}

/// A source tensor's logical shape: the safetensors header's own shape for a
/// direct tensor, or its `weight_shape` companion's declared value for a
/// packed INT8 one -- the exact same distinction `Shard::verify_tensor`'s
/// callers make when loading, not a second convention invented here.
fn source_shape(
    dir: &Path,
    map: &BTreeMap<String, String>,
    name: &str,
    shards: &mut BTreeMap<String, Shard>,
) -> Vec<u64> {
    if let Some(base) = name.strip_suffix(".weight_packed") {
        let shape_name = format!("{base}.weight_shape");
        let file = map
            .get(&shape_name)
            .unwrap_or_else(|| panic!("{shape_name} is not in the index"));
        let shard = open_shard(dir, shards, file);
        let bytes = shard
            .tensor_bytes(&shape_name)
            .unwrap_or_else(|e| panic!("read {shape_name}: {e}"));
        return bytes
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().expect("8-byte chunk")) as u64)
            .collect();
    }
    let file = map
        .get(name)
        .unwrap_or_else(|| panic!("{name} is not in the index"));
    let shard = open_shard(dir, shards, file);
    shard
        .header()
        .get(name)
        .unwrap_or_else(|e| panic!("{name} header: {e}"))
        .shape
        .clone()
}

/// The exact index name a role's bare source resolves to: a linear's base
/// name gets whichever suffix the checkpoint actually stored it under, and
/// everything else already carries `.weight`.
fn resolved_name(source: &str, map: &BTreeMap<String, String>) -> Option<String> {
    if map.contains_key(source) {
        return Some(source.to_string());
    }
    [".weight", ".weight_packed"]
        .into_iter()
        .map(|suffix| format!("{source}{suffix}"))
        .find(|candidate| map.contains_key(candidate))
}

fn assert_config_matches_artifact(config: &TextConfig, artifact: ArtifactGeometry) {
    assert_eq!(config.hidden, artifact.hidden, "hidden");
    assert_eq!(config.layers, artifact.layers, "layers");
    assert_eq!(config.heads, artifact.heads, "heads");
    assert_eq!(
        config.local_kv_heads, artifact.local_kv_heads,
        "local_kv_heads"
    );
    assert_eq!(
        config.local_head_dim, artifact.local_head_dim,
        "local_head_dim"
    );
    assert_eq!(
        config.global_kv_heads, artifact.global_kv_heads,
        "global_kv_heads"
    );
    assert_eq!(
        config.global_head_dim, artifact.global_head_dim,
        "global_head_dim"
    );
    assert_eq!(config.intermediate, artifact.intermediate, "intermediate");
    assert_eq!(config.vocab, artifact.vocab, "vocab");
    assert_eq!(
        config.global_stride, artifact.global_stride,
        "global_stride"
    );
    assert_eq!(
        config.sliding_window, artifact.sliding_window,
        "sliding_window"
    );
    assert_eq!(config.rms_eps, artifact.rms_eps, "rms_eps");
    assert_eq!(
        config.sliding_rope_theta, artifact.sliding_rope_theta,
        "sliding_rope_theta"
    );
    assert_eq!(
        config.global_rope_theta, artifact.global_rope_theta,
        "global_rope_theta"
    );
    assert_eq!(
        config.final_logit_softcap, artifact.final_logit_softcap,
        "final_logit_softcap"
    );
    assert_eq!(
        config.max_trained_position, artifact.max_trained_position,
        "max_trained_position"
    );
    assert_eq!(config.moe, artifact.moe, "moe");
}

fn check_checkpoint(dir_str: &str, artifact: ArtifactGeometry) {
    let dir = Path::new(dir_str);
    let graph = from_checkpoint(dir).expect("a full-size graph composes from the checkpoint");

    // (a)
    assert_config_matches_artifact(&graph.config, artifact);

    let map = weight_map(dir);
    let mut shards = BTreeMap::new();
    let mut bound_tensors = BTreeSet::new();

    for bound in &graph.composition.weights {
        let key = (bound.role.name.clone(), bound.role.layer, bound.role.expert);
        let source = graph
            .role_to_source_tensor
            .get(&key)
            .unwrap_or_else(|| panic!("{key:?} is not in the role -> source map"))
            .clone();

        if bound.role.name == "v_norm_unit_gain" {
            assert!(source.is_none(), "v_norm_unit_gain has a source tensor");
            continue;
        }
        let source = source.unwrap_or_else(|| panic!("{key:?} has no declared source tensor"));
        let resolved = resolved_name(&source, &map).unwrap_or_else(|| {
            panic!("{source} is not in the index under .weight or .weight_packed")
        });
        assert!(
            bound_tensors.insert(resolved.clone()),
            "{resolved} is bound by more than one role"
        );

        // (c)
        let spec = graph
            .composition
            .graph
            .spec(bound.value)
            .unwrap_or_else(|| panic!("{key:?} has no graph value"));
        let graph_shape: Vec<u64> = spec
            .shape
            .iter()
            .map(|d| match d {
                Dim::Const(n) => *n,
                other => panic!("{key:?} has a non-constant extent {other:?}"),
            })
            .collect();
        let source_shape = source_shape(dir, &map, &resolved, &mut shards);
        assert_eq!(
            graph_shape, source_shape,
            "{key:?} graph shape disagrees with {resolved}'s source shape"
        );
    }

    // (b): every text tensor the index declares is bound exactly once,
    // excluding vision/audio, the affine companions and `layer_scalar`
    // (consumed while building the config, not while composing).
    let expected: BTreeSet<String> = map
        .keys()
        .filter(|name| name.starts_with("model.language_model."))
        .filter(|name| !name.contains("vision") && !name.contains("audio"))
        .filter(|name| !name.ends_with(".weight_scale"))
        .filter(|name| !name.ends_with(".weight_shape"))
        .filter(|name| !name.ends_with(".layer_scalar"))
        .cloned()
        .collect();
    assert_eq!(bound_tensors, expected);

    // (d): the graph validated during composition (`finish` refuses an
    // unregistered operation or a shape mismatch), and composing again at
    // both row counts the task names succeeds too.
    let mut oracles = moxie_graph::OracleRegistry::new();
    moxie_oracles::register(&mut oracles).unwrap();
    let model = moxie_models::gemma4::Gemma4Text::full(graph.config.clone(), &graph.revision)
        .expect("the derived config still builds a full-size model");
    for rows in [1u32, 8] {
        model
            .compose(&oracles, SymbolId(rows))
            .unwrap_or_else(|e| panic!("composition at rows {rows} failed: {e}"));
    }
}

#[test]
#[ignore = "needs the checkpoints under /fast/models"]
fn dense_checkpoint_composes_a_full_size_graph() {
    check_checkpoint(DENSE, ARTIFACT);
}

#[test]
#[ignore = "needs the checkpoints under /fast/models"]
fn moe_checkpoint_composes_a_full_size_graph() {
    check_checkpoint(MOE, ARTIFACT_A4B);
}
