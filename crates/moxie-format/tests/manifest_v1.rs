//! Manifest v1 acceptance: every documented rejection rule actually rejects.
//!
//! Malformed manifests are built here as strings, one per rule, so each
//! rejection names the rule it proves.

use std::collections::BTreeMap;

use moxie_format::manifest::{
    self, MAX_ARCH_DEPTH, MAX_ARCH_NODES, MAX_MANIFEST_BYTES, MAX_TENSORS, SCHEMA_VERSION,
};

const HEX: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

fn header() -> String {
    format!(
        r#"schema_version = {SCHEMA_VERSION}
required_features = []
endianness = "little"

[source]
model = "example"
revision = "r1"
license = "apache-2.0"
[[source.files]]
path = "orig.bin"
sha256 = "{HEX}"

[tokenizer]
name = "tok"
version = "1"
digest = "{HEX}"

[template]
name = "tpl"
version = "1"
digest = "{HEX}"

[architecture]
name = "arch"
version = "1"
[architecture.metadata]
hidden = 8

[provenance]
scale_convention = "affine-v1"
quantizer = "none"
calibration = "none"
"#
    )
}

fn bf16_tensor(
    role: &str,
    shape: &str,
    chunk: &str,
    offset: i64,
    length: i64,
    order: i64,
) -> String {
    format!(
        r#"[[tensors]]
role = "{role}"
shape = [{shape}]
precision = "bf16-v1"
chunk = "{chunk}"
offset = {offset}
length = {length}
sha256 = "{HEX}"
alignment = 2
logical_order = {order}
"#
    )
}

fn manifest_with(tensors: &str) -> String {
    format!(
        r#"{h}
{tensors}
[[excluded]]
role = "old"
reason = "removed upstream"

[completeness]
status = "complete"
missing = []
"#,
        h = header()
    )
}

fn valid_one() -> String {
    manifest_with(&bf16_tensor("w", "2, 2", "c.bin", 0, 8, 0))
}

fn err_contains(result: moxie_types::Result<impl std::fmt::Debug>, needle: &str) -> String {
    match result {
        Ok(v) => panic!("expected rejection containing {needle:?}, got Ok({v:?})"),
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains(needle),
                "error {msg:?} does not name the rule ({needle:?})"
            );
            assert_eq!(e.kind(), "invalid_artifact");
            msg
        }
    }
}

#[test]
fn the_committed_fixture_parses() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/fixtures/manifest.toml"
    ))
    .expect("committed fixture exists");
    let m = manifest::parse(&text).expect("fixture is valid");
    assert_eq!(m.tensors.len(), 2);
    assert!(matches!(
        m.completeness,
        moxie_format::manifest::Completeness::Complete
    ));
    // Identity is stable across parses.
    let again = manifest::parse(&text).unwrap();
    assert_eq!(
        manifest::artifact_identity(&m),
        manifest::artifact_identity(&again)
    );
}

#[test]
fn schema_version_is_refused_by_version_before_any_other_field() {
    let bad = manifest_with(&bf16_tensor("w", "2, 2", "c.bin", 0, 8, 0)).replacen(
        "schema_version = 1",
        "schema_version = 2",
        1,
    );
    err_contains(manifest::parse(&bad), "version");
}

#[test]
fn a_future_version_is_refused_even_when_v1_fields_are_gone() {
    // First review, reproduced: `parse("schema_version = 2\\n")` used to
    // report a missing `required_features`, because full v1 deserialization
    // ran before the version check. The version is gated first now, so a
    // future schema with missing or changed v1 fields still fails on version.
    err_contains(
        manifest::parse("schema_version = 2\n"),
        "refused by version",
    );
    let future = "schema_version = 99\nrequired_features = []\nendianness = \"sideways\"\n";
    err_contains(manifest::parse(future), "refused by version");
    // A manifest with no version at all says so, rather than failing on
    // whatever v1 field the deserializer reaches first.
    match manifest::parse("endianness = \"little\"\n") {
        Ok(_) => panic!("a versionless manifest must not parse"),
        Err(e) => assert!(e.to_string().contains("schema_version"), "{e}"),
    }
}
#[test]
fn unknown_required_features_are_refused_by_name() {
    let bad = manifest_with(&bf16_tensor("w", "2, 2", "c.bin", 0, 8, 0)).replacen(
        "required_features = []",
        "required_features = [\"future-x\"]",
        1,
    );
    err_contains(manifest::parse(&bad), "future-x");
}

#[test]
fn an_unrecognized_key_is_a_rejection_not_a_silent_drop() {
    let bad = valid_one().replacen(
        "endianness = \"little\"",
        "endianness = \"little\"\ntypo_field = 1",
        1,
    );
    let msg = err_contains(manifest::parse(&bad), "typo_field");
    assert!(
        msg.contains("does not parse") || msg.contains("unknown"),
        "{msg}"
    );
}

#[test]
fn big_endian_is_refused_rather_than_byte_swapped() {
    let bad = valid_one().replacen("little", "big", 1);
    err_contains(manifest::parse(&bad), "byte-swapped");
}

#[test]
fn a_missing_required_section_is_a_rejection_not_a_default() {
    // Drop the whole provenance table: absence must reject, never default.
    let bad = valid_one()
        .lines()
        .filter(|l| {
            !l.starts_with("scale_convention")
                && !l.starts_with("quantizer")
                && !l.starts_with("calibration")
                && *l != "[provenance]"
        })
        .collect::<Vec<_>>()
        .join("\n");
    match manifest::parse(&bad) {
        Ok(_) => panic!("a manifest without provenance must not parse"),
        Err(e) => assert_eq!(e.kind(), "invalid_artifact"),
    }
}

#[test]
fn duplicate_roles_are_rejected_before_either_wins() {
    let two = format!(
        "{}{}",
        bf16_tensor("w", "2, 2", "c.bin", 0, 8, 0),
        bf16_tensor("w", "2, 2", "c.bin", 8, 8, 1)
    );
    err_contains(
        manifest::parse(&manifest_with(&two)),
        "duplicate tensor role",
    );
}

#[test]
fn duplicate_logical_orders_are_rejected() {
    let two = format!(
        "{}{}",
        bf16_tensor("a", "1", "c.bin", 0, 2, 0),
        bf16_tensor("b", "1", "c.bin", 2, 2, 0)
    );
    err_contains(manifest::parse(&manifest_with(&two)), "logical_order");
}

#[test]
fn offset_plus_length_that_wraps_is_an_error_not_an_in_range_value() {
    // offset = u64::MAX, length 16: the sum wraps, so the check must use
    // checked_add rather than trusting the wrapped value.
    let m = manifest::parse(&manifest_with(&bf16_tensor("w", "1", "c.bin", 0, 2, 0))).unwrap();
    let mut chunks = BTreeMap::new();
    chunks.insert("c.bin".to_string(), 16u64);
    // Sanity: a fitting range validates.
    manifest::validate_chunks(&m, &chunks).unwrap();
    // Now the wrapping range, expressed directly (TOML i64 cannot hold u64::MAX,
    // so this exercises validate_chunks' checked_add via a hand-built manifest).
    let mut evil = m;
    evil.tensors[0].offset = u64::MAX;
    evil.tensors[0].length = 16;
    err_contains(manifest::validate_chunks(&evil, &chunks), "overflows");
}

#[test]
fn a_manifest_describing_more_bytes_than_exist_is_truncation() {
    let m = manifest::parse(&manifest_with(&bf16_tensor("w", "2, 2", "c.bin", 0, 8, 0))).unwrap();
    let mut chunks = BTreeMap::new();
    chunks.insert("c.bin".to_string(), 7u64);
    err_contains(manifest::validate_chunks(&m, &chunks), "truncation");
}

#[test]
fn a_chunk_that_does_not_exist_is_rejected() {
    let m = manifest::parse(&manifest_with(&bf16_tensor("w", "2, 2", "c.bin", 0, 8, 0))).unwrap();
    let chunks = BTreeMap::new();
    err_contains(manifest::validate_chunks(&m, &chunks), "does not exist");
}

#[test]
fn overlapping_ranges_reject_including_total_containment() {
    // Partial overlap.
    let two = format!(
        "{}{}",
        bf16_tensor("a", "4", "c.bin", 0, 8, 0),
        bf16_tensor("b", "4", "c.bin", 4, 8, 1)
    );
    err_contains(manifest::parse(&manifest_with(&two)), "overlap");
    // Total containment: b wholly inside a. A pairwise "starts inside" test
    // that only compares starts would miss nothing here, but the historically
    // missed case is containment reported the other way: sorting by offset and
    // comparing neighbours catches both.
    let two = format!(
        "{}{}",
        bf16_tensor("big", "8", "c.bin", 0, 16, 0),
        bf16_tensor("small", "2", "c.bin", 4, 4, 1)
    );
    err_contains(manifest::parse(&manifest_with(&two)), "overlap");
    // Adjacent ranges are fine.
    let two = format!(
        "{}{}",
        bf16_tensor("a", "4", "c.bin", 0, 8, 0),
        bf16_tensor("b", "4", "c.bin", 8, 8, 1)
    );
    manifest::parse(&manifest_with(&two)).expect("adjacent ranges are valid");
    // Same offsets in different chunks are fine.
    let two = format!(
        "{}{}",
        bf16_tensor("a", "4", "c0.bin", 0, 8, 0),
        bf16_tensor("b", "4", "c1.bin", 0, 8, 1)
    );
    manifest::parse(&manifest_with(&two)).expect("per-chunk overlap only");
}

#[test]
fn alignment_must_be_a_power_of_two_and_divide_the_offset() {
    let bad = manifest_with(&bf16_tensor("w", "2, 2", "c.bin", 0, 8, 0)).replacen(
        "alignment = 2",
        "alignment = 3",
        1,
    );
    err_contains(manifest::parse(&bad), "power of two");
    let bad = manifest_with(&bf16_tensor("w", "2, 2", "c.bin", 2, 8, 0)).replacen(
        "alignment = 2",
        "alignment = 4",
        1,
    );
    err_contains(manifest::parse(&bad), "multiple of alignment");
}

#[test]
fn shape_and_byte_count_that_disagree_are_rejected() {
    // 2x2 BF16 needs 8 bytes, not 10.
    let bad = manifest_with(&bf16_tensor("w", "2, 2", "c.bin", 0, 10, 0));
    err_contains(manifest::parse(&bad), "disagree");
}

#[test]
fn a_shape_product_that_overflows_is_an_error_not_a_wrap() {
    // u64::MAX/2 squared overflows u64.
    let huge = format!("{}, {}", u64::MAX / 2, u64::MAX / 2);
    let bad = manifest_with(&bf16_tensor("w", &huge, "c.bin", 0, 8, 0));
    err_contains(manifest::parse(&bad), "overflow");
}

#[test]
fn zero_or_negative_dimensions_are_rejected() {
    let bad = manifest_with(&bf16_tensor("w", "2, 0", "c.bin", 0, 8, 0));
    err_contains(manifest::parse(&bad), "positive");
}

#[test]
fn affine_descriptor_closed_set_is_enforced_at_the_manifest() {
    let affine = |extra: &str| {
        format!(
            r#"[[tensors]]
role = "q"
shape = [8, 32]
precision = "affine-int4-v1"
chunk = "c.bin"
offset = 0
length = 128
sha256 = "{HEX}"
alignment = 16
logical_order = 0
{extra}
"#
        )
    };
    let good =
        affine("group_rule = \"contiguous-32\"\nscale_dtype = \"f16\"\nzero_point = \"symmetric\"");
    manifest::parse(&manifest_with(&good)).expect("closed descriptor validates");
    for (bad_extra, needle) in [
        (
            "group_rule = \"contiguous-64\"\nscale_dtype = \"f16\"\nzero_point = \"symmetric\"",
            "closed set",
        ),
        (
            "group_rule = \"contiguous-32\"\nscale_dtype = \"f8\"\nzero_point = \"symmetric\"",
            "scale_dtype",
        ),
        (
            "group_rule = \"contiguous-32\"\nscale_dtype = \"f16\"\nzero_point = \"sometimes\"",
            "zero_point",
        ),
        (
            "scale_dtype = \"f16\"\nzero_point = \"symmetric\"",
            "group_rule",
        ),
    ] {
        err_contains(manifest::parse(&manifest_with(&affine(bad_extra))), needle);
    }
    // group-128 is the other closed value; per-channel is explicit.
    for rule in ["contiguous-128", "per-channel"] {
        let ok = affine(&format!(
            "group_rule = \"{rule}\"\nscale_dtype = \"bf16\"\nzero_point = \"per-group\""
        ));
        manifest::parse(&manifest_with(&ok)).expect(rule);
    }
    // A BF16 tensor must not carry affine fields.
    let bad = format!(
        "{}\ngroup_rule = \"contiguous-32\"\nscale_dtype = \"f16\"\nzero_point = \"symmetric\"",
        bf16_tensor("w", "2, 2", "c.bin", 0, 8, 0).trim_end()
    );
    err_contains(
        manifest::parse(&manifest_with(&bad)),
        "must not carry affine",
    );
    // Unknown precision string.
    let bad =
        manifest_with(&bf16_tensor("w", "2, 2", "c.bin", 0, 8, 0)).replacen("bf16-v1", "fp8-v1", 1);
    err_contains(manifest::parse(&bad), "not one of");
}

#[test]
fn group_index_map_must_match_the_row_and_its_groups() {
    let affine_with = |idx: &str| {
        format!(
            r#"[[tensors]]
role = "q"
shape = [1, 64]
precision = "affine-int8-v1"
chunk = "c.bin"
offset = 0
length = 64
sha256 = "{HEX}"
alignment = 16
logical_order = 0
group_rule = "contiguous-32"
scale_dtype = "f32"
zero_point = "per-group"
group_index = [{idx}]
"#
        )
    };
    // 64 columns alternating groups 0/1: valid.
    let idx: Vec<String> = (0..64).map(|k| (k % 2).to_string()).collect();
    manifest::parse(&manifest_with(&affine_with(&idx.join(", ")))).expect("valid map");
    // Wrong length.
    let idx: Vec<String> = (0..63).map(|_| "0".to_string()).collect();
    err_contains(
        manifest::parse(&manifest_with(&affine_with(&idx.join(", ")))),
        "entries for 64 input channels",
    );
    // Out-of-range group id (only groups 0,1 exist at width 64 / size 32).
    let idx: Vec<String> = (0..64).map(|_| "2".to_string()).collect();
    err_contains(
        manifest::parse(&manifest_with(&affine_with(&idx.join(", ")))),
        "only 2 groups",
    );
}

#[test]
fn a_group_index_that_orphans_a_scale_is_rejected_by_the_shared_descriptor() {
    // First review, reproduced: `[1,64]` contiguous-32 with an all-zero map
    // passed the manifest's own checks while `AffineDescriptor::validate`
    // rejects it (group 1 owns a scale no column reads). The manifest now
    // runs the shared validator, so the disagreement is a rejection here.
    let idx: Vec<String> = (0..64).map(|_| "0".to_string()).collect();
    let text = format!(
        r#"[[tensors]]
role = "q"
shape = [1, 64]
precision = "affine-int8-v1"
chunk = "c.bin"
offset = 0
length = 64
sha256 = "{HEX}"
alignment = 16
logical_order = 0
group_rule = "contiguous-32"
scale_dtype = "f32"
zero_point = "per-group"
group_index = [{}]
"#,
        idx.join(", ")
    );
    err_contains(
        manifest::parse(&manifest_with(&text)),
        "shared affine descriptor",
    );
}

#[test]
fn chunk_paths_are_confined_on_the_string_before_any_join() {
    // Literal TOML strings (single quotes): no escape processing, so what is
    // written is what the validator sees. With basic strings `\` would be
    // needed for one backslash, and `\b` would unescape to a backspace.
    let with_chunk = |chunk_toml: &str| {
        manifest_with(&bf16_tensor("w", "1", "PLACEHOLDER", 0, 2, 0)).replacen(
            "chunk = \"PLACEHOLDER\"",
            &format!("chunk = {chunk_toml}"),
            1,
        )
    };
    for bad in [
        "'a/b.bin'",
        "'..'",
        "'.'",
        "'/abs.bin'",
        "'C:win.bin'",
        "''",
        "'a\\\\b.bin'",
    ] {
        match manifest::parse(&with_chunk(bad)) {
            Ok(_) => panic!("chunk name {bad:?} must be rejected on the string"),
            Err(e) => assert_eq!(e.kind(), "invalid_artifact"),
        }
    }
    manifest::validate_chunk_name("chunk0.bin").expect("single component is fine");
}

#[test]
fn completeness_partial_is_a_shape_with_named_missing_tensors() {
    let complete_but_missing = manifest_with(&bf16_tensor("w", "1", "c.bin", 0, 2, 0)).replacen(
        "missing = []",
        "missing = [\"x\"]",
        1,
    );
    err_contains(
        manifest::parse(&complete_but_missing),
        "complete' but lists missing",
    );
    let partial_empty = manifest_with(&bf16_tensor("w", "1", "c.bin", 0, 2, 0)).replacen(
        "status = \"complete\"",
        "status = \"partial\"",
        1,
    );
    err_contains(manifest::parse(&partial_empty), "names no missing");
    let partial = manifest_with(&bf16_tensor("w", "1", "c.bin", 0, 2, 0)).replacen(
        "status = \"complete\"\nmissing = []",
        "status = \"partial\"\nmissing = [\"gone\"]",
        1,
    );
    let m = manifest::parse(&partial).expect("partial opens for inspection");
    assert!(matches!(
        m.completeness,
        moxie_format::manifest::Completeness::Partial { .. }
    ));
}

#[test]
fn excluded_entries_need_reasons_and_completeness_needs_a_status() {
    let bad = manifest_with(&bf16_tensor("w", "1", "c.bin", 0, 2, 0)).replacen(
        "role = \"old\"\nreason = \"removed upstream\"",
        "role = \"old\"\nreason = \"\"",
        1,
    );
    err_contains(manifest::parse(&bad), "non-empty");
    let bad = manifest_with(&bf16_tensor("w", "1", "c.bin", 0, 2, 0)).replacen(
        "status = \"complete\"",
        "status = \"almost\"",
        1,
    );
    err_contains(manifest::parse(&bad), "expected 'complete' or 'partial'");
}

#[test]
fn tensor_entries_are_capped_before_validation() {
    // The bound is on the deserialized value, checked before any O(n) pass.
    // A manifest just over the cap must fail with the cap named. Generating
    // 1M+ entries costs ~150 MB transiently; that is the price of proving the
    // bound is stated rather than incidental.
    // Build the tensor section separately to keep the header parseable alone.
    let mut tensors = String::new();
    for i in 0..=MAX_TENSORS {
        tensors.push_str(&format!(
            "[[tensors]]\nrole = \"t{i}\"\nshape = [1]\nprecision = \"bf16-v1\"\nchunk = \"c.bin\"\noffset = 0\nlength = 2\nsha256 = \"{HEX}\"\nalignment = 1\nlogical_order = {i}\n"
        ));
    }
    let text = format!(
        "{header}\n{tensors}\n[[excluded]]\nrole = \"x\"\nreason = \"y\"\n\n[completeness]\nstatus = \"complete\"\nmissing = []\n",
        header = header()
    );
    match manifest::parse(&text) {
        Ok(_) => panic!("{} tensors must exceed the cap", MAX_TENSORS + 1),
        Err(e) => {
            let msg = e.to_string();
            assert!(msg.contains("above the"), "{msg}");
        }
    }
}

#[test]
fn architecture_metadata_depth_is_bounded_during_traversal() {
    // 65 nested levels, one over the 64 cap. Dotted table headers nest
    // without repeating the parent path.
    let mut meta = String::from("[architecture.metadata]\n");
    let mut path = String::new();
    for i in 0..=MAX_ARCH_DEPTH {
        if !path.is_empty() {
            path.push('.');
        }
        path.push_str(&format!("l{i}"));
        meta.push_str(&format!("[architecture.metadata.{path}]\nv = {i}\n"));
    }
    let text = header().replacen("[architecture.metadata]\nhidden = 8", &meta, 1)
        + &bf16_tensor("w", "1", "c.bin", 0, 2, 0)
        + "\n[[excluded]]\nrole = \"x\"\nreason = \"y\"\n\n[completeness]\nstatus = \"complete\"\nmissing = []\n";
    err_contains(manifest::parse(&text), "deeper than");
}

#[test]
fn architecture_metadata_node_count_is_bounded_during_traversal() {
    // One key per node, just over the 65,536 cap.
    let mut keys = String::from("[architecture.metadata]\n");
    for i in 0..=MAX_ARCH_NODES {
        keys.push_str(&format!("k{i} = 1\n"));
    }
    let text = header().replacen("[architecture.metadata]\nhidden = 8", &keys, 1)
        + &bf16_tensor("w", "1", "c.bin", 0, 2, 0)
        + "\n[[excluded]]\nrole = \"x\"\nreason = \"y\"\n\n[completeness]\nstatus = \"complete\"\nmissing = []\n";
    err_contains(manifest::parse(&text), "more than");
}

#[test]
fn artifact_identity_hashes_the_opaque_tree_with_sorted_keys() {
    let a = manifest::parse(&valid_one()).unwrap();
    let reordered = valid_one().replacen("hidden = 8", "hidden = 8\nzzz = 1", 1);
    let b = manifest::parse(&reordered).unwrap();
    assert_ne!(
        manifest::artifact_identity(&a),
        manifest::artifact_identity(&b),
        "metadata participates in identity"
    );
    assert_eq!(manifest::artifact_identity(&a).len(), 64);
}

#[test]
fn manifest_size_cap_is_visible_where_it_is_enforced() {
    // The 4 MiB file-size cap is enforced by the storage crate's capped reader
    // before parsing (a bound on the parse itself); it is asserted there with
    // a real file. This names the constant here so the two stay linked.
    assert_eq!(MAX_MANIFEST_BYTES, 4 * 1024 * 1024);
}

/// Build a manifest identical to `valid_one` except for the source identity.
/// `model_toml`/`revision_toml` are TOML string literals (caller escapes).
fn manifest_with_source(model_toml: &str, revision_toml: &str) -> String {
    let mut text = valid_one();
    text = text.replacen("model = \"example\"", &format!("model = {model_toml}"), 1);
    text = text.replacen(
        "revision = \"r1\"",
        &format!("revision = {revision_toml}"),
        1,
    );
    text
}

#[test]
fn delimiter_characters_cannot_merge_two_fields_into_one_identity() {
    // First review, reproduced: NUL and newline are legal inside parsed
    // strings, so any delimiter-joined encoding hashes A and B identically.
    // Length-prefixing makes concatenation injective.
    let a = manifest::parse(&manifest_with_source(r#""x\u0000y""#, r#""z""#)).unwrap();
    let b = manifest::parse(&manifest_with_source(r#""x""#, r#""y\u0000z""#)).unwrap();
    assert_eq!(a.source.model, "x\0y");
    assert_eq!(b.source.revision, "y\0z");
    assert_ne!(
        manifest::artifact_identity(&a),
        manifest::artifact_identity(&b),
        "model/revision boundary crossed by an embedded NUL"
    );
    // Newline variant of the same confusion.
    let c = manifest::parse(&manifest_with_source(r#""x\ny""#, r#""z""#)).unwrap();
    let d = manifest::parse(&manifest_with_source(r#""x""#, r#""y\nz""#)).unwrap();
    assert_ne!(
        manifest::artifact_identity(&c),
        manifest::artifact_identity(&d),
        "model/revision boundary crossed by an embedded newline"
    );
}

#[test]
fn every_identity_field_boundary_is_length_delimited() {
    // The same confusion, moved across each other multi-field line the old
    // encoding joined: tokenizer name/version, excluded role/reason, and a
    // tensor role carrying the delimiter itself.
    let tok = |name: &str, version: &str| {
        valid_one()
            .replacen("name = \"tok\"", &format!("name = {name}"), 1)
            .replacen("version = \"1\"", &format!("version = {version}"), 1)
    };
    let a = manifest::parse(&tok(r#""t\u0000k""#, r#""1""#)).unwrap();
    let b = manifest::parse(&tok(r#""t""#, r#""k\u00001""#)).unwrap();
    assert_ne!(
        manifest::artifact_identity(&a),
        manifest::artifact_identity(&b),
        "tokenizer name/version boundary"
    );
    let exc = |role: &str, reason: &str| {
        valid_one()
            .replacen("role = \"old\"", &format!("role = {role}"), 1)
            .replacen(
                "reason = \"removed upstream\"",
                &format!("reason = {reason}"),
                1,
            )
    };
    let a = manifest::parse(&exc(r#""o\u0000ld""#, r#""why""#)).unwrap();
    let b = manifest::parse(&exc(r#""o""#, r#""ld\u0000why""#)).unwrap();
    assert_ne!(
        manifest::artifact_identity(&a),
        manifest::artifact_identity(&b),
        "excluded role/reason boundary"
    );
    // A role carrying NUL/newline is still a distinct role, never a merge.
    let t = |role: &str| {
        manifest_with(&bf16_tensor("PLACEHOLDER", "1", "c.bin", 0, 2, 0)).replacen(
            "role = \"PLACEHOLDER\"",
            &format!("role = {role}"),
            1,
        )
    };
    let a = manifest::parse(&t(r#""w\u0000""#)).unwrap();
    let b = manifest::parse(&t(r#""w""#)).unwrap();
    assert_ne!(
        manifest::artifact_identity(&a),
        manifest::artifact_identity(&b)
    );
}

#[test]
fn opaque_metadata_with_delimiters_is_stable_and_distinct() {
    // Embedded delimiters inside the opaque tree hash as data, distinctly.
    let with_meta = |value_toml: &str| {
        header().replacen("[architecture.metadata]\nhidden = 8", value_toml, 1)
            + &bf16_tensor("w", "1", "c.bin", 0, 2, 0)
            + "\n[[excluded]]\nrole = \"x\"\nreason = \"y\"\n\n[completeness]\nstatus = \"complete\"\nmissing = []\n"
    };
    let a = manifest::parse(&with_meta("[architecture.metadata]\ns = \"x\\u0000y\"")).unwrap();
    let b = manifest::parse(&with_meta(
        "[architecture.metadata]\ns = \"x\"\nt = \"y\\u0000z\"",
    ))
    .unwrap();
    assert_ne!(
        manifest::artifact_identity(&a),
        manifest::artifact_identity(&b)
    );
    assert_eq!(
        manifest::artifact_identity(&a),
        manifest::artifact_identity(
            &manifest::parse(&with_meta("[architecture.metadata]\ns = \"x\\u0000y\"")).unwrap()
        )
    );
}
