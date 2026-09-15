// The substitutions, generated once from the batteries this replaces and then
// owned here.
//
// Included by `mutationcheck.rs` rather than declared as a module: it is a
// table, not an interface, and nothing outside the battery may name it.

/// The `T0006` battery's lanes, in the order they run.
const LANES_T0006: &[Lane] = &[
    Lane { name: "format", argv: &[r#"test"#, r#"-p"#, r#"moxie-format"#, r#"--lib"#, r#"--offline"#, r#"--locked"#] },
    Lane { name: "manifest", argv: &[r#"test"#, r#"-p"#, r#"moxie-format"#, r#"--offline"#, r#"--locked"#, r#"--test"#, r#"manifest_v1"#] },
    Lane { name: "publication", argv: &[r#"test"#, r#"-p"#, r#"moxie-repack"#, r#"--offline"#, r#"--locked"#, r#"--test"#, r#"publication"#] },
    Lane { name: "workflow", argv: &[r#"test"#, r#"-p"#, r#"moxie-repack"#, r#"--offline"#, r#"--locked"#, r#"--test"#, r#"workflow"#] },
    Lane { name: "roundtrip", argv: &[r#"test"#, r#"-p"#, r#"moxie-repack"#, r#"--offline"#, r#"--locked"#, r#"--test"#, r#"round_trip"#] },
    Lane { name: "cli", argv: &[r#"test"#, r#"-p"#, r#"moxie-repack"#, r#"--offline"#, r#"--locked"#, r#"--test"#, r#"cli"#] },
    // `--test-threads=1` because this lane measures peak **live** heap through
    // a global allocator. Its own lock stops one test resetting the other's
    // peak, but not the other test's live bytes being counted into it, so the
    // lane is load-dependent: two baseline runs on an identical clean tree
    // reported "fails" and "disagrees with itself", and the battery correctly
    // refused to build verdicts on either. Serialising the executable is not a
    // weakened assertion -- it is the isolation the measurement already
    // assumes. The **fix** belongs in `crates/moxie-repack/tests/budget.rs`,
    // whose two tests should not share a process-wide counter at all; until
    // then this keeps the battery runnable, which an unrunnable battery is not.
    Lane { name: "budget", argv: &[r#"test"#, r#"-p"#, r#"moxie-repack"#, r#"--offline"#, r#"--locked"#, r#"--test"#, r#"budget"#, r#"--"#, r#"--test-threads=1"#] },
    Lane { name: "malformed", argv: &[r#"test"#, r#"-p"#, r#"moxie-repack"#, r#"--offline"#, r#"--locked"#, r#"--test"#, r#"malformed"#] },
    Lane { name: "round2", argv: &[r#"test"#, r#"-p"#, r#"moxie-repack"#, r#"--offline"#, r#"--locked"#, r#"--test"#, r#"round2"#] },
    Lane { name: "admission", argv: &[r#"test"#, r#"-p"#, r#"moxie-repack"#, r#"--offline"#, r#"--locked"#, r#"--test"#, r#"admission"#] },
    Lane { name: "storage", argv: &[r#"test"#, r#"-p"#, r#"moxie-storage"#, r#"--offline"#, r#"--locked"#, r#"--test"#, r#"artifact"#] },
    Lane { name: "arch", argv: &[r#"run"#, r#"--offline"#, r#"--locked"#, r#"-q"#, r#"--bin"#, r#"xtask"#, r#"--"#, r#"arch-check"#] },
];

/// The `T0006` battery.
const BATTERY_T0006: &[Mutation] = &[
    Mutation {
        name: "unit-checksum-not-compared",
        file: "crates/moxie-repack/src/write/run.rs",
        from: r#"        if got != unit.sha256 {"#,
        to: r#"        if false && got != unit.sha256 {"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "tensor-hash-never-sees-the-bytes",
        file: "crates/moxie-repack/src/write/run.rs",
        from: r#"        progress.hasher.update(bytes);"#,
        to: r#"        let _ = &mut progress.hasher;"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "published-validation-skipped",
        file: "crates/moxie-repack/src/write/run.rs",
        from: r#"        faults.check(Site::Validate)?;
        let artifact = Artifact::open_unpublished(&self.dest, &staged, ByteBudget::default())?;
        let mut roles: Vec<&str> = sealed.iter().map(|t| t.role.as_str()).collect();"#,
        to: r#"        let artifact = Artifact::open_unpublished(&self.dest, &staged, ByteBudget::default())?;
        let mut roles: Vec<&str> = sealed.iter().take(0).map(|t| t.role.as_str()).collect();"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "source-digest-is-a-constant",
        file: "crates/moxie-repack/src/lib.rs",
        from: r#"        let Some((digest, bytes)) =
            sources.file_digest_cancellable(&file, buffers.source_tile_mut(), cancelled)?
        else {"#,
        to: r#"        let Some((digest, bytes)) = Some(("0".repeat(64), 0u64)).filter(|_| {
            sources
                .file_digest_cancellable(&file, buffers.source_tile_mut(), cancelled)
                .is_ok()
        }) else {"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "unit-source-digest-is-recorded-not-checked",
        file: "crates/moxie-repack/src/work.rs",
        from: r#"fn sha256_of(bytes: &[u8]) -> String {
    let mut h = StreamingSha256::new();
    h.update(bytes);
    h.finalize_hex()
}"#,
        to: r#"fn sha256_of(bytes: &[u8]) -> String {
    let _ = bytes;
    "0".repeat(64)
}"#,
        expect: Expect::Survivor,
    },
    Mutation {
        name: "resume-ignores-its-binding",
        file: "crates/moxie-repack/src/write/run.rs",
        from: r#"        if recorded != self.binding {"#,
        to: r#"        if false && recorded != self.binding {"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "run-binding-omits-the-source-digests",
        file: "crates/moxie-repack/src/lib.rs",
        from: r#"    field(b"repack-run-v1");
    field(plan.as_bytes());
    field(selection.as_bytes());
    for (file, digest) in sources {
        field(file.as_bytes());
        field(digest.as_bytes());
    }"#,
        to: r#"    field(b"repack-run-v1");
    field(plan.as_bytes());
    field(selection.as_bytes());
    let _ = sources;"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "journal-version-not-checked",
        file: "crates/moxie-format/src/journal.rs",
        from: r#"                if v != JOURNAL_VERSION {"#,
        to: r#"                if false && v != JOURNAL_VERSION {"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "selection-version-not-checked",
        file: "crates/moxie-format/src/selection.rs",
        from: r#"    if v.version != SELECTION_VERSION {"#,
        to: r#"    if false && v.version != SELECTION_VERSION {"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "manifest-schema-version-not-checked",
        file: "crates/moxie-format/src/manifest.rs",
        from: r#"    if !SUPPORTED_SCHEMA_VERSIONS.contains(&v.schema_version) {"#,
        to: r#"    if false && !SUPPORTED_SCHEMA_VERSIONS.contains(&v.schema_version) {"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "manifest-staged-under-its-final-name",
        file: "crates/moxie-repack/src/write/run.rs",
        from: r#"        let staged = self.dest.join(STAGED_MANIFEST_FILE);"#,
        to: r#"        let staged = self.dest.join(MANIFEST_FILE);"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "incomplete-tensors-can-be-sealed",
        file: "crates/moxie-repack/src/write/run.rs",
        from: r#"            if p.done != c.len {"#,
        to: r#"            if false && p.done != c.len {"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "journal-recorded-before-the-bytes-are-durable",
        file: "crates/moxie-repack/src/write/run.rs",
        from: r#"        write_at(&mut file, offset, bytes, faults, Site::ChunkWrite)
            .map_err(|e| invalid(format!("cannot write {}: {e}", path.display())))?;
        faults.check(Site::ChunkSync)?;
        file.sync_all()
            .map_err(|e| invalid(format!("cannot sync {}: {e}", path.display())))?;

        let unit = CompletedUnit {"#,
        to: r#"        let unit = CompletedUnit {"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "journal-recorded-before-the-bytes-are-durable-tail",
        file: "crates/moxie-repack/src/write/run.rs",
        from: r#"        let line = journal::unit_line(&unit);
        self.charge_journal(line.len() as u64)?;
        append_durably(&mut self.journal, &line, faults)?;"#,
        to: r#"        let line = journal::unit_line(&unit);
        self.charge_journal(line.len() as u64)?;
        append_durably(&mut self.journal, &line, faults)?;
        write_at(&mut file, offset, bytes, faults, Site::ChunkWrite)
            .map_err(|e| invalid(format!("cannot write {}: {e}", path.display())))?;
        faults.check(Site::ChunkSync)?;
        file.sync_all()
            .map_err(|e| invalid(format!("cannot sync {}: {e}", path.display())))?;"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "resume-trusts-the-journals-offsets",
        file: "crates/moxie-repack/src/write/run.rs",
        from: r#"        if unit.chunk != planned.file || unit.offset != planned.file_offset + progress.done {"#,
        to: r#"        if false && (unit.chunk != planned.file || unit.offset != planned.file_offset + progress.done) {"#,
        expect: Expect::Survivor,
    },
    Mutation {
        name: "chunk-sync-skipped",
        file: "crates/moxie-repack/src/write/run.rs",
        from: r#"        faults.check(Site::ChunkSync)?;
        file.sync_all()
            .map_err(|e| invalid(format!("cannot sync {}: {e}", path.display())))?;"#,
        to: r#"        let _ = &file;"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "directory-sync-skipped",
        file: "crates/moxie-repack/src/write/run.rs",
        from: r#"        faults.check(Site::DirectorySync)?;
        let dir = File::open(&self.dest).map_err(|e| {"#,
        to: r#"        if true {
            return Ok(());
        }
        let dir = File::open(&self.dest).map_err(|e| {"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "write-errors-are-swallowed",
        file: "crates/moxie-repack/src/write/run.rs",
        from: r#"        write_at(&mut file, offset, bytes, faults, Site::ChunkWrite)
            .map_err(|e| invalid(format!("cannot write {}: {e}", path.display())))?;"#,
        to: r#"        let _ = write_at(&mut file, offset, bytes, faults, Site::ChunkWrite);"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "header-failure-leaks-the-admission",
        file: "crates/moxie-repack/src/write/run.rs",
        from: r#"            if let Err(e) = run.write_shard_headers(faults) {
                run.abandon(ledger)?;
                return Err(e);
            }
            Ok(Start::Fresh(run))"#,
        to: r#"            run.write_shard_headers(faults)?;
            Ok(Start::Fresh(run))"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "shard-header-sync-skipped",
        file: "crates/moxie-repack/src/write/run.rs",
        from: r#"            faults.check(Site::ShardHeaderSync)?;
            file.sync_all()
                .map_err(|e| invalid(format!("cannot sync '{}': {e}", shard.file)))?;"#,
        to: r#"            let _ = &file;"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "shard-header-never-written",
        file: "crates/moxie-repack/src/write/run.rs",
        from: r#"            write_at(&mut file, 0, header, faults, Site::ShardHeaderWrite).map_err(|e| {
                invalid(format!("cannot write the header of '{}': {e}", shard.file))
            })?;"#,
        to: r#"            let _ = (&mut file, header);"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "journal-sync-skipped",
        file: "crates/moxie-repack/src/write/run.rs",
        from: r#"    faults.check(Site::JournalSync)?;
    file.sync_data()
        .map_err(|e| invalid(format!("cannot sync the journal: {e}")))"#,
        to: r#"    Ok(())"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "unit-size-not-checked-against-the-scratch",
        file: "crates/moxie-repack/src/write/run.rs",
        from: r#"        if bytes.len() > self.budget.scratch_bytes() {"#,
        to: r#"        if false && bytes.len() > self.budget.scratch_bytes() {"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "disk-budget-not-checked-while-writing",
        file: "crates/moxie-repack/src/write/run.rs",
        from: r#"        if new_disk > self.budget.disk_bytes() {"#,
        to: r#"        if false && new_disk > self.budget.disk_bytes() {"#,
        expect: Expect::Survivor,
    },
    Mutation {
        name: "chunk-file-limit-ignored-by-the-plan",
        file: "crates/moxie-repack/src/write/plan.rs",
        from: r#"            if alone > budget.chunk_file_bytes() {"#,
        to: r#"            if false && alone > budget.chunk_file_bytes() {"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "plan-ignores-the-disk-budget",
        file: "crates/moxie-repack/src/write/plan.rs",
        from: r#"        if total_disk > budget.disk_bytes() {"#,
        to: r#"        if false && total_disk > budget.disk_bytes() {"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "tile-is-the-whole-scratch",
        file: "crates/moxie-repack/src/lib.rs",
        from: r#"        (self.scratch_bytes / 2).max(1)"#,
        to: r#"        (4usize << 20).max(self.scratch_bytes)"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "header-budget-flag-ignored",
        file: "crates/moxie-repack/src/lib.rs",
        from: r#"    HeaderBudget::new(budgets.header_bytes).ok_or_else(|| {"#,
        to: r#"    Some(HeaderBudget::DEFAULT).ok_or_else(|| {"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "cancellation-not-checked-between-units",
        file: "crates/moxie-repack/src/lib.rs",
        from: r#"            if cancelled() {
                let outcome = run_slot.take().expect("a run").cancel(ledger)?;"#,
        to: r#"            if false && cancelled() {
                let outcome = run_slot.take().expect("a run").cancel(ledger)?;"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "cancellation-not-checked-before-publish",
        file: "crates/moxie-repack/src/write/run.rs",
        from: r#"        if cancelled() {
            // The last point at which cancellation can be honoured: after this
            // the artifact exists.
            let bytes = self.progress.values().map(|p| p.done).sum();
            return self.cancel_with(bytes, ledger);
        }"#,
        to: r#"        let _ = &cancelled;"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "payload-sections-in-the-wrong-order",
        file: "crates/moxie-format/src/payload.rs",
        from: r#"    Ok(PayloadExtents {
        codes: 0..code_bytes,
        scales: code_bytes..scales_end,
        zero_points,
    })"#,
        to: r#"    Ok(PayloadExtents {
        codes: (scales_end - code_bytes)..scales_end,
        scales: 0..scale_bytes,
        zero_points,
    })"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "zero-point-section-dropped",
        file: "crates/moxie-format/src/payload.rs",
        from: r#"            Some(scales_end..end)"#,
        to: r#"            let _ = end;
            None"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "scale-block-not-validated",
        file: "crates/moxie-format/src/payload.rs",
        from: r#"        if !v.is_finite() || v <= 0.0 {"#,
        to: r#"        if false && (!v.is_finite() || v <= 0.0) {"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "affine-length-rule-deleted",
        file: "crates/moxie-format/src/manifest.rs",
        from: r#"            if let Placement::Chunk { length, .. } = &placement
                && need != *length
            {
                return Err(invalid(format_args!(
                    "tensor '{role}': a {} {out_features}x{in_features} tensor with {} \"#,
        to: r#"            if let Placement::Chunk { length, .. } = &placement
                && false
                && need != *length
            {
                return Err(invalid(format_args!(
                    "tensor '{role}': a {} {out_features}x{in_features} tensor with {} \"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "component-dtype-and-shape-not-checked",
        file: "crates/moxie-storage/src/lib.rs",
        from: r#"                    if entry.dtype != want.dtype
                        || entry.shape != want.shape
                        || entry.len() != want.len
                    {"#,
        to: r#"                    if false
                        && (entry.dtype != want.dtype
                            || entry.shape != want.shape
                            || entry.len() != want.len)
                    {"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "component-order-not-checked",
        file: "crates/moxie-format/src/manifest.rs",
        from: r#"            if got.kind != want.kind || got.name != want.name {"#,
        to: r#"            if false && (got.kind != want.kind || got.name != want.name) {"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "payload-coverage-not-required",
        file: "crates/moxie-format/src/safetensors.rs",
        from: r#"        if covered != self.payload_len {"#,
        to: r#"        if false && covered != self.payload_len {"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "hard-links-are-followed",
        file: "crates/moxie-repack/src/write/run.rs",
        from: r#"        if meta.nlink() != 1 {"#,
        to: r#"        if false && meta.nlink() != 1 {"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "source-identity-not-rebound",
        file: "crates/moxie-repack/src/source.rs",
        from: r#"                Some(seen) if *seen != key => {"#,
        to: r#"                Some(seen) if false && *seen != key => {"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "source-replacement-not-detected",
        file: "crates/moxie-repack/src/source.rs",
        from: r#"        self.confirm_identities()?;"#,
        to: r#"        let _ = self.confirm_identities();"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "metadata-bound-is-a-constant-again",
        file: "crates/moxie-repack/src/lib.rs",
        from: r#"        self.header_bytes
            + moxie_format::manifest::MAX_MANIFEST_BYTES as u64
            + selection_bytes.saturating_mul(Self::ROLE_RETENTIONS)
            + tensors.saturating_mul(Self::PER_TENSOR_METADATA_BYTES)"#,
        to: r#"        let _ = (selection_bytes, tensors);
        self.metadata_floor_bytes()"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "staging-bound-ignores-the-names",
        file: "crates/moxie-repack/src/lib.rs",
        from: r#"            let component_name = t.role_bytes + 16;"#,
        to: r#"            let component_name = 0;"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "resume-rewrites-before-it-checks",
        file: "crates/moxie-repack/src/write/run.rs",
        from: r#"        if resuming {
            let report = match run.recover(&journal_path, faults, cancelled) {"#,
        to: r#"        if resuming {
            if run.write_shard_headers(faults).is_err() {
                run.abandon(ledger)?;
                return Err(invalid("header pass failed".into()));
            }
            let report = match run.recover(&journal_path, faults, cancelled) {"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "torn-tail-found-after-decoding",
        file: "crates/moxie-format/src/journal.rs",
        from: r#"    let (complete_bytes, torn) = match bytes.iter().rposition(|b| *b == b'\n') {
        Some(at) => (&bytes[..=at], bytes.len() - at - 1),
        None => (&bytes[..0], bytes.len()),
    };
    let complete = core::str::from_utf8(complete_bytes).map_err(|e| {"#,
        to: r#"    let checked = core::str::from_utf8(bytes).map_err(|e| {
        invalid(format_args!("the journal is not UTF-8: {e}"))
    })?;
    let (complete_bytes, torn) = match checked.rfind('\n') {
        Some(at) => (&bytes[..=at], bytes.len() - at - 1),
        None => (&bytes[..0], bytes.len()),
    };
    let complete = core::str::from_utf8(complete_bytes).map_err(|e| {"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "capped-read-uses-the-manifests-cap",
        file: "crates/moxie-storage/src/lib.rs",
        from: r#"    read_file_capped_bytes(path, cap)"#,
        to: r#"    read_file_capped_bytes(path, moxie_format::manifest::MAX_MANIFEST_BYTES)"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "journal-cap-not-enforced-on-the-plan",
        file: "crates/moxie-repack/src/lib.rs",
        from: r#"    if estimate.journal_bound_bytes > moxie_format::journal::MAX_JOURNAL_BYTES as u64 {"#,
        to: r#"    if false && estimate.journal_bound_bytes > moxie_format::journal::MAX_JOURNAL_BYTES as u64 {"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "cancelled-hashing-is-an-error-again",
        file: "crates/moxie-repack/src/source.rs",
        from: r#"        let shard = self.shard(file)?;
        let digested = shard.digest_whole_file(scratch, cancelled)?;"#,
        to: r#"        let shard = self.shard(file)?;
        let digested = shard.digest_whole_file(scratch, &|| false)?;
        let _ = cancelled;"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "split-module-companion-search-narrowed",
        file: "crates/moxie-repack/src/lib.rs",
        from: r#"                        let mut searched: Vec<&str> = files.values().map(String::as_str).collect();"#,
        to: r#"                        let mut searched: Vec<&str> = vec![packed_file.as_str()];"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "relative-destinations-are-not-resolved",
        file: "crates/moxie-repack/src/lib.rs",
        from: r#"    let mut probe = if destination.is_absolute() {
        destination.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| invalid(format!("cannot read the current directory: {e}")))?
            .join(destination)
    };"#,
        to: r#"    let mut probe = destination.to_path_buf();"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "v1-identity-follows-the-current-schema",
        file: "crates/moxie-format/src/manifest.rs",
        from: r#"    w.field_u64("schema", schema_version_of(manifest) as u64);"#,
        to: r#"    w.field_u64("schema", SCHEMA_VERSION as u64);"#,
        expect: Expect::Caught,
    },
    // --- round 3 -------------------------------------------------------------
    Mutation {
        name: "source-length-not-rechecked",
        file: "crates/moxie-storage/src/lib.rs",
        from: r#"        if now != self.len {"#,
        to: r#"        if false && now != self.len {"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "source-header-not-bound",
        file: "crates/moxie-repack/src/source.rs",
        from: r#"            header_sha256: shard.header_sha256().to_string(),"#,
        to: r#"            header_sha256: String::new(),"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "partial-admission-leaks",
        file: "crates/moxie-repack/src/work.rs",
        from: r#"            Err(e) => {
                let _ = ledger.release(metadata_charge);
                return Err(e);
            }"#,
        to: r#"            Err(e) => {
                return Err(e);
            }"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "recovery-cancellation-is-an-error-again",
        file: "crates/moxie-repack/src/write/run.rs",
        from: r#"            if cancelled() {
                return Ok(None);
            }"#,
        to: r#"            if cancelled() {
                return Err(invalid("cancelled while rehashing".into()));
            }"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "journal-history-not-compacted",
        file: "crates/moxie-repack/src/write/run.rs",
        from: r#"            for index in &kept {
                let line = journal::unit_line(&state.units[*index]);"#,
        to: r#"            for index in 0..state.units.len() {
                let _ = &kept;
                let line = journal::unit_line(&state.units[index]);"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "manifest-global-allowance-is-a-constant",
        file: "crates/moxie-repack/src/lib.rs",
        from: r#"        let mut manifest = Self::MANIFEST_FIXED_BOUND
            + selection_bytes.saturating_mul(Self::MANIFEST_GLOBAL_EXPANSION);"#,
        to: r#"        let _ = selection_bytes;
        let mut manifest = Self::MANIFEST_FIXED_BOUND;"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "recovery-is-not-admitted",
        file: "crates/moxie-repack/src/lib.rs",
        from: r#"    buffers.admit_recovery(
        ledger,
        Budgets::recovery_bound(journal_bytes, units, widest_role),
    )?;"#,
        to: r#"    let _ = (journal_bytes, units, widest_role);"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "schema-version-inferred-again",
        file: "crates/moxie-format/src/manifest.rs",
        from: r#"pub fn schema_version_of(manifest: &Manifest) -> u32 {
    manifest.schema_version
}"#,
        to: r#"pub fn schema_version_of(manifest: &Manifest) -> u32 {
    match manifest.tensors.first().map(|t| &t.placement) {
        Some(Placement::Chunk { .. }) => 1,
        _ => SCHEMA_VERSION,
    }
}"#,
        expect: Expect::Caught,
    },
    // --- round 4 -------------------------------------------------------------
    Mutation {
        name: "compaction-truncates-in-place",
        file: "crates/moxie-repack/src/write/run.rs",
        from: r#"            let mut file = open_confined(&self.dest, COMPACT_JOURNAL_FILE, false)?;"#,
        to: r#"            let mut file = open_confined(&self.dest, JOURNAL_FILE, false)?;"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "journal-cap-not-enforced-per-append",
        file: "crates/moxie-repack/src/write/run.rs",
        from: r#"        if used > journal::MAX_JOURNAL_BYTES as u64 {"#,
        to: r#"        if false && used > journal::MAX_JOURNAL_BYTES as u64 {"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "journal-sizing-ignores-escaping",
        file: "crates/moxie-repack/src/lib.rs",
        from: r#"            role_bytes: moxie_format::journal::escaped_len(&r.role) as u64,"#,
        to: r#"            role_bytes: r.role.len() as u64,"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "escaped-len-undercounts",
        file: "crates/moxie-format/src/journal.rs",
        from: r#"            c if (c as u32) < 0x20 || c as u32 == 0x7f => 6,"#,
        to: r#"            c if (c as u32) < 0x20 || c as u32 == 0x7f => 1,"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "compaction-peak-not-budgeted",
        file: "crates/moxie-repack/src/write/run.rs",
        from: r#"        let peak = self.overhead_used + self.journal_used;
        if peak > self.plan.overhead_bytes() {"#,
        to: r#"        let peak = self.overhead_used + self.journal_used;
        if false && peak > self.plan.overhead_bytes() {"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "plan-budgets-for-one-journal",
        file: "crates/moxie-repack/src/lib.rs",
        from: r#"    let total = 2 * estimate.journal_bound_bytes + estimate.manifest_bound_bytes;"#,
        to: r#"    let total = estimate.journal_bound_bytes + estimate.manifest_bound_bytes;"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "leftover-replacement-is-kept",
        file: "crates/moxie-repack/src/write/run.rs",
        from: r#"    let leftover = dest.join(COMPACT_JOURNAL_FILE);
    if leftover.exists() {"#,
        to: r#"    let leftover = dest.join(COMPACT_JOURNAL_FILE);
    if false && leftover.exists() {"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "leftover-cleared-before-the-binding-check",
        file: "crates/moxie-repack/src/write/run.rs",
        from: r#"        if !resuming {
            remove_leftover_replacement(&dest)?;
        }"#,
        to: r#"        remove_leftover_replacement(&dest)?;"#,
        expect: Expect::Caught,
    },
];

/// What `T0006` rebuilds between substitutions.
/// The `T0028` battery's lanes.
///
/// One host lane and two device lanes. The device lanes are what make this
/// battery worth its runtime: a kernel's answer can only be wrong on a GPU, and
/// a host lane cannot tell a correct tile from a transposed one.
const LANES_T0028: &[Lane] = &[
    Lane { name: "w4a16-host", argv: &[r#"test"#, r#"-p"#, r#"moxie-executor"#, r#"--lib"#, r#"--offline"#, r#"--locked"#, r#"affine_linear"#] },
    Lane { name: "w4a16-kernels", argv: &[r#"test"#, r#"-p"#, r#"moxie-kernels"#, r#"--features"#, r#"fatbin"#, r#"--offline"#, r#"--locked"#] },
    Lane { name: "w4a16-device", argv: &[r#"test"#, r#"-p"#, r#"moxie-executor"#, r#"--features"#, r#"driver"#, r#"--offline"#, r#"--locked"#, r#"--test"#, r#"affine_linear_device"#] },
    // The injected-fault lane. Three of the review's findings are about what
    // happens **after** something has been enqueued, and no unfaulted test can
    // reach that window.
    Lane { name: "w4a16-faults", argv: &[r#"test"#, r#"-p"#, r#"moxie-executor"#, r#"--features"#, r#"driver"#, r#"--offline"#, r#"--locked"#, r#"--test"#, r#"driver_faults"#] },
    // The review found three defects in `moxie-repack`'s user surface while
    // reading this batch. They are task 0027's code and this is where their
    // regressions are measured, because T0006 measures the publication path
    // rather than the CLI.
    Lane { name: "two-command", argv: &[r#"test"#, r#"-p"#, r#"moxie-repack"#, r#"--offline"#, r#"--locked"#, r#"--test"#, r#"two_command"#] },
    // The allocation-failure lane. A refusal that aborts cannot be caught by a
    // test that never fails an allocation, and the re-review's reproduction was
    // a `SIGABRT`, which no assertion inside the process can observe.
    Lane { name: "allocation-refusal", argv: &[r#"test"#, r#"-p"#, r#"moxie-executor"#, r#"--offline"#, r#"--locked"#, r#"--test"#, r#"allocation_refusal"#] },
];

/// The `T0028` battery: can task 0028's gates, and its review's, actually fail?
///
/// Every substitution here is a way of getting a **plausible wrong answer**
/// rather than a crash -- a transposed tile, a nibble pair read backwards, a
/// zero point that is never subtracted, one group's scale applied to another's
/// codes. Those are exactly the defects a tolerance cannot catch by being
/// tight, which is why they are measured rather than asserted.
const BATTERY_T0028: &[Mutation] = &[
    // --- the re-review of the first seven findings ----------------------------
    //
    // Three of the seven were reported as fixed and were not. Each substitution
    // below is the state the re-review actually reproduced, not a paraphrase of
    // it: a process that aborts, a buffer freed under an unknown completion, and
    // a plan that publishes the wrong tensors as complete.
    Mutation {
        // Finding 3, re-reviewed: the arguments were allocated **before** the
        // fallible sink received them, so one failed allocation aborted the
        // process through the path built to prevent that.
        name: "refusal-prose-allocated-before-the-fallible-sink",
        file: "crates/moxie-executor/src/affine_linear.rs",
        from: r#"            declared.map_or("none", |p| p.get().name()),"#,
        to: r#"            declared.map_or_else(|| "none".to_string(), |p| p.to_string()),"#,
        expect: Expect::Caught,
    },
    Mutation {
        // The same finding at its root: a "fallible" sink that is not fallible.
        // Every refusal in this module composes its prose through it, so this
        // is the one substitution that puts the abort back everywhere at once.
        name: "the-fallible-sink-grows-infallibly",
        file: "crates/moxie-executor/src/affine_linear.rs",
        from: r#"    match sink.write_fmt(args) {
        Ok(()) => sink.0,
        Err(_) => String::new(),
    }"#,
        to: r#"    let _ = &sink;
    args.to_string()"#,
        expect: Expect::Caught,
    },
    Mutation {
        // Finding 2, re-reviewed: `close` refused while quarantined but nothing
        // stopped the value being dropped, and dropping it freed the host bytes
        // an asynchronous copy may still be reading.
        name: "quarantined-run-releases-its-operands-on-drop",
        file: "crates/moxie-executor/src/affine_linear.rs",
        from: r#"            if !self.quarantined {
                return;
            }"#,
        to: r#"            if true {
                return;
            }"#,
        expect: Expect::Caught,
    },
    Mutation {
        // The physical half of the same finding: a failed context synchronise is
        // the one answer meaning "completion unknown", and the free proceeded.
        name: "device-buffer-freed-under-a-failed-synchronize",
        file: "crates/moxie-cuda/src/driver.rs",
        from: r#"            if ffi::cuCtxSynchronize() != ffi::CUDA_SUCCESS {
                self.ptr = 0;
                return;
            }"#,
        to: r#"            let _ = ffi::cuCtxSynchronize();"#,
        expect: Expect::Caught,
    },
    Mutation {
        // Finding 5, re-reviewed: the first repair compared the **count** of
        // planned tensors with the bound index. Two different sets satisfy one
        // count equally, and review found the pair -- a tensor duplicated and
        // another dropped. This substitution is that repair, verbatim.
        name: "completeness-compares-a-count-not-a-set",
        file: "crates/moxie-repack/src/discover.rs",
        from: r#"    if !duplicated.is_empty() {"#,
        to: r#"    let accounted: usize = selection
        .tensors
        .iter()
        .map(|t| match &t.kind {
            moxie_format::selection::SelectionKind::Bf16 { .. } => 1,
            moxie_format::selection::SelectionKind::PackQuantized { files, .. } => files.len(),
        })
        .sum();
    if accounted == index_tensors {
        return Ok(());
    }
    if !duplicated.is_empty() {"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "int4-nibble-pair-read-backwards",
        file: "crates/moxie-kernels/cuda/affine_linear.cu",
        from: r#"        (k & 1ULL) ? (byte >> 4) : (byte & 0x0FU));"#,
        to: r#"        (k & 1ULL) ? (byte & 0x0FU) : (byte >> 4));"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "zero-point-never-subtracted",
        file: "crates/moxie-kernels/cuda/affine_linear.cu",
        from: r#"                value = __fmul_rn(static_cast<float>(code - zero), scale);"#,
        to: r#"                value = __fmul_rn(static_cast<float>(code), scale);"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "every-group-uses-the-first-groups-scale",
        file: "crates/moxie-kernels/cuda/affine_linear.cu",
        from: r#"            const unsigned long long group =
                (groups_per_row == 1ULL)
                    ? 0ULL
                    : ((k0 + half) / static_cast<unsigned long long>(group_size));"#,
        to: r#"            const unsigned long long group = 0ULL;"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "bf16-scales-decoded-as-f16",
        file: "crates/moxie-kernels/cuda/affine_linear.cu",
        from: r#"    if (kind == 1U) {"#,
        to: r#"    if (kind == 9U) {"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "weight-tile-loaded-untransposed",
        file: "crates/moxie-kernels/cuda/affine_linear.cu",
        from: r#"        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;"#,
        to: r#"        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "host-group-map-off-by-one",
        file: "crates/moxie-executor/src/affine_linear.rs",
        from: r#"            k / self.group_size"#,
        to: r#"            (k + 1) / self.group_size"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "permuted-tensor-accepted-as-contiguous",
        file: "crates/moxie-executor/src/affine_linear.rs",
        from: r#"        if descriptor.group_index.is_some() {"#,
        to: r#"        if false && descriptor.group_index.is_some() {"#,
        expect: Expect::Caught,
    },
    // The review's findings, each with the substitution that puts the defect
    // back. Every one of them is a plausible wrong answer or a freed buffer
    // that work may still be reading -- never a crash.
    Mutation {
        name: "another-devices-lease-accepted",
        file: "crates/moxie-executor/src/affine_linear.rs",
        from: r#"                if lease.scope() != Scope::Device(device) {"#,
        to: r#"                if false && lease.scope() != Scope::Device(device) {"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "admission-trusts-the-descriptor-it-is-handed",
        file: "crates/moxie-executor/src/affine_linear.rs",
        from: r#"            if let Err(error) = super::descriptor_serves(
                &descriptor,
                moxie_types::WeightPrecision::expect(launch.width().precision()),
                &launch,
            ) {
                return Err(fail(error));
            }"#,
        to: r#"            let _ = &launch;"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "an-unprovable-launch-hands-its-operands-back",
        file: "crates/moxie-executor/src/affine_linear.rs",
        from: r#"                    // `enqueue` already quarantined, so the operands stay here.
                    return Err(AffineRunRefused {
                        error,
                        weight: None,
                        activations: None,
                    });"#,
        to: r#"                    return Err(AffineRunRefused {
                        error,
                        weight: self.held_weight.take(),
                        activations: self.held_activations.take(),
                    });"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "an-edited-plan-publishes-as-complete",
        file: "crates/moxie-repack/src/discover.rs",
        from: r#"    if !matches!(
        selection.completeness,
        moxie_format::selection::Completeness::Complete
    ) {
        return Ok(());
    }"#,
        to: r#"    if true {
        return Ok(());
    }"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "a-staging-path-is-deleted-without-ownership",
        file: "crates/moxie-repack/src/main.rs",
        from: r#"                Ok(_) if !flags.force => {"#,
        to: r#"                Ok(_) if false => {"#,
        expect: Expect::Caught,
    },
    Mutation {
        name: "the-plan-is-read-before-its-cap-applies",
        file: "crates/moxie-repack/src/main.rs",
        from: r#"    let plan_text = match moxie_storage::read_text_capped(
        &selection_path,
        moxie_format::selection::MAX_SELECTION_BYTES,
    ) {"#,
        to: r#"    let plan_text = match std::fs::read_to_string(&selection_path) {"#,
        expect: Expect::Caught,
    },
    // Independence control. No fixture supplies a component shorter than its
    // descriptor implies, so removing this check must leave the battery green:
    // if a numerical lane goes red here, the answer was depending on a bounds
    // check rather than on the arithmetic, which is a different fact.
    Mutation {
        name: "resident-component-length-unchecked",
        file: "crates/moxie-executor/src/affine_linear.rs",
        from: r#"            if len < need {"#,
        to: r#"            if false && len < need {"#,
        expect: Expect::Survivor,
    },
];

const BUILDS_T0028: &[&[&str]] = &[
    &["-p", "moxie-kernels", "--features", "fatbin"],
    &["-p", "moxie-executor", "--features", "driver", "--tests"],
    &["-p", "moxie-repack", "--tests"],
];

const BUILDS_T0006: &[&[&str]] = &[
    &["-p", "moxie-format"],
    &["-p", "moxie-storage", "--tests"],
    &["-p", "moxie-repack", "--tests"],
    &["-p", "xtask"],
];

