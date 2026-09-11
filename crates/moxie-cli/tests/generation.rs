use moxie_cli::{Options, fixture, render};
use moxie_engine::{
    Cancel, GenerationEvent as Event, GenerationRequest as Request, HostTensor, Value,
    service::{GenerationService, StartError},
};
use moxie_interp::{Interpreter, KvCache, paged::PagedExecution};
use moxie_memory::{CapacitySnapshot, Ledger};
use moxie_state::{KvGeometry, PagedSequence, ROOT, SequenceState, StateKind};
use moxie_types::{Error, Precision, Scope};

fn ledger() -> Ledger {
    Ledger::new([CapacitySnapshot::new(Scope::Host, 1 << 30, 1).unwrap()]).unwrap()
}
fn request(prompt: &[u32], chunk: usize, temp: f64) -> Request<'_> {
    Request {
        prompt,
        max_new_tokens: 5,
        prefill_chunk: chunk,
        temperature: temp,
        seed: 42,
    }
}
fn tokens(service: &mut GenerationService<'_, '_>) -> Vec<u32> {
    let mut tokens = Vec::new();
    while let Some(event) = service.next_event(&Cancel::never()) {
        match event {
            Event::Token { id, .. } => tokens.push(id),
            Event::Finished { usage } => assert_eq!(usage.completion_tokens, tokens.len()),
            Event::Failed { error, .. } => panic!("{error}"),
            Event::Cancelled { .. } => panic!("unexpected cancellation"),
            _ => {}
        }
    }
    assert!(service.is_idle());
    assert_eq!(service.charged_bytes(), 0);
    tokens
}

#[test]
fn whole_chunked_decode_seed_and_repeated_generation_agree() {
    for (heads, dim, vocab, layers) in [(2, 4, 16, 1), (3, 4, 7, 2)] {
        let fixture = fixture::build(heads, dim, vocab, layers).unwrap();
        let prompt: Vec<_> = (0..19).map(|i| i % vocab as u32).collect();
        for temperature in [0.0, 0.25, 1.0, 10.0] {
            let mut owner = ledger();
            let mut service = GenerationService::new(&mut owner, fixture.program());
            service.start(request(&prompt, 19, temperature)).unwrap();
            let reference = tokens(&mut service);
            for chunk in [1, 3, 7, 8, 19, 32] {
                service.start(request(&prompt, chunk, temperature)).unwrap();
                assert_eq!(
                    reference,
                    tokens(&mut service),
                    "chunk {chunk}, T {temperature}"
                );
            }
        }
    }
}

#[test]
fn admission_rejects_a_graph_that_cannot_execute_tail_or_decode_rows() {
    let fixed = fixture::build_fixed_rows(2, 4, 16, 1, 2).unwrap();
    let mut owner = ledger();
    let mut service = GenerationService::new(&mut owner, fixed.program());
    let request = Request {
        prompt: &[0, 1],
        max_new_tokens: 2,
        prefill_chunk: 2,
        temperature: 0.0,
        seed: 0,
    };
    assert!(matches!(
        service.start(request),
        Err(StartError::Rejected(Error::InvalidRequest {
            field: "program",
            ..
        }))
    ));
    assert!(service.is_idle());
    assert_eq!(service.charged_bytes(), 0);

    // A fixed two-row graph is coherent when no tail or decode row is needed.
    let one_token = Request {
        max_new_tokens: 1,
        ..request
    };
    service.start(one_token).unwrap();
    assert_eq!(tokens(&mut service).len(), 1);
}

#[test]
fn paged_history_is_owned_by_one_immutable_program() {
    let first = fixture::build(2, 4, 16, 1).unwrap();
    let second = fixture::build(2, 4, 16, 1).unwrap();
    let mut owner = ledger();
    let geometry = KvGeometry {
        layers: 1,
        kv_heads: 2,
        key_dim: 4,
        value_dim: 4,
        precision: Precision::Bf16,
        page_tokens: 3,
        max_tokens: 4,
    };
    let mut pages = PagedSequence::with_sampling(&mut owner, geometry, 16, 2, 0).unwrap();
    pages.append_prompt(2).unwrap();
    let execution = PagedExecution::bind(
        &first.graph,
        &first.weights,
        first.tokens,
        first.positions,
        &mut pages,
    )
    .unwrap();
    let txn = pages.begin().unwrap();
    execution
        .run(&mut pages, txn, &[0], &[0], &Cancel::never())
        .unwrap();
    pages.commit_prefix(txn, 0).unwrap();
    pages.clear_logits().unwrap();

    let error = PagedExecution::bind(
        &second.graph,
        &second.weights,
        second.tokens,
        second.positions,
        &mut pages,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        Error::InvalidRequest {
            field: "execution_configuration",
            ..
        }
    ));
    let txn = pages.begin().unwrap();
    execution
        .run(&mut pages, txn, &[1], &[1], &Cancel::never())
        .unwrap();
    pages.commit_prefix(txn, 0).unwrap();
    pages.close(&mut owner).unwrap();
    assert!(owner.outstanding().is_empty());
}

#[test]
fn forward_cancellation_restores_existing_rows_frontiers_and_lineage() {
    for (heads, dim, vocab, layers) in [(2, 4, 16, 1), (3, 4, 7, 2)] {
        let fixture = fixture::build(heads, dim, vocab, layers).unwrap();
        let mut owner = ledger();
        let geometry = KvGeometry {
            layers: layers as usize,
            kv_heads: heads as usize,
            key_dim: dim as usize,
            value_dim: dim as usize,
            precision: Precision::Bf16,
            page_tokens: 3,
            max_tokens: 16,
        };
        let mut pages =
            PagedSequence::with_sampling(&mut owner, geometry, vocab as usize, 4, 0).unwrap();
        pages.append_prompt(8).unwrap();
        let execution = PagedExecution::bind(
            &fixture.graph,
            &fixture.weights,
            fixture.tokens,
            fixture.positions,
            &mut pages,
        )
        .unwrap();
        let txn = pages.begin().unwrap();
        execution
            .run(&mut pages, txn, &[0, 1], &[0, 1], &Cancel::never())
            .unwrap();
        assert!(pages.clear_logits().is_err());
        assert!(pages.record_logits(txn).is_err()); // at most one live result
        pages.commit_prefix(txn, 0).unwrap();
        pages.clear_logits().unwrap();
        let before = pages.state().frontiers(ROOT).unwrap();
        let lineage = pages.state().lineage_at(ROOT, 2).unwrap();
        let stored: Vec<_> = (0..layers as usize)
            .flat_map(|layer| (0..2).map(move |p| (layer, p)))
            .map(|(layer, p)| {
                let row = pages.row(layer, p).unwrap();
                (row.key.to_vec(), row.value.to_vec())
            })
            .collect();
        let continuation = [2, 3, 4, 5, 6, 0];
        let positions = [2, 3, 4, 5, 6, 7];
        let counter = Cancel::after(1000);
        let txn = pages.begin().unwrap();
        execution
            .run(&mut pages, txn, &continuation, &positions, &counter)
            .unwrap();
        pages.abort(txn).unwrap();
        let boundaries = 1000 - counter.remaining();
        for stop in 0..boundaries {
            let txn = pages.begin().unwrap();
            let error = execution
                .run(
                    &mut pages,
                    txn,
                    &continuation,
                    &positions,
                    &Cancel::after(stop),
                )
                .unwrap_err();
            assert!(matches!(error, Error::Cancelled { .. }));
            assert_eq!(pages.usage().rows, 2);
            assert_eq!(pages.state().frontiers(ROOT).unwrap(), before);
            assert_eq!(pages.state().lineage_at(ROOT, 2).unwrap(), lineage);
            assert!(pages.state().open_transactions().is_empty());
            assert!(pages.state().live_results().is_empty());
            for (index, (key, value)) in stored.iter().enumerate() {
                let row = pages.row(index / 2, (index % 2) as u64).unwrap();
                assert_eq!(row.key, key);
                assert_eq!(row.value, value);
            }
        }
        pages.close(&mut owner).unwrap();
        assert!(owner.outstanding().is_empty());
    }
}

#[test]
fn paged_forward_is_bit_exact_with_dense_reference_and_rejects_stale_outputs() {
    for (heads, dim, vocab, layers) in [(2, 4, 16, 1), (3, 4, 7, 2)] {
        let fixture = fixture::build(heads, dim, vocab, layers).unwrap();
        for chunk in [1, 3, 7, 19] {
            let mut owner = ledger();
            let geometry = KvGeometry {
                layers: layers as usize,
                kv_heads: heads as usize,
                key_dim: dim as usize,
                value_dim: dim as usize,
                precision: Precision::Bf16,
                page_tokens: 7,
                max_tokens: 32,
            };
            let mut pages =
                PagedSequence::with_sampling(&mut owner, geometry, vocab as usize, 10, 42).unwrap();
            let mut dense = SequenceState::new([StateKind::KvPages]);
            let mut cache = KvCache::for_branch(layers as usize, &dense, ROOT).unwrap();
            pages.append_prompt(19).unwrap();
            let execution = PagedExecution::bind(
                &fixture.graph,
                &fixture.weights,
                fixture.tokens,
                fixture.positions,
                &mut pages,
            )
            .unwrap();
            dense.append_prompt(ROOT, 19).unwrap();
            let mut last = None;
            for start in (0..19).step_by(chunk) {
                let end = (start + chunk).min(19);
                let mut bindings = fixture.weights.clone();
                bindings.set(
                    fixture.tokens,
                    Value::Index((start..end).map(|i| i as u64 % vocab).collect()),
                );
                bindings.set(
                    fixture.positions,
                    Value::Index((start as u64..end as u64).collect()),
                );
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
                let input_tokens: Vec<_> = (start..end).map(|i| i as u64 % vocab).collect();
                let input_positions: Vec<_> = (start as u64..end as u64).collect();
                let actual = execution
                    .run(
                        &mut pages,
                        txn,
                        &input_tokens,
                        &input_positions,
                        &Cancel::never(),
                    )
                    .unwrap();
                assert_eq!(reference.logits.data(), actual.logits().data());
                pages.commit_prefix(txn, 0).unwrap();
                last = Some(actual);
            }
            let output = last.unwrap();
            let mut foreign =
                PagedSequence::with_sampling(&mut owner, geometry, vocab as usize, 10, 42).unwrap();
            let foreign_txn = foreign.begin().unwrap();
            assert!(
                output
                    .stage(&mut foreign, foreign_txn, 0.0, &Cancel::never())
                    .is_err()
            );
            foreign.abort(foreign_txn).unwrap();
            foreign.close(&mut owner).unwrap();
            let txn = pages.begin().unwrap();
            let token = output
                .stage(&mut pages, txn, 0.0, &Cancel::never())
                .unwrap();
            pages.commit_prefix(txn, 1).unwrap();
            let txn = pages.begin().unwrap();
            assert!(
                output
                    .stage(&mut pages, txn, 0.0, &Cancel::never())
                    .is_err()
            );
            pages.abort(txn).unwrap();
            // Materialize the actual pending sampled token through both paths.
            dense.accept(ROOT, 1).unwrap();
            let mut bindings = fixture.weights.clone();
            bindings.set(fixture.tokens, Value::Index(vec![token as u64]));
            bindings.set(fixture.positions, Value::Index(vec![19]));
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
                .run(&mut pages, txn, &[token as u64], &[19], &Cancel::never())
                .unwrap();
            assert_eq!(reference.logits.data(), actual.logits().data());
            assert!(
                output
                    .stage(&mut pages, txn, 0.0, &Cancel::never())
                    .is_err()
            );
            pages.abort(txn).unwrap();
            let txn = pages.begin().unwrap();
            assert!(
                actual
                    .stage(&mut pages, txn, 0.0, &Cancel::never())
                    .is_err()
            );
            pages.abort(txn).unwrap();
            // Same numerical prefix after replay is a different result. A
            // counter-only guard would now accept the aborted output.
            let txn = pages.begin().unwrap();
            let fresh = execution
                .run(&mut pages, txn, &[token as u64], &[19], &Cancel::never())
                .unwrap();
            assert!(
                actual
                    .stage(&mut pages, txn, 0.0, &Cancel::never())
                    .is_err()
            );
            fresh.stage(&mut pages, txn, 0.0, &Cancel::never()).unwrap();
            pages.commit_prefix(txn, 1).unwrap();
            pages.close(&mut owner).unwrap();
            assert!(owner.outstanding().is_empty());
        }
    }
}

#[test]
fn every_service_boundary_cancels_without_tentative_events_and_allows_restart() {
    let fixture = fixture::build(3, 4, 7, 2).unwrap();
    let prompt = [0, 1, 2, 3, 4, 5, 6, 0, 1];
    let mut owner = ledger();
    let mut service = GenerationService::new(&mut owner, fixture.program());
    let req = request(&prompt, 4, 1.0);
    service.start(req).unwrap();
    let counter = Cancel::after(10000);
    while service.next_event(&counter).is_some() {}
    let boundaries = 10000 - counter.remaining();
    assert!(boundaries > 40);
    for stop in 0..boundaries {
        service.start(req).unwrap();
        assert_eq!(service.start(req), Err(StartError::Busy));
        let cancel = Cancel::after(stop);
        let mut emitted = 0;
        let mut terminal = 0;
        while let Some(event) = service.next_event(&cancel) {
            match event {
                Event::Token { position, .. } => {
                    assert_eq!(position, prompt.len() as u64 + emitted as u64);
                    emitted += 1;
                }
                Event::Cancelled { usage } => {
                    terminal += 1;
                    assert_eq!(usage.completion_tokens, emitted);
                    assert_eq!(usage.prompt_tokens, prompt.len());
                    assert_eq!(service.charged_bytes(), 0);
                }
                Event::Failed { error, .. } => panic!("unexpected failure at {stop}: {error}"),
                Event::Finished { .. } => panic!("missed cancellation {stop}/{boundaries}"),
                _ => {}
            }
        }
        assert_eq!(terminal, 1);
        assert!(service.next_event(&Cancel::never()).is_none());
        service.start(req).unwrap();
        assert_eq!(tokens(&mut service).len(), 5);
    }
    println!("cancelled every one of {boundaries} service boundaries; every restart passed");
}

#[test]
fn request_refusal_failure_disconnect_and_atomic_signal_release_resources() {
    let first_cancel = Cancel::never();
    first_cancel
        .signal()
        .store(true, std::sync::atomic::Ordering::Relaxed);
    assert!(first_cancel.check("local").is_err());
    assert!(Cancel::never().check("unrelated generation").is_ok());
    let mut fixture = fixture::build(2, 4, 16, 1).unwrap();
    let mut owner = ledger();
    {
        let mut service = GenerationService::new(&mut owner, fixture.program());
        let base = request(&[0, 1], 2, 0.0);
        for bad in [
            Request {
                prompt: &[],
                ..base
            },
            Request {
                prompt: &[16],
                ..base
            },
            Request {
                max_new_tokens: 0,
                ..base
            },
            Request {
                max_new_tokens: usize::MAX,
                ..base
            },
            Request {
                prefill_chunk: 0,
                ..base
            },
            Request {
                prefill_chunk: 257,
                ..base
            },
            Request {
                temperature: f64::NAN,
                ..base
            },
            Request {
                temperature: -1.0,
                ..base
            },
        ] {
            assert!(matches!(service.start(bad), Err(StartError::Rejected(_))));
            assert_eq!(service.charged_bytes(), 0);
        }
        service.start(base).unwrap();
        let signal = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let cancel = Cancel::with_signal(signal);
        assert!(matches!(
            service.next_event(&cancel),
            Some(Event::Admitted { .. })
        ));
        assert!(matches!(
            service.next_event(&cancel),
            Some(Event::Cancelled { .. })
        ));
        assert_eq!(service.charged_bytes(), 0);
        service.start(base).unwrap();
        service.next_event(&Cancel::never()); // disconnect/drop before any token
    }
    assert!(owner.outstanding().is_empty());
    // Finite but extreme weights cause a real numerical execution failure.
    for id in fixture.graph.weights() {
        let old = fixture.weights.get(*id).unwrap().as_float().unwrap();
        fixture.weights.set(
            *id,
            Value::Float(
                HostTensor::bf16(
                    vec![f32::from_bits(0x7f7f0000); old.data().len()],
                    old.shape().to_vec(),
                )
                .unwrap(),
            ),
        );
    }
    let mut service = GenerationService::new(&mut owner, fixture.program());
    service.start(request(&[0, 1], 2, 1.0)).unwrap();
    let mut failed = false;
    while let Some(event) = service.next_event(&Cancel::never()) {
        if let Event::Failed { error, usage } = event {
            assert!(matches!(error, Error::Numerical { .. }));
            assert_eq!(usage.completion_tokens, 0);
            failed = true;
        }
    }
    assert!(failed);
    assert_eq!(service.charged_bytes(), 0);
}

#[test]
fn cli_and_service_share_tokens_and_reject_unknown_or_duplicate_options() {
    let args: Vec<String> =
        "diagnostic --shape b --prompt 0,1,2,3 --max-new 5 --chunk 3 --temperature 1 --seed 42"
            .split_whitespace()
            .map(str::to_owned)
            .collect();
    let options = Options::parse(&args).unwrap();
    let fixture = fixture::build(3, 4, 7, 2).unwrap();
    let mut owner = ledger();
    let mut service = GenerationService::new(&mut owner, fixture.program());
    service.start(options.request()).unwrap();
    let reference = tokens(&mut service);
    service.start(options.request()).unwrap();
    let mut output = Vec::new();
    assert!(render(&mut service, &Cancel::never(), &mut output).unwrap());
    let text = String::from_utf8(output).unwrap();
    let actual: Vec<u32> = text
        .lines()
        .filter(|l| l.starts_with("event=token "))
        .map(|l| {
            l.split_whitespace()
                .nth(1)
                .unwrap()
                .strip_prefix("id=")
                .unwrap()
                .parse()
                .unwrap()
        })
        .collect();
    assert_eq!(reference, actual);
    let binary = std::process::Command::new(env!("CARGO_BIN_EXE_moxie"))
        .args(&args)
        .output()
        .unwrap();
    assert!(
        binary.status.success(),
        "{}",
        String::from_utf8_lossy(&binary.stderr)
    );
    assert_eq!(text, String::from_utf8(binary.stdout).unwrap());
    for bad in [
        "diagnostic --top-p 0.5",
        "diagnostic --seed 1 --seed 2",
        "diagnostic --prompt",
        "chat",
    ] {
        assert!(
            Options::parse(
                &bad.split_whitespace()
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            )
            .is_err()
        );
    }
    struct Broken;
    impl std::io::Write for Broken {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::ErrorKind::BrokenPipe.into())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    service.start(options.request()).unwrap();
    assert!(render(&mut service, &Cancel::never(), &mut Broken).is_err());
    drop(service);
    assert!(owner.outstanding().is_empty());
}
