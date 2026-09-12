//! Isolated process-wide allocator instrumentation; one test avoids cross-test noise.
use moxie_cli::{fixture, gemma};
use moxie_engine::{
    Cancel, GenerationEvent, GenerationRequest,
    service::{GenerationService, StartError},
};
use moxie_interp::paged::PagedExecution;
use moxie_memory::{CapacitySnapshot, Ledger};
use moxie_state::{KvGeometry, PagedSequence, ROOT};
use moxie_types::{Error, HostTier, Precision, Scope, Tier};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static FAIL: AtomicUsize = AtomicUsize::new(0);
struct Count;
// SAFETY: all successful allocations/deallocations delegate unchanged to System;
// the only injected failure is the allocator's documented null return.
unsafe impl GlobalAlloc for Count {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if FAIL
            .compare_exchange(layout.size(), 0, SeqCst, SeqCst)
            .is_ok()
        {
            return std::ptr::null_mut();
        }
        // SAFETY: layout is the caller's allocator contract.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let now = LIVE.fetch_add(layout.size(), SeqCst) + layout.size();
            PEAK.fetch_max(now, SeqCst);
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), SeqCst);
        // SAFETY: forwarded with the original allocation layout.
        unsafe { System.dealloc(ptr, layout) };
    }
}
#[global_allocator]
static ALLOC: Count = Count;

fn ledger(bytes: u64) -> Ledger {
    Ledger::new([CapacitySnapshot::new(Scope::Host, bytes + 1, 1).unwrap()]).unwrap()
}
fn req(prompt: &[u32], chunk: usize, max: usize) -> GenerationRequest<'_> {
    GenerationRequest {
        prompt,
        prefill_chunk: chunk,
        max_new_tokens: max,
        temperature: 1.0,
        seed: 42,
    }
}

#[test]
fn admitted_peak_cleanup_repeated_generations_and_allocation_failure() {
    for (heads, dim, vocab, layers, prompt_len, chunk) in [
        (2, 4, 16, 1, 37, 13),
        (3, 4, 7, 2, 251, 65),
        (4, 64, 256, 8, 255, 255),
    ] {
        let fixture = fixture::build(heads, dim, vocab, layers).unwrap();
        let prompt: Vec<_> = (0..prompt_len).map(|i| i % vocab as u32).collect();
        let before_ledger = LIVE.load(SeqCst);
        let mut owner = ledger(1 << 30);
        let baseline = LIVE.load(SeqCst);
        PEAK.store(baseline, SeqCst);
        let mut service = GenerationService::new(&mut owner, fixture.program());
        service.start(req(&prompt, chunk, 1)).unwrap();
        let charged = service.charged_bytes();
        while let Some(event) = service.next_event(&Cancel::never()) {
            assert!(!matches!(
                event,
                GenerationEvent::Failed { .. } | GenerationEvent::Cancelled { .. }
            ));
            assert!(
                PEAK.load(SeqCst) <= baseline + charged as usize,
                "peak {} > reserve {}",
                PEAK.load(SeqCst) - baseline,
                charged
            );
        }
        assert_eq!(service.charged_bytes(), 0);
        drop(service);
        assert!(owner.outstanding().is_empty());
        let peak = PEAK.load(SeqCst) - baseline;
        // Ledger retains bounded zero-valued tier accounting nodes until drop.
        drop(owner);
        assert!(LIVE.load(SeqCst) <= before_ledger);
        println!(
            "shape={heads}x{dim} layers={layers} prompt={prompt_len} chunk={chunk} peak_delta={peak} admitted={charged}"
        );
    }
    // The reduced Gemma-like graphs, which are wider per layer than the
    // synthetic fixtures and reserve less per stored row: under grouped-query
    // attention the pages hold `kv_heads * head_dim`, not `heads * head_dim`.
    // A reserve computed from the query width would over-admit here and the
    // printed numbers are what shows it does not.
    for (shape, prompt_len, chunk, maximum) in [
        (gemma::Shape::A, 37usize, 13usize, 1usize),
        (gemma::Shape::A, 251, 65, 1),
        (gemma::Shape::B, 255, 255, 1),
    ] {
        let config = shape.config();
        let built = gemma::build(shape).unwrap();
        let prompt: Vec<u32> = (0..prompt_len)
            .map(|i| (i as u64 % config.vocab) as u32)
            .collect();
        let before_ledger = LIVE.load(SeqCst);
        let mut owner = ledger(1 << 30);
        let baseline = LIVE.load(SeqCst);
        PEAK.store(baseline, SeqCst);
        let mut service = GenerationService::new(&mut owner, built.program());
        service.start(req(&prompt, chunk, maximum)).unwrap();
        let charged = service.charged_bytes();
        while let Some(event) = service.next_event(&Cancel::never()) {
            assert!(!matches!(
                event,
                GenerationEvent::Failed { .. } | GenerationEvent::Cancelled { .. }
            ));
            assert!(
                PEAK.load(SeqCst) <= baseline + charged as usize,
                "{shape:?}: peak {} > reserve {}",
                PEAK.load(SeqCst) - baseline,
                charged
            );
        }
        assert_eq!(service.charged_bytes(), 0);
        drop(service);
        assert!(owner.outstanding().is_empty());
        let peak = PEAK.load(SeqCst) - baseline;
        drop(owner);
        assert!(LIVE.load(SeqCst) <= before_ledger);
        println!(
            "shape={} layers={} heads={} local_kv={}x{} global_kv={}x{} window={} \
             prompt={prompt_len} chunk={chunk} peak_delta={peak} admitted={charged}",
            shape.name(),
            config.layers,
            config.heads,
            config.local_kv_heads,
            config.local_head_dim,
            config.global_kv_heads,
            config.global_head_dim,
            config.sliding_window
        );
    }
    // Repeated generations on a reduced Gemma graph retain nothing: the third
    // run's live heap is the first's, which is what "no retained growth" means
    // when the graph has six layers of paged history rather than one.
    {
        let built = gemma::build(gemma::Shape::A).unwrap();
        let prompt: Vec<u32> = (0..23).map(|i| i % 11).collect();
        let mut owner = ledger(1 << 30);
        let mut service = GenerationService::new(&mut owner, built.program());
        service.start(req(&prompt, 5, 2)).unwrap();
        while service.next_event(&Cancel::never()).is_some() {}
        let settled = LIVE.load(SeqCst);
        for round in 0..64 {
            service.start(req(&prompt, 5, 2)).unwrap();
            while service.next_event(&Cancel::never()).is_some() {}
            assert_eq!(service.charged_bytes(), 0, "round {round}");
            assert!(
                LIVE.load(SeqCst) <= settled,
                "round {round}: {} > {settled}",
                LIVE.load(SeqCst)
            );
        }
        drop(service);
        assert!(owner.outstanding().is_empty());
    }
    let fixture = fixture::build(3, 4, 7, 2).unwrap();
    let prompt = [0; 37];
    let mut owner = ledger(1 << 30);
    let mut service = GenerationService::new(&mut owner, fixture.program());
    // Initialize the existing ledger's fixed tier nodes before testing growth.
    service.start(req(&prompt, 7, 3)).unwrap();
    while service.next_event(&Cancel::never()).is_some() {}
    let baseline = LIVE.load(SeqCst);
    for i in 0..1000 {
        service.start(req(&prompt, 7, 3)).unwrap();
        let cancel = Cancel::after(i % 75);
        while service.next_event(&cancel).is_some() {}
        assert_eq!(service.charged_bytes(), 0);
        assert!(LIVE.load(SeqCst) <= baseline);
    }
    drop(service);
    assert!(owner.outstanding().is_empty());

    // Review regression: the widest fixture has a 65,536-element BF16-valued
    // weight payload. Fail its 262,144-byte forward clone after admission. The
    // service must return a typed terminal event, close, and accept a retry.
    let wide = fixture::build(4, 64, 256, 8).unwrap();
    let wide_prompt: Vec<_> = (0..255).map(|i| i as u32).collect();
    let mut wide_owner = ledger(1 << 30);
    let mut wide_service = GenerationService::new(&mut wide_owner, wide.program());
    wide_service.start(req(&wide_prompt, 255, 1)).unwrap();
    assert!(matches!(
        wide_service.next_event(&Cancel::never()),
        Some(GenerationEvent::Admitted { .. })
    ));
    FAIL.store(262_144, SeqCst);
    assert!(matches!(
        wide_service.next_event(&Cancel::never()),
        Some(GenerationEvent::Failed {
            error: Error::CapacityExceeded {
                tier: Some(Tier::Host(HostTier::CpuWorkspace)),
                requested_bytes: 262_144,
                available_bytes: 0,
            },
            ..
        })
    ));
    assert_eq!(FAIL.swap(0, SeqCst), 0, "forward fault was not injected");
    assert!(wide_service.is_idle());
    assert_eq!(wide_service.charged_bytes(), 0);
    wide_service.start(req(&wide_prompt, 255, 1)).unwrap();
    while let Some(event) = wide_service.next_event(&Cancel::never()) {
        assert!(!matches!(
            event,
            GenerationEvent::Failed { .. } | GenerationEvent::Cancelled { .. }
        ));
    }
    drop(wide_service);
    assert!(wide_owner.outstanding().is_empty());

    // Direct API regression: preparation failure on a later call must abort
    // the entire already-mutated transaction, not only the failing call.
    let direct = fixture::build(4, 64, 256, 1).unwrap();
    let geometry = KvGeometry::uniform(1, 4, 64, 64, Precision::Bf16, 7, 8);
    let mut direct_owner = ledger(1 << 30);
    let mut pages = PagedSequence::with_sampling(&mut direct_owner, geometry, 256, 4, 0).unwrap();
    pages.append_prompt(2).unwrap();
    let baseline = pages.state().frontiers(ROOT).unwrap();
    let execution = PagedExecution::bind(
        &direct.graph,
        &direct.weights,
        direct.tokens,
        direct.positions,
        &mut pages,
    )
    .unwrap();
    let txn = pages.begin().unwrap();
    execution
        .run(&mut pages, txn, &[0], &[0], &Cancel::never())
        .unwrap();
    assert_eq!(pages.usage().rows, 1);
    assert_eq!(pages.state().frontiers(ROOT).unwrap().executed, 1);
    assert_eq!(pages.state().live_results().len(), 1);
    FAIL.store(262_144, SeqCst);
    assert!(matches!(
        execution.run(&mut pages, txn, &[1], &[1], &Cancel::never()),
        Err(Error::CapacityExceeded {
            tier: Some(Tier::Host(HostTier::CpuWorkspace)),
            requested_bytes: 262_144,
            available_bytes: 0,
        })
    ));
    assert_eq!(FAIL.swap(0, SeqCst), 0, "direct fault was not injected");
    assert_eq!(pages.usage().rows, 0);
    assert!(pages.row(0, 0).is_err());
    assert_eq!(pages.state().frontiers(ROOT).unwrap(), baseline);
    assert!(pages.state().live_results().is_empty());
    assert!(pages.state().open_transactions().is_empty());

    let retry = pages.begin().unwrap();
    execution
        .run(&mut pages, retry, &[0, 1], &[0, 1], &Cancel::never())
        .unwrap();
    pages.commit_prefix(retry, 0).unwrap();
    assert_eq!(pages.usage().rows, 2);
    assert_eq!(pages.state().frontiers(ROOT).unwrap().executed, 2);
    assert!(pages.state().open_transactions().is_empty());
    pages.close(&mut direct_owner).unwrap();
    assert!(direct_owner.outstanding().is_empty());

    // Fail only the newly owned prompt payload and the existing paged lineage
    // allocation after both admissions. Check the original error attribution.
    // Use extents distinct from the planner's small metadata allocations.
    let failure_prompt = [0; 193];
    for (bytes, tier) in [
        (193 * 4, Some(Tier::Host(HostTier::StateSpill))),
        ((193 + 3 + 1) * 8, Some(Tier::Host(HostTier::Pageable))),
        // 28 pages per layer, and pages now belong to one layer, so the page
        // table has 2 * 28 entries rather than 28 shared ones. The pools are
        // unchanged: 196 rows * 2 layers * 48 K/V bytes. Plus 16*(3+7)
        // history/count bytes and 8*7 workspace bytes.
        (2 * 28 * 8 + 196 * 2 * 48 + 16 * (3 + 7) + 8 * 7, None),
    ] {
        let mut service = GenerationService::new(&mut owner, fixture.program());
        FAIL.store(bytes, SeqCst);
        let error = service.start(req(&failure_prompt, 7, 3)).unwrap_err();
        assert_eq!(
            FAIL.swap(0, SeqCst),
            0,
            "allocation failure was not injected"
        );
        assert_eq!(
            error,
            StartError::Rejected(Error::CapacityExceeded {
                tier,
                requested_bytes: bytes as u64,
                available_bytes: 0
            })
        );
        assert_eq!(service.charged_bytes(), 0);
        drop(service);
        assert!(owner.outstanding().is_empty());
        for tier in [
            HostTier::CpuWorkspace,
            HostTier::StateSpill,
            HostTier::Pageable,
        ] {
            assert_eq!(owner.committed(Scope::Host, Tier::Host(tier)), 0);
        }
    }
    let mut tiny = ledger(1024);
    let mut service = GenerationService::new(&mut tiny, fixture.program());
    assert!(matches!(
        service.start(req(&prompt, 7, 3)),
        Err(StartError::Rejected(Error::CapacityExceeded { .. }))
    ));
    assert_eq!(service.charged_bytes(), 0);
    println!(
        "1000 cancellation/restart cycles: no retained growth; allocation refusals: full cleanup"
    );
}
