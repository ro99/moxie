//! Isolated process-wide allocator instrumentation; one test avoids cross-test noise.
use moxie_cli::fixture;
use moxie_engine::{
    Cancel, GenerationEvent, GenerationRequest,
    service::{GenerationService, StartError},
};
use moxie_memory::{CapacitySnapshot, Ledger};
use moxie_types::{Error, HostTier, Scope, Tier};
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

    // Fail only the newly owned prompt payload and the existing paged lineage
    // allocation after both admissions. Check the original error attribution.
    // Use extents distinct from the planner's small metadata allocations.
    let failure_prompt = [0; 193];
    for (bytes, tier) in [
        (193 * 4, Some(Tier::Host(HostTier::StateSpill))),
        ((193 + 3 + 1) * 8, Some(Tier::Host(HostTier::Pageable))),
        // 28 page-table entries + 196 rows * 2 layers * 48 K/V bytes,
        // plus 16*(3+7) history/count bytes and 8*7 workspace bytes.
        (28 * 8 + 196 * 2 * 48 + 16 * (3 + 7) + 8 * 7, None),
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
