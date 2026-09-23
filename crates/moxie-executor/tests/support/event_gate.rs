// Test-only real stream gate: enqueue a bounded host function immediately
// before the actual event record. Copies remain real; completion cannot race
// past the first sweep. No production API or fabricated event status is used.
use std::ffi::{c_char, c_int, c_void};
use std::sync::atomic::{AtomicBool, Ordering::SeqCst};
use std::time::{Duration, Instant};

#[allow(dead_code)]
pub(super) static BLOCK_NEXT: AtomicBool = AtomicBool::new(false);
static RELEASE: AtomicBool = AtomicBool::new(true);
pub(super) static TIMED_OUT: AtomicBool = AtomicBool::new(false);

#[link(name = "dl")]
unsafe extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

unsafe extern "C" fn pending_work(_: *mut c_void) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !RELEASE.load(SeqCst) {
        if Instant::now() >= deadline {
            TIMED_OUT.store(true, SeqCst);
            break;
        }
        std::thread::park_timeout(Duration::from_millis(1));
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuEventRecord(event: *mut c_void, stream: *mut c_void) -> c_int {
    // SAFETY: RTLD_NEXT finds the real CUDA ABI symbols after this executable.
    let (launch, record) = unsafe {
        let launch = dlsym((-1isize) as *mut c_void, c"cuLaunchHostFunc".as_ptr());
        let record = dlsym((-1isize) as *mut c_void, c"cuEventRecord".as_ptr());
        if launch.is_null() || record.is_null() {
            return 1;
        }
        (
            std::mem::transmute::<
                *mut c_void,
                unsafe extern "C" fn(
                    *mut c_void,
                    unsafe extern "C" fn(*mut c_void),
                    *mut c_void,
                ) -> c_int,
            >(launch),
            std::mem::transmute::<
                *mut c_void,
                unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int,
            >(record),
        )
    };
    if BLOCK_NEXT.swap(false, SeqCst) {
        // SAFETY: stream comes unchanged from CUDA; callback uses static state,
        // calls no CUDA API and exits within ten seconds even on test failure.
        let result = unsafe { launch(stream, pending_work, std::ptr::null_mut()) };
        if result != 0 {
            return result;
        }
    }
    // SAFETY: event and stream are forwarded unchanged to the real driver.
    unsafe { record(event, stream) }
}

pub(super) struct PendingGate;

impl PendingGate {
    pub(super) fn arm() -> Self {
        RELEASE.store(false, SeqCst);
        TIMED_OUT.store(false, SeqCst);
        BLOCK_NEXT.store(true, SeqCst);
        Self
    }
}

impl Drop for PendingGate {
    fn drop(&mut self) {
        BLOCK_NEXT.store(false, SeqCst);
        RELEASE.store(true, SeqCst);
    }
}
