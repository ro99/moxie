unsafe extern "C" {
    fn launch_my_private_kernel(p: *mut core::ffi::c_void) -> i32;
}

pub fn graph() {}
