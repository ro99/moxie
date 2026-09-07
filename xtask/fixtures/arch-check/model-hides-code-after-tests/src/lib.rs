pub fn graph() {}

#[cfg(test)]
mod tests {
    #[test]
    fn a_brace_in_a_string_must_not_confuse_the_matcher() {
        let _ = "}";
    }
}

// Production code again. Everything below here was invisible to the checker
// before the brace-matching fix.
unsafe extern "C" {
    fn launch_private_kernel(p: *mut core::ffi::c_void) -> i32;
}

pub fn evict_expert() {
    let _ = std::fs::read("/weights");
}
