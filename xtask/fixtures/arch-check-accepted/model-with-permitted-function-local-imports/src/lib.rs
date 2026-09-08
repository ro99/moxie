//! A model definition: metadata and graph composition, nothing else.

pub mod layers;

pub fn describe() -> String {
    use std::{collections::BTreeMap, fmt::Write as _};
    let mut counts: BTreeMap<&str, u32> = BTreeMap::new();
    counts.insert("linear", 2);
    let mut out = String::new();
    for (name, n) in &counts {
        let _ = write!(out, "{name}={n};");
    }
    out
}

pub fn width() -> usize {
    use std::cmp::max as biggest;
    biggest(core::mem::size_of::<u32>(), 2)
}

mod inner {
    pub fn ratio(a: u32, b: u32) -> f64 {
        f64::from(a) / f64::from(b)
    }
}

pub fn ratio() -> f64 {
    inner::ratio(3, 4)
}

#[cfg(test)]
mod tests {
    use std::{fs};

    #[test]
    fn a_test_harness_may_read_a_fixture_file() {
        // Permitted: document 02 exempts the dev harness. The traversal now
        // visits function bodies, so this exemption has to hold there too.
        let _ = fs::read("/dev/null");
        let _ = std::thread::current().id();
        assert!(super::ratio() > 0.0);
    }
}
