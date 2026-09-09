//! Opaque identifiers.
//!
//! These are newtypes rather than bare integers so that a chunk id cannot be
//! passed where a layout id is expected. Document 03 requires canonical chunk
//! identity and prepared-layout identity to be *separate*: a buffer packed for
//! one kernel must not be consumed by another under a different interpretation.

macro_rules! opaque_id {
    ($(#[$m:meta])* $name:ident, $inner:ty) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub $inner);

        impl $name {
            pub const fn get(self) -> $inner { self.0 }
        }

        impl core::fmt::Display for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(f, "{}({})", stringify!($name), self.0)
            }
        }
    };
}

opaque_id!(
    /// A canonical artifact: one converted checkpoint at one precision profile.
    ArtifactId, u64);
opaque_id!(
    /// A logical tensor role within an artifact.
    TensorId, u64);
opaque_id!(
    /// Immutable canonical chunk identity: (artifact, tensor/expert, logical
    /// range, format version). Document 03.
    ChunkId, u64);
opaque_id!(
    /// Prepared-layout identity: (chunk, device capability, kernel layout
    /// version). Deliberately *not* interchangeable with `ChunkId` -- see R17.
    LayoutId, u64);
opaque_id!(
    /// A compiled semantic graph.
    GraphId, u64);
opaque_id!(
    /// A copy-on-write branch of sequence state.
    BranchId, u64);
opaque_id!(
    /// An open state transaction. Committed only on `commit_prefix`.
    StateTransactionId, u64);
opaque_id!(
    /// A CUDA device ordinal. Diagnostic only -- evidence identifies a GPU by
    /// UUID (document 07). Valid only under CUDA_DEVICE_ORDER=PCI_BUS_ID.
    DeviceId, u32);
opaque_id!(
    /// An execution rank. One rank owns one device context (document 01).
    RankId, u32);

/// A GPU's stable identity: the 16 bytes behind CUDA's canonical
/// `GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` string.
///
/// AGENTS.md: "Ordinals are diagnostics only -- plans, manifests and results
/// identify a GPU by UUID." That is why this is a value type that can key a map
/// while [`DeviceId`] cannot: an ordinal is only meaningful inside one process's
/// visible device set, and on this machine ordinal 0 is the 5060 Ti rather than
/// one of the 3090 pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DeviceUuid([u8; 16]);

impl DeviceUuid {
    /// The groups of hex digits in the canonical form, in order.
    const GROUPS: [usize; 5] = [8, 4, 4, 4, 12];

    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        DeviceUuid(bytes)
    }

    pub const fn to_bytes(self) -> [u8; 16] {
        self.0
    }

    /// Parse the canonical form CUDA and `nvidia-smi` report. Strict on purpose:
    /// a UUID that round-trips through a lenient parser is a UUID that can be
    /// written two ways, and two spellings of one device is exactly the identity
    /// collision this type exists to prevent.
    pub fn parse(s: &str) -> crate::Result<Self> {
        let bad = |detail: String| crate::Error::InvalidRequest {
            field: "device_uuid",
            detail,
        };
        let body = s.strip_prefix("GPU-").ok_or_else(|| {
            bad(format!(
                "{s:?} does not start with the canonical `GPU-` prefix"
            ))
        })?;
        let mut bytes = [0u8; 16];
        let mut out = 0usize;
        let mut rest = body;
        for (i, width) in Self::GROUPS.iter().enumerate() {
            if i > 0 {
                rest = rest.strip_prefix('-').ok_or_else(|| {
                    bad(format!("{s:?} is missing the separator before group {i}"))
                })?;
            }
            if rest.len() < *width {
                return Err(bad(format!("{s:?} is truncated in group {i}")));
            }
            let (group, tail) = rest.split_at(*width);
            rest = tail;
            let mut chars = group.chars();
            for _ in 0..width / 2 {
                // Two hex digits per byte; every group has an even width.
                let hi = chars.next().expect("group width checked above");
                let lo = chars.next().expect("group width checked above");
                let digit = |c: char| {
                    c.to_digit(16)
                        .filter(|_| !c.is_ascii_uppercase())
                        .ok_or_else(|| bad(format!("{s:?} has non-canonical hex digit {c:?}")))
                };
                bytes[out] = (digit(hi)? * 16 + digit(lo)?) as u8;
                out += 1;
            }
        }
        if !rest.is_empty() {
            return Err(bad(format!("{s:?} has trailing characters {rest:?}")));
        }
        debug_assert_eq!(out, 16);
        Ok(DeviceUuid(bytes))
    }
}

impl core::fmt::Display for DeviceUuid {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("GPU-")?;
        let mut at = 0usize;
        for (i, width) in DeviceUuid::GROUPS.iter().enumerate() {
            if i > 0 {
                f.write_str("-")?;
            }
            for _ in 0..width / 2 {
                write!(f, "{:02x}", self.0[at])?;
                at += 1;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_and_layout_ids_are_distinct_types() {
        // This is the R17 lesson expressed in the type system: the same integer
        // is a different thing depending on which identity it names.
        let chunk = ChunkId(7);
        let layout = LayoutId(7);
        assert_eq!(chunk.get(), layout.get());
        assert_eq!(chunk.to_string(), "ChunkId(7)");
        assert_eq!(layout.to_string(), "LayoutId(7)");
        // `chunk == layout` does not compile, which is the point.
    }

    #[test]
    fn a_device_uuid_round_trips_through_its_canonical_string() {
        let text = "GPU-1a2b3c4d-5e6f-4718-9a0b-c1d2e3f40506";
        let u = DeviceUuid::parse(text).unwrap();
        assert_eq!(u.to_string(), text);
        assert_eq!(DeviceUuid::from_bytes(u.to_bytes()), u);
    }

    #[test]
    fn only_the_canonical_spelling_parses() {
        // Every rejection here is a spelling that would otherwise give one
        // physical GPU two identities.
        for bad in [
            "1a2b3c4d-5e6f-4718-9a0b-c1d2e3f40506",       // no prefix
            "GPU-1A2B3C4D-5E6F-4718-9A0B-C1D2E3F40506",   // upper case
            "GPU-1a2b3c4d5e6f47189a0bc1d2e3f40506",       // no separators
            "GPU-1a2b3c4d-5e6f-4718-9a0b-c1d2e3f4050",    // truncated
            "GPU-1a2b3c4d-5e6f-4718-9a0b-c1d2e3f4050600", // trailing
            "GPU-1a2b3c4d-5e6f-4718-9a0b-c1d2e3g40506",   // not hex
            "",
        ] {
            let e = DeviceUuid::parse(bad).unwrap_err();
            assert_eq!(e.kind(), "invalid_request", "{bad:?} must be refused");
        }
    }

    #[test]
    fn distinct_devices_are_distinct_and_ordered() {
        let a = DeviceUuid::parse("GPU-00000000-0000-0000-0000-000000000001").unwrap();
        let b = DeviceUuid::parse("GPU-00000000-0000-0000-0000-000000000002").unwrap();
        assert_ne!(a, b);
        assert!(a < b);
        // An ordinal is not identity: the same ordinal may name either of these
        // depending on CUDA_VISIBLE_DEVICES, which is why DeviceId keys nothing.
        assert_eq!(DeviceId(0).get(), 0);
    }
}
