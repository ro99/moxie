//! The host key/value cache.
//!
//! One [`KvHistory`] per layer, indexed by absolute sequence position. This is
//! the M1 reference form: a dense `Vec` per layer, no pages, no eviction, no
//! device memory. Document 06 puts paged device state in M4 and the real
//! residency authority in M2, and this must not grow into either -- what it is
//! for is giving the interpreter somewhere to append so the state contracts have
//! a consumer.
//!
//! `StateKind::KvPages` is `RestoreCapability::Truncate`, and [`KvCache::rollback_to`]
//! is what that means physically.

use moxie_oracles::attention::KvHistory;
use moxie_types::{Error, Result};

/// Per-layer key/value history.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct KvCache {
    layers: Vec<KvHistory>,
}

impl KvCache {
    /// A cache for `layers` attention layers.
    pub fn new(layers: usize) -> Self {
        Self {
            layers: vec![KvHistory::new(); layers],
        }
    }

    pub fn layers(&self) -> usize {
        self.layers.len()
    }

    pub fn history(&self, layer: u32) -> Result<&KvHistory> {
        self.layers
            .get(layer as usize)
            .ok_or(Error::InvalidRequest {
                field: "layer",
                detail: format!("layer {layer} of {}", self.layers.len()),
            })
    }

    pub fn append(
        &mut self,
        layer: u32,
        position: u64,
        key: Vec<f32>,
        value: Vec<f32>,
    ) -> Result<()> {
        let n = self.layers.len();
        self.layers
            .get_mut(layer as usize)
            .ok_or(Error::InvalidRequest {
                field: "layer",
                detail: format!("layer {layer} of {n}"),
            })?
            .append(position, key, value)
    }

    /// Drop everything at or after `prefix`, on every layer.
    ///
    /// The physical half of a rollback. `moxie-state` decides *whether* the
    /// rollback is allowed -- published output, restore evidence, lineage -- and
    /// this performs it once that has been agreed.
    pub fn rollback_to(&mut self, prefix: u64) {
        for l in &mut self.layers {
            l.truncate(prefix);
        }
    }

    /// The number of positions held, which must agree across layers.
    pub fn len(&self) -> usize {
        self.layers.first().map(KvHistory::len).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether every layer holds the same number of positions.
    ///
    /// A layer that fell behind means a step wrote some layers and not others,
    /// which is precisely what staging appends is meant to prevent.
    pub fn is_coherent(&self) -> bool {
        let n = self.len();
        self.layers.iter().all(|l| l.len() == n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_cache_is_empty_and_coherent() {
        let kv = KvCache::new(3);
        assert_eq!(kv.layers(), 3);
        assert!(kv.is_empty());
        assert!(kv.is_coherent());
    }

    #[test]
    fn appends_are_per_layer_and_positions_must_be_dense() {
        let mut kv = KvCache::new(2);
        kv.append(0, 0, vec![1.0], vec![2.0]).unwrap();
        assert!(!kv.is_coherent(), "layer 1 has not been written yet");
        kv.append(1, 0, vec![3.0], vec![4.0]).unwrap();
        assert!(kv.is_coherent());
        assert_eq!(kv.len(), 1);

        assert!(kv.append(0, 2, vec![1.0], vec![2.0]).is_err(), "gap");
        assert!(
            kv.append(9, 1, vec![1.0], vec![2.0]).is_err(),
            "no such layer"
        );
        assert!(kv.history(9).is_err());
    }

    #[test]
    fn rollback_truncates_every_layer_to_the_same_prefix() {
        let mut kv = KvCache::new(2);
        for p in 0..5u64 {
            kv.append(0, p, vec![p as f32], vec![-(p as f32)]).unwrap();
            kv.append(1, p, vec![p as f32 * 2.0], vec![0.0]).unwrap();
        }
        assert_eq!(kv.len(), 5);
        kv.rollback_to(3);
        assert_eq!(kv.len(), 3);
        assert!(kv.is_coherent());

        // Truncation restores exactly the earlier cache, which is what
        // RestoreCapability::Truncate claims.
        let mut fresh = KvCache::new(2);
        for p in 0..3u64 {
            fresh
                .append(0, p, vec![p as f32], vec![-(p as f32)])
                .unwrap();
            fresh.append(1, p, vec![p as f32 * 2.0], vec![0.0]).unwrap();
        }
        assert_eq!(kv, fresh);
    }
}
