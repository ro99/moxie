//! Which rank holds which GPU, process-wide.
//!
//! A registry rather than a convention: two contexts on one card is exactly the
//! shared mutable CUDA state that rank ownership exists to remove (document 01),
//! and nothing in the driver prevents it.
//!
//! This module is compiled in **both** lanes and touches no driver symbol. The
//! bookkeeping is where the exclusivity guarantee actually lives, and it is
//! worth testing without a GPU present: the ordering rule below cannot be
//! observed from the outside without pausing a CUDA call.

use std::collections::BTreeMap;
use std::sync::{LazyLock, Mutex, MutexGuard};

use moxie_types::{DeviceUuid, Error, RankId, Result};

#[derive(Debug, Clone)]
pub(crate) enum Claim {
    Held(RankId),
    /// The holder's context teardown failed. The card is not handed out again:
    /// a release that errored leaves a context reference in an unknown state,
    /// and advertising it as free is the one answer nothing can recover from.
    TeardownFailed {
        rank: RankId,
        detail: String,
    },
}

fn table() -> MutexGuard<'static, BTreeMap<DeviceUuid, Claim>> {
    static CLAIMS: LazyLock<Mutex<BTreeMap<DeviceUuid, Claim>>> =
        LazyLock::new(|| Mutex::new(BTreeMap::new()));
    // A poisoned registry still describes which devices are held. Refusing to
    // read it would strand every card in the process.
    CLAIMS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Register `rank` as the holder of `uuid`.
pub(crate) fn claim(uuid: DeviceUuid, rank: RankId) -> Result<()> {
    let mut held = table();
    match held.get(&uuid) {
        Some(Claim::Held(holder)) => {
            return Err(Error::Unsupported {
                capability: "rank_context",
                reason: format!("{uuid} is already held by rank {}", holder.get()),
            });
        }
        Some(Claim::TeardownFailed { rank, detail }) => {
            return Err(Error::Unsupported {
                capability: "rank_context",
                reason: format!(
                    "{uuid} is not being handed out: teardown for rank {} failed ({detail})",
                    rank.get()
                ),
            });
        }
        None => {}
    }
    if let Some((other, _)) = held
        .iter()
        .find(|(_, c)| matches!(c, Claim::Held(r) if *r == rank))
    {
        // A rank whose teardown failed is not counted as holding anything: the
        // device is withheld, but the rank is not blamed for it forever.
        return Err(Error::Unsupported {
            capability: "rank_context",
            reason: format!(
                "rank {} already holds {other}; one rank owns one device",
                rank.get()
            ),
        });
    }
    held.insert(uuid, Claim::Held(rank));
    Ok(())
}

/// Drop a claim that was never attached to a context.
pub(crate) fn abandon(uuid: DeviceUuid) {
    table().remove(&uuid);
}

/// Release `uuid`, running `teardown` **while the claim is still held**.
///
/// The order is the guarantee. Removing the claim first lets another rank
/// acquire the card while the previous context reference is still outstanding,
/// which is two live primary contexts on one device -- and it can defeat the
/// device reset that happens when the last reference goes.
///
/// The lock is deliberately not held across `teardown`: the claim's presence is
/// what excludes another rank, not the mutex, and holding it would deadlock any
/// teardown that needed to consult the registry.
pub(crate) fn release_with<F>(uuid: DeviceUuid, rank: RankId, teardown: F)
where
    F: FnOnce() -> Result<()>,
{
    let outcome = teardown();
    let mut held = table();
    match outcome {
        Ok(()) => {
            held.remove(&uuid);
        }
        Err(e) => {
            // Fail closed, and say why, so the next `claim` reports the real
            // reason rather than a bare refusal.
            held.insert(
                uuid,
                Claim::TeardownFailed {
                    rank,
                    detail: e.to_string(),
                },
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uuid(n: u8) -> DeviceUuid {
        let mut bytes = [0u8; 16];
        bytes[15] = n;
        // A second byte keeps these apart from the fixtures other modules use.
        bytes[0] = 0xc1;
        DeviceUuid::from_bytes(bytes)
    }

    #[test]
    fn a_held_device_is_refused_and_the_refusal_names_the_holder() {
        let (d, r) = (uuid(1), RankId(101));
        claim(d, r).unwrap();
        let e = claim(d, RankId(102)).unwrap_err();
        assert_eq!(e.kind(), "unsupported");
        assert!(e.to_string().contains(&d.to_string()), "{e}");
        assert!(e.to_string().contains("rank 101"), "{e}");
        release_with(d, r, || Ok(()));
    }

    #[test]
    fn one_rank_owns_one_device() {
        let (a, b, r) = (uuid(2), uuid(3), RankId(103));
        claim(a, r).unwrap();
        let e = claim(b, r).unwrap_err();
        assert!(e.to_string().contains("one rank owns one device"), "{e}");
        release_with(a, r, || Ok(()));
        assert!(claim(b, r).is_ok());
        release_with(b, r, || Ok(()));
    }

    #[test]
    fn the_device_stays_held_until_teardown_has_finished() {
        // The property the ordering exists for. Removing the claim first lets
        // another rank acquire the card while the previous context reference is
        // still outstanding, which is two live primary contexts on one device.
        let (d, r) = (uuid(4), RankId(104));
        claim(d, r).unwrap();
        release_with(d, r, || {
            let e = claim(d, RankId(105))
                .expect_err("the device must not be available while teardown is still running");
            assert!(e.to_string().contains("rank 104"), "{e}");
            Ok(())
        });
        // ...and available immediately afterwards.
        claim(d, RankId(105)).unwrap();
        release_with(d, RankId(105), || Ok(()));
    }

    #[test]
    fn a_device_whose_teardown_failed_is_not_handed_out() {
        // Fail closed. A release that errored leaves a context reference in an
        // unknown state, and advertising the card as free is the one answer that
        // cannot be recovered from.
        let (d, r) = (uuid(5), RankId(106));
        claim(d, r).unwrap();
        release_with(d, r, || {
            Err(Error::DeviceLost {
                device: 0,
                detail: "release failed".into(),
            })
        });
        let e = claim(d, RankId(107)).unwrap_err();
        assert_eq!(e.kind(), "unsupported");
        assert!(e.to_string().contains("teardown"), "{e}");
        assert!(e.to_string().contains("rank 106"), "{e}");
    }
}
