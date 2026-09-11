//! Pure base sampling over caller-owned, bounded byte storage.
//! No allocation, state authority, cache access or device integration.

use moxie_types::{DimError, Error, HostTier, Result, Tier};

pub const RNG_PROFILE: &str = "moxie-philox4x32-10-cdf-v1";

fn invalid(field: &'static str) -> Error {
    Error::InvalidRequest {
        field,
        detail: "invalid sampling input".into(),
    }
}

fn numerical() -> Error {
    Error::Numerical {
        detail: "no finite legal distribution".into(),
    }
}

fn get(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("checked storage"),
    )
}
fn put(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

/// Ascending-order compensated FP64 reduction. A plain sum of 100,000 uniform
/// probabilities exceeds the declared normalization bound solely from rounding.
#[derive(Default)]
struct Sum {
    value: f64,
    correction: f64,
}
impl Sum {
    fn add(&mut self, value: f64) {
        let corrected = value - self.correction;
        let next = self.value + corrected;
        self.correction = (next - self.value) - corrected;
        self.value = next;
    }
    fn of(values: impl Iterator<Item = f64>) -> f64 {
        let mut sum = Self::default();
        for value in values {
            sum.add(value);
        }
        sum.value
    }
}

/// Fixed byte geometry, checked before admission. No pointers or allocations.
#[derive(Debug, Clone, Copy)]
pub struct Layout {
    vocabulary: usize,
    capacity: usize,
    state: usize,
    workspace: usize,
}

impl Layout {
    pub fn new(vocabulary: usize, capacity: usize) -> Result<Self> {
        if vocabulary == 0 || vocabulary > u32::MAX as usize || capacity == 0 {
            return Err(invalid("sampling_geometry"));
        }
        let state = capacity
            .checked_add(vocabulary)
            .and_then(|n| n.checked_mul(16))
            .ok_or(DimError::Overflow)?;
        let workspace = vocabulary.checked_mul(8).ok_or(DimError::Overflow)?;
        if state.checked_add(workspace).ok_or(DimError::Overflow)? > isize::MAX as usize {
            return Err(DimError::Overflow.into());
        }
        Ok(Self {
            vocabulary,
            capacity,
            state,
            workspace,
        })
    }
    pub fn vocabulary(self) -> usize {
        self.vocabulary
    }
    pub fn capacity(self) -> usize {
        self.capacity
    }
    pub fn state_bytes(self) -> usize {
        self.state
    }
    pub fn workspace_bytes(self) -> usize {
        self.workspace
    }
}

/// History metadata; all variable-size data lives in the supplied byte storage.
/// This has no transaction identity or independent commit/abort authority.
#[derive(Debug)]
pub struct History {
    layout: Layout,
    len: usize,
    committed: usize,
}

impl History {
    pub fn new(layout: Layout, bytes: &mut [u8]) -> Result<Self> {
        if bytes.len() != layout.state_bytes() {
            return Err(invalid("history_storage"));
        }
        bytes.fill(0);
        Ok(Self {
            layout,
            len: 0,
            committed: 0,
        })
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn committed_len(&self) -> usize {
        self.committed
    }
    pub fn layout(&self) -> Layout {
        self.layout
    }
    fn count_offset(&self, token: usize, committed: bool) -> usize {
        16 * self.layout.capacity + 8 * (token + if committed { self.layout.vocabulary } else { 0 })
    }
    pub fn view<'a>(&'a self, bytes: &'a [u8], committed: bool) -> Result<HistoryView<'a>> {
        if bytes.len() != self.layout.state_bytes() {
            return Err(invalid("history_storage"));
        }
        Ok(HistoryView {
            history: self,
            bytes,
            committed,
        })
    }
    pub fn append(&mut self, bytes: &mut [u8], position: u64, token: u32) -> Result<()> {
        if bytes.len() != self.layout.state_bytes() || token as usize >= self.layout.vocabulary {
            return Err(invalid("history_token"));
        }
        if self.len == self.layout.capacity {
            return Err(Error::CapacityExceeded {
                tier: Some(Tier::Host(HostTier::StateSpill)),
                requested_bytes: 16,
                available_bytes: 0,
            });
        }
        if self.len > 0 && position <= get(bytes, 16 * (self.len - 1)) {
            return Err(invalid("history_position"));
        }
        let count = self.count_offset(token as usize, false);
        put(bytes, count, get(bytes, count) + 1);
        put(bytes, 16 * self.len, position);
        put(bytes, 16 * self.len + 8, token as u64);
        self.len += 1;
        Ok(())
    }
    /// Apply suffix undo, including accumulated counts, without copying history.
    pub fn truncate(&mut self, bytes: &mut [u8], len: usize) -> Result<()> {
        if bytes.len() != self.layout.state_bytes() || len > self.len {
            return Err(invalid("history_prefix"));
        }
        while self.len > len {
            self.len -= 1;
            let token = get(bytes, 16 * self.len + 8) as usize;
            let count = self.count_offset(token, false);
            put(bytes, count, get(bytes, count) - 1);
            if self.len < self.committed {
                let count = self.count_offset(token, true);
                put(bytes, count, get(bytes, count) - 1);
                self.committed -= 1;
            }
            bytes[16 * self.len..16 * (self.len + 1)].fill(0);
        }
        Ok(())
    }
    /// Publish the prefix selected by the owning state transaction.
    pub fn publish(&mut self, bytes: &mut [u8], len: usize) -> Result<()> {
        if bytes.len() != self.layout.state_bytes() || len < self.committed || len > self.len {
            return Err(invalid("history_prefix"));
        }
        self.truncate(bytes, len)?;
        while self.committed < len {
            let token = get(bytes, 16 * self.committed + 8) as usize;
            let count = self.count_offset(token, true);
            put(bytes, count, get(bytes, count) + 1);
            self.committed += 1;
        }
        Ok(())
    }
    /// Rebuild counts from retained tokens for a completed explicit replay.
    /// Normal transaction abort instead uses the bounded suffix undo above.
    pub fn replay_prefix(&mut self, bytes: &mut [u8], len: usize) -> Result<()> {
        if len > self.committed {
            return Err(invalid("history_prefix"));
        }
        self.truncate(bytes, len)?;
        bytes[16 * self.layout.capacity..].fill(0);
        for i in 0..len {
            let token = get(bytes, 16 * i + 8) as usize;
            for committed in [false, true] {
                let offset = self.count_offset(token, committed);
                put(bytes, offset, get(bytes, offset) + 1);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HistoryView<'a> {
    history: &'a History,
    bytes: &'a [u8],
    committed: bool,
}
impl<'a> HistoryView<'a> {
    pub fn len(self) -> usize {
        if self.committed {
            self.history.committed
        } else {
            self.history.len
        }
    }
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
    pub fn entries(self, window: usize) -> impl Iterator<Item = (u64, u32)> + 'a {
        let start = if window == 0 {
            0
        } else {
            self.len().saturating_sub(window)
        };
        (start..self.len())
            .map(move |i| (get(self.bytes, 16 * i), get(self.bytes, 16 * i + 8) as u32))
    }
    pub fn count(self, token: u32) -> Result<u64> {
        if token as usize >= self.history.layout.vocabulary {
            return Err(invalid("token"));
        }
        Ok(get(
            self.bytes,
            self.history.count_offset(token as usize, self.committed),
        ))
    }
}

/// Normalized probabilities in ascending token order. Construction validates
/// every value. Borrowing prevents the backing workspace being overwritten.
#[derive(Debug, Clone, Copy)]
pub struct Distribution<'a> {
    bytes: &'a [u8],
}
impl<'a> Distribution<'a> {
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self> {
        if bytes.is_empty() || !bytes.len().is_multiple_of(8) || bytes.len() / 8 > u32::MAX as usize
        {
            return Err(invalid("distribution"));
        }
        let d = Self { bytes };
        let mut sum = Sum::default();
        for p in d.probabilities() {
            if !p.is_finite() || p < 0.0 {
                return Err(numerical());
            }
            sum.add(p);
        }
        if !sum.value.is_finite() || (sum.value - 1.0).abs() > 1e-12 {
            return Err(numerical());
        }
        Ok(d)
    }
    pub fn probabilities(self) -> impl Iterator<Item = f64> + 'a {
        self.bytes
            .chunks_exact(8)
            .map(|b| f64::from_bits(u64::from_le_bytes(b.try_into().expect("eight bytes"))))
    }
    pub fn draw_uniform(self, u: f64) -> Result<u32> {
        if !(0.0..1.0).contains(&u) {
            return Err(invalid("uniform"));
        }
        let target = u * Sum::of(self.probabilities());
        let mut cumulative = Sum::default();
        let mut last = None;
        for (i, p) in self.probabilities().enumerate() {
            if p > 0.0 {
                last = Some(i as u32);
                cumulative.add(p);
                if cumulative.value > target {
                    return Ok(i as u32);
                }
            }
        }
        last.ok_or_else(numerical)
    }
    pub fn draw(self, seed: u64, step: u64, greedy: bool) -> Result<u32> {
        if step == u64::MAX {
            return Err(invalid("rng_step"));
        }
        if greedy {
            let mut best = 0;
            let mut mass = -1.0;
            for (i, p) in self.probabilities().enumerate() {
                if p > mass {
                    best = i as u32;
                    mass = p;
                }
            }
            Ok(best)
        } else {
            self.draw_uniform(target_uniform(seed, step))
        }
    }
}

pub fn distribution<'a>(
    logits: &[f32],
    legal: Option<&[bool]>,
    temperature: f64,
    workspace: &'a mut [u8],
) -> Result<Distribution<'a>> {
    if logits.is_empty()
        || logits.len() > u32::MAX as usize
        || logits.len().checked_mul(8) != Some(workspace.len())
        || legal.is_some_and(|m| m.len() != logits.len())
    {
        return Err(invalid("vocabulary"));
    }
    if !temperature.is_finite() || !(0.0..=10.0).contains(&temperature) {
        return Err(invalid("temperature"));
    }
    let mut best = None;
    let mut maximum = f64::NEG_INFINITY;
    for (i, &l) in logits.iter().enumerate() {
        if l.is_nan() || l == f32::INFINITY {
            return Err(numerical());
        }
        if legal.is_none_or(|m| m[i]) && l.is_finite() && (l as f64) > maximum {
            best = Some(i);
            maximum = l as f64;
        }
    }
    let best = best.ok_or_else(numerical)?;
    let mut sum = Sum::default();
    for (i, &l) in logits.iter().enumerate() {
        let p = if temperature == 0.0 {
            if i == best { 1.0 } else { 0.0 }
        } else if legal.is_none_or(|m| m[i]) && l.is_finite() {
            ((l as f64 - maximum) / temperature).exp()
        } else {
            0.0
        };
        put(workspace, i * 8, p.to_bits());
        sum.add(p);
    }
    for i in 0..logits.len() {
        let p = f64::from_bits(get(workspace, i * 8)) / sum.value;
        put(workspace, i * 8, p.to_bits());
    }
    Distribution::from_bytes(workspace)
}

/// Closed RNG domains. Reserving vocabulary is not enabling those processors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Domain {
    Target = 0,
    Proposal = 1,
    Acceptance = 2,
    Filtering = 3,
}
pub fn counter(step: u64, domain: Domain) -> [u32; 4] {
    [step as u32, (step >> 32) as u32, domain as u32, 0]
}
pub fn philox(mut c: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
    for round in 0..10 {
        let a = 0xD2511F53u64 * c[0] as u64;
        let b = 0xCD9E8D57u64 * c[2] as u64;
        c = [
            (b >> 32) as u32 ^ c[1] ^ key[0],
            b as u32,
            (a >> 32) as u32 ^ c[3] ^ key[1],
            a as u32,
        ];
        if round != 9 {
            key[0] = key[0].wrapping_add(0x9E3779B9);
            key[1] = key[1].wrapping_add(0xBB67AE85);
        }
    }
    c
}
pub fn target_uniform(seed: u64, step: u64) -> f64 {
    let words = philox(
        counter(step, Domain::Target),
        [seed as u32, (seed >> 32) as u32],
    );
    let word = ((words[0] as u64) << 32) | words[1] as u64;
    (word >> 11) as f64 * (1.0 / 9007199254740992.0)
}
