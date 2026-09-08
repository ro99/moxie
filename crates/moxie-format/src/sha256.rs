//! SHA-256, FIPS 180-4.
//!
//! The single canonical implementation in the workspace. `xtask spec-check`
//! consumes this rather than keeping its own copy, and
//! `crates/moxie-kernels/build.rs` includes the raw file rather than
//! duplicating it, so the digest that identifies a kernel image is the same
//! function that checks a manifest checksum.
//!
//! No third-party dependency and no shell-out, for the same reason as before:
//! the hash is needed in build scripts and in I/O-free validation alike.

#![forbid(unsafe_code)]

include!("sha256_raw.rs");

/// Incremental SHA-256: feed slices as they are read, finalize once.
///
/// The bounded reader hashes *while* reading so a multi-gigabyte tensor never
/// sits in memory twice. This is the same compression function as
/// [`sha256_hex`], with the padding applied at finalization.
#[derive(Debug, Clone)]
pub struct StreamingSha256 {
    h: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total_len: u64,
}

impl StreamingSha256 {
    pub fn new() -> Self {
        Self {
            h: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buf: [0u8; 64],
            buf_len: 0,
            total_len: 0,
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.total_len = self.total_len.wrapping_add(data.len() as u64);
        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let block = self.buf;
                self.compress(&block);
                self.buf_len = 0;
            }
        }
        while data.len() >= 64 {
            let (block, rest) = data.split_at(64);
            let mut arr = [0u8; 64];
            arr.copy_from_slice(block);
            self.compress(&arr);
            data = rest;
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

    fn compress(&mut self, chunk: &[u8; 64]) {
        // The single hashing core, shared with the one-shot path: there is no
        // second compression implementation to drift.
        compress_block(&mut self.h, chunk);
    }

    pub fn finalize_hex(self) -> String {
        let bytes = self.finalize_bytes();
        bytes_to_hex(&bytes)
    }

    /// Finalize to raw bytes with no heap allocation.
    ///
    /// The bounded reader verifies checksums on this path so that hashing a
    /// multi-gigabyte tensor never allocates proportionally to it -- or at
    /// all. Padding runs through `update` over a stack buffer.
    pub fn finalize_bytes(mut self) -> [u8; 32] {
        let bit_len = self.total_len.wrapping_mul(8);
        let mut pad = [0u8; 128];
        pad[0] = 0x80;
        // 0x80, zeros to ≡56 (mod 64), then the 8 length bytes.
        let pad_len = if self.buf_len < 56 {
            64 - self.buf_len
        } else {
            128 - self.buf_len
        };
        pad[pad_len - 8..pad_len].copy_from_slice(&bit_len.to_be_bytes());
        self.update(&pad[..pad_len]);
        debug_assert_eq!(self.buf_len, 0);
        let mut out = [0u8; 32];
        for (i, w) in self.h.iter().enumerate() {
            out[4 * i..4 * i + 4].copy_from_slice(&w.to_be_bytes());
        }
        out
    }
}

impl Default for StreamingSha256 {
    fn default() -> Self {
        Self::new()
    }
}

/// Lowercase hex of 32 digest bytes. No allocation beyond the returned String.
fn bytes_to_hex(bytes: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(64);
    for b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 15) as usize] as char);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_the_published_vectors() {
        // Moved with the function from `xtask::speccheck`, per task 0005's
        // deletion plan. A home-grown digest that has never been checked
        // against a published vector is not an identity.
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        // Multi-block, crossing the length-padding boundary.
        assert_eq!(
            sha256_hex(&[b'a'; 1000]),
            "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3"
        );
    }

    #[test]
    fn finalize_bytes_agrees_with_finalize_hex() {
        // The allocation-free path must be the same digest, not a second one.
        let data: Vec<u8> = (0..300).map(|i| (i * 37 % 251) as u8).collect();
        for len in [0, 1, 55, 56, 64, 119, 120, 300] {
            let mut s = StreamingSha256::new();
            s.update(&data[..len]);
            let bytes = s.finalize_bytes();
            let mut s2 = StreamingSha256::new();
            s2.update(&data[..len]);
            assert_eq!(s2.finalize_hex(), sha256_hex(&data[..len]));
            assert_eq!(hex_of(&bytes), sha256_hex(&data[..len]));
        }
    }

    fn hex_of(bytes: &[u8; 32]) -> String {
        super::bytes_to_hex(bytes)
    }

    #[test]
    fn streaming_agrees_with_oneshot_on_every_boundary() {
        // Feed the same bytes in every split position around the block edge so
        // the buffered path cannot hide a length bug.
        let data: Vec<u8> = (0..300).map(|i| (i * 37 % 251) as u8).collect();
        let want = sha256_hex(&data);
        for split in [0, 1, 55, 56, 57, 63, 64, 65, 119, 120, 200] {
            let mut s = StreamingSha256::new();
            s.update(&data[..split.min(data.len())]);
            s.update(&data[split.min(data.len())..]);
            assert_eq!(s.finalize_hex(), want, "split at {split}");
        }
        // Byte by byte.
        let mut s = StreamingSha256::new();
        for b in &data {
            s.update(&[*b]);
        }
        assert_eq!(s.finalize_hex(), want);
    }
}
