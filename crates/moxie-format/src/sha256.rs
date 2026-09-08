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
        let mut w = [0u32; 64];
        for (i, word) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) = (
            self.h[0], self.h[1], self.h[2], self.h[3], self.h[4], self.h[5], self.h[6], self.h[7],
        );
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (dst, v) in self.h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *dst = dst.wrapping_add(v);
        }
    }

    pub fn finalize_hex(mut self) -> String {
        let bit_len = self.total_len.wrapping_mul(8);
        let mut pad = [0u8; 64];
        pad[0] = 0x80;
        let pad_len = if self.buf_len < 56 {
            56 - self.buf_len
        } else {
            120 - self.buf_len
        };
        let (p1, _) = pad.split_at(pad_len);
        let mut tmp = Vec::with_capacity(p1.len() + 8);
        tmp.extend_from_slice(p1);
        tmp.extend_from_slice(&bit_len.to_be_bytes());
        self.update(&tmp);
        debug_assert_eq!(self.buf_len, 0);
        self.h.iter().map(|w| format!("{w:08x}")).collect()
    }
}

impl Default for StreamingSha256 {
    fn default() -> Self {
        Self::new()
    }
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
