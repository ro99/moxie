//! Fixtures for task 0025: synthetic shards, selections and scratch
//! directories, built here so that every test states its own source bytes.
//!
//! Nothing in this module calls the code under test. A shard is assembled from
//! a JSON header and a payload written by hand, and the expected canonical
//! bytes are computed from the values that went in -- so a test comparing them
//! is comparing the repacker against an independent statement of the format,
//! not against itself.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// A temporary directory that removes itself.
pub struct Scratch {
    pub path: PathBuf,
}

impl Scratch {
    pub fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "moxie-repack-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("a clock after 1970")
                .as_nanos()
        ));
        std::fs::create_dir_all(&path).expect("a scratch directory");
        Self { path }
    }

    pub fn join(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }

    /// Total bytes under this directory, for the disk-budget gates.
    pub fn bytes_used(&self) -> u64 {
        fn walk(dir: &Path) -> u64 {
            let mut total = 0;
            let Ok(entries) = std::fs::read_dir(dir) else {
                return 0;
            };
            for entry in entries.filter_map(|e| e.ok()) {
                let meta = match entry.metadata() {
                    Ok(m) => m,
                    Err(_) => continue,
                };
                total += if meta.is_dir() {
                    walk(&entry.path())
                } else {
                    meta.len()
                };
            }
            total
        }
        walk(&self.path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// BF16 bits of an f32, round-to-nearest-even -- the same rule the codec uses,
/// written out here so the fixture does not borrow it.
pub fn bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounded = bits + 0x7FFF + lsb;
    (rounded >> 16) as u16
}

pub fn bf16_bytes(v: f32) -> [u8; 2] {
    bf16_bits(v).to_le_bytes()
}

/// One tensor in a synthetic safetensors shard.
pub struct Entry {
    pub name: String,
    pub dtype: &'static str,
    pub shape: Vec<u64>,
    pub data: Vec<u8>,
}

impl Entry {
    pub fn new(name: &str, dtype: &'static str, shape: Vec<u64>, data: Vec<u8>) -> Self {
        Self {
            name: name.to_string(),
            dtype,
            shape,
            data,
        }
    }
}

/// Write a safetensors file: an eight-byte length, a JSON header, payloads.
pub fn write_shard(path: &Path, entries: &[Entry]) {
    let mut header = String::from("{");
    let mut payload = Vec::new();
    for (i, e) in entries.iter().enumerate() {
        if i > 0 {
            header.push(',');
        }
        let shape: Vec<String> = e.shape.iter().map(|d| d.to_string()).collect();
        header.push_str(&format!(
            "\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            e.name,
            e.dtype,
            shape.join(","),
            payload.len(),
            payload.len() + e.data.len()
        ));
        payload.extend_from_slice(&e.data);
    }
    header.push('}');
    // Pad to eight bytes, which the reader accepts and real writers emit.
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&payload);
    std::fs::write(path, out).expect("writing a shard");
}

/// A `pack-quantized` module's source payloads, from explicit values.
pub struct Module {
    pub rows: usize,
    pub columns: usize,
    pub group: usize,
    pub bits: u32,
    /// Unsigned source codes, `[rows][columns]`.
    pub codes: Vec<Vec<u32>>,
    /// Scales, `[rows][groups]`, as f32 values.
    pub scales: Vec<Vec<f32>>,
    /// Unsigned source zero points, `[rows][groups]`; empty when symmetric.
    pub zeros: Vec<Vec<u32>>,
}

impl Module {
    pub fn groups(&self) -> usize {
        if self.group == 0 {
            1
        } else {
            self.columns.div_ceil(self.group)
        }
    }

    pub fn values_per_word(&self) -> usize {
        32 / self.bits as usize
    }

    /// `weight_packed`: I32 words along the **input** axis.
    pub fn packed(&self) -> Vec<u8> {
        let per_word = self.values_per_word();
        let words = self.columns.div_ceil(per_word);
        let mut out = Vec::new();
        for row in &self.codes {
            for w in 0..words {
                let mut word: u32 = 0;
                for lane in 0..per_word {
                    let k = w * per_word + lane;
                    let code = row.get(k).copied().unwrap_or(0);
                    word |= code << (lane as u32 * self.bits);
                }
                out.extend_from_slice(&word.to_le_bytes());
            }
        }
        out
    }

    pub fn packed_shape(&self) -> Vec<u64> {
        vec![
            self.rows as u64,
            self.columns.div_ceil(self.values_per_word()) as u64,
        ]
    }

    /// `weight_zero_point`: I32 words along the **output** axis.
    pub fn zero_point(&self) -> Vec<u8> {
        let per_word = self.values_per_word();
        let word_rows = self.rows.div_ceil(per_word);
        let mut out = Vec::new();
        for wr in 0..word_rows {
            for g in 0..self.groups() {
                let mut word: u32 = 0;
                for lane in 0..per_word {
                    let o = wr * per_word + lane;
                    let z = self
                        .zeros
                        .get(o)
                        .and_then(|r| r.get(g))
                        .copied()
                        .unwrap_or(0);
                    word |= z << (lane as u32 * self.bits);
                }
                out.extend_from_slice(&word.to_le_bytes());
            }
        }
        out
    }

    pub fn zero_point_shape(&self) -> Vec<u64> {
        vec![
            self.rows.div_ceil(self.values_per_word()) as u64,
            self.groups() as u64,
        ]
    }

    /// `weight_scale`, in one of the three encodings.
    pub fn scale_payload(&self, dtype: &str) -> Vec<u8> {
        let mut out = Vec::new();
        for row in &self.scales {
            for v in row {
                match dtype {
                    "BF16" => out.extend_from_slice(&bf16_bytes(*v)),
                    "F16" => out.extend_from_slice(&f16_bytes(*v)),
                    "F32" => out.extend_from_slice(&v.to_le_bytes()),
                    other => panic!("unsupported scale dtype {other}"),
                }
            }
        }
        out
    }

    pub fn scale_shape(&self) -> Vec<u64> {
        vec![self.rows as u64, self.groups() as u64]
    }

    pub fn weight_shape(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(self.rows as i64).to_le_bytes());
        out.extend_from_slice(&(self.columns as i64).to_le_bytes());
        out
    }

    /// The canonical payload ADR 0023 describes, built from the values above
    /// rather than by the codec: codes, then scales, then zero points.
    pub fn expected_canonical(&self, scale_dtype: &str) -> Vec<u8> {
        let bias = 1i32 << (self.bits - 1);
        let mut out = Vec::new();
        // Codes: logical row-major, each row byte-aligned, low nibble first.
        for row in &self.codes {
            let mut bytes = Vec::new();
            match self.bits {
                4 => {
                    let stride = self.columns.div_ceil(2);
                    bytes.resize(stride, 0u8);
                    for (k, code) in row.iter().enumerate() {
                        let signed = (*code as i32 - bias) as i8 as u8 & 0x0F;
                        if k.is_multiple_of(2) {
                            bytes[k / 2] |= signed;
                        } else {
                            bytes[k / 2] |= signed << 4;
                        }
                    }
                }
                8 => {
                    for code in row {
                        bytes.push((*code as i32 - bias) as i8 as u8);
                    }
                }
                other => panic!("unsupported width {other}"),
            }
            out.extend_from_slice(&bytes);
        }
        // Scales: row-major, in the source encoding, unchanged.
        out.extend_from_slice(&self.scale_payload(scale_dtype));
        // Zero points: row-major, signed i16 little-endian.
        for row in &self.zeros {
            for z in row {
                let signed = (*z as i32 - bias) as i16;
                out.extend_from_slice(&signed.to_le_bytes());
            }
        }
        out
    }
}

/// IEEE-754 binary16 bytes of an f32, for the values these fixtures use.
///
/// Deliberately simple: the fixtures only use values that are exact in f16, and
/// a rounding rule this file does not need is a rounding rule it should not
/// contain.
pub fn f16_bytes(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    let sign = ((bits >> 31) as u16) << 15;
    let exponent = ((bits >> 23) & 0xFF) as i32 - 127;
    let mantissa = bits & 0x007F_FFFF;
    assert!(
        (-14..=15).contains(&exponent),
        "{v} is outside the normal f16 range this fixture uses"
    );
    assert_eq!(
        mantissa & 0x1FFF,
        0,
        "{v} does not fit in an f16 mantissa exactly"
    );
    let out = sign | (((exponent + 15) as u16) << 10) | ((mantissa >> 13) as u16);
    out.to_le_bytes()
}

/// A selection document naming one or more tensors.
pub struct SelectionBuilder {
    pub model: String,
    pub revision: String,
    pub completeness: String,
    pub tensors: Vec<String>,
}

impl SelectionBuilder {
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            revision: "r0".into(),
            completeness: "status = \"partial\"\nmissing = [\"everything else\"]".into(),
            tensors: Vec::new(),
        }
    }

    pub fn complete(mut self) -> Self {
        self.completeness = "status = \"complete\"\nmissing = []".into();
        self
    }

    pub fn pack_quantized(
        mut self,
        role: &str,
        module: &str,
        width: &str,
        group: &str,
        zero_points: &str,
        files: &BTreeMap<&str, &str>,
    ) -> Self {
        let mut entry = format!(
            "\n[[tensor]]\nrole = \"{role}\"\nkind = \"pack-quantized\"\nmodule = \"{module}\"\n\
             width = \"{width}\"\ngroup = {group}\nzero_points = \"{zero_points}\"\n\
             [tensor.files]\n"
        );
        for (suffix, file) in files {
            entry.push_str(&format!("{suffix} = \"{file}\"\n"));
        }
        self.tensors.push(entry);
        self
    }

    pub fn bf16(mut self, role: &str, name: &str, file: &str) -> Self {
        self.tensors.push(format!(
            "\n[[tensor]]\nrole = \"{role}\"\nkind = \"bf16\"\nname = \"{name}\"\nfile = \"{file}\"\n"
        ));
        self
    }

    pub fn text(&self) -> String {
        const HEX: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        format!(
            "version = 1\n\n[source]\nmodel = \"{}\"\nrevision = \"{}\"\n\
             license = \"unlicensed-test-fixture\"\n\n\
             [tokenizer]\nname = \"none\"\nversion = \"not-selected\"\ndigest = \"{HEX}\"\n\n\
             [template]\nname = \"none\"\nversion = \"not-selected\"\ndigest = \"{HEX}\"\n\n\
             [architecture]\nname = \"synthetic\"\nversion = \"1\"\n\
             [architecture.metadata]\nnote = \"opaque to every shared crate\"\n\n\
             [provenance]\nscale_convention = \"affine-v1\"\nquantizer = \"none\"\n\
             calibration = \"none\"\n\n[completeness]\n{}\n{}",
            self.model,
            self.revision,
            self.completeness,
            self.tensors.join("")
        )
    }

    pub fn write(&self, path: &Path) {
        std::fs::write(path, self.text()).expect("writing a selection");
    }
}

/// The repack binary, as Cargo built it.
pub fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_moxie-repack"))
}

/// What one invocation of the program said.
#[derive(Debug)]
pub struct Run {
    pub status: i32,
    pub stdout: String,
    pub stderr: String,
}

impl Run {
    /// The value of one `key: value` line, if it is there.
    pub fn field(&self, key: &str) -> Option<&str> {
        self.stdout
            .lines()
            .find_map(|l| l.strip_prefix(&format!("{key}: ")))
    }

    pub fn outcome(&self) -> &str {
        self.field("outcome").unwrap_or("<no outcome line>")
    }

    pub fn says(&self, needle: &str) -> bool {
        self.stdout.contains(needle) || self.stderr.contains(needle)
    }
}

/// Run the real binary.
pub fn run(args: &[&str]) -> Run {
    let out = std::process::Command::new(binary())
        .args(args)
        .output()
        .expect("the repack binary runs");
    Run {
        status: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// The acceptance profile's budgets, as command-line arguments.
pub fn budgets() -> Vec<&'static str> {
    vec![
        "--total-bytes",
        "128MiB",
        "--header-bytes",
        "64MiB",
        "--scratch-bytes",
        "1MiB",
        "--chunk-file-bytes",
        "4MiB",
        "--disk-bytes",
        "64MiB",
    ]
}
