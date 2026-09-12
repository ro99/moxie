//! Host reference execution against the admitted physical pages. Dense histories
//! below are ephemeral oracle scratch, never another persistent cache or journal.
use moxie_format::bf16::{bf16_bits_to_f32, f32_to_bf16_bits};
use moxie_graph::{Bindings, Graph, OpParams, ValueId};
use moxie_oracles::attention::KvHistory;
use moxie_state::{KvRow, LogitsHandle, PagedExecutionBinding, PagedSequence, ROOT};
use moxie_types::{Error, Precision, Result, StateTransactionId, SymbolTable};

use crate::{
    Cancel, HostTensor, Interpreter, KvCache, Value, bound_precision, try_clone_slice, try_vec,
};

/// One immutable graph/weight configuration with sole authority to extend a
/// particular paged sequence. Shared borrows keep the graph and weight payloads
/// immutable for the binding's lifetime; a second configuration cannot claim
/// the sequence after rows exist.
#[derive(Debug)]
pub struct PagedExecution<'a> {
    graph: &'a Graph,
    weights: &'a Bindings<Value>,
    tokens: ValueId,
    positions: ValueId,
    binding: PagedExecutionBinding,
}

impl<'a> PagedExecution<'a> {
    pub fn bind(
        graph: &'a Graph,
        weights: &'a Bindings<Value>,
        tokens: ValueId,
        positions: ValueId,
        sequence: &mut PagedSequence,
    ) -> Result<Self> {
        if tokens == positions
            || graph.inputs().len() != 2
            || !graph.inputs().contains(&tokens)
            || !graph.inputs().contains(&positions)
            || weights.len() != graph.weights().len()
        {
            return Err(invalid(
                "program",
                "exact token/position inputs and weight bindings required",
            ));
        }
        for id in graph.weights() {
            if weights.get(*id).is_none() {
                return Err(invalid("weights", "missing graph weight"));
            }
        }
        let binding = sequence.claim_execution()?;
        Ok(Self {
            graph,
            weights,
            tokens,
            positions,
            binding,
        })
    }

    /// Execute dynamic input rows with fallible payload copies. Any failure
    /// after a valid transaction is presented aborts that transaction.
    pub fn run(
        &self,
        sequence: &mut PagedSequence,
        txn: StateTransactionId,
        tokens: &[u64],
        positions: &[u64],
        cancel: &Cancel,
    ) -> Result<PagedOutput> {
        // Establish transaction authority before doing fallible preparation.
        // Every later error belongs to this transaction and therefore restores
        // all earlier work in it, including work from a prior successful call.
        sequence.validate_transaction(txn)?;
        let result = (|| {
            sequence.validate_execution(self.binding)?;
            let mut bindings = Bindings::new();
            for id in self.graph.weights() {
                bindings.set(
                    *id,
                    self.weights
                        .get(*id)
                        .expect("validated immutable program")
                        .try_clone()?,
                );
            }
            bindings.set(self.tokens, Value::Index(try_clone_slice(tokens)?));
            bindings.set(self.positions, Value::Index(try_clone_slice(positions)?));
            Interpreter::new().evaluate_paged(self.graph, &bindings, sequence, txn, cancel)
        })();
        if result.is_err() && sequence.validate_transaction(txn).is_ok() {
            sequence.abort(txn).expect("validated local transaction");
        }
        result
    }
}

pub(crate) trait HistorySource {
    fn read_history(&self, layer: u32) -> Result<KvHistory>;
}
impl HistorySource for KvCache {
    fn read_history(&self, layer: u32) -> Result<KvHistory> {
        Ok(self.history(layer)?.clone())
    }
}
impl HistorySource for PagedSequence {
    fn read_history(&self, layer: u32) -> Result<KvHistory> {
        let mut history = KvHistory::try_with_capacity(self.usage().rows)?;
        for position in 0..self.usage().rows as u64 {
            let row = self.row(layer as usize, position)?;
            let decode = |bytes: &[u8]| -> Result<Vec<f32>> {
                let mut decoded = try_vec(bytes.len() / 2)?;
                decoded.extend(
                    bytes
                        .chunks_exact(2)
                        .map(|b| bf16_bits_to_f32(u16::from_le_bytes([b[0], b[1]]))),
                );
                Ok(decoded)
            };
            history.append(position, decode(row.key)?, decode(row.value)?)?;
        }
        Ok(history)
    }
}

/// Immutable output paired with the existing sequence's result identity.
/// Callers can inspect numerical values but cannot replace the bound logits.
#[derive(Debug)]
pub struct PagedOutput {
    logits: HostTensor,
    retained: LogitsHandle,
    prefix: u64,
}
impl PagedOutput {
    pub fn logits(&self) -> &HostTensor {
        &self.logits
    }
    pub fn prefix(&self) -> u64 {
        self.prefix
    }

    /// Revalidate the live sequence-issued identity, then use the accepted
    /// sampler. The output's last row predicts the token after its prefix.
    pub fn stage(
        &self,
        sequence: &mut PagedSequence,
        txn: StateTransactionId,
        temperature: f64,
        cancel: &Cancel,
    ) -> Result<u32> {
        sequence.validate_transaction(txn)?;
        if sequence.state().retained_logits(ROOT)? != Some(self.retained)
            || !sequence.state().next_logits_valid(ROOT)
        {
            return Err(invalid(
                "logits",
                "forward output is foreign, stale or pending materialization",
            ));
        }
        cancel.check("sample/prepare")?;
        sequence
            .prepare_sample(
                txn,
                self.prefix,
                self.logits.row(self.logits.rows() - 1)?,
                None,
                temperature,
            )?
            .stage(cancel.signal())
    }
}

fn invalid(field: &'static str, detail: &str) -> Error {
    Error::InvalidRequest {
        field,
        detail: detail.into(),
    }
}

impl Interpreter {
    fn evaluate_paged(
        &self,
        graph: &Graph,
        bindings: &Bindings<Value>,
        sequence: &mut PagedSequence,
        txn: StateTransactionId,
        cancel: &Cancel,
    ) -> Result<PagedOutput> {
        cancel.check("forward/start")?;
        let geometry = sequence.geometry();
        if geometry.precision != Precision::Bf16
            || graph.attention_layers().is_empty()
            || graph.attention_layers().len() != geometry.layers
        {
            return Err(invalid(
                "graph",
                "paged reference requires exact BF16 attention layer coverage",
            ));
        }
        for node in graph.nodes() {
            // `kv_heads`, not `heads`: the pages store keys and values, and
            // under GQA there are fewer of those than there are query heads.
            // The two coincided while every graph was multi-head, which is how
            // this read `heads` and still passed.
            if let OpParams::Attention {
                kv_heads,
                head_dim,
                layer,
                ..
            } = node.params
                && (kv_heads as usize != geometry.kv_heads
                    || head_dim as usize != geometry.key_dim
                    || head_dim as usize != geometry.value_dim
                    || layer as usize >= geometry.layers)
            {
                return Err(invalid(
                    "graph",
                    "attention geometry differs from physical pages",
                ));
            }
        }
        if bindings.len() != graph.inputs().len() + graph.weights().len() {
            return Err(invalid(
                "bindings",
                "exact graph input and weight bindings required",
            ));
        }
        let mut values = try_vec(graph.value_count())?;
        values.resize_with(graph.value_count(), || None);
        for id in graph.inputs().iter().chain(graph.weights()) {
            values[id.0 as usize] = Some(
                bindings
                    .get(*id)
                    .ok_or_else(|| invalid("bindings", "missing graph value"))?
                    .try_clone()?,
            );
        }
        let positions = self.positions_of(graph, &values)?;
        let rows = positions.len();
        let start = sequence.usage().rows as u64;
        if rows == 0
            || start
                .checked_add(rows as u64)
                .is_none_or(|end| end > geometry.max_tokens as u64)
            || positions
                .iter()
                .enumerate()
                .any(|(i, p)| *p != start + i as u64)
        {
            return Err(invalid(
                "positions",
                "nonempty contiguous absolute positions must fit pages",
            ));
        }
        let mut symbols = SymbolTable::new();
        symbols.declare(graph.rows_symbol(), "rows");
        symbols.bind(graph.rows_symbol(), rows as u64);
        for id in graph.inputs().iter().chain(graph.weights()) {
            let value = values[id.0 as usize].as_ref().expect("bound above");
            let spec = graph.spec(*id).expect("graph value");
            let shape: Vec<u64> = match value {
                Value::Float(t) => {
                    let mut shape = try_vec(t.shape().len())?;
                    shape.extend(t.shape().iter().map(|x| *x as u64));
                    shape
                }
                Value::Index(v) => {
                    let mut shape = try_vec(1)?;
                    shape.push(v.len() as u64);
                    shape
                }
            };
            if shape != spec.extent(&symbols)? || bound_precision(value) != spec.role.precision() {
                return Err(invalid("bindings", "shape or precision differs from graph"));
            }
            if let Value::Float(t) = value
                && t.data().iter().any(|x| !x.is_finite())
            {
                return Err(Error::InvalidArtifact {
                    detail: "nonfinite binding".into(),
                });
            }
        }
        let mut staged = try_vec(
            rows.checked_mul(geometry.layers)
                .ok_or(moxie_types::DimError::Overflow)?,
        )?;
        for node in graph.nodes() {
            cancel.check(node.params.op().name())?;
            let value = self.eval(node, &values, sequence, &mut staged, &positions)?;
            if let Value::Float(t) = &value
                && t.data().iter().any(|x| !x.is_finite())
            {
                return Err(Error::Numerical {
                    detail: format!("{} produced nonfinite output", node.params.op().name()),
                });
            }
            values[node.output.0 as usize] = Some(value);
        }
        let logits = values[graph.output().0 as usize]
            .take()
            .ok_or_else(|| invalid("graph", "no output"))?
            .as_float()?
            .try_clone()?;
        if logits.rows() != rows || logits.precision() != Precision::F32 {
            return Err(invalid(
                "logits",
                "expected FP32 logits for each executed row",
            ));
        }
        for position in positions {
            let mut encoded = try_vec(geometry.layers)?;
            for layer in 0..geometry.layers {
                let row = staged
                    .iter()
                    .find(|a| a.layer == layer as u32 && a.position == position)
                    .ok_or_else(|| invalid("graph", "missing layer row"))?;
                let encode = |data: &[f32]| -> Result<Vec<u8>> {
                    let mut out = try_vec(
                        data.len()
                            .checked_mul(2)
                            .ok_or(moxie_types::DimError::Overflow)?,
                    )?;
                    out.extend(data.iter().flat_map(|x| f32_to_bf16_bits(*x).to_le_bytes()));
                    Ok(out)
                };
                encoded.push((encode(&row.key)?, encode(&row.value)?));
            }
            let mut views = try_vec(encoded.len())?;
            views.extend(encoded.iter().map(|(k, v)| KvRow { key: k, value: v }));
            cancel.check("forward/before_append")?;
            sequence.append(txn, position, &views, cancel.signal())?;
            cancel.check("forward/after_append")?;
        }
        let retained = sequence.record_logits(txn)?;
        cancel.check("forward/after_logits")?;
        Ok(PagedOutput {
            logits,
            retained,
            prefix: start + rows as u64,
        })
    }
}
