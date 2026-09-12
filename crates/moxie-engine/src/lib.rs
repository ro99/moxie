//! Single-sequence orchestration. This first profile is explicitly a bounded
//! host-reference diagnostic, using the shared interpreter and paged transactions.
#![forbid(unsafe_code)]
pub mod service;

use moxie_graph::{Bindings, Graph, OpParams, ValueId, Visibility};
use moxie_interp::paged::{PagedExecution, PagedOutput};
pub use moxie_interp::{Cancel, HostTensor, Value};
use moxie_memory::{BufferRequest, HostBuffer, Ledger, PlanRequest, Reservation, StageSpan};
use moxie_state::{KvGeometry, LayerKv, PagedSequence, Retention};
use moxie_types::{DimError, Error, HostTier, Precision, Result, Scope, SymbolTable, Tier};

/// Immutable mathematical inputs supplied by the composition root. No execution
/// callback, mutable model state or client-owned token loop is accepted.
#[derive(Debug, Clone, Copy)]
pub struct Program<'a> {
    pub graph: &'a Graph,
    pub weights: &'a Bindings<Value>,
    pub tokens: ValueId,
    pub positions: ValueId,
}

#[derive(Debug, Clone, Copy)]
pub struct GenerationRequest<'a> {
    pub prompt: &'a [u32],
    pub max_new_tokens: usize,
    pub prefill_chunk: usize,
    pub temperature: f64,
    pub seed: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

#[derive(Debug, PartialEq)]
pub enum GenerationEvent {
    Admitted {
        profile: &'static str,
        sampler: &'static str,
        temperature: f64,
        seed: u64,
        context: usize,
        reserved_bytes: u64,
    },
    Prefill {
        processed: usize,
        total: usize,
    },
    Token {
        id: u32,
        position: u64,
    },
    Finished {
        usage: Usage,
    },
    Cancelled {
        usage: Usage,
    },
    Failed {
        error: Error,
        usage: Usage,
    },
}

fn invalid(field: &'static str, detail: &str) -> Error {
    Error::InvalidRequest {
        field,
        detail: detail.into(),
    }
}
fn unsupported(reason: &str) -> Error {
    Error::Unsupported {
        capability: "host-reference diagnostic",
        reason: reason.into(),
    }
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or(DimError::Overflow.into())
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or(DimError::Overflow.into())
}
fn try_vec<T>(capacity: usize) -> Result<Vec<T>> {
    let requested_bytes = mul(capacity, std::mem::size_of::<T>())?;
    let mut out = Vec::new();
    out.try_reserve_exact(capacity)
        .map_err(|_| Error::CapacityExceeded {
            tier: Some(Tier::Host(HostTier::CpuWorkspace)),
            requested_bytes: requested_bytes as u64,
            available_bytes: 0,
        })?;
    Ok(out)
}

fn try_clone_slice<T: Clone>(values: &[T]) -> Result<Vec<T>> {
    let mut out = try_vec(values.len())?;
    out.extend_from_slice(values);
    Ok(out)
}

#[derive(Debug)]
struct Envelope {
    geometry: KvGeometry,
    vocabulary: usize,
    workspace: usize,
}

impl Program<'_> {
    fn validate(self, request: GenerationRequest<'_>) -> Result<Envelope> {
        if request.prompt.is_empty() || request.max_new_tokens == 0 || request.prefill_chunk == 0 {
            return Err(invalid(
                "length",
                "prompt, maximum output and chunk must be positive",
            ));
        }
        if !request.temperature.is_finite() || !(0.0..=10.0).contains(&request.temperature) {
            return Err(invalid(
                "temperature",
                "expected finite temperature in [0,10]",
            ));
        }
        let context = add(request.prompt.len(), request.max_new_tokens)?;
        if context > 256 || request.prefill_chunk > 256 || self.graph.value_count() > 384 {
            return Err(unsupported(
                "context/chunk <=256 and graph values <=384 required",
            ));
        }
        if self.tokens == self.positions
            || self.graph.inputs().len() != 2
            || !self.graph.inputs().contains(&self.tokens)
            || !self.graph.inputs().contains(&self.positions)
            || self.weights.len() != self.graph.weights().len()
        {
            return Err(invalid(
                "program",
                "exact token/position inputs and weight bindings required",
            ));
        }
        let full = request.prefill_chunk.min(request.prompt.len());
        let tail = request.prompt.len() % request.prefill_chunk;
        let mut row_counts = [0usize; 3];
        let mut row_count_len = 0;
        for rows in [full, tail, usize::from(request.max_new_tokens > 1)] {
            if rows != 0 && !row_counts[..row_count_len].contains(&rows) {
                row_counts[row_count_len] = rows;
                row_count_len += 1;
            }
        }
        let row_counts = &row_counts[..row_count_len];
        let maximum_rows = *row_counts.iter().max().expect("nonempty prompt");
        let mut symbols = SymbolTable::new();
        symbols.declare(self.graph.rows_symbol(), "rows");
        symbols.bind(self.graph.rows_symbol(), maximum_rows as u64);
        let mut elements = 0;
        for rows in row_counts {
            symbols.bind(self.graph.rows_symbol(), *rows as u64);
            let mut row_elements = 0;
            for spec in self.graph.values() {
                let shape = spec.extent(&symbols)?;
                let count = shape.iter().try_fold(1usize, |n, x| {
                    mul(n, usize::try_from(*x).map_err(|_| DimError::Overflow)?)
                })?;
                if shape.is_empty() || shape.len() > 4 || count == 0 || count > 65_536 {
                    return Err(unsupported("tensor rank 1..4 and extent 1..65536 required"));
                }
                row_elements = add(row_elements, count)?;
            }
            elements = elements.max(row_elements);
        }
        for rows in row_counts {
            symbols.bind(self.graph.rows_symbol(), *rows as u64);
            for id in [self.tokens, self.positions] {
                let spec = self.graph.spec(id).expect("graph input");
                if !spec.role.is_index() || spec.extent(&symbols)? != [*rows as u64] {
                    return Err(invalid(
                        "program",
                        "token/position inputs must match every prefill tail and decode row count",
                    ));
                }
            }
        }
        symbols.bind(self.graph.rows_symbol(), maximum_rows as u64);
        for id in self.graph.weights() {
            let spec = self.graph.spec(*id).expect("graph weight");
            let value = self
                .weights
                .get(*id)
                .ok_or_else(|| invalid("weights", "missing weight"))?
                .as_float()?;
            let value_shape = try_clone_slice(value.shape())?;
            if value.precision() != Precision::Bf16
                || spec.role.precision() != Some(Precision::Bf16)
                || !value_shape
                    .iter()
                    .map(|x| *x as u64)
                    .eq(spec.extent(&symbols)?)
                || value.data().iter().any(|x| !x.is_finite())
            {
                return Err(invalid(
                    "weights",
                    "finite BF16 weights of the exact graph shape required",
                ));
            }
            // Weight dimensions must not depend on the dynamic row symbol, even
            // when this particular request happens to execute only one row count.
            let expected = spec.extent(&symbols)?;
            symbols.bind(
                self.graph.rows_symbol(),
                if maximum_rows == 1 { 2 } else { 1 },
            );
            if spec.extent(&symbols)? != expected {
                return Err(unsupported("row-dependent weight"));
            }
            symbols.bind(self.graph.rows_symbol(), maximum_rows as u64);
        }
        let Some(output) = self
            .graph
            .nodes()
            .iter()
            .find(|n| n.output == self.graph.output())
        else {
            return Err(invalid("graph", "missing vocabulary projection"));
        };
        let OpParams::VocabProjection { vocab, .. } = output.params else {
            return Err(unsupported("output must be vocabulary projection"));
        };
        let vocabulary = usize::try_from(vocab).map_err(|_| DimError::Overflow)?;
        if request.prompt.iter().any(|t| *t as usize >= vocabulary) {
            return Err(invalid("prompt", "token outside vocabulary"));
        }
        let layers = self.graph.attention_layers();
        if layers.is_empty() || layers.len() > 8 {
            return Err(unsupported("1..8 attention layers required"));
        }
        // One entry per attention layer, in layer order. Layers need not agree
        // with each other any more, so each one's own width and retention is
        // admitted; what they must agree with is their own graph node.
        let mut per_layer: Vec<Option<LayerKv>> = try_vec(layers.len())?;
        per_layer.resize(layers.len(), None);
        for node in self.graph.nodes() {
            match node.params {
                OpParams::Embedding { vocab: v, .. }
                    if v != vocab || node.inputs[0] != self.tokens =>
                {
                    return Err(invalid("graph", "embedding input/vocabulary mismatch"));
                }
                OpParams::Rope { .. } if node.inputs[1] != self.positions => {
                    return Err(invalid("graph", "rope must consume absolute positions"));
                }
                OpParams::Attention {
                    kv_heads,
                    head_dim,
                    layer,
                    visibility,
                    ..
                } => {
                    if node.inputs[3] != self.positions {
                        return Err(unsupported("attention requires absolute positions"));
                    }
                    // `layers` is the sorted list of layer indices the graph
                    // uses, so a gap or a duplicate is caught by position
                    // rather than assumed away.
                    let Some(slot) = layers
                        .iter()
                        .position(|l| *l == layer)
                        .and_then(|i| per_layer.get_mut(i))
                    else {
                        return Err(invalid("graph", "attention layer is not in this graph"));
                    };
                    if slot.is_some() {
                        return Err(invalid(
                            "graph",
                            "two attention nodes claim the same key/value layer",
                        ));
                    }
                    // The paged rows hold keys and values, so the row width
                    // follows the key/value heads, not the query heads. Under
                    // GQA the two differ and charging for the query width
                    // would over-reserve.
                    *slot = Some(LayerKv {
                        kv_heads: kv_heads as usize,
                        key_dim: head_dim as usize,
                        value_dim: head_dim as usize,
                        // The store's retention comes from the mask, so the two
                        // cannot drift: a layer keeps exactly what it can see.
                        retention: match visibility {
                            Visibility::Causal => Retention::All,
                            Visibility::SlidingWindow { window } => Retention::Window {
                                window: window as usize,
                            },
                        },
                    });
                }
                _ => {}
            }
        }
        let mut kv_layers = try_vec(layers.len())?;
        for slot in per_layer {
            kv_layers.push(slot.ok_or_else(|| invalid("graph", "attention layer has no node"))?);
        }
        // Workspace follows the widest layer: the interpreter holds one layer's
        // dense scratch at a time, and charging the narrowest would under-admit.
        let width = kv_layers
            .iter()
            .map(|l| mul(l.kv_heads, l.key_dim))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .expect("attention layers");
        let workspace = add(
            add(
                add(1_048_576, mul(64, elements)?)?,
                mul(4096, self.graph.value_count())?,
            )?,
            mul(mul(mul(256, context)?, add(width, 16)?)?, layers.len() + 1)?,
        )?;
        Ok(Envelope {
            vocabulary,
            workspace,
            geometry: KvGeometry {
                layers: kv_layers,
                precision: Precision::Bf16,
                page_tokens: 7,
                max_tokens: context,
                // One prefill chunk is the longest transaction this service
                // opens, so it is exactly the undo headroom a reclaiming layer
                // has to be admitted for. ADR 0014.
                tentative_rows: request.prefill_chunk.max(1),
            },
        })
    }
}

#[derive(Debug)]
struct Session<'program> {
    sequence: Option<PagedSequence>,
    execution: PagedExecution<'program>,
    prompt: HostBuffer,
    reserve: Option<Reservation>,
    output: Option<PagedOutput>,
    prompt_len: usize,
    processed: usize,
    generated: usize,
    maximum: usize,
    chunk: usize,
    temperature: f64,
    seed: u64,
    admitted: bool,
    reserved_bytes: u64,
    pending: Option<u32>,
}

impl<'program> Session<'program> {
    fn create(
        program: Program<'program>,
        request: GenerationRequest<'_>,
        ledger: &mut Ledger,
    ) -> Result<Self> {
        let envelope = program.validate(request)?;
        let baseline = ledger.scope_committed(Scope::Host);
        let mut plan = PlanRequest::new("host reference generation scratch", ["generation"])?;
        plan.buffer(BufferRequest::new(
            "bounded interpreter scratch",
            Scope::Host,
            Tier::Host(HostTier::CpuWorkspace),
            envelope.workspace as u64,
            StageSpan::at(0),
        ))?;
        let reserve = ledger.admit(&plan).map_err(Error::from)?;
        let mut prompt =
            match HostBuffer::allocate(ledger, "generation prompt", request.prompt.len() * 4, 0) {
                Ok(x) => x,
                Err(e) => {
                    ledger.release(reserve).expect("admitting ledger");
                    return Err(e);
                }
            };
        let mut sequence = match PagedSequence::with_sampling(
            ledger,
            envelope.geometry,
            envelope.vocabulary,
            request.max_new_tokens,
            request.seed,
        ) {
            Ok(x) => x,
            Err(e) => {
                prompt.release(ledger).expect("admitting ledger");
                ledger.release(reserve).expect("admitting ledger");
                return Err(e);
            }
        };
        for (target, token) in prompt.bytes_mut().chunks_exact_mut(4).zip(request.prompt) {
            target.copy_from_slice(&token.to_le_bytes());
        }
        sequence
            .append_prompt(request.prompt.len() as u64)
            .expect("validated prompt capacity");
        let execution = match PagedExecution::bind(
            program.graph,
            program.weights,
            program.tokens,
            program.positions,
            &mut sequence,
        ) {
            Ok(execution) => execution,
            Err(error) => {
                sequence
                    .close(ledger)
                    .expect("service holds the admitting ledger");
                prompt
                    .release(ledger)
                    .expect("service holds the admitting ledger");
                ledger
                    .release(reserve)
                    .expect("service holds the admitting ledger");
                return Err(error);
            }
        };
        Ok(Self {
            sequence: Some(sequence),
            execution,
            prompt,
            reserve: Some(reserve),
            output: None,
            prompt_len: request.prompt.len(),
            processed: 0,
            generated: 0,
            maximum: request.max_new_tokens,
            chunk: request.prefill_chunk,
            temperature: request.temperature,
            seed: request.seed,
            admitted: false,
            reserved_bytes: ledger.scope_committed(Scope::Host) - baseline,
            pending: None,
        })
    }

    fn usage(&self) -> Usage {
        Usage {
            prompt_tokens: self.prompt_len,
            completion_tokens: self.generated,
        }
    }

    fn forward(
        &mut self,
        tokens: &[u64],
        cancel: &Cancel,
        txn: moxie_types::StateTransactionId,
    ) -> Result<PagedOutput> {
        let sequence = self.sequence.as_mut().expect("live session");
        let start = sequence.usage().rows as u64;
        let mut positions = try_vec(tokens.len())?;
        positions.extend(start..start + tokens.len() as u64);
        self.execution
            .run(sequence, txn, tokens, &positions, cancel)
    }

    fn step(&mut self, cancel: &Cancel) -> Result<GenerationEvent> {
        if !self.admitted {
            self.admitted = true;
            return Ok(GenerationEvent::Admitted {
                profile: "host-reference",
                sampler: if self.temperature == 0.0 {
                    "greedy"
                } else {
                    "temperature"
                },
                temperature: self.temperature,
                seed: self.seed,
                context: self.prompt_len + self.maximum,
                reserved_bytes: self.reserved_bytes,
            });
        }
        if self.generated == self.maximum {
            return Ok(GenerationEvent::Finished {
                usage: self.usage(),
            });
        }
        cancel.check("generation/step")?;
        if self.processed < self.prompt_len {
            let end = (self.processed + self.chunk).min(self.prompt_len);
            let mut tokens = try_vec(end - self.processed)?;
            tokens.extend(
                self.prompt.bytes()[self.processed * 4..end * 4]
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes(b.try_into().expect("four bytes")) as u64),
            );
            self.output = None;
            self.sequence.as_mut().unwrap().clear_logits()?;
            let txn = self.sequence.as_mut().unwrap().begin()?;
            let result = (|| {
                let output = self.forward(&tokens, cancel, txn)?;
                cancel.check("prefill/before_commit")?;
                self.sequence.as_mut().unwrap().commit_prefix_cancellable(
                    txn,
                    0,
                    cancel.signal(),
                )?;
                Ok(output)
            })();
            match result {
                Ok(output) => self.output = Some(output),
                Err(e) => {
                    self.abort_if_open(txn);
                    return Err(e);
                }
            }
            self.processed = end;
            return Ok(GenerationEvent::Prefill {
                processed: end,
                total: self.prompt_len,
            });
        }
        if self.pending.is_some() {
            self.output = None;
            self.sequence.as_mut().unwrap().clear_logits()?;
        }
        let txn = self.sequence.as_mut().unwrap().begin()?;
        let result = (|| {
            if let Some(token) = self.pending {
                self.output = Some(self.forward(&[token as u64], cancel, txn)?);
            }
            let sequence = self.sequence.as_mut().unwrap();
            let token = self
                .output
                .as_ref()
                .expect("successful prefill/forward")
                .stage(sequence, txn, self.temperature, cancel)?;
            cancel.check("sample/after_stage")?;
            sequence.commit_prefix_cancellable(txn, 1, cancel.signal())?;
            // Commit is the linearization point. Cancellation after it is
            // observed on the next pull; this event reports committed work.
            sequence
                .emit(1)
                .expect("one committed token, transaction closed");
            Ok(token)
        })();
        let token = match result {
            Ok(t) => t,
            Err(e) => {
                self.abort_if_open(txn);
                return Err(e);
            }
        };
        self.pending = Some(token);
        self.generated += 1;
        Ok(GenerationEvent::Token {
            id: token,
            position: (self.prompt_len + self.generated - 1) as u64,
        })
    }

    fn abort_if_open(&mut self, txn: moxie_types::StateTransactionId) {
        let sequence = self.sequence.as_mut().unwrap();
        if sequence.validate_transaction(txn).is_ok() {
            sequence.abort(txn).expect("validated local transaction");
        }
    }
    fn close(mut self, ledger: &mut Ledger) {
        self.output = None;
        self.sequence
            .take()
            .unwrap()
            .close(ledger)
            .expect("service holds the admitting ledger");
        self.prompt
            .release(ledger)
            .expect("service holds the admitting ledger");
        ledger
            .release(self.reserve.take().unwrap())
            .expect("service holds the admitting ledger");
    }
}
