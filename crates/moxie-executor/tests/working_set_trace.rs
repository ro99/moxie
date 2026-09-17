//! The trace's product, enumerated -- and every equality made to fail on purpose.
//!
//! Task 0023's deliverable is a reconciliation, so two things have to be true of
//! its tests, and they are different things. The first is that a whole working
//! set's trace reconciles across the combinations that can arise: candidate mix,
//! cache pressure on either side, a warm cache, and a layer that fails. The
//! second is that **each named equality can fail** -- a `reconcile()` that can
//! only succeed is a stub, and this repository has twice found a check that was
//! never load-bearing because nothing ever violated it.
//!
//! The working set here is synthetic and small. Its purpose is the product; the
//! designated artifact's real 45,675,970,560 B of expert weights are the device
//! lane's, in `whole_working_set.rs`, which cannot run without a GPU.

use std::io::Write;
use std::path::{Path, PathBuf};

use moxie_executor::grouped::{ExpertRoles, GroupedRun};
use moxie_executor::residency::{ChunkSource, ShardSource};
use moxie_executor::trace::{
    LayerSnapshot, LayerTrace, StepSnapshot, StepTrace, TRACE_SCHEMA_VERSION,
};
use moxie_graph::{CombineOrder, ExpertActivation, OpParams};
use moxie_kernels::cpu_expert::{bf16_round, to_bf16_bits};
use moxie_memory::{
    ArtifactId, CapacitySnapshot, ChunkId, Ledger, ResidencyAuthority, ResidencyRequest, TurnId,
};
use moxie_plan::expert::{
    ChunkResidency, Exactness, ExpertBudget, ExpertKernels, ExpertPolicy, ResidentChunks,
    compile_experts,
};
use moxie_storage::Shard;
use moxie_types::{
    AccumulationPolicy, ActivationPrecision, DeviceCapability, DeviceUuid, Error, KernelCatalogue,
    KernelOperand, KernelSymbol, Precision, Result, RoundingProfile, Scope,
    SemanticKernelDescriptor, SemanticKernelOp, SmVersion, StrategyControl, TensorLayout,
    WeightPrecision, WorkspaceExpression,
};

const HIDDEN: u64 = 16;
const INTERMEDIATE: u64 = 8;
const EXPERTS: u64 = 6;
const TOP_K: u64 = 2;
const ROWS: u64 = 4;
/// `2 * intermediate * hidden * 2`, the gate-up chunk.
const GATE_UP: u64 = 2 * INTERMEDIATE * HIDDEN * 2;
/// `hidden * intermediate * 2`, the down chunk.
const DOWN: u64 = HIDDEN * INTERMEDIATE * 2;
const CHUNK: u64 = GATE_UP + DOWN;
const BUS: &str = "0000:82:00.0";
const LAYERS: u32 = 3;

/// Layer `l`'s route: four rows of top-k 2, rotated so no two layers demand the
/// same set. A whole working set whose every layer wanted the same experts would
/// measure one layer three times.
fn route(layer: u32) -> Vec<u32> {
    let base: [u32; 8] = [0, 1, 0, 2, 1, 3, 0, 4];
    base.iter().map(|e| (e + layer) % EXPERTS as u32).collect()
}

fn union_bytes(layer: u32) -> u64 {
    let mut distinct = route(layer);
    distinct.sort_unstable();
    distinct.dedup();
    distinct.len() as u64 * CHUNK
}

// ---------------------------------------------------------------------------
// The fixture: one shard holding every layer's experts
// ---------------------------------------------------------------------------

struct Values(u64);

impl Values {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let unit = ((self.0 >> 40) as f32) / ((1u32 << 24) as f32) - 0.5;
        bf16_round(unit * 2.0)
    }
    fn block(&mut self, len: usize) -> Vec<f32> {
        (0..len).map(|_| self.next()).collect()
    }
}

fn to_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for v in values {
        out.extend_from_slice(&to_bf16_bits(*v).to_le_bytes());
    }
    out
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("moxie-trace-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// One shard with `gate_up`/`down` tensors for every layer, written in layer
/// order.
fn write_shard(dir: &Path, seed: u64) -> (PathBuf, Vec<u8>) {
    let mut v = Values::new(seed);
    let mut payload = Vec::new();
    let mut entries = Vec::new();
    for layer in 0..LAYERS {
        for (name, len) in [
            ("gate_up", (EXPERTS * 2 * INTERMEDIATE * HIDDEN) as usize),
            ("down", (EXPERTS * HIDDEN * INTERMEDIATE) as usize),
        ] {
            let block = to_bytes(&v.block(len));
            let start = payload.len();
            payload.extend_from_slice(&block);
            let shape = if name == "gate_up" {
                format!("[{EXPERTS},{},{HIDDEN}]", 2 * INTERMEDIATE)
            } else {
                format!("[{EXPERTS},{HIDDEN},{INTERMEDIATE}]")
            };
            entries.push(format!(
                "\"layers.{layer}.{name}_proj\":{{\"dtype\":\"BF16\",\"shape\":{shape},\
                 \"data_offsets\":[{start},{}]}}",
                payload.len()
            ));
        }
    }
    let header = format!("{{{}}}", entries.join(","));
    let path = dir.join("model.safetensors");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    f.write_all(header.as_bytes()).unwrap();
    f.write_all(&payload).unwrap();
    f.flush().unwrap();
    let x = to_bytes(&v.block((ROWS * HIDDEN) as usize));
    (path, x)
}

fn artifact() -> ArtifactId {
    ArtifactId::new("whole-working-set-fixture-v1").unwrap()
}

fn roles(layer: u32) -> ExpertRoles {
    ExpertRoles {
        per_expert: None,
        artifact: artifact(),
        gate_up_role: format!("experts_gate_up.{layer}"),
        down_role: format!("experts_down.{layer}"),
        format_version: 1,
    }
}

fn source(path: &Path) -> ShardSource {
    let mut src = ShardSource::new(artifact(), vec![Shard::open(path).unwrap()]);
    for layer in 0..LAYERS {
        src = src
            .role(
                format!("experts_gate_up.{layer}"),
                0,
                format!("layers.{layer}.gate_up_proj"),
            )
            .unwrap()
            .role(
                format!("experts_down.{layer}"),
                0,
                format!("layers.{layer}.down_proj"),
            )
            .unwrap();
    }
    src
}

/// A source that fails the `n`th read it is asked for.
struct FailingSource {
    inner: ShardSource,
    fail_at: Option<u32>,
    seen: u32,
}

impl ChunkSource for FailingSource {
    fn read_chunk(&mut self, chunk: &ChunkId, into: &mut [u8]) -> Result<()> {
        let seen = self.seen;
        self.seen += 1;
        if self.fail_at == Some(seen) {
            return Err(Error::InvalidArtifact {
                detail: format!("injected read failure on {chunk}").into(),
            });
        }
        self.inner.read_chunk(chunk, into)
    }
}

fn uuid() -> DeviceUuid {
    DeviceUuid::parse("GPU-3032cfa3-19df-028f-5ebd-43314911e0b9").unwrap()
}

fn mlp() -> OpParams {
    OpParams::ExpertMlp {
        hidden: HIDDEN,
        intermediate: INTERMEDIATE,
        experts: EXPERTS,
        top_k: TOP_K,
        activation: ExpertActivation::GeGlu,
    }
}

fn combine() -> OpParams {
    OpParams::Combine {
        hidden: HIDDEN,
        top_k: TOP_K,
        order: CombineOrder::AscendingExpertId,
        output_scale: 1.0,
    }
}

fn capability() -> DeviceCapability {
    DeviceCapability {
        uuid: uuid(),
        ordinal: 1,
        name: "RTX 3090".into(),
        compute_major: 8,
        compute_minor: 6,
        total_memory_bytes: 24 << 30,
        multiprocessor_count: 82,
        pci_bus_id: BUS.into(),
        peer_access: Vec::new(),
    }
}

fn catalogue() -> KernelCatalogue {
    KernelCatalogue::new(vec![SemanticKernelDescriptor {
        id: moxie_types::KernelId("trace-expert".into()),
        abi_version: moxie_plan::expert::EXPERT_ABI_VERSION,
        operation: SemanticKernelOp::ExpertMlp(moxie_types::GateTransform::GeluTanh),
        inputs: vec![
            KernelOperand::Activation(ActivationPrecision::expect(Precision::Bf16)),
            KernelOperand::RouteIndex,
            KernelOperand::Weight(WeightPrecision::expect(Precision::Bf16)),
            KernelOperand::Weight(WeightPrecision::expect(Precision::Bf16)),
        ],
        output: ActivationPrecision::expect(Precision::Bf16),
        accumulation: AccumulationPolicy::Bf16InF32Acc,
        rounding: RoundingProfile::FinalBf16Rne,
        layout: TensorLayout::ContiguousRowMajorV1,
        shape: moxie_types::KernelShapeBounds {
            max_rows: 1024,
            max_input: 1024,
            max_output: 1024,
        },
        sm: SmVersion::SM86,
        workspace: WorkspaceExpression::RowsTimesIntermediateF32,
        image_sha256: [9; 32],
        symbols: vec![KernelSymbol("project".into()), KernelSymbol("down".into())],
    }])
    .expect("one descriptor")
}

/// Kernel launches one device group submits: one per symbol of the selected
/// descriptor, which for the expert kernel is a projection and a reduction.
const KERNELS_PER_GROUP: u32 = 2;

/// A device lane that performs uploads and computes nothing.
///
/// The trace is about bytes and counts, and the numerical answer is task 0021's
/// gate on real hardware. What this stands in for is the *lifecycle*: an upload
/// really is completed against the authority, so every byte identity this sweep
/// checks is checked on the device path too, on a machine with no GPU.
#[derive(Debug, Default)]
struct Lane;

impl moxie_executor::grouped::ExpertDeviceLane for Lane {
    fn load_activations(
        &mut self,
        _x: &[u8],
    ) -> std::result::Result<(), moxie_executor::grouped::LaunchRefused> {
        Ok(())
    }

    fn perform_upload(
        &mut self,
        authority: &mut ResidencyAuthority,
        order: &moxie_memory::WorkOrder,
    ) -> Result<()> {
        authority.complete_upload(order.ticket(), moxie_memory::Outcome::Completed)
    }

    fn run_group(
        &mut self,
        _authority: &ResidencyAuthority,
        _group: &moxie_plan::expert::ExpertGroup,
        _gate_up: &moxie_memory::ResidencyLease,
        _down: &moxie_memory::ResidencyLease,
        _staging: moxie_executor::grouped::ExpertStaging<'_>,
        _host_slots: &mut [u8],
    ) -> std::result::Result<u32, moxie_executor::grouped::LaunchRefused> {
        // What the production lane submits: one launch per symbol of the
        // selected descriptor. A double reporting anything else would make the
        // launch equality mean something different here than on a card.
        Ok(KERNELS_PER_GROUP)
    }

    fn close(&mut self, _ledger: &mut Ledger) -> Result<()> {
        Ok(())
    }

    fn is_quarantined(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// The product
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Placement {
    Host,
    Device,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Room {
    /// The cache holds a whole layer's union: nothing this layer admits can be
    /// evicted before the layer is done with it, so the prediction is exact.
    Roomy,
    /// A quarter of it, which is where a backpressure retry can re-admit a
    /// chunk and the prediction becomes a declared lower bound.
    Tight,
}

impl Room {
    fn bytes(self, union: u64) -> u64 {
        // Whole cache ranges: the authority's arena is aligned, and a cap that
        // is not is a refusal rather than a small cache.
        let align = |bytes: u64| bytes.div_ceil(256) * 256;
        match self {
            // Every layer of the step, **plus the alignment each chunk can
            // waste**. Sizing it to the step's bytes exactly leaves the last
            // layer no room for padding, and a prediction that cannot promise
            // the padding fits is a lower bound -- which would make "roomy" a
            // configuration that never reaches the exact branch.
            Room::Roomy => align(u64::from(LAYERS) * union + u64::from(LAYERS) * 8 * 2 * 256),
            Room::Tight => align(union.div_ceil(4).max(CHUNK)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Warm {
    /// Every layer runs once.
    Cold,
    /// Each layer runs once untraced first, and the traced run is given the
    /// resident set the authority actually has -- read, not assumed.
    Repeat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Case {
    placement: Placement,
    device_room: Room,
    host_room: Room,
    warm: Warm,
    layers: u32,
    fail_read_at: Option<u32>,
}

/// What one layer's traced run produced.
struct LayerOutcome {
    /// Every layer execution this call performed: the warm pass, when the case
    /// asked for one, and then the traced run. Both are traced, because the
    /// step's outer boundary counts the bytes of both.
    traces: Vec<LayerTrace>,
    /// The traced run's own record, for the assertions that are about it.
    trace: LayerTrace,
    failed: bool,
}

#[allow(clippy::too_many_arguments)]
fn run_layer(
    case: Case,
    layer: u32,
    path: &Path,
    x: &[u8],
    ledger: &mut Ledger,
    authority: &mut ResidencyAuthority,
    turn: u64,
) -> LayerOutcome {
    let union = union_bytes(layer);
    let device = case.placement == Placement::Device;
    let policy = ExpertPolicy {
        device: if device {
            StrategyControl::Required
        } else {
            StrategyControl::Off
        },
        host: if device {
            StrategyControl::Off
        } else {
            StrategyControl::Auto
        },
        host_placement: StrategyControl::Off,
        max_transfer_bytes_per_row: CHUNK,
        ..ExpertPolicy::default()
    };
    let cat = catalogue();
    let cap = capability();
    let mut traces: Vec<LayerTrace> = Vec::new();

    // A warm pass, if this case asked for one. It is an ordinary run through the
    // ordinary path; nothing is placed by hand.
    if case.warm == Warm::Repeat {
        let plan = compile_experts(
            &mlp(),
            &combine(),
            &route(layer),
            &budget(case, layer, union, ResidentChunks::none(), 0, 0),
            &policy,
            None,
            device.then_some(ExpertKernels {
                capability: &cap,
                catalogue: &cat,
            }),
        )
        .unwrap();
        let snapshot = LayerSnapshot::take(layer, authority).unwrap();
        let mut run = GroupedRun::admit(ledger, plan, roles(layer), None).unwrap();
        if device {
            run.install_lane(Box::new(Lane)).unwrap();
        }
        run.load_activations(x).unwrap();
        let mut src = FailingSource {
            inner: source(path),
            fail_at: None,
            seen: 0,
        };
        run.run_to_completion(authority, &mut src, TurnId::new(turn), 0, u64::MAX)
            .unwrap();
        traces.push(snapshot.close(&run, authority, ledger).unwrap());
        run.close(ledger).unwrap();
        authority.check_invariants().unwrap();
    }

    // The snapshot the planner is compiled against: what the authority actually
    // holds, not what a warm pass was expected to leave behind.
    let scope = if device {
        Scope::Device(uuid())
    } else {
        Scope::Host
    };
    let resident = resident_chunks(authority, device.then_some(scope), layer);

    let plan = compile_experts(
        &mlp(),
        &combine(),
        &route(layer),
        &budget(
            case,
            layer,
            union,
            resident,
            if device {
                largest_free_of(authority, scope)
            } else {
                0
            },
            largest_free_of(authority, Scope::Host),
        ),
        &policy,
        None,
        device.then_some(ExpertKernels {
            capability: &cap,
            catalogue: &cat,
        }),
    )
    .unwrap();

    let snapshot = LayerSnapshot::take(layer, authority).unwrap();
    let mut run = GroupedRun::admit(ledger, plan, roles(layer), None).unwrap();
    if device {
        run.install_lane(Box::new(Lane)).unwrap();
    }
    run.load_activations(x).unwrap();
    let mut src = FailingSource {
        inner: source(path),
        fail_at: case.fail_read_at.filter(|_| layer == 1),
        seen: 0,
    };
    let outcome = run.run_to_completion(authority, &mut src, TurnId::new(turn), 0, u64::MAX);
    let failed = outcome.is_err();
    run.check_invariants().unwrap();
    authority.check_invariants().unwrap();
    let trace = snapshot.close(&run, authority, ledger).unwrap();
    if failed {
        run.cancel(authority);
    }
    let _ = run.close(ledger);
    authority.check_invariants().unwrap();
    traces.push(trace.clone());
    LayerOutcome {
        traces,
        trace,
        failed,
    }
}

/// The largest contiguous free range in a scope's cache.
///
/// What admission actually needs, and what two rounds of review showed a
/// prediction cannot do without: eviction does not reason about one plan's share
/// of a cache, and it does not reason about totals either -- a cache with ample
/// free bytes in pieces too small for a chunk evicts to admit one.
fn largest_free_of(authority: &ResidencyAuthority, scope: Scope) -> u64 {
    authority
        .occupancy(scope)
        .map_or(0, |o| o.largest_free_bytes)
}

/// Where this layer's expert chunks are, read from the authority **per chunk**.
///
/// A cache really can hold one of an expert's two chunks -- the authority admits
/// and evicts them separately -- and the device can hold one while the host
/// holds the other. A snapshot that rounded that to "resident" or "absent", or
/// even to a byte count per expert, would hand the planner a number that cannot
/// answer whether an upload has a host source to copy from.
fn resident_chunks(
    authority: &ResidencyAuthority,
    device_scope: Option<Scope>,
    layer: u32,
) -> ResidentChunks {
    let roles = [
        format!("experts_gate_up.{layer}"),
        format!("experts_down.{layer}"),
    ];
    let mut out = ResidentChunks::none();
    let mut seen: Vec<(u32, u64)> = Vec::new();
    for chunk in authority.outstanding() {
        if !chunk.state.is_ready() {
            continue;
        }
        let Some(expert) = chunk.chunk.slot().expert_index() else {
            continue;
        };
        if !roles.iter().any(|r| r == chunk.chunk.slot().role()) {
            continue;
        }
        let on_device = Some(chunk.scope) == device_scope;
        let in_host = chunk.scope == Scope::Host;
        // One entry per (expert, chunk) pair, merging the two scopes' answers
        // about the same chunk: the same bytes can be in both caches at once,
        // and pushing them twice would say an expert has twice the bytes it has.
        let key = (expert, chunk.bytes);
        if let Some(index) = seen.iter().position(|k| *k == key) {
            let entry = &mut out.chunks_mut()[index];
            entry.on_device |= on_device;
            entry.in_host |= in_host;
        } else {
            seen.push(key);
            out.push(ChunkResidency {
                expert,
                bytes: chunk.bytes,
                on_device,
                in_host,
            });
        }
    }
    out
}

fn budget(
    case: Case,
    layer: u32,
    union: u64,
    resident: ResidentChunks,
    device_free_bytes: u64,
    host_free_bytes: u64,
) -> ExpertBudget {
    let device = case.placement == Placement::Device;
    let _ = layer;
    ExpertBudget {
        device: uuid(),
        device_pci_bus_id: BUS.into(),
        device_cache_cap_bytes: if device {
            case.device_room.bytes(union)
        } else {
            0
        },
        device_cache_leased_bytes: 0,
        device_cache_largest_free_bytes: device_free_bytes,
        device_arena_free_bytes: if device { 1 << 20 } else { 0 },
        host_workspace_bytes: 1 << 20,
        host_buffer_bytes: 1 << 20,
        host_cache_cap_bytes: case.host_room.bytes(union),
        host_cache_leased_bytes: 0,
        host_cache_largest_free_bytes: host_free_bytes,
        cache_alignment_bytes: 256,
        chunks_per_expert: 2,
        resident,
    }
}

fn ledger_for(case: Case) -> Ledger {
    let mut scopes = vec![CapacitySnapshot::new(Scope::Host, 1 << 26, 1 << 20).unwrap()];
    if case.placement == Placement::Device {
        scopes.push(CapacitySnapshot::new(Scope::Device(uuid()), 1 << 26, 1 << 20).unwrap());
    }
    Ledger::new(scopes).unwrap()
}

/// One case: run every layer, assemble the trace, reconcile it.
fn run_case(case: Case, path: &Path, x: &[u8]) -> (StepTrace, u32) {
    let mut ledger = ledger_for(case);
    let union = union_bytes(0);
    let mut request = ResidencyRequest::new("whole working set", case.host_room.bytes(union));
    if case.placement == Placement::Device {
        request = request.device(uuid(), case.device_room.bytes(union));
    }
    let mut authority = ResidencyAuthority::open(&mut ledger, &request).unwrap();
    // Before the first layer, and before any warm pass: every byte that enters a
    // cache from here on has to belong to a layer of this trace.
    let start = StepSnapshot::take(&authority).unwrap();

    let mut layers = Vec::new();
    let mut failures = 0;
    for layer in 0..case.layers {
        let outcome = run_layer(
            case,
            layer,
            path,
            x,
            &mut ledger,
            &mut authority,
            u64::from(layer) + 1,
        );
        if outcome.failed {
            failures += 1;
        }
        // A failed layer's trace is **kept**. Its bytes were charged, moved and
        // given back like any other layer's, and a step that dropped them would
        // reconcile a subset of itself. A warm pass is kept for the same reason:
        // it is a layer execution, its bytes entered the caches, and the step's
        // outer boundary counts them whether this test does or not.
        layers.extend(outcome.traces);
        authority.end_turn(TurnId::new(u64::from(layer) + 1));
    }

    for scope in [Scope::Host, Scope::Device(uuid())] {
        authority.retire_all(scope);
    }
    authority.check_invariants().unwrap();
    // The authority gives its own cache envelope back **before** the trace is
    // assembled: `nothing-outstanding` asks whether anything is still charged
    // when the step is over, and a step that is still holding its cache is not
    // over.
    authority.close(&mut ledger).unwrap();
    let trace = StepTrace::new(
        artifact().as_str().to_string(),
        format!("{case:?}"),
        layers,
        &start,
        &authority,
        &ledger,
    )
    .unwrap();
    (trace, failures)
}

/// The sweep: every combination, with the invariants checked after every
/// operation and the product printed rather than described.
#[test]
fn every_trace_combination_reconciles() {
    let dir = scratch("sweep");
    let (path, x) = write_shard(&dir, 0x0023_2026);

    let mut cases = 0;
    let mut layers_traced = 0;
    let mut equalities = 0;
    let mut bounds = 0;
    let mut failed_layers = 0;
    let mut exact_layers = 0;
    let mut bounded_layers = 0;
    let mut with_device_hits = 0;
    let mut with_source_reuse = 0;
    let mut with_evictions = 0;
    let mut incomplete = 0;
    let mut skipped = 0;

    for placement in [Placement::Host, Placement::Device] {
        for device_room in [Room::Roomy, Room::Tight] {
            for host_room in [Room::Roomy, Room::Tight] {
                for warm in [Warm::Cold, Warm::Repeat] {
                    for layers in [1, LAYERS] {
                        for fail_read_at in [None, Some(0)] {
                            let case = Case {
                                placement,
                                device_room,
                                host_room,
                                warm,
                                layers,
                                fail_read_at,
                            };
                            cases += 1;
                            let (trace, failures) = run_case(case, &path, &x);
                            failed_layers += failures;
                            layers_traced += trace.layers.len();
                            for layer in &trace.layers {
                                if !layer.completed {
                                    incomplete += 1;
                                }
                                match layer.predicted.exactness {
                                    Exactness::Exact => exact_layers += 1,
                                    Exactness::LowerBound { .. } => bounded_layers += 1,
                                }
                                let device: u64 = layer
                                    .scopes
                                    .iter()
                                    .filter(|s| matches!(s.scope, Scope::Device(_)))
                                    .map(|s| s.flow.hit_bytes)
                                    .sum();
                                if device > 0 {
                                    with_device_hits += 1;
                                }
                                let reuse: u64 =
                                    layer.scopes.iter().map(|s| s.flow.source_reuse_bytes).sum();
                                if reuse > 0 {
                                    with_source_reuse += 1;
                                }
                                let evicted: u64 =
                                    layer.scopes.iter().map(|s| s.flow.evicted_bytes).sum();
                                if evicted > 0 {
                                    with_evictions += 1;
                                }
                            }
                            // The high-water mark is a mark: it may not follow
                            // the level down. Found by mutation -- setting it to
                            // the current level at every admission survived
                            // everything, because nothing ever compared two
                            // readings of it.
                            for (index, layer) in trace.layers.iter().enumerate().skip(1) {
                                for delta in &layer.scopes {
                                    let before = trace.layers[index - 1]
                                        .scopes
                                        .iter()
                                        .find(|s| s.scope == delta.scope)
                                        .map_or(0, |s| s.peak_resident_bytes);
                                    assert!(
                                        delta.peak_resident_bytes >= before,
                                        "{case:?}: {} peak fell from {before} to {} at layer \
                                         {index}",
                                        delta.scope,
                                        delta.peak_resident_bytes
                                    );
                                }
                            }
                            let checked = trace
                                .reconcile()
                                .unwrap_or_else(|e| panic!("{case:?} did not reconcile: {e}"));
                            equalities += checked.equalities_checked;
                            bounds += checked.bounds_checked;
                            skipped += checked.predictions_skipped;
                        }
                    }
                }
            }
        }
    }

    println!(
        "whole-working-set trace sweep: {cases} case(s), {layers_traced} layer(s) traced, \
         {equalities} equality check(s) of which {bounds} were the declared lower bound"
    );
    println!(
        "  layers by prediction: {exact_layers} exact, {bounded_layers} lower-bound; \
         {failed_layers} layer(s) failed, {incomplete} of them traced as incomplete and \
         {skipped} prediction(s) skipped for it"
    );
    println!(
        "  layers exercising: {with_device_hits} with device hits, {with_source_reuse} with host \
         source reuse, {with_evictions} with evictions"
    );

    assert_eq!(cases, 2 * 2 * 2 * 2 * 2 * 2);
    // Every axis has to *do* something. An axis that is exercised but never
    // reaches the behaviour it names is an axis that checks nothing -- task
    // 0021's fourth review found exactly that, in a sweep whose 144 combinations
    // all passed with the branch it was aimed at deleted.
    assert!(exact_layers > 0, "no layer produced an exact prediction");
    assert!(bounded_layers > 0, "no layer produced a lower bound");
    assert!(with_device_hits > 0, "no layer ever hit on the device");
    assert!(
        with_source_reuse > 0,
        "no device upload ever reused a host source"
    );
    assert!(with_evictions > 0, "nothing was ever evicted");
    assert!(failed_layers > 0, "no layer ever failed");
    assert_eq!(
        incomplete, failed_layers,
        "a failed layer did not reach the trace as an incomplete one"
    );
    assert_eq!(
        skipped, incomplete,
        "an incomplete layer's prediction was compared anyway"
    );
    assert!(
        bounds > 0 && bounds < equalities,
        "one form of check never ran"
    );
}

// ---------------------------------------------------------------------------
// Every equality, made to fail
// ---------------------------------------------------------------------------

/// One trace that reconciles, from a case that exercises hits, reuse and an
/// exact prediction.
fn a_reconciling_trace(dir: &Path) -> StepTrace {
    let (path, x) = write_shard(dir, 0x0023_1234);
    let case = Case {
        placement: Placement::Device,
        device_room: Room::Roomy,
        host_room: Room::Roomy,
        warm: Warm::Repeat,
        layers: LAYERS,
        fail_read_at: None,
    };
    let (trace, failures) = run_case(case, &path, &x);
    assert_eq!(failures, 0);
    trace.reconcile().expect("the unmutated trace reconciles");
    trace
}

/// Each named equality, violated on purpose, must be the one that is reported.
///
/// This is the battery, not a formality. A `reconcile()` that cannot fail is a
/// stub, and a check nothing ever violates is indistinguishable from one that
/// was never written -- AGENTS.md records three separate occasions where a
/// regression asserted the symptom rather than the check and would have passed
/// with the check deleted.
#[test]
fn every_equality_can_fail_and_names_itself() {
    let dir = scratch("violations");
    let good = a_reconciling_trace(&dir);

    /// One mutation aimed at one named equality.
    type Mutation = (&'static str, Box<dyn Fn(&mut StepTrace)>);

    let mutations: Vec<Mutation> = vec![
        (
            "schema-version-is-this-one",
            Box::new(|t: &mut StepTrace| t.schema_version += 1),
        ),
        (
            "ledger-is-the-runs-own",
            Box::new(|t: &mut StepTrace| t.layers[0].ledger_id += 1),
        ),
        (
            "every-reservation-is-charged",
            Box::new(|t: &mut StepTrace| t.layers[0].reservations_named += 1),
        ),
        (
            "step-covers-every-byte",
            Box::new(|t: &mut StepTrace| t.whole[0].1.read_bytes += 1),
        ),
        (
            "ledger-charges-what-is-held",
            Box::new(|t: &mut StepTrace| t.layers[0].ledger[0].2 += 64),
        ),
        (
            "ledger-charges-what-is-held",
            Box::new(|t: &mut StepTrace| {
                let tier = t.layers[0].ledger[0].1;
                t.layers[0].ledger.push((Scope::Host, tier, 4096));
            }),
        ),
        (
            "cache-cap-is-the-reservation",
            Box::new(|t: &mut StepTrace| {
                t.layers[0].scopes[0].resident_bytes = t.layers[0].scopes[0].cap_bytes + 1;
            }),
        ),
        (
            "requests-are-attempts",
            Box::new(|t: &mut StepTrace| t.layers[0].cost.acquires_issued[0] += 1),
        ),
        (
            "requests-are-attempts",
            Box::new(|t: &mut StepTrace| t.layers[0].cost.acquires_issued[1] += 1),
        ),
        (
            "run-ran-the-plan",
            Box::new(|t: &mut StepTrace| t.layers[0].cost.groups_run += 1),
        ),
        (
            "run-ran-the-plan",
            Box::new(|t: &mut StepTrace| t.layers[0].cost.host_groups += 1),
        ),
        (
            "launches-match-device-groups",
            Box::new(|t: &mut StepTrace| t.layers[0].cost.launches += 1),
        ),
        (
            "slots-are-written-once",
            Box::new(|t: &mut StepTrace| t.layers[0].cost.slots_written += 1),
        ),
        (
            "leases-balance",
            Box::new(|t: &mut StepTrace| t.layers[0].cost.leases_released += 1),
        ),
        (
            "predicted-uploads-are-uploaded",
            Box::new(|t: &mut StepTrace| t.layers[0].predicted.device_upload_bytes += 1),
        ),
        (
            "predicted-reads-are-read",
            Box::new(|t: &mut StepTrace| t.layers[0].predicted.host_read_bytes += 1),
        ),
        (
            "predicted-hits-are-hit",
            Box::new(|t: &mut StepTrace| t.layers[0].predicted.device_hit_bytes += 1),
        ),
        (
            "predicted-source-reuse-is-reused",
            Box::new(|t: &mut StepTrace| t.layers[0].predicted.host_source_reuse_bytes += 1),
        ),
        (
            "step-is-the-sum-of-layers",
            Box::new(|t: &mut StepTrace| t.totals[0].1.admitted_bytes += 1),
        ),
        (
            "nothing-outstanding",
            Box::new(|t: &mut StepTrace| t.outstanding_reservations += 1),
        ),
        (
            "nothing-outstanding",
            Box::new(|t: &mut StepTrace| t.final_accounts[0].resident_bytes += 1),
        ),
        (
            "nothing-outstanding",
            Box::new(|t: &mut StepTrace| t.final_accounts[0].flow.retired_bytes += 1),
        ),
    ];

    let mut named: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for (expected, mutate) in &mutations {
        let mut trace = good.clone();
        mutate(&mut trace);
        let failure = trace
            .reconcile()
            .expect_err(&format!("the mutation aimed at {expected} was not caught"));
        assert_eq!(
            failure.check, *expected,
            "a mutation aimed at {expected} was reported as {}: {failure}",
            failure.check
        );
        *named.entry(*expected).or_default() += 1;
    }

    println!(
        "violation battery: {} mutation(s), {} distinct equalities, all reported by name",
        mutations.len(),
        named.len()
    );
    for (check, count) in &named {
        println!("  {check}: {count}");
    }
    // Every equality `reconcile` can report must appear here. A new check with
    // no violating fixture is a check nobody has shown to be load-bearing.
    assert_eq!(named.len(), 17);

    // The *bound* form of the prediction checks is a separate branch and needs
    // its own violations: a lower bound fails when the outcome is below it, not
    // when it differs from it, and a battery that only ever mutated an exact
    // layer would leave that branch unexercised.
    let tight = {
        let (path, x) = write_shard(&dir, 0x0023_5678);
        let case = Case {
            placement: Placement::Device,
            device_room: Room::Tight,
            host_room: Room::Tight,
            warm: Warm::Repeat,
            layers: LAYERS,
            fail_read_at: None,
        };
        let (trace, _) = run_case(case, &path, &x);
        trace
            .reconcile()
            .expect("the unmutated tight trace reconciles");
        trace
    };
    let bounded = tight
        .layers
        .iter()
        .position(|l| !matches!(l.predicted.exactness, Exactness::Exact))
        .expect("the tight case has a lower-bound layer");
    for (expected, mutate) in [
        (
            "predicted-uploads-are-uploaded",
            Box::new(|t: &mut StepTrace, i: usize| {
                t.layers[i].predicted.device_upload_bytes = u64::MAX;
            }) as Box<dyn Fn(&mut StepTrace, usize)>,
        ),
        (
            "predicted-reads-are-read",
            Box::new(|t: &mut StepTrace, i: usize| {
                t.layers[i].predicted.host_read_bytes = u64::MAX;
            }),
        ),
        (
            "predicted-hits-are-hit",
            Box::new(|t: &mut StepTrace, i: usize| {
                t.layers[i].predicted.device_hit_bytes = u64::MAX;
            }),
        ),
    ] {
        let mut trace = tight.clone();
        mutate(&mut trace, bounded);
        let failure = trace
            .reconcile()
            .expect_err(&format!("the bound aimed at {expected} was not caught"));
        assert_eq!(
            failure.check, expected,
            "reported as {}: {failure}",
            failure.check
        );
    }
    println!("  and 3 more against the declared-lower-bound branch");

    // A layer that did **not** finish still reconciles, and its counts are
    // bounds rather than equalities. That branch needs its own violations: a
    // failed layer that ran *more* groups than its plan had is still a defect,
    // and so is one whose leases neither came back nor are withheld.
    let interrupted = {
        let (path, x) = write_shard(&dir, 0x0023_9999);
        let case = Case {
            placement: Placement::Host,
            device_room: Room::Roomy,
            host_room: Room::Roomy,
            warm: Warm::Cold,
            layers: LAYERS,
            fail_read_at: Some(0),
        };
        let (trace, failures) = run_case(case, &path, &x);
        assert!(failures > 0, "the interrupted case did not fail a layer");
        trace
            .reconcile()
            .expect("an interrupted step still reconciles");
        trace
    };
    let failed = interrupted
        .layers
        .iter()
        .position(|l| !l.completed)
        .expect("one layer did not finish");
    for (expected, mutate) in [
        (
            "run-ran-the-plan",
            Box::new(|t: &mut StepTrace, i: usize| {
                t.layers[i].cost.groups_run = t.layers[i].groups + 1;
            }) as Box<dyn Fn(&mut StepTrace, usize)>,
        ),
        (
            "slots-are-written-once",
            Box::new(|t: &mut StepTrace, i: usize| {
                t.layers[i].cost.slots_written = t.layers[i].rows * t.layers[i].top_k + 1;
            }),
        ),
        (
            "leases-balance",
            Box::new(|t: &mut StepTrace, i: usize| t.layers[i].withheld_leases += 1),
        ),
    ] {
        let mut trace = interrupted.clone();
        mutate(&mut trace, failed);
        let failure = trace.reconcile().expect_err(&format!(
            "the interrupted-layer mutation for {expected} was not caught"
        ));
        assert_eq!(
            failure.check, expected,
            "reported as {}: {failure}",
            failure.check
        );
    }
    println!("  and 3 more against a layer that did not finish");
}

/// The schema is a number this repository pins, not a value that drifts.
#[test]
fn the_trace_schema_is_pinned_and_says_what_it_does_not_measure() {
    assert_eq!(TRACE_SCHEMA_VERSION, 1);
    // Document 07's result `counters` block has fields this trace does not
    // measure. They are named as unmeasured rather than reported as zero.
    for field in ["d2h_bytes", "p2p_bytes", "sync_wait_ms", "collective_bytes"] {
        assert!(
            moxie_executor::trace::UNMEASURED.contains(&field),
            "{field} is neither measured nor declared unmeasured"
        );
    }
    println!(
        "trace schema v{TRACE_SCHEMA_VERSION}; {} counter(s) declared unmeasured: {:?}",
        moxie_executor::trace::UNMEASURED.len(),
        moxie_executor::trace::UNMEASURED
    );
}

/// A charge nobody in the step can name must be found, and named as that.
///
/// Found by mutation: deleting the filter that keeps only the authority's and
/// the run's own reservations survived everything, because in every case the
/// only reservations the ledger held **were** those. A fixture on which two
/// behaviours agree tests neither, and "no third charger" was the property with
/// no fixture at all.
#[test]
fn a_charge_nobody_can_name_is_reported() {
    use moxie_memory::{BufferRequest, PlanRequest, StageSpan};
    use moxie_types::{HostTier, Tier};

    let dir = scratch("third-charger");
    let (path, x) = write_shard(&dir, 0x0023_4321);
    let case = Case {
        placement: Placement::Host,
        device_room: Room::Roomy,
        host_room: Room::Roomy,
        warm: Warm::Cold,
        layers: 1,
        fail_read_at: None,
    };

    let mut ledger = ledger_for(case);
    let union = union_bytes(0);
    let mut authority = ResidencyAuthority::open(
        &mut ledger,
        &ResidencyRequest::new("third charger", case.host_room.bytes(union)),
    )
    .unwrap();
    let start = StepSnapshot::take(&authority).unwrap();

    // Somebody else's reservation, live for the whole layer. It is charged to a
    // tier this step also uses, so it cannot be spotted by its tier alone.
    let mut request = PlanRequest::new("an unrelated consumer", ["live"]).unwrap();
    request
        .buffer(BufferRequest::new(
            "not this step's",
            Scope::Host,
            Tier::Host(HostTier::Pageable),
            1 << 16,
            StageSpan::at(0),
        ))
        .unwrap();
    let stranger = ledger.admit(&request).unwrap();

    let outcome = run_layer(case, 0, &path, &x, &mut ledger, &mut authority, 1);
    assert!(!outcome.failed);
    let failure = StepTrace::new(
        artifact().as_str().to_string(),
        "third charger".to_string(),
        outcome.traces,
        &start,
        &authority,
        &ledger,
    )
    .unwrap()
    .reconcile()
    .expect_err("a charge nobody can name must not reconcile");
    assert_eq!(failure.check, "ledger-charges-what-is-held", "{failure}");
    println!("third charger reported as: {failure}");

    ledger.release(stranger).unwrap();
    authority.end_turn(TurnId::new(1));
    authority.retire_all(Scope::Host);
    authority.close(&mut ledger).unwrap();
}

/// A warm chunk this layer needs, older than a cached chunk it does not.
///
/// The review's counterexample, kept. Experts 4 and 5 are warmed; the traced
/// layer demands 0–4 through a cache that holds five experts. Expert 4 is the
/// least recently used of everything resident, so this layer's own admissions
/// evict exactly the chunk it predicted a hit on, and it reads it again —
/// **3,840 B read against 3,072 B predicted**, when the prediction called itself
/// exact.
///
/// The rule is now "nothing has to be evicted", which this configuration fails,
/// so the prediction is a declared lower bound and the run reconciles against
/// it. What makes this a regression rather than a one-off is the assertion that
/// the planner does **not** call it exact: an exactness rule that drifted back
/// to reasoning about one plan's share of the cache would fail here by name.
#[test]
fn a_warm_chunk_older_than_an_unrelated_one_is_not_an_exact_prediction() {
    let dir = scratch("lru-victim");
    let (path, x) = write_shard(&dir, 0x0023_0023);
    let case = Case {
        placement: Placement::Host,
        device_room: Room::Roomy,
        host_room: Room::Roomy,
        warm: Warm::Cold,
        layers: 1,
        fail_read_at: None,
    };
    let mut ledger = ledger_for(case);
    let union = union_bytes(0);
    // Exactly the five experts layer 0 demands, and not one byte more.
    let mut authority =
        ResidencyAuthority::open(&mut ledger, &ResidencyRequest::new("lru", union)).unwrap();
    let start = StepSnapshot::take(&authority).unwrap();

    // Warm experts 4 and 5. Expert 4 is in layer 0's route; expert 5 is not, and
    // it is touched later, so it is the *newer* of the two.
    let policy = ExpertPolicy {
        device: StrategyControl::Off,
        host: StrategyControl::Auto,
        host_placement: StrategyControl::Off,
        ..ExpertPolicy::default()
    };
    let warm_route = [4, 5, 4, 5, 4, 5, 4, 5];
    let warm_plan = compile_experts(
        &mlp(),
        &combine(),
        &warm_route,
        &budget(case, 0, union, ResidentChunks::none(), 0, 0),
        &policy,
        None,
        None,
    )
    .unwrap();
    // The warm pass is a layer execution and its bytes enter the cache, so it is
    // traced. `step-covers-every-byte` catches it if it is not -- it caught this
    // very test the first time it was written.
    let warm_snapshot = LayerSnapshot::take(0, &authority).unwrap();
    let mut warm = GroupedRun::admit(&mut ledger, warm_plan, roles(0), None).unwrap();
    warm.load_activations(&x).unwrap();
    let mut src = FailingSource {
        inner: source(&path),
        fail_at: None,
        seen: 0,
    };
    warm.run_to_completion(&mut authority, &mut src, TurnId::new(90), 0, u64::MAX)
        .unwrap();
    let warm_trace = warm_snapshot.close(&warm, &authority, &ledger).unwrap();
    warm.close(&mut ledger).unwrap();
    authority.end_turn(TurnId::new(90));

    let outcome = run_layer(case, 0, &path, &x, &mut ledger, &mut authority, 91);
    assert!(!outcome.failed);
    assert!(
        !matches!(outcome.trace.predicted.exactness, Exactness::Exact),
        "a cache that must evict to admit this layer cannot promise an exact \
         prediction: {:?}",
        outcome.trace.predicted.exactness
    );
    let host = outcome
        .trace
        .scopes
        .iter()
        .find(|s| s.scope == Scope::Host)
        .expect("a host scope");
    assert!(
        host.flow.read_bytes > outcome.trace.predicted.host_read_bytes,
        "this configuration is supposed to re-read an evicted warm chunk; it read \
         {} B against {} B predicted",
        host.flow.read_bytes,
        outcome.trace.predicted.host_read_bytes
    );
    println!(
        "warm chunk evicted by its own layer: {} B read against {} B predicted as a lower bound",
        host.flow.read_bytes, outcome.trace.predicted.host_read_bytes
    );

    authority.end_turn(TurnId::new(91));
    authority.retire_all(Scope::Host);
    authority.close(&mut ledger).unwrap();
    let mut layers = vec![warm_trace];
    layers.extend(outcome.traces);
    let trace = StepTrace::new(
        artifact().as_str().to_string(),
        "lru victim".to_string(),
        layers,
        &start,
        &authority,
        &ledger,
    )
    .unwrap();
    trace
        .reconcile()
        .expect("a lower-bound prediction reconciles against what happened");
}

/// A layer that ran and is missing from the trace must be found.
///
/// The review's second counterexample, kept. Three layers executed and 11,520 B
/// of reads happened; dropping the middle record and re-deriving the totals gave
/// a trace that reconciled **35 checks** over two layers and never mentioned the
/// third. Totals derived from the records supplied, and then re-summed from the
/// same records, cannot notice one that is not there.
///
/// The step's own boundary can: `whole` is the authority's delta from before the
/// first layer, and the layers must account for every byte that **entered** a
/// cache inside it.
#[test]
fn a_layer_that_ran_may_not_be_left_out_of_the_step() {
    let dir = scratch("omitted");
    let (path, x) = write_shard(&dir, 0x0023_7070);
    let case = Case {
        placement: Placement::Host,
        device_room: Room::Roomy,
        host_room: Room::Roomy,
        warm: Warm::Cold,
        layers: LAYERS,
        fail_read_at: None,
    };
    let (complete, failures) = run_case(case, &path, &x);
    assert_eq!(failures, 0);
    assert_eq!(complete.layers.len(), LAYERS as usize);
    complete.reconcile().expect("the complete step reconciles");

    let read: u64 = complete
        .whole
        .iter()
        .filter(|(scope, _)| *scope == Scope::Host)
        .map(|(_, flow)| flow.read_bytes)
        .sum();
    assert!(read > 0, "the step read nothing; there is nothing to omit");

    // Drop the middle layer and re-derive the totals exactly as `new` would, so
    // the forgery is internally consistent: only the outer boundary can tell.
    let mut forged = complete.clone();
    let dropped = forged.layers.remove(1);
    let mut totals: std::collections::BTreeMap<Scope, moxie_memory::ByteFlow> =
        std::collections::BTreeMap::new();
    for layer in &forged.layers {
        for delta in &layer.scopes {
            let entry = totals.entry(delta.scope).or_default();
            *entry = entry.plus(&delta.flow);
        }
    }
    forged.totals = totals.into_iter().collect();
    let failure = forged
        .reconcile()
        .expect_err("a step that omits an executed layer must not reconcile");
    assert_eq!(failure.check, "step-covers-every-byte", "{failure}");
    println!(
        "omitted layer {} of a {}-layer step reported as: {failure}",
        dropped.layer, LAYERS
    );
}

/// A trace read out of somebody else's ledger must be refused.
///
/// The review's third counterexample, kept. Handing `close` a separate, empty
/// ledger made the observed charges empty and the accounted charges empty, and
/// empty equals empty: a completed run reconciled with **no recorded ledger
/// charges at all**. Two numbers agreeing is worth nothing when both can be read
/// from the wrong place.
#[test]
fn a_trace_read_from_another_ledger_is_refused() {
    let dir = scratch("foreign-ledger");
    let (path, x) = write_shard(&dir, 0x0023_8080);
    let case = Case {
        placement: Placement::Host,
        device_room: Room::Roomy,
        host_room: Room::Roomy,
        warm: Warm::Cold,
        layers: 1,
        fail_read_at: None,
    };
    let mut ledger = ledger_for(case);
    let union = union_bytes(0);
    let mut authority =
        ResidencyAuthority::open(&mut ledger, &ResidencyRequest::new("foreign", union)).unwrap();
    let start = StepSnapshot::take(&authority).unwrap();
    let stranger = ledger_for(case);

    let policy = ExpertPolicy {
        device: StrategyControl::Off,
        host: StrategyControl::Auto,
        host_placement: StrategyControl::Off,
        ..ExpertPolicy::default()
    };
    let plan = compile_experts(
        &mlp(),
        &combine(),
        &route(0),
        &budget(case, 0, union, ResidentChunks::none(), 0, 0),
        &policy,
        None,
        None,
    )
    .unwrap();
    let snapshot = LayerSnapshot::take(0, &authority).unwrap();
    let mut run = GroupedRun::admit(&mut ledger, plan, roles(0), None).unwrap();
    run.load_activations(&x).unwrap();
    let mut src = FailingSource {
        inner: source(&path),
        fail_at: None,
        seen: 0,
    };
    run.run_to_completion(&mut authority, &mut src, TurnId::new(1), 0, u64::MAX)
        .unwrap();
    // The whole point: a ledger that never admitted any of this.
    let trace = snapshot.close(&run, &authority, &stranger).unwrap();
    run.close(&mut ledger).unwrap();
    authority.end_turn(TurnId::new(1));
    authority.retire_all(Scope::Host);
    authority.close(&mut ledger).unwrap();

    let failure = StepTrace::new(
        artifact().as_str().to_string(),
        "foreign ledger".to_string(),
        vec![trace],
        &start,
        &authority,
        &stranger,
    )
    .unwrap()
    .reconcile()
    .expect_err("a trace read from another ledger must not reconcile");
    assert_eq!(failure.check, "ledger-is-the-runs-own", "{failure}");
    println!("foreign ledger reported as: {failure}");
}

/// Free space that is ample in total and useless in pieces.
///
/// The review's second counterexample, kept and built the same way: a cache of
/// six experts, holding five, with the free space cut into 256 B holes by
/// retiring chunks out of order. The plan predicts **768 B of reads** and the
/// first 512 B chunk it needs fits no hole, so admission evicts the warm expert
/// the plan was counting on and reads it again — **1,536 B**.
///
/// The rule now asks whether **one contiguous run** can hold what the layer
/// admits, padding included, so this configuration declares a lower bound and
/// reconciles against it. The assertion that the planner does **not** call it
/// exact is the regression: a rule that drifted back to comparing totals would
/// fail here by name.
#[test]
fn free_space_in_pieces_is_not_an_exact_prediction() {
    use moxie_executor::residency::drain_reads;
    use moxie_memory::{AcquireRequest, Acquired, ChunkId, Content, UseClass};

    let dir = scratch("fragmented");
    let (path, x) = write_shard(&dir, 0x0023_0999);
    let case = Case {
        placement: Placement::Host,
        device_room: Room::Roomy,
        host_room: Room::Roomy,
        warm: Warm::Cold,
        layers: 1,
        fail_read_at: None,
    };
    let mut ledger = ledger_for(case);
    let mut authority =
        ResidencyAuthority::open(&mut ledger, &ResidencyRequest::new("fragment", 6 * CHUNK))
            .unwrap();
    let mut src = source(&path);
    let (shape, _) = moxie_plan::expert::shape_of(&mlp(), &combine()).unwrap();

    // Acquire one chunk and let it go: the placement stays, which is what makes
    // the cache full, and the lease does not, which is what makes it evictable.
    let mut load = |authority: &mut ResidencyAuthority, chunk: &ChunkId, now: u64| {
        let acquired = authority
            .acquire(AcquireRequest {
                chunk,
                destination: Scope::Host,
                now,
                deadline: u64::MAX,
                class: UseClass::demand(Content::Expert),
                turn: TurnId::new(90),
            })
            .unwrap();
        let lease = match acquired {
            Acquired::Ready(lease) => lease,
            Acquired::Pending { lease, work, .. } => {
                drain_reads(authority, &mut src, work).unwrap();
                lease
            }
        };
        authority.release(lease).unwrap();
    };

    for (now, expert) in [4, 5, 1, 2, 3, 0].into_iter().enumerate() {
        let (gate_up, down) = roles(0).chunks(expert, shape).unwrap();
        load(&mut authority, &gate_up, now as u64);
        load(&mut authority, &down, now as u64);
    }
    // Cut the free space into pieces: retire one 512 B chunk, fill it with a
    // chunk from another layer, then retire three 256 B chunks spread through
    // the arena. Ample free bytes, no hole bigger than 256.
    authority
        .retire(Scope::Host, &roles(0).chunks(0, shape).unwrap().0)
        .unwrap();
    load(&mut authority, &roles(1).chunks(0, shape).unwrap().0, 10);
    for expert in [0, 1, 2] {
        authority
            .retire(Scope::Host, &roles(0).chunks(expert, shape).unwrap().1)
            .unwrap();
    }
    authority.check_invariants().unwrap();

    let occupancy = authority.occupancy(Scope::Host).unwrap();
    assert!(
        occupancy.free_bytes > occupancy.largest_free_bytes,
        "this fixture is supposed to fragment the arena: {occupancy:?}"
    );
    println!(
        "fragmented host cache: {} B free in {} range(s), largest {} B",
        occupancy.free_bytes, occupancy.free_ranges, occupancy.largest_free_bytes
    );

    let resident = resident_chunks(&authority, None, 0);
    let policy = ExpertPolicy {
        device: StrategyControl::Off,
        host: StrategyControl::Auto,
        host_placement: StrategyControl::Off,
        ..ExpertPolicy::default()
    };
    let plan = compile_experts(
        &mlp(),
        &combine(),
        &[0, 4, 0, 4, 0, 4, 0, 4],
        &budget(
            case,
            0,
            6 * CHUNK,
            resident,
            0,
            occupancy.largest_free_bytes,
        ),
        &policy,
        None,
        None,
    )
    .unwrap();
    assert!(
        !matches!(plan.envelope().predicted.exactness, Exactness::Exact),
        "free space in pieces cannot promise an exact prediction: {:?}",
        plan.envelope().predicted.exactness
    );

    let start = StepSnapshot::take(&authority).unwrap();
    let snapshot = LayerSnapshot::take(0, &authority).unwrap();
    let mut run = GroupedRun::admit(&mut ledger, plan, roles(0), None).unwrap();
    run.load_activations(&x).unwrap();
    run.run_to_completion(&mut authority, &mut src, TurnId::new(91), 20, u64::MAX)
        .unwrap();
    authority.check_invariants().unwrap();
    let trace = snapshot.close(&run, &authority, &ledger).unwrap();
    run.close(&mut ledger).unwrap();
    authority.close(&mut ledger).unwrap();

    let host = trace
        .scopes
        .iter()
        .find(|s| s.scope == Scope::Host)
        .expect("a host scope");
    println!(
        "fragmentation forced {} B of reads against {} B predicted as a lower bound",
        host.flow.read_bytes, trace.predicted.host_read_bytes
    );
    let step = StepTrace::new(
        artifact().as_str().to_string(),
        "fragmentation".to_string(),
        vec![trace],
        &start,
        &authority,
        &ledger,
    )
    .unwrap();
    step.reconcile()
        .expect("a lower-bound prediction reconciles against what happened");
}
