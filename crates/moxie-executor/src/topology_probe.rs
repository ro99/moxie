//! Bounded measurements of direct transfer and device-memory paths.

use std::{
    sync::{
        Barrier, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

use moxie_cuda::{
    DeviceBuffer, Event, Module, ModuleImage, PinnedHostBuffer, RankContext, ResolvedModule,
    Stream, TrustedImage, query_device,
};
use moxie_plan::{DeviceCost, Endpoint, LinkCost, TopologyCosts};
use moxie_types::{DeviceCapability, Error, RankId, Result};

#[derive(Debug, Clone, Copy)]
pub struct ProbeConfig {
    pub small_bytes: usize,
    pub bulk_bytes: usize,
    pub reps: usize,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            small_bytes: 16 * 1024,
            bulk_bytes: 64 * 1024 * 1024,
            reps: 5,
        }
    }
}

/// Measure pageable host links, granted peer links, simultaneous traffic and device memory.
///
/// Mapped host memory remains unmeasured; neither mapped nor pinned memory is
/// used by the admitted execution paths.
pub fn probe_topology(ordinals: &[u32], config: ProbeConfig) -> Result<TopologyCosts> {
    if ordinals.is_empty()
        || config.small_bytes == 0
        || config.bulk_bytes == 0
        || config.reps == 0
        || config.small_bytes > config.bulk_bytes
    {
        return Err(invalid(
            "ordinals, positive sizes and repetitions are required; small_bytes must not exceed bulk_bytes",
        ));
    }
    let mut ordinals = ordinals.to_vec();
    ordinals.sort_unstable();
    if ordinals.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(invalid("device ordinals must be unique"));
    }
    let capabilities = ordinals
        .iter()
        .map(|ordinal| query_device(*ordinal))
        .collect::<Result<Vec<_>>>()?;
    let allocation = config
        .bulk_bytes
        .checked_mul(2)
        .ok_or_else(|| invalid("bulk buffer extent overflowed"))?;
    let (devices, mut links) = isolated(&capabilities, config, allocation)?;
    let h2d = concurrent_host(&ordinals, config.bulk_bytes, config.reps, Direction::H2d)?;
    let d2h = concurrent_host(&ordinals, config.bulk_bytes, config.reps, Direction::D2h)?;
    for (i, capability) in capabilities.iter().enumerate() {
        let uuid = Endpoint::Device(capability.uuid);
        links
            .iter_mut()
            .find(|link| link.from == Endpoint::Host && link.to == uuid)
            .unwrap()
            .concurrent_gbps = h2d[i];
        links
            .iter_mut()
            .find(|link| link.from == uuid && link.to == Endpoint::Host)
            .unwrap()
            .concurrent_gbps = d2h[i];
    }
    links.sort_by_key(|link| (link.from, link.to));
    Ok(TopologyCosts { devices, links })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    H2d,
    D2h,
}

#[derive(Debug, Clone, Copy)]
pub struct PinnedLink {
    pub device: moxie_types::DeviceUuid,
    pub direction: Direction,
    pub pageable_gbps: f64,
    pub pinned_gbps: f64,
    pub pageable_issue_us: f64,
    pub pinned_issue_us: f64,
    pub overlap: f64,
}

/// Measure pageable and pinned host transfers and pinned H2D copy/compute overlap.
pub fn probe_pinned(ordinals: &[u32], config: ProbeConfig) -> Result<Vec<PinnedLink>> {
    if ordinals.is_empty() || config.bulk_bytes < 4 || config.reps == 0 {
        return Err(invalid(
            "ordinals, at least four bulk bytes and positive repetitions are required",
        ));
    }
    let mut ordinals = ordinals.to_vec();
    ordinals.sort_unstable();
    if ordinals.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(invalid("device ordinals must be unique"));
    }
    let float_count = config.bulk_bytes / core::mem::size_of::<f32>();
    let float_count = u32::try_from(float_count)
        .map_err(|_| invalid("bulk buffer is too large for the smoke kernel"))?;
    let capabilities = ordinals
        .iter()
        .map(|ordinal| query_device(*ordinal))
        .collect::<Result<Vec<_>>>()?;
    let mut links = Vec::with_capacity(capabilities.len() * 2);
    for capability in capabilities {
        links.extend(probe_pinned_device(&capability, config, float_count)?);
    }
    Ok(links)
}

struct PinnedProbeResources<'ctx> {
    pinned: PinnedHostBuffer<'ctx>,
    stream: Stream<'ctx>,
    compute_stream: Stream<'ctx>,
    transfer: DeviceBuffer<'ctx>,
    x: DeviceBuffer<'ctx>,
    y: DeviceBuffer<'ctx>,
    pageable: Vec<u8>,
    pageable_out: Vec<u8>,
    package: ResolvedModule<'ctx>,
    start: Event<'ctx>,
    end: Event<'ctx>,
    both_start: Event<'ctx>,
    both_end: Event<'ctx>,
    kernel_end: Event<'ctx>,
}

fn probe_pinned_device(
    capability: &DeviceCapability,
    config: ProbeConfig,
    float_count: u32,
) -> Result<[PinnedLink; 2]> {
    let ctx = RankContext::acquire(RankId(capability.ordinal), capability.ordinal)?;
    // Everything is owned outside the fallible measurement closure. If any
    // enqueue path fails, the final context sync runs before these resources
    // can be dropped.
    // SAFETY: the image is the immutable nvcc output embedded by this build.
    let image = unsafe { TrustedImage::from_build_output(moxie_kernels::SMOKE_FATBIN)? };
    let package = Module::load(&ctx, ModuleImage::Binary(image))?
        .resolve_all(&[moxie_kernels::AXPY_F32.to_owned()])?;
    let mut resources = PinnedProbeResources {
        pinned: PinnedHostBuffer::alloc(&ctx, config.bulk_bytes)?,
        stream: Stream::new(&ctx)?,
        compute_stream: Stream::new(&ctx)?,
        transfer: DeviceBuffer::alloc(&ctx, config.bulk_bytes)?,
        x: DeviceBuffer::alloc(&ctx, config.bulk_bytes)?,
        y: DeviceBuffer::alloc(&ctx, config.bulk_bytes)?,
        pageable: zeroed(config.bulk_bytes)?,
        pageable_out: zeroed(config.bulk_bytes)?,
        package,
        start: Event::new(&ctx)?,
        end: Event::new(&ctx)?,
        both_start: Event::new(&ctx)?,
        both_end: Event::new(&ctx)?,
        kernel_end: Event::new(&ctx)?,
    };
    resources.pinned.as_mut_slice().fill(0x5a);
    resources.pageable.fill(0x5a);
    resources.transfer.copy_from_host(&resources.pageable)?;
    resources.x.copy_from_host(&resources.pageable)?;
    resources.y.copy_from_host(&resources.pageable)?;
    let measurements = (|| {
        let PinnedProbeResources {
            pinned,
            stream,
            compute_stream,
            transfer,
            x,
            y,
            pageable,
            pageable_out,
            package,
            start,
            end,
            both_start,
            both_end,
            kernel_end,
        } = &mut resources;
        // Warm each transfer and kernel path before collecting samples.
        // SAFETY: host slices and the transfer allocation remain live until the
        // following stream synchronization observes all four copies.
        unsafe {
            transfer.copy_from_host_async_at(0, pageable, stream)?;
            transfer.copy_from_host_async_at(0, pinned.as_slice(), stream)?;
            transfer.copy_to_host_async(pageable_out, stream)?;
            transfer.copy_to_host_async(pinned.as_mut_slice(), stream)?;
        }
        stream.synchronize()?;
        launch_axpy(package, x, y, float_count, compute_stream)?;
        compute_stream.synchronize()?;

        let pageable_h2d = event_times(config.reps, stream, start, end, || {
            // SAFETY: the pageable source, device buffer and stream outlive the
            // measured operation, and `event_times` observes completion.
            unsafe { transfer.copy_from_host_async_at(0, pageable, stream) }
        })?;
        let pageable_h2d_issue = issue_times(config.reps, stream, || {
            // SAFETY: the source remains live until the stream is synchronized.
            unsafe { transfer.copy_from_host_async_at(0, pageable, stream) }
        })?;
        let pinned_h2d = event_times(config.reps, stream, start, end, || {
            // SAFETY: pinned host storage, destination and stream remain live;
            // `event_times` waits before the next sample.
            unsafe { transfer.copy_from_host_async_at(0, pinned.as_slice(), stream) }
        })?;
        let pinned_h2d_issue = issue_times(config.reps, stream, || {
            // SAFETY: the pinned source remains live until stream synchronization.
            unsafe { transfer.copy_from_host_async_at(0, pinned.as_slice(), stream) }
        })?;
        let pageable_d2h = event_times(config.reps, stream, start, end, || {
            // SAFETY: the destination, device buffer and stream outlive the
            // measured operation, and `event_times` observes completion.
            unsafe { transfer.copy_to_host_async(pageable_out, stream) }
        })?;
        let pageable_d2h_issue = issue_times(config.reps, stream, || {
            // SAFETY: the destination remains live until the stream is synchronized.
            unsafe { transfer.copy_to_host_async(pageable_out, stream) }
        })?;
        let pinned_d2h = event_times(config.reps, stream, start, end, || {
            // SAFETY: pinned host storage, source and stream remain live;
            // `event_times` waits before the next sample.
            unsafe { transfer.copy_to_host_async(pinned.as_mut_slice(), stream) }
        })?;
        let pinned_d2h_issue = issue_times(config.reps, stream, || {
            // SAFETY: the pinned destination remains live until synchronization.
            unsafe { transfer.copy_to_host_async(pinned.as_mut_slice(), stream) }
        })?;

        let copy_calibration = event_times(1, stream, start, end, || {
            // SAFETY: the pinned source remains live until event completion.
            unsafe { transfer.copy_from_host_async_at(0, pinned.as_slice(), stream) }
        })?[0];
        let kernel_calibration = event_times(1, compute_stream, start, end, || {
            launch_axpy(package, x, y, float_count, compute_stream)
        })?[0];
        let kernel_reps = (copy_calibration / kernel_calibration).ceil().max(1.0) as usize;
        let kernel_alone = event_times(config.reps, compute_stream, start, end, || {
            for _ in 0..kernel_reps {
                launch_axpy(package, x, y, float_count, compute_stream)?;
            }
            Ok(())
        })?;
        let mut both = Vec::with_capacity(config.reps);
        for _ in 0..config.reps {
            both_start.record(stream)?;
            compute_stream.wait_event(both_start)?;
            // SAFETY: pinned host storage and destination remain live until the
            // end event, which is ordered after the copy and compute streams.
            unsafe {
                transfer.copy_from_host_async_at(0, pinned.as_slice(), stream)?;
            }
            for _ in 0..kernel_reps {
                launch_axpy(package, x, y, float_count, compute_stream)?;
            }
            kernel_end.record(compute_stream)?;
            stream.wait_event(kernel_end)?;
            both_end.record(stream)?;
            both_end.synchronize()?;
            both.push(f64::from(Event::elapsed_ms(both_start, both_end)?) / 1e3);
        }

        let copy_alone = median_metric(&pinned_h2d, |seconds| seconds)?;
        let kernel_alone = median_metric(&kernel_alone, |seconds| seconds)?;
        let both = median_metric(&both, |seconds| seconds)?;
        let overlap =
            ((copy_alone + kernel_alone - both) / copy_alone.min(kernel_alone)).clamp(0.0, 1.0);
        Ok([
            PinnedLink {
                device: ctx.uuid(),
                direction: Direction::H2d,
                pageable_gbps: median_metric(&pageable_h2d, |seconds| {
                    config.bulk_bytes as f64 / seconds / 1e9
                })?,
                pinned_gbps: median_metric(&pinned_h2d, |seconds| {
                    config.bulk_bytes as f64 / seconds / 1e9
                })?,
                pageable_issue_us: median_metric(&pageable_h2d_issue, |seconds| seconds * 1e6)?,
                pinned_issue_us: median_metric(&pinned_h2d_issue, |seconds| seconds * 1e6)?,
                overlap,
            },
            PinnedLink {
                device: ctx.uuid(),
                direction: Direction::D2h,
                pageable_gbps: median_metric(&pageable_d2h, |seconds| {
                    config.bulk_bytes as f64 / seconds / 1e9
                })?,
                pinned_gbps: median_metric(&pinned_d2h, |seconds| {
                    config.bulk_bytes as f64 / seconds / 1e9
                })?,
                pageable_issue_us: median_metric(&pageable_d2h_issue, |seconds| seconds * 1e6)?,
                pinned_issue_us: median_metric(&pinned_d2h_issue, |seconds| seconds * 1e6)?,
                overlap: 0.0,
            },
        ])
    })();
    match ctx.synchronize() {
        Ok(()) => measurements,
        Err(sync_error) => {
            // If completion is unknown, withhold the resources the queued work may use.
            std::mem::forget(resources);
            match measurements {
                Err(original_error) => Err(original_error),
                Ok(_) => Err(sync_error),
            }
        }
    }
}

fn launch_axpy(
    package: &ResolvedModule<'_>,
    x: &DeviceBuffer<'_>,
    y: &DeviceBuffer<'_>,
    elements: u32,
    stream: &Stream<'_>,
) -> Result<()> {
    let (mut x_ptr, mut y_ptr, mut scale, mut count) =
        (x.device_ptr(), y.device_ptr(), 1.0f32, elements);
    let mut params = [
        (&raw mut x_ptr).cast(),
        (&raw mut y_ptr).cast(),
        (&raw mut scale).cast(),
        (&raw mut count).cast(),
    ];
    // SAFETY: arguments match the smoke AXPY ABI, both device buffers hold at
    // least `elements` floats, and they remain live through event completion.
    unsafe {
        package.launch_async(
            0,
            stream,
            (elements.div_ceil(256), 1, 1),
            (256, 1, 1),
            0,
            &mut params,
        )
    }
}

fn isolated(
    caps: &[DeviceCapability],
    config: ProbeConfig,
    allocation: usize,
) -> Result<(Vec<DeviceCost>, Vec<LinkCost>)> {
    let contexts = caps
        .iter()
        .map(|cap| RankContext::acquire(RankId(cap.ordinal), cap.ordinal))
        .collect::<Result<Vec<_>>>()?;
    let free = contexts
        .iter()
        .map(|ctx| ctx.measure().map(|m| m.free_bytes))
        .collect::<Result<Vec<_>>>()?;
    if free.contains(&0) {
        return Err(invalid("usable device memory must be positive"));
    }
    let streams = contexts
        .iter()
        .map(Stream::new)
        .collect::<Result<Vec<_>>>()?;
    let buffers = contexts
        .iter()
        .map(|ctx| DeviceBuffer::alloc(ctx, allocation))
        .collect::<Result<Vec<_>>>()?;
    let (small, bulk) = (zeroed(config.small_bytes)?, zeroed(config.bulk_bytes)?);
    let (mut small_out, mut bulk_out) = (zeroed(config.small_bytes)?, zeroed(config.bulk_bytes)?);
    let (mut devices, mut links) = (Vec::with_capacity(caps.len()), Vec::new());

    for i in 0..caps.len() {
        let hs = host_times(config.reps, || buffers[i].copy_from_host_at(0, &small))?;
        let hb = host_times(config.reps, || buffers[i].copy_from_host_at(0, &bulk))?;
        let ds = host_times(config.reps, || {
            buffers[i].copy_to_host_at(0, &mut small_out)
        })?;
        let db = host_times(config.reps, || buffers[i].copy_to_host_at(0, &mut bulk_out))?;
        let host = Endpoint::Host;
        let device = Endpoint::Device(contexts[i].uuid());
        links.push(LinkCost {
            from: host,
            to: device,
            latency_us: median_metric(&hs, |seconds| seconds * 1e6)?,
            bandwidth_gbps: median_metric(&hb, |seconds| config.bulk_bytes as f64 / seconds / 1e9)?,
            concurrent_gbps: 1.0,
        });
        links.push(LinkCost {
            from: device,
            to: host,
            latency_us: median_metric(&ds, |seconds| seconds * 1e6)?,
            bandwidth_gbps: median_metric(&db, |seconds| config.bulk_bytes as f64 / seconds / 1e9)?,
            concurrent_gbps: 1.0,
        });

        let (start, end) = (Event::new(&contexts[i])?, Event::new(&contexts[i])?);
        let mut memory_copy = || {
            // SAFETY: disjoint extents in this live buffer and stream remain valid through event completion.
            unsafe {
                buffers[i].copy_from_peer_async_at(
                    config.bulk_bytes,
                    &buffers[i],
                    0,
                    config.bulk_bytes,
                    &streams[i],
                )
            }
        };
        memory_copy()?;
        streams[i].synchronize()?;
        let times = event_times(config.reps, &streams[i], &start, &end, &mut memory_copy)?;
        let memory_bytes = config
            .bulk_bytes
            .checked_mul(2)
            .ok_or_else(|| invalid("memory byte count overflowed"))?;
        devices.push(DeviceCost {
            device: contexts[i].uuid(),
            memory_gbps: median_metric(&times, |seconds| memory_bytes as f64 / seconds / 1e9)?,
            usable_bytes: free[i],
        });
    }

    let mut peers = Vec::new();
    for (from, source) in caps.iter().enumerate() {
        for (to, destination) in caps.iter().enumerate() {
            if from != to && destination.can_access_peer(source.ordinal) {
                peers.push((from, to));
            }
        }
    }
    for &(from, to) in &peers {
        contexts[to].enable_peer_access(&contexts[from])?;
    }
    for &(from, to) in &peers {
        peer_copy(
            &buffers[to],
            &buffers[from],
            config.small_bytes,
            &streams[to],
        )?;
        peer_copy(
            &buffers[to],
            &buffers[from],
            config.bulk_bytes,
            &streams[to],
        )?;
    }
    for stream in &streams {
        stream.synchronize()?;
    }

    let timers = peers
        .iter()
        .map(|(_, to)| Ok((Event::new(&contexts[*to])?, Event::new(&contexts[*to])?)))
        .collect::<Result<Vec<_>>>()?;
    let peer_link_start = links.len();
    for (&(from, to), (start, end)) in peers.iter().zip(&timers) {
        let small_times = event_times(config.reps, &streams[to], start, end, || {
            peer_copy(
                &buffers[to],
                &buffers[from],
                config.small_bytes,
                &streams[to],
            )
        })?;
        let bulk_times = event_times(config.reps, &streams[to], start, end, || {
            peer_copy(
                &buffers[to],
                &buffers[from],
                config.bulk_bytes,
                &streams[to],
            )
        })?;
        links.push(LinkCost {
            from: Endpoint::Device(caps[from].uuid),
            to: Endpoint::Device(caps[to].uuid),
            latency_us: median_metric(&small_times, |seconds| seconds * 1e6)?,
            bandwidth_gbps: median_metric(&bulk_times, |seconds| {
                config.bulk_bytes as f64 / seconds / 1e9
            })?,
            concurrent_gbps: 1.0,
        });
    }

    let mut samples = vec![Vec::with_capacity(config.reps); peers.len()];
    for _ in 0..config.reps {
        for ((_, to), (start, _)) in peers.iter().zip(&timers) {
            start.record(&streams[*to])?;
        }
        for &(from, to) in &peers {
            peer_copy(
                &buffers[to],
                &buffers[from],
                config.bulk_bytes,
                &streams[to],
            )?;
        }
        for ((_, to), (_, end)) in peers.iter().zip(&timers) {
            end.record(&streams[*to])?;
        }
        for (_, end) in &timers {
            end.synchronize()?;
        }
        for (sample, (start, end)) in samples.iter_mut().zip(&timers) {
            sample.push(f64::from(Event::elapsed_ms(start, end)?) / 1e3);
        }
    }
    for (link, sample) in links[peer_link_start..].iter_mut().zip(samples) {
        link.concurrent_gbps =
            median_metric(&sample, |seconds| config.bulk_bytes as f64 / seconds / 1e9)?;
    }
    Ok((devices, links))
}

fn concurrent_host(
    ordinals: &[u32],
    bytes: usize,
    reps: usize,
    direction: Direction,
) -> Result<Vec<f64>> {
    let barrier = Barrier::new(ordinals.len());
    let failed = AtomicBool::new(false);
    let first_error = Mutex::new(None::<Error>);
    std::thread::scope(|scope| {
        let handles = ordinals
            .iter()
            .map(|&ordinal| {
                let (barrier, failed, first_error) = (&barrier, &failed, &first_error);
                scope.spawn(move || -> Result<f64> {
                    let mut ready = false;
                    let result = (|| {
                        let ctx = RankContext::acquire(RankId(ordinal), ordinal)?;
                        let mut device = DeviceBuffer::alloc(&ctx, bytes)?;
                        let mut host = zeroed(bytes)?;
                        match direction {
                            Direction::H2d => device.copy_from_host(&host)?,
                            Direction::D2h => {
                                device.copy_from_host(&host)?;
                                device.copy_to_host(&mut host)?;
                            }
                        }
                        ready = true;
                        barrier.wait();
                        if failed.load(Ordering::Acquire) {
                            return Ok(0.0);
                        }
                        let times = (0..reps)
                            .map(|_| {
                                barrier.wait();
                                let time = Instant::now();
                                let result = match direction {
                                    Direction::H2d => device.copy_from_host(&host),
                                    Direction::D2h => device.copy_to_host(&mut host),
                                };
                                result.map(|()| time.elapsed().as_secs_f64())
                            })
                            .collect::<Vec<_>>()
                            .into_iter()
                            .collect::<Result<Vec<_>>>()?;
                        median_metric(&times, |seconds| bytes as f64 / seconds / 1e9)
                    })();
                    if !ready {
                        if let Err(error) = result {
                            let mut first = first_error.lock().unwrap();
                            if first.is_none() {
                                *first = Some(error);
                            }
                        }
                        failed.store(true, Ordering::Release);
                        barrier.wait();
                        return Ok(0.0);
                    }
                    result
                })
            })
            .collect::<Vec<_>>();
        let mut results = Vec::with_capacity(handles.len());
        let mut join_error = None;
        for handle in handles {
            match handle.join() {
                Ok(result) => results.push(result),
                Err(_) if join_error.is_none() => {
                    join_error = Some(invalid("probe worker panicked"));
                }
                Err(_) => {}
            }
        }
        if let Some(error) = first_error.lock().unwrap().take() {
            return Err(error);
        }
        if let Some(error) = join_error {
            return Err(error);
        }
        results.into_iter().collect()
    })
}

fn host_times(reps: usize, mut copy: impl FnMut() -> Result<()>) -> Result<Vec<f64>> {
    copy()?;
    (0..reps)
        .map(|_| {
            let now = Instant::now();
            copy()?;
            Ok(now.elapsed().as_secs_f64())
        })
        .collect()
}

fn issue_times(
    reps: usize,
    stream: &Stream<'_>,
    mut enqueue: impl FnMut() -> Result<()>,
) -> Result<Vec<f64>> {
    (0..reps)
        .map(|_| {
            let start = Instant::now();
            enqueue()?;
            let elapsed = start.elapsed().as_secs_f64();
            stream.synchronize()?;
            Ok(elapsed)
        })
        .collect()
}

fn event_times(
    reps: usize,
    stream: &Stream<'_>,
    start: &Event<'_>,
    end: &Event<'_>,
    mut copy: impl FnMut() -> Result<()>,
) -> Result<Vec<f64>> {
    (0..reps)
        .map(|_| {
            start.record(stream)?;
            copy()?;
            end.record(stream)?;
            end.synchronize()?;
            Ok(f64::from(Event::elapsed_ms(start, end)?) / 1e3)
        })
        .collect()
}

fn peer_copy(
    dst: &DeviceBuffer<'_>,
    src: &DeviceBuffer<'_>,
    bytes: usize,
    stream: &Stream<'_>,
) -> Result<()> {
    // SAFETY: caller retains both buffers and the destination stream until this copy completes.
    unsafe { dst.copy_from_peer_async_at(bytes, src, 0, bytes, stream) }
}

fn zeroed(bytes: usize) -> Result<Vec<u8>> {
    let mut data = Vec::new();
    data.try_reserve_exact(bytes)
        .map_err(|e| invalid(format!("host buffer allocation failed: {e}")))?;
    data.resize(bytes, 0);
    Ok(data)
}

fn median_metric(seconds: &[f64], mut convert: impl FnMut(f64) -> f64) -> Result<f64> {
    let mut values = seconds
        .iter()
        .map(|time| {
            if !time.is_finite() || *time <= 0.0 {
                return Err(invalid("measurement must be finite and positive"));
            }
            let value = convert(*time);
            if !value.is_finite() || value <= 0.0 {
                Err(invalid("measurement must be finite and positive"))
            } else {
                Ok(value)
            }
        })
        .collect::<Result<Vec<_>>>()?;
    if values.is_empty() {
        return Err(invalid("measurement must be finite and positive"));
    }
    values.sort_by(f64::total_cmp);
    let n = values.len();
    let value = if n.is_multiple_of(2) {
        let lower = values[n / 2 - 1];
        lower + (values[n / 2] - lower) / 2.0
    } else {
        values[n / 2]
    };
    if !value.is_finite() || value <= 0.0 {
        return Err(invalid("median must be finite and positive"));
    }
    Ok(value)
}

fn invalid(detail: impl Into<String>) -> Error {
    Error::InvalidRequest {
        field: "topology_probe",
        detail: detail.into(),
    }
}
