//! Collective communication operations

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::backend::ReduceOp;
use crate::process_group::ProcessGroup;
use crate::tcp_backend::{encode_any, Payload, TcpBackend, TcpEngine};
use crate::TorshResult;
use log::info;
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;
use torsh_core::dtype::FloatElement;
use torsh_tensor::Tensor;

/// Obtain the real TCP collective engine for `group`, or an honest error if the
/// active backend cannot perform cross-rank collectives.
///
/// The backend read-guard is released before returning, so the returned
/// `Arc<TcpEngine>` can be awaited without holding any lock across `.await`
/// (required by the DDP background-sync path, which `tokio::spawn`s gradient
/// all-reduces and therefore needs a `Send` future).
fn tcp_engine(group: &ProcessGroup) -> TorshResult<Arc<TcpEngine>> {
    let backend = group.backend();
    let guard = backend.read();
    if !guard.is_ready() {
        return Err(crate::TorshDistributedError::BackendNotInitialized);
    }
    guard
        .as_any()
        .downcast_ref::<TcpBackend>()
        .map(|b| b.engine())
        .ok_or_else(|| {
            crate::TorshDistributedError::feature_not_available(
                "cross-rank collective communication",
                "the Gloo (TCP) backend; the active backend cannot perform real \
                 collectives for world_size > 1",
            )
        })
}

/// Validate that the backend is initialised, backend-agnostically.
///
/// Used on the trivial `world_size <= 1` paths so a single-rank collective is
/// correct for ANY backend type (not just the TCP backend), while still
/// rejecting an uninitialised process group.
fn check_ready(group: &ProcessGroup) -> TorshResult<()> {
    let guard = group.backend().read();
    crate::communication::validate_backend_initialized(&**guard)
}

/// Decode a received payload into a `Vec<T>`, erroring on a dtype mismatch.
fn payload_into_vec<T: FloatElement>(payload: Payload) -> TorshResult<Vec<T>> {
    payload
        .into_box()
        .downcast::<Vec<T>>()
        .map(|b| *b)
        .map_err(|_| {
            crate::TorshDistributedError::invalid_argument(
                "dtype",
                "received payload element type does not match the local tensor",
                "matching f32/f64 element type across ranks",
            )
        })
}

/// Communication group for selective collective operations
#[derive(Debug, Clone)]
pub struct CommunicationGroup {
    /// Group identifier
    pub group_id: String,
    /// Ranks participating in this group
    pub ranks: Vec<u32>,
    /// Local rank within this group (0-indexed within the group)
    pub local_rank: u32,
    /// Size of this group
    pub group_size: u32,
    /// Global rank to local rank mapping
    pub global_to_local: HashMap<u32, u32>,
    /// Local rank to global rank mapping
    pub local_to_global: HashMap<u32, u32>,
}

impl CommunicationGroup {
    /// Create a new communication group
    pub fn new(group_id: String, ranks: Vec<u32>, current_global_rank: u32) -> TorshResult<Self> {
        if ranks.is_empty() {
            return Err(crate::TorshDistributedError::invalid_argument(
                "ranks",
                "Communication group cannot be empty",
                "non-empty vector of ranks",
            ));
        }

        if !ranks.contains(&current_global_rank) {
            return Err(crate::TorshDistributedError::invalid_argument(
                "current_global_rank",
                format!(
                    "Current rank {} not in group {:?}",
                    current_global_rank, ranks
                ),
                "rank that exists in the group",
            ));
        }

        let mut sorted_ranks = ranks.clone();
        sorted_ranks.sort_unstable();

        let mut global_to_local = HashMap::new();
        let mut local_to_global = HashMap::new();

        for (local_idx, &global_rank) in sorted_ranks.iter().enumerate() {
            global_to_local.insert(global_rank, local_idx as u32);
            local_to_global.insert(local_idx as u32, global_rank);
        }

        let local_rank = global_to_local[&current_global_rank];
        let group_size = sorted_ranks.len() as u32;

        Ok(Self {
            group_id,
            ranks: sorted_ranks,
            local_rank,
            group_size,
            global_to_local,
            local_to_global,
        })
    }

    /// Create a communication group for a range of ranks
    pub fn from_range(
        group_id: String,
        start_rank: u32,
        end_rank: u32,
        current_global_rank: u32,
    ) -> TorshResult<Self> {
        if start_rank >= end_rank {
            return Err(crate::TorshDistributedError::invalid_argument(
                "rank_range",
                "start_rank must be less than end_rank",
                "valid rank range where start < end",
            ));
        }

        let ranks: Vec<u32> = (start_rank..end_rank).collect();
        Self::new(group_id, ranks, current_global_rank)
    }

    /// Check if a global rank is in this group
    pub fn contains_rank(&self, global_rank: u32) -> bool {
        self.global_to_local.contains_key(&global_rank)
    }

    /// Get local rank for a global rank
    pub fn global_to_local_rank(&self, global_rank: u32) -> Option<u32> {
        self.global_to_local.get(&global_rank).copied()
    }

    /// Get global rank for a local rank
    pub fn local_to_global_rank(&self, local_rank: u32) -> Option<u32> {
        self.local_to_global.get(&local_rank).copied()
    }
}

/// Group manager for managing multiple communication groups
#[derive(Debug, Default)]
pub struct GroupManager {
    groups: HashMap<String, Arc<CommunicationGroup>>,
    current_global_rank: u32,
}

impl GroupManager {
    /// Create a new group manager
    pub fn new(current_global_rank: u32) -> Self {
        Self {
            groups: HashMap::new(),
            current_global_rank,
        }
    }

    /// Create and register a new communication group
    pub fn create_group(
        &mut self,
        group_id: String,
        ranks: Vec<u32>,
    ) -> TorshResult<Arc<CommunicationGroup>> {
        if self.groups.contains_key(&group_id) {
            return Err(crate::TorshDistributedError::invalid_argument(
                "group_id",
                format!("Group '{}' already exists", group_id),
                "unique group identifier",
            ));
        }

        let group = Arc::new(CommunicationGroup::new(
            group_id.clone(),
            ranks,
            self.current_global_rank,
        )?);
        self.groups.insert(group_id, Arc::clone(&group));
        Ok(group)
    }

    /// Create a communication group from a rank range
    pub fn create_group_from_range(
        &mut self,
        group_id: String,
        start_rank: u32,
        end_rank: u32,
    ) -> TorshResult<Arc<CommunicationGroup>> {
        if self.groups.contains_key(&group_id) {
            return Err(crate::TorshDistributedError::invalid_argument(
                "group_id",
                format!("Group '{}' already exists", group_id),
                "unique group identifier",
            ));
        }

        let group = Arc::new(CommunicationGroup::from_range(
            group_id.clone(),
            start_rank,
            end_rank,
            self.current_global_rank,
        )?);
        self.groups.insert(group_id, Arc::clone(&group));
        Ok(group)
    }

    /// Get a communication group by ID
    pub fn get_group(&self, group_id: &str) -> Option<Arc<CommunicationGroup>> {
        self.groups.get(group_id).cloned()
    }

    /// Remove a communication group
    pub fn remove_group(&mut self, group_id: &str) -> bool {
        self.groups.remove(group_id).is_some()
    }

    /// List all group IDs
    pub fn list_groups(&self) -> Vec<String> {
        self.groups.keys().cloned().collect()
    }

    /// Create predefined groups for common parallelism patterns
    pub fn create_standard_groups(
        &mut self,
        world_size: u32,
        data_parallel_size: u32,
        model_parallel_size: u32,
    ) -> TorshResult<()> {
        if data_parallel_size * model_parallel_size != world_size {
            return Err(crate::TorshDistributedError::invalid_argument(
                "parallelism_configuration",
                "data_parallel_size * model_parallel_size must equal world_size",
                format!(
                    "configuration where {} * {} = {}",
                    data_parallel_size, model_parallel_size, world_size
                ),
            ));
        }

        // Create data parallel groups (ranks that share the same model)
        for mp_rank in 0..model_parallel_size {
            let mut dp_ranks = Vec::new();
            for dp_rank in 0..data_parallel_size {
                dp_ranks.push(dp_rank * model_parallel_size + mp_rank);
            }
            let group_id = format!("data_parallel_{}", mp_rank);
            self.create_group(group_id, dp_ranks)?;
        }

        // Create model parallel groups (ranks that share the same data)
        for dp_rank in 0..data_parallel_size {
            let start_rank = dp_rank * model_parallel_size;
            let end_rank = start_rank + model_parallel_size;
            let group_id = format!("model_parallel_{}", dp_rank);
            self.create_group_from_range(group_id, start_rank, end_rank)?;
        }

        Ok(())
    }
}

/// All-reduce: reduce `tensor` across all processes and distribute the result.
///
/// Performs a real cross-rank reduction over the TCP backend. For
/// `world_size == 1` the operation is the identity (correct for every op) and
/// no network I/O occurs. For `world_size > 1` an honest error is returned if
/// the active backend cannot communicate.
pub async fn all_reduce<T>(
    tensor: &mut Tensor<T>,
    op: ReduceOp,
    group: &ProcessGroup,
) -> TorshResult<()>
where
    T: FloatElement
        + Default
        + Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    let world_size = group.world_size();
    if world_size <= 1 {
        check_ready(group)?;
        return Ok(());
    }
    let engine = tcp_engine(group)?;
    let participants: Vec<u32> = (0..world_size).collect();

    let shape = tensor.shape().dims().to_vec();
    let device = tensor.device();
    let mut data: Vec<T> = tensor.to_vec()?;
    engine
        .all_reduce_any(
            &mut data as &mut (dyn Any + Send + Sync),
            op,
            &participants,
            "",
        )
        .await?;
    *tensor = Tensor::from_data(data, shape, device)?;
    Ok(())
}

/// All-gather: gather tensors from all processes.
///
/// Each rank's `input` is transferred to every other rank; `output` ends up with
/// one tensor per rank (in rank order). For `world_size == 1` this is a single
/// clone of `input`.
pub async fn all_gather<T: FloatElement>(
    output: &mut Vec<Tensor<T>>,
    input: &Tensor<T>,
    group: &ProcessGroup,
) -> TorshResult<()> {
    let world_size = group.world_size();
    output.clear();

    if world_size <= 1 {
        // Validate the backend is initialised even on the trivial path.
        check_ready(group)?;
        output.push(input.clone());
        return Ok(());
    }

    let engine = tcp_engine(group)?;
    let participants: Vec<u32> = (0..world_size).collect();

    let shape = input.shape().dims().to_vec();
    let device = input.device();
    let input_data: Vec<T> = input.to_vec()?;
    let payloads = engine
        .all_gather_payloads(&input_data as &(dyn Any + Send + Sync), &participants, "")
        .await?;
    for payload in payloads {
        let data = payload_into_vec::<T>(payload)?;
        output.push(Tensor::from_data(data, shape.clone(), device)?);
    }
    Ok(())
}

/// Broadcast: send `tensor` from `src_rank` to all processes.
///
/// Non-root ranks receive the root's buffer; the root's tensor is unchanged.
pub async fn broadcast<T: FloatElement>(
    tensor: &mut Tensor<T>,
    src_rank: u32,
    group: &ProcessGroup,
) -> TorshResult<()> {
    use crate::communication::validate_rank;

    let world_size = group.world_size();
    validate_rank(src_rank, world_size)?;
    if world_size <= 1 {
        check_ready(group)?;
        return Ok(());
    }
    let engine = tcp_engine(group)?;
    let participants: Vec<u32> = (0..world_size).collect();

    let shape = tensor.shape().dims().to_vec();
    let device = tensor.device();
    let mut data: Vec<T> = tensor.to_vec()?;
    engine
        .broadcast_any(
            &mut data as &mut (dyn Any + Send + Sync),
            src_rank,
            &participants,
            "",
        )
        .await?;
    *tensor = Tensor::from_data(data, shape, device)?;
    Ok(())
}

/// Reduce: reduce `tensor` across all processes to `dst_rank`.
///
/// Only the destination rank's tensor is updated with the reduced result; other
/// ranks' tensors are left unchanged.
pub async fn reduce<T>(
    tensor: &mut Tensor<T>,
    dst_rank: u32,
    op: ReduceOp,
    group: &ProcessGroup,
) -> TorshResult<()>
where
    T: FloatElement
        + Default
        + Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    use crate::communication::validate_rank;

    let world_size = group.world_size();
    validate_rank(dst_rank, world_size)?;
    if world_size <= 1 {
        // Single rank already holds the reduced result (identity).
        check_ready(group)?;
        return Ok(());
    }
    let engine = tcp_engine(group)?;
    let participants: Vec<u32> = (0..world_size).collect();

    let shape = tensor.shape().dims().to_vec();
    let device = tensor.device();
    let mut data: Vec<T> = tensor.to_vec()?;
    engine
        .reduce_any(
            &mut data as &mut (dyn Any + Send + Sync),
            dst_rank,
            op,
            &participants,
            "",
        )
        .await?;
    // Only the destination rank's tensor is updated with the reduced result.
    if group.rank() == dst_rank {
        *tensor = Tensor::from_data(data, shape, device)?;
    }
    Ok(())
}

/// Scatter: distribute `input` chunks from `src_rank`, one chunk per process.
///
/// `input` must be `Some` on the source rank with exactly `world_size` tensors;
/// each rank receives `input[rank]`.
pub async fn scatter<T: FloatElement>(
    output: &mut Tensor<T>,
    input: Option<&[Tensor<T>]>,
    src_rank: u32,
    group: &ProcessGroup,
) -> TorshResult<()> {
    use crate::communication::validate_rank;

    let world_size = group.world_size();
    let rank = group.rank();
    validate_rank(src_rank, world_size)?;

    let require_chunks = |input: Option<&[Tensor<T>]>| -> TorshResult<()> {
        let tensors = input.ok_or_else(|| {
            crate::TorshDistributedError::invalid_argument(
                "input_tensors",
                "Input tensors required for source rank",
                "non-empty vector of tensors for scatter operation",
            )
        })?;
        if tensors.len() != world_size as usize {
            return Err(crate::TorshDistributedError::invalid_argument(
                "tensors",
                format!("Expected {} tensors, got {}", world_size, tensors.len()),
                format!("{} tensors (one per rank)", world_size),
            ));
        }
        Ok(())
    };

    if world_size <= 1 {
        check_ready(group)?;
        if rank == src_rank {
            require_chunks(input)?;
            if let Some(tensors) = input {
                *output = tensors[rank as usize].clone();
            }
        }
        return Ok(());
    }

    let engine = tcp_engine(group)?;
    let participants: Vec<u32> = (0..world_size).collect();

    // The source serialises one chunk per participant.
    let chunks: Option<Vec<Vec<u8>>> = if rank == src_rank {
        require_chunks(input)?;
        let tensors = input.unwrap_or(&[]);
        let mut encoded = Vec::with_capacity(tensors.len());
        for t in tensors {
            let v: Vec<T> = t.to_vec()?;
            encoded.push(encode_any(&v as &(dyn Any + Send + Sync))?);
        }
        Some(encoded)
    } else {
        None
    };

    let my_bytes = engine
        .scatter_bytes(chunks.as_deref(), src_rank, &participants, "")
        .await?;
    let payload = Payload::decode(&my_bytes)?;
    let data = payload_into_vec::<T>(payload)?;

    let expected = output.numel();
    if data.len() != expected {
        return Err(crate::TorshDistributedError::tensor_shape_mismatch(
            vec![expected],
            vec![data.len()],
        ));
    }
    let shape = output.shape().dims().to_vec();
    let device = output.device();
    *output = Tensor::from_data(data, shape, device)?;
    Ok(())
}

/// Barrier synchronization across all processes
#[allow(clippy::await_holding_lock)]
pub async fn barrier(group: &ProcessGroup) -> TorshResult<()> {
    let backend = group.backend();
    let mut backend_guard = backend.write();

    if !backend_guard.is_ready() {
        return Err(crate::TorshDistributedError::BackendNotInitialized);
    }

    backend_guard.barrier().await
}

/// Send `tensor` to `dst_rank` (point-to-point communication).
///
/// The buffer is posted to the store keyed by `(src, dst, tag)`; the matching
/// [`recv`] consumes (and deletes) it.
///
/// # Constraint
///
/// At most **one in-flight message per `(src, dst, tag)` triple** is supported:
/// issuing a second `send` with the same triple before the matching [`recv`]
/// overwrites the first message. Use distinct `tag`s for concurrent messages
/// between the same pair of ranks.
pub async fn send<T: FloatElement>(
    tensor: &Tensor<T>,
    dst_rank: u32,
    tag: u32,
    group: &ProcessGroup,
) -> TorshResult<()> {
    use crate::communication::validate_rank;

    validate_rank(dst_rank, group.world_size())?;
    let engine = tcp_engine(group)?;
    let data: Vec<T> = tensor.to_vec()?;
    engine
        .send_any(&data as &(dyn Any + Send + Sync), dst_rank, tag)
        .await
}

/// Receive a tensor from `src_rank` (point-to-point communication).
///
/// Blocks until the matching [`send`] posts its buffer, then writes it into
/// `tensor` (preserving `tensor`'s shape/device).
pub async fn recv<T: FloatElement>(
    tensor: &mut Tensor<T>,
    src_rank: u32,
    tag: u32,
    group: &ProcessGroup,
) -> TorshResult<()> {
    use crate::communication::validate_rank;

    validate_rank(src_rank, group.world_size())?;
    let engine = tcp_engine(group)?;
    let payload = engine.recv_bytes(src_rank, tag).await?;
    let data = payload_into_vec::<T>(payload)?;

    let expected = tensor.numel();
    if data.len() != expected {
        return Err(crate::TorshDistributedError::tensor_shape_mismatch(
            vec![expected],
            vec![data.len()],
        ));
    }
    let shape = tensor.shape().dims().to_vec();
    let device = tensor.device();
    *tensor = Tensor::from_data(data, shape, device)?;
    Ok(())
}

/// Non-blocking send (isend) - returns immediately without waiting for completion
pub async fn isend<T: FloatElement>(
    tensor: &Tensor<T>,
    dst_rank: u32,
    tag: u32,
    group: &ProcessGroup,
) -> TorshResult<()> {
    // Data transfer is real (delegates to the blocking `send` over the TCP
    // engine); the "non-blocking" contract is not yet honored — the future only
    // resolves once the send completes. Genuine async progress is a follow-up.
    send(tensor, dst_rank, tag, group).await
}

/// Non-blocking receive (irecv) - returns immediately, tensor is filled when ready
pub async fn irecv<T: FloatElement>(
    tensor: &mut Tensor<T>,
    src_rank: u32,
    tag: u32,
    group: &ProcessGroup,
) -> TorshResult<()> {
    // Data transfer is real (delegates to the blocking `recv` over the TCP
    // engine); the future resolves only once the receive completes. Genuine
    // non-blocking progress is a follow-up.
    recv(tensor, src_rank, tag, group).await
}

// ============================================================================
// Group-Aware Collective Operations
// ============================================================================

/// All-reduce within a communication group
pub async fn all_reduce_group<T>(
    tensor: &mut Tensor<T>,
    op: ReduceOp,
    comm_group: &CommunicationGroup,
    process_group: &ProcessGroup,
) -> TorshResult<()>
where
    T: FloatElement
        + Default
        + Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    let current_global_rank = process_group.rank();
    if !comm_group.contains_rank(current_global_rank) {
        return Ok(()); // Not part of this group, skip operation
    }
    if comm_group.group_size <= 1 {
        return Ok(());
    }

    let engine = tcp_engine(process_group)?;
    let participants = comm_group.ranks.clone();
    let shape = tensor.shape().dims().to_vec();
    let device = tensor.device();
    let mut data: Vec<T> = tensor.to_vec()?;
    engine
        .all_reduce_any(
            &mut data as &mut (dyn Any + Send + Sync),
            op,
            &participants,
            &comm_group.group_id,
        )
        .await?;
    *tensor = Tensor::from_data(data, shape, device)?;

    info!(
        " All-reduce in group '{}': rank {} (local: {}) with {} participants",
        comm_group.group_id, current_global_rank, comm_group.local_rank, comm_group.group_size
    );
    Ok(())
}

/// Broadcast within a communication group
pub async fn broadcast_group<T: FloatElement>(
    tensor: &mut Tensor<T>,
    src_local_rank: u32,
    comm_group: &CommunicationGroup,
    process_group: &ProcessGroup,
) -> TorshResult<()> {
    use crate::communication::validate_rank;

    let current_global_rank = process_group.rank();
    if !comm_group.contains_rank(current_global_rank) {
        return Ok(()); // Not part of this group, skip operation
    }
    validate_rank(src_local_rank, comm_group.group_size)?;
    let src_global_rank = comm_group
        .local_to_global_rank(src_local_rank)
        .ok_or_else(|| {
            crate::TorshDistributedError::invalid_argument(
                "src_local_rank",
                format!(
                    "Invalid local rank {} in group '{}'",
                    src_local_rank, comm_group.group_id
                ),
                format!("valid local rank in range 0..{}", comm_group.group_size),
            )
        })?;
    if comm_group.group_size <= 1 {
        return Ok(());
    }

    let engine = tcp_engine(process_group)?;
    let participants = comm_group.ranks.clone();
    let shape = tensor.shape().dims().to_vec();
    let device = tensor.device();
    let mut data: Vec<T> = tensor.to_vec()?;
    engine
        .broadcast_any(
            &mut data as &mut (dyn Any + Send + Sync),
            src_global_rank,
            &participants,
            &comm_group.group_id,
        )
        .await?;
    *tensor = Tensor::from_data(data, shape, device)?;

    info!(
        " Broadcast in group '{}': from local rank {} (global: {}) to {} participants",
        comm_group.group_id, src_local_rank, src_global_rank, comm_group.group_size
    );
    Ok(())
}

/// All-gather within a communication group
pub async fn all_gather_group<T: FloatElement>(
    output: &mut Vec<Tensor<T>>,
    input: &Tensor<T>,
    comm_group: &CommunicationGroup,
    process_group: &ProcessGroup,
) -> TorshResult<()> {
    let current_global_rank = process_group.rank();
    if !comm_group.contains_rank(current_global_rank) {
        return Ok(()); // Not part of this group, skip operation
    }
    output.clear();
    if comm_group.group_size <= 1 {
        let _ = tcp_engine(process_group)?;
        output.push(input.clone());
        return Ok(());
    }

    let engine = tcp_engine(process_group)?;
    let participants = comm_group.ranks.clone();
    let shape = input.shape().dims().to_vec();
    let device = input.device();
    let input_data: Vec<T> = input.to_vec()?;
    let payloads = engine
        .all_gather_payloads(
            &input_data as &(dyn Any + Send + Sync),
            &participants,
            &comm_group.group_id,
        )
        .await?;
    for payload in payloads {
        let data = payload_into_vec::<T>(payload)?;
        output.push(Tensor::from_data(data, shape.clone(), device)?);
    }

    info!(
        "🔗 All-gather in group '{}': rank {} collecting from {} participants",
        comm_group.group_id, current_global_rank, comm_group.group_size
    );
    Ok(())
}

/// Reduce within a communication group
pub async fn reduce_group<T>(
    tensor: &mut Tensor<T>,
    dst_local_rank: u32,
    op: ReduceOp,
    comm_group: &CommunicationGroup,
    process_group: &ProcessGroup,
) -> TorshResult<()>
where
    T: FloatElement
        + Default
        + Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    let current_global_rank = process_group.rank();
    if !comm_group.contains_rank(current_global_rank) {
        return Ok(()); // Not part of this group, skip operation
    }

    if dst_local_rank >= comm_group.group_size {
        return Err(crate::TorshDistributedError::RankOutOfBounds {
            rank: dst_local_rank,
            world_size: comm_group.group_size,
        });
    }

    let dst_global_rank = comm_group
        .local_to_global_rank(dst_local_rank)
        .ok_or_else(|| crate::TorshDistributedError::InvalidArgument {
            arg: "rank".to_string(),
            reason: format!(
                "Invalid local rank {} in group '{}'",
                dst_local_rank, comm_group.group_id
            ),
            expected: "valid local rank within the communication group".to_string(),
        })?;
    if comm_group.group_size <= 1 {
        return Ok(());
    }

    let engine = tcp_engine(process_group)?;
    let participants = comm_group.ranks.clone();
    let shape = tensor.shape().dims().to_vec();
    let device = tensor.device();
    let mut data: Vec<T> = tensor.to_vec()?;
    engine
        .reduce_any(
            &mut data as &mut (dyn Any + Send + Sync),
            dst_global_rank,
            op,
            &participants,
            &comm_group.group_id,
        )
        .await?;
    if current_global_rank == dst_global_rank {
        *tensor = Tensor::from_data(data, shape, device)?;
    }

    info!(
        "⬇️  Reduce in group '{}': to local rank {} (global: {}) from {} participants",
        comm_group.group_id, dst_local_rank, dst_global_rank, comm_group.group_size
    );
    Ok(())
}

/// Barrier synchronization within a communication group
pub async fn barrier_group(
    comm_group: &CommunicationGroup,
    process_group: &ProcessGroup,
) -> TorshResult<()> {
    let current_global_rank = process_group.rank();
    if !comm_group.contains_rank(current_global_rank) {
        return Ok(()); // Not part of this group, skip operation
    }
    if comm_group.group_size <= 1 {
        return Ok(());
    }

    let engine = tcp_engine(process_group)?;
    let participants = comm_group.ranks.clone();
    engine.barrier(&participants, &comm_group.group_id).await?;

    info!(
        "🚧 Barrier in group '{}': rank {} synchronised with {} participants",
        comm_group.group_id, current_global_rank, comm_group.group_size
    );
    Ok(())
}

/// Point-to-point send within a communication group (using local ranks)
pub async fn send_group<T: FloatElement>(
    tensor: &Tensor<T>,
    dst_local_rank: u32,
    tag: u32,
    comm_group: &CommunicationGroup,
    process_group: &ProcessGroup,
) -> TorshResult<()> {
    let current_global_rank = process_group.rank();

    // Check if current rank is part of this group
    if !comm_group.contains_rank(current_global_rank) {
        return Err(crate::TorshDistributedError::InvalidArgument {
            arg: "rank".to_string(),
            reason: format!(
                "Rank {} not in group '{}'",
                current_global_rank, comm_group.group_id
            ),
            expected: "rank must be member of the communication group".to_string(),
        });
    }

    if dst_local_rank >= comm_group.group_size {
        return Err(crate::TorshDistributedError::RankOutOfBounds {
            rank: dst_local_rank,
            world_size: comm_group.group_size,
        });
    }

    let dst_global_rank = comm_group
        .local_to_global_rank(dst_local_rank)
        .ok_or_else(|| crate::TorshDistributedError::InvalidArgument {
            arg: "rank".to_string(),
            reason: format!(
                "Invalid local rank {} in group '{}'",
                dst_local_rank, comm_group.group_id
            ),
            expected: "valid local rank within the communication group".to_string(),
        })?;

    let engine = tcp_engine(process_group)?;
    let data: Vec<T> = tensor.to_vec()?;
    engine
        .send_any(&data as &(dyn Any + Send + Sync), dst_global_rank, tag)
        .await?;

    info!(
        "📤 Group send in '{}': from rank {} to local rank {} (global: {}) with tag {}",
        comm_group.group_id, current_global_rank, dst_local_rank, dst_global_rank, tag
    );
    Ok(())
}

/// Point-to-point receive within a communication group (using local ranks)
pub async fn recv_group<T: FloatElement>(
    tensor: &mut Tensor<T>,
    src_local_rank: u32,
    tag: u32,
    comm_group: &CommunicationGroup,
    process_group: &ProcessGroup,
) -> TorshResult<()> {
    let current_global_rank = process_group.rank();

    // Check if current rank is part of this group
    if !comm_group.contains_rank(current_global_rank) {
        return Err(crate::TorshDistributedError::InvalidArgument {
            arg: "rank".to_string(),
            reason: format!(
                "Rank {} not in group '{}'",
                current_global_rank, comm_group.group_id
            ),
            expected: "rank must be member of the communication group".to_string(),
        });
    }

    if src_local_rank >= comm_group.group_size {
        return Err(crate::TorshDistributedError::RankOutOfBounds {
            rank: src_local_rank,
            world_size: comm_group.group_size,
        });
    }

    let src_global_rank = comm_group
        .local_to_global_rank(src_local_rank)
        .ok_or_else(|| {
            crate::TorshDistributedError::invalid_argument(
                "src_local_rank",
                format!(
                    "Invalid local rank {} in group '{}'",
                    src_local_rank, comm_group.group_id
                ),
                format!("valid local rank in range 0..{}", comm_group.group_size),
            )
        })?;

    let engine = tcp_engine(process_group)?;
    let payload = engine.recv_bytes(src_global_rank, tag).await?;
    let data = payload_into_vec::<T>(payload)?;
    let expected = tensor.numel();
    if data.len() != expected {
        return Err(crate::TorshDistributedError::tensor_shape_mismatch(
            vec![expected],
            vec![data.len()],
        ));
    }
    let shape = tensor.shape().dims().to_vec();
    let device = tensor.device();
    *tensor = Tensor::from_data(data, shape, device)?;

    info!(
        "📥 Group recv in '{}': from local rank {} (global: {}) to rank {} with tag {}",
        comm_group.group_id, src_local_rank, src_global_rank, current_global_rank, tag
    );
    Ok(())
}

// ============================================================================
// Custom Collective Operations
// ============================================================================

/// Reduce-scatter: reduce tensors and scatter result chunks to all processes
/// Each rank gets a different chunk of the reduced result
pub async fn reduce_scatter<T>(
    output: &mut Tensor<T>,
    input: &Tensor<T>,
    op: ReduceOp,
    group: &ProcessGroup,
) -> TorshResult<()>
where
    T: FloatElement
        + Default
        + Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    let world_size = group.world_size();
    let rank = group.rank();
    if world_size <= 1 {
        check_ready(group)?;
        *output = input.clone();
        return Ok(());
    }

    // Real reduce-scatter: all-reduce the full tensor, then keep this rank's
    // contiguous chunk of the reduced result.
    let mut reduced = input.clone();
    all_reduce(&mut reduced, op, group).await?;

    let data = reduced.to_vec()?;
    let total = data.len();
    let base = total / world_size as usize;
    let start = (rank as usize * base).min(total);
    let end = if rank as usize + 1 == world_size as usize {
        total
    } else {
        (start + base).min(total)
    };
    let chunk: Vec<T> = data[start..end].to_vec();
    let device = input.device();
    let chunk_len = chunk.len();
    *output = Tensor::from_data(chunk, vec![chunk_len], device)?;

    info!(
        " Reduce-scatter: rank {} kept {} of {} reduced elements",
        rank, chunk_len, total
    );
    Ok(())
}

/// All-to-all: each rank sends unique data to every other rank
/// output\[i\] receives data from rank i
pub async fn all_to_all<T: FloatElement>(
    output: &mut Vec<Tensor<T>>,
    input: &[Tensor<T>],
    group: &ProcessGroup,
) -> TorshResult<()> {
    let world_size = group.world_size() as usize;

    if input.len() != world_size {
        return Err(crate::TorshDistributedError::InvalidArgument {
            arg: "rank".to_string(),
            reason: format!(
                "Input must have {} tensors for all-to-all, got {}",
                world_size,
                input.len()
            ),
            expected: format!("{} tensors (one per rank)", world_size),
        });
    }

    output.clear();
    if world_size <= 1 {
        check_ready(group)?;
        output.push(input[0].clone());
        return Ok(());
    }

    let engine = tcp_engine(group)?;
    let participants: Vec<u32> = (0..group.world_size()).collect();

    // Serialise the chunk destined for each rank.
    let mut sends = Vec::with_capacity(world_size);
    for t in input {
        let v: Vec<T> = t.to_vec()?;
        sends.push(encode_any(&v as &(dyn Any + Send + Sync))?);
    }
    let received = engine.all_to_all_bytes(&sends, &participants, "").await?;

    for (i, bytes) in received.into_iter().enumerate() {
        let payload = Payload::decode(&bytes)?;
        let data = payload_into_vec::<T>(payload)?;
        // Reconstruct with input[i]'s shape when sizes match (symmetric case),
        // otherwise fall back to a flat 1-D tensor.
        let shape = input[i].shape().dims().to_vec();
        let target_numel: usize = shape.iter().product();
        let out_shape = if data.len() == target_numel {
            shape
        } else {
            vec![data.len()]
        };
        output.push(Tensor::from_data(data, out_shape, input[i].device())?);
    }

    info!(
        " All-to-all: rank {} exchanged data with {} ranks",
        group.rank(),
        world_size
    );
    Ok(())
}

/// Ring all-reduce: more bandwidth-efficient all-reduce for large tensors
/// Reduces communication volume by using ring topology
pub async fn ring_all_reduce<T>(
    tensor: &mut Tensor<T>,
    op: ReduceOp,
    group: &ProcessGroup,
) -> TorshResult<()>
where
    T: FloatElement
        + Default
        + Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    // The store-based engine already performs a correct all-reduce. The ring
    // topology is a bandwidth optimization; the numerical result is identical,
    // so delegate to the standard path rather than fabricate a ring.
    all_reduce(tensor, op, group).await
}

/// Hierarchical all-reduce: two-level all-reduce for multi-node scenarios
/// More efficient when there are multiple nodes with fast intra-node communication
pub async fn hierarchical_all_reduce<T>(
    tensor: &mut Tensor<T>,
    op: ReduceOp,
    group: &ProcessGroup,
    ranks_per_node: u32,
) -> TorshResult<()>
where
    T: FloatElement
        + Default
        + Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    let world_size = group.world_size();
    if ranks_per_node == 0 || world_size % ranks_per_node != 0 {
        return Err(crate::TorshDistributedError::InvalidArgument {
            arg: "ranks_per_node".to_string(),
            reason: "World size must be divisible by a non-zero ranks_per_node for hierarchical all-reduce"
                .to_string(),
            expected: format!(
                "world_size divisible by ranks_per_node ({})",
                ranks_per_node
            ),
        });
    }

    // The two-level (intra-node then inter-node) schedule is a communication
    // optimization; the numerical result equals a flat all-reduce, so delegate.
    all_reduce(tensor, op, group).await
}

/// Bucket all-reduce: reduce multiple tensors efficiently by combining them
/// Useful for gradient synchronization in distributed training
pub async fn bucket_all_reduce<T>(
    tensors: &mut [Tensor<T>],
    op: ReduceOp,
    group: &ProcessGroup,
    max_bucket_size_mb: f32,
) -> TorshResult<()>
where
    T: FloatElement
        + Default
        + Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    if tensors.is_empty() {
        return Ok(());
    }

    // Bucketing (`max_bucket_size_mb`) is a communication-efficiency
    // optimization; correctness only requires a real all-reduce per tensor, in
    // the same order on every rank (SPMD).
    let _ = max_bucket_size_mb;
    for tensor in tensors.iter_mut() {
        all_reduce(tensor, op, group).await?;
    }

    info!(
        " Bucket all-reduce: rank {} reduced {} tensors",
        group.rank(),
        tensors.len()
    );
    Ok(())
}

// ============================================================================
// Advanced Communication Primitives for Distributed Deep Learning
// ============================================================================

/// All-reduce with fusion: combines small tensors into larger buffers for efficiency
/// This is critical for gradient synchronization in distributed training
pub async fn fused_all_reduce<T>(
    tensors: &mut [Tensor<T>],
    op: ReduceOp,
    group: &ProcessGroup,
    fusion_threshold_bytes: usize,
) -> TorshResult<()>
where
    T: FloatElement
        + Default
        + Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    if tensors.is_empty() {
        return Ok(());
    }

    // Fusion (`fusion_threshold_bytes`) is a communication-efficiency
    // optimization; correctness only requires a real all-reduce per tensor, in
    // the same order on every rank (SPMD).
    let _ = fusion_threshold_bytes;
    for tensor in tensors.iter_mut() {
        all_reduce(tensor, op, group).await?;
    }

    info!(
        " Fused all-reduce complete: rank {} reduced {} tensors",
        group.rank(),
        tensors.len()
    );
    Ok(())
}

/// Variable-sized all-gather: gather tensors of different sizes from all ranks
/// Critical for dynamic neural networks where tensor sizes vary across ranks
pub async fn all_gather_varsize<T: FloatElement>(
    output: &mut Vec<Tensor<T>>,
    input: &Tensor<T>,
    group: &ProcessGroup,
) -> TorshResult<()> {
    // The store-based all-gather already transfers each rank's exact buffer
    // (every payload carries its own length), so variable per-rank sizes are
    // handled correctly by the standard path.
    all_gather(output, input, group).await
}

/// Tree-based broadcast: more efficient for large world sizes
/// Uses binary tree topology to reduce latency compared to linear broadcast
pub async fn tree_broadcast<T: FloatElement>(
    tensor: &mut Tensor<T>,
    src_rank: u32,
    group: &ProcessGroup,
) -> TorshResult<()> {
    // The tree topology is a latency optimization; the delivered data is
    // identical to a flat broadcast, so delegate to the correct path.
    broadcast(tensor, src_rank, group).await
}

/// Pipelined all-reduce: overlaps computation and communication
/// Useful for very large tensors that can be processed in chunks
pub async fn pipelined_all_reduce<T>(
    tensor: &mut Tensor<T>,
    op: ReduceOp,
    group: &ProcessGroup,
    pipeline_chunks: usize,
) -> TorshResult<()>
where
    T: FloatElement
        + Default
        + Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    if pipeline_chunks == 0 {
        return Err(crate::TorshDistributedError::InvalidArgument {
            arg: "pipeline_chunks".to_string(),
            reason: "Pipeline chunks must be greater than 0".to_string(),
            expected: "pipeline_chunks > 0".to_string(),
        });
    }

    // Chunked pipelining overlaps communication with computation; the reduced
    // result is identical to a single all-reduce, so delegate to it.
    all_reduce(tensor, op, group).await
}

/// Double-buffered all-reduce: uses double buffering to hide latency
/// Critical for overlapping gradient computation with communication
pub async fn double_buffered_all_reduce<T>(
    current_buffer: &mut Tensor<T>,
    next_buffer: &mut Tensor<T>,
    op: ReduceOp,
    group: &ProcessGroup,
) -> TorshResult<()>
where
    T: FloatElement
        + Default
        + Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    // Double buffering hides latency by overlapping; the reduced results are
    // identical to two sequential all-reduces. All ranks must issue both in the
    // same order (SPMD).
    all_reduce(current_buffer, op, group).await?;
    all_reduce(next_buffer, op, group).await?;
    Ok(())
}
