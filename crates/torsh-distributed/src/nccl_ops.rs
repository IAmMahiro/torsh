//! NCCL-oriented collective operation wrappers.
//!
//! These wrappers expose a NCCL-style API surface but, because no real
//! NVIDIA NCCL / GPU transport is wired in yet, they delegate to the crate's
//! generic (and now genuinely functional) collective implementations in
//! [`crate::collectives`], which perform real cross-rank communication over the
//! pure-Rust TCP backend.
//!
//! There is deliberately **no fabricated GPU simulation** here: the previous
//! implementation invented reduction/gather results (e.g. multiplying by
//! `world_size`), which silently corrupted data. When a real NCCL backend
//! (via future `cudarc`+`nccl` / `oxicuda-comm` bindings) becomes available,
//! these wrappers should dispatch to it for `BackendType::Nccl` process groups.
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use torsh_distributed::nccl_ops::nccl_all_reduce;
//! use torsh_distributed::{ReduceOp, init_process_group, BackendType};
//! use torsh_tensor::Tensor;
//!
//! async fn example() -> Result<(), Box<dyn std::error::Error>> {
//!     let pg = init_process_group(BackendType::Gloo, 0, 1, "127.0.0.1", 29500).await?;
//!     let mut tensor: Tensor<f32> = Tensor::from_vec(vec![1.0; 1000], &[1000])?;
//!     nccl_all_reduce(&mut tensor, ReduceOp::Sum, &pg).await?;
//!     Ok(())
//! }
//! ```

use crate::{ProcessGroup, ReduceOp, TorshDistributedError, TorshResult};
use torsh_core::dtype::FloatElement;
use torsh_tensor::Tensor;

/// NCCL-style all-reduce.
///
/// Delegates to the real [`crate::collectives::all_reduce`], performing a
/// genuine cross-rank reduction (no fabricated results).
pub async fn nccl_all_reduce<T>(
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
    crate::collectives::all_reduce(tensor, op, group).await
}

/// NCCL-style broadcast. Delegates to the real [`crate::collectives::broadcast`].
pub async fn nccl_broadcast<T: FloatElement>(
    tensor: &mut Tensor<T>,
    src_rank: u32,
    group: &ProcessGroup,
) -> TorshResult<()> {
    crate::collectives::broadcast(tensor, src_rank, group).await
}

/// NCCL-style reduce-scatter.
///
/// Delegates to the real [`crate::collectives::reduce_scatter`], which performs
/// a full all-reduce followed by keeping this rank's contiguous chunk.
pub async fn nccl_reduce_scatter<T: FloatElement + Default + Copy>(
    input: &Tensor<T>,
    output: &mut Tensor<T>,
    op: ReduceOp,
    group: &ProcessGroup,
) -> TorshResult<()> {
    crate::collectives::reduce_scatter(output, input, op, group).await
}

/// NCCL-style all-gather. Delegates to the real [`crate::collectives::all_gather`].
pub async fn nccl_all_gather<T: FloatElement>(
    input: &Tensor<T>,
    output: &mut Vec<Tensor<T>>,
    group: &ProcessGroup,
) -> TorshResult<()> {
    crate::collectives::all_gather(output, input, group).await
}

/// Batch of collective operations.
///
/// The batched fast-path (`ncclGroupStart`/`ncclGroupEnd`) requires a real NCCL
/// backend, which is not yet available; [`NcclBatch::execute`] therefore returns
/// an honest error rather than pretending to execute the batch.
pub struct NcclBatch {
    operations: Vec<NcclOperation>,
}

// Fields are retained to describe queued operations for a future real batched
// NCCL implementation; `execute` currently returns an honest error without
// reading them, so they are intentionally unread for now.
#[derive(Debug)]
#[allow(dead_code)]
enum NcclOperation {
    AllReduce {
        tensor_id: usize,
        op: ReduceOp,
    },
    Broadcast {
        tensor_id: usize,
        src_rank: u32,
    },
    ReduceScatter {
        input_id: usize,
        output_id: usize,
        op: ReduceOp,
    },
}

impl NcclBatch {
    /// Create a new batch.
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    /// Add an all-reduce operation to the batch.
    pub fn all_reduce(&mut self, tensor_id: usize, op: ReduceOp) -> &mut Self {
        self.operations
            .push(NcclOperation::AllReduce { tensor_id, op });
        self
    }

    /// Add a broadcast operation to the batch.
    pub fn broadcast(&mut self, tensor_id: usize, src_rank: u32) -> &mut Self {
        self.operations.push(NcclOperation::Broadcast {
            tensor_id,
            src_rank,
        });
        self
    }

    /// Add a reduce-scatter operation to the batch.
    pub fn reduce_scatter(&mut self, input_id: usize, output_id: usize, op: ReduceOp) -> &mut Self {
        self.operations.push(NcclOperation::ReduceScatter {
            input_id,
            output_id,
            op,
        });
        self
    }

    /// Number of operations queued in the batch.
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Whether the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Execute all operations in the batch.
    ///
    /// Returns an honest error: grouped NCCL execution requires a real
    /// `cudarc`+`nccl` backend that is not yet available. Callers should issue
    /// the individual collectives via [`crate::collectives`] instead.
    pub async fn execute(&self, _group: &ProcessGroup) -> TorshResult<()> {
        Err(TorshDistributedError::feature_not_available(
            "batched NCCL group execution",
            "a real cudarc+nccl backend (issue individual collectives instead)",
        ))
    }
}

impl Default for NcclBatch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{init_process_group, BackendType};
    use torsh_tensor::Tensor;

    #[tokio::test]
    async fn test_nccl_all_reduce_single_rank_identity() {
        // With the real backend, a single-rank all-reduce is the identity.
        let pg = init_process_group(BackendType::Gloo, 0, 1, "127.0.0.1", 29610)
            .await
            .expect("init");

        let mut tensor: Tensor<f32> = Tensor::from_vec(vec![3.0; 10], &[10]).expect("tensor");
        nccl_all_reduce(&mut tensor, ReduceOp::Sum, &pg)
            .await
            .expect("all_reduce");
        assert_eq!(tensor.to_vec().expect("to_vec"), vec![3.0; 10]);
    }

    #[tokio::test]
    async fn test_nccl_broadcast_single_rank() {
        let pg = init_process_group(BackendType::Gloo, 0, 1, "127.0.0.1", 29611)
            .await
            .expect("init");

        let mut tensor: Tensor<f32> = Tensor::from_vec(vec![0.0; 10], &[10]).expect("tensor");
        nccl_broadcast(&mut tensor, 0, &pg)
            .await
            .expect("broadcast");
    }

    #[tokio::test]
    async fn test_nccl_batch_execute_is_honest_error() {
        let pg = init_process_group(BackendType::Gloo, 0, 1, "127.0.0.1", 29612)
            .await
            .expect("init");

        let mut batch = NcclBatch::new();
        batch
            .all_reduce(0, ReduceOp::Sum)
            .broadcast(1, 0)
            .reduce_scatter(2, 3, ReduceOp::Sum);
        assert_eq!(batch.len(), 3);

        // Honest: batched NCCL execution is not available without a real backend.
        let result = batch.execute(&pg).await;
        assert!(result.is_err());
    }
}
