//! Tests for collective operations.
//!
//! These exercise the `world_size == 1` path, where every collective is a
//! correct identity (no network I/O). REAL multi-rank collectives (genuine
//! cross-rank reductions over TCP) are covered by `hardening_distributed.rs`.
//!
//! Historically these tests ran against a mock backend with `world_size = 4`
//! and asserted that tensors were left UNCHANGED — i.e. they asserted the
//! fabricated no-op. With honest collectives, a single-process `world_size > 1`
//! collective returns an error instead of silently succeeding, so these tests
//! use `world_size = 1`, where the same "unchanged" assertions hold for a real
//! reason (a single-rank reduction is genuinely the identity).

use torsh_core::Result;
use torsh_distributed::{
    backend::BackendType,
    backend::ReduceOp,
    collectives::{all_gather, all_reduce, barrier, broadcast, reduce, scatter},
    init_process_group,
};
use torsh_tensor::creation::{eye, full, ones, zeros};
use torsh_tensor::Tensor;

#[tokio::test]
async fn test_all_reduce_single_rank_identity() -> Result<()> {
    let pg = init_process_group(BackendType::Gloo, 0, 1, "127.0.0.1", 29520).await?;

    let mut tensor = ones::<f32>(&[2, 3])?;
    all_reduce(&mut tensor, ReduceOp::Sum, &pg).await?;

    // Single-rank sum is the identity (correct, not fabricated).
    let expected = ones::<f32>(&[2, 3])?;
    let data = tensor.to_vec()?;
    let expected_data = expected.to_vec()?;
    assert_eq!(data.len(), expected_data.len());
    for (a, b) in data.iter().zip(expected_data.iter()) {
        assert!((a - b).abs() < 1e-6_f32);
    }

    Ok(())
}

#[tokio::test]
async fn test_broadcast_single_rank() -> Result<()> {
    let pg = init_process_group(BackendType::Gloo, 0, 1, "127.0.0.1", 29521).await?;

    let rank = pg.rank() as f32;
    let mut tensor = full::<f32>(&[3, 3], rank)?;
    broadcast(&mut tensor, 0, &pg).await?;

    // Root's own tensor is unchanged.
    let expected = full::<f32>(&[3, 3], rank)?;
    let data = tensor.to_vec()?;
    let expected_data = expected.to_vec()?;
    assert_eq!(data, expected_data);

    Ok(())
}

#[tokio::test]
async fn test_all_gather_single_rank() -> Result<()> {
    let pg = init_process_group(BackendType::Gloo, 0, 1, "127.0.0.1", 29522).await?;

    let input = eye::<f32>(3)?;
    let mut output = Vec::new();
    all_gather(&mut output, &input, &pg).await?;

    // One rank -> one gathered tensor, equal to the input.
    assert_eq!(output.len(), 1);
    let input_data = input.to_vec()?;
    for tensor in &output {
        assert_eq!(tensor.to_vec()?, input_data);
    }

    Ok(())
}

#[tokio::test]
async fn test_reduce_single_rank_identity() -> Result<()> {
    let pg = init_process_group(BackendType::Gloo, 0, 1, "127.0.0.1", 29523).await?;

    let mut tensor = full::<f32>(&[2, 2], 2.0)?;
    reduce(&mut tensor, 0, ReduceOp::Sum, &pg).await?;

    // Single-rank reduce leaves the destination holding its own values.
    let expected = full::<f32>(&[2, 2], 2.0)?;
    let data: Vec<f32> = tensor.to_vec()?;
    let expected_data = expected.to_vec()?;
    for (a, b) in data.iter().zip(expected_data.iter()) {
        assert!(
            (a - b).abs() < 1e-6_f32,
            "Values don't match: {} vs {}",
            a,
            b
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_scatter_single_rank() -> Result<()> {
    let pg = init_process_group(BackendType::Gloo, 0, 1, "127.0.0.1", 29524).await?;

    // One rank -> exactly one chunk.
    let tensors: Vec<Tensor<f32>> = vec![full(&[2, 2], 0.0)?];
    let mut output = zeros::<f32>(&[2, 2])?;
    scatter(&mut output, Some(&tensors), 0, &pg).await?;

    let expected = zeros::<f32>(&[2, 2])?;
    assert_eq!(output.to_vec()?, expected.to_vec()?);

    Ok(())
}

#[tokio::test]
async fn test_barrier_single_rank() -> Result<()> {
    let pg = init_process_group(BackendType::Gloo, 0, 1, "127.0.0.1", 29525).await?;
    barrier(&pg).await?;
    Ok(())
}

#[test]
fn test_reduce_ops() {
    let ops = vec![
        ReduceOp::Sum,
        ReduceOp::Product,
        ReduceOp::Min,
        ReduceOp::Max,
        ReduceOp::Band,
        ReduceOp::Bor,
        ReduceOp::Bxor,
    ];

    for op in ops {
        assert_eq!(op, op);
    }
}

#[test]
fn test_backend_availability() {
    use torsh_distributed::{is_available, is_mpi_available, is_nccl_available};

    assert!(is_available());

    #[cfg(feature = "mpi")]
    assert!(is_mpi_available());

    #[cfg(not(feature = "mpi"))]
    assert!(!is_mpi_available());

    #[cfg(feature = "nccl")]
    assert!(is_nccl_available());

    #[cfg(not(feature = "nccl"))]
    assert!(!is_nccl_available());
}
