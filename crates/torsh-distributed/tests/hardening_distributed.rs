//! Wave-3 hardening tests for `torsh-distributed`.
//!
//! These tests prove that distributed collectives are now HONEST and perform
//! REAL cross-rank communication over the pure-Rust TCP backend, rather than
//! fabricating results or silently no-op'ing (findings F009-F015, F103-F106).
//!
//! Multi-rank tests spawn one OS thread per rank, each with its own tokio
//! runtime and process group, exchanging data through real 127.0.0.1 TCP
//! sockets. Rank 0 runs the store server; its runtime is deliberately kept
//! alive until every worker thread has joined so the server never disappears
//! mid-collective.

use torsh_distributed::backend::{BackendType, ReduceOp};
use torsh_distributed::collectives::{
    all_gather, all_reduce, barrier, broadcast, recv, reduce, scatter, send,
};
use torsh_distributed::{init_process_group, is_available, is_gloo_available, ProcessGroup};
use torsh_tensor::Tensor;

/// Run `f` on every rank of a `world_size`-rank Gloo process group, each on its
/// own thread + runtime, and return the per-rank results in rank order.
///
/// Rank 0's runtime AND its process group (which together host and own the
/// store server) are kept alive until all worker threads have joined, so the
/// server never disappears mid-collective — even after rank 0 has completed its
/// own part of the collective.
fn run_ranks<F, Fut, T>(world_size: u32, port: u16, f: F) -> Vec<T>
where
    F: Fn(u32, std::sync::Arc<ProcessGroup>) -> Fut + Clone + Send + 'static,
    Fut: std::future::Future<Output = T>,
    T: Send + 'static,
{
    use std::sync::Arc;
    use std::thread;

    // Rank 0 runs in this runtime; keep it alive until workers join.
    let rt0 = tokio::runtime::Runtime::new().expect("rank0 runtime");

    let mut handles = Vec::new();
    for rank in 1..world_size {
        let f = f.clone();
        handles.push(thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().expect("worker runtime");
            rt.block_on(async move {
                let pg = Arc::new(
                    init_process_group(BackendType::Gloo, rank, world_size, "127.0.0.1", port)
                        .await
                        .expect("init worker process group"),
                );
                f(rank, pg).await
            })
        }));
    }

    // Retain rank 0's process group (`_pg0`) until after every worker joins.
    let (r0, _pg0) = rt0.block_on(async {
        let pg = Arc::new(
            init_process_group(BackendType::Gloo, 0, world_size, "127.0.0.1", port)
                .await
                .expect("init rank0 process group"),
        );
        let result = f(0, pg.clone()).await;
        (result, pg)
    });

    let mut out: Vec<(u32, T)> = vec![(0, r0)];
    for (i, h) in handles.into_iter().enumerate() {
        out.push(((i + 1) as u32, h.join().expect("worker thread panicked")));
    }
    // Now safe to tear down the server: everyone has finished reading.
    drop(_pg0);
    drop(rt0);

    out.sort_by_key(|(r, _)| *r);
    out.into_iter().map(|(_, v)| v).collect()
}

// ---------------------------------------------------------------------------
// F009 / F010 / F012 / F014: collectives really reduce across ranks.
// If any collective were still a no-op or fabricated, these exact-value
// assertions would fail.
// ---------------------------------------------------------------------------

#[test]
fn real_all_reduce_sum_across_three_ranks() {
    // rank r contributes [(r+1); 4]; the true sum is 1+2+3 = 6.
    let results = run_ranks(3, 29700, |rank, pg| async move {
        let mut t = Tensor::<f32>::from_vec(vec![(rank + 1) as f32; 4], &[4]).expect("tensor");
        all_reduce(&mut t, ReduceOp::Sum, &pg)
            .await
            .expect("all_reduce");
        t.to_vec().expect("to_vec")
    });
    for r in &results {
        assert_eq!(r, &vec![6.0f32; 4], "every rank must see the true sum");
    }
}

#[test]
fn real_all_reduce_mean_is_average() {
    // Mean of {1,2,3} = 2. This is exactly the DDP averaging path (F013):
    // a genuine cross-rank sum, then divide by world_size.
    let results = run_ranks(3, 29701, |rank, pg| async move {
        let mut t = Tensor::<f32>::from_vec(vec![(rank + 1) as f32; 2], &[2]).expect("tensor");
        all_reduce(&mut t, ReduceOp::Mean, &pg)
            .await
            .expect("all_reduce");
        t.to_vec().expect("to_vec")
    });
    for r in &results {
        assert_eq!(r, &vec![2.0f32; 2]);
    }
}

#[test]
fn real_all_reduce_max_and_min() {
    let maxes = run_ranks(3, 29702, |rank, pg| async move {
        let mut t = Tensor::<f32>::from_vec(vec![(rank + 1) as f32; 2], &[2]).expect("tensor");
        all_reduce(&mut t, ReduceOp::Max, &pg).await.expect("max");
        t.to_vec().expect("to_vec")
    });
    for r in &maxes {
        assert_eq!(r, &vec![3.0f32; 2]);
    }

    let mins = run_ranks(3, 29703, |rank, pg| async move {
        let mut t = Tensor::<f32>::from_vec(vec![(rank + 1) as f32; 2], &[2]).expect("tensor");
        all_reduce(&mut t, ReduceOp::Min, &pg).await.expect("min");
        t.to_vec().expect("to_vec")
    });
    for r in &mins {
        assert_eq!(r, &vec![1.0f32; 2]);
    }
}

#[test]
fn real_all_reduce_product() {
    // 1 * 2 * 3 = 6.
    let results = run_ranks(3, 29704, |rank, pg| async move {
        let mut t = Tensor::<f32>::from_vec(vec![(rank + 1) as f32; 2], &[2]).expect("tensor");
        all_reduce(&mut t, ReduceOp::Product, &pg)
            .await
            .expect("product");
        t.to_vec().expect("to_vec")
    });
    for r in &results {
        assert_eq!(r, &vec![6.0f32; 2]);
    }
}

#[test]
fn real_all_reduce_rejects_bitwise_on_floats() {
    // Bitwise reductions are undefined for floating point and must error, not
    // silently succeed.
    let results = run_ranks(2, 29705, |_rank, pg| async move {
        let mut t = Tensor::<f32>::from_vec(vec![1.0; 2], &[2]).expect("tensor");
        all_reduce(&mut t, ReduceOp::Band, &pg).await.is_err()
    });
    assert!(results.iter().all(|&is_err| is_err));
}

#[test]
fn real_broadcast_delivers_root_data() {
    // Root (rank 0) holds [7;4]; other ranks start at [0;4] and must receive it.
    let results = run_ranks(3, 29706, |rank, pg| async move {
        let init = if rank == 0 { 7.0 } else { 0.0 };
        let mut t = Tensor::<f32>::from_vec(vec![init; 4], &[4]).expect("tensor");
        broadcast(&mut t, 0, &pg).await.expect("broadcast");
        t.to_vec().expect("to_vec")
    });
    for r in &results {
        assert_eq!(r, &vec![7.0f32; 4], "all ranks must receive root's data");
    }
}

#[test]
fn real_all_gather_collects_every_rank() {
    // rank r contributes [r]; every rank must end with [[0],[1],[2]].
    let results = run_ranks(3, 29707, |rank, pg| async move {
        let input = Tensor::<f32>::from_vec(vec![rank as f32], &[1]).expect("tensor");
        let mut out: Vec<Tensor<f32>> = Vec::new();
        all_gather(&mut out, &input, &pg).await.expect("all_gather");
        out.iter()
            .map(|t| t.to_vec().expect("to_vec")[0])
            .collect::<Vec<f32>>()
    });
    for r in &results {
        assert_eq!(r, &vec![0.0f32, 1.0, 2.0]);
    }
}

#[test]
fn real_reduce_to_destination_only() {
    // Sum reduce to rank 1: rank 1 gets 6, others keep their own value.
    let results = run_ranks(3, 29708, |rank, pg| async move {
        let mut t = Tensor::<f32>::from_vec(vec![(rank + 1) as f32], &[1]).expect("tensor");
        reduce(&mut t, 1, ReduceOp::Sum, &pg).await.expect("reduce");
        t.to_vec().expect("to_vec")[0]
    });
    assert_eq!(results[0], 1.0, "rank 0 unchanged");
    assert_eq!(results[1], 6.0, "rank 1 holds the reduced sum");
    assert_eq!(results[2], 3.0, "rank 2 unchanged");
}

#[test]
fn real_scatter_distributes_chunks() {
    // Root scatters [[10],[20],[30]]; rank r receives (r+1)*10.
    let results = run_ranks(3, 29709, |rank, pg| async move {
        let mut out = Tensor::<f32>::from_vec(vec![0.0], &[1]).expect("out");
        let chunks: Vec<Tensor<f32>> = (0..3)
            .map(|i| Tensor::<f32>::from_vec(vec![((i + 1) * 10) as f32], &[1]).expect("chunk"))
            .collect();
        let input = if rank == 0 {
            Some(chunks.as_slice())
        } else {
            None
        };
        scatter(&mut out, input, 0, &pg).await.expect("scatter");
        out.to_vec().expect("to_vec")[0]
    });
    assert_eq!(results, vec![10.0, 20.0, 30.0]);
}

#[test]
fn real_barrier_synchronizes() {
    let results = run_ranks(
        3,
        29710,
        |_rank, pg| async move { barrier(&pg).await.is_ok() },
    );
    assert!(results.iter().all(|&ok| ok));
}

#[test]
fn real_send_recv_point_to_point() {
    // rank 0 sends [42;3] to rank 1; rank 1 receives it.
    let results = run_ranks(2, 29711, |rank, pg| async move {
        if rank == 0 {
            let t = Tensor::<f32>::from_vec(vec![42.0; 3], &[3]).expect("tensor");
            send(&t, 1, 7, &pg).await.expect("send");
            vec![]
        } else {
            let mut t = Tensor::<f32>::from_vec(vec![0.0; 3], &[3]).expect("tensor");
            recv(&mut t, 0, 7, &pg).await.expect("recv");
            t.to_vec().expect("to_vec")
        }
    });
    assert_eq!(results[1], vec![42.0f32; 3]);
}

// ---------------------------------------------------------------------------
// F013: DDP-style averaging is correct because the underlying all-reduce
// genuinely sums. (Previously the sum was a no-op but the /world_size division
// still ran, scaling every gradient to 1/N with no synchronisation.)
// ---------------------------------------------------------------------------

#[test]
fn ddp_style_gradient_averaging_is_correct() {
    // Each rank's "gradient" is [(r+1); 3]. DDP does all_reduce(Sum) then
    // div_scalar(world_size). The correct averaged gradient is (1+2+3)/3 = 2.
    let results = run_ranks(3, 29712, |rank, pg| async move {
        let mut grad = Tensor::<f32>::from_vec(vec![(rank + 1) as f32; 3], &[3]).expect("grad");
        all_reduce(&mut grad, ReduceOp::Sum, &pg)
            .await
            .expect("all_reduce");
        let averaged = grad.div_scalar(3.0).expect("div_scalar");
        averaged.to_vec().expect("to_vec")
    });
    for r in &results {
        assert_eq!(
            r,
            &vec![2.0f32; 3],
            "averaged gradient must be the true cross-rank mean"
        );
    }
}

// ---------------------------------------------------------------------------
// world_size == 1: collectives are the identity and never touch the network.
// This is the correct trivial case (NOT fabricated) and must not hang or bind.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn single_rank_all_reduce_is_identity() {
    let pg = init_process_group(BackendType::Gloo, 0, 1, "127.0.0.1", 29713)
        .await
        .expect("init");
    let mut t = Tensor::<f32>::from_vec(vec![5.0, -1.0, 2.5], &[3]).expect("tensor");
    all_reduce(&mut t, ReduceOp::Sum, &pg)
        .await
        .expect("all_reduce");
    assert_eq!(t.to_vec().expect("to_vec"), vec![5.0, -1.0, 2.5]);
}

// ---------------------------------------------------------------------------
// F011 / F015: create_backend is honest — no mock substitution, no identity
// spoofing.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn nccl_backend_is_honestly_unavailable() {
    // No real NCCL transport exists, so init must error rather than hand back a
    // mock that fabricates collectives.
    let r = init_process_group(BackendType::Nccl, 0, 1, "127.0.0.1", 29714).await;
    assert!(r.is_err(), "NCCL must be an honest error, not a mock");
}

#[tokio::test]
async fn custom_backend_is_honestly_unavailable() {
    let r = init_process_group(
        BackendType::Custom("does-not-exist"),
        0,
        1,
        "127.0.0.1",
        29715,
    )
    .await;
    assert!(r.is_err());
}

#[tokio::test]
async fn gloo_reports_its_true_identity() {
    let pg = init_process_group(BackendType::Gloo, 0, 1, "127.0.0.1", 29716)
        .await
        .expect("init");
    // Not spoofed: a Gloo group reports Gloo.
    assert_eq!(pg.backend_type(), BackendType::Gloo);
    assert!(is_available());
    assert!(is_gloo_available());
}
