# torsh-distributed

Distributed training support for ToRSh with PyTorch-compatible API.

**Status**: Partial — real MPI collectives (including a Universe-lifetime-correct `barrier()`) and
tensor-parallel all-gather are implemented and tested, but the high-level `ProcessGroup`
convenience API currently always runs against an in-process mock backend, and NCCL/Gloo remain
unimplemented (see "Backends" below). **Tests**: 388 passing, 6 skipped (`cargo nextest run
--all-features`).

## Overview

This crate provides distributed and parallel training capabilities including:

- **Data Parallel Training**: DistributedDataParallel (DDP)
- **Model Parallel Training**: Pipeline and tensor parallelism
- **Communication Backends**: NCCL, Gloo, MPI support
- **RPC Framework**: Remote procedure calls for distributed computing
- **Collective Operations**: All-reduce, broadcast, gather, scatter

## Usage

### Basic Distributed Training

> **Note:** `init_process_group`/`ProcessGroup` (the convenience API below) currently construct an
> in-process `MockBackend` for **every** requested `BackendType` (`Nccl`, `Mpi`, and `Gloo` all
> resolve to the mock today) — useful for testing training loops without a real cluster, but it is
> not yet wired up to real networking. Genuine MPI collective communication (including a real,
> Universe-lifetime-correct `barrier()`) is implemented on `torsh_distributed::backend::MpiBackend`
> (behind the `mpi` feature), but must currently be constructed and driven directly rather than
> through `ProcessGroup`. See "Backends" below for the accurate per-backend status.

```rust
use torsh_distributed::prelude::*;
use torsh_nn::prelude::*;

// rank/world_size are typically supplied by the launcher via RANK / WORLD_SIZE env vars
let rank: Rank = std::env::var("RANK")?.parse()?;
let world_size: WorldSize = std::env::var("WORLD_SIZE")?.parse()?;

// Initialize process group (backend, rank, world_size, master_addr, master_port)
let process_group = std::sync::Arc::new(
    init_process_group(BackendType::Nccl, rank, world_size, "localhost", 29500).await?,
);

// Get rank and world size back from the process group
let rank = process_group.rank();
let world_size = process_group.world_size();

// Create model and wrap with DDP
let model = create_model();
let ddp_model = DistributedDataParallel::new(
    model,
    process_group,
    vec![rank as usize],       // device_ids
    Some(rank as usize),       // output_device
    false,                     // broadcast_buffers
    25.0,                      // bucket_cap_mb
)?;

// Distributed optimizer
let optimizer = DistributedOptimizer::new(
    SGD::new(ddp_model.parameters(), 0.1, None, None, None, false),
)?;

// Training loop
for epoch in 0..num_epochs {
    for batch in dataloader {
        let output = ddp_model.forward(&batch.input)?;
        let loss = compute_loss(&output, &batch.target)?;
        
        loss.backward()?;
        optimizer.step()?;
        optimizer.zero_grad();
    }
}

// Cleanup
destroy_process_group()?;
```

### Collective Operations

> **Note:** these are `async fn`s that take a `&ProcessGroup` (so they need `.await` plus a group
> obtained from `init_process_group`). More importantly, as of this release `all_reduce`,
> `broadcast`, `reduce`, `send`, and `recv` all have explicitly-commented **mock** bodies that do
> not perform real cross-process communication (e.g. `all_reduce` never touches the tensor at all;
> `all_gather` just clones the local input `world_size` times). This matches `ProcessGroup` always
> using `MockBackend` today (see "Backends" above) — useful for exercising training-loop control
> flow, but not yet real distributed communication.

```rust
use torsh_distributed::collectives::*;

// All-reduce: sum tensors across all processes
let mut tensor = create_tensor();
all_reduce(&mut tensor, ReduceOp::Sum, &process_group).await?;

// Broadcast: send tensor from rank 0 to all others
broadcast(&mut tensor, 0, &process_group).await?;

// Gather: collect tensors from all ranks
let mut gathered = Vec::new();
all_gather(&mut gathered, &tensor, &process_group).await?;

// Scatter: distribute chunks to different ranks
let mut chunk = create_tensor();
scatter(&mut chunk, Some(&all_chunks), 0, &process_group).await?;

// Reduce: aggregate to specific rank
reduce(&mut tensor, 0, ReduceOp::Sum, &process_group).await?;
```

### RPC Framework

```rust
use torsh_distributed::rpc::*;

// Initialize RPC
init_rpc(
    "worker",
    rank,
    world_size,
    Some(RpcBackendOptions::default()),
)?;

// Remote procedure call
let future = rpc_async(
    "worker1",
    "process_data",
    &[tensor.clone()],
)?;

// Get result
let result = future.wait()?;

// Remote reference
let rref = remote(&tensor, "worker2")?;
let local_value = rref.to_here()?;

// Shutdown RPC
shutdown_rpc()?;
```

### Pipeline Parallelism

```rust
use torsh_distributed::pipeline::*;

// Split model into stages
let stages = vec![
    stage1_layers,
    stage2_layers,
    stage3_layers,
    stage4_layers,
];

// Create pipeline
let pipeline = PipelineParallel::new(
    stages,
    num_microbatches,
    device_placement,
)?;

// Forward with micro-batching
let output = pipeline.forward(input)?;
```

### Tensor Parallelism

```rust
use torsh_distributed::{TensorParallel, TensorParallelConfig, TensorParallelLayer};

// `tp_layer` wraps a module (Box<dyn Module>) and shards its parameters across
// the tensor-parallel process group according to `TensorParallelConfig`/`TensorParallelLayer`.
let tp_layer = TensorParallel::new(module, tp_process_group, tp_config, layer_info)?;

// Gather sharded output back into the full, unsharded tensor. `parallel_all_gather`
// concatenates every rank's shard along `shard_dim` (requires the `scirs2-memory`
// feature) — earlier versions of this function silently discarded all but one
// shard; it now correctly reconstructs the full tensor.
let shard_dim = 0;
let full_tensor = tp_layer.parallel_all_gather(&local_shard, shard_dim).await?;
```

### Gradient Compression

```rust
use torsh_distributed::compression::*;

// Configure gradient compression
let compressor = GradientCompressor::new()
    .algorithm(CompressionAlgorithm::TopK(0.1))
    .memory(CompressorMemory::Residual);

// Apply to DDP
let ddp_model = DistributedDataParallel::new(model, ...)
    .with_compression(compressor)?;
```

### Fault Tolerance

```rust
use torsh_distributed::elastic::*;

// Elastic training with dynamic workers
let elastic_agent = ElasticAgent::new()
    .min_workers(2)
    .max_workers(8)
    .checkpoint_dir("./checkpoints");

elastic_agent.run(train_fn)?;
```

### Monitoring

```rust
use torsh_distributed::monitoring::*;

// Track distributed metrics
let monitor = DistributedMonitor::new();

// Log communication time
monitor.log_comm_time("all_reduce", duration);

// Get statistics
let stats = monitor.get_stats();
println!("Total communication time: {:?}", stats.total_comm_time);
```

## Backends

### NCCL (NVIDIA GPUs) — mocked, not yet functional
- The `nccl` feature builds a `NcclBackend` that is explicitly documented as using **mock
  implementations** internally; there are no real NVIDIA NCCL Rust bindings on crates.io today
  (the `nccl` crate that does exist is an unrelated configuration library), so no actual
  GPUDirect/NVLink communication happens yet
- Tracked as permanently blocked pending real NCCL-equivalent Rust bindings — not a near-term fix

### Gloo (CPU and GPU) — not implemented
- `BackendType::Gloo` is accepted by the API but always resolves to the same in-process
  `MockBackend` used for testing; there is no real TCP or InfiniBand transport implemented, and no
  `gloo`-equivalent dependency is linked (it isn't available on crates.io either)

### MPI (HPC environments) — real, behind the `mpi` feature
- A genuine `MpiBackend` (using the `mpi` crate) implements real collective communication,
  including `barrier()`, which calls the actual `MPI_Barrier` collective and correctly keeps the
  MPI `Universe` alive for the backend's lifetime (an earlier version dropped `Universe`
  immediately after initialization, which finalized MPI before `barrier()` could ever work — fixed)
- Currently reachable via `torsh_distributed::backend::MpiBackend` directly; not yet wired into the
  `ProcessGroup`/`init_process_group` convenience path (see the note above)

## Environment Variables

```bash
# Basic setup
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export RANK=0
export WORLD_SIZE=4

# NCCL specific
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0

# Gloo specific
export GLOO_SOCKET_IFNAME=eth0
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../../LICENSE) for details.