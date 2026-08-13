# torsh-backend

Unified backend implementation for ToRSh with PyTorch-compatible API, leveraging SciRS2's GPU acceleration.

## Overview

This crate provides a unified backend system with feature-gated compute backends:

- **CPU Backend**: Optimized CPU operations with SIMD and parallelism, via scirs2-core (oxiblas-backed)
- **CUDA Backend**: torsh-backend does **not** provide its own CUDA compute stack. The `cuda` feature only selects a pure-Rust honest-fallback API (every device op returns an error or delegates to the CPU). Real NVIDIA GPU acceleration lives in `torsh-tensor`'s oxicuda-based `GpuDispatch` (runtime driver load, no CUDA SDK at build time).
- **Metal Backend**: Apple GPU acceleration via the `metal`/`objc2` crates directly
- **ROCm Backend**: AMD GPU acceleration (feature flag reserved; no HIP/ROCm bindings implemented yet)
- **WebGPU Backend**: Cross-platform GPU support implemented natively in this crate via `wgpu` (not via scirs2-core)

Note: All backend implementations are unified in this single crate using feature flags, eliminating the need for separate torsh-backend-* crates.

## Architecture

The backend system exposes a unified device/backend abstraction (CPU via scirs2-core; CUDA, Metal, and WebGPU implemented directly against their native crates):

```rust
use torsh_backend::{Backend, BackendType, Device};

// Unified backend with runtime selection
let backend = Backend::new(BackendType::Auto)?;  // Auto-detect best backend
let backend = Backend::new(BackendType::Cuda)?;  // Explicit CUDA
let backend = Backend::new(BackendType::Metal)?; // Explicit Metal
```

## Feature Flags

```toml
[dependencies]
torsh-backend = { version = "0.2.0", features = ["cuda", "metal"] }

# Available features:
# - "cpu" (default): CPU backend with SIMD optimizations (scirs2-core parallel/simd, oxiblas-backed)
# - "cuda": pure-Rust honest-fallback CUDA API only (ops error or route to CPU); real GPU
#           compute lives in torsh-tensor's oxicuda GpuDispatch, not here
# - "metal": Apple GPU backend via the metal/objc2 crates
# - "rocm": AMD GPU backend (feature flag reserved; not yet implemented)
# - "webgpu": WebGPU backend via the wgpu crate, implemented natively in this crate
```

## Usage

### Unified Backend API

```rust
use torsh_backend::prelude::*;

// Automatic backend selection based on availability
let backend = Backend::auto()?;

// Query available backends
for backend_type in Backend::available() {
    println!("Available: {:?}", backend_type);
}

// Create backend with specific configuration
let backend = BackendBuilder::new()
    .backend_type(BackendType::Cuda)
    .device_id(0)
    .memory_pool_size(4 * 1024 * 1024 * 1024)  // 4GB
    .enable_tensor_cores(true)
    .build()?;

// All backends use the same API
let a = backend.randn(&[1024, 1024], DType::F32)?;
let b = backend.randn(&[1024, 1024], DType::F32)?;
let c = backend.matmul(&a, &b)?;
```

### CPU Backend

```rust
// CPU backend leverages scirs2-core's optimized operations
let cpu_backend = Backend::cpu()
    .num_threads(8)
    .enable_simd(true)
    .build()?;

// Uses OpenBLAS/MKL/Accelerate via scirs2
let result = cpu_backend.gemm(&a, &b, 1.0, &c, 0.0)?;
```

### CUDA Backend

torsh-backend's `cuda` feature exposes only a pure-Rust honest-fallback API:
`Backend::cuda()` never yields a working device here (it returns an error), and
`torsh_backend::cuda::is_available()` is always `false`. For real GPU compute,
use `torsh-tensor`'s oxicuda-based path, which loads the CUDA driver at runtime
without a build-time SDK.

### Metal Backend

```rust  
#[cfg(feature = "metal")]
{
    // Metal backend with MPS via scirs2-core
    let metal_backend = Backend::metal()
        .enable_mps(true)  // Metal Performance Shaders
        .build()?;
    
    // Leverages Apple's optimized kernels
    let result = metal_backend.conv2d(&input, &kernel, ConvConfig {
        stride: [1, 1],
        padding: [1, 1],
        dilation: [1, 1],
        groups: 1,
    })?;
}
```

### Unified Operations

```rust
// All backends support the same operations via scirs2
impl Backend {
    // BLAS operations (via scirs2-core)
    pub fn gemm(&self, a: &Tensor, b: &Tensor, alpha: f32, 
                c: &Tensor, beta: f32) -> Result<Tensor>;
    pub fn gemv(&self, a: &Tensor, x: &Tensor, alpha: f32,
                y: &Tensor, beta: f32) -> Result<Tensor>;
    
    // DNN operations (via scirs2's GPU kernels)
    pub fn conv2d(&self, input: &Tensor, weight: &Tensor,
                  config: ConvConfig) -> Result<Tensor>;
    pub fn batch_norm(&self, input: &Tensor, mean: &Tensor,
                      var: &Tensor, training: bool) -> Result<Tensor>;
    
    // Optimized fused operations from scirs2
    pub fn fused_adam_step(&self, params: &mut [Tensor], 
                           grads: &[Tensor], state: &mut AdamState,
                           lr: f32, betas: (f32, f32)) -> Result<()>;
}
```

## Device Management

```rust
// Unified device abstraction
let devices = Backend::list_devices()?;
for device in devices {
    println!("{}: {} ({}GB memory)", 
             device.id(), device.name(), device.memory_gb());
}

// Multi-device support
let backend_gpu0 = Backend::new(BackendType::Cuda)?.device(0)?;
let backend_gpu1 = Backend::new(BackendType::Cuda)?.device(1)?;

// Device synchronization
backend.synchronize()?;
```

## Memory Management

```rust
// Unified memory pool leveraging scirs2's allocators
let backend = Backend::new(BackendType::Auto)?
    .memory_pool(MemoryPoolConfig {
        initial_size: 1 << 30,      // 1GB
        max_size: 4 << 30,          // 4GB  
        strategy: AllocationStrategy::BestFit,
        enable_defrag: true,
    })?;

// Zero-copy host-device transfers (when supported)
let pinned = backend.alloc_pinned(&[1024, 1024], DType::F32)?;
backend.copy_host_to_device_async(&host_data, &mut pinned).await?;
```

## Performance Features

```rust
// Auto-tuning (via scirs2's auto-tuning infrastructure)
let backend = Backend::new(BackendType::Cuda)?
    .enable_autotuning(true)
    .autotuning_cache_file("tuning_cache.json")?;

// Mixed precision training
let backend = backend.enable_mixed_precision(MixedPrecisionConfig {
    compute_dtype: DType::F16,
    accum_dtype: DType::F32,
    scale_factor: 65536.0,
})?;

```

## Integration with SciRS2

This crate uses scirs2-core for CPU (parallel/SIMD, oxiblas) and integrates its own native GPU implementations for the rest:

### Backend Implementation Status

| Backend | Implementation | Features |
|---------|----------------|----------|
| CPU | ✅ scirs2-core (`parallel`, `simd` features; oxiblas-backed) | SIMD, Rayon/scirs2 parallelism, auto-tuning |
| CUDA | ➡️ Delegated to `torsh-tensor` (oxicuda). torsh-backend's `cuda` feature is a pure-Rust honest fallback only (ops error or route to CPU) | Real GPU compute via `torsh-tensor`'s runtime-loaded oxicuda `GpuDispatch` — no CUDA SDK at build time |
| Metal | ✅ Direct via `metal`/`objc2` (feature `metal`) | Metal Performance Shaders, Neural Engine hooks, unified memory |
| ROCm | 🚧 Not implemented | `rocm` feature flag reserved (`rocm = []`); no HIP/ROCm bindings yet |
| WebGPU | ✅ Direct via `wgpu`, implemented natively in this crate (feature `webgpu`) | Device/buffer/pipeline/multi-device management with real cross-device buffer copies |

### Implementation Notes

- **GPU Kernels**: CPU ops route through scirs2-core (oxiblas-backed); Metal and WebGPU backends are implemented directly against their native crates (`metal`/`objc2`, `wgpu`). CUDA compute is delegated to `torsh-tensor`'s oxicuda path rather than duplicated here
- **Auto-tuning**: Kernel selection via this crate's own auto-tuning system (`cpu/autotuning.rs`)
- **Memory Management**: Backend-specific memory pools plus a unified memory pool implemented in this crate
- **Async Execution**: WebGPU and CUDA async APIs implemented natively in this crate
- **BLAS/LAPACK**: CPU ops via scirs2-core's oxiblas integration

### Migration from Separate Backend Crates

The previous separate backend crates (`torsh-backend-cpu`, `torsh-backend-cuda`, `torsh-backend-metal`) are now deprecated. Use feature flags instead:

```toml
# Old (deprecated)
torsh-backend-cuda = "0.1.3"

# New (unified)
torsh-backend = { version = "0.2.0", features = ["cuda"] }
```

## Dependencies

Key dependency versions used by this crate:

- `wgpu 30.x` — WebGPU compute backend (cross-platform GPU support)

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../../LICENSE) for details.