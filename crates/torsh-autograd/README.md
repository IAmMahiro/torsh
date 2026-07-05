# torsh-autograd

Automatic differentiation engine for ToRSh, providing PyTorch-compatible autograd functionality powered by scirs2.

## Overview

This crate leverages scirs2's powerful automatic differentiation capabilities to provide:

- Reverse-mode automatic differentiation
- PyTorch-compatible gradient computation API
- Advanced features like Jacobian/Hessian computation
- Gradient accumulation and checkpointing
- Memory-efficient training utilities

## Features

- **Full scirs2 Integration**: Built on top of scirs2-autograd for robust AD (default `autograd` feature)
- **Gradient Modes**: Support for no_grad, inference_mode, and anomaly detection
- **Higher-Order Gradients**: `HigherOrderGradient` API surface for Jacobian/Hessian computation (infrastructure in place; numerical core is still a placeholder — see TODO.md)
- **Custom Functions**: Define your own differentiable operations via the `Function` trait
- **Memory Efficiency**: Gradient accumulation via the `GradientAccumulation` trait; computation-graph checkpoint snapshots via `GraphCheckpoint`
- **Performance**: Profiling and optimization utilities

## Usage

### Basic Gradient Computation

```rust
use torsh_autograd::prelude::*;
use torsh_tensor::prelude::*;

// Enable gradient computation
let x = tensor![2.0].requires_grad_(true);
let y = x.pow(2.0)?;

// Compute gradients (Tensor::backward is the real entry point; there is no
// separate free-standing `backward()` function in this crate)
y.backward()?;

// Access gradient
let grad = x.grad().unwrap();
assert_eq!(grad.item(), 4.0); // dy/dx = 2x = 4
```

### Gradient Modes

```rust
// no_grad() is re-exported from the crate prelude; inference_mode() and
// detect_anomaly() live in `grad_mode` and need an explicit import
use torsh_autograd::grad_mode::{detect_anomaly, inference_mode};

// Disable gradient computation
{
    let _guard = no_grad();
    // Operations here won't track gradients
    let z = x.mul(&y)?;
}

// Inference mode for maximum performance
{
    let _guard = inference_mode();
    // No graph building, pure computation
    let output = model.forward(&input)?;
}

// Anomaly detection for debugging
{
    let _guard = detect_anomaly();
    // Will detect NaN/Inf in gradients
    loss.backward()?;
}
```

### Advanced Gradient Functions

`torsh-autograd` exposes a `HigherOrderGradient` type (`torsh_autograd::prelude::HigherOrderGradient`)
with a `compute_hessian`/`compute_jacobian`/`hessian_vector_product` API surface operating on flat
`&[f32]` data:

```rust
use torsh_autograd::prelude::HigherOrderGradient;

let mut higher_order = HigherOrderGradient::new();
let hessian = higher_order.compute_hessian(&data, param_count)?;
```

Note: as of this writing the numerical core of `compute_hessian`/`compute_jacobian` is a documented
placeholder (it returns a zero-filled matrix rather than a real second-derivative computation) —
see TODO.md for status. There is no closure-based, JAX-style `jacobian()`/`hessian()`/`vjp()`/`jvp()`
free function that takes an arbitrary closure and a `Tensor`. A lower-level, real (non-placeholder)
`vjp_optimization::VjpOptimizer::compute_vjp` exists for vector-Jacobian products over an explicitly
constructed graph of `VjpNode`s (add/mul/relu and a few other ops); see that module's tests for the
construction pattern.

### Custom Autograd Functions

The real `Function` trait (`torsh_autograd::function::{Function, FunctionContext}`) operates on
`&dyn AutogradTensor<T>` / `Box<dyn AutogradTensor<T>>`, not the concrete `torsh_tensor::Tensor<T>`
type directly, and there is currently no generic `apply_function()` dispatch helper — a `Function`
impl's `forward`/`backward` are called directly:

```rust
use torsh_autograd::autograd_traits::AutogradTensor;
use torsh_autograd::function::{Function, FunctionContext};

struct MyReLU;

impl Function for MyReLU {
    fn forward<T: TensorElement>(
        &self,
        ctx: &mut FunctionContext,
        inputs: &[&dyn AutogradTensor<T>],
    ) -> Result<Vec<Box<dyn AutogradTensor<T>>>> {
        ctx.save_value(0.0_f64); // save whatever backward() will need
        let data: Vec<T> = inputs[0].to_vec();
        // elided: apply max(0, x) element-wise to `data` here (via to_f64()/from_f64()
        // round-tripping, as TensorElement has no direct Ord bound) before wrapping back up
        let output = inputs[0].with_data(data)?;
        Ok(vec![output])
    }

    fn backward<T: TensorElement>(
        &self,
        ctx: &mut FunctionContext,
        grad_outputs: &[&dyn AutogradTensor<T>],
    ) -> Result<Vec<Option<Box<dyn AutogradTensor<T>>>>> {
        // elided: recover what forward() saved from `ctx`, mask grad_outputs by (input > 0)
        Ok(vec![None])
    }
}
```

### Gradient Accumulation

There is no `GradientAccumulator` struct in this crate; gradient accumulation is exposed as the
`GradientAccumulation` trait (`torsh_autograd::autograd_traits::GradientAccumulation`), implemented
per-tensor:

```rust
use torsh_autograd::autograd_traits::GradientAccumulation;

// `tensor` implements GradientAccumulation
for batch in batches {
    let loss = model.forward(&batch)?;
    loss.backward()?;
    tensor.accumulate_grad(loss.grad().expect("gradient after backward"))?;
}

let accumulated = tensor.grad();
```

### Memory-Efficient Training

There is no closure-based `checkpoint()` helper in this crate. `context::optimization` exposes a
`GraphCheckpoint` snapshot (node/edge/cache counts plus a timestamp) for inspecting computation-graph
growth rather than a PyTorch-style recompute-to-save-memory wrapper:

```rust
use torsh_autograd::context::optimization::GraphCheckpoint;

let checkpoint: GraphCheckpoint = ctx.checkpoint();
println!("nodes={} edges={}", checkpoint.node_count, checkpoint.edge_count);
```

### Gradient Clipping

`clip_grad_norm`/`clip_grad_value` live at the crate root (`torsh_autograd::clip`), not under
`grad_mode::clip` (that module is currently commented out), and operate on `&dyn AutogradTensor<T>`
rather than a `model.parameters()`-style collection:

```rust
use torsh_autograd::clip::{clip_grad_norm, clip_grad_value};

// Returns the computed gradient norm (Result<T>); note this computes the norm
// but does not itself mutate `gradients` in place
let total_norm = clip_grad_norm(&gradients, 1.0f32, 2.0)?;

// Returns a new clipped Vec<T> bounded to [min_value, max_value]
let clipped = clip_grad_value(gradient, -0.5f32, 0.5)?;
```

## Integration with SciRS2

This crate fully leverages scirs2-autograd's capabilities:

- **Variable Environment**: Managed gradient storage
- **Computation Graphs**: Efficient graph construction and traversal
- **Tensor Operations**: All operations use scirs2's optimized implementations
- **Memory Management**: Benefit from scirs2's memory optimization

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../../LICENSE) for details.