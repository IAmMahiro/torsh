# torsh-functional

Functional operations for ToRSh tensors, providing PyTorch-compatible functional API.

## Overview

This crate provides a comprehensive set of functional operations that work on tensors:

- **Mathematical Operations**: Element-wise, reduction, and special functions
- **Neural Network Functions**: Activations, normalization, loss functions
- **Linear Algebra**: Matrix operations, decompositions, solvers
- **Signal Processing**: FFT, convolution, filtering
- **Image Operations**: Transforms, filters, augmentations

Note: This crate integrates with various scirs2 modules (scirs2-linalg, scirs2-special, scirs2-signal, scirs2-fft) for optimized implementations.

## Usage

### Mathematical Operations

```rust
use torsh_functional as F;
use torsh_tensor::prelude::*;

// Element-wise arithmetic is provided by Tensor methods, not torsh_functional
let a = tensor![[1.0, 2.0], [3.0, 4.0]];
let b = tensor![[5.0, 6.0], [7.0, 8.0]];

let sum = a.add_op(&b)?;
let product = a.mul_op(&b)?;
let power = F::pow(&a, 2.0)?;

// Trigonometric functions
let angles = tensor![0.0, PI/4.0, PI/2.0];
let sines = F::sin(&angles)?;
let cosines = F::cos(&angles)?;

// Reductions
let sum_all = F::sum(&a)?;
let mean = F::mean(&a)?;
let std = F::std(&a, true)?;  // unbiased
let (max_vals, max_idx) = F::max_dim(&a, 1, true)?;  // dim, keepdim -> (values, indices)
```

### Neural Network Functions

```rust
// Activation functions
let x = randn(&[10, 20]);
let relu = F::relu(&x)?;
let sigmoid = F::sigmoid(&x)?;
let tanh = F::tanh(&x)?;
let gelu = F::gelu(&x)?;
let swish = F::silu(&x)?;

// Softmax with temperature
let logits = randn(&[32, 10]);
let probs = F::softmax(&logits, -1)?;
let log_probs = F::log_softmax(&logits, -1)?;

// Normalization
let normalized = F::layer_norm(&x, &[20], None, None, 1e-5)?;
let batch_normed = F::batch_norm(&x, None, None, None, None, true, 0.1, 1e-5)?;

// Dropout
let dropped = F::dropout(&x, 0.5, true)?;  // training mode
```

### Loss Functions

```rust
use torsh_functional::loss::ReductionType;

// Classification losses (cross_entropy/nll_loss still take a &str reduction;
// note the current nll_loss implementation only supports weight=None, ignore_index=None)
let logits = model.forward(&input)?;
let targets = tensor![0, 1, 2, 3];

let ce_loss = F::cross_entropy(&logits, &targets, None, "mean", None, 0.0)?; // label_smoothing = 0.0
let nll = F::nll_loss(&log_probs, &targets, None, "mean", None)?;

// Regression losses take a `ReductionType` enum, not a string
let predictions = model.forward(&input)?;
let targets = randn(&predictions.shape());

let mse = F::mse_loss(&predictions, &targets, ReductionType::Mean)?;
let mae = F::l1_loss(&predictions, &targets, ReductionType::Mean)?;
let huber = F::smooth_l1_loss(&predictions, &targets, ReductionType::Mean, 1.0)?; // beta = 1.0

// Binary classification
let binary_logits = model.forward(&input)?;
let binary_targets = rand(&binary_logits.shape())?;

let bce = F::binary_cross_entropy_with_logits(
    &binary_logits,
    &binary_targets,
    None,
    ReductionType::Mean,
    None, // pos_weight
)?;
```

### Convolution and Pooling

```rust
// 2D Convolution (stride/padding/dilation are (usize, usize) tuples, not slices)
let input = randn(&[1, 3, 224, 224]);  // NCHW
let weight = randn(&[64, 3, 7, 7]);
let bias = randn(&[64]);

let output = F::conv2d(
    &input,
    &weight,
    Some(&bias),
    (2, 2),  // stride
    (3, 3),  // padding
    (1, 1),  // dilation
    1,       // groups
)?;

// Pooling operations (max_pool2d returns (values, Option<indices>))
let (pooled, _indices) = F::max_pool2d(
    &input, (2, 2), Some((2, 2)), (0, 0), (1, 1), false, false,
)?; // kernel, stride, padding, dilation, ceil_mode, return_indices
let avg_pooled = F::avg_pool2d(
    &input, (2, 2), Some((2, 2)), (0, 0), false, true, None,
)?; // kernel, stride, padding, ceil_mode, count_include_pad, divisor_override
let adaptive = F::adaptive_avg_pool2d(&input, (1, 1))?;  // global pooling
```

### Linear Algebra

```rust
// Matrix operations (leveraging scirs2-linalg)
let a = randn(&[10, 20]);
let b = randn(&[20, 30]);

let c = F::matmul(&a, &b)?;
let det = F::det(&square_matrix)?;
let inv = F::inv(&square_matrix)?;

// Eigenvalues and eigenvectors
let (eigenvalues, eigenvectors) = F::eig(&symmetric_matrix)?;

// SVD
let (u, s, v) = F::svd(&matrix, true, true)?;

// Solve linear systems
let x = F::solve(&a, &b)?;  // Solve Ax = b
```

### Signal Processing

```rust
// FFT operations (leveraging scirs2-signal)
let signal = randn(&[1024]);
let spectrum = F::fft(&signal)?;
let reconstructed = F::ifft(&spectrum)?;

// 2D FFT for images
let image = randn(&[1, 3, 256, 256]);
let freq_domain = F::fft2(&image)?;
```

### Advanced Operations

```rust
// Note: there are two distinct `InterpolationMode` enums in this crate.
// `torsh_functional::InterpolationMode` (re-exported from `image`) has
// Nearest/Bilinear/Bicubic/Area. interp2d/grid_sample instead take
// `torsh_functional::interpolation::InterpolationMode` (Linear/Nearest/Cubic/Spline) -
// import it explicitly to get the right type.
use torsh_functional::interpolation::InterpolationMode;

// Coordinate-based 2D interpolation (there is no PyTorch-style size-based
// F::interpolate/affine_grid; interp2d resamples at explicit coordinates)
let sampled_at_coords = F::interp2d(&input_2d, &x_coords, &y_coords, InterpolationMode::Linear)?;

// Grid sampling (align_corners is a plain bool, not Option<bool>)
let grid = randn(&[1, 32, 32, 2]); // [N, H_out, W_out, 2]
let sampled = F::grid_sample(&input, &grid, InterpolationMode::Linear, "zeros", true)?;
```

### Utilities

```rust
use torsh_functional::advanced_manipulation::PaddingMode;

// Tensor manipulation (reshape is in torsh_functional; permute is a Tensor method)
let reshaped = F::reshape(&tensor, &[-1, 10])?;
let permuted = tensor.permute(&[0, 2, 3, 1])?;

// Padding (mode is a PaddingMode enum, value is a plain f32)
let padded = F::pad(&tensor, &[1, 1, 2, 2], PaddingMode::Constant, 0.0)?;

// Concatenation and stacking (stack is a Tensor::stack associated function, not F::)
let tensors = vec![a.clone(), b.clone(), c.clone()];
let concatenated = F::cat(&tensors, 0)?;
let stacked = Tensor::stack(&tensors, 1)?;

// Splitting
let chunks = F::chunk(&tensor, 4, 0)?;
let splits = F::split(&tensor, &[10, 20, 30], 0)?;
```

## Integration with SciRS2

This crate leverages multiple scirs2 modules for optimized implementations:

- **scirs2-linalg**: For linear algebra operations (matrix multiplication, decompositions)
- **scirs2-special**: For special mathematical functions (bessel, gamma, etc.)
- **scirs2-signal**: For signal processing operations
- **scirs2-fft**: For Fast Fourier Transform operations
- **scirs2-core**: For SIMD operations and memory management
- **scirs2-neural**: For neural network specific operations

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../../LICENSE) for details.