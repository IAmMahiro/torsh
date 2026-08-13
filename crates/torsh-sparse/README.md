# torsh-sparse

Sparse tensor operations for ToRSh, leveraging scirs2-sparse for efficient sparse matrix computations.

**Status**: Stable for CPU sparse formats, operations, neural-network layers, and linear algebra;
GPU/cuSPARSE acceleration is a documented placeholder, not yet functional (see "GPU Acceleration"
below). **Tests**: 259/259 passing (`cargo nextest run --all-features`).

## Overview

This crate provides comprehensive sparse tensor support:

- **Sparse Formats**: COO, CSR, CSC, BSR, DIA, DSR, ELL, RLE, and Symmetric formats
- **Operations**: Sparse matrix multiplication, addition, elementwise ops, transpose, softmax
- **Conversions**: Dense to sparse, format conversions
- **GPU Scaffolding**: `CudaSparseTensor`/`CudaSparseOps` types exist but are currently placeholders
  (see "GPU Acceleration" below) — there is no working cuSPARSE backend yet
- **Integration**: Seamless integration with scirs2-sparse

All of the free functions below are called directly on the crate root / `torsh_sparse::ops` /
`torsh_sparse::linalg` paths shown — there is no `sparse::` namespace in this crate.

## Usage

### Creating Sparse Tensors

```rust
use torsh_sparse::{CooTensor, CsrTensor, sparse_from_dense, SparseFormat};
use torsh_core::Shape;
use torsh_tensor::prelude::*;

// From COO format (row indices, column indices, values, shape)
let row_indices = vec![0, 1, 1];
let col_indices = vec![2, 0, 2];
let values = vec![3.0f32, 4.0, 5.0];
let sparse_coo = CooTensor::new(row_indices, col_indices, values, Shape::new(vec![3, 4]))?;

// Or from (row, col, value) triplets
let coo2 = CooTensor::from_triplets(vec![(0, 0, 1.0f32), (1, 1, 2.0)], (2, 2))?;

// From a dense tensor (any target format, values below `threshold` are dropped)
let dense = tensor![[1.0, 0.0, 2.0],
                     [0.0, 0.0, 3.0],
                     [4.0, 5.0, 0.0]];
let sparse = sparse_from_dense(&dense, SparseFormat::Coo, Some(0.0))?;

// From CSR format (row pointers, column indices, values, shape)
let row_ptr = vec![0, 2, 3, 5]; // row pointers, len = rows + 1
let col_idx = vec![0, 2, 2, 0, 1];
let csr_values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
let sparse_csr = CsrTensor::new(row_ptr, col_idx, csr_values, Shape::new(vec![3, 3]))?;
```

### Sparse Operations

```rust
use torsh_sparse::ops;

// Sparse-sparse matrix multiplication
let c = ops::sparse_matmul(&a, &b)?;

// Sparse-dense multiplication (SpMM): sparse @ dense -> dense
let dense_vector = randn::<f32>(&[1000, 1])?;
let result = ops::spmm(&sparse_matrix, &dense_vector)?;

// Element-wise operations
let sum = ops::element_add(&sparse_a, &sparse_b)?;
let hadamard_product = ops::sphadamard(&sparse_a, &sparse_b)?;

// Transpose
let transposed = ops::transpose(&sparse_matrix)?;
// ...or, equivalently, the inherent method:
let transposed = sparse_matrix.transpose()?;
```

### Format Conversions

```rust
// Convert between formats (methods on the SparseTensor trait)
let csr = sparse_coo.to_csr()?;
let csc = sparse_coo.to_csc()?;

// Convert to dense
let dense = sparse_tensor.to_dense()?;
```

### Advanced Sparse Operations

```rust
// Sparse linear algebra
use torsh_sparse::linalg::{conjugate_gradient, gmres, bicgstab, SparseCholesky};

// Incomplete Cholesky decomposition (for symmetric positive-definite matrices)
let cholesky = SparseCholesky::new(&symmetric_sparse, /* fill_factor */ 1.5)?;

// Iterative solvers: Ax = b
// conjugate_gradient(matrix, b, x0, tol, max_iter) -> (x, iterations, residual)
let (x, iters, residual) = conjugate_gradient(&sparse_a, &b, None, 1e-6, 1000)?;
let (x, iters, residual) = gmres(&sparse_a, &b, None, 1e-6, 1000, 50)?;
let (x, iters, residual) = bicgstab(&sparse_a, &b, None, 1e-6, 1000)?;
```

### Sparse Neural Network Layers

```rust
use torsh_sparse::nn::{SparseLinear, SparseEmbedding, GraphConvolution};

// Sparse Linear layer: new(in_features, out_features, sparsity, use_bias)
let sparse_linear = SparseLinear::new(1000, 100, 0.9, true)?; // 90% sparse

// Sparse Embedding
let sparse_embedding = SparseEmbedding::new(10_000, 300, 0.9)?;

// Graph Convolution (for GNNs): new(in_features, out_features, use_bias, add_self_loops, normalize)
let gcn = GraphConvolution::new(64, 32, true, true, true)?;
```

### GPU Acceleration

`torsh_sparse::gpu` defines the shape of a future cuSPARSE-backed API
(`CudaSparseTensor`, `CudaSparseOps`, `CudaSparseTensorFactory`), but as of this release it is a
placeholder: `CudaSparseTensor::is_cuda_available()` always returns `false`, and operations like
`spmm`, `spgemm`, dense↔sparse GPU conversion, and batched ops all return
`"not yet implemented"` errors regardless of the `cuda` feature flag. Real cuSPARSE bindings are
not currently wired up (the `cudarc` dependency behind the optional `cuda` feature is not yet used
by this module). Treat this as a documented placeholder, not a working GPU backend, until this
note is updated.

### Sparse Gradients

```rust
// Sparse optimizer for sparse gradients (new(lr, beta1, beta2, eps, weight_decay, amsgrad))
use torsh_sparse::optimizers::SparseAdam;

let sparse_adam = SparseAdam::new(0.001, 0.9, 0.999, 1e-8, 0.0, false);

// Gradient accumulation for sparse tensors, keyed by tensor id
use torsh_sparse::autograd::SparseGradientAccumulator;

let mut grad_accumulator = SparseGradientAccumulator::new();
grad_accumulator.accumulate(tensor_id, sparse_gradient)?;
```

### Utilities

```rust
use torsh_sparse::{analyze_sparse_tensor, compare_format_performance};

// Analyze sparsity
let stats = analyze_sparse_tensor(&tensor)?;
println!("Sparsity: {:.2}%", stats.sparsity * 100.0);
println!("NNZ: {}", stats.nnz);
println!("Detected pattern: {:?}", stats.pattern);
println!("Recommended format: {:?}", stats.recommended_format);

// Benchmark and compare all sparse formats for a given tensor
let comparison = compare_format_performance(&sparse_tensor, /* include_operations */ true)?;
```

## Integration with SciRS2

This crate fully leverages scirs2-sparse for:
- Optimized sparse BLAS operations
- Efficient sparse matrix formats
- Hardware-accelerated sparse computations
- Advanced sparse linear algebra

## Performance Tips

1. Choose the right format for your access pattern
2. Use CSR for row-wise operations, CSC for column-wise
3. Consider hybrid formats for mixed access patterns
4. Use batched operations when possible
5. Profile different sparse formats for your use case (see `compare_format_performance` above)

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../../LICENSE) for details.
