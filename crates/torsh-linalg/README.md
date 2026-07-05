# torsh-linalg

Linear algebra operations for ToRSh, leveraging scirs2-linalg for optimized implementations.

## Overview

This crate provides comprehensive linear algebra functionality by wrapping scirs2-linalg with a PyTorch-compatible API:

- **Matrix Operations**: Multiplication, decompositions, solving
- **Eigenvalue Problems**: Eigenvalues, eigenvectors, SVD
- **Matrix Functions**: Inverse, determinant, trace, norms
- **Specialized Solvers**: Linear systems, least squares, Cholesky
- **Tensor Operations**: Einstein summation, tensor contractions

## Usage

### Basic Matrix Operations

There is no `linalg` module/namespace in this crate — `use torsh_linalg::prelude::*;` brings these
functions directly into scope (no `linalg::` prefix). The matrix-vector product function is named
`matvec`, not `mv`:

```rust
use torsh_linalg::prelude::*;
use torsh_tensor::prelude::*;

// Matrix multiplication (2D only today — see note below on batching)
let a = randn(&[10, 20])?;
let b = randn(&[20, 30])?;
let c = matmul(&a, &b)?;

// Batch matrix multiplication (dedicated batched entry point)
let batch_a = randn(&[32, 10, 20])?;
let batch_b = randn(&[32, 20, 30])?;
let batch_c = bmm(&batch_a, &batch_b)?;

// Matrix-vector multiplication (function name is `matvec`, not `mv`)
let matrix = randn(&[10, 20])?;
let vector = randn(&[20])?;
let result = matvec(&matrix, &vector)?;
```

Note: `matmul`'s own doc comment states it currently delegates straight to `Tensor::matmul` for
plain 2D inputs — general N-D broadcasting/batching through `matmul` itself is called out in the
source as a future enhancement; use `bmm` for the batched case shown above.

### Decompositions

There is no separate `lu_factor` function — `lu` alone returns the full `(P, L, U)` factorization
(such that `P @ A = L @ U`), and `cholesky`/`svd` take a required `bool` argument (there is no
1-argument overload):

```rust
// LU decomposition with partial pivoting: returns (P, L, U) where P @ A = L @ U
let (p, l, u) = lu(&matrix)?;

// QR decomposition
let (q, r) = qr(&matrix)?;

// Cholesky decomposition (for positive definite matrices); `upper` selects
// upper- vs. lower-triangular factor — there is no argument-less overload
let l = cholesky(&pos_def_matrix, false)?;

// Eigenvalue decomposition
let (eigenvalues, eigenvectors) = eig(&square_matrix)?;

// Singular Value Decomposition (SVD); `full_matrices` is required, not optional
let (u, s, v) = svd(&matrix, true)?;
let (u_reduced, s_reduced, v_reduced) = svd(&matrix, false)?; // reduced SVD
```

### Solving Linear Systems

There is no `tril`/`cholesky_solve` function in this crate, and `solve_triangular`/`lstsq` take
different argument lists than shown previously (no separate "solve with Cholesky" helper — use
`cholesky` to factor, then `solve_triangular` against the factor):

```rust
// Solve Ax = b
let a = randn(&[10, 10])?;
let b = randn(&[10, 5])?;
let x = solve(&a, &b)?;

// Solve a triangular system (`upper` selects which triangle `lower` is read
// from — there is no standalone `tril()` extraction helper in this crate)
let lower = cholesky(&pos_def_matrix, false)?;
let x = solve_triangular(&lower, &b, false)?; // 3 args: (matrix, rhs, upper)

// Least squares solution: returns (solution, residuals, rank, singular_values),
// not just `x`, and `rcond` is a required (if Option-typed) third argument
let a = randn(&[20, 10])?; // overdetermined
let b = randn(&[20, 1])?;
let (x, _residuals, _rank, _singular_values) = lstsq(&a, &b, None)?;
```

### Matrix Properties

The matrix-norm function is named `matrix_norm` (not `norm`), takes `Option<&str>` (there is no
numeric-typed overload — spectral norm is requested via the string `"2"`), and `pinv` requires an
explicit `rcond` argument:

```rust
// Determinant
let det = det(&square_matrix)?;

// Inverse
let inv = inv(&square_matrix)?;
let pinv_result = pinv(&matrix, None)?; // pseudo-inverse; rcond: Option<f32>

// Matrix norms (function is `matrix_norm`, ord is `Option<&str>`)
let frobenius = matrix_norm(&matrix, Some("fro"))?;
let nuclear = matrix_norm(&matrix, Some("nuc"))?;
let spectral = matrix_norm(&matrix, Some("2"))?;

// Condition number
let cond = cond(&matrix, None)?;

// Rank
let rank = matrix_rank(&matrix, None)?;

// Trace
let trace = trace(&square_matrix)?;
```

### Advanced Operations

The Kronecker product function is named `kronecker`, not `kron`, and `matrix_power`'s exponent is a
plain `i32` — there is no fractional-power / matrix-square-root overload (`0.5` below would not
type-check):

```rust
// Einstein summation
let result = einsum("ij,jk->ik", &[&a, &b])?;
let batch_result = einsum("bij,bjk->bik", &[&batch_a, &batch_b])?;

// Kronecker product (function is `kronecker`; both inputs must be 2D)
let kron_result = kronecker(&a, &b)?;

// Matrix exponential
let exp_matrix = matrix_exp(&square_matrix)?;

// Matrix power (integer exponents only)
let matrix_squared = matrix_power(&square_matrix, 2)?;
```

### Special Matrix Constructors

`eye` takes two arguments (`n`, `m`), not three, and `diag` always requires a `diagonal: i32` offset
argument (for both constructing a diagonal matrix from a vector and extracting a diagonal from a
matrix — the same function dispatches on the input's rank). Note also that `torsh_tensor::prelude`
exports its own `eye` (a tensor-creation helper, different signature) — glob-importing both preludes
together makes bare `eye(...)` ambiguous; qualify as `torsh_linalg::eye(...)` if you hit that:

```rust
// Identity matrix
let eye_matrix = eye(10, None)?;

// Diagonal matrix (offset 0 = main diagonal)
let diag_vals = tensor![1.0, 2.0, 3.0, 4.0];
let diag_matrix = diag(&diag_vals, 0)?;

// Extract diagonal
let diagonal = diag(&matrix, 0)?;

// Vandermonde matrix
let x = tensor![1.0, 2.0, 3.0, 4.0];
let vander_matrix = vander(&x, None, true)?;
```

### Batch Operations

Dedicated batched entry points exist for some operations (e.g. `bmm` above); whether every function
below actually loops over leading batch dimensions internally has not been verified against the
implementation, so treat this section as indicative rather than guaranteed:

```rust
// Batch inverse
let batch_matrices = randn(&[32, 10, 10])?;
let batch_inv = inv(&batch_matrices)?;

// Batch solve
let batch_a = randn(&[32, 10, 10])?;
let batch_b = randn(&[32, 10, 5])?;
let batch_x = solve(&batch_a, &batch_b)?;

// There is no separate eigenvalues-only `eigvals` function — use `eig`,
// which returns both eigenvalues and eigenvectors
let (batch_eigenvalues, _batch_eigenvectors) = eig(&batch_matrices)?;
```

### Performance Considerations

This crate leverages scirs2-linalg which uses:
- Optimized BLAS/LAPACK implementations
- Multi-threading for large operations
- GPU acceleration when available
- Efficient memory layouts

## Integration with SciRS2

All operations are implemented via scirs2-linalg, ensuring:
- Consistent numerical behavior
- Optimized performance
- Hardware acceleration support
- Compatibility with the scirs2 ecosystem

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../../LICENSE) for details.