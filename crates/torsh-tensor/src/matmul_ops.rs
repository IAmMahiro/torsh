//! Matrix multiplication with PyTorch broadcasting semantics and a real GEMM.
//!
//! # Dispatch
//!
//! The 2-D kernel ([`gemm_into`]) picks the fastest available path:
//!
//! 1. `f32` / `f64` contiguous inputs are handed to
//!    `scirs2_core::ndarray::linalg::general_mat_mul`, the blocked, SIMD
//!    (`matrixmultiply`) GEMM behind `ndarray`'s `Dot` implementation. Input and
//!    output buffers are wrapped as views, so no matrix is copied.
//! 2. Any other element type uses a cache-blocked `i-k-j` kernel with register
//!    tiling over the output row. This keeps the innermost loop sequential in
//!    both `B` and `C`, unlike the classic `i-j-k` dot-product order which
//!    strides `B` by `n`.
//!
//! # Shape rules (`torch.matmul`)
//!
//! | lhs | rhs | result |
//! |-----|-----|--------|
//! | `[k]` | `[k]` | `[]` (dot product) |
//! | `[k]` | `[k, n]` | `[n]` |
//! | `[m, k]` | `[k]` | `[m]` |
//! | `[m, k]` | `[k, n]` | `[m, n]` |
//! | `[..b, m, k]` | `[..b, k, n]` | `[..broadcast(b), m, n]` |
//!
//! # Autograd
//!
//! Every rank listed above is recorded on the autograd tape: the backward rule
//! in `core_ops::autograd` promotes 1-D operands exactly like the forward pass,
//! walks the broadcast batch axes, and accumulates each operand's gradient over
//! the batch axes it was broadcast along.

use std::sync::Arc;

use torsh_core::{
    dtype::TensorElement,
    error::{Result, TorshError},
};

use crate::core_ops::Tensor;

/// Cache blocking parameters for the generic fallback kernel.
const BLOCK_K: usize = 64;
const BLOCK_N: usize = 128;

/// Compute `c = a @ b` for row-major `a` (`m x k`), `b` (`k x n`), `c` (`m x n`).
///
/// `c` is fully overwritten. Returns an error when the slice lengths do not
/// match the requested extents.
pub(crate) fn gemm_into<T>(
    m: usize,
    k: usize,
    n: usize,
    a: &[T],
    b: &[T],
    c: &mut [T],
) -> Result<()>
where
    T: TensorElement + Copy + num_traits::Float,
{
    if a.len() < m * k || b.len() < k * n || c.len() < m * n {
        return Err(TorshError::InvalidArgument(format!(
            "matmul: buffer too small for {}x{} @ {}x{} (got {}, {}, {})",
            m,
            k,
            k,
            n,
            a.len(),
            b.len(),
            c.len()
        )));
    }

    if m == 0 || n == 0 {
        return Ok(());
    }

    if gemm_blas_f32(m, k, n, a, b, c) || gemm_blas_f64(m, k, n, a, b, c) {
        return Ok(());
    }

    gemm_blocked(m, k, n, a, b, c);
    Ok(())
}

/// `f32` fast path through `ndarray`'s blocked SIMD GEMM. Returns `false` when
/// `T` is not `f32`.
fn gemm_blas_f32<T: TensorElement + Copy>(
    m: usize,
    k: usize,
    n: usize,
    a: &[T],
    b: &[T],
    c: &mut [T],
) -> bool {
    if std::any::TypeId::of::<T>() != std::any::TypeId::of::<f32>() {
        return false;
    }
    // Safety: TypeId confirmed T == f32, so the slices have identical layout.
    let a_f32: &[f32] = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f32, m * k) };
    let b_f32: &[f32] = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, k * n) };
    let c_f32: &mut [f32] =
        unsafe { std::slice::from_raw_parts_mut(c.as_mut_ptr() as *mut f32, m * n) };
    gemm_ndarray_f32(m, k, n, a_f32, b_f32, c_f32)
}

/// `f64` fast path through `ndarray`'s blocked SIMD GEMM.
fn gemm_blas_f64<T: TensorElement + Copy>(
    m: usize,
    k: usize,
    n: usize,
    a: &[T],
    b: &[T],
    c: &mut [T],
) -> bool {
    if std::any::TypeId::of::<T>() != std::any::TypeId::of::<f64>() {
        return false;
    }
    // Safety: TypeId confirmed T == f64, so the slices have identical layout.
    let a_f64: &[f64] = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f64, m * k) };
    let b_f64: &[f64] = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f64, k * n) };
    let c_f64: &mut [f64] =
        unsafe { std::slice::from_raw_parts_mut(c.as_mut_ptr() as *mut f64, m * n) };
    gemm_ndarray_f64(m, k, n, a_f64, b_f64, c_f64)
}

macro_rules! ndarray_gemm_impl {
    ($name:ident, $ty:ty) => {
        /// Run `general_mat_mul` over borrowed row-major buffers.
        fn $name(m: usize, k: usize, n: usize, a: &[$ty], b: &[$ty], c: &mut [$ty]) -> bool {
            use scirs2_core::ndarray::{linalg::general_mat_mul, ArrayView2, ArrayViewMut2};

            let a_view = match ArrayView2::from_shape((m, k), &a[..m * k]) {
                Ok(view) => view,
                Err(_) => return false,
            };
            let b_view = match ArrayView2::from_shape((k, n), &b[..k * n]) {
                Ok(view) => view,
                Err(_) => return false,
            };
            let mut c_view = match ArrayViewMut2::from_shape((m, n), &mut c[..m * n]) {
                Ok(view) => view,
                Err(_) => return false,
            };
            general_mat_mul(1.0, &a_view, &b_view, 0.0, &mut c_view);
            true
        }
    };
}

ndarray_gemm_impl!(gemm_ndarray_f32, f32);
ndarray_gemm_impl!(gemm_ndarray_f64, f64);

/// Cache-blocked generic GEMM used for element types without a BLAS kernel.
fn gemm_blocked<T>(m: usize, k: usize, n: usize, a: &[T], b: &[T], c: &mut [T])
where
    T: TensorElement + Copy + num_traits::Float,
{
    let zero = <T as num_traits::Zero>::zero();
    for value in c[..m * n].iter_mut() {
        *value = zero;
    }

    for k_block in (0..k).step_by(BLOCK_K) {
        let k_end = (k_block + BLOCK_K).min(k);
        for n_block in (0..n).step_by(BLOCK_N) {
            let n_end = (n_block + BLOCK_N).min(n);
            for i in 0..m {
                let a_row = &a[i * k..i * k + k];
                let c_row = &mut c[i * n..i * n + n];
                for (kk, &a_ik) in a_row.iter().enumerate().take(k_end).skip(k_block) {
                    if a_ik == zero {
                        continue;
                    }
                    let b_row = &b[kk * n + n_block..kk * n + n_end];
                    for (c_val, &b_val) in c_row[n_block..n_end].iter_mut().zip(b_row.iter()) {
                        *c_val = *c_val + a_ik * b_val;
                    }
                }
            }
        }
    }
}

/// Broadcast two batch-dimension lists following NumPy/PyTorch rules.
fn broadcast_batch_dims(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>> {
    let rank = lhs.len().max(rhs.len());
    let mut out = vec![1usize; rank];
    for i in 0..rank {
        let l = if i + lhs.len() >= rank {
            lhs[i + lhs.len() - rank]
        } else {
            1
        };
        let r = if i + rhs.len() >= rank {
            rhs[i + rhs.len() - rank]
        } else {
            1
        };
        out[i] = if l == r {
            l
        } else if l == 1 {
            r
        } else if r == 1 {
            l
        } else {
            return Err(TorshError::ShapeMismatch {
                expected: lhs.to_vec(),
                got: rhs.to_vec(),
            });
        };
    }
    Ok(out)
}

/// Flat batch offset of `coords` inside a (possibly broadcast) batch shape.
fn batch_offset(coords: &[usize], shape: &[usize], matrix_stride: usize) -> usize {
    let rank = shape.len();
    let mut offset = 0usize;
    let mut stride = matrix_stride;
    for axis in (0..rank).rev() {
        let coord_idx = coords.len() + axis - rank;
        let coord = if shape[axis] == 1 {
            0
        } else {
            coords[coord_idx]
        };
        offset += coord * stride;
        stride *= shape[axis];
    }
    offset
}

impl<T: TensorElement + Copy> Tensor<T> {
    /// Matrix multiplication following `torch.matmul` shape rules.
    ///
    /// Supports 1-D/2-D operands and batched (N-D) operands with broadcasting of
    /// the leading batch dimensions. Autograd is recorded for every one of those
    /// shapes — the backward rule mirrors the promotion and broadcasting done
    /// here.
    pub fn matmul(&self, other: &Self) -> Result<Self>
    where
        T: num_traits::Float + std::iter::Sum,
    {
        let mut result = self.basic_matmul(other)?;

        // Record the matmul operation for autograd. Backward computes, per batch,
        // dL/dlhs = grad @ rhsᵀ and dL/drhs = lhsᵀ @ grad, summing each operand's
        // contribution over the batch axes it was broadcast along.
        if crate::should_record_grad(self.requires_grad || other.requires_grad) {
            result.requires_grad = true;
            result.operation = crate::core_ops::Operation::MatMul {
                lhs: Arc::new(self.clone()),
                rhs: Arc::new(other.clone()),
            };
        }
        Ok(result)
    }

    /// Shape-aware matrix multiplication kernel shared by [`Tensor::matmul`] and
    /// [`Tensor::matmul_scirs2`].
    pub(crate) fn basic_matmul(&self, other: &Self) -> Result<Self>
    where
        T: num_traits::Float + std::iter::Sum,
    {
        let lhs_binding = self.shape();
        let lhs_shape = lhs_binding.dims().to_vec();
        let rhs_binding = other.shape();
        let rhs_shape = rhs_binding.dims().to_vec();

        if lhs_shape.is_empty() || rhs_shape.is_empty() {
            return Err(TorshError::InvalidArgument(
                "matmul requires tensors with at least one dimension".to_string(),
            ));
        }

        // Promote 1-D operands the way PyTorch does, remembering what to strip.
        let lhs_was_vector = lhs_shape.len() == 1;
        let rhs_was_vector = rhs_shape.len() == 1;

        let mut lhs_dims = lhs_shape.clone();
        if lhs_was_vector {
            lhs_dims.insert(0, 1);
        }
        let mut rhs_dims = rhs_shape.clone();
        if rhs_was_vector {
            rhs_dims.push(1);
        }

        let m = lhs_dims[lhs_dims.len() - 2];
        let k = lhs_dims[lhs_dims.len() - 1];
        let k_rhs = rhs_dims[rhs_dims.len() - 2];
        let n = rhs_dims[rhs_dims.len() - 1];

        if k != k_rhs {
            return Err(TorshError::ShapeMismatch {
                expected: vec![m, k],
                got: vec![k_rhs, n],
            });
        }

        let lhs_batch = lhs_dims[..lhs_dims.len() - 2].to_vec();
        let rhs_batch = rhs_dims[..rhs_dims.len() - 2].to_vec();
        let batch_shape = broadcast_batch_dims(&lhs_batch, &rhs_batch)?;
        let batch_count: usize = batch_shape.iter().product();

        let mut result_data = vec![<T as num_traits::Zero>::zero(); batch_count * m * n];

        // Materialise the right-hand side before borrowing the left one: both
        // operands can be backed by the *same* `RwLock` (`a.matmul(&a)`, or any
        // pair sharing an `Arc` after a clone), and taking two read guards on one
        // lock from a single thread is allowed to deadlock or panic.
        let rhs_data = other.to_vec()?;

        self.with_contiguous_data(|lhs_data| {
            {
                let rhs_data: &[T] = &rhs_data;
                if batch_shape.is_empty() {
                    return gemm_into(m, k, n, lhs_data, rhs_data, &mut result_data);
                }

                let mut coords = vec![0usize; batch_shape.len()];
                for batch in 0..batch_count {
                    // Decode the flat batch index into per-axis coordinates.
                    let mut remaining = batch;
                    for axis in (0..batch_shape.len()).rev() {
                        coords[axis] = remaining % batch_shape[axis];
                        remaining /= batch_shape[axis];
                    }

                    let lhs_offset = batch_offset(&coords, &lhs_batch, m * k);
                    let rhs_offset = batch_offset(&coords, &rhs_batch, k * n);
                    let out_offset = batch * m * n;

                    gemm_into(
                        m,
                        k,
                        n,
                        &lhs_data[lhs_offset..lhs_offset + m * k],
                        &rhs_data[rhs_offset..rhs_offset + k * n],
                        &mut result_data[out_offset..out_offset + m * n],
                    )?;
                }
                Ok(())
            }
        })?;

        // Assemble the output shape and strip the promoted vector axes.
        let mut out_shape = batch_shape;
        out_shape.push(m);
        out_shape.push(n);
        if rhs_was_vector {
            out_shape.pop();
        }
        if lhs_was_vector {
            let row_axis = out_shape.len() - if rhs_was_vector { 1 } else { 2 };
            out_shape.remove(row_axis);
        }

        Self::from_data(result_data, out_shape, self.device())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn gemm_into_matches_reference_for_f32() {
        let (m, k, n) = (3usize, 4usize, 2usize);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 - 3.0).collect();
        let mut c = vec![0.0f32; m * n];
        gemm_into(m, k, n, &a, &b, &mut c).expect("gemm should succeed");

        let mut expected = vec![0.0f32; m * n];
        for (i, row) in expected.chunks_mut(n).enumerate() {
            for kk in 0..k {
                for (j, out) in row.iter_mut().enumerate() {
                    *out += a[i * k + kk] * b[kk * n + j];
                }
            }
        }
        assert_eq!(c, expected);
    }

    #[test]
    fn gemm_blocked_matches_reference() {
        // Directly exercise the generic fallback kernel (used for non f32/f64).
        let (m, k, n) = (5usize, 7usize, 6usize);
        let a: Vec<f64> = (0..m * k).map(|i| (i % 5) as f64 - 2.0).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i % 7) as f64 - 3.0).collect();
        let mut c = vec![0.0f64; m * n];
        gemm_blocked(m, k, n, &a, &b, &mut c);

        let mut expected = vec![0.0f64; m * n];
        for (i, row) in expected.chunks_mut(n).enumerate() {
            for kk in 0..k {
                for (j, out) in row.iter_mut().enumerate() {
                    *out += a[i * k + kk] * b[kk * n + j];
                }
            }
        }
        assert_eq!(c, expected);
    }

    #[test]
    fn matmul_batched_autograd_is_recorded() {
        let a = Tensor::from_data(
            (1..=12).map(|v| v as f32).collect(),
            vec![2, 2, 3],
            DeviceType::Cpu,
        )
        .expect("tensor creation should succeed")
        .requires_grad_(true);
        let b = Tensor::from_data(
            (1..=12).map(|v| v as f32).collect(),
            vec![2, 3, 2],
            DeviceType::Cpu,
        )
        .expect("tensor creation should succeed");

        let product = a
            .matmul(&b)
            .expect("batched matmul must record the graph, not error out");
        assert_eq!(product.shape().dims(), &[2, 2, 2]);
        assert!(
            product.requires_grad(),
            "a batched matmul over a requires_grad operand must join the graph"
        );
        assert!(
            matches!(product.operation, crate::core_ops::Operation::MatMul { .. }),
            "the recorded node must be a MatMul"
        );

        product.sum().expect("sum").backward().expect("backward");
        let grad = a.grad().expect("lhs gradient").to_vec().expect("to_vec");
        assert_eq!(grad.len(), 12);
    }

    /// Both operands may be backed by the same storage lock; taking two read
    /// guards on one `RwLock` from one thread can deadlock, so the kernel must
    /// borrow at most one operand at a time.
    #[test]
    fn matmul_with_itself_and_with_a_clone_does_not_deadlock() {
        // Small enough to land in lock-based storage rather than lock-free SimdOptimized.
        let a = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu)
            .expect("tensor creation should succeed");
        let self_product = a.matmul(&a).expect("a @ a should succeed");
        assert_eq!(
            self_product.to_vec().expect("to_vec"),
            vec![7.0, 10.0, 15.0, 22.0]
        );

        let b = a.clone();
        let clone_product = a.matmul(&b).expect("a @ a.clone() should succeed");
        assert_eq!(
            clone_product.to_vec().expect("to_vec"),
            vec![7.0, 10.0, 15.0, 22.0]
        );
    }

    #[test]
    fn broadcast_batch_dims_rules() {
        assert_eq!(
            broadcast_batch_dims(&[2, 1], &[3]).expect("broadcast"),
            vec![2, 3]
        );
        assert_eq!(broadcast_batch_dims(&[], &[4]).expect("broadcast"), vec![4]);
        assert!(broadcast_batch_dims(&[2], &[3]).is_err());
    }
}
