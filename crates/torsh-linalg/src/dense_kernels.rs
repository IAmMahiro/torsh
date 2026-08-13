//! Dense numeric kernels backing the decomposition routines.
//!
//! Everything in this module operates on row-major `f64` buffers so that the
//! `f32` tensor API does not limit the achievable accuracy. Two kinds of
//! kernels live here:
//!
//! * **Local implementations** — Householder QR, one-sided Jacobi SVD, cyclic
//!   Jacobi symmetric eigensolver, LU with partial pivoting and inverse
//!   iteration for eigenvectors. These are always available.
//! * **SciRS2 delegation** — the eigenvalues of a general (non-symmetric)
//!   matrix come from `scirs2-linalg`, which is LAPACK-backed through OxiBLAS.
//!   That path is compiled only when the `scirs2-integration` feature is
//!   enabled (the default); with the feature off it reports an honest error
//!   instead of guessing.
//!
//! The SVD and the symmetric eigensolver are deliberately *not* delegated:
//! `scirs2_linalg::eigh` returns duplicated eigenvectors for repeated
//! eigenvalues (e.g. the 2x2 identity yields the same vector twice), which
//! propagates into `scirs2_linalg::svd` and destroys the factorisation of any
//! matrix with a degenerate spectrum. The Jacobi kernels below handle
//! degeneracy exactly.

use crate::TorshResult;
use torsh_core::{DeviceType, TorshError};
use torsh_tensor::Tensor;

/// Read a 2-D tensor into a row-major `f64` buffer.
///
/// Returns the buffer together with the matrix dimensions.
pub(crate) fn tensor_to_f64(tensor: &Tensor) -> TorshResult<(Vec<f64>, usize, usize)> {
    if tensor.shape().ndim() != 2 {
        return Err(TorshError::InvalidArgument(
            "expected a 2D tensor".to_string(),
        ));
    }
    let (m, n) = (tensor.shape().dims()[0], tensor.shape().dims()[1]);
    let mut data = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            data.push(tensor.get(&[i, j])? as f64);
        }
    }
    Ok((data, m, n))
}

/// Build an `f32` tensor from a row-major `f64` buffer.
pub(crate) fn f64_to_tensor(
    data: &[f64],
    rows: usize,
    cols: usize,
    device: DeviceType,
) -> TorshResult<Tensor> {
    if data.len() != rows * cols {
        return Err(TorshError::InvalidArgument(format!(
            "buffer of {} elements cannot be reshaped to {rows}x{cols}",
            data.len()
        )));
    }
    let values: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    Tensor::from_data(values, vec![rows, cols], device)
}

/// Build a 1-D `f32` tensor from an `f64` buffer.
pub(crate) fn f64_to_vector(data: &[f64], device: DeviceType) -> TorshResult<Tensor> {
    let values: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let len = values.len();
    Tensor::from_data(values, vec![len], device)
}

/// Infinity norm (max absolute row sum) of a square row-major matrix.
pub(crate) fn matrix_inf_norm(a: &[f64], n: usize) -> f64 {
    let mut best = 0.0f64;
    for i in 0..n {
        let mut row = 0.0f64;
        for j in 0..n {
            row += a[i * n + j].abs();
        }
        if row > best {
            best = row;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Householder QR
// ---------------------------------------------------------------------------

/// Thin (reduced) Householder QR factorisation of an `m x n` matrix.
///
/// Returns `(q, r)` with `q` of shape `m x k` and `r` of shape `k x n`, where
/// `k = min(m, n)`. Unlike classical Gram-Schmidt this stays orthogonal to
/// working precision on ill-conditioned input and never fails on rank-deficient
/// or wide (`m < n`) matrices.
pub(crate) fn householder_qr(a: &[f64], m: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let k = m.min(n);
    let mut work = a.to_vec();
    let mut reflectors: Vec<(Vec<f64>, f64)> = Vec::with_capacity(k);

    for j in 0..k {
        // Norm of the sub-column below (and including) the diagonal.
        let mut norm_sq = 0.0f64;
        for i in j..m {
            let v = work[i * n + j];
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt();

        if norm <= f64::EPSILON * f64::EPSILON {
            reflectors.push((vec![0.0; m], 0.0));
            continue;
        }

        // Choose the sign that avoids cancellation.
        let alpha = if work[j * n + j] > 0.0 { -norm } else { norm };

        let mut v = vec![0.0f64; m];
        for i in j..m {
            v[i] = work[i * n + j];
        }
        v[j] -= alpha;

        let v_norm_sq: f64 = v[j..].iter().map(|x| x * x).sum();
        if v_norm_sq <= f64::MIN_POSITIVE {
            reflectors.push((v, 0.0));
            continue;
        }
        let beta = 2.0 / v_norm_sq;

        // R <- (I - beta v v^T) R
        for col in j..n {
            let mut dot = 0.0f64;
            for i in j..m {
                dot += v[i] * work[i * n + col];
            }
            let factor = beta * dot;
            for i in j..m {
                work[i * n + col] -= factor * v[i];
            }
        }

        reflectors.push((v, beta));
    }

    // Q = H_0 H_1 ... H_{k-1} applied to the first k columns of the identity.
    let mut q = vec![0.0f64; m * k];
    for i in 0..k {
        q[i * k + i] = 1.0;
    }
    for j in (0..k).rev() {
        let (v, beta) = &reflectors[j];
        if *beta == 0.0 {
            continue;
        }
        for col in 0..k {
            let mut dot = 0.0f64;
            for i in j..m {
                dot += v[i] * q[i * k + col];
            }
            let factor = beta * dot;
            for i in j..m {
                q[i * k + col] -= factor * v[i];
            }
        }
    }

    // R is the leading k x n upper-triangular block of the working matrix.
    let mut r = vec![0.0f64; k * n];
    for i in 0..k {
        for j in i..n {
            r[i * n + j] = work[i * n + j];
        }
    }

    (q, r)
}

// ---------------------------------------------------------------------------
// LU with partial pivoting (used by inverse iteration)
// ---------------------------------------------------------------------------

/// In-place LU factorisation with partial pivoting.
///
/// Pivots whose magnitude falls below `tiny` are replaced by `tiny` (with the
/// original sign). This is the standard LAPACK trick that lets inverse
/// iteration run with an *exact* eigenvalue shift, where the shifted matrix is
/// singular by construction.
fn lu_factor(mut a: Vec<f64>, n: usize, tiny: f64) -> (Vec<f64>, Vec<usize>) {
    let mut perm: Vec<usize> = (0..n).collect();

    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_val = a[k * n + k].abs();
        for i in (k + 1)..n {
            let candidate = a[i * n + k].abs();
            if candidate > pivot_val {
                pivot_val = candidate;
                pivot_row = i;
            }
        }

        if pivot_row != k {
            for j in 0..n {
                a.swap(k * n + j, pivot_row * n + j);
            }
            perm.swap(k, pivot_row);
        }

        if a[k * n + k].abs() < tiny {
            a[k * n + k] = if a[k * n + k] < 0.0 { -tiny } else { tiny };
        }

        let pivot = a[k * n + k];
        for i in (k + 1)..n {
            let factor = a[i * n + k] / pivot;
            a[i * n + k] = factor;
            if factor != 0.0 {
                for j in (k + 1)..n {
                    a[i * n + j] -= factor * a[k * n + j];
                }
            }
        }
    }

    (a, perm)
}

/// Solve `LU x = P b` for a factorisation produced by [`lu_factor`].
fn lu_solve(lu: &[f64], perm: &[usize], n: usize, b: &[f64]) -> Vec<f64> {
    let mut x = vec![0.0f64; n];
    for i in 0..n {
        x[i] = b[perm[i]];
    }
    // Forward substitution (unit lower triangular).
    for i in 1..n {
        let mut sum = x[i];
        for j in 0..i {
            sum -= lu[i * n + j] * x[j];
        }
        x[i] = sum;
    }
    // Back substitution.
    for i in (0..n).rev() {
        let mut sum = x[i];
        for j in (i + 1)..n {
            sum -= lu[i * n + j] * x[j];
        }
        x[i] = sum / lu[i * n + i];
    }
    x
}

fn normalize(v: &mut [f64]) -> f64 {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 && norm.is_finite() {
        for value in v.iter_mut() {
            *value /= norm;
        }
    }
    norm
}

/// Fix the sign/phase of an eigenvector so results are reproducible: the
/// component with the largest magnitude is made positive (real case).
fn canonicalize_sign(v: &mut [f64]) {
    let mut best = 0usize;
    for i in 0..v.len() {
        if v[i].abs() > v[best].abs() {
            best = i;
        }
    }
    if v[best] < 0.0 {
        for value in v.iter_mut() {
            *value = -*value;
        }
    }
}

// ---------------------------------------------------------------------------
// Eigenvectors by inverse iteration
// ---------------------------------------------------------------------------

/// Compute a unit eigenvector for the real eigenvalue `lambda` of `a`.
///
/// Uses shifted inverse iteration with a floored pivot. Returns `None` when the
/// resulting vector does not satisfy `A v = lambda v` to the requested accuracy
/// (defective or badly clustered eigenvalue) so callers can report an honest
/// error instead of a fabricated vector.
pub(crate) fn eigenvector_real(a: &[f64], n: usize, lambda: f64) -> Option<Vec<f64>> {
    let scale = matrix_inf_norm(a, n).max(1.0);
    let tiny = f64::EPSILON * scale;

    let mut shifted = a.to_vec();
    for i in 0..n {
        shifted[i * n + i] -= lambda;
    }
    let (lu, perm) = lu_factor(shifted, n, tiny);

    let mut v: Vec<f64> = (0..n).map(|i| 1.0 / (i as f64 + 1.5)).collect();
    normalize(&mut v);

    for _ in 0..4 {
        let mut w = lu_solve(&lu, &perm, n, &v);
        if w.iter().any(|x| !x.is_finite()) {
            return None;
        }
        if normalize(&mut w) == 0.0 {
            return None;
        }
        v = w;
    }

    // Residual check: ||A v - lambda v||_inf must be small relative to ||A||.
    let mut residual = 0.0f64;
    for i in 0..n {
        let mut av = 0.0f64;
        for j in 0..n {
            av += a[i * n + j] * v[j];
        }
        residual = residual.max((av - lambda * v[i]).abs());
    }
    if residual > 1e-6 * scale {
        return None;
    }

    canonicalize_sign(&mut v);
    Some(v)
}

/// Compute a unit eigenvector for the complex eigenvalue `re + i*im` of the
/// real matrix `a`.
///
/// The complex system `(A - lambda I) v = w` is solved through its equivalent
/// real `2n x 2n` embedding
///
/// ```text
/// [ A - re*I    im*I   ] [ x ]   [ p ]
/// [  -im*I    A - re*I ] [ y ] = [ q ]
/// ```
///
/// with `v = x + i*y`, so the same real LU solver can drive inverse iteration.
/// Returns `None` if the iteration does not converge.
pub(crate) fn eigenvector_complex(
    a: &[f64],
    n: usize,
    re: f64,
    im: f64,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let scale = matrix_inf_norm(a, n).max(1.0);
    let tiny = f64::EPSILON * scale;
    let size = 2 * n;

    let mut m = vec![0.0f64; size * size];
    for i in 0..n {
        for j in 0..n {
            m[i * size + j] = a[i * n + j];
            m[(i + n) * size + (j + n)] = a[i * n + j];
        }
        m[i * size + i] -= re;
        m[(i + n) * size + (i + n)] -= re;
        m[i * size + (i + n)] = im;
        m[(i + n) * size + i] = -im;
    }
    let (lu, perm) = lu_factor(m, size, tiny);

    let mut v: Vec<f64> = (0..size).map(|i| 1.0 / (i as f64 + 1.5)).collect();
    normalize(&mut v);

    for _ in 0..5 {
        let mut w = lu_solve(&lu, &perm, size, &v);
        if w.iter().any(|x| !x.is_finite()) {
            return None;
        }
        if normalize(&mut w) == 0.0 {
            return None;
        }
        v = w;
    }

    let x = v[..n].to_vec();
    let y = v[n..].to_vec();

    // Residual of the complex eigen-equation.
    let mut residual = 0.0f64;
    for i in 0..n {
        let (mut ax, mut ay) = (0.0f64, 0.0f64);
        for j in 0..n {
            ax += a[i * n + j] * x[j];
            ay += a[i * n + j] * y[j];
        }
        let target_re = re * x[i] - im * y[i];
        let target_im = re * y[i] + im * x[i];
        residual = residual.max((ax - target_re).abs().max((ay - target_im).abs()));
    }
    if residual > 1e-6 * scale {
        return None;
    }

    Some((x, y))
}

// ---------------------------------------------------------------------------
// SciRS2-backed dense kernels
// ---------------------------------------------------------------------------

/// Result of a dense SVD: `(u, u_cols, s, vt_rows, vt)` in row-major layout.
pub(crate) struct SvdResult {
    /// Left singular vectors, `m x u_cols`.
    pub u: Vec<f64>,
    /// Number of columns of `u`.
    pub u_cols: usize,
    /// Singular values, descending.
    pub s: Vec<f64>,
    /// Right singular vectors transposed, `vt_rows x n`.
    pub vt: Vec<f64>,
    /// Number of rows of `vt`.
    pub vt_rows: usize,
}

/// Orthonormally complete a set of column vectors to `target` columns.
///
/// Existing columns are assumed orthonormal (columns that are numerically zero
/// are replaced). New directions are taken from the canonical basis and
/// re-orthogonalised twice for stability.
fn complete_basis(columns: &mut Vec<Vec<f64>>, rows: usize, target: usize) {
    let mut candidate = 0usize;
    while columns.len() < target && candidate < rows {
        let mut v = vec![0.0f64; rows];
        v[candidate] = 1.0;
        candidate += 1;

        for _ in 0..2 {
            for existing in columns.iter() {
                let dot: f64 = existing.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                for i in 0..rows {
                    v[i] -= dot * existing[i];
                }
            }
        }

        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-8 {
            for value in v.iter_mut() {
                *value /= norm;
            }
            columns.push(v);
        }
    }
}

/// One-sided Jacobi SVD of a tall (or square) `rows x cols` matrix.
///
/// Returns `(w, v)` where `w` holds the scaled left vectors `A * V` (row-major
/// `rows x cols`) and `v` the right singular vectors (row-major `cols x cols`,
/// column `j` is `v_j`). The singular values are the column norms of `w`.
///
/// One-sided Jacobi is chosen over the normal-equation route (`A^T A` plus a
/// symmetric eigensolver) because it neither squares the condition number nor
/// loses orthogonality on repeated singular values.
fn one_sided_jacobi(a: &[f64], rows: usize, cols: usize) -> (Vec<f64>, Vec<f64>) {
    let mut w = a.to_vec();
    let mut v = vec![0.0f64; cols * cols];
    for i in 0..cols {
        v[i * cols + i] = 1.0;
    }

    let max_sweeps = 60;
    for _ in 0..max_sweeps {
        let mut rotated = false;

        for p in 0..cols.saturating_sub(1) {
            for q in (p + 1)..cols {
                let mut alpha = 0.0f64;
                let mut beta = 0.0f64;
                let mut gamma = 0.0f64;
                for i in 0..rows {
                    let wp = w[i * cols + p];
                    let wq = w[i * cols + q];
                    alpha += wp * wp;
                    beta += wq * wq;
                    gamma += wp * wq;
                }

                if gamma.abs() <= 1e-15 * (alpha * beta).sqrt() || gamma == 0.0 {
                    continue;
                }
                rotated = true;

                let zeta = (beta - alpha) / (2.0 * gamma);
                let sign = if zeta >= 0.0 { 1.0 } else { -1.0 };
                let t = sign / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;

                for i in 0..rows {
                    let wp = w[i * cols + p];
                    let wq = w[i * cols + q];
                    w[i * cols + p] = c * wp - s * wq;
                    w[i * cols + q] = s * wp + c * wq;
                }
                for i in 0..cols {
                    let vp = v[i * cols + p];
                    let vq = v[i * cols + q];
                    v[i * cols + p] = c * vp - s * vq;
                    v[i * cols + q] = s * vp + c * vq;
                }
            }
        }

        if !rotated {
            break;
        }
    }

    (w, v)
}

/// Singular value decomposition of an `m x n` matrix (one-sided Jacobi).
pub(crate) fn dense_svd(
    a: &[f64],
    m: usize,
    n: usize,
    full_matrices: bool,
) -> TorshResult<SvdResult> {
    let k = m.min(n);

    // The kernel needs a tall matrix; for wide input factor the transpose and
    // swap the roles of U and V afterwards.
    let transposed = m < n;
    let (rows, cols) = if transposed { (n, m) } else { (m, n) };
    let work: Vec<f64> = if transposed {
        let mut t = vec![0.0f64; rows * cols];
        for i in 0..m {
            for j in 0..n {
                t[j * cols + i] = a[i * n + j];
            }
        }
        t
    } else {
        a.to_vec()
    };

    let (w, v) = one_sided_jacobi(&work, rows, cols);

    // Column norms are the singular values; sort descending.
    let mut order: Vec<usize> = (0..cols).collect();
    let sigma: Vec<f64> = (0..cols)
        .map(|j| {
            (0..rows)
                .map(|i| w[i * cols + j] * w[i * cols + j])
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    order.sort_by(|&i, &j| {
        sigma[j]
            .partial_cmp(&sigma[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let scale = sigma.iter().fold(0.0f64, |acc, &x| acc.max(x));
    let cutoff = scale * f64::EPSILON * (rows.max(cols) as f64);

    // Left vectors of the (possibly transposed) problem.
    let mut left: Vec<Vec<f64>> = Vec::with_capacity(cols);
    let mut right: Vec<Vec<f64>> = Vec::with_capacity(cols);
    let mut values: Vec<f64> = Vec::with_capacity(cols);
    for &j in &order {
        let s_j = sigma[j];
        values.push(s_j);
        right.push((0..cols).map(|i| v[i * cols + j]).collect());
        if s_j > cutoff && s_j > 0.0 {
            left.push((0..rows).map(|i| w[i * cols + j] / s_j).collect());
        }
    }
    // Rank-deficient input: complete the left basis so U still has orthonormal
    // columns (the corresponding singular values are zero).
    complete_basis(&mut left, rows, cols);

    let left_target = if transposed {
        if full_matrices {
            cols
        } else {
            k
        }
    } else if full_matrices {
        rows
    } else {
        k
    };
    let right_target = if transposed {
        if full_matrices {
            rows
        } else {
            k
        }
    } else if full_matrices {
        cols
    } else {
        k
    };

    if transposed {
        // A = (A^T)^T = right * diag(s) * left^T
        complete_basis(&mut right, cols, left_target);
        complete_basis(&mut left, rows, right_target);
    } else {
        complete_basis(&mut left, rows, left_target);
        complete_basis(&mut right, cols, right_target);
    }

    let (u_cols_vecs, u_rows, vt_cols_vecs, vt_dim) = if transposed {
        (right, cols, left, rows)
    } else {
        (left, rows, right, cols)
    };

    let u_cols = if full_matrices { u_rows } else { k };
    let vt_rows = if full_matrices { vt_dim } else { k };

    let mut u = vec![0.0f64; u_rows * u_cols];
    for (j, column) in u_cols_vecs.iter().take(u_cols).enumerate() {
        for (i, &value) in column.iter().enumerate().take(u_rows) {
            u[i * u_cols + j] = value;
        }
    }

    let mut vt = vec![0.0f64; vt_rows * vt_dim];
    for (j, column) in vt_cols_vecs.iter().take(vt_rows).enumerate() {
        for (i, &value) in column.iter().enumerate().take(vt_dim) {
            vt[j * vt_dim + i] = value;
        }
    }

    values.truncate(k);
    while values.len() < k {
        values.push(0.0);
    }

    Ok(SvdResult {
        u,
        u_cols,
        s: values,
        vt,
        vt_rows,
    })
}

/// Symmetric eigendecomposition by the cyclic Jacobi method.
///
/// Returns `(eigenvalues, eigenvectors)` with the eigenvalues in ascending
/// order and the eigenvectors stored column-wise in a row-major `n x n` buffer.
/// The Jacobi iteration is backward stable and, unlike tridiagonal QL/QR
/// variants, delivers orthonormal vectors even for repeated eigenvalues.
pub(crate) fn dense_symmetric_eig(a: &[f64], n: usize) -> TorshResult<(Vec<f64>, Vec<f64>)> {
    // Work on the symmetric part so tiny asymmetries cannot bias the result.
    let mut m = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            m[i * n + j] = 0.5 * (a[i * n + j] + a[j * n + i]);
        }
    }

    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let frobenius: f64 = m.iter().map(|x| x * x).sum::<f64>().sqrt();
    let threshold = 1e-15 * frobenius.max(1.0);

    for _ in 0..100 {
        let mut off = 0.0f64;
        for p in 0..n {
            for q in (p + 1)..n {
                off += m[p * n + q] * m[p * n + q];
            }
        }
        if off.sqrt() <= threshold {
            break;
        }

        for p in 0..n.saturating_sub(1) {
            for q in (p + 1)..n {
                let apq = m[p * n + q];
                if apq.abs() <= f64::MIN_POSITIVE {
                    continue;
                }

                let theta = (m[q * n + q] - m[p * n + p]) / (2.0 * apq);
                let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
                let t = sign / (theta.abs() + (1.0 + theta * theta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;

                // M <- J^T M J with J = [[c, s], [-s, c]] acting on (p, q).
                for i in 0..n {
                    let mip = m[i * n + p];
                    let miq = m[i * n + q];
                    m[i * n + p] = c * mip - s * miq;
                    m[i * n + q] = s * mip + c * miq;
                }
                for j in 0..n {
                    let mpj = m[p * n + j];
                    let mqj = m[q * n + j];
                    m[p * n + j] = c * mpj - s * mqj;
                    m[q * n + j] = s * mpj + c * mqj;
                }
                for i in 0..n {
                    let vip = v[i * n + p];
                    let viq = v[i * n + q];
                    v[i * n + p] = c * vip - s * viq;
                    v[i * n + q] = s * vip + c * viq;
                }
            }
        }
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        m[i * n + i]
            .partial_cmp(&m[j * n + j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let values: Vec<f64> = order.iter().map(|&i| m[i * n + i]).collect();
    let mut vectors = vec![0.0f64; n * n];
    for (new_col, &old_col) in order.iter().enumerate() {
        for row in 0..n {
            vectors[row * n + new_col] = v[row * n + old_col];
        }
    }

    Ok((values, vectors))
}

#[cfg(feature = "scirs2-integration")]
fn to_array(
    data: &[f64],
    rows: usize,
    cols: usize,
) -> TorshResult<scirs2_core::ndarray::Array2<f64>> {
    scirs2_core::ndarray::Array2::from_shape_vec((rows, cols), data.to_vec())
        .map_err(|e| TorshError::ComputeError(format!("matrix conversion failed: {e}")))
}

/// Eigenvalues of a general (non-symmetric) real matrix as `(real, imaginary)`.
#[cfg(feature = "scirs2-integration")]
pub(crate) fn dense_general_eigenvalues(a: &[f64], n: usize) -> TorshResult<(Vec<f64>, Vec<f64>)> {
    let array = to_array(a, n, n)?;
    let (values, _vectors) = scirs2_linalg::eig(&array.view(), None)
        .map_err(|e| TorshError::ComputeError(format!("eigenvalue solver failed: {e}")))?;
    let re = values.iter().map(|z| z.re).collect();
    let im = values.iter().map(|z| z.im).collect();
    Ok((re, im))
}

/// Eigenvalues of a general matrix (unavailable without `scirs2-integration`).
#[cfg(not(feature = "scirs2-integration"))]
pub(crate) fn dense_general_eigenvalues(
    _a: &[f64],
    _n: usize,
) -> TorshResult<(Vec<f64>, Vec<f64>)> {
    Err(TorshError::ComputeError(
        "eigendecomposition requires the 'scirs2-integration' feature of torsh-linalg".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn householder_qr_reconstructs_and_is_orthogonal() {
        let m = 5;
        let n = 3;
        let a: Vec<f64> = (0..m * n).map(|i| ((i * 37) % 11) as f64 - 5.0).collect();
        let (q, r) = householder_qr(&a, m, n);
        let k = m.min(n);

        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for t in 0..k {
                    acc += q[i * k + t] * r[t * n + j];
                }
                assert!((acc - a[i * n + j]).abs() < 1e-10);
            }
        }

        for i in 0..k {
            for j in 0..k {
                let mut acc = 0.0;
                for t in 0..m {
                    acc += q[t * k + i] * q[t * k + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((acc - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn inverse_iteration_recovers_known_eigenvector() {
        // Upper triangular matrix: eigenvalues are 1, 2, 3.
        let a = vec![1.0, 2.0, 3.0, 0.0, 2.0, 4.0, 0.0, 0.0, 3.0];
        let v = eigenvector_real(&a, 3, 2.0).expect("eigenvector for lambda=2");
        // A v should equal 2 v
        for i in 0..3 {
            let mut av = 0.0;
            for j in 0..3 {
                av += a[i * 3 + j] * v[j];
            }
            assert!((av - 2.0 * v[i]).abs() < 1e-8);
        }
    }

    #[test]
    fn complex_inverse_iteration_recovers_rotation_eigenvector() {
        // [[0, -1], [1, 0]] has eigenvalues +/- i.
        let a = vec![0.0, -1.0, 1.0, 0.0];
        let (x, y) = eigenvector_complex(&a, 2, 0.0, 1.0).expect("complex eigenvector");
        for i in 0..2 {
            let (mut ax, mut ay) = (0.0, 0.0);
            for j in 0..2 {
                ax += a[i * 2 + j] * x[j];
                ay += a[i * 2 + j] * y[j];
            }
            // lambda = i  =>  A v = i v  =>  re = -y, im = x
            assert!((ax + y[i]).abs() < 1e-8);
            assert!((ay - x[i]).abs() < 1e-8);
        }
    }
}
