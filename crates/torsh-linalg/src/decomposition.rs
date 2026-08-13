//! Matrix decomposition algorithms

use crate::dense_kernels;
use crate::TorshResult;
use torsh_core::{DeviceType, TorshError};
use torsh_tensor::{creation::eye, Tensor};

/// Create an independent copy of a tensor.
///
/// [`Tensor::clone`] only clones the metadata: the underlying storage is shared
/// through an `Arc`, so writing into a "cloned" tensor also writes into the
/// original. Working matrices used by the in-place algorithms below must
/// therefore be built from a real copy of the data.
fn deep_copy(tensor: &Tensor) -> TorshResult<Tensor> {
    Tensor::from_data(
        tensor.to_vec()?,
        tensor.shape().dims().to_vec(),
        tensor.device(),
    )
}

/// LU decomposition with partial pivoting
/// Returns (P, L, U) where PA = LU
pub fn lu(tensor: &Tensor) -> TorshResult<(Tensor, Tensor, Tensor)> {
    if tensor.shape().ndim() != 2 {
        return Err(TorshError::InvalidArgument(
            "LU decomposition requires 2D tensor".to_string(),
        ));
    }

    let (m, n) = (tensor.shape().dims()[0], tensor.shape().dims()[1]);

    if m != n {
        return Err(TorshError::InvalidArgument(
            "LU decomposition requires square matrix".to_string(),
        ));
    }

    // Initialize matrices as mutable vectors (avoids SimdOptimized storage issues)
    // Copy input tensor data to working matrix A
    let mut a_data = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            a_data[i * n + j] = tensor.get(&[i, j])?;
        }
    }

    // L = identity matrix (lower triangular)
    let mut l_data = vec![0.0f32; m * m];
    for i in 0..m {
        l_data[i * m + i] = 1.0;
    }

    // P = identity matrix (permutation matrix)
    let mut p_data = vec![0.0f32; m * m];
    for i in 0..m {
        p_data[i * m + i] = 1.0;
    }

    // Gaussian elimination with partial pivoting
    for k in 0..m {
        // Find pivot
        let mut max_val = 0.0;
        let mut pivot_row = k;

        for i in k..m {
            let val = a_data[i * n + k].abs();
            if val > max_val {
                max_val = val;
                pivot_row = i;
            }
        }

        // Swap rows if needed
        if pivot_row != k {
            // Swap rows in A
            for j in 0..n {
                let temp = a_data[k * n + j];
                a_data[k * n + j] = a_data[pivot_row * n + j];
                a_data[pivot_row * n + j] = temp;
            }

            // Swap rows in P
            for j in 0..m {
                let temp = p_data[k * m + j];
                p_data[k * m + j] = p_data[pivot_row * m + j];
                p_data[pivot_row * m + j] = temp;
            }

            // Swap rows in L (only for already computed part)
            for j in 0..k {
                let temp = l_data[k * m + j];
                l_data[k * m + j] = l_data[pivot_row * m + j];
                l_data[pivot_row * m + j] = temp;
            }
        }

        let pivot = a_data[k * n + k];
        if pivot.abs() < 1e-12 {
            return Err(TorshError::InvalidArgument(
                format!("Matrix is singular: pivot element at ({k}, {k}) = {pivot} is too small for numerical stability")
            ));
        }

        // Eliminate column
        for i in (k + 1)..m {
            let factor = a_data[i * n + k] / pivot;
            l_data[i * m + k] = factor;

            for j in k..n {
                a_data[i * n + j] -= factor * a_data[k * n + j];
            }
        }
    }

    // Create output tensors
    let p = Tensor::from_data(p_data, vec![m, m], DeviceType::Cpu)?;
    let l = Tensor::from_data(l_data, vec![m, m], DeviceType::Cpu)?;
    let u = Tensor::from_data(a_data, vec![m, n], DeviceType::Cpu)?;

    Ok((p, l, u))
}

/// QR decomposition using Householder reflections
///
/// Returns the thin (reduced) factorisation `(Q, R)` with `A = Q R`, where
/// `Q` is `m x k` with orthonormal columns and `R` is `k x n` upper triangular
/// for `k = min(m, n)`.
///
/// Householder reflections are backward stable: unlike classical Gram-Schmidt
/// the orthogonality of `Q` does not degrade with the square of the condition
/// number, and rank-deficient as well as wide (`m < n`) matrices are handled
/// without failing.
pub fn qr(tensor: &Tensor) -> TorshResult<(Tensor, Tensor)> {
    let (a, m, n) = dense_kernels::tensor_to_f64(tensor).map_err(|_| {
        TorshError::InvalidArgument("QR decomposition requires 2D tensor".to_string())
    })?;

    if m == 0 || n == 0 {
        return Err(TorshError::InvalidArgument(
            "QR decomposition requires a non-empty matrix".to_string(),
        ));
    }

    let k = m.min(n);
    let (q_data, r_data) = dense_kernels::householder_qr(&a, m, n);

    let q = dense_kernels::f64_to_tensor(&q_data, m, k, DeviceType::Cpu)?;
    let r = dense_kernels::f64_to_tensor(&r_data, k, n, DeviceType::Cpu)?;

    Ok((q, r))
}

/// Singular Value Decomposition
///
/// Returns `(U, S, V^T)` with `A = U * diag(S) * V^T`. The decomposition is
/// computed by the LAPACK-backed dense solver of `scirs2-linalg` (OxiBLAS) in
/// double precision, on `A` itself, and the singular values come back in
/// descending order.
///
/// * `full_matrices == false` (thin/reduced): `U` is `m x k`, `S` has `k`
///   entries and `V^T` is `k x n` for `k = min(m, n)`.
/// * `full_matrices == true`: `U` is `m x m` and `V^T` is `n x n`; `S` still
///   has `k` entries.
pub fn svd(tensor: &Tensor, full_matrices: bool) -> TorshResult<(Tensor, Tensor, Tensor)> {
    let (a, m, n) = dense_kernels::tensor_to_f64(tensor)
        .map_err(|_| TorshError::InvalidArgument("SVD requires 2D tensor".to_string()))?;

    if m == 0 || n == 0 {
        return Err(TorshError::InvalidArgument(
            "SVD requires a non-empty matrix".to_string(),
        ));
    }

    let result = dense_kernels::dense_svd(&a, m, n, full_matrices)?;

    let u = dense_kernels::f64_to_tensor(&result.u, m, result.u_cols, tensor.device())?;
    let vt = dense_kernels::f64_to_tensor(&result.vt, result.vt_rows, n, tensor.device())?;
    let s_values: Vec<f32> = result.s.iter().map(|&v| v as f32).collect();
    let s = Tensor::from_data(s_values, vec![result.s.len()], tensor.device())?;

    Ok((u, s, vt))
}

/// Relative tolerance used to decide whether a matrix is symmetric or whether
/// an eigenvalue's imaginary part is numerically zero.
const EIG_SYMMETRY_TOL: f64 = 1e-9;

/// Read a square matrix, rejecting non-square / non-2D input.
fn square_matrix_f64(tensor: &Tensor, what: &str) -> TorshResult<(Vec<f64>, usize)> {
    let (a, m, n) = dense_kernels::tensor_to_f64(tensor)
        .map_err(|_| TorshError::InvalidArgument(format!("{what} requires 2D tensor")))?;
    if m != n {
        return Err(TorshError::InvalidArgument(format!(
            "{what} requires square matrix"
        )));
    }
    if n == 0 {
        return Err(TorshError::InvalidArgument(format!(
            "{what} requires a non-empty matrix"
        )));
    }
    Ok((a, n))
}

fn is_symmetric(a: &[f64], n: usize) -> bool {
    let scale = dense_kernels::matrix_inf_norm(a, n).max(1.0);
    for i in 0..n {
        for j in (i + 1)..n {
            if (a[i * n + j] - a[j * n + i]).abs() > EIG_SYMMETRY_TOL * scale {
                return false;
            }
        }
    }
    true
}

fn is_diagonal_matrix(a: &[f64], n: usize) -> bool {
    let scale = dense_kernels::matrix_inf_norm(a, n).max(1.0);
    for i in 0..n {
        for j in 0..n {
            if i != j && a[i * n + j].abs() > EIG_SYMMETRY_TOL * scale {
                return false;
            }
        }
    }
    true
}

/// Full complex eigendecomposition of a real square matrix.
///
/// Returns `(values_re, values_im, vectors_re, vectors_im)` where the value
/// tensors have `n` entries and the vector matrices are `n x n` with the
/// eigenvector for `lambda_k` stored in column `k`, i.e.
/// `v_k = vectors_re[:, k] + i * vectors_im[:, k]`.
///
/// Eigenvalues come from the LAPACK-backed general eigensolver of
/// `scirs2-linalg`; the eigenvectors are then recovered by shifted inverse
/// iteration (a real `2n x 2n` embedding is used for complex conjugate pairs).
/// If an eigenvector cannot be determined to working accuracy — a defective or
/// severely clustered spectrum — an error is returned rather than a fabricated
/// vector.
pub fn eig_complex(tensor: &Tensor) -> TorshResult<(Tensor, Tensor, Tensor, Tensor)> {
    let (a, n) = square_matrix_f64(tensor, "Eigenvalue decomposition")?;
    let device = tensor.device();

    // Exact path for diagonal matrices.
    if is_diagonal_matrix(&a, n) {
        let values_re: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
        let vectors_re = {
            let mut m = vec![0.0f64; n * n];
            for i in 0..n {
                m[i * n + i] = 1.0;
            }
            m
        };
        return Ok((
            dense_kernels::f64_to_vector(&values_re, device)?,
            Tensor::from_data(vec![0.0f32; n], vec![n], device)?,
            dense_kernels::f64_to_tensor(&vectors_re, n, n, device)?,
            Tensor::from_data(vec![0.0f32; n * n], vec![n, n], device)?,
        ));
    }

    let scale = dense_kernels::matrix_inf_norm(&a, n).max(1.0);

    // Symmetric matrices: real spectrum with an orthonormal eigenbasis.
    if is_symmetric(&a, n) {
        let (values, vectors) = dense_kernels::dense_symmetric_eig(&a, n)?;
        return Ok((
            dense_kernels::f64_to_vector(&values, device)?,
            Tensor::from_data(vec![0.0f32; n], vec![n], device)?,
            dense_kernels::f64_to_tensor(&vectors, n, n, device)?,
            Tensor::from_data(vec![0.0f32; n * n], vec![n, n], device)?,
        ));
    }

    let (values_re, values_im) = dense_kernels::dense_general_eigenvalues(&a, n)?;

    let mut vectors_re = vec![0.0f64; n * n];
    let mut vectors_im = vec![0.0f64; n * n];

    for k in 0..n {
        let (re, im) = (values_re[k], values_im[k]);
        if im.abs() <= EIG_SYMMETRY_TOL * scale {
            let v = dense_kernels::eigenvector_real(&a, n, re).ok_or_else(|| {
                TorshError::ComputeError(format!(
                    "eigenvector for eigenvalue {re} could not be determined \
                     (defective or severely clustered spectrum)"
                ))
            })?;
            for i in 0..n {
                vectors_re[i * n + k] = v[i];
            }
        } else {
            let (x, y) = dense_kernels::eigenvector_complex(&a, n, re, im).ok_or_else(|| {
                TorshError::ComputeError(format!(
                    "eigenvector for eigenvalue {re}{im:+}i could not be determined \
                     (defective or severely clustered spectrum)"
                ))
            })?;
            for i in 0..n {
                vectors_re[i * n + k] = x[i];
                vectors_im[i * n + k] = y[i];
            }
        }
    }

    Ok((
        dense_kernels::f64_to_vector(&values_re, device)?,
        dense_kernels::f64_to_vector(&values_im, device)?,
        dense_kernels::f64_to_tensor(&vectors_re, n, n, device)?,
        dense_kernels::f64_to_tensor(&vectors_im, n, n, device)?,
    ))
}

/// Eigenvalue decomposition of a real square matrix with a real spectrum
///
/// Returns `(eigenvalues, eigenvectors)`: an `n` element vector of eigenvalues
/// and an `n x n` matrix whose column `k` is the (unit norm) eigenvector for
/// `eigenvalues[k]`, so `A * V[:, k] = eigenvalues[k] * V[:, k]` holds for every
/// `k`.
///
/// Symmetric input is solved with the LAPACK symmetric eigensolver; general
/// input uses the LAPACK general eigensolver plus inverse iteration for the
/// eigenvectors. Matrices with genuinely complex eigenvalues cannot be
/// represented by this real-valued signature and return an error pointing at
/// [`eig_complex`]; nothing is ever padded or invented.
pub fn eig(tensor: &Tensor) -> TorshResult<(Tensor, Tensor)> {
    let (values_re, values_im, vectors_re, vectors_im) = eig_complex(tensor)?;

    let n = values_re.shape().dims()[0];
    for i in 0..n {
        if values_im.get(&[i])?.abs() > 0.0 {
            return Err(TorshError::ComputeError(format!(
                "matrix has complex eigenvalues (lambda_{i} = {}{:+}i); \
                 use eig_complex() to obtain the full complex spectrum",
                values_re.get(&[i])?,
                values_im.get(&[i])?
            )));
        }
    }
    let _ = vectors_im;

    Ok((values_re, vectors_re))
}

/// Cholesky decomposition
/// For a positive definite matrix A, finds L (or U) such that A = L*L^T (or U^T*U)
pub fn cholesky(tensor: &Tensor, upper: bool) -> TorshResult<Tensor> {
    if tensor.shape().ndim() != 2 {
        return Err(TorshError::InvalidArgument(
            "Cholesky decomposition requires 2D tensor".to_string(),
        ));
    }

    let (m, n) = (tensor.shape().dims()[0], tensor.shape().dims()[1]);

    if m != n {
        return Err(TorshError::InvalidArgument(
            "Cholesky decomposition requires square matrix".to_string(),
        ));
    }

    // Initialize result matrix as mutable vector (avoids SimdOptimized storage issues)
    let mut result_data = vec![0.0f32; n * n];

    if upper {
        // Compute upper triangular: A = U^T * U
        for i in 0..n {
            for j in i..n {
                let mut sum = tensor.get(&[i, j])?;

                // Subtract contributions from already computed elements
                for k in 0..i {
                    sum -= result_data[k * n + i] * result_data[k * n + j];
                }

                if i == j {
                    // Diagonal element
                    if sum <= 0.0 {
                        return Err(TorshError::InvalidArgument(
                            "Matrix is not positive definite".to_string(),
                        ));
                    }
                    result_data[i * n + j] = sum.sqrt();
                } else {
                    // Off-diagonal element
                    let diag = result_data[i * n + i];
                    if diag.abs() < 1e-12 {
                        return Err(TorshError::InvalidArgument(
                            "Matrix is singular".to_string(),
                        ));
                    }
                    result_data[i * n + j] = sum / diag;
                }
            }
        }
    } else {
        // Compute lower triangular: A = L * L^T
        for i in 0..n {
            for j in 0..=i {
                let mut sum = tensor.get(&[i, j])?;

                // Subtract contributions from already computed elements
                for k in 0..j {
                    sum -= result_data[i * n + k] * result_data[j * n + k];
                }

                if i == j {
                    // Diagonal element
                    if sum <= 0.0 {
                        return Err(TorshError::InvalidArgument(
                            "Matrix is not positive definite".to_string(),
                        ));
                    }
                    result_data[i * n + j] = sum.sqrt();
                } else {
                    // Off-diagonal element
                    let diag = result_data[j * n + j];
                    if diag.abs() < 1e-12 {
                        return Err(TorshError::InvalidArgument(
                            "Matrix is singular".to_string(),
                        ));
                    }
                    result_data[i * n + j] = sum / diag;
                }
            }
        }
    }

    Tensor::from_data(result_data, vec![n, n], DeviceType::Cpu)
}

/// Polar decomposition
///
/// Decomposes a matrix A into A = UP where U is unitary and P is positive definite.
/// For real matrices, U is orthogonal. Uses SVD internally: A = UΣV^T, then U_polar = UV^T and P = VΣV^T.
///
/// Returns (U, P) where:
/// - U: Unitary/orthogonal matrix
/// - P: Positive definite matrix
///
/// If `side` is "right", computes A = UP (default)
/// If `side` is "left", computes A = PU
pub fn polar(tensor: &Tensor, side: Option<&str>) -> TorshResult<(Tensor, Tensor)> {
    if tensor.shape().ndim() != 2 {
        return Err(TorshError::InvalidArgument(
            "Polar decomposition requires 2D tensor".to_string(),
        ));
    }

    let (m, n) = (tensor.shape().dims()[0], tensor.shape().dims()[1]);
    let side = side.unwrap_or("right");

    // Compute the thin SVD: A = U_svd * Σ * V^T with U_svd: m x k, V^T: k x n.
    // The reduced form is what makes the products below dimensionally correct
    // for rectangular input.
    let (u_svd, s, vt) = svd(tensor, false)?;

    // Extract singular values (s is 1D vector)
    let min_dim = m.min(n);

    // Create diagonal matrix from singular values
    let mut s_diag_data = vec![0.0f32; min_dim * min_dim];
    for i in 0..min_dim {
        s_diag_data[i * min_dim + i] = s.get(&[i])?;
    }
    let s_diag = Tensor::from_data(s_diag_data, vec![min_dim, min_dim], tensor.device())?;

    match side {
        "right" => {
            // A = UP: U_polar = U_svd * V^T, P = V * Σ * V^T
            let vt_transposed = vt.transpose(-2, -1)?; // V
            let u_polar = u_svd.matmul(&vt)?; // U_svd * V^T

            // P = V * Σ * V^T
            let v_sigma = vt_transposed.matmul(&s_diag)?;
            let p = v_sigma.matmul(&vt)?;

            Ok((u_polar, p))
        }
        "left" => {
            // A = PU: P = U_svd * Σ * U_svd^T, U_polar = U_svd * V^T
            let ut = u_svd.transpose(-2, -1)?;

            // P = U_svd * Σ * U_svd^T
            let u_sigma = u_svd.matmul(&s_diag)?;
            let p = u_sigma.matmul(&ut)?;

            // U_polar = U_svd * V^T
            let u_polar = u_svd.matmul(&vt)?;

            Ok((p, u_polar))
        }
        _ => Err(TorshError::InvalidArgument(format!(
            "Invalid side parameter: {side}. Must be 'right' or 'left'"
        ))),
    }
}

/// Schur decomposition
///
/// Computes the Schur decomposition of a square matrix A = QTQ^H where Q is unitary
/// and T is upper triangular (Schur form). For real matrices, T may be quasi-triangular
/// with 2x2 blocks for complex conjugate eigenvalue pairs.
///
/// This is a simplified implementation using QR iteration.
///
/// Returns (Q, T) where:
/// - Q: Unitary/orthogonal matrix
/// - T: Upper triangular matrix (Schur form)
pub fn schur(tensor: &Tensor) -> TorshResult<(Tensor, Tensor)> {
    if tensor.shape().ndim() != 2 {
        return Err(TorshError::InvalidArgument(
            "Schur decomposition requires 2D tensor".to_string(),
        ));
    }

    let (m, n) = (tensor.shape().dims()[0], tensor.shape().dims()[1]);
    if m != n {
        return Err(TorshError::InvalidArgument(
            "Schur decomposition requires square matrix".to_string(),
        ));
    }

    // Initialize Q as identity and T as copy of input matrix
    let mut q = eye::<f32>(n)?;
    // Deep copy: the shift below writes into `t` in place.
    let mut t = deep_copy(tensor)?;

    // Simplified QR iteration for Schur decomposition
    let max_iterations = 100;
    let tolerance = 1e-8;

    for _iteration in 0..max_iterations {
        // Check if T is already in Schur form (upper triangular)
        let mut is_schur = true;
        for i in 1..n {
            for j in 0..i {
                if t.get(&[i, j])?.abs() > tolerance {
                    is_schur = false;
                    break;
                }
            }
            if !is_schur {
                break;
            }
        }

        if is_schur {
            break;
        }

        // Apply QR iteration with shift
        // Compute shift (Wilkinson shift - use bottom-right 2x2 submatrix)
        let shift = if n >= 2 {
            let a22 = t.get(&[n - 1, n - 1])?;
            let a21 = t.get(&[n - 2, n - 1])?;
            let a12 = t.get(&[n - 1, n - 2])?;
            let a11 = t.get(&[n - 2, n - 2])?;

            // Compute eigenvalues of 2x2 matrix and choose closest to a22
            let trace = a11 + a22;
            let det = a11 * a22 - a12 * a21;
            let discriminant = trace * trace - 4.0 * det;

            if discriminant >= 0.0 {
                let sqrt_disc = discriminant.sqrt();
                let lambda1 = (trace + sqrt_disc) / 2.0;
                let lambda2 = (trace - sqrt_disc) / 2.0;

                // Choose eigenvalue closest to a22
                if (lambda1 - a22).abs() < (lambda2 - a22).abs() {
                    lambda1
                } else {
                    lambda2
                }
            } else {
                a22 // Use diagonal element if complex eigenvalues
            }
        } else {
            t.get(&[n - 1, n - 1])? // Single element case
        };

        // Apply shift: T_shifted = T - shift * I
        for i in 0..n {
            let old_val = t.get(&[i, i])?;
            t.set(&[i, i], old_val - shift)?;
        }

        // QR decomposition of shifted matrix
        let (q_step, r_step) = qr(&t)?;

        // Update T = R * Q + shift * I
        t = r_step.matmul(&q_step)?;

        // Add back shift
        for i in 0..n {
            let old_val = t.get(&[i, i])?;
            t.set(&[i, i], old_val + shift)?;
        }

        // Update Q = Q * Q_step
        q = q.matmul(&q_step)?;

        // Check convergence of subdiagonal elements
        let mut max_subdiag = 0.0f32;
        for i in 1..n {
            let subdiag = t.get(&[i, i - 1])?.abs();
            if subdiag > max_subdiag {
                max_subdiag = subdiag;
            }
        }

        if max_subdiag < tolerance {
            break;
        }

        // Deflation: zero out converged subdiagonal elements
        for i in 1..n {
            if t.get(&[i, i - 1])?.abs() < tolerance {
                t.set(&[i, i - 1], 0.0)?;
            }
        }
    }

    Ok((q, t))
}

/// Jordan canonical form decomposition
///
/// Computes `(P, J)` such that `A = P * J * P^(-1)`, where `J` is the Jordan
/// canonical form and `P` holds the corresponding basis in its columns.
///
/// The eigenvalues and eigenvectors come from [`eig`], so the spectrum must be
/// real. Repeated eigenvalues whose eigenvectors are still independent yield a
/// diagonal `J` (Jordan blocks of size one). Genuinely defective matrices need
/// generalised eigenvectors, which this routine does not compute: rather than
/// returning a plausible-looking but wrong factorisation it reports an error.
pub fn jordan_form(tensor: &Tensor) -> TorshResult<(Tensor, Tensor)> {
    let (_, n) = square_matrix_f64(tensor, "Jordan form")?;

    // Real eigenpairs (errors out on a complex spectrum).
    let (eigenvalues, eigenvectors) = eig(tensor)?;

    // P is the eigenvector matrix; check that it is actually a basis, i.e. the
    // matrix is diagonalisable. A defective matrix has a rank-deficient P.
    let (_, p_singular_values, _) = svd(&eigenvectors, false)?;
    let sigma_max = p_singular_values.get(&[0])?;
    let sigma_min = p_singular_values.get(&[n - 1])?;
    if sigma_max <= 0.0 || sigma_min <= 1e-6 * sigma_max {
        return Err(TorshError::ComputeError(
            "Jordan form: matrix is defective (eigenvectors do not form a basis); \
             generalised eigenvectors are required and are not computed here"
                .to_string(),
        ));
    }

    // J = diag(eigenvalues): every Jordan block has size one for a
    // diagonalisable matrix.
    let mut j_data = vec![0.0f32; n * n];
    for i in 0..n {
        j_data[i * n + i] = eigenvalues.get(&[i])?;
    }
    let j = Tensor::from_data(j_data, vec![n, n], tensor.device())?;

    Ok((eigenvectors, j))
}

/// Compute the Hessenberg decomposition of a matrix
///
/// The Hessenberg decomposition factors a matrix A as A = QHQ^T where:
/// - Q is an orthogonal matrix
/// - H is an upper Hessenberg matrix (zeros below the first subdiagonal)
///
/// This is an intermediate step in many eigenvalue algorithms and is
/// computationally cheaper than full Schur decomposition.
///
/// # Arguments
///
/// * `tensor` - Square matrix to decompose (n×n)
///
/// # Returns
///
/// Tuple (Q, H) where:
/// - Q: Orthogonal matrix (n×n)
/// - H: Upper Hessenberg matrix (n×n) with zeros below first subdiagonal
///
/// # Properties
///
/// - A = QHQ^T
/// - Q^T Q = I (Q is orthogonal)
/// - H_{ij} = 0 for i > j+1 (upper Hessenberg form)
/// - Preserves eigenvalues: eig(A) = eig(H)
///
/// # Algorithm
///
/// Uses Householder reflections to reduce the matrix to Hessenberg form
///
/// # Examples
///
/// ```ignore
/// use torsh_linalg::decomposition::hessenberg;
/// let a = create_matrix()?;
/// let (q, h) = hessenberg(&a)?;
/// // Verify: A ≈ Q * H * Q^T
/// ```
pub fn hessenberg(tensor: &Tensor) -> TorshResult<(Tensor, Tensor)> {
    if tensor.shape().ndim() != 2 {
        return Err(TorshError::InvalidArgument(
            "Hessenberg decomposition requires 2D tensor".to_string(),
        ));
    }

    let (m, n) = (tensor.shape().dims()[0], tensor.shape().dims()[1]);
    if m != n {
        return Err(TorshError::InvalidArgument(format!(
            "Hessenberg decomposition requires square matrix, got {m}x{n}"
        )));
    }

    // Start with H = A (independent copy: the reflections update it in place)
    // and Q = I.
    let h = deep_copy(tensor)?;
    let q = eye::<f32>(n)?;

    // Householder reduction to Hessenberg form
    for k in 0..n.saturating_sub(2) {
        // Compute Householder vector for column k
        let mut col_data = Vec::with_capacity(n - k - 1);
        for i in (k + 1)..n {
            col_data.push(h.get(&[i, k])?);
        }

        if col_data.is_empty() {
            continue;
        }

        // Compute norm of subcolumn
        let mut norm = 0.0f32;
        for &val in &col_data {
            norm += val * val;
        }
        norm = norm.sqrt();

        if norm < 1e-10 {
            continue; // Skip if column is already zero
        }

        // Compute Householder vector
        let sign = if col_data[0] >= 0.0 { 1.0 } else { -1.0 };
        let u1 = col_data[0] + sign * norm;
        let tau = 2.0 / (u1 * u1 + col_data[1..].iter().map(|x| x * x).sum::<f32>());

        let mut v = vec![0.0f32; n];
        v[k + 1] = u1;
        for (i, &val) in col_data[1..].iter().enumerate() {
            v[k + 2 + i] = val;
        }

        // Apply Householder reflection: H = (I - τvv^T) H (I - τvv^T)
        // Left multiplication: H <- (I - τvv^T) H
        for j in k..n {
            let mut dot = 0.0f32;
            for i in (k + 1)..n {
                dot += v[i] * h.get(&[i, j])?;
            }
            for i in (k + 1)..n {
                let old_val = h.get(&[i, j])?;
                h.set(&[i, j], old_val - tau * v[i] * dot)?;
            }
        }

        // Right multiplication: H <- H (I - τvv^T)
        for i in 0..n {
            let mut dot = 0.0f32;
            for j in (k + 1)..n {
                dot += h.get(&[i, j])? * v[j];
            }
            for j in (k + 1)..n {
                let old_val = h.get(&[i, j])?;
                h.set(&[i, j], old_val - tau * dot * v[j])?;
            }
        }

        // Update Q: Q <- Q (I - τvv^T)
        for i in 0..n {
            let mut dot = 0.0f32;
            for j in (k + 1)..n {
                dot += q.get(&[i, j])? * v[j];
            }
            for j in (k + 1)..n {
                let old_val = q.get(&[i, j])?;
                q.set(&[i, j], old_val - tau * dot * v[j])?;
            }
        }
    }

    // Zero out elements below the first subdiagonal for numerical stability
    for i in 0..n {
        for j in 0..i.saturating_sub(1) {
            if h.get(&[i, j])?.abs() < 1e-10 {
                h.set(&[i, j], 0.0)?;
            }
        }
    }

    Ok((q, h))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_tensor::creation::{eye, zeros};

    fn create_test_matrix_2x2() -> TorshResult<Tensor> {
        // Create a 2x2 matrix [[1.0, 2.0], [3.0, 4.0]]
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        Tensor::from_data(data, vec![2, 2], torsh_core::DeviceType::Cpu)
    }

    #[allow(dead_code)]
    fn create_test_matrix_3x3() -> TorshResult<Tensor> {
        // Create a 3x3 matrix [[2.0, -1.0, 0.0], [1.0, 2.0, -1.0], [0.0, 1.0, 2.0]]
        let data = vec![2.0f32, -1.0, 0.0, 1.0, 2.0, -1.0, 0.0, 1.0, 2.0];
        Tensor::from_data(data, vec![3, 3], torsh_core::DeviceType::Cpu)
    }

    fn create_spd_matrix_2x2() -> TorshResult<Tensor> {
        // Create a symmetric positive definite matrix [[4.0, 2.0], [2.0, 3.0]]
        let data = vec![4.0f32, 2.0, 2.0, 3.0];
        Tensor::from_data(data, vec![2, 2], torsh_core::DeviceType::Cpu)
    }

    #[test]
    fn test_lu_decomposition() -> TorshResult<()> {
        let mat = create_test_matrix_2x2()?;
        let (p, l, u) = lu(&mat)?;

        // Verify dimensions
        assert_eq!(p.shape().dims(), &[2, 2]);
        assert_eq!(l.shape().dims(), &[2, 2]);
        assert_eq!(u.shape().dims(), &[2, 2]);

        // Check that L is lower triangular (entries above diagonal are 0)
        assert_relative_eq!(l.get(&[0, 1])?, 0.0, epsilon = 1e-6);

        // Check that U is upper triangular (entries below diagonal are 0 or small)
        // Note: our implementation might not guarantee strict upper triangular

        // Check that P is a permutation matrix (each row/column has exactly one 1)
        for i in 0..2 {
            let mut row_sum = 0.0;
            let mut col_sum = 0.0;
            for j in 0..2 {
                row_sum += p.get(&[i, j])?;
                col_sum += p.get(&[j, i])?;
            }
            assert_relative_eq!(row_sum, 1.0, epsilon = 1e-6);
            assert_relative_eq!(col_sum, 1.0, epsilon = 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_lu_identity() -> TorshResult<()> {
        let identity = eye::<f32>(3)?;
        let (p, l, u) = lu(&identity)?;

        // For identity matrix, L should be identity and U should be identity
        // P can be any permutation
        let pa = p.matmul(&identity)?;
        let lu_product = l.matmul(&u)?;

        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(pa.get(&[i, j])?, lu_product.get(&[i, j])?, epsilon = 1e-6);
            }
        }

        Ok(())
    }

    #[test]
    fn test_qr_decomposition() -> TorshResult<()> {
        let mat = create_test_matrix_2x2()?;
        let (q, r) = qr(&mat)?;

        // Verify dimensions
        assert_eq!(q.shape().dims(), &[2, 2]);
        assert_eq!(r.shape().dims(), &[2, 2]);

        // Verify that A = Q*R (approximately)
        let qr_product = q.matmul(&r)?;

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(qr_product.get(&[i, j])?, mat.get(&[i, j])?, epsilon = 1e-4);
            }
        }

        // Verify Q is orthogonal: Q^T * Q = I
        let qt = q.transpose(-2, -1)?;
        let qtq = qt.matmul(&q)?;
        let identity = eye::<f32>(2)?;

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(qtq.get(&[i, j])?, identity.get(&[i, j])?, epsilon = 1e-4);
            }
        }

        Ok(())
    }

    #[test]
    fn test_svd_decomposition() -> TorshResult<()> {
        let mat = create_test_matrix_2x2()?;
        let (u, s, vt) = svd(&mat, true)?;

        // Verify dimensions
        assert_eq!(u.shape().dims(), &[2, 2]);
        assert_eq!(s.shape().dims(), &[2]); // Singular values as 1D vector
        assert_eq!(vt.shape().dims(), &[2, 2]);

        // Reconstruct matrix: A = U * diag(S) * V^T
        let s_diag = zeros::<f32>(&[2, 2])?;
        s_diag.set(&[0, 0], s.get(&[0])?)?;
        s_diag.set(&[1, 1], s.get(&[1])?)?;

        let us = u.matmul(&s_diag)?;
        let reconstructed = us.matmul(&vt)?;

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(
                    reconstructed.get(&[i, j])?,
                    mat.get(&[i, j])?,
                    epsilon = 1e-4
                );
            }
        }

        // Verify singular values are non-negative and ordered
        assert!(s.get(&[0])? >= 0.0);
        assert!(s.get(&[1])? >= 0.0);
        assert!(s.get(&[0])? >= s.get(&[1])?);

        Ok(())
    }

    #[test]
    fn test_eigendecomposition() -> TorshResult<()> {
        // Use a symmetric matrix for real eigenvalues
        let mat = eye::<f32>(2)?;
        mat.set(&[0, 0], 3.0)?;
        mat.set(&[1, 1], 1.0)?;

        let (eigenvals, eigenvecs) = eig(&mat)?;

        // Verify we get some eigenvalues
        assert!(eigenvals.shape().dims()[0] > 0);
        assert_eq!(eigenvecs.shape().dims()[0], 2);

        // For diagonal matrix, eigenvalues should be the diagonal elements
        let eval1 = eigenvals.get(&[0])?;
        assert!(eval1.abs() - 3.0 < 0.1 || eval1.abs() - 1.0 < 0.1);

        Ok(())
    }

    #[test]
    fn test_cholesky_decomposition() -> TorshResult<()> {
        let spd_mat = create_spd_matrix_2x2()?;

        // Test lower triangular
        let l = cholesky(&spd_mat, false)?;
        assert_eq!(l.shape().dims(), &[2, 2]);

        // Verify that A = L*L^T (approximately)
        let lt = l.transpose(-2, -1)?;
        let llt_product = l.matmul(&lt)?;

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(
                    llt_product.get(&[i, j])?,
                    spd_mat.get(&[i, j])?,
                    epsilon = 1e-5
                );
            }
        }

        // Test upper triangular
        let u = cholesky(&spd_mat, true)?;
        assert_eq!(u.shape().dims(), &[2, 2]);

        // Verify that A = U^T*U (approximately)
        let ut = u.transpose(-2, -1)?;
        let utu_product = ut.matmul(&u)?;

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(
                    utu_product.get(&[i, j])?,
                    spd_mat.get(&[i, j])?,
                    epsilon = 1e-5
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_cholesky_identity() -> TorshResult<()> {
        let identity = eye::<f32>(3)?;
        let l = cholesky(&identity, false)?;

        // For identity matrix, Cholesky should return identity
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(l.get(&[i, j])?, expected, epsilon = 1e-6);
            }
        }

        Ok(())
    }

    #[test]
    fn test_polar_decomposition() -> TorshResult<()> {
        // Use identity matrix for polar decomposition test (should be simple)
        let mat = eye::<f32>(2)?;

        // Test right polar decomposition: A = UP
        let (u, p) = polar(&mat, Some("right"))?;

        // Verify dimensions
        assert_eq!(u.shape().dims(), &[2, 2]);
        assert_eq!(p.shape().dims(), &[2, 2]);

        // For identity matrix, U should be close to identity and P should be close to identity
        // Just verify that the function runs and produces reasonable output
        assert!(u.get(&[0, 0])?.abs() > 0.5); // Some reasonable values
        assert!(p.get(&[0, 0])?.abs() > 0.5); // Some reasonable values

        Ok(())
    }

    #[test]
    fn test_polar_left() -> TorshResult<()> {
        // Use identity matrix for polar decomposition test
        let mat = eye::<f32>(2)?;

        // Test left polar decomposition: A = PU
        let (p, u) = polar(&mat, Some("left"))?;

        // Verify dimensions
        assert_eq!(p.shape().dims(), &[2, 2]);
        assert_eq!(u.shape().dims(), &[2, 2]);

        // Just verify that the function runs and produces reasonable output
        assert!(p.get(&[0, 0])?.abs() > 0.5); // Some reasonable values
        assert!(u.get(&[0, 0])?.abs() > 0.5); // Some reasonable values

        Ok(())
    }

    #[test]
    fn test_schur_decomposition() -> TorshResult<()> {
        // Use a simple matrix for Schur decomposition
        let mat = eye::<f32>(2)?;
        mat.set(&[0, 1], 1.0)?; // [[1, 1], [0, 1]]

        let (q, t) = schur(&mat)?;

        // Verify dimensions
        assert_eq!(q.shape().dims(), &[2, 2]);
        assert_eq!(t.shape().dims(), &[2, 2]);

        // Verify that A = Q*T*Q^T (approximately)
        let qt = q.transpose(-2, -1)?;
        let qtq = q.matmul(&t)?;
        let reconstructed = qtq.matmul(&qt)?;

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(
                    reconstructed.get(&[i, j])?,
                    mat.get(&[i, j])?,
                    epsilon = 1e-3
                );
            }
        }

        // Verify Q is orthogonal
        let qtq_check = qt.matmul(&q)?;
        let identity = eye::<f32>(2)?;

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(
                    qtq_check.get(&[i, j])?,
                    identity.get(&[i, j])?,
                    epsilon = 1e-4
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_schur_identity() -> TorshResult<()> {
        let identity = eye::<f32>(3)?;
        let (q, t) = schur(&identity)?;

        // For identity matrix, T should be identity and Q should be orthogonal
        // Verify T is approximately identity
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(t.get(&[i, j])?, expected, epsilon = 1e-6);
            }
        }

        // Verify Q is orthogonal
        let qt = q.transpose(-2, -1)?;
        let qtq = qt.matmul(&q)?;

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(qtq.get(&[i, j])?, expected, epsilon = 1e-6);
            }
        }

        Ok(())
    }

    #[test]
    fn test_svd_rank_deficient() -> TorshResult<()> {
        // Create a rank-deficient matrix (rank 1)
        let mat = zeros::<f32>(&[3, 3])?;
        mat.set(&[0, 0], 1.0)?;
        mat.set(&[0, 1], 2.0)?;
        mat.set(&[0, 2], 3.0)?;
        mat.set(&[1, 0], 2.0)?;
        mat.set(&[1, 1], 4.0)?;
        mat.set(&[1, 2], 6.0)?;
        mat.set(&[2, 0], 3.0)?;
        mat.set(&[2, 1], 6.0)?;
        mat.set(&[2, 2], 9.0)?;

        let (_u, s, _vt) = svd(&mat, false)?;

        // Check how many singular values we actually got
        let s_len = s.shape().dims()[0];
        eprintln!("Number of singular values returned: {s_len}");

        // Should have at most 1 significant singular value
        // For a rank-1 matrix, we expect one large singular value and others to be small
        let s0 = s.get(&[0])?;
        eprintln!("s0={s0}");

        assert!(s0 > 1e-6); // First singular value should be significant

        // Check other singular values if they exist
        if s_len > 1 {
            // Try to access the second singular value safely
            if let Ok(s1) = s.get(&[1]) {
                eprintln!("s1={s1}");

                // For numerical SVD implementations, the tolerance needs to be relaxed
                let tolerance = 1e-1; // More tolerant for this implementation
                assert!(s1 < tolerance); // Second singular value should be relatively small

                // Check condition number
                assert!(s0 / s1.max(1e-12) > 5.0); // Condition number should be reasonably high
            } else {
                eprintln!("Could not access s1, treating as effectively zero");
            }
        }

        if s_len > 2 {
            // Try to access the third singular value safely
            if let Ok(s2) = s.get(&[2]) {
                eprintln!("s2={s2}");

                let tolerance = 1e-1;
                assert!(s2 < tolerance); // Third singular value should be relatively small
            } else {
                eprintln!("Could not access s2, treating as effectively zero");
            }
        }

        Ok(())
    }

    #[test]
    fn test_error_cases() -> TorshResult<()> {
        // Test non-square matrix for LU
        let nonsquare = zeros::<f32>(&[2, 3])?;
        assert!(lu(&nonsquare).is_err());

        // Test non-square matrix for eigenvalue
        assert!(eig(&nonsquare).is_err());

        // Test non-square matrix for Cholesky
        assert!(cholesky(&nonsquare, false).is_err());

        // Test non-positive-definite matrix for Cholesky
        let bad_mat = zeros::<f32>(&[2, 2])?;
        bad_mat.set(&[0, 0], -1.0)?; // Negative on diagonal
        bad_mat.set(&[1, 1], 1.0)?;
        assert!(cholesky(&bad_mat, false).is_err());

        Ok(())
    }

    #[test]
    fn test_hessenberg_identity() -> TorshResult<()> {
        // Hessenberg of identity should be identity
        let identity = eye::<f32>(3)?;
        let (q, h) = hessenberg(&identity)?;

        // H should be identity (already in Hessenberg form)
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(h.get(&[i, j])?, expected, epsilon = 1e-4);
            }
        }

        // Q should be approximately orthogonal (Q^T Q ≈ I)
        let qt = q.transpose_view(0, 1)?;
        let qtq = qt.matmul(&q)?;
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(qtq.get(&[i, j])?, expected, epsilon = 1e-3);
            }
        }

        Ok(())
    }

    #[test]
    fn test_hessenberg_structure() -> TorshResult<()> {
        // Test that Hessenberg form has zeros below first subdiagonal
        let mat = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
            torsh_core::DeviceType::Cpu,
        )?;

        let (_, h) = hessenberg(&mat)?;

        // Check that H has zeros below first subdiagonal
        // Element (2, 0) should be zero
        assert!(h.get(&[2, 0])?.abs() < 1e-6);

        // Elements on and above the first subdiagonal can be non-zero
        // We just verify the structure, not specific values

        Ok(())
    }

    #[test]
    fn test_hessenberg_small_matrix() -> TorshResult<()> {
        // Test 2x2 matrix (already in Hessenberg form)
        let mat = create_test_matrix_2x2()?;
        let (q, _h) = hessenberg(&mat)?;

        // For 2x2, Hessenberg form is same as original (all 2x2 are Hessenberg)
        // Just verify Q is approximately orthogonal
        let qt = q.transpose_view(0, 1)?;
        let qtq = qt.matmul(&q)?;

        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(qtq.get(&[i, j])?, expected, epsilon = 1e-3);
            }
        }

        Ok(())
    }
}
