//! Regression tests for the production-hardening campaign (torsh-linalg).
//!
//! Findings covered: F026 (aliasing mutation in `eig`/`jordan_form`),
//! F027 (fabricated eigenpairs), F028 (garbage SVD beyond 5x5),
//! F221 (classical Gram-Schmidt QR).

use torsh_core::DeviceType;
use torsh_linalg::decomposition::{eig, eig_complex, hessenberg, jordan_form, qr, schur, svd};
use torsh_linalg::TorshResult;
use torsh_tensor::Tensor;

/// Tiny deterministic LCG so the tests are reproducible without extra deps.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let bits = (self.0 >> 33) as u32;
        (bits as f32 / (1u32 << 31) as f32) - 1.0
    }
}

fn random_matrix(rows: usize, cols: usize, seed: u64) -> TorshResult<Tensor> {
    let mut rng = Lcg::new(seed);
    let data: Vec<f32> = (0..rows * cols).map(|_| rng.next_f32()).collect();
    Tensor::from_data(data, vec![rows, cols], DeviceType::Cpu)
}

fn random_symmetric(n: usize, seed: u64) -> TorshResult<Tensor> {
    let a = random_matrix(n, n, seed)?;
    let mut data = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let v = 0.5 * (a.get(&[i, j])? + a.get(&[j, i])?);
            data[i * n + j] = v;
        }
    }
    Tensor::from_data(data, vec![n, n], DeviceType::Cpu)
}

fn matrix_data(t: &Tensor) -> TorshResult<Vec<f32>> {
    let dims = t.shape().dims().to_vec();
    let mut out = Vec::with_capacity(dims[0] * dims[1]);
    for i in 0..dims[0] {
        for j in 0..dims[1] {
            out.push(t.get(&[i, j])?);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// F026: eig()/jordan_form() must not mutate the caller's matrix
// ---------------------------------------------------------------------------

#[test]
fn f026_eig_does_not_mutate_input() -> TorshResult<()> {
    let a = random_symmetric(6, 12_345)?;
    let before = matrix_data(&a)?;
    let _ = eig(&a)?;
    let after = matrix_data(&a)?;
    assert_eq!(before, after, "eig() mutated the caller's matrix");
    Ok(())
}

#[test]
fn f026_jordan_form_does_not_mutate_input() -> TorshResult<()> {
    let a = random_symmetric(5, 999)?;
    let before = matrix_data(&a)?;
    let _ = jordan_form(&a)?;
    let after = matrix_data(&a)?;
    assert_eq!(before, after, "jordan_form() mutated the caller's matrix");
    Ok(())
}

#[test]
fn f026_schur_does_not_mutate_input() -> TorshResult<()> {
    // schur() applies its shift in place through the same shallow clone.
    let a = random_symmetric(4, 5150)?;
    let before = matrix_data(&a)?;
    let _ = schur(&a)?;
    let after = matrix_data(&a)?;
    assert_eq!(before, after, "schur() mutated the caller's matrix");
    Ok(())
}

#[test]
fn f026_hessenberg_does_not_mutate_input() -> TorshResult<()> {
    let a = random_matrix(4, 4, 8080)?;
    let before = matrix_data(&a)?;
    let _ = hessenberg(&a)?;
    let after = matrix_data(&a)?;
    assert_eq!(before, after, "hessenberg() mutated the caller's matrix");
    Ok(())
}

// ---------------------------------------------------------------------------
// F027: eig() must return genuine eigenpairs, never padded zeros/identity
// ---------------------------------------------------------------------------

fn check_eigenpairs(a: &Tensor, tol: f32) -> TorshResult<()> {
    let n = a.shape().dims()[0];
    let (values, vectors) = eig(a)?;
    assert_eq!(values.shape().dims(), &[n]);
    assert_eq!(vectors.shape().dims(), &[n, n]);

    for k in 0..n {
        let lambda = values.get(&[k])?;
        // residual ||A v - lambda v|| / ||v||
        let mut vnorm = 0.0f32;
        for i in 0..n {
            let v = vectors.get(&[i, k])?;
            vnorm += v * v;
        }
        let vnorm = vnorm.sqrt();
        assert!(vnorm > 1e-4, "eigenvector {k} is (near) zero");

        let mut residual = 0.0f32;
        for i in 0..n {
            let mut av = 0.0f32;
            for j in 0..n {
                av += a.get(&[i, j])? * vectors.get(&[j, k])?;
            }
            let d = av - lambda * vectors.get(&[i, k])?;
            residual += d * d;
        }
        let residual = residual.sqrt() / vnorm;
        assert!(
            residual < tol,
            "eigenpair {k} residual {residual} exceeds {tol} (lambda={lambda})"
        );
    }
    Ok(())
}

#[test]
fn f027_eig_symmetric_20x20_real_eigenpairs() -> TorshResult<()> {
    let a = random_symmetric(20, 4_242)?;
    check_eigenpairs(&a, 1e-3)
}

#[test]
fn f027_eig_symmetric_8x8_real_eigenpairs() -> TorshResult<()> {
    let a = random_symmetric(8, 7)?;
    check_eigenpairs(&a, 1e-4)
}

#[test]
fn f027_eig_nonsymmetric_real_spectrum() -> TorshResult<()> {
    // A = M D M^-1 with a well separated real spectrum is diagonalisable over R.
    // Build it as an upper triangular matrix similar to diag(1..=6) instead:
    // an upper triangular matrix has its diagonal as spectrum and real eigenvectors.
    let n = 6;
    let mut data = vec![0.0f32; n * n];
    let mut rng = Lcg::new(31);
    for i in 0..n {
        data[i * n + i] = (i + 1) as f32;
        for j in (i + 1)..n {
            data[i * n + j] = 0.5 * rng.next_f32();
        }
    }
    let a = Tensor::from_data(data, vec![n, n], DeviceType::Cpu)?;
    check_eigenpairs(&a, 1e-3)
}

// ---------------------------------------------------------------------------
// F028: SVD must reconstruct A for min(m, n) > 5 and for non-square input
// ---------------------------------------------------------------------------

fn check_svd(rows: usize, cols: usize, seed: u64, tol: f32) -> TorshResult<()> {
    let a = random_matrix(rows, cols, seed)?;
    let k = rows.min(cols);
    let (u, s, vt) = svd(&a, false)?;

    assert_eq!(u.shape().dims(), &[rows, k], "U has wrong shape");
    assert_eq!(s.shape().dims(), &[k], "S has wrong shape");
    assert_eq!(vt.shape().dims(), &[k, cols], "V^T has wrong shape");

    // Singular values non-negative and sorted descending.
    for i in 0..k {
        assert!(s.get(&[i])? >= -1e-6, "negative singular value at {i}");
        if i + 1 < k {
            assert!(
                s.get(&[i])? >= s.get(&[i + 1])? - 1e-5,
                "singular values not sorted at {i}"
            );
        }
    }

    // Reconstruction: A ~= U * diag(S) * V^T
    let mut max_err = 0.0f32;
    for i in 0..rows {
        for j in 0..cols {
            let mut acc = 0.0f32;
            for t in 0..k {
                acc += u.get(&[i, t])? * s.get(&[t])? * vt.get(&[t, j])?;
            }
            max_err = max_err.max((acc - a.get(&[i, j])?).abs());
        }
    }
    assert!(
        max_err < tol,
        "SVD reconstruction error {max_err} exceeds {tol} for {rows}x{cols}"
    );
    Ok(())
}

#[test]
fn f028_svd_reconstructs_20x20() -> TorshResult<()> {
    check_svd(20, 20, 1_001, 1e-3)
}

#[test]
fn f028_svd_reconstructs_8x8() -> TorshResult<()> {
    check_svd(8, 8, 2_002, 1e-4)
}

#[test]
fn f028_svd_reconstructs_tall_10x6() -> TorshResult<()> {
    check_svd(10, 6, 3_003, 1e-4)
}

#[test]
fn f028_svd_reconstructs_wide_6x10() -> TorshResult<()> {
    check_svd(6, 10, 4_004, 1e-4)
}

#[test]
fn f028_svd_full_matrices_shapes() -> TorshResult<()> {
    let a = random_matrix(6, 4, 5_005)?;
    let (u, s, vt) = svd(&a, true)?;
    assert_eq!(u.shape().dims(), &[6, 6], "full U must be m x m");
    assert_eq!(s.shape().dims(), &[4]);
    assert_eq!(vt.shape().dims(), &[4, 4], "full V^T must be n x n");
    Ok(())
}

// ---------------------------------------------------------------------------
// F221: QR must stay orthogonal on ill-conditioned input
// ---------------------------------------------------------------------------

#[test]
fn f221_qr_orthogonal_on_hilbert() -> TorshResult<()> {
    let n = 8;
    let mut data = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            data[i * n + j] = 1.0 / ((i + j + 1) as f32);
        }
    }
    let a = Tensor::from_data(data, vec![n, n], DeviceType::Cpu)?;
    let (q, r) = qr(&a)?;

    // Q^T Q = I
    let mut max_dev = 0.0f32;
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f32;
            for t in 0..n {
                acc += q.get(&[t, i])? * q.get(&[t, j])?;
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            max_dev = max_dev.max((acc - expected).abs());
        }
    }
    assert!(
        max_dev < 1e-4,
        "||Q^T Q - I||_max = {max_dev} is too large (loss of orthogonality)"
    );

    // A = QR
    let mut max_err = 0.0f32;
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f32;
            for t in 0..n {
                acc += q.get(&[i, t])? * r.get(&[t, j])?;
            }
            max_err = max_err.max((acc - a.get(&[i, j])?).abs());
        }
    }
    assert!(max_err < 1e-5, "||A - QR||_max = {max_err} is too large");
    Ok(())
}

#[test]
fn f221_qr_rank_deficient_is_supported() -> TorshResult<()> {
    // Rank-1 3x3 matrix: Householder QR handles this, Gram-Schmidt bails out.
    let data = vec![1.0f32, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0];
    let a = Tensor::from_data(data, vec![3, 3], DeviceType::Cpu)?;
    let (q, r) = qr(&a)?;
    let mut max_err = 0.0f32;
    for i in 0..3 {
        for j in 0..3 {
            let mut acc = 0.0f32;
            for t in 0..3 {
                acc += q.get(&[i, t])? * r.get(&[t, j])?;
            }
            max_err = max_err.max((acc - a.get(&[i, j])?).abs());
        }
    }
    assert!(max_err < 1e-5, "rank-deficient QR error {max_err}");
    Ok(())
}

#[test]
fn f221_qr_wide_matrix_is_supported() -> TorshResult<()> {
    let a = random_matrix(3, 5, 606)?;
    let (q, r) = qr(&a)?;
    assert_eq!(q.shape().dims(), &[3, 3]);
    assert_eq!(r.shape().dims(), &[3, 5]);
    let mut max_err = 0.0f32;
    for i in 0..3 {
        for j in 0..5 {
            let mut acc = 0.0f32;
            for t in 0..3 {
                acc += q.get(&[i, t])? * r.get(&[t, j])?;
            }
            max_err = max_err.max((acc - a.get(&[i, j])?).abs());
        }
    }
    assert!(max_err < 1e-5, "wide QR error {max_err}");
    Ok(())
}

// ---------------------------------------------------------------------------
// F027: complex spectra are reported honestly, never silently truncated
// ---------------------------------------------------------------------------

#[test]
fn f027_eig_rejects_complex_spectrum() -> TorshResult<()> {
    // Rotation by 90 degrees: eigenvalues are +/- i, no real eigenvector exists.
    let a = Tensor::from_data(vec![0.0f32, -1.0, 1.0, 0.0], vec![2, 2], DeviceType::Cpu)?;
    assert!(
        eig(&a).is_err(),
        "a complex spectrum must be reported, not padded with zeros"
    );
    Ok(())
}

#[test]
fn f027_eig_complex_returns_true_complex_eigenpairs() -> TorshResult<()> {
    let a = Tensor::from_data(vec![0.0f32, -1.0, 1.0, 0.0], vec![2, 2], DeviceType::Cpu)?;
    let (re, im, vec_re, vec_im) = eig_complex(&a)?;

    assert_eq!(re.shape().dims(), &[2]);
    for k in 0..2 {
        assert!(re.get(&[k])?.abs() < 1e-5, "real part must vanish");
        assert!(
            (im.get(&[k])?.abs() - 1.0).abs() < 1e-5,
            "imaginary part must be +/-1, got {}",
            im.get(&[k])?
        );

        // (A v)_i == lambda * v_i in complex arithmetic.
        for i in 0..2 {
            let (mut av_re, mut av_im) = (0.0f32, 0.0f32);
            for j in 0..2 {
                av_re += a.get(&[i, j])? * vec_re.get(&[j, k])?;
                av_im += a.get(&[i, j])? * vec_im.get(&[j, k])?;
            }
            let target_re =
                re.get(&[k])? * vec_re.get(&[i, k])? - im.get(&[k])? * vec_im.get(&[i, k])?;
            let target_im =
                re.get(&[k])? * vec_im.get(&[i, k])? + im.get(&[k])? * vec_re.get(&[i, k])?;
            assert!(
                (av_re - target_re).abs() < 1e-4,
                "complex eigenpair {k} residual"
            );
            assert!(
                (av_im - target_im).abs() < 1e-4,
                "complex eigenpair {k} residual"
            );
        }
    }
    Ok(())
}

#[test]
fn f027_eig_diagonal_stays_exact() -> TorshResult<()> {
    let a = Tensor::from_data(
        vec![3.0f32, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 5.0],
        vec![3, 3],
        DeviceType::Cpu,
    )?;
    let (values, vectors) = eig(&a)?;
    assert_eq!(values.get(&[0])?, 3.0);
    assert_eq!(values.get(&[1])?, -2.0);
    assert_eq!(values.get(&[2])?, 5.0);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_eq!(vectors.get(&[i, j])?, expected);
        }
    }
    Ok(())
}

#[test]
fn f028_svd_handles_degenerate_spectrum() -> TorshResult<()> {
    // Repeated singular values (the identity) broke the eigen-decomposition
    // route: U and V came back with duplicated columns.
    for n in [2usize, 3, 5] {
        let mut data = vec![0.0f32; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        let a = Tensor::from_data(data, vec![n, n], DeviceType::Cpu)?;
        let (u, s, vt) = svd(&a, false)?;
        for i in 0..n {
            assert!((s.get(&[i])? - 1.0).abs() < 1e-5);
            for j in 0..n {
                let mut acc = 0.0f32;
                for t in 0..n {
                    acc += u.get(&[i, t])? * s.get(&[t])? * vt.get(&[t, j])?;
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (acc - expected).abs() < 1e-5,
                    "identity {n}x{n} not reconstructed at ({i},{j}): {acc}"
                );
            }
        }
    }
    Ok(())
}
