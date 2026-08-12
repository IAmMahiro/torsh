//! Regression tests for the production-hardening campaign (torsh-sparse).
//!
//! Findings covered: F255 (duplicate entries never coalesced), F256 (sparse
//! add/multiply unimplemented), F311 (MATLAB .mat round trip).

use torsh_core::Shape;
use torsh_sparse::{CooTensor, CsrTensor, SparseTensor, TorshResult};
use torsh_tensor::creation::tensor_1d;

fn coo_with_duplicates() -> TorshResult<CooTensor> {
    // (0,0) appears twice (1.0 + 2.5) and (1,2) appears twice (3.0 + -1.0).
    CooTensor::new(
        vec![0, 1, 0, 1, 2],
        vec![0, 2, 0, 2, 1],
        vec![1.0, 3.0, 2.5, -1.0, 4.0],
        Shape::new(vec![3, 3]),
    )
}

// ---------------------------------------------------------------------------
// F255: duplicates must be summed consistently everywhere
// ---------------------------------------------------------------------------

#[test]
fn f255_coo_to_dense_sums_duplicates() -> TorshResult<()> {
    let coo = coo_with_duplicates()?;
    let dense = coo.to_dense()?;
    assert_eq!(dense.get(&[0, 0])?, 3.5, "duplicates must be summed");
    assert_eq!(dense.get(&[1, 2])?, 2.0, "duplicates must be summed");
    assert_eq!(dense.get(&[2, 1])?, 4.0);
    Ok(())
}

#[test]
fn f255_coalesce_merges_and_reports_flag() -> TorshResult<()> {
    let mut coo = coo_with_duplicates()?;
    assert!(!coo.is_coalesced());
    coo.coalesce();
    assert!(coo.is_coalesced());
    assert_eq!(
        coo.nnz(),
        3,
        "5 entries with 2 duplicate pairs -> 3 entries"
    );

    let triplets = coo.triplets();
    assert_eq!(triplets[0], (0, 0, 3.5));
    assert_eq!(triplets[1], (1, 2, 2.0));
    assert_eq!(triplets[2], (2, 1, 4.0));
    Ok(())
}

#[test]
fn f255_coalesce_drops_cancelling_entries() -> TorshResult<()> {
    let mut coo = CooTensor::new(
        vec![0, 0],
        vec![1, 1],
        vec![2.0, -2.0],
        Shape::new(vec![2, 2]),
    )?;
    coo.coalesce();
    assert_eq!(coo.nnz(), 0, "entries that cancel out must be dropped");
    Ok(())
}

#[test]
fn f255_csr_matvec_agrees_with_dense() -> TorshResult<()> {
    let coo = coo_with_duplicates()?;
    let csr = coo.to_csr()?;

    let v = tensor_1d(&[1.0f32, 2.0, 3.0])?;
    let sparse_result = csr.matvec(&v)?;

    let dense = csr.to_dense()?;
    for row in 0..3 {
        let mut expected = 0.0f32;
        for col in 0..3 {
            expected += dense.get(&[row, col])? * v.get(&[col])?;
        }
        assert!(
            (sparse_result.get(&[row])? - expected).abs() < 1e-6,
            "matvec disagrees with to_dense at row {row}: {} vs {expected}",
            sparse_result.get(&[row])?
        );
    }
    Ok(())
}

#[test]
fn f255_csr_from_coo_has_no_duplicate_columns() -> TorshResult<()> {
    let coo = coo_with_duplicates()?;
    let csr = CsrTensor::from_coo(&coo)?;
    assert_eq!(csr.nnz(), 3);
    for row in 0..3 {
        let (cols, _) = csr.get_row(row)?;
        for window in cols.windows(2) {
            assert!(
                window[0] < window[1],
                "column indices within a row must be strictly increasing"
            );
        }
    }
    Ok(())
}

#[test]
fn f255_csr_new_rejects_unsorted_columns() {
    // Row 0 has columns [2, 1] which breaks the sortedness invariant `get()`
    // and the two-pointer matmul rely on.
    let result = CsrTensor::new(
        vec![0, 2, 2],
        vec![2, 1],
        vec![1.0, 2.0],
        Shape::new(vec![2, 3]),
    );
    assert!(result.is_err(), "unsorted column indices must be rejected");
}

#[test]
fn f255_csr_new_rejects_duplicate_columns() {
    let result = CsrTensor::new(
        vec![0, 2, 2],
        vec![1, 1],
        vec![1.0, 2.0],
        Shape::new(vec![2, 3]),
    );
    assert!(result.is_err(), "duplicate column indices must be rejected");
}

#[test]
fn f255_csr_new_rejects_bad_row_ptr() {
    let result = CsrTensor::new(
        vec![0, 2, 1],
        vec![0, 1],
        vec![1.0, 2.0],
        Shape::new(vec![2, 3]),
    );
    assert!(result.is_err(), "decreasing row_ptr must be rejected");
}

#[test]
fn f255_csc_from_coo_coalesces() -> TorshResult<()> {
    let coo = coo_with_duplicates()?;
    let csc = coo.to_csc()?;
    assert_eq!(csc.nnz(), 3);
    let dense = csc.to_dense()?;
    assert_eq!(dense.get(&[0, 0])?, 3.5);
    Ok(())
}

#[test]
fn f255_from_unsorted_parts_normalises_external_data() -> TorshResult<()> {
    // Row 0 holds columns [2, 1, 1]: unsorted *and* duplicated, as SciPy, HDF5
    // or MATLAB may deliver it.
    let csr = CsrTensor::from_unsorted_parts(
        vec![0, 3, 3],
        vec![2, 1, 1],
        vec![1.0, 2.0, 3.0],
        Shape::new(vec![2, 3]),
    )?;
    assert_eq!(csr.nnz(), 2);
    let (cols, values) = csr.get_row(0)?;
    assert_eq!(cols, vec![1, 2]);
    assert_eq!(values, vec![5.0, 1.0]);

    let csc = torsh_sparse::CscTensor::from_unsorted_parts(
        vec![0, 0, 3],
        vec![1, 0, 1],
        vec![1.0, 2.0, 3.0],
        Shape::new(vec![2, 2]),
    )?;
    assert_eq!(csc.nnz(), 2);
    let dense = csc.to_dense()?;
    assert_eq!(dense.get(&[1, 1])?, 4.0);
    assert_eq!(dense.get(&[0, 1])?, 2.0);
    Ok(())
}

// ---------------------------------------------------------------------------
// F256: sparse add / multiply must actually compute
// ---------------------------------------------------------------------------

fn simple_coo(values: [f32; 3]) -> TorshResult<CooTensor> {
    CooTensor::new(
        vec![0, 1, 2],
        vec![0, 1, 2],
        values.to_vec(),
        Shape::new(vec![3, 3]),
    )
}

#[test]
fn f256_coo_add_and_multiply() -> TorshResult<()> {
    let a = simple_coo([1.0, 2.0, 3.0])?;
    let b = CooTensor::new(
        vec![0, 1, 0],
        vec![0, 1, 2],
        vec![10.0, 20.0, 5.0],
        Shape::new(vec![3, 3]),
    )?;

    let sum = a.add_coo(&b)?;
    let dense = sum.to_dense()?;
    assert_eq!(dense.get(&[0, 0])?, 11.0);
    assert_eq!(dense.get(&[1, 1])?, 22.0);
    assert_eq!(dense.get(&[2, 2])?, 3.0);
    assert_eq!(dense.get(&[0, 2])?, 5.0);

    let product = a.multiply_coo(&b)?;
    let dense = product.to_dense()?;
    assert_eq!(dense.get(&[0, 0])?, 10.0);
    assert_eq!(dense.get(&[1, 1])?, 40.0);
    assert_eq!(dense.get(&[2, 2])?, 0.0, "no overlap -> zero");
    assert_eq!(dense.get(&[0, 2])?, 0.0, "no overlap -> zero");
    Ok(())
}

#[test]
fn f256_csr_add_and_multiply() -> TorshResult<()> {
    let a = simple_coo([1.0, 2.0, 3.0])?.to_csr()?;
    let b = CooTensor::new(
        vec![0, 1, 0],
        vec![0, 1, 2],
        vec![10.0, 20.0, 5.0],
        Shape::new(vec![3, 3]),
    )?
    .to_csr()?;

    let sum = a.add_csr(&b)?;
    let dense = sum.to_dense()?;
    assert_eq!(dense.get(&[0, 0])?, 11.0);
    assert_eq!(dense.get(&[1, 1])?, 22.0);
    assert_eq!(dense.get(&[0, 2])?, 5.0);

    let product = a.multiply_csr(&b)?;
    let dense = product.to_dense()?;
    assert_eq!(dense.get(&[0, 0])?, 10.0);
    assert_eq!(dense.get(&[1, 1])?, 40.0);
    assert_eq!(dense.get(&[2, 2])?, 0.0);
    Ok(())
}

#[test]
fn f256_shape_mismatch_is_rejected() -> TorshResult<()> {
    let a = simple_coo([1.0, 2.0, 3.0])?;
    let b = CooTensor::new(vec![0], vec![0], vec![1.0], Shape::new(vec![2, 2]))?;
    assert!(a.add_coo(&b).is_err());
    assert!(a.multiply_coo(&b).is_err());
    Ok(())
}

// ---------------------------------------------------------------------------
// F311: MATLAB Level-5 .mat round trip
// ---------------------------------------------------------------------------

#[cfg(feature = "matlab")]
#[test]
fn f311_mat_file_round_trip() -> TorshResult<()> {
    use torsh_sparse::matlab_compat::MatlabSparseCompat;

    let coo = CooTensor::new(
        vec![0, 2, 1],
        vec![0, 1, 3],
        vec![1.5, -2.5, 7.0],
        Shape::new(vec![3, 4]),
    )?;

    let mut path = std::env::temp_dir();
    path.push(format!("torsh_sparse_hardening_{}.mat", std::process::id()));

    MatlabSparseCompat::export_to_mat_file(&coo, &path, "spmat")?;
    let imported = MatlabSparseCompat::import_from_mat_file(&path, "spmat")?;
    let _ = std::fs::remove_file(&path);

    assert_eq!(imported.shape().dims(), &[3, 4]);
    let dense = imported.to_dense()?;
    assert!((dense.get(&[0, 0])? - 1.5).abs() < 1e-6);
    assert!((dense.get(&[2, 1])? + 2.5).abs() < 1e-6);
    assert!((dense.get(&[1, 3])? - 7.0).abs() < 1e-6);
    assert!(dense.get(&[0, 1])?.abs() < 1e-6);
    Ok(())
}
