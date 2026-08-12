//! Hardening regression tests for the wave-3 dead-code sweep (F066/F262/F264/F312)
//! and the `normal_`/`multinomial` port that had to happen before the sweep could
//! delete `src/ops.rs.backup`.
//!
//! `src/ops/` (13k LOC of disabled SIMD/GPU dispatch), `src/ops.rs.backup`
//! (7.8k LOC), `src/ops_legacy.rs`, `src/lib_new.rs` and `src/lazy_ops.rs` were
//! never reachable from `lib.rs` (no live `mod` declaration pointed at any of
//! them) and are deleted by this pass. Before deleting `ops.rs.backup`, its
//! only-known implementations of `Tensor::normal_` and `Tensor::multinomial`
//! were ported for real into `src/creation.rs`, using the wave-1
//! process-global RNG (`with_rng` / `manual_seed`) instead of the banned
//! direct `rand`/`rand_distr` crates the backup file used.

// ---------------------------------------------------------- dead code sweep ---

/// F066/F262/F264/F312: the disabled `src/ops/` tree and its orphan siblings
/// must not exist in the shipped crate. This is intentionally a plain
/// filesystem check (not a compile-time one): the whole point of these files
/// is that nothing in the crate references them, so there is no symbol to
/// call that would fail to compile if they came back — only their presence
/// on disk (and in the published tarball) is observable.
#[test]
fn deleted_dead_code_files_are_gone() {
    let src = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src"));
    for orphan in [
        "ops.rs.backup",
        "ops_legacy.rs",
        "lib_new.rs",
        "lazy_ops.rs",
        "ops", // the whole disabled src/ops/ directory tree
    ] {
        let path = src.join(orphan);
        assert!(
            !path.exists(),
            "{orphan} should have been deleted as dead/unreachable code, \
             but still exists at {path:?}"
        );
    }
}

// ------------------------------------------------------- normal_ / multinomial ---
//
// `ops.rs.backup` held the only implementations of `Tensor::normal_` and
// `Tensor::multinomial` (referenced by the block-commented tests at
// tests/tensor_tests.rs:202,232). Both were re-implemented for real in
// `src/creation.rs` on top of the wave-1 process-global RNG before the backup
// file was deleted above. These tests mirror (and extend, with the error-path
// cases the original commented tests did not cover) that original coverage.

use torsh_tensor::creation::{manual_seed, tensor_1d, tensor_2d, zeros};
use torsh_tensor::Tensor;

#[test]
fn normal_inplace_fills_nonzero_values() {
    let mut t = zeros::<f32>(&[10, 10]).expect("zeros should succeed");
    t.normal_(0.0, 1.0).expect("normal_ should succeed");

    let data = t.to_vec().expect("to_vec should succeed");
    let non_zero_count = data.iter().filter(|&&x| x != 0.0).count();
    assert!(
        non_zero_count > 0,
        "normal_ should fill tensor with non-zero values"
    );
}

#[test]
fn normal_inplace_mean_is_close_to_requested_mean() {
    manual_seed(12345);
    let mut t = zeros::<f32>(&[200, 200]).expect("zeros should succeed");
    t.normal_(5.0, 2.0).expect("normal_ should succeed");

    let data = t.to_vec().expect("to_vec should succeed");
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    assert!(
        (mean - 5.0).abs() < 0.5,
        "mean {mean} should be close to the requested mean 5.0"
    );
}

#[test]
fn normal_inplace_rejects_negative_std() {
    let mut t = zeros::<f32>(&[4]).expect("zeros should succeed");
    assert!(t.normal_(0.0, -1.0).is_err());
}

#[test]
fn normal_inplace_rejects_non_finite_parameters() {
    let mut t = zeros::<f32>(&[4]).expect("zeros should succeed");
    assert!(t.normal_(f64::NAN, 1.0).is_err());
    assert!(t.normal_(0.0, f64::INFINITY).is_err());
    assert!(t.normal_(f64::NEG_INFINITY, 1.0).is_err());
}

#[test]
fn normal_inplace_accepts_zero_std_as_a_point_mass() {
    let mut t = zeros::<f32>(&[16]).expect("zeros should succeed");
    t.normal_(3.0, 0.0)
        .expect("std=0.0 should be accepted, matching PyTorch's normal_ semantics");
    let data = t.to_vec().expect("to_vec should succeed");
    assert!(data.iter().all(|&x| (x - 3.0).abs() < 1e-6));
}

#[test]
fn normal_inplace_rejects_tensors_that_require_grad() {
    let mut t = zeros::<f32>(&[4]).expect("zeros should succeed");
    t = t.requires_grad_(true);
    assert!(
        t.normal_(0.0, 1.0).is_err(),
        "normal_ must refuse to mutate a tensor that requires grad, like every \
         other in-place op on this type"
    );
}

#[test]
fn multinomial_with_replacement_returns_valid_indices() {
    let weights = tensor_1d(&[0.1f32, 0.2, 0.3, 0.4]).expect("tensor_1d should succeed");
    let samples = Tensor::multinomial(&weights, 100, true).expect("multinomial should succeed");

    assert_eq!(samples.shape().dims(), &[100]);
    assert_eq!(samples.dtype(), torsh_core::DType::I64);

    let sample_data = samples.to_vec().expect("to_vec should succeed");
    for &sample in &sample_data {
        assert!(
            (0i64..4).contains(&sample),
            "sample index {sample} out of bounds"
        );
    }
}

#[test]
fn multinomial_without_replacement_rejects_oversampling() {
    let weights = tensor_1d(&[0.1f32, 0.2, 0.3, 0.4]).expect("tensor_1d should succeed");
    assert!(Tensor::multinomial(&weights, 5, false).is_err());
}

#[test]
fn multinomial_without_replacement_returns_unique_indices() {
    let weights = tensor_1d(&[0.1f32, 0.2, 0.3, 0.4]).expect("tensor_1d should succeed");
    let samples = Tensor::multinomial(&weights, 3, false).expect("multinomial should succeed");
    assert_eq!(samples.shape().dims(), &[3]);

    let mut sample_data = samples.to_vec().expect("to_vec should succeed");
    sample_data.sort_unstable();
    let before = sample_data.len();
    sample_data.dedup();
    assert_eq!(
        sample_data.len(),
        before,
        "samples without replacement should be unique"
    );
}

#[test]
fn multinomial_rejects_all_zero_weights() {
    let zero_weights = tensor_1d(&[0.0f32, 0.0, 0.0, 0.0]).expect("tensor_1d should succeed");
    assert!(Tensor::multinomial(&zero_weights, 1, true).is_err());
}

#[test]
fn multinomial_rejects_negative_weights() {
    let weights = tensor_1d(&[0.5f32, -0.1, 0.6]).expect("tensor_1d should succeed");
    assert!(Tensor::multinomial(&weights, 1, true).is_err());
}

#[test]
fn multinomial_rejects_non_finite_weights() {
    let weights = tensor_1d(&[0.5f32, f32::NAN, 0.6]).expect("tensor_1d should succeed");
    assert!(Tensor::multinomial(&weights, 1, true).is_err());
}

#[test]
fn multinomial_rejects_non_1d_weights() {
    let weights = tensor_2d(&[&[0.5f32, 0.5], &[0.5, 0.5]]).expect("tensor_2d should succeed");
    assert!(Tensor::multinomial(&weights, 1, true).is_err());
}

#[test]
fn multinomial_never_samples_a_zero_weight_category() {
    // Regression test for the boundary-selection fallback ported from
    // ops.rs.backup: a zero-weight category must never be selected.
    let weights = tensor_1d(&[1.0f32, 0.0, 1.0, 0.0]).expect("tensor_1d should succeed");
    let samples = Tensor::multinomial(&weights, 200, true).expect("multinomial should succeed");
    let data = samples.to_vec().expect("to_vec should succeed");
    assert!(
        data.iter().all(|&idx| idx == 0 || idx == 2),
        "multinomial must never select a zero-weight category"
    );
}
