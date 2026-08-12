//! Hardening tests for torsh-metrics production-hardening findings.
//!
//! Each test is a minimal reproducer for one assigned finding. See the
//! campaign brief for finding IDs and full descriptions.

use torsh_core::device::DeviceType;
use torsh_metrics::classification::{AverageMethod, F1Score, MultiClassMetrics, Precision};
use torsh_metrics::regression::{R2Score, MAE};
use torsh_metrics::{Metric, SklearnPrecision, SklearnRecall};
use torsh_tensor::creation::from_vec;

/// F125: an unknown `average` string must not abort the process, and a
/// user-slice length mismatch must not either -- both are ordinary
/// programmer/user errors, not invariant violations.
#[test]
fn f125_sklearn_precision_mismatched_lengths_does_not_panic() {
    let y_true = vec![0usize, 1, 0];
    let y_pred = vec![0usize, 1]; // deliberately mismatched length
    let result = SklearnPrecision::new().compute(&y_true, &y_pred);
    assert!(
        result.is_nan(),
        "mismatched-length input should surface as NaN, not panic or a plausible value"
    );
}

#[test]
fn f125_sklearn_precision_unknown_average_does_not_panic() {
    let y_true = vec![0usize, 1, 0, 1];
    let y_pred = vec![0usize, 1, 1, 1];
    let result = SklearnPrecision::new()
        .with_average("weighed") // typo of "weighted"
        .compute(&y_true, &y_pred);
    assert!(
        result.is_nan(),
        "unknown average string should surface as NaN, not panic"
    );
}

#[test]
fn f125_sklearn_recall_mismatched_lengths_and_unknown_average_do_not_panic() {
    let y_true = vec![0usize, 1, 0];
    let y_pred = vec![0usize, 1];
    assert!(SklearnRecall::new().compute(&y_true, &y_pred).is_nan());

    let y_true = vec![0usize, 1, 0, 1];
    let y_pred = vec![0usize, 1, 1, 1];
    assert!(SklearnRecall::new()
        .with_average("bogus")
        .compute(&y_true, &y_pred)
        .is_nan());
}

/// F222: macro-F1 must be the mean of per-class F1 scores, not the
/// harmonic mean of macro-precision and macro-recall.
///
/// class 0: tp=1, fp=0, fn=1 -> P=1.0, R=0.5, F1=0.6667
/// class 1: tp=1, fp=1, fn=0 -> P=0.5, R=1.0, F1=0.6667
/// correct macro-F1 = mean(0.6667, 0.6667) = 0.6667;
/// the buggy formula computes F1(mean_P=0.75, mean_R=0.75) = 0.75.
#[test]
fn f222_macro_f1_is_mean_of_per_class_f1_not_harmonic_of_macro_pr() {
    let predictions = from_vec(
        vec![
            0.9, 0.1, // sample 0: pred 0, true 0 (TP for class 0)
            0.1, 0.9, // sample 1: pred 1, true 0 (FN for class 0, FP for class 1)
            0.2, 0.8, // sample 2: pred 1, true 1 (TP for class 1)
        ],
        &[3, 2],
        DeviceType::Cpu,
    )
    .unwrap();
    let targets = from_vec(vec![0.0, 0.0, 1.0], &[3], DeviceType::Cpu).unwrap();

    let f1 = F1Score::macro_averaged();
    let result = f1.compute(&predictions, &targets);
    assert!(
        (result - 0.6667).abs() < 0.01,
        "macro-F1 should be the mean of per-class F1 (~0.6667), got {result}"
    );
}

/// F223: `AverageMethod::Weighted` must compute a real support-weighted
/// average, not silently fall through to the macro value while still
/// reporting itself as "weighted".
///
/// 9 samples true=0/pred=0 (correct), 1 sample true=1/pred=0 (wrong):
/// F1[0]=0.9474 (support 9), F1[1]=0.0 (support 1).
/// macro F1  = (0.9474 + 0.0) / 2      = 0.4737
/// weighted F1 = (0.9474*9 + 0.0*1) / 10 = 0.8526
/// These differ substantially, so returning the macro value under a
/// "weighted" label is observably wrong.
#[test]
fn f223_weighted_average_is_support_weighted_not_macro() {
    let mut pred_data = Vec::new();
    let mut target_data = Vec::new();
    for _ in 0..9 {
        pred_data.extend_from_slice(&[0.9, 0.1]); // predicts class 0
        target_data.push(0.0); // true class 0 (correct)
    }
    pred_data.extend_from_slice(&[0.9, 0.1]); // predicts class 0
    target_data.push(1.0); // true class 1 (wrong)

    let predictions = from_vec(pred_data, &[10, 2], DeviceType::Cpu).unwrap();
    let targets = from_vec(target_data, &[10], DeviceType::Cpu).unwrap();

    let f1 = F1Score::new(AverageMethod::Weighted);
    let result = f1.compute(&predictions, &targets);
    assert!(
        (result - 0.8526).abs() < 0.01,
        "weighted F1 should be support-weighted (~0.8526), not the macro value (~0.4737); got {result}"
    );
}

/// F224: 1-D predictions (positive-class probabilities, the natural shape
/// for binary classification -- same convention `compute_standard_accuracy`
/// already uses) must be supported by precision/recall/F1, not silently
/// bailed out to 0.0.
#[test]
fn f224_precision_recall_f1_support_1d_binary_predictions() {
    // Perfect binary predictions at the 0.5 threshold.
    let predictions = from_vec(vec![0.9, 0.1, 0.8, 0.2], &[4], DeviceType::Cpu).unwrap();
    let targets = from_vec(vec![1.0, 0.0, 1.0, 0.0], &[4], DeviceType::Cpu).unwrap();

    let precision = Precision::micro().compute(&predictions, &targets);
    assert!(
        (precision - 1.0).abs() < 1e-6,
        "1-D perfect predictions should give precision 1.0, not the 0.0 shape-bailout value; got {precision}"
    );
}

/// F225: the ROC/PR threshold list must stay sorted even when scores fall
/// outside [0, 1] (e.g. raw logits), otherwise the trapezoidal AUC
/// integration double-counts area.
#[test]
fn f225_threshold_metrics_thresholds_stay_monotonic_for_out_of_unit_range_scores() {
    use torsh_metrics::classification::ThresholdMetrics;

    let predictions = from_vec(vec![-2.0, 1.0, 3.0], &[3], DeviceType::Cpu).unwrap();
    let targets = from_vec(vec![0.0, 1.0, 1.0], &[3], DeviceType::Cpu).unwrap();

    let metrics = ThresholdMetrics::compute(&predictions, &targets);
    let sorted = metrics.thresholds.windows(2).all(|w| w[0] <= w[1]);
    assert!(
        sorted,
        "thresholds must stay monotonic even for scores outside [0,1]: {:?}",
        metrics.thresholds
    );

    let auc = metrics.auc_roc();
    assert!(
        (0.0..=1.0).contains(&auc),
        "AUC-ROC must stay within [0,1]; got {auc}"
    );
}

/// F226: shape/length mismatches must surface as an honest NaN, not a
/// plausible-looking 0.0 that reads as "perfect model".
///
/// (MAE, not MSE: `tests/integration_tests.rs::test_mismatched_sizes` pins
/// MSE's mismatched-length behavior to exactly `0.0` and is out of scope
/// for this wave, so that one case is a documented exception -- see the
/// FOLLOW-UP comment on `regression::compute_mse`. MAE has no such pin and
/// gets the real fix.)
#[test]
fn f226_mae_mismatched_lengths_returns_nan_not_zero() {
    let predictions = from_vec(vec![1.0, 2.0, 3.0], &[3], DeviceType::Cpu).unwrap();
    let targets = from_vec(vec![1.0, 2.0], &[2], DeviceType::Cpu).unwrap();

    let mae = MAE;
    let result = mae.compute(&predictions, &targets);
    assert!(
        result.is_nan(),
        "mismatched-length inputs must surface as NaN, not a plausible 0.0 MAE; got {result}"
    );
}

/// F226 (sklearn alignment): a perfect constant-target prediction (zero
/// variance in targets, zero residual) should score R2=1.0, matching
/// sklearn's `force_finite` behavior, not an unconditional 0.0.
#[test]
fn f226_r2_perfect_constant_prediction_returns_one_not_zero() {
    let predictions = from_vec(vec![5.0, 5.0, 5.0, 5.0], &[4], DeviceType::Cpu).unwrap();
    let targets = from_vec(vec![5.0, 5.0, 5.0, 5.0], &[4], DeviceType::Cpu).unwrap();

    let r2 = R2Score::new();
    let result = r2.compute(&predictions, &targets);
    assert!(
        (result - 1.0).abs() < 1e-9,
        "a perfect constant-target prediction should score R2=1.0 (sklearn force_finite behavior), got {result}"
    );
}

/// F227: the confusion-matrix class count must be derived from the
/// observed data, not forced to a minimum of 2 -- a legitimate
/// single-class evaluation batch should not have its macro metrics halved
/// by an invented, always-empty second class.
#[test]
fn f227_confusion_matrix_class_count_matches_data_not_forced_to_two() {
    let predictions =
        from_vec(vec![0.9, 0.1, 0.8, 0.2, 0.7, 0.3], &[3, 2], DeviceType::Cpu).unwrap();
    let targets = from_vec(vec![0.0, 0.0, 0.0], &[3], DeviceType::Cpu).unwrap();

    let metrics = MultiClassMetrics::compute(&predictions, &targets)
        .expect("well-formed single-class input should not error");
    assert_eq!(
        metrics.num_classes(),
        1,
        "single-class data should report 1 class, not a forced minimum of 2"
    );
    assert!(
        (metrics.macro_avg - 1.0).abs() < 1e-9,
        "a perfect single-class classifier should score macro F1 = 1.0, not 0.5; got {}",
        metrics.macro_avg
    );
}
