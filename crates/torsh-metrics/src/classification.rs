//! Classification metrics

use crate::Metric;
use torsh_core::error::TorshError;
use torsh_tensor::Tensor;
// Enhanced with scirs2-metrics integration
// use scirs2_metrics::classification::*; // Will be added when API stabilizes

/// Averaging method for multi-class metrics
#[derive(Debug, Clone)]
pub enum AverageMethod {
    Micro,
    Macro,
    Weighted,
    None, // Per-class results
}

/// Accuracy metric
#[derive(Clone)]
pub struct Accuracy {
    top_k: Option<usize>,
}

impl Accuracy {
    /// Create a new accuracy metric
    pub fn new() -> Self {
        Self { top_k: None }
    }

    /// Create a top-k accuracy metric
    pub fn top_k(k: usize) -> Self {
        Self { top_k: Some(k) }
    }
}

impl Metric for Accuracy {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        let result = if let Some(k) = self.top_k {
            // Top-k accuracy
            compute_top_k_accuracy(predictions, targets, k)
        } else {
            // Standard accuracy using robust implementation
            compute_standard_accuracy(predictions, targets)
        };
        // `Metric::compute` is infallible by trait signature; surface
        // shape/argument errors as NaN (visibly wrong) rather than a
        // plausible-looking 0.0. Use `compute_standard_accuracy`/
        // `compute_top_k_accuracy` directly for the `Result` form.
        result.unwrap_or(f64::NAN)
    }

    fn name(&self) -> &str {
        if self.top_k.is_some() {
            "top_k_accuracy"
        } else {
            "accuracy"
        }
    }
}

/// Precision metric - Enhanced with scirs2-metrics compatibility
pub struct Precision {
    average: AverageMethod,
}

impl Precision {
    /// Create a new precision metric
    pub fn new(average: AverageMethod) -> Self {
        Self { average }
    }

    /// Create micro-averaged precision
    pub fn micro() -> Self {
        Self {
            average: AverageMethod::Micro,
        }
    }

    /// Create macro-averaged precision
    pub fn macro_averaged() -> Self {
        Self {
            average: AverageMethod::Macro,
        }
    }
}

impl Metric for Precision {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        // `Metric::compute` is infallible by trait signature; surface
        // shape/argument errors (and `AverageMethod::None`, which this
        // scalar API cannot express) as NaN. Use `compute_precision`
        // directly, or [`MultiClassMetrics::compute`], for the `Result`/
        // per-class forms.
        compute_precision(predictions, targets, &self.average).unwrap_or(f64::NAN)
    }

    fn name(&self) -> &str {
        match self.average {
            AverageMethod::Micro => "precision_micro",
            AverageMethod::Macro => "precision_macro",
            AverageMethod::Weighted => "precision_weighted",
            AverageMethod::None => "precision",
        }
    }
}

/// Recall metric - Enhanced with scirs2-metrics compatibility
pub struct Recall {
    average: AverageMethod,
}

impl Recall {
    /// Create a new recall metric
    pub fn new(average: AverageMethod) -> Self {
        Self { average }
    }

    /// Create micro-averaged recall
    pub fn micro() -> Self {
        Self {
            average: AverageMethod::Micro,
        }
    }

    /// Create macro-averaged recall
    pub fn macro_averaged() -> Self {
        Self {
            average: AverageMethod::Macro,
        }
    }
}

impl Metric for Recall {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        // See the note on `Metric for Precision::compute` above.
        compute_recall(predictions, targets, &self.average).unwrap_or(f64::NAN)
    }

    fn name(&self) -> &str {
        match self.average {
            AverageMethod::Micro => "recall_micro",
            AverageMethod::Macro => "recall_macro",
            AverageMethod::Weighted => "recall_weighted",
            AverageMethod::None => "recall",
        }
    }
}

/// F1 Score metric - Enhanced with scirs2-metrics compatibility
pub struct F1Score {
    average: AverageMethod,
}

impl F1Score {
    /// Create a new F1 score metric
    pub fn new(average: AverageMethod) -> Self {
        Self { average }
    }

    /// Create micro-averaged F1 score
    pub fn micro() -> Self {
        Self {
            average: AverageMethod::Micro,
        }
    }

    /// Create macro-averaged F1 score
    pub fn macro_averaged() -> Self {
        Self {
            average: AverageMethod::Macro,
        }
    }
}

impl Metric for F1Score {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        // See the note on `Metric for Precision::compute` above.
        compute_f1_score(predictions, targets, &self.average).unwrap_or(f64::NAN)
    }

    fn name(&self) -> &str {
        match self.average {
            AverageMethod::Micro => "f1_micro",
            AverageMethod::Macro => "f1_macro",
            AverageMethod::Weighted => "f1_weighted",
            AverageMethod::None => "f1",
        }
    }
}

// Implementation functions for the metrics

/// Compute standard classification accuracy.
///
/// Accepts either a 1-D tensor of positive-class probabilities (binary
/// classification, thresholded at 0.5) or a 2-D `[n_rows, n_classes]`
/// tensor of per-class scores (argmax per row).
fn compute_standard_accuracy(predictions: &Tensor, targets: &Tensor) -> Result<f64, TorshError> {
    if predictions.numel() == 0 || targets.numel() == 0 {
        // An empty input has no accuracy to report. Returning `Ok(0.0)` here
        // would read as "the model got everything wrong"; the honest answer is
        // an error, which the `Metric::compute` boundary surfaces as NaN.
        return Err(TorshError::InvalidArgument(
            "accuracy is undefined for empty predictions or targets".to_string(),
        ));
    }

    let pred_vec = predictions
        .to_vec()
        .map_err(|e| TorshError::InvalidArgument(format!("failed to read predictions: {e}")))?;
    let targets_vec = targets
        .to_vec()
        .map_err(|e| TorshError::InvalidArgument(format!("failed to read targets: {e}")))?;

    let shape = predictions.shape();
    let dims = shape.dims();

    // Handle both 1D and 2D predictions
    let (rows, cols) = if dims.len() == 1 {
        // 1D predictions - binary classification with threshold 0.5
        let rows = dims[0];
        if targets_vec.len() != rows {
            return Err(TorshError::InvalidArgument(format!(
                "targets length {} does not match predictions length {rows}",
                targets_vec.len()
            )));
        }

        let mut correct = 0;
        for i in 0..rows {
            let predicted_class = if pred_vec[i] >= 0.5 { 1.0 } else { 0.0 };
            if (predicted_class - targets_vec[i]).abs() < 1e-6 {
                correct += 1;
            }
        }
        return Ok(correct as f64 / rows as f64);
    } else if dims.len() == 2 {
        let rows = dims[0];
        let cols = dims[1];
        if rows == 0 || cols == 0 {
            return Err(TorshError::InvalidArgument(
                "predictions tensor has a zero-sized dimension".to_string(),
            ));
        }
        if targets_vec.len() != rows {
            return Err(TorshError::InvalidArgument(format!(
                "targets length {} does not match predictions rows {rows}",
                targets_vec.len()
            )));
        }
        (rows, cols)
    } else {
        return Err(TorshError::InvalidArgument(format!(
            "predictions tensor must be 1-D or 2-D, got {}-D",
            dims.len()
        )));
    };

    let mut correct = 0;

    // Manually compute argmax for each row
    for i in 0..rows {
        let mut max_idx = 0;
        let mut max_val = pred_vec[i * cols];

        for j in 1..cols {
            let val = pred_vec[i * cols + j];
            if val > max_val {
                max_val = val;
                max_idx = j;
            }
        }

        if max_idx as i64 == targets_vec[i] as i64 {
            correct += 1;
        }
    }

    Ok(correct as f64 / rows as f64)
}

fn compute_top_k_accuracy(
    predictions: &Tensor,
    targets: &Tensor,
    k: usize,
) -> Result<f64, TorshError> {
    if predictions.numel() == 0 || targets.numel() == 0 {
        return Err(TorshError::InvalidArgument(
            "predictions/targets tensor is empty".to_string(),
        ));
    }

    let pred_vec = predictions
        .to_vec()
        .map_err(|e| TorshError::InvalidArgument(format!("failed to read predictions: {e}")))?;
    let targets_vec = targets
        .to_vec()
        .map_err(|e| TorshError::InvalidArgument(format!("failed to read targets: {e}")))?;

    let shape = predictions.shape();
    let dims = shape.dims();

    if dims.len() != 2 {
        return Err(TorshError::InvalidArgument(format!(
            "top-k accuracy requires a 2-D [n_rows, n_classes] predictions tensor, got {}-D",
            dims.len()
        )));
    }

    let rows = dims[0];
    let cols = dims[1];

    if rows == 0 || cols == 0 {
        return Err(TorshError::InvalidArgument(
            "predictions tensor has a zero-sized dimension".to_string(),
        ));
    }
    if targets_vec.len() != rows {
        return Err(TorshError::InvalidArgument(format!(
            "targets length {} does not match predictions rows {rows}",
            targets_vec.len()
        )));
    }
    if k > cols {
        return Err(TorshError::InvalidArgument(format!(
            "k ({k}) exceeds the number of classes ({cols})"
        )));
    }

    let mut correct = 0;

    // Manually compute top-k for each row
    for i in 0..rows {
        let target = targets_vec[i] as usize;

        // Get values for this row and find top-k indices
        let mut row_values: Vec<(f32, usize)> =
            (0..cols).map(|j| (pred_vec[i * cols + j], j)).collect();

        // Sort by value in descending order
        row_values.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .expect("row values should be comparable")
        });

        // Check if target is in top-k
        for j in 0..k.min(row_values.len()) {
            if row_values[j].1 == target {
                correct += 1;
                break;
            }
        }
    }

    Ok(correct as f64 / rows as f64)
}

/// Mean of per-class precision (macro average).
fn per_class_precision(tp: &[f64], fp: &[f64]) -> Vec<f64> {
    tp.iter()
        .zip(fp.iter())
        .map(|(&t, &f)| if t + f > 0.0 { t / (t + f) } else { 0.0 })
        .collect()
}

/// Per-class recall from confusion components.
fn per_class_recall(tp: &[f64], fn_: &[f64]) -> Vec<f64> {
    tp.iter()
        .zip(fn_.iter())
        .map(|(&t, &f)| if t + f > 0.0 { t / (t + f) } else { 0.0 })
        .collect()
}

/// Per-class F1 from per-class precision/recall (the binary F1 formula
/// applied class-by-class, *not* applied to already-averaged P/R -- see
/// F222).
fn per_class_f1(precision: &[f64], recall: &[f64]) -> Vec<f64> {
    precision
        .iter()
        .zip(recall.iter())
        .map(|(&p, &r)| {
            if p + r > 0.0 {
                2.0 * p * r / (p + r)
            } else {
                0.0
            }
        })
        .collect()
}

/// Per-class support (number of true instances), i.e. `tp + fn`.
fn support_from_components(tp: &[f64], fn_: &[f64]) -> Vec<f64> {
    tp.iter().zip(fn_.iter()).map(|(&t, &f)| t + f).collect()
}

/// Unweighted mean across classes (macro average). `0.0` for no classes.
fn macro_average(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// Support-weighted mean across classes. `0.0` when total support is 0.
fn weighted_average(values: &[f64], support: &[f64]) -> f64 {
    let total: f64 = support.iter().sum();
    if total > 0.0 {
        values
            .iter()
            .zip(support.iter())
            .map(|(&v, &s)| v * s)
            .sum::<f64>()
            / total
    } else {
        0.0
    }
}

/// Error returned by [`compute_precision`]/[`compute_recall`]/
/// [`compute_f1_score`] for `AverageMethod::None`: a scalar `f64` cannot
/// express a per-class result, so this is a real error rather than a
/// silent fallback to another averaging method.
fn per_class_not_representable_error() -> TorshError {
    TorshError::InvalidArgument(
        "AverageMethod::None requests per-class results, which a scalar metric cannot express; \
         use MultiClassMetrics::compute() for per-class precision/recall/F1"
            .to_string(),
    )
}

fn compute_precision(
    predictions: &Tensor,
    targets: &Tensor,
    average: &AverageMethod,
) -> Result<f64, TorshError> {
    let (tp, fp, fn_) = compute_confusion_components(predictions, targets)?;

    match average {
        AverageMethod::Micro => {
            let total_tp: f64 = tp.iter().sum();
            let total_fp: f64 = fp.iter().sum();

            Ok(if total_tp + total_fp > 0.0 {
                total_tp / (total_tp + total_fp)
            } else {
                0.0
            })
        }
        AverageMethod::Macro => Ok(macro_average(&per_class_precision(&tp, &fp))),
        AverageMethod::Weighted => {
            let support = support_from_components(&tp, &fn_);
            Ok(weighted_average(&per_class_precision(&tp, &fp), &support))
        }
        AverageMethod::None => Err(per_class_not_representable_error()),
    }
}

fn compute_recall(
    predictions: &Tensor,
    targets: &Tensor,
    average: &AverageMethod,
) -> Result<f64, TorshError> {
    let (tp, _fp, fn_) = compute_confusion_components(predictions, targets)?;

    match average {
        AverageMethod::Micro => {
            let total_tp: f64 = tp.iter().sum();
            let total_fn: f64 = fn_.iter().sum();

            Ok(if total_tp + total_fn > 0.0 {
                total_tp / (total_tp + total_fn)
            } else {
                0.0
            })
        }
        AverageMethod::Macro => Ok(macro_average(&per_class_recall(&tp, &fn_))),
        AverageMethod::Weighted => {
            let support = support_from_components(&tp, &fn_);
            Ok(weighted_average(&per_class_recall(&tp, &fn_), &support))
        }
        AverageMethod::None => Err(per_class_not_representable_error()),
    }
}

fn compute_f1_score(
    predictions: &Tensor,
    targets: &Tensor,
    average: &AverageMethod,
) -> Result<f64, TorshError> {
    match average {
        AverageMethod::Micro => {
            // Micro-F1 is mathematically the harmonic mean of micro
            // precision/recall (unlike macro/weighted-F1, see F222), so
            // computing it from the already-aggregated P/R is correct here.
            let precision = compute_precision(predictions, targets, &AverageMethod::Micro)?;
            let recall = compute_recall(predictions, targets, &AverageMethod::Micro)?;
            Ok(if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            })
        }
        AverageMethod::Macro => {
            let (tp, fp, fn_) = compute_confusion_components(predictions, targets)?;
            let precision = per_class_precision(&tp, &fp);
            let recall = per_class_recall(&tp, &fn_);
            Ok(macro_average(&per_class_f1(&precision, &recall)))
        }
        AverageMethod::Weighted => {
            let (tp, fp, fn_) = compute_confusion_components(predictions, targets)?;
            let precision = per_class_precision(&tp, &fp);
            let recall = per_class_recall(&tp, &fn_);
            let f1 = per_class_f1(&precision, &recall);
            let support = support_from_components(&tp, &fn_);
            Ok(weighted_average(&f1, &support))
        }
        AverageMethod::None => Err(per_class_not_representable_error()),
    }
}

/// Derive one predicted class index per row from a predictions tensor, and
/// the corresponding true class index per row from `targets`.
///
/// Accepts either a 1-D tensor of positive-class scores/probabilities
/// (binary classification, thresholded at 0.5 -- the same convention
/// [`compute_standard_accuracy`] uses) or a 2-D `[n_rows, n_classes]`
/// tensor of per-class scores (argmax per row).
fn predicted_and_target_classes(
    predictions: &Tensor,
    targets: &Tensor,
) -> Result<(Vec<usize>, Vec<usize>), TorshError> {
    let pred_vec = predictions
        .to_vec()
        .map_err(|e| TorshError::InvalidArgument(format!("failed to read predictions: {e}")))?;
    let targets_vec = targets
        .to_vec()
        .map_err(|e| TorshError::InvalidArgument(format!("failed to read targets: {e}")))?;

    let shape = predictions.shape();
    let dims = shape.dims();

    let preds: Vec<usize> = match dims.len() {
        1 => {
            let rows = dims[0];
            if rows == 0 {
                return Err(TorshError::InvalidArgument(
                    "predictions tensor is empty".to_string(),
                ));
            }
            pred_vec
                .iter()
                .map(|&p| if p >= 0.5 { 1 } else { 0 })
                .collect()
        }
        2 => {
            let rows = dims[0];
            let cols = dims[1];
            if rows == 0 || cols == 0 {
                return Err(TorshError::InvalidArgument(
                    "predictions tensor has a zero-sized dimension".to_string(),
                ));
            }
            let mut preds = Vec::with_capacity(rows);
            for i in 0..rows {
                let mut max_idx = 0;
                let mut max_val = pred_vec[i * cols];
                for j in 1..cols {
                    let val = pred_vec[i * cols + j];
                    if val > max_val {
                        max_val = val;
                        max_idx = j;
                    }
                }
                preds.push(max_idx);
            }
            preds
        }
        d => {
            return Err(TorshError::InvalidArgument(format!(
                "predictions tensor must be 1-D or 2-D, got {d}-D"
            )))
        }
    };

    if targets_vec.len() != preds.len() {
        return Err(TorshError::InvalidArgument(format!(
            "targets length {} does not match predictions row count {}",
            targets_vec.len(),
            preds.len()
        )));
    }

    let targets: Vec<usize> = targets_vec.iter().map(|&t| t as usize).collect();

    Ok((preds, targets))
}

/// Compute confusion matrix components (TP, FP, FN) for each class.
///
/// The number of classes is `max(observed target or predicted label) + 1`
/// -- derived from the data, never forced to a minimum (see F227), so a
/// legitimate single-class batch reports exactly 1 class.
fn compute_confusion_components(
    predictions: &Tensor,
    targets: &Tensor,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), TorshError> {
    let (preds_vec, targets_vec) = predicted_and_target_classes(predictions, targets)?;

    let num_classes = preds_vec
        .iter()
        .chain(targets_vec.iter())
        .max()
        .copied()
        .unwrap_or(0)
        + 1;

    let mut tp = vec![0.0; num_classes];
    let mut fp = vec![0.0; num_classes];
    let mut fn_ = vec![0.0; num_classes];

    // Compute confusion matrix components. Every index is guaranteed
    // in-bounds by construction: num_classes is `max(preds ∪ targets) + 1`.
    for i in 0..targets_vec.len() {
        let target = targets_vec[i];
        let pred = preds_vec[i];

        if target == pred {
            tp[target] += 1.0;
        } else {
            fp[pred] += 1.0;
            fn_[target] += 1.0;
        }
    }

    Ok((tp, fp, fn_))
}

/// Multi-class classification metrics with per-class statistics
#[derive(Debug, Clone)]
pub struct MultiClassMetrics {
    /// Per-class precision scores
    pub per_class_precision: Vec<f64>,
    /// Per-class recall scores
    pub per_class_recall: Vec<f64>,
    /// Per-class F1 scores
    pub per_class_f1: Vec<f64>,
    /// Macro-averaged F1 score (unweighted mean across classes)
    pub macro_avg: f64,
    /// Weighted-averaged F1 score (weighted by support)
    pub weighted_avg: f64,
    /// Per-class support (number of true instances for each class)
    pub support: Vec<usize>,
}

impl MultiClassMetrics {
    /// Compute multi-class metrics from predictions and targets.
    ///
    /// Errors if `predictions`/`targets` are empty, mismatched in length,
    /// or `predictions` is neither 1-D (binary, thresholded at 0.5) nor 2-D
    /// (`[n_rows, n_classes]`, argmax per row).
    pub fn compute(predictions: &Tensor, targets: &Tensor) -> Result<Self, TorshError> {
        let (tp, fp, fn_) = compute_confusion_components(predictions, targets)?;

        let per_class_precision = per_class_precision(&tp, &fp);
        let per_class_recall = per_class_recall(&tp, &fn_);
        let per_class_f1 = per_class_f1(&per_class_precision, &per_class_recall);
        let support: Vec<usize> = support_from_components(&tp, &fn_)
            .into_iter()
            .map(|s| s as usize)
            .collect();
        let support_f64: Vec<f64> = support.iter().map(|&s| s as f64).collect();

        let macro_avg = macro_average(&per_class_f1);
        let weighted_avg = weighted_average(&per_class_f1, &support_f64);

        Ok(MultiClassMetrics {
            per_class_precision,
            per_class_recall,
            per_class_f1,
            macro_avg,
            weighted_avg,
            support,
        })
    }

    /// Get the number of classes
    pub fn num_classes(&self) -> usize {
        self.per_class_f1.len()
    }

    /// Get metrics for a specific class
    pub fn class_metrics(&self, class_idx: usize) -> Option<(f64, f64, f64, usize)> {
        if class_idx < self.num_classes() {
            Some((
                self.per_class_precision[class_idx],
                self.per_class_recall[class_idx],
                self.per_class_f1[class_idx],
                self.support[class_idx],
            ))
        } else {
            None
        }
    }

    /// Format metrics as a human-readable string
    pub fn format(&self) -> String {
        let mut result = String::new();
        result.push_str("Multi-Class Metrics:\n");
        result.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        for i in 0..self.num_classes() {
            result.push_str(&format!(
                "Class {}: Precision={:.4}, Recall={:.4}, F1={:.4}, Support={}\n",
                i,
                self.per_class_precision[i],
                self.per_class_recall[i],
                self.per_class_f1[i],
                self.support[i]
            ));
        }

        result.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        result.push_str(&format!("Macro Avg F1: {:.4}\n", self.macro_avg));
        result.push_str(&format!("Weighted Avg F1: {:.4}\n", self.weighted_avg));

        result
    }
}

/// Confusion Matrix for multi-class classification
#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    /// The confusion matrix as a 2D array (rows=true labels, cols=predicted labels)
    pub matrix: Vec<Vec<usize>>,
    /// Number of classes
    pub num_classes: usize,
    /// Class labels (optional)
    pub labels: Option<Vec<String>>,
}

impl ConfusionMatrix {
    /// Create a confusion matrix from predictions and targets.
    ///
    /// Errors under the same conditions as
    /// [`MultiClassMetrics::compute`]. The matrix is sized from the
    /// observed labels (never forced to a minimum of 2x2, see F227).
    pub fn compute(predictions: &Tensor, targets: &Tensor) -> Result<Self, TorshError> {
        let (matrix, num_classes) = compute_confusion_matrix(predictions, targets)?;

        Ok(ConfusionMatrix {
            matrix,
            num_classes,
            labels: None,
        })
    }

    /// Create a confusion matrix with custom class labels.
    ///
    /// Errors under the same conditions as [`Self::compute`].
    pub fn compute_with_labels(
        predictions: &Tensor,
        targets: &Tensor,
        labels: Vec<String>,
    ) -> Result<Self, TorshError> {
        let (matrix, num_classes) = compute_confusion_matrix(predictions, targets)?;

        Ok(ConfusionMatrix {
            matrix,
            num_classes,
            labels: Some(labels),
        })
    }

    /// Get the value at position (true_class, predicted_class)
    pub fn get(&self, true_class: usize, predicted_class: usize) -> Option<usize> {
        if true_class < self.num_classes && predicted_class < self.num_classes {
            Some(self.matrix[true_class][predicted_class])
        } else {
            None
        }
    }

    /// Get the total number of samples
    pub fn total(&self) -> usize {
        self.matrix
            .iter()
            .map(|row| row.iter().sum::<usize>())
            .sum()
    }

    /// Get accuracy from the confusion matrix
    pub fn accuracy(&self) -> f64 {
        let correct: usize = (0..self.num_classes).map(|i| self.matrix[i][i]).sum();
        let total = self.total();

        if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Get the diagonal (true positives for each class)
    pub fn diagonal(&self) -> Vec<usize> {
        (0..self.num_classes).map(|i| self.matrix[i][i]).collect()
    }

    /// Normalize the confusion matrix by true labels (row-wise)
    pub fn normalize_by_true(&self) -> Vec<Vec<f64>> {
        self.matrix
            .iter()
            .map(|row| {
                let sum: usize = row.iter().sum();
                if sum > 0 {
                    row.iter().map(|&val| val as f64 / sum as f64).collect()
                } else {
                    vec![0.0; self.num_classes]
                }
            })
            .collect()
    }

    /// Normalize the confusion matrix by predicted labels (column-wise)
    pub fn normalize_by_pred(&self) -> Vec<Vec<f64>> {
        let mut result = vec![vec![0.0; self.num_classes]; self.num_classes];

        // Compute column sums
        let mut col_sums = vec![0; self.num_classes];
        for row in &self.matrix {
            for (j, &val) in row.iter().enumerate() {
                col_sums[j] += val;
            }
        }

        // Normalize
        for i in 0..self.num_classes {
            for j in 0..self.num_classes {
                if col_sums[j] > 0 {
                    result[i][j] = self.matrix[i][j] as f64 / col_sums[j] as f64;
                }
            }
        }

        result
    }

    /// Normalize the confusion matrix by all samples
    pub fn normalize_all(&self) -> Vec<Vec<f64>> {
        let total = self.total();
        if total > 0 {
            self.matrix
                .iter()
                .map(|row| row.iter().map(|&val| val as f64 / total as f64).collect())
                .collect()
        } else {
            vec![vec![0.0; self.num_classes]; self.num_classes]
        }
    }

    /// Format the confusion matrix as a string
    pub fn format(&self) -> String {
        let mut result = String::new();
        result.push_str("Confusion Matrix:\n");

        // Header
        result.push_str("        ");
        for j in 0..self.num_classes {
            if let Some(ref labels) = self.labels {
                if j < labels.len() {
                    result.push_str(&format!("{:>8} ", labels[j]));
                } else {
                    result.push_str(&format!("{:>8} ", j));
                }
            } else {
                result.push_str(&format!("{:>8} ", j));
            }
        }
        result.push('\n');

        // Matrix rows
        for i in 0..self.num_classes {
            if let Some(ref labels) = self.labels {
                if i < labels.len() {
                    result.push_str(&format!("{:>8} ", labels[i]));
                } else {
                    result.push_str(&format!("{:>8} ", i));
                }
            } else {
                result.push_str(&format!("{:>8} ", i));
            }

            for j in 0..self.num_classes {
                result.push_str(&format!("{:>8} ", self.matrix[i][j]));
            }
            result.push('\n');
        }

        result
    }

    /// Format normalized confusion matrix (by true labels)
    pub fn format_normalized(&self) -> String {
        let normalized = self.normalize_by_true();
        let mut result = String::new();
        result.push_str("Normalized Confusion Matrix (by true labels):\n");

        // Header
        result.push_str("        ");
        for j in 0..self.num_classes {
            if let Some(ref labels) = self.labels {
                if j < labels.len() {
                    result.push_str(&format!("{:>8} ", labels[j]));
                } else {
                    result.push_str(&format!("{:>8} ", j));
                }
            } else {
                result.push_str(&format!("{:>8} ", j));
            }
        }
        result.push('\n');

        // Matrix rows
        for i in 0..self.num_classes {
            if let Some(ref labels) = self.labels {
                if i < labels.len() {
                    result.push_str(&format!("{:>8} ", labels[i]));
                } else {
                    result.push_str(&format!("{:>8} ", i));
                }
            } else {
                result.push_str(&format!("{:>8} ", i));
            }

            for j in 0..self.num_classes {
                result.push_str(&format!("{:>8.4} ", normalized[i][j]));
            }
            result.push('\n');
        }

        result
    }
}

/// Helper function to compute the confusion matrix.
///
/// See [`predicted_and_target_classes`] for the accepted input shapes and
/// [`compute_confusion_components`]'s doc comment for the class-count
/// derivation (never forced to a minimum, see F227).
fn compute_confusion_matrix(
    predictions: &Tensor,
    targets: &Tensor,
) -> Result<(Vec<Vec<usize>>, usize), TorshError> {
    let (preds_vec, targets_vec) = predicted_and_target_classes(predictions, targets)?;

    let num_classes = preds_vec
        .iter()
        .chain(targets_vec.iter())
        .max()
        .copied()
        .unwrap_or(0)
        + 1;

    let mut matrix = vec![vec![0; num_classes]; num_classes];

    for i in 0..targets_vec.len() {
        matrix[targets_vec[i]][preds_vec[i]] += 1;
    }

    Ok((matrix, num_classes))
}

/// Threshold-dependent metrics for binary classification
#[derive(Debug, Clone)]
pub struct ThresholdMetrics {
    /// Optimal threshold for classification
    pub optimal_threshold: f64,
    /// Precision-recall curve (precision, recall) pairs at different thresholds
    pub precision_recall_curve: (Vec<f64>, Vec<f64>),
    /// ROC curve (false positive rate, true positive rate) pairs
    pub roc_curve: (Vec<f64>, Vec<f64>),
    /// Thresholds used for the curves
    pub thresholds: Vec<f64>,
}

impl ThresholdMetrics {
    /// Compute threshold metrics for binary classification
    /// predictions: probabilities for the positive class (shape: \[n\])
    /// targets: binary labels (0 or 1) (shape: \[n\])
    pub fn compute(predictions: &Tensor, targets: &Tensor) -> Self {
        let (pred_vec, targets_vec) = match (predictions.to_vec(), targets.to_vec()) {
            (Ok(p), Ok(t)) => (p, t),
            _ => {
                return ThresholdMetrics {
                    optimal_threshold: 0.5,
                    precision_recall_curve: (vec![0.0], vec![0.0]),
                    roc_curve: (vec![0.0], vec![0.0]),
                    thresholds: vec![0.5],
                }
            }
        };

        if pred_vec.is_empty() || targets_vec.is_empty() || pred_vec.len() != targets_vec.len() {
            return ThresholdMetrics {
                optimal_threshold: 0.5,
                precision_recall_curve: (vec![0.0], vec![0.0]),
                roc_curve: (vec![0.0], vec![0.0]),
                thresholds: vec![0.5],
            };
        }

        // Generate thresholds
        let mut thresholds: Vec<f64> = pred_vec.iter().map(|&x| x as f64).collect();
        thresholds.sort_by(|a, b| {
            a.partial_cmp(b)
                .expect("threshold values should be comparable")
        });
        thresholds.dedup();

        // Add boundary thresholds strictly below the minimum and above the
        // maximum observed score, so the sequence stays monotonic
        // regardless of the score range (scores confined to [0,1] are the
        // common case, but nothing here guarantees it -- raw logits can be
        // negative or exceed 1.0). Inserting fixed 0.0/1.0 sentinels broke
        // monotonicity whenever scores fell outside [0,1] (see F225): e.g.
        // scores [-2.0, 1.0, 3.0] became [0.0, -2.0, 1.0, 3.0], which the
        // trapezoidal AUC in `calculate_auc` sums as unsigned area,
        // inflating the result past 1.0 instead of cancelling out.
        //
        // The epsilon is scaled to the magnitude of the observed scores
        // (floored at 1.0) rather than a fixed constant: thresholds are
        // narrowed to `f32` before comparison in
        // `compute_binary_confusion_matrix`, and a fixed epsilon like
        // `1e-9` rounds away to nothing once the scores are large enough
        // that it falls below `f32`'s representable precision at that
        // magnitude.
        let min_score = *thresholds.first().unwrap_or(&0.0);
        let max_score = *thresholds.last().unwrap_or(&1.0);
        let magnitude = min_score.abs().max(max_score.abs()).max(1.0);
        let epsilon = magnitude * 1e-3;
        thresholds.insert(0, min_score - epsilon);
        thresholds.push(max_score + epsilon);

        let mut precisions = Vec::new();
        let mut recalls = Vec::new();
        let mut fprs = Vec::new();
        let mut tprs = Vec::new();

        // Compute metrics at each threshold
        for &threshold in &thresholds {
            let (tp, fp, tn, fn_) =
                compute_binary_confusion_matrix(&pred_vec, &targets_vec, threshold as f32);

            let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 1.0 };

            let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };

            let fpr = if fp + tn > 0.0 { fp / (fp + tn) } else { 0.0 };

            let tpr = recall;

            precisions.push(precision);
            recalls.push(recall);
            fprs.push(fpr);
            tprs.push(tpr);
        }

        // Find optimal threshold (maximize F1 score)
        let mut best_f1 = 0.0;
        let mut optimal_threshold = 0.5;

        for (i, &threshold) in thresholds.iter().enumerate() {
            let precision = precisions[i];
            let recall = recalls[i];

            let f1 = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };

            if f1 > best_f1 {
                best_f1 = f1;
                optimal_threshold = threshold;
            }
        }

        ThresholdMetrics {
            optimal_threshold,
            precision_recall_curve: (precisions, recalls),
            roc_curve: (fprs, tprs),
            thresholds,
        }
    }

    /// Calculate Area Under the ROC Curve (AUC-ROC)
    pub fn auc_roc(&self) -> f64 {
        let (fprs, tprs) = &self.roc_curve;
        calculate_auc(fprs, tprs)
    }

    /// Calculate Area Under the Precision-Recall Curve (AUC-PR)
    pub fn auc_pr(&self) -> f64 {
        let (precisions, recalls) = &self.precision_recall_curve;
        calculate_auc(recalls, precisions)
    }

    /// Get precision and recall at the optimal threshold
    pub fn optimal_metrics(&self) -> (f64, f64) {
        if let Some(idx) = self
            .thresholds
            .iter()
            .position(|&t| (t - self.optimal_threshold).abs() < 1e-9)
        {
            (
                self.precision_recall_curve.0[idx],
                self.precision_recall_curve.1[idx],
            )
        } else {
            (0.0, 0.0)
        }
    }
}

/// Helper function to compute binary confusion matrix at a threshold
fn compute_binary_confusion_matrix(
    predictions: &[f32],
    targets: &[f32],
    threshold: f32,
) -> (f64, f64, f64, f64) {
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut tn = 0.0;
    let mut fn_ = 0.0;

    for (&pred, &target) in predictions.iter().zip(targets.iter()) {
        let pred_class = if pred >= threshold { 1.0 } else { 0.0 };

        if target > 0.5 {
            // Positive class
            if pred_class > 0.5 {
                tp += 1.0;
            } else {
                fn_ += 1.0;
            }
        } else {
            // Negative class
            if pred_class > 0.5 {
                fp += 1.0;
            } else {
                tn += 1.0;
            }
        }
    }

    (tp, fp, tn, fn_)
}

/// Calculate Area Under Curve using trapezoidal rule
fn calculate_auc(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }

    let mut auc = 0.0;
    for i in 1..x.len() {
        let dx = x[i] - x[i - 1];
        let avg_y = (y[i] + y[i - 1]) / 2.0;
        auc += dx.abs() * avg_y;
    }

    auc
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;
    use torsh_tensor::creation::from_vec;

    #[test]
    fn test_multi_class_metrics() {
        // Create sample predictions and targets
        let predictions = from_vec(
            vec![
                0.9, 0.1, 0.0, // Class 0
                0.2, 0.7, 0.1, // Class 1
                0.1, 0.2, 0.7, // Class 2
                0.8, 0.1, 0.1, // Class 0
                0.1, 0.8, 0.1, // Class 1
            ],
            &[5, 3],
            DeviceType::Cpu,
        )
        .unwrap();
        let targets = from_vec(vec![0.0, 1.0, 2.0, 0.0, 1.0], &[5], DeviceType::Cpu).unwrap();

        let metrics = MultiClassMetrics::compute(&predictions, &targets).unwrap();

        // Perfect predictions, so all metrics should be 1.0
        assert_eq!(metrics.num_classes(), 3);
        assert!((metrics.macro_avg - 1.0).abs() < 1e-6);
        assert!((metrics.weighted_avg - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_confusion_matrix() {
        let predictions = from_vec(
            vec![
                0.9, 0.1, 0.0, // Predicts 0, actually 0 (TP)
                0.2, 0.7, 0.1, // Predicts 1, actually 1 (TP)
                0.1, 0.2, 0.7, // Predicts 2, actually 2 (TP)
            ],
            &[3, 3],
            DeviceType::Cpu,
        )
        .unwrap();
        let targets = from_vec(vec![0.0, 1.0, 2.0], &[3], DeviceType::Cpu).unwrap();

        let cm = ConfusionMatrix::compute(&predictions, &targets).unwrap();

        assert_eq!(cm.num_classes, 3);
        assert_eq!(cm.total(), 3);
        assert!((cm.accuracy() - 1.0).abs() < 1e-6);
        assert_eq!(cm.diagonal(), vec![1, 1, 1]);
    }

    #[test]
    fn test_threshold_metrics() {
        // Binary classification probabilities
        let predictions = from_vec(vec![0.9, 0.8, 0.3, 0.2, 0.7], &[5], DeviceType::Cpu).unwrap();
        let targets = from_vec(vec![1.0, 1.0, 0.0, 0.0, 1.0], &[5], DeviceType::Cpu).unwrap();

        let metrics = ThresholdMetrics::compute(&predictions, &targets);

        // Should find a reasonable threshold
        assert!(metrics.optimal_threshold > 0.0);
        assert!(metrics.optimal_threshold < 1.0);

        // AUC should be high for this good prediction
        let auc = metrics.auc_roc();
        assert!(auc > 0.8);
    }

    #[test]
    fn test_confusion_matrix_normalization() {
        let predictions = from_vec(
            vec![
                0.9, 0.1, // Predicts 0, actually 0
                0.2, 0.8, // Predicts 1, actually 1
                0.7, 0.3, // Predicts 0, actually 0
                0.3, 0.7, // Predicts 1, actually 1
            ],
            &[4, 2],
            DeviceType::Cpu,
        )
        .unwrap();
        let targets = from_vec(vec![0.0, 1.0, 0.0, 1.0], &[4], DeviceType::Cpu).unwrap();

        let cm = ConfusionMatrix::compute(&predictions, &targets).unwrap();

        let normalized = cm.normalize_by_true();
        // Each row should sum to 1.0
        for row in &normalized {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }
}
