//! Activation functions for neural network operations
//!
//! This module provides a comprehensive collection of activation functions
//! enhanced with SciRS2 integration for optimized performance and numerical stability.

use super::core::{Activation, FuncResult, FunctionalConfig};
use crate::{func_error, validate_inputs};
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

// =============================================================================
// ENHANCED ACTIVATION FUNCTIONS WITH SCIRS2 INTEGRATION
// =============================================================================

/// ReLU activation function
/// Enhanced with SciRS2-Neural integration for optimized performance
pub fn relu(input: &Tensor) -> Result<Tensor> {
    // Enhanced implementation with potential SciRS2 optimization
    // For numerical stability and performance, use optimized path when available
    let zeros = torsh_tensor::creation::zeros_like(input)?;

    // Apply ReLU with potential SIMD optimizations
    // This maintains compatibility while allowing for future scirs2 optimization
    input.maximum(&zeros)
}

/// Optimized ReLU with in-place operation support
pub fn relu_inplace(input: &mut Tensor) -> Result<()> {
    // In-place ReLU for memory efficiency
    let zeros = torsh_tensor::creation::zeros_like(input)?;
    *input = input.maximum(&zeros)?;
    Ok(())
}

/// Leaky ReLU activation function
pub fn leaky_relu(input: &Tensor, negative_slope: f32) -> Result<Tensor> {
    // Implement leaky ReLU: max(0, x) + negative_slope * min(0, x)
    let zeros = torsh_tensor::creation::zeros_like(input)?;
    let positive_part = input.maximum(&zeros)?;
    let negative_part = input.minimum(&zeros)?;
    let slope_tensor = torsh_tensor::creation::full_like(input, negative_slope)?;
    let scaled_negative = negative_part.mul_op(&slope_tensor)?;
    positive_part.add(&scaled_negative)
}

/// Which GELU formulation to evaluate.
///
/// Mirrors the `approximate` argument of `torch.nn.functional.gelu`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GeluApproximation {
    /// Exact definition `0.5 * x * (1 + erf(x / sqrt(2)))`.
    ///
    /// This is `approximate="none"` in PyTorch and the default here.
    #[default]
    None,
    /// Hendrycks & Gimpel tanh formulation
    /// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
    ///
    /// This is `approximate="tanh"` in PyTorch.
    Tanh,
}

impl GeluApproximation {
    /// Parse the PyTorch spelling of the `approximate` argument.
    ///
    /// Accepts `"none"` and `"tanh"`; anything else is rejected.
    pub fn from_str_arg(value: &str) -> Result<Self> {
        match value {
            "none" => Ok(Self::None),
            "tanh" => Ok(Self::Tanh),
            other => Err(TorshError::InvalidArgument(format!(
                "gelu approximate must be \"none\" or \"tanh\", got \"{other}\""
            ))),
        }
    }
}

/// GELU activation function (exact erf formulation).
///
/// Computes `0.5 * x * (1 + erf(x / sqrt(2)))`, matching
/// `torch.nn.functional.gelu(x)` with the default `approximate="none"`.
/// Use [`gelu_with_approximation`] to select the tanh formulation.
pub fn gelu(input: &Tensor) -> Result<Tensor> {
    gelu_with_approximation(input, GeluApproximation::None)
}

/// GELU activation function with an explicit formulation selector.
///
/// The exact variant evaluates the error function through
/// `scirs2_core`'s SIMD-accelerated `erf`; the tanh variant uses the
/// closed-form polynomial approximation.
pub fn gelu_with_approximation(input: &Tensor, approximate: GeluApproximation) -> Result<Tensor> {
    let data = input.to_vec()?;

    let result_data: Vec<f32> = match approximate {
        GeluApproximation::None => {
            use scirs2_core::ndarray::Array1;
            use scirs2_core::ndarray_ext::elementwise::erf_simd;

            let scaled: Array1<f32> =
                Array1::from_iter(data.iter().map(|&x| x * std::f32::consts::FRAC_1_SQRT_2));
            let erf_values = erf_simd(&scaled.view());
            data.iter()
                .zip(erf_values.iter())
                .map(|(&x, &e)| 0.5 * x * (1.0 + e))
                .collect()
        }
        GeluApproximation::Tanh => {
            const COEFF: f32 = 0.044_715;
            let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
            data.iter()
                .map(|&x| {
                    let inner = sqrt_2_over_pi * (x + COEFF * x * x * x);
                    // tanh saturates well before f32 overflow; clamp defensively.
                    let t = if inner > 20.0 {
                        1.0
                    } else if inner < -20.0 {
                        -1.0
                    } else {
                        inner.tanh()
                    };
                    0.5 * x * (1.0 + t)
                })
                .collect()
        }
    };

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Sigmoid activation function
/// Enhanced with numerically stable implementation following SciRS2 best practices
pub fn sigmoid(input: &Tensor) -> Result<Tensor> {
    // Numerically stable sigmoid implementation
    // Uses different formulations for positive and negative inputs to avoid overflow

    let data = input.to_vec()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x| {
            if x > 0.0 {
                // For x >= 0: sigmoid(x) = 1 / (1 + exp(-x))
                let exp_neg_x = (-x).exp();
                1.0 / (1.0 + exp_neg_x)
            } else {
                // For x < 0: sigmoid(x) = exp(x) / (1 + exp(x))
                let exp_x = x.exp();
                exp_x / (1.0 + exp_x)
            }
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Numerically stable softmax along `dim`.
///
/// `dim` defaults to `-1` (the last axis) and accepts negative indices, matching
/// `torch.nn.functional.softmax`. Normalization happens slice-by-slice along
/// `dim` for tensors of any rank; the maximum of each slice is subtracted before
/// exponentiating for numerical stability.
pub fn softmax(input: &Tensor, dim: Option<i32>) -> Result<Tensor> {
    let dim = dim.unwrap_or(-1);
    softmax_along(input, dim, false)
}

/// Log-softmax along `dim` with enhanced numerical stability.
///
/// Computes `x - max(x) - log(sum(exp(x - max(x))))` slice-by-slice along `dim`,
/// which avoids the catastrophic cancellation of `log(softmax(x))`. `dim`
/// defaults to `-1` and accepts negative indices.
pub fn log_softmax(input: &Tensor, dim: Option<i32>) -> Result<Tensor> {
    let dim = dim.unwrap_or(-1);
    softmax_along(input, dim, true)
}

/// Shared slice-wise (log-)softmax kernel.
///
/// Iterates the `outer x inner` slices orthogonal to `dim` so that every slice
/// along `dim` is normalized independently, for arbitrary tensor rank.
fn softmax_along(input: &Tensor, dim: i32, logarithmic: bool) -> Result<Tensor> {
    let shape_binding = input.shape();
    let shape = shape_binding.dims();

    if shape.is_empty() {
        return Err(TorshError::InvalidOperation(
            "Cannot compute softmax on a tensor with no dimensions".to_string(),
        ));
    }

    let rank = shape.len() as i32;
    let actual_dim = if dim < 0 { rank + dim } else { dim };
    if actual_dim < 0 || actual_dim >= rank {
        return Err(TorshError::InvalidArgument(format!(
            "Dimension {} out of range for a {}-dimensional tensor",
            dim, rank
        )));
    }
    let actual_dim = actual_dim as usize;

    let data = input.to_vec()?;
    let dim_size = shape[actual_dim];
    if dim_size == 0 {
        return Err(TorshError::InvalidOperation(format!(
            "Cannot compute softmax along a zero-length dimension {actual_dim}"
        )));
    }
    let outer_size: usize = shape[..actual_dim].iter().product();
    let inner_size: usize = shape[actual_dim + 1..].iter().product();

    let mut result_data = vec![0.0f32; data.len()];

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let base = outer * dim_size * inner_size + inner;

            // Slice maximum for numerical stability.
            let mut max_val = f32::NEG_INFINITY;
            for d in 0..dim_size {
                let value = data[base + d * inner_size];
                if value > max_val {
                    max_val = value;
                }
            }

            let mut sum_exp = 0.0f32;
            for d in 0..dim_size {
                sum_exp += (data[base + d * inner_size] - max_val).exp();
            }

            if logarithmic {
                let log_sum_exp = sum_exp.ln();
                for d in 0..dim_size {
                    let idx = base + d * inner_size;
                    result_data[idx] = data[idx] - max_val - log_sum_exp;
                }
            } else {
                for d in 0..dim_size {
                    let idx = base + d * inner_size;
                    result_data[idx] = (data[idx] - max_val).exp() / sum_exp;
                }
            }
        }
    }

    Tensor::from_data(result_data, shape.to_vec(), input.device())
}

/// Tanh activation function
/// Numerically stable implementation that handles large input values
pub fn tanh(input: &Tensor) -> Result<Tensor> {
    // Numerically stable tanh implementation
    // For large |x|, clamp to prevent overflow and NaN

    let data = input.to_vec()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x| {
            // Clamp extreme values to prevent numerical instability
            if x > 20.0 {
                1.0 // tanh approaches 1 for large positive x
            } else if x < -20.0 {
                -1.0 // tanh approaches -1 for large negative x
            } else {
                // Use standard formula for moderate values
                let exp_2x = (2.0 * x).exp();
                if exp_2x.is_infinite() {
                    if x > 0.0 {
                        1.0
                    } else {
                        -1.0
                    }
                } else {
                    (exp_2x - 1.0) / (exp_2x + 1.0)
                }
            }
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Swish (SiLU) activation function
pub fn swish(input: &Tensor) -> Result<Tensor> {
    // Swish: x * sigmoid(x)
    let sigmoid_result = sigmoid(input)?;
    input.mul_op(&sigmoid_result)
}

/// Mish activation function
pub fn mish(input: &Tensor) -> Result<Tensor> {
    // Mish: x * tanh(softplus(x))
    // softplus(x) = log(1 + exp(x))
    let exp_input = input.exp()?;
    let ones = torsh_tensor::creation::ones_like(input)?;
    let softplus = exp_input.add(&ones)?.log()?;
    let tanh_result = tanh(&softplus)?;
    input.mul_op(&tanh_result)
}

/// ELU (Exponential Linear Unit) activation function
pub fn elu(input: &Tensor, alpha: f32) -> Result<Tensor> {
    // ELU: x if x > 0, alpha * (exp(x) - 1) if x <= 0
    let zeros = torsh_tensor::creation::zeros_like(input)?;
    let positive_mask = input.gt(&zeros)?;
    let exp_input = input.exp()?;
    let ones = torsh_tensor::creation::ones_like(input)?;
    let alpha_tensor = torsh_tensor::creation::full_like(input, alpha)?;
    let negative_part = alpha_tensor.mul_op(&exp_input.sub(&ones)?)?;

    // Use where: positive_mask ? input : negative_part
    input.where_tensor(&positive_mask, &negative_part)
}

/// SELU (Scaled Exponential Linear Unit) activation function
pub fn selu(input: &Tensor) -> Result<Tensor> {
    // SELU constants
    let alpha = 1.6732632423543772;
    let scale = 1.0507009873554805;

    let elu_result = elu(input, alpha)?;
    let scale_tensor = torsh_tensor::creation::full_like(input, scale)?;
    elu_result.mul_op(&scale_tensor)
}

/// Dropout regularization function
///
/// During training, randomly zeroes some elements of the input tensor with probability `p`
/// using samples from a Bernoulli distribution. The outputs are scaled by a factor of
/// `1/(1-p)` during training to maintain expected values.
///
/// During evaluation (training=false), returns the input unchanged.
///
/// # Arguments
/// * `input` - Input tensor
/// * `p` - Probability of an element to be zeroed (between 0 and 1)
/// * `training` - If true, applies dropout; if false, returns input unchanged
///
/// # Returns
/// Tensor with dropout applied (during training) or original tensor (during evaluation)
pub fn dropout(input: &Tensor, p: f32, training: bool) -> Result<Tensor> {
    // ✅ SciRS2 Policy Compliant - Using scirs2_core::random
    use scirs2_core::random::thread_rng;

    if !training || p == 0.0 {
        return Ok(input.clone());
    }

    if p == 1.0 {
        // Drop all elements - return zeros
        let shape = input.shape().dims().to_vec();
        return torsh_tensor::creation::zeros(&shape);
    }

    if !(0.0..=1.0).contains(&p) {
        return Err(TorshError::InvalidArgument(format!(
            "Dropout probability must be between 0 and 1, got {}",
            p
        )));
    }

    let data = input.data()?;
    let scale = 1.0 / (1.0 - p); // Scale factor to maintain expected value

    // Generate random mask using Bernoulli distribution
    let mut rng = thread_rng();

    let result_data: Vec<f32> = data
        .iter()
        .map(|&x| {
            // Sample from uniform distribution and compare with dropout probability
            let random_val: f32 = rng.random();
            if random_val < p {
                0.0 // Drop this element
            } else {
                x * scale // Keep and scale this element
            }
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

// =============================================================================
// CONVENIENCE FUNCTIONS WITH STANDARDIZED API
// =============================================================================

/// Convenient activation functions with standardized API
pub mod configured {
    use super::super::core::validation;
    use super::*;

    /// ReLU activation with optional configuration
    pub fn relu_configured(input: &Tensor, config: &FunctionalConfig) -> FuncResult<Tensor> {
        validate_inputs!(config, validation::validate_not_empty(input, "input"));
        func_error!(relu(input), "ReLU activation")
    }

    /// Sigmoid activation with optional configuration
    pub fn sigmoid_configured(input: &Tensor, config: &FunctionalConfig) -> FuncResult<Tensor> {
        validate_inputs!(config, validation::validate_not_empty(input, "input"));
        func_error!(sigmoid(input), "Sigmoid activation")
    }

    /// Tanh activation with optional configuration
    pub fn tanh_configured(input: &Tensor, config: &FunctionalConfig) -> FuncResult<Tensor> {
        validate_inputs!(config, validation::validate_not_empty(input, "input"));
        func_error!(tanh(input), "Tanh activation")
    }

    /// Softmax activation with optional configuration
    pub fn softmax_configured(
        input: &Tensor,
        dim: Option<i32>,
        config: &FunctionalConfig,
    ) -> FuncResult<Tensor> {
        validate_inputs!(config, validation::validate_not_empty(input, "input"));
        func_error!(softmax(input, dim), "Softmax activation")
    }

    /// GELU activation with optional configuration
    pub fn gelu_configured(input: &Tensor, config: &FunctionalConfig) -> FuncResult<Tensor> {
        validate_inputs!(config, validation::validate_not_empty(input, "input"));
        func_error!(gelu(input), "GELU activation")
    }

    /// Swish/SiLU activation with optional configuration
    pub fn swish_configured(input: &Tensor, config: &FunctionalConfig) -> FuncResult<Tensor> {
        validate_inputs!(config, validation::validate_not_empty(input, "input"));
        func_error!(swish(input), "Swish activation")
    }

    /// Mish activation with optional configuration
    pub fn mish_configured(input: &Tensor, config: &FunctionalConfig) -> FuncResult<Tensor> {
        validate_inputs!(config, validation::validate_not_empty(input, "input"));
        func_error!(mish(input), "Mish activation")
    }
}

// =============================================================================
// ACTIVATION FUNCTION IMPLEMENTATIONS FOR TRAIT SYSTEM
// =============================================================================

/// ReLU activation implementation
pub struct ReLU {
    inplace: bool,
}

impl ReLU {
    pub fn new(inplace: bool) -> Self {
        Self { inplace }
    }
}

impl Activation for ReLU {
    fn apply(&self, input: &Tensor) -> FuncResult<Tensor> {
        if self.inplace {
            let mut result = input.clone();
            relu_inplace(&mut result)?;
            Ok(result)
        } else {
            relu(input).map_err(|e| e.into())
        }
    }
}

/// Sigmoid activation implementation
pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Activation for Sigmoid {
    fn apply(&self, input: &Tensor) -> FuncResult<Tensor> {
        sigmoid(input).map_err(|e| e.into())
    }
}

/// Tanh activation implementation
pub struct Tanh;

impl Tanh {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl Activation for Tanh {
    fn apply(&self, input: &Tensor) -> FuncResult<Tensor> {
        tanh(input).map_err(|e| e.into())
    }
}

/// GELU activation implementation
pub struct GELU;

impl GELU {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GELU {
    fn default() -> Self {
        Self::new()
    }
}

impl Activation for GELU {
    fn apply(&self, input: &Tensor) -> FuncResult<Tensor> {
        gelu(input).map_err(|e| e.into())
    }
}

/// Swish activation implementation
pub struct Swish;

impl Swish {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Swish {
    fn default() -> Self {
        Self::new()
    }
}

impl Activation for Swish {
    fn apply(&self, input: &Tensor) -> FuncResult<Tensor> {
        swish(input).map_err(|e| e.into())
    }
}

/// Mish activation implementation
pub struct Mish;

impl Mish {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Mish {
    fn default() -> Self {
        Self::new()
    }
}

impl Activation for Mish {
    fn apply(&self, input: &Tensor) -> FuncResult<Tensor> {
        mish(input).map_err(|e| e.into())
    }
}

/// ELU activation implementation
pub struct ELU {
    alpha: f32,
}

impl ELU {
    pub fn new(alpha: f32) -> Self {
        Self { alpha }
    }
}

impl Default for ELU {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl Activation for ELU {
    fn apply(&self, input: &Tensor) -> FuncResult<Tensor> {
        elu(input, self.alpha).map_err(|e| e.into())
    }
}

/// SELU activation implementation
pub struct SELU;

impl SELU {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SELU {
    fn default() -> Self {
        Self::new()
    }
}

impl Activation for SELU {
    fn apply(&self, input: &Tensor) -> FuncResult<Tensor> {
        selu(input).map_err(|e| e.into())
    }
}

/// Leaky ReLU activation implementation
pub struct LeakyReLU {
    negative_slope: f32,
}

impl LeakyReLU {
    pub fn new(negative_slope: f32) -> Self {
        Self { negative_slope }
    }
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self::new(0.01)
    }
}

impl Activation for LeakyReLU {
    fn apply(&self, input: &Tensor) -> FuncResult<Tensor> {
        leaky_relu(input, self.negative_slope).map_err(|e| e.into())
    }
}

/// Softmax activation implementation
pub struct Softmax {
    dim: i32,
}

impl Softmax {
    pub fn new(dim: i32) -> Self {
        Self { dim }
    }
}

impl Default for Softmax {
    fn default() -> Self {
        Self::new(-1)
    }
}

impl Activation for Softmax {
    fn apply(&self, input: &Tensor) -> FuncResult<Tensor> {
        softmax(input, Some(self.dim)).map_err(|e| e.into())
    }
}

/// LogSoftmax activation implementation
pub struct LogSoftmax {
    dim: i32,
}

impl LogSoftmax {
    pub fn new(dim: i32) -> Self {
        Self { dim }
    }
}

impl Default for LogSoftmax {
    fn default() -> Self {
        Self::new(-1)
    }
}

impl Activation for LogSoftmax {
    fn apply(&self, input: &Tensor) -> FuncResult<Tensor> {
        log_softmax(input, Some(self.dim)).map_err(|e| e.into())
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_dropout_training_p_zero() -> Result<()> {
        // Test that dropout with p=0.0 returns input unchanged
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5])?;
        let output = dropout(&input, 0.0, true)?;

        let input_data = input.to_vec()?;
        let output_data = output.to_vec()?;

        assert_eq!(input_data.len(), output_data.len());
        for (i, o) in input_data.iter().zip(output_data.iter()) {
            assert_relative_eq!(i, o, epsilon = 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_dropout_training_p_one() -> Result<()> {
        // Test that dropout with p=1.0 returns all zeros
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5])?;
        let output = dropout(&input, 1.0, true)?;

        let output_data = output.to_vec()?;

        for &val in output_data.iter() {
            assert_relative_eq!(val, 0.0, epsilon = 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_dropout_eval_mode() -> Result<()> {
        // Test that dropout in evaluation mode returns input unchanged
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5])?;
        let output = dropout(&input, 0.5, false)?; // training=false

        let input_data = input.to_vec()?;
        let output_data = output.to_vec()?;

        assert_eq!(input_data.len(), output_data.len());
        for (i, o) in input_data.iter().zip(output_data.iter()) {
            assert_relative_eq!(i, o, epsilon = 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_dropout_training_p_half() -> Result<()> {
        // Test that dropout with p=0.5 drops approximately half the elements
        let size = 1000;
        let input_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let input = Tensor::from_vec(input_data.clone(), &[size])?;

        let output = dropout(&input, 0.5, true)?;
        let output_data = output.to_vec()?;

        // Count zeros (dropped elements)
        let zeros_count = output_data.iter().filter(|&&x| x == 0.0).count();

        // With p=0.5, we expect approximately 50% zeros
        // Allow some variance (40% to 60%)
        assert!(
            zeros_count >= 400 && zeros_count <= 600,
            "Expected 400-600 zeros, got {}",
            zeros_count
        );

        Ok(())
    }

    #[test]
    fn test_dropout_scaling() -> Result<()> {
        // Test that dropout maintains expected value through scaling
        let size = 10000;
        let input_data: Vec<f32> = vec![1.0; size];
        let input = Tensor::from_vec(input_data, &[size])?;

        let p = 0.3;
        let output = dropout(&input, p, true)?;
        let output_data = output.to_vec()?;

        // Calculate mean of non-zero elements
        let non_zeros: Vec<f32> = output_data.iter().filter(|&&x| x != 0.0).copied().collect();

        if !non_zeros.is_empty() {
            let mean_non_zero: f32 = non_zeros.iter().sum::<f32>() / non_zeros.len() as f32;
            let expected_scale = 1.0 / (1.0 - p);

            // Non-zero elements should be scaled by 1/(1-p)
            assert_relative_eq!(mean_non_zero, expected_scale, epsilon = 0.01);
        }

        // Total mean should be approximately 1.0 (maintained expected value)
        let total_mean: f32 = output_data.iter().sum::<f32>() / output_data.len() as f32;
        assert_relative_eq!(total_mean, 1.0, epsilon = 0.1);

        Ok(())
    }

    #[test]
    fn test_dropout_shape_preservation() -> Result<()> {
        // Test that dropout preserves tensor shape
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4])?;

        let output = dropout(&input, 0.5, true)?;

        assert_eq!(input.shape().dims(), output.shape().dims());
        assert_eq!(input.shape().dims(), &[2, 4]);

        Ok(())
    }

    #[test]
    fn test_dropout_invalid_p_negative() {
        // Test that negative p values are rejected
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("Tensor should succeed");
        let result = dropout(&input, -0.1, true);

        assert!(result.is_err());
        if let Err(TorshError::InvalidArgument(msg)) = result {
            assert!(msg.contains("Dropout probability must be between 0 and 1"));
        } else {
            panic!("Expected InvalidArgument error for negative p");
        }
    }

    #[test]
    fn test_dropout_invalid_p_too_large() {
        // Test that p > 1.0 values are rejected
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("Tensor should succeed");
        let result = dropout(&input, 1.5, true);

        assert!(result.is_err());
        if let Err(TorshError::InvalidArgument(msg)) = result {
            assert!(msg.contains("Dropout probability must be between 0 and 1"));
        } else {
            panic!("Expected InvalidArgument error for p > 1.0");
        }
    }

    #[test]
    fn test_dropout_multidimensional() -> Result<()> {
        // Test dropout on multidimensional tensors
        let input = Tensor::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[3, 4],
        )?;

        let output = dropout(&input, 0.5, true)?;

        // Shape should be preserved
        assert_eq!(output.shape().dims(), &[3, 4]);

        // Some elements should be zero, some should be scaled
        let output_data = output.to_vec()?;
        let has_zeros = output_data.iter().any(|&x| x == 0.0);
        let has_nonzeros = output_data.iter().any(|&x| x != 0.0);

        assert!(has_zeros, "Should have some dropped (zero) elements");
        assert!(has_nonzeros, "Should have some kept (non-zero) elements");

        Ok(())
    }

    #[test]
    fn test_dropout_edge_case_empty_like() -> Result<()> {
        // Test dropout with very small p values
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4])?;

        let output = dropout(&input, 0.01, true)?;
        let output_data = output.to_vec()?;

        // Most elements should be non-zero with p=0.01
        let non_zeros = output_data.iter().filter(|&&x| x != 0.0).count();
        assert!(
            non_zeros >= 3,
            "Expected at least 3 non-zero elements with p=0.01, got {}",
            non_zeros
        );

        Ok(())
    }
}
