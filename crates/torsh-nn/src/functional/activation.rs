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

/// ReLU activation function.
///
/// # Autograd
///
/// Delegates to [`Tensor::relu`], which records
/// [`UnaryKind::Relu`](torsh_tensor::core_ops::UnaryKind) and keeps every
/// dispatch band (scalar / parallel / f32 SIMD) behind one derivative.
///
/// The previous body was `input.maximum(&zeros_like(input))`. That became
/// differentiable when `Tensor::maximum` started recording, but it built a
/// throw-away `zeros` tensor on every call and split the gradient `0.5 / 0.5`
/// at the tie `x == 0`; `Tensor::relu`'s recorded predicate is `x > 0`, which is
/// what `torch.relu` propagates there. Forward values are bit-identical:
/// measured over a 1631-point sweep of `[-30, 30]` plus `+/-0.0` and the
/// denormal edges, `maximum(x, 0)` and `relu(x)` agreed on every bit, including
/// the sign of zero at `x == -0.0` (both return `+0.0`).
pub fn relu(input: &Tensor) -> Result<Tensor> {
    input.relu()
}

/// Optimized ReLU with in-place operation support.
///
/// Rebinds `input` to the activated tensor, which stays on the autograd graph:
/// the tensor handed in is still reachable as the recorded operand, so a
/// `backward()` further downstream reaches whatever produced it.
pub fn relu_inplace(input: &mut Tensor) -> Result<()> {
    *input = input.relu()?;
    Ok(())
}

/// Leaky ReLU activation function.
///
/// # Autograd
///
/// Delegates to [`Tensor::leaky_relu`], which records
/// [`Operation::LeakyRelu`](torsh_tensor::Operation) with the slope stored
/// alongside the operand, so backward evaluates the same `x > 0` predicate the
/// forward did.
///
/// The previous body was `maximum(x, 0) + slope * minimum(x, 0)`, five tensor
/// allocations for one element-wise pass. Its gradient at the kink was
/// `0.5 * (1 + slope)` — the average of the two one-sided derivatives, because
/// both `maximum` and `minimum` split ties evenly. PyTorch propagates `slope`
/// there, which is what the delegation now does. Forward values are
/// bit-identical apart from the sign of zero at `x == -0.0` (the composition
/// returned `+0.0`, the delegation returns `-0.0 * slope == -0.0`).
pub fn leaky_relu(input: &Tensor, negative_slope: f32) -> Result<Tensor> {
    input.leaky_relu(negative_slope)
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
/// `scirs2_core`'s SIMD-accelerated `erf`; the tanh variant delegates to
/// [`Tensor::gelu`], which evaluates the identical closed form.
///
/// # Autograd
///
/// `GeluApproximation::Tanh` is exactly what
/// [`UnaryKind::Gelu`](torsh_tensor::core_ops::UnaryKind) records, so that arm
/// is a one-line delegation. `GeluApproximation::None` has no recording tensor
/// primitive — torsh-tensor ships no `erf` — so it keeps its own kernel and
/// attaches the analytic derivative through `with_analytic_gradient`; see
/// that helper for why the forward values stay bit-identical.
pub fn gelu_with_approximation(input: &Tensor, approximate: GeluApproximation) -> Result<Tensor> {
    match approximate {
        GeluApproximation::None => gelu_exact(input),
        // Tensor::gelu is `0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))`,
        // the same closed form this arm used to evaluate by hand. It needs no
        // defensive |inner| > 20 clamp because it calls `tanh` directly rather
        // than routing through `exp`, and it deliberately has no SIMD band (the
        // clamped Pade kernel that used to serve numel > 1000 computed a
        // measurably different function).
        GeluApproximation::Tanh => input.gelu(),
    }
}

/// Exact (erf) GELU: `0.5 * x * (1 + erf(x / sqrt(2)))`.
///
/// Kept as its own kernel rather than delegated, because delegating to
/// [`Tensor::gelu`] would silently swap the documented `approximate="none"`
/// definition for the tanh approximation — a 4e-4 change on a typical batch,
/// three orders of magnitude larger than any rounding difference.
fn gelu_exact(input: &Tensor) -> Result<Tensor> {
    use scirs2_core::ndarray::Array1;
    use scirs2_core::ndarray_ext::elementwise::erf_simd;

    /// `1 / sqrt(2 * pi)`, the standard normal density's normalising constant.
    const INV_SQRT_2PI: f32 = 0.398_942_28;

    let data = input.to_vec()?;
    let dims = input.shape().dims().to_vec();

    let scaled: Array1<f32> =
        Array1::from_iter(data.iter().map(|&x| x * std::f32::consts::FRAC_1_SQRT_2));
    let erf_values = erf_simd(&scaled.view());
    let forward_data: Vec<f32> = data
        .iter()
        .zip(erf_values.iter())
        .map(|(&x, &e)| 0.5 * x * (1.0 + e))
        .collect();
    let forward = Tensor::from_data(forward_data, dims.clone(), input.device())?;

    if !input.requires_grad() {
        return Ok(forward);
    }

    // d/dx [0.5*x*(1 + erf(x/sqrt2))] = 0.5*(1 + erf(x/sqrt2)) + x*phi(x),
    // with phi the standard normal density. The `erf` values are reused, so the
    // derivative costs one extra pass over the data and no extra `erf` call.
    let derivative_data: Vec<f32> = data
        .iter()
        .zip(erf_values.iter())
        .map(|(&x, &e)| 0.5 * (1.0 + e) + x * (-0.5 * x * x).exp() * INV_SQRT_2PI)
        .collect();
    let derivative = Tensor::from_data(derivative_data, dims.clone(), input.device())?;

    with_analytic_gradient(input, forward, &derivative, data, dims)
}

/// Attach an analytic derivative to a forward value that no recording tensor
/// operation can express.
///
/// Returns `forward + (input - frozen) * derivative`, where `frozen` is a fresh
/// leaf holding `input`'s own values. Element-wise:
///
/// * the residual `input - frozen` is `+0.0` everywhere, so
///   `forward + 0.0 * derivative` is **bit-identical** to `forward` for every
///   finite input (adding a zero of either sign to a float leaves it unchanged,
///   and `0.0 * d` is finite for the finite `d` this is used with);
/// * the residual's derivative with respect to `input` is `1`, so the product
///   contributes exactly `derivative` to `d(result)/d(input)`.
///
/// `frozen` is built with [`Tensor::from_data`] and **not** with
/// `Tensor::detach()`: `detach` clones the operand's recorded `operation`, so
/// the residual's negative leg would flow back into `input`'s own subgraph and
/// cancel the positive one. A leaf-input gradient check cannot see that; the
/// non-leaf check in `tests/hardening_nn_activations.rs` can.
///
/// This is the tensor-level equivalent of a hand-written
/// `torch.autograd.Function`, and it is a stopgap: the clean fix is a recording
/// `erf` (or an exact-GELU `UnaryKind`) in torsh-tensor.
fn with_analytic_gradient(
    input: &Tensor,
    forward: Tensor,
    derivative: &Tensor,
    values: Vec<f32>,
    dims: Vec<usize>,
) -> Result<Tensor> {
    let frozen = Tensor::from_data(values, dims, input.device())?;
    let residual = input.sub(&frozen)?;
    let correction = residual.mul_op(derivative)?;
    forward.add(&correction)
}

/// Sigmoid activation function.
///
/// # Autograd
///
/// Delegates to [`Tensor::sigmoid`], which records
/// [`UnaryKind::Sigmoid`](torsh_tensor::core_ops::UnaryKind).
///
/// The previous body evaluated the branch-stable form
/// (`1/(1+exp(-x))` for `x > 0`, `exp(x)/(1+exp(x))` otherwise) and handed the
/// values to `Tensor::from_data`, i.e. returned a detached leaf: `swish`,
/// `compile_time`'s MLP and every `Sigmoid` layer built on it silently lost
/// their gradients. `Tensor::sigmoid` evaluates `1/(1+exp(-x))` on every
/// branch, which cannot overflow in f32 either (`exp` saturates to `+inf` and
/// `1/inf` is `0`); the two agree to at most 1 ULP — measured worst case
/// 1.2e-7 absolute over `[-30, 30]`.
pub fn sigmoid(input: &Tensor) -> Result<Tensor> {
    input.sigmoid()
}

/// Numerically stable softmax along `dim`.
///
/// `dim` defaults to `-1` (the last axis) and accepts negative indices, matching
/// `torch.nn.functional.softmax`. Normalization happens slice-by-slice along
/// `dim` for tensors of any rank; the maximum of each slice is subtracted before
/// exponentiating for numerical stability.
///
/// # Autograd
///
/// Delegates to [`Tensor::softmax`], which builds the normalization out of
/// recording tensor operations (`sub` / `exp` / `sum_dim` / `div`) so the result
/// stays attached to `input`. The previous slice-wise kernel rebuilt its output
/// with `Tensor::from_data` and returned a detached leaf, which silently cut
/// every consumer — most visibly `cross_entropy` — off the graph.
pub fn softmax(input: &Tensor, dim: Option<i32>) -> Result<Tensor> {
    let dim = dim.unwrap_or(-1);
    let actual_dim = normalize_softmax_dim(input, dim)?;
    input.softmax(actual_dim as i32)
}

/// Log-softmax along `dim` with enhanced numerical stability.
///
/// Computes `x - max(x) - log(sum(exp(x - max(x))))` slice-by-slice along `dim`,
/// which avoids the catastrophic cancellation of `log(softmax(x))`. `dim`
/// defaults to `-1` and accepts negative indices.
///
/// # Autograd
///
/// Delegates to [`Tensor::log_softmax`], which records the exact log-softmax
/// Jacobian (`Operation::LogSoftmax`) rather than the composed sub/exp/sum/log
/// graph. This is what makes `cross_entropy` differentiable end to end.
pub fn log_softmax(input: &Tensor, dim: Option<i32>) -> Result<Tensor> {
    let dim = dim.unwrap_or(-1);
    let actual_dim = normalize_softmax_dim(input, dim)?;
    input.log_softmax(actual_dim as i32)
}

/// Validate `dim` against `input`'s shape and normalize it to a non-negative axis.
///
/// Shared by [`softmax`] and [`log_softmax`] so both keep the exact error
/// contract the old slice-wise kernel enforced: rank-0 inputs, out-of-range axes
/// (negative indices included) and zero-length axes are all rejected before any
/// work happens. The tensor-level operations reject these too, but with
/// different error variants and wording, so the checks stay here.
fn normalize_softmax_dim(input: &Tensor, dim: i32) -> Result<usize> {
    let shape_binding = input.shape();
    let shape = shape_binding.dims();

    if shape.is_empty() {
        return Err(TorshError::InvalidOperation(
            "Cannot compute softmax on a tensor with no dimensions".to_string(),
        ));
    }

    let rank = shape.len() as i32;
    let actual_dim = if dim < 0 { rank + dim } else { dim };
    let dim_size = if actual_dim < 0 {
        None
    } else {
        shape.get(actual_dim as usize).copied()
    };
    let dim_size = dim_size.ok_or_else(|| {
        TorshError::InvalidArgument(format!(
            "Dimension {} out of range for a {}-dimensional tensor",
            dim, rank
        ))
    })?;
    let actual_dim = actual_dim as usize;

    if dim_size == 0 {
        return Err(TorshError::InvalidOperation(format!(
            "Cannot compute softmax along a zero-length dimension {actual_dim}"
        )));
    }

    Ok(actual_dim)
}

/// Tanh activation function.
///
/// # Autograd
///
/// Delegates to [`Tensor::tanh`], which records
/// [`UnaryKind::Tanh`](torsh_tensor::core_ops::UnaryKind).
///
/// The previous body evaluated `(exp(2x) - 1) / (exp(2x) + 1)` with a defensive
/// `|x| > 20` clamp and returned a detached `Tensor::from_data` leaf, which is
/// what made `mish`'s gradient wrong (sign-flipped at `x = -1.3`) and every
/// `Tanh` layer untrainable. `Tensor::tanh` calls the platform `tanh`, which is
/// both attached and *more* accurate: the old expression cancels catastrophically
/// near the origin, returning exactly `0.0` for `|x| < ~6e-8` where the true
/// value is `x`. Away from that region the two agree to at most 1 ULP (measured
/// worst case 1.2e-7 absolute over `[-30, 30]`).
pub fn tanh(input: &Tensor) -> Result<Tensor> {
    input.tanh()
}

/// Swish (SiLU) activation function, `x * sigmoid(x)`.
///
/// # Autograd
///
/// Correct by composition: both factors record, so backward assembles
/// `sigma(x) + x * sigma(x) * (1 - sigma(x))`. Before [`sigmoid`] was
/// delegated, the inner factor was a detached leaf and only the outer multiply
/// recorded — the result *was* attached and `backward()` *did* succeed, it just
/// returned `grad * sigma(x)`, which is 21.9% wrong at `x = 1.1` and has the
/// wrong sign for `x < -1.28`. That is the failure mode this wave exists to
/// remove: no error, no `None` gradient, just a number that trains slowly.
pub fn swish(input: &Tensor) -> Result<Tensor> {
    let sigmoid_result = sigmoid(input)?;
    input.mul_op(&sigmoid_result)
}

/// Mish activation function, `x * tanh(softplus(x))`.
///
/// # Autograd
///
/// Correct by composition now that [`tanh`] records; see `softplus` for why
/// the inner term is no longer evaluated as `ln(exp(x) + 1)`.
pub fn mish(input: &Tensor) -> Result<Tensor> {
    let tanh_result = softplus(input)?.tanh()?;
    input.mul_op(&tanh_result)
}

/// Numerically stable `softplus(x) = ln(1 + exp(x))`, evaluated as
/// `max(x, 0) + ln(1 + exp(-|x|))`.
///
/// The naive `ln(exp(x) + 1)` overflows to `+inf` for `x > 88.7` in f32. Its
/// *forward* survived that — `tanh(inf)` is `1`, so `mish(x)` still came out as
/// `x` — but the gradient does not: `d/du ln(u)` is `1/inf == 0` while
/// `d/dx exp(x)` is `inf`, and `0 * inf` is `NaN`. Recording `tanh` would
/// therefore have traded a silently-wrong gradient for a `NaN` one above the
/// overflow knee. Every operation here records
/// (`abs`, `mul_scalar`, `exp`, `add_scalar`, `ln`, `clamp_min`, `add`), and the
/// two expressions agree to 9.5e-7 absolute wherever the naive one is finite.
///
/// The `clamp_min` arm's derivative is `1` at exactly `x == 0` where the true
/// `softplus'(0)` is `0.5`; `mish` multiplies that term by `x`, so its own
/// gradient at the origin is unaffected.
fn softplus(input: &Tensor) -> Result<Tensor> {
    let tail = input
        .abs()?
        .mul_scalar(-1.0)?
        .exp()?
        .add_scalar(1.0)?
        .ln()?;
    input.clamp_min(0.0)?.add(&tail)
}

/// ELU (Exponential Linear Unit) activation function:
/// `x` for `x > 0`, `alpha * (exp(x) - 1)` otherwise.
///
/// # Autograd
///
/// Composed from recording operations around a **detached 0/1 indicator** of the
/// positive branch. Freezing that indicator is exact rather than an
/// approximation: the branch predicate is locally constant, so its derivative is
/// zero almost everywhere — the same argument that lets `dropout` freeze its
/// Bernoulli mask.
///
/// The previous body selected with `input.where_tensor(&input.gt(&zeros), ..)`.
/// Neither `gt` nor `where_tensor` records, so the result was a detached leaf
/// and `selu` — which is just a scaled `elu` — inherited that.
///
/// Two details keep the forward bit-identical to the `where_tensor` select
/// rather than merely equal on the common case:
///
/// * the positive branch is `clamp_min(x, 0) * [x > 0]`, not `x * [x > 0]`, so
///   `x = -inf` multiplies a hard zero instead of producing `-inf * 0 == NaN`;
/// * the negative branch exponentiates `min(x, 0)`, so `exp` cannot overflow for
///   large positive `x` (where the mask discards the branch anyway) and
///   `alpha * (inf - 1) * 0 == NaN` cannot arise.
///
/// The gradient at exactly `x == 0` is `alpha`, matching ATen's
/// `elu_backward` (`output > 0 ? grad : grad * (output + alpha)`).
pub fn elu(input: &Tensor, alpha: f32) -> Result<Tensor> {
    let data = input.to_vec()?;
    let dims = input.shape().dims().to_vec();

    let positive: Vec<f32> = data
        .iter()
        .map(|&x| if x > 0.0 { 1.0 } else { 0.0 })
        .collect();
    let negative: Vec<f32> = positive.iter().map(|&m| 1.0 - m).collect();
    let positive_mask = Tensor::from_data(positive, dims.clone(), input.device())?;
    let negative_mask = Tensor::from_data(negative, dims, input.device())?;

    let positive_part = input.clamp_min(0.0)?.mul_op(&positive_mask)?;
    let negative_part = input
        .clamp_max(0.0)?
        .exp()?
        .add_scalar(-1.0)?
        .mul_scalar(alpha)?
        .mul_op(&negative_mask)?;
    positive_part.add(&negative_part)
}

/// SELU (Scaled Exponential Linear Unit) activation function.
///
/// # Autograd
///
/// Differentiable by composition now that [`elu`] records; the body is
/// unchanged, so the forward values are bit-identical to the previous release.
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
///
/// # Autograd
///
/// The Bernoulli draw is materialised as its own mask tensor and applied with a
/// multiply, so the result records `Operation::Mul` and the gradient reaching
/// `input` is `grad * mask`. Rebuilding the scaled values through
/// `Tensor::from_data` would return a detached leaf and cut every layer that
/// sits behind the dropout (inter-layer dropout of a multi-layer LSTM/GRU, for
/// instance).
pub fn dropout(input: &Tensor, p: f32, training: bool) -> Result<Tensor> {
    // ✅ SciRS2 Policy Compliant - Using scirs2_core::random
    use scirs2_core::random::thread_rng;

    if !training || p == 0.0 {
        return Ok(input.clone());
    }

    if p == 1.0 {
        // Drop all elements: scaling by zero keeps the result on the graph with
        // the correct (identically zero) gradient.
        return input.mul_scalar(0.0);
    }

    if !(0.0..=1.0).contains(&p) {
        return Err(TorshError::InvalidArgument(format!(
            "Dropout probability must be between 0 and 1, got {}",
            p
        )));
    }

    let numel = input.numel();
    let scale = 1.0 / (1.0 - p); // Scale factor to maintain expected value

    // Generate random mask using Bernoulli distribution: one draw per element,
    // in element order, so the consumption pattern is unchanged.
    let mut rng = thread_rng();

    let mask_data: Vec<f32> = (0..numel)
        .map(|_| {
            // Sample from uniform distribution and compare with dropout probability
            let random_val: f32 = rng.random();
            if random_val < p {
                0.0 // Drop this element
            } else {
                scale // Keep and scale this element
            }
        })
        .collect();

    let mask = Tensor::from_data(mask_data, input.shape().dims().to_vec(), input.device())?;
    input.mul_op(&mask)
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
