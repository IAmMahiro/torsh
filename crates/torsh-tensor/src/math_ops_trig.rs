//! Transcendental and trigonometric math operations for tensors.
//!
//! This module is included by `math_ops.rs` via `#[path]` and re-exported with `pub use`.
//! It covers all floating-point transcendental functions:
//! - Square root, exponential, logarithms
//! - Trigonometric: sin, cos, tan, asin, acos, atan
//! - Hyperbolic: sinh, cosh, tanh
//! - Activation: GELU, leaky ReLU
//! - Rounding: floor, ceil, round, trunc, fract
//! - Power, negation, sign

use super::*;

// Mathematical functions for floating-point tensors
impl<T: TensorElement + Copy> Tensor<T>
where
    T: scirs2_core::numeric::Float + torsh_core::dtype::FloatElement,
{
    /// Square root of all elements
    pub fn sqrt(&self) -> Result<Self> {
        let result = self.map(|x| x.sqrt())?;
        Ok(self.record_unary(result, UnaryKind::Sqrt))
    }

    /// Square of all elements
    ///
    /// Records [`Operation::Power`] with exponent `2.0`: `map` is
    /// forward-only (see [`Tensor::neg`]'s doc for why), so without this
    /// `square()` would return a `requires_grad` leaf that swallows the
    /// gradient of everything upstream of it. The `Power` backward arm
    /// already exists and needs no extra trait bound here -- `2.0` is a
    /// literal `f32`, not a value converted from `T` (that conversion is
    /// what [`Tensor::pow`]'s `Into<f32>` bound is for).
    pub fn square(&self) -> Result<Self> {
        let mut result = self.map(|x| x * x)?;
        if crate::should_record_grad(self.requires_grad) {
            result.requires_grad = true;
            result.operation = Operation::Power {
                input: Arc::new(self.clone()),
                exponent: 2.0,
            };
        }
        Ok(result)
    }

    /// Reciprocal square root of all elements (1/sqrt(x))
    pub fn rsqrt(&self) -> Result<Self> {
        let result =
            self.map(|x| T::from(1.0).expect("numeric conversion should succeed") / x.sqrt())?;
        Ok(self.record_unary(result, UnaryKind::Rsqrt))
    }

    /// Reciprocal of all elements (1/x)
    pub fn reciprocal(&self) -> Result<Self> {
        let result = self.map(|x| T::from(1.0).expect("numeric conversion should succeed") / x)?;
        Ok(self.record_unary(result, UnaryKind::Reciprocal))
    }

    /// Exponential of all elements
    pub fn exp(&self) -> Result<Self> {
        let result = self.map(|x| x.exp())?;
        Ok(self.record_unary(result, UnaryKind::Exp))
    }

    /// Natural logarithm of all elements
    pub fn ln(&self) -> Result<Self> {
        let result = self.map(|x| x.ln())?;
        Ok(self.record_unary(result, UnaryKind::Ln))
    }

    /// Logarithm base 10 of all elements
    pub fn log10(&self) -> Result<Self> {
        let result = self.map(|x| x.log10())?;
        Ok(self.record_unary(result, UnaryKind::Log10))
    }

    /// Logarithm base 2 of all elements
    pub fn log2(&self) -> Result<Self> {
        let result = self.map(|x| x.log2())?;
        Ok(self.record_unary(result, UnaryKind::Log2))
    }

    /// Natural logarithm of all elements
    pub fn log(&self) -> Result<Self> {
        let result = self.map(|x| x.ln())?;
        Ok(self.record_unary(result, UnaryKind::Ln))
    }

    /// Sine of all elements
    pub fn sin(&self) -> Result<Self> {
        let result = self.map(|x| x.sin())?;
        Ok(self.record_unary(result, UnaryKind::Sin))
    }

    /// Cosine of all elements
    pub fn cos(&self) -> Result<Self> {
        let result = self.map(|x| x.cos())?;
        Ok(self.record_unary(result, UnaryKind::Cos))
    }

    /// Tangent of all elements
    pub fn tan(&self) -> Result<Self> {
        let result = self.map(|x| x.tan())?;
        Ok(self.record_unary(result, UnaryKind::Tan))
    }

    /// GELU (Gaussian Error Linear Unit) activation, in the tanh
    /// approximation `0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))`.
    ///
    /// One closed form at every tensor size — see the private `gelu_forward`
    /// below for why the f32 SIMD kernel is deliberately not used — and one recorded
    /// derivative ([`crate::core_ops::UnaryKind::Gelu`]) that matches it exactly.
    pub fn gelu(&self) -> Result<Self> {
        let result = self.gelu_forward()?;
        Ok(self.record_unary(result, UnaryKind::Gelu))
    }

    /// Forward-only GELU (parallel or scalar path); autograd is recorded by
    /// the public [`Tensor::gelu`] wrapper, so every dispatch path shares one
    /// recorded derivative.
    ///
    /// **There is deliberately no f32 SIMD path.** Until Wave 4, tensors with
    /// `numel > 1000` were routed to scirs2-core's `adaptive_simd_gelu_f32`,
    /// which substitutes a clamped Pade rational for `tanh`. That is a
    /// *different function*, not a faster evaluation of the same one: the
    /// forward jumped by up to 0.0211 absolute (515% relative at `x = -2.40`)
    /// the moment a tensor crossed 1000 elements, and the recorded backward —
    /// which differentiates the exact-tanh closed form — no longer matched the
    /// forward that had run. Both defects are pinned by
    /// `tests/hardening_autograd_primitives.rs`
    /// (`gelu_forward_is_continuous_across_the_dispatch_threshold` and
    /// `gelu_backward_matches_finite_difference_above_the_simd_threshold`).
    /// The parallel path below still fans large tensors out across worker
    /// threads; it just evaluates the same `compute_gelu_scalar` everywhere.
    fn gelu_forward(&self) -> Result<Self> {
        // GPU fast path (deferred): GELU is not yet a variant of oxicuda's
        // `ComputeBackend` `UnaryOp`.  oxicuda-blas already ships
        // `elementwise::unary::gelu`, so once `UnaryOp::Gelu` is added upstream
        // (TODO(oxicuda-unaryop-gelu)) dispatch here via
        // `crate::gpu_dispatch::try_unary_f32(self, UnaryOp::Gelu)`.

        // ✅ SciRS2 Parallel Processing - Use parallel computation for medium
        // and large tensors (there is no separate SIMD band; see above).
        #[cfg(feature = "parallel")]
        {
            if self.numel() > 100 {
                return self.parallel_map(|x| self.compute_gelu_scalar(x));
            }
        }

        // Fallback to sequential processing
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        self.map(|x| self.compute_gelu_scalar(x))
    }

    /// Compute GELU for a single scalar value
    fn compute_gelu_scalar(&self, x: T) -> T {
        let pi = T::from(std::f64::consts::PI).expect("numeric conversion should succeed");
        let two = T::from(2.0).expect("numeric conversion should succeed");
        let sqrt_2_over_pi = (two / pi).sqrt();
        let point_044715 = T::from(0.044715).expect("numeric conversion should succeed");
        let one = <T as scirs2_core::numeric::One>::one();
        let half = T::from(0.5).expect("numeric conversion should succeed");

        let x_cubed = x * x * x;
        let tanh_input = sqrt_2_over_pi * (x + point_044715 * x_cubed);
        half * x * (one + tanh_input.tanh())
    }

    /// Leaky ReLU activation function with negative slope
    pub fn leaky_relu(&self, negative_slope: T) -> Result<Self> {
        let result = self.leaky_relu_forward(negative_slope)?;
        Ok(self.record_leaky_relu(result, negative_slope))
    }

    /// Forward-only leaky ReLU; autograd is recorded by the public
    /// [`Tensor::leaky_relu`] wrapper.
    fn leaky_relu_forward(&self, negative_slope: T) -> Result<Self> {
        // GPU fast path (deferred): parameterized LeakyReLU has no oxicuda
        // `ComputeBackend` `UnaryOp` variant (the flat `unary` op takes no
        // scalar parameter).  TODO(oxicuda-unaryop-leakyrelu): add a
        // parameterized variant upstream, then dispatch via gpu_dispatch.
        self.map(|x| {
            if x > scirs2_core::numeric::Zero::zero() {
                x
            } else {
                negative_slope * x
            }
        })
    }

    /// Arcsine of all elements
    pub fn asin(&self) -> Result<Self> {
        let result = self.map(|x| x.asin())?;
        Ok(self.record_unary(result, UnaryKind::Asin))
    }

    /// Arccosine of all elements
    pub fn acos(&self) -> Result<Self> {
        let result = self.map(|x| x.acos())?;
        Ok(self.record_unary(result, UnaryKind::Acos))
    }

    /// Arctangent of all elements
    pub fn atan(&self) -> Result<Self> {
        let result = self.map(|x| x.atan())?;
        Ok(self.record_unary(result, UnaryKind::Atan))
    }

    /// Hyperbolic sine of all elements
    pub fn sinh(&self) -> Result<Self> {
        let result = self.map(|x| x.sinh())?;
        Ok(self.record_unary(result, UnaryKind::Sinh))
    }

    /// Hyperbolic cosine of all elements
    pub fn cosh(&self) -> Result<Self> {
        let result = self.map(|x| x.cosh())?;
        Ok(self.record_unary(result, UnaryKind::Cosh))
    }

    /// Hyperbolic tangent of all elements
    pub fn tanh(&self) -> Result<Self> {
        let result = self.tanh_forward()?;
        Ok(self.record_unary(result, UnaryKind::Tanh))
    }

    /// Forward-only tanh (GPU or scalar path); autograd is recorded by the
    /// public [`Tensor::tanh`] wrapper.
    fn tanh_forward(&self) -> Result<Self> {
        // GPU fast path: f32 CUDA tensors dispatch to oxicuda's ComputeBackend.
        // Declines to None (CPU fallback) unless a GPU backend is active.
        #[cfg(feature = "gpu")]
        if let Some(result) =
            crate::gpu_dispatch::try_unary_f32(self, crate::gpu_dispatch::UnaryOp::Tanh)
        {
            return Ok(result);
        }

        self.map(|x| x.tanh())
    }

    /// Power function (element-wise)
    pub fn pow(&self, exponent: T) -> Result<Self>
    where
        T: TensorElement + Into<f32>,
    {
        // Convert T to f32 for the Operation::Power storage
        let exponent_f32: f32 = exponent.into();

        let mut result = self.map(|x| x.powf(exponent))?;

        // Set up gradient computation if needed
        if crate::should_record_grad(self.requires_grad) {
            result.requires_grad = true;
            result.operation = Operation::Power {
                input: Arc::new(self.clone()),
                exponent: exponent_f32,
            };
        }

        Ok(result)
    }

    /// Power function with scalar exponent (alias for pow)
    pub fn pow_scalar(&self, exponent: T) -> Result<Self>
    where
        T: TensorElement + Into<f32>,
    {
        self.pow(exponent)
    }

    /// Power function with tensor exponents
    pub fn pow_tensor(&self, exponent: &Self) -> Result<Self> {
        self.elementwise_operation(exponent, |base, exp| base.powf(exp))
    }

    /// Floor of all elements
    ///
    /// a.e. zero derivative: the result is detached.
    pub fn floor(&self) -> Result<Self> {
        self.map(|x| x.floor())
    }

    /// Ceiling of all elements
    ///
    /// a.e. zero derivative: the result is detached.
    pub fn ceil(&self) -> Result<Self> {
        self.map(|x| x.ceil())
    }

    /// Round to nearest integer
    ///
    /// a.e. zero derivative: the result is detached.
    pub fn round(&self) -> Result<Self> {
        self.map(|x| x.round())
    }

    /// Truncate to integer part
    ///
    /// a.e. zero derivative: the result is detached.
    pub fn trunc(&self) -> Result<Self> {
        self.map(|x| x.trunc())
    }

    /// Fractional part
    ///
    /// a.e. zero derivative (`fract` is `x - trunc(x)`, and the recorded
    /// pass-through would be wrong at every integer): the result is detached.
    pub fn fract(&self) -> Result<Self> {
        self.map(|x| x.fract())
    }

    /// Negation of all elements
    ///
    /// Records [`Operation::MulScalar`] with `-1`: `map` is forward-only, so
    /// without this the negated tensor would be a `requires_grad` leaf that
    /// swallows the gradient of everything upstream of it.
    pub fn neg(&self) -> Result<Self>
    where
        T: std::ops::Neg<Output = T>,
    {
        let mut result = self.map(|x| -x)?;
        if crate::should_record_grad(self.requires_grad) {
            result.requires_grad = true;
            result.operation = Operation::MulScalar {
                input: Arc::new(self.clone()),
                scalar: -<T as TensorElement>::one(),
            };
        }
        Ok(result)
    }

    /// Sign of all elements (-1, 0, or 1)
    ///
    /// a.e. zero derivative: the result is detached.
    pub fn sign(&self) -> Result<Self> {
        self.map(|x| {
            if x > <T as scirs2_core::numeric::Zero>::zero() {
                <T as scirs2_core::numeric::One>::one()
            } else if x < <T as scirs2_core::numeric::Zero>::zero() {
                -<T as scirs2_core::numeric::One>::one()
            } else {
                <T as scirs2_core::numeric::Zero>::zero()
            }
        })
    }
}
