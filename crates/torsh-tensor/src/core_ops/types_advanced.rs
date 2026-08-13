/// Advanced operations for Tensor: comparisons, shape manipulation, logical ops, and Operation enum.
///
/// This module is included by `types.rs` via `#[path = "types_advanced.rs"]`.
use super::*;

/// Comparison operations for tensors
impl<T: TensorElement + PartialOrd + Copy> Tensor<T> {
    /// Element-wise greater than comparison
    pub fn gt(&self, other: &Self) -> Result<Tensor<bool>> {
        if self.shape != other.shape {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: other.shape.dims().to_vec(),
            });
        }
        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;
        let result_data: Vec<bool> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| a > b)
            .collect();
        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }
    /// Element-wise less than comparison
    pub fn lt(&self, other: &Self) -> Result<Tensor<bool>> {
        if self.shape != other.shape {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: other.shape.dims().to_vec(),
            });
        }
        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;
        let result_data: Vec<bool> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| a < b)
            .collect();
        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }
    /// Element-wise greater than or equal comparison
    pub fn ge(&self, other: &Self) -> Result<Tensor<bool>> {
        if self.shape != other.shape {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: other.shape.dims().to_vec(),
            });
        }
        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;
        let result_data: Vec<bool> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| a >= b)
            .collect();
        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }
    /// Element-wise less than or equal comparison
    pub fn le(&self, other: &Self) -> Result<Tensor<bool>> {
        if self.shape != other.shape {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: other.shape.dims().to_vec(),
            });
        }
        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;
        let result_data: Vec<bool> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| a <= b)
            .collect();
        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }
    /// Element-wise equality comparison
    pub fn eq(&self, other: &Self) -> Result<Tensor<bool>> {
        if self.shape != other.shape {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: other.shape.dims().to_vec(),
            });
        }
        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;
        let result_data: Vec<bool> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| a == b)
            .collect();
        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }
    /// Element-wise inequality comparison
    pub fn ne(&self, other: &Self) -> Result<Tensor<bool>> {
        if self.shape != other.shape {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: other.shape.dims().to_vec(),
            });
        }
        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;
        let result_data: Vec<bool> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| a != b)
            .collect();
        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }
    /// Scalar comparison methods
    /// Element-wise equality comparison with scalar
    pub fn eq_scalar(&self, value: T) -> Result<Tensor<bool>>
    where
        T: PartialEq + Copy,
    {
        let self_data = self.to_vec()?;
        let result_data: Vec<bool> = self_data.iter().map(|&a| a == value).collect();
        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }
    /// Element-wise inequality comparison with scalar
    pub fn ne_scalar(&self, value: T) -> Result<Tensor<bool>>
    where
        T: PartialEq + Copy,
    {
        let self_data = self.to_vec()?;
        let result_data: Vec<bool> = self_data.iter().map(|&a| a != value).collect();
        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }
    /// Element-wise greater than comparison with scalar
    pub fn gt_scalar(&self, value: T) -> Result<Tensor<bool>>
    where
        T: PartialOrd + Copy,
    {
        let self_data = self.to_vec()?;
        let result_data: Vec<bool> = self_data.iter().map(|&a| a > value).collect();
        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }
    /// Element-wise less than comparison with scalar
    pub fn lt_scalar(&self, value: T) -> Result<Tensor<bool>>
    where
        T: PartialOrd + Copy,
    {
        let self_data = self.to_vec()?;
        let result_data: Vec<bool> = self_data.iter().map(|&a| a < value).collect();
        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }
    /// Element-wise less than or equal comparison with scalar
    pub fn le_scalar(&self, value: T) -> Result<Tensor<bool>>
    where
        T: PartialOrd + Copy,
    {
        let self_data = self.to_vec()?;
        let result_data: Vec<bool> = self_data.iter().map(|&a| a <= value).collect();
        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }
    /// Element-wise greater than or equal comparison with scalar
    pub fn ge_scalar(&self, value: T) -> Result<Tensor<bool>>
    where
        T: PartialOrd + Copy,
    {
        let self_data = self.to_vec()?;
        let result_data: Vec<bool> = self_data.iter().map(|&a| a >= value).collect();
        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }
}
/// Shape manipulation operations for tensors
impl<T: TensorElement> Tensor<T> {
    /// Flatten tensor to 1D
    pub fn flatten(&self) -> Result<Self> {
        let total_elements = self.numel();
        self.view(&[total_elements as i32])
    }
    /// Conditional tensor selection - where condition is true, select from self, otherwise from other
    pub fn where_tensor(&self, condition: &Tensor<bool>, other: &Self) -> Result<Self> {
        if self.shape != condition.shape || self.shape != other.shape {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: condition.shape.dims().to_vec(),
            });
        }
        let self_data = self.to_vec()?;
        let condition_data = condition.to_vec()?;
        let other_data = other.to_vec()?;
        let result_data: Vec<T> = self_data
            .iter()
            .zip(condition_data.iter())
            .zip(other_data.iter())
            .map(
                |((&self_val, &cond), &other_val)| {
                    if cond {
                        self_val
                    } else {
                        other_val
                    }
                },
            )
            .collect();
        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }
    /// Add bias vector to tensor (element-wise addition)
    pub fn add_bias(&self, bias: &Self) -> Result<Self>
    where
        T: std::ops::Add<Output = T>,
    {
        self.add(bias)
    }
}
/// Logical operations for boolean tensors
impl Tensor<bool> {
    /// Element-wise logical AND operation
    pub fn logical_and(&self, other: &Self) -> Result<Self> {
        if self.shape != other.shape {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: other.shape.dims().to_vec(),
            });
        }
        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;
        let result_data: Vec<bool> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| a && b)
            .collect();
        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }
    /// Element-wise logical OR operation
    pub fn logical_or(&self, other: &Self) -> Result<Self> {
        if self.shape != other.shape {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: other.shape.dims().to_vec(),
            });
        }
        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;
        let result_data: Vec<bool> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| a || b)
            .collect();
        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }
    /// Element-wise logical XOR operation
    pub fn logical_xor(&self, other: &Self) -> Result<Self> {
        if self.shape != other.shape {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: other.shape.dims().to_vec(),
            });
        }
        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;
        let result_data: Vec<bool> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| a ^ b)
            .collect();
        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }
}
/// Operation type for gradient computation
#[derive(Debug, Clone)]
pub enum Operation<T: TensorElement> {
    /// Leaf node (no operation)
    Leaf,
    /// Power operation: x^n
    Power {
        input: Arc<Tensor<T>>,
        exponent: f32,
    },
    /// Addition operation: a + b
    Add {
        lhs: Arc<Tensor<T>>,
        rhs: Arc<Tensor<T>>,
    },
    /// Subtraction operation: a - b
    Sub {
        lhs: Arc<Tensor<T>>,
        rhs: Arc<Tensor<T>>,
    },
    /// Multiplication operation: a * b
    Mul {
        lhs: Arc<Tensor<T>>,
        rhs: Arc<Tensor<T>>,
    },
    /// Division operation: a / b
    Div {
        /// Numerator
        lhs: Arc<Tensor<T>>,
        /// Denominator
        rhs: Arc<Tensor<T>>,
    },
    /// Multiplication by a constant: `input * scalar`.
    ///
    /// Recorded separately from [`Operation::Mul`] so a scalar multiply does not
    /// have to materialise a whole tensor of copies of the constant; the
    /// backward rule is simply `grad * scalar`.
    MulScalar {
        /// The tensor that was scaled
        input: Arc<Tensor<T>>,
        /// The constant factor
        scalar: T,
    },
    /// Addition of a constant: `input + scalar`.
    ///
    /// Backward is the identity, so a scalar shift never has to materialise a
    /// tensor of copies of the constant. [`Tensor::sub_scalar`] records this
    /// variant with the negated constant.
    AddScalar {
        /// The tensor that was shifted
        input: Arc<Tensor<T>>,
        /// The constant that was added
        scalar: T,
    },
    /// Division by a constant: `input / scalar`. Backward is `grad / scalar`.
    DivScalar {
        /// The tensor that was divided
        input: Arc<Tensor<T>>,
        /// The constant divisor
        scalar: T,
    },
    /// Mean reduction operation: mean(input)
    Mean {
        /// The input tensor before reduction
        input: Arc<Tensor<T>>,
        /// Number of elements that were averaged
        count: f64,
    },
    /// Sum reduction operation: sum(input) over all elements
    Sum {
        /// The input tensor before reduction
        input: Arc<Tensor<T>>,
    },
    /// Dimension-wise sum reduction ([`Tensor::sum_dim`]).
    ///
    /// `dims` is stored **normalised** — negative indices resolved, sorted and
    /// deduplicated exactly as the forward pass reduced them — because the
    /// backward rule rebuilds the keepdim-shaped layout from it. `keepdim`
    /// records whether those axes survived with extent `1`. The input shape is
    /// read back from `input`: a `Tensor`'s `shape` field is owned (not shared),
    /// so the recorded clone can never drift from the shape that was reduced.
    ///
    /// Backward re-inserts the reduced axes with extent `1` and broadcasts the
    /// gradient over them: every input element fed exactly one output slot, so
    /// it receives that slot's gradient unchanged (`d/dx_i sum = 1`).
    SumDim {
        /// The tensor that was reduced
        input: Arc<Tensor<T>>,
        /// Reduced axes: normalised, sorted, deduplicated
        dims: Vec<usize>,
        /// Whether the reduced axes were kept with extent `1`
        keepdim: bool,
    },
    /// Matrix multiplication: lhs @ rhs (2-D, row-major)
    MatMul {
        /// Left operand
        lhs: Arc<Tensor<T>>,
        /// Right operand
        rhs: Arc<Tensor<T>>,
    },
    /// Custom operation with name and inputs
    Custom(String, Vec<Weak<Tensor<T>>>),
    /// Shape-only view of `input` (reshape, axis permutation or broadcast).
    ///
    /// The forward pass moves no data (or, for a non-contiguous source, copies
    /// it unchanged), so the backward pass only has to map the gradient back
    /// onto the input's layout — see `kind`.
    View {
        /// The tensor the view was taken of
        input: Arc<Tensor<T>>,
        /// How the view re-indexes its input
        kind: ViewKind,
    },
    /// im2col (unfold) of an `[N, C, H, W]` tensor into per-group patch
    /// matrices, the gather that turns a convolution into a matrix product.
    ///
    /// The forward pass copies each sliding window into one row; the backward
    /// pass is col2im, which scatter-adds every patch element back onto the
    /// input position it was read from (overlapping windows accumulate).
    Im2Col {
        /// The tensor that was unfolded
        input: Arc<Tensor<T>>,
        /// Geometry needed to invert the gather
        config: Im2ColConfig,
    },
    /// Concatenation of several tensors along `dim` ([`Tensor::cat`]).
    ///
    /// The backward pass narrows the upstream gradient back to each input's
    /// extent along `dim` (each input's grad is a contiguous slab of the seed).
    Concat {
        /// The tensors that were concatenated, in order
        inputs: Vec<Arc<Tensor<T>>>,
        /// Axis the inputs were joined along
        dim: usize,
    },
    /// Stacking of several equally-shaped tensors along a *new* axis `dim`
    /// ([`Tensor::stack`]).
    ///
    /// The backward pass selects each input's slice of the upstream gradient at
    /// its index along `dim`, dropping the inserted axis.
    Stack {
        /// The tensors that were stacked, in order
        inputs: Vec<Arc<Tensor<T>>>,
        /// The newly-inserted axis
        dim: usize,
    },
    /// A gather (basic/advanced indexing, `narrow`, `select`, `slice_tensor`).
    ///
    /// `index_map[o]` is the *logical* flat index of `input` that supplied
    /// output element `o`. The backward pass scatter-adds each output gradient
    /// back onto that input position (duplicated reads accumulate), which is the
    /// zeros-everywhere-except-the-slice rule for contiguous slices.
    Gather {
        /// The tensor that was indexed
        input: Arc<Tensor<T>>,
        /// Output-position → input logical flat index
        index_map: Arc<Vec<usize>>,
    },
    /// `log_softmax(input, dim)` ([`Tensor::log_softmax`]).
    ///
    /// The backward rule is the exact, numerically stable Jacobian-vector
    /// product `g - softmax(input) * sum(g, dim, keepdim)`; it needs no gradient
    /// through the max shift.
    LogSoftmax {
        /// The tensor the log-softmax was taken of
        input: Arc<Tensor<T>>,
        /// Axis the normalisation ran over
        dim: usize,
    },
    /// A differentiable element-wise unary function (`exp`, `ln`, `sqrt`,
    /// `rsqrt`, `reciprocal`, `log10`, `log2`, `sin`, `cos`, `tan`, `asin`,
    /// `acos`, `atan`, `sinh`, `cosh`, `tanh`, `sigmoid`, `relu`, `gelu`).
    ///
    /// The forward value may have been produced by any dispatch path (SIMD,
    /// parallel, GPU or scalar); the derivative is recomputed from `input` in
    /// the backward pass, so every path shares one correct rule.
    Unary {
        /// The operand
        input: Arc<Tensor<T>>,
        /// Which function was applied
        kind: UnaryKind,
    },
    /// `leaky_relu(input, negative_slope)` ([`Tensor::leaky_relu`]).
    ///
    /// Recorded as its own variant rather than a [`UnaryKind`] because it
    /// carries a per-call scalar; the slope is stored as `T` — lossless and
    /// bound-free — exactly like [`Operation::MulScalar`]'s `scalar`.
    ///
    /// Sub-gradient: `1` where `input > 0`, `negative_slope` elsewhere. The
    /// backward uses the same `>` predicate as the forward, so the two agree
    /// at `x == 0` (PyTorch's convention: the slope branch).
    LeakyRelu {
        /// The operand
        input: Arc<Tensor<T>>,
        /// Slope applied to the non-positive half
        negative_slope: T,
    },
    /// `maximum(lhs, rhs)` ([`Tensor::maximum`]), element-wise and
    /// broadcasting.
    ///
    /// Backward routes each element's gradient to whichever operand won, using
    /// the *same* `lhs > rhs` predicate the forward uses:
    ///
    /// * `lhs > rhs` — all of the gradient goes to `lhs`;
    /// * `lhs == rhs` (an exact tie) — `0.5 * grad` to each, PyTorch's
    ///   convention for `torch.maximum`;
    /// * otherwise (including any `NaN` operand, for which both comparisons are
    ///   false) — all of the gradient goes to `rhs`, mirroring the forward,
    ///   which returns `rhs` in exactly that case.
    ///
    /// The *un-broadcast* operands are recorded, so each contribution is folded
    /// back to its own shape with `reduce_grad_to_shape` — the same discipline
    /// as [`Operation::Add`] and [`Operation::Mul`].
    Maximum {
        /// Left operand
        lhs: Arc<Tensor<T>>,
        /// Right operand
        rhs: Arc<Tensor<T>>,
    },
    /// `minimum(lhs, rhs)` ([`Tensor::minimum`]), the mirror of
    /// [`Operation::Maximum`].
    ///
    /// Backward uses the forward's `lhs < rhs` predicate: `lhs` wins outright
    /// when it is strictly smaller, an exact tie splits `0.5 * grad` each way,
    /// and every other case (including `NaN`) sends the whole gradient to
    /// `rhs`. Operands are recorded un-broadcast and folded back per operand.
    Minimum {
        /// Left operand
        lhs: Arc<Tensor<T>>,
        /// Right operand
        rhs: Arc<Tensor<T>>,
    },
    /// `clamp(input, min, max)` ([`Tensor::clamp`], [`Tensor::clamp_min`],
    /// [`Tensor::clamp_max`]).
    ///
    /// Named `ClampBounds` rather than `Clamp` because both bounds are optional
    /// and the variant carries them, not a clamped tensor.
    ///
    /// Backward is ATen's `clamp_backward`,
    /// `grad * (x >= min) * (x <= max)`, with an absent bound treated as
    /// always-satisfied. **Both comparisons are inclusive**: an element sitting
    /// exactly on a bound keeps its full gradient (PyTorch picks the
    /// sub-gradient `1` there, not `0` and not `0.5`), while an element that
    /// was actually moved by the clamp gets `0`.
    ///
    /// `NaN` inputs get a zero gradient, because both `x >= min` and
    /// `x <= max` are false for `NaN` — even though the forward passes `NaN`
    /// through unchanged. That asymmetry is PyTorch's too.
    ClampBounds {
        /// The operand
        input: Arc<Tensor<T>>,
        /// Lower bound, or `None` when only an upper bound was applied
        min: Option<T>,
        /// Upper bound, or `None` when only a lower bound was applied
        max: Option<T>,
    },
}

/// The differentiable element-wise unary functions recorded by
/// [`Operation::Unary`]. Each fully determines its own derivative.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryKind {
    /// `exp(x)`; derivative `exp(x)`.
    Exp,
    /// `ln(x)`; derivative `1/x`.
    Ln,
    /// `sqrt(x)`; derivative `1/(2*sqrt(x))`.
    Sqrt,
    /// `sin(x)`; derivative `cos(x)`.
    Sin,
    /// `cos(x)`; derivative `-sin(x)`.
    Cos,
    /// `tanh(x)`; derivative `1 - tanh(x)^2`.
    Tanh,
    /// `sigmoid(x)`; derivative `s*(1-s)` with `s = sigmoid(x)`.
    Sigmoid,
    /// `relu(x)`; sub-gradient `1` for `x > 0`, else `0`.
    Relu,
    /// `gelu(x)` in the **tanh approximation** computed by
    /// [`Tensor::gelu`]: `0.5*x*(1 + tanh(k*(x + c*x^3)))` with
    /// `k = sqrt(2/pi)` and `c = 0.044715`.
    ///
    /// The recorded derivative differentiates *that* closed form, not the
    /// exact erf-GELU — and every forward dispatch path now computes exactly
    /// that closed form, so the derivative matches the forward on all of them.
    ///
    /// Until Wave 4 the f32 SIMD path (`numel > 1000`) called scirs2-core's
    /// `adaptive_simd_gelu_f32`, a clamped-Pade rational standing in for
    /// `tanh`. That made the forward *discontinuous* across the `numel = 1000`
    /// dispatch threshold (measured: up to 0.0211 absolute, 515% relative at
    /// `x = -2.40`) and left the recorded derivative differentiating a
    /// function the large-tensor forward never evaluated. The dispatch was
    /// removed rather than the rule relaxed; `gelu` is now one function at
    /// every size, and `hardening_autograd_primitives.rs` pins both the
    /// cross-threshold continuity and a finite-difference gradcheck at
    /// `n = 2048`, which the Pade path could not pass.
    Gelu,
    /// `abs(x) = |x|`; sub-gradient `sign(x)`, i.e. `1` for `x > 0`, `-1` for
    /// `x < 0` and **`0` exactly at `x == 0`** (PyTorch's convention for the
    /// kink; the true sub-differential there is the whole interval
    /// `[-1, 1]`).
    ///
    /// Only *real* element types record this. `Tensor::abs` is defined for
    /// every [`torsh_core::dtype::ComplexElement`], and for a genuinely
    /// complex `T` it maps `Tensor<T> -> Tensor<T::Real>` — a different tensor
    /// type, whose derivative is the Wirtinger `z/|z|` rather than `sign(x)`.
    /// The recording hook (`Tensor::record_abs_if_real`) downcasts and simply
    /// declines when input and output types differ, so complex `abs` stays
    /// detached exactly as before.
    Abs,
    /// `rsqrt(x) = 1/sqrt(x)`; derivative `-0.5 * x^(-3/2)`, computed as
    /// `-0.5 / (x * sqrt(x))` to reuse the same `sqrt` the forward pass took
    /// (no `powf`, matching the crate's other arms). Diverges to `-inf` as
    /// `x -> 0+` and is `NaN` for `x < 0`, exactly like `rsqrt` itself.
    Rsqrt,
    /// `reciprocal(x) = 1/x`; derivative `-1/x^2`. Diverges to `-inf` at
    /// `x == 0` from either side (`x^2` is never negative).
    Reciprocal,
    /// `log10(x)`; derivative `1/(x*ln(10))`.
    Log10,
    /// `log2(x)`; derivative `1/(x*ln(2))`.
    Log2,
    /// `tan(x)`; derivative `1 + tan(x)^2` (`= sec(x)^2`). Diverges at every
    /// odd multiple of `pi/2`, where `tan(x)` itself is already `+-inf`.
    Tan,
    /// `asin(x)`; derivative `1/sqrt(1 - x^2)`. Diverges to `+inf` at the
    /// domain edges `x == +-1`; `NaN` outside `[-1, 1]`, matching `asin`
    /// itself.
    Asin,
    /// `acos(x)`; derivative `-1/sqrt(1 - x^2)`. Diverges to `-inf` at the
    /// domain edges `x == +-1`; `NaN` outside `[-1, 1]`, matching `acos`
    /// itself.
    Acos,
    /// `atan(x)`; derivative `1/(1 + x^2)`. Finite everywhere: `1 + x^2` is
    /// never zero, so `atan` is the one inverse trig function here with no
    /// domain-edge case to test.
    Atan,
    /// `sinh(x)`; derivative `cosh(x)`.
    Sinh,
    /// `cosh(x)`; derivative `sinh(x)`.
    Cosh,
}

/// Geometry of an [`Operation::Im2Col`] gather.
///
/// Everything the backward pass needs to map a patch element back onto the
/// input element it was copied from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Im2ColConfig {
    /// Input shape `[batch, channels, height, width]`
    pub input_shape: [usize; 4],
    /// Kernel extent `(kh, kw)`
    pub kernel: (usize, usize),
    /// Stride `(sh, sw)`
    pub stride: (usize, usize),
    /// Zero padding `(ph, pw)` applied to the input before gathering
    pub padding: (usize, usize),
    /// Dilation `(dh, dw)` of the kernel taps
    pub dilation: (usize, usize),
    /// Number of channel groups the patches are split into
    pub groups: usize,
    /// Output spatial extent `(out_h, out_w)`
    pub output: (usize, usize),
}

/// How a [`Operation::View`] node re-indexes the tensor it was taken of.
///
/// Each variant fully determines the backward rule, which is why the view
/// constructors can share a single `Operation` variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ViewKind {
    /// Row-major reshape: the element order is unchanged, so the gradient is
    /// simply reshaped back to the input's shape.
    Reshape,
    /// Axis permutation, where `perm[i]` is the input axis that became output
    /// axis `i`. The gradient is permuted by the inverse of `perm`.
    Permute(Vec<usize>),
    /// Broadcast expansion. The gradient is summed over the axes that were
    /// prepended or stretched from extent 1.
    Expand,
    /// Contiguous slice `[start, start + len)` along `dim`, every other axis
    /// taken whole — the geometry `narrow`, a step-1 `slice_with_step` and
    /// `slice_tensor` all build.
    ///
    /// Recorded instead of an [`Operation::Gather`] so a slice neither stores
    /// one index per output element nor scatters element-by-element on the way
    /// back: the gradient is one zero-padded slab. `len` is read back from the
    /// gradient and the input extent from the recorded `input` (a `Tensor`'s
    /// `shape` field is owned, so the clone can never drift), which is why only
    /// the two geometric parameters live here.
    Narrow {
        /// The sliced axis
        dim: usize,
        /// First index of the slice along `dim`
        start: usize,
    },
}
