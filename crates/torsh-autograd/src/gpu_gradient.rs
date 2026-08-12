//! Activation forward/backward kernels with a GPU-offload policy layer.
//!
//! ## What this module actually does today
//!
//! It computes activation values and activation gradients **on the CPU**, and
//! carries the configuration and accounting that a GPU offload path would need
//! (backend selection, a minimum size below which offloading is not worth it,
//! and [`GpuStats`]).
//!
//! No GPU backend is reachable from this crate: `torsh-autograd`'s `gpu` feature
//! enables no device dependency, so [`GpuBackend::is_available`] is `false` for
//! every backend and [`GpuGradientComputer::should_use_gpu`] never selects the
//! device path. The working device path in this workspace is
//! `torsh_tensor::gpu_dispatch`, which dispatches elementwise work through
//! oxicuda.
//!
//! Consequently [`GpuStats::kernel_launches`] and
//! [`GpuStats::memory_transferred`] stay at zero here — they are only ever
//! incremented by real kernel launches and real host/device copies, so a
//! profile taken through this API cannot overstate GPU utilisation.
//!
//! ## Supported activations
//!
//! GELU (tanh approximation), LeakyReLU, Swish/SiLU, ReLU, Tanh and Sigmoid,
//! forward via [`GpuGradientComputer::gpu_activation`] and backward via
//! [`GpuGradientComputer::gpu_activation_backward`].
//!
//! ## Usage
//!
//! ```rust
//! use torsh_autograd::gpu_gradient::{ActivationType, GpuBackend, GpuGradientComputer};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let mut computer = GpuGradientComputer::new(GpuBackend::Auto)?;
//!
//! let input = [-1.0f32, 0.0, 2.0];
//! let activated = computer.gpu_activation(&input, ActivationType::ReLU)?;
//! assert_eq!(activated, vec![0.0, 0.0, 2.0]);
//!
//! let upstream = [1.0f32, 1.0, 1.0];
//! let grad = computer.gpu_activation_backward(&input, &upstream, ActivationType::ReLU)?;
//! assert_eq!(grad, vec![0.0, 0.0, 1.0]);
//! # Ok(())
//! # }
//! ```

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::error_handling::AutogradResult;

/// Negative-side slope of LeakyReLU, matching the framework default.
const LEAKY_RELU_SLOPE: f64 = 0.01;

/// Cubic coefficient of the tanh GELU approximation, matching `Tensor::gelu`.
const GELU_CUBIC_COEFFICIENT: f64 = 0.044715;

/// Supported GPU backends
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuBackend {
    /// NVIDIA CUDA backend
    CUDA,
    /// Apple Metal backend
    Metal,
    /// Cross-platform WebGPU backend
    WebGPU,
    /// AMD ROCm backend
    ROCm,
    /// OpenCL backend
    OpenCL,
    /// Automatic backend selection
    Auto,
}

impl GpuBackend {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::CUDA => "CUDA",
            Self::Metal => "Metal",
            Self::WebGPU => "WebGPU",
            Self::ROCm => "ROCm",
            Self::OpenCL => "OpenCL",
            Self::Auto => "Auto",
        }
    }

    /// Check if this backend is available on the current system
    ///
    /// Always `false`: this crate has no GPU dependency in either feature
    /// configuration (`gpu = []` enables no transport), so no backend can be
    /// reached from here. The real device path is
    /// `torsh_tensor::gpu_dispatch`, which dispatches through oxicuda. This
    /// method reports the truth rather than probing for hardware that nothing
    /// in this crate could then use.
    pub fn is_available(&self) -> bool {
        false
    }

    /// Get the best available backend for this system
    pub fn auto_select() -> Option<Self> {
        for backend in &[
            Self::CUDA,
            Self::Metal,
            Self::WebGPU,
            Self::ROCm,
            Self::OpenCL,
        ] {
            if backend.is_available() {
                return Some(*backend);
            }
        }
        None
    }
}

/// Configuration for GPU gradient computation
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// GPU backend to use
    pub backend: GpuBackend,
    /// Device index (for multi-GPU systems)
    pub device_id: usize,
    /// Minimum tensor size for GPU computation
    pub min_gpu_size: usize,
    /// Enable memory pooling
    pub memory_pooling: bool,
    /// Enable kernel fusion
    pub kernel_fusion: bool,
    /// Enable mixed precision computation
    pub mixed_precision: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            backend: GpuBackend::Auto,
            device_id: 0,
            min_gpu_size: 50000,
            memory_pooling: true,
            kernel_fusion: true,
            mixed_precision: false,
        }
    }
}

impl GpuConfig {
    /// Create a new GPU configuration
    pub fn new(backend: GpuBackend) -> Self {
        Self {
            backend,
            ..Default::default()
        }
    }

    /// Set device ID for multi-GPU systems
    pub fn with_device_id(mut self, device_id: usize) -> Self {
        self.device_id = device_id;
        self
    }

    /// Set minimum tensor size for GPU computation
    pub fn with_min_gpu_size(mut self, min_size: usize) -> Self {
        self.min_gpu_size = min_size;
        self
    }

    /// Enable or disable memory pooling
    pub fn with_memory_pooling(mut self, enabled: bool) -> Self {
        self.memory_pooling = enabled;
        self
    }

    /// Enable or disable kernel fusion
    pub fn with_kernel_fusion(mut self, enabled: bool) -> Self {
        self.kernel_fusion = enabled;
        self
    }

    /// Enable or disable mixed precision
    pub fn with_mixed_precision(mut self, enabled: bool) -> Self {
        self.mixed_precision = enabled;
        self
    }
}

/// Statistics about GPU computation
#[derive(Debug, Clone, Default)]
pub struct GpuStats {
    /// Total number of GPU operations executed
    pub total_ops: usize,
    /// Total time spent in GPU operations (ms)
    pub total_time_ms: f64,
    /// Average speedup vs CPU
    pub avg_speedup: f64,
    /// Total memory transferred to GPU (bytes)
    pub memory_transferred: usize,
    /// Number of kernel launches
    pub kernel_launches: usize,
}

/// GPU gradient computer
pub struct GpuGradientComputer {
    config: GpuConfig,
    available: bool,
    stats: GpuStats,
    #[cfg(feature = "gpu")]
    _gpu_context: Option<()>, // Placeholder for actual GPU context
}

impl GpuGradientComputer {
    /// Create a new GPU gradient computer
    pub fn new(backend: GpuBackend) -> AutogradResult<Self> {
        let config = GpuConfig::new(backend);
        Self::with_config(config)
    }

    /// Create with custom configuration
    pub fn with_config(config: GpuConfig) -> AutogradResult<Self> {
        let backend = if config.backend == GpuBackend::Auto {
            GpuBackend::auto_select().unwrap_or(GpuBackend::CUDA)
        } else {
            config.backend
        };

        let available = backend.is_available();

        if !available {
            tracing::warn!(
                "GPU backend {:?} not available, will fall back to CPU",
                backend
            );
        }

        Ok(Self {
            config,
            available,
            stats: GpuStats::default(),
            #[cfg(feature = "gpu")]
            _gpu_context: None,
        })
    }

    /// Check if GPU is available
    pub fn is_available(&self) -> bool {
        self.available
    }

    /// Get current configuration
    pub fn config(&self) -> &GpuConfig {
        &self.config
    }

    /// Get backend
    pub fn backend(&self) -> GpuBackend {
        self.config.backend
    }

    /// Get statistics
    pub fn stats(&self) -> &GpuStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = GpuStats::default();
    }

    /// Check if a tensor should be computed on GPU
    pub fn should_use_gpu(&self, tensor_size: usize) -> bool {
        self.available && tensor_size >= self.config.min_gpu_size
    }

    /// Apply an activation function elementwise.
    ///
    /// Supported: GELU (tanh approximation, matching `Tensor::gelu`), LeakyReLU
    /// (slope 0.01), Swish/SiLU, ReLU, Tanh and Sigmoid.
    ///
    /// # Execution
    ///
    /// The work runs on the CPU. No GPU transport is wired into this crate (see
    /// [`GpuBackend::is_available`]), so [`should_use_gpu`](Self::should_use_gpu)
    /// is always false and no kernel-launch or host/device-transfer statistics
    /// are recorded — the counters in [`GpuStats`] only ever move when real GPU
    /// work happens. Elapsed time and the operation count are recorded because
    /// they are measured, not assumed.
    ///
    /// # Errors
    ///
    /// Returns an error if a required floating-point constant cannot be
    /// represented in `T`.
    pub fn gpu_activation<T>(
        &mut self,
        data: &[T],
        activation: ActivationType,
    ) -> AutogradResult<Vec<T>>
    where
        T: num_traits::Float,
    {
        let start = std::time::Instant::now();
        let result = Self::activation_forward(data, activation)?;
        self.record_cpu_op(start);
        Ok(result)
    }

    /// Backward pass of [`gpu_activation`](Self::gpu_activation).
    ///
    /// Returns `grad_output * f'(input)` elementwise, where `f` is `activation`
    /// and `input` is the tensor that was fed to the forward pass. This is the
    /// quantity a gradient computer actually needs; the forward activation alone
    /// is not a gradient.
    ///
    /// # Errors
    ///
    /// Returns an error if `input` and `grad_output` have different lengths, or
    /// if a required floating-point constant cannot be represented in `T`.
    pub fn gpu_activation_backward<T>(
        &mut self,
        input: &[T],
        grad_output: &[T],
        activation: ActivationType,
    ) -> AutogradResult<Vec<T>>
    where
        T: num_traits::Float,
    {
        if input.len() != grad_output.len() {
            return Err(crate::error_handling::AutogradError::shape_mismatch(
                "gpu_activation_backward",
                vec![input.len()],
                vec![grad_output.len()],
            ));
        }

        let start = std::time::Instant::now();
        let derivative = Self::activation_derivative(input, activation)?;
        let result = derivative
            .into_iter()
            .zip(grad_output.iter())
            .map(|(d, &g)| d * g)
            .collect();
        self.record_cpu_op(start);
        Ok(result)
    }

    /// Record one honestly-measured CPU operation.
    ///
    /// Deliberately leaves `kernel_launches` and `memory_transferred` alone: a
    /// profile that reports launches and transfers that never happened is worse
    /// than no profile at all.
    fn record_cpu_op(&mut self, start: std::time::Instant) {
        self.stats.total_ops += 1;
        self.stats.total_time_ms += start.elapsed().as_secs_f64() * 1000.0;
    }

    /// Convert an `f64` constant into `T`, or report which constant failed.
    fn constant<T: num_traits::Float>(value: f64, what: &str) -> AutogradResult<T> {
        T::from(value).ok_or_else(|| {
            crate::error_handling::AutogradError::gradient_computation(
                "activation",
                format!("cannot represent {what} ({value}) in the element type"),
            )
        })
    }

    /// Elementwise activation values.
    fn activation_forward<T: num_traits::Float>(
        data: &[T],
        activation: ActivationType,
    ) -> AutogradResult<Vec<T>> {
        let zero = T::zero();
        let one = T::one();
        let half: T = Self::constant(0.5, "0.5")?;
        let leaky_slope: T = Self::constant(LEAKY_RELU_SLOPE, "the LeakyReLU slope")?;
        let gelu_coefficient: T = Self::constant(GELU_CUBIC_COEFFICIENT, "the GELU coefficient")?;
        let sqrt_2_over_pi: T = Self::constant((2.0 / std::f64::consts::PI).sqrt(), "sqrt(2/pi)")?;

        Ok(data
            .iter()
            .map(|&x| match activation {
                ActivationType::ReLU => {
                    if x > zero {
                        x
                    } else {
                        zero
                    }
                }
                ActivationType::LeakyReLU => {
                    if x > zero {
                        x
                    } else {
                        leaky_slope * x
                    }
                }
                ActivationType::Tanh => x.tanh(),
                ActivationType::Sigmoid => one / (one + (-x).exp()),
                ActivationType::Swish => x * (one / (one + (-x).exp())),
                ActivationType::GELU => {
                    let inner = sqrt_2_over_pi * (x + gelu_coefficient * x * x * x);
                    half * x * (one + inner.tanh())
                }
            })
            .collect())
    }

    /// Elementwise derivative of the activation with respect to its input.
    fn activation_derivative<T: num_traits::Float>(
        data: &[T],
        activation: ActivationType,
    ) -> AutogradResult<Vec<T>> {
        let zero = T::zero();
        let one = T::one();
        let three: T = Self::constant(3.0, "3.0")?;
        let half: T = Self::constant(0.5, "0.5")?;
        let leaky_slope: T = Self::constant(LEAKY_RELU_SLOPE, "the LeakyReLU slope")?;
        let gelu_coefficient: T = Self::constant(GELU_CUBIC_COEFFICIENT, "the GELU coefficient")?;
        let sqrt_2_over_pi: T = Self::constant((2.0 / std::f64::consts::PI).sqrt(), "sqrt(2/pi)")?;

        Ok(data
            .iter()
            .map(|&x| match activation {
                // Sub-gradient 0 is the usual convention at the kink.
                ActivationType::ReLU => {
                    if x > zero {
                        one
                    } else {
                        zero
                    }
                }
                ActivationType::LeakyReLU => {
                    if x > zero {
                        one
                    } else {
                        leaky_slope
                    }
                }
                // d/dx tanh(x) = 1 - tanh(x)^2
                ActivationType::Tanh => {
                    let t = x.tanh();
                    one - t * t
                }
                // d/dx sigma(x) = sigma(x) (1 - sigma(x))
                ActivationType::Sigmoid => {
                    let s = one / (one + (-x).exp());
                    s * (one - s)
                }
                // d/dx (x sigma(x)) = sigma(x) (1 + x (1 - sigma(x)))
                ActivationType::Swish => {
                    let s = one / (one + (-x).exp());
                    s * (one + x * (one - s))
                }
                // With u(x) = sqrt(2/pi) (x + c x^3):
                //   d/dx [0.5 x (1 + tanh u)] = 0.5 (1 + tanh u) + 0.5 x (1 - tanh^2 u) u'
                ActivationType::GELU => {
                    let inner = sqrt_2_over_pi * (x + gelu_coefficient * x * x * x);
                    let tanh_inner = inner.tanh();
                    let inner_derivative =
                        sqrt_2_over_pi * (one + three * gelu_coefficient * x * x);
                    half * (one + tanh_inner)
                        + half * x * (one - tanh_inner * tanh_inner) * inner_derivative
                }
            })
            .collect())
    }

    /// Report current performance statistics
    pub fn report_performance(&self) -> String {
        format!(
            "GPU Gradient Computation Statistics:\n\
             - Backend: {}\n\
             - Available: {}\n\
             - Total operations: {}\n\
             - Total time: {:.2}ms\n\
             - Kernel launches: {}\n\
             - Memory transferred: {:.2} MB\n\
             - Average speedup: {:.2}x\n\
             - Average time per op: {:.2}ms",
            self.config.backend.name(),
            self.available,
            self.stats.total_ops,
            self.stats.total_time_ms,
            self.stats.kernel_launches,
            self.stats.memory_transferred as f64 / (1024.0 * 1024.0),
            self.stats.avg_speedup,
            if self.stats.total_ops > 0 {
                self.stats.total_time_ms / self.stats.total_ops as f64
            } else {
                0.0
            }
        )
    }
}

/// Activation function types supported on GPU
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationType {
    GELU,
    LeakyReLU,
    Swish,
    ReLU,
    Tanh,
    Sigmoid,
}

/// Global GPU gradient computer instance
static GLOBAL_GPU_COMPUTER: once_cell::sync::Lazy<
    parking_lot::RwLock<Option<GpuGradientComputer>>,
> = once_cell::sync::Lazy::new(|| parking_lot::RwLock::new(None));

/// Get the global GPU gradient computer
pub fn get_global_gpu_computer(
) -> parking_lot::RwLockReadGuard<'static, Option<GpuGradientComputer>> {
    GLOBAL_GPU_COMPUTER.read()
}

/// Initialize the global GPU gradient computer
pub fn initialize_global_gpu(backend: GpuBackend) -> AutogradResult<()> {
    let mut computer_lock = GLOBAL_GPU_COMPUTER.write();
    *computer_lock = Some(GpuGradientComputer::new(backend)?);
    Ok(())
}

/// Check if global GPU computer is initialized and available
pub fn is_global_gpu_available() -> bool {
    GLOBAL_GPU_COMPUTER
        .read()
        .as_ref()
        .map_or(false, |c| c.is_available())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_backend_names() {
        assert_eq!(GpuBackend::CUDA.name(), "CUDA");
        assert_eq!(GpuBackend::Metal.name(), "Metal");
        assert_eq!(GpuBackend::WebGPU.name(), "WebGPU");
    }

    #[test]
    fn test_gpu_config() {
        let config = GpuConfig::new(GpuBackend::CUDA)
            .with_device_id(1)
            .with_min_gpu_size(100000)
            .with_memory_pooling(false);

        assert_eq!(config.backend, GpuBackend::CUDA);
        assert_eq!(config.device_id, 1);
        assert_eq!(config.min_gpu_size, 100000);
        assert!(!config.memory_pooling);
    }

    #[test]
    fn test_gpu_computer_creation() {
        // This will create with CPU fallback since GPU is not available in tests
        let result = GpuGradientComputer::new(GpuBackend::CUDA);
        assert!(result.is_ok());

        let computer = result.unwrap();
        assert_eq!(computer.backend(), GpuBackend::CUDA);
    }

    #[test]
    fn test_should_use_gpu() {
        let computer = GpuGradientComputer::new(GpuBackend::CUDA).unwrap();

        // Small tensors should not use GPU
        assert!(!computer.should_use_gpu(1000));

        // Large tensors would use GPU if available
        // (but GPU is not available in tests, so this returns false)
        assert!(!computer.should_use_gpu(100000));
    }

    #[test]
    fn test_report_performance() {
        let computer = GpuGradientComputer::new(GpuBackend::CUDA).unwrap();
        let report = computer.report_performance();

        assert!(report.contains("GPU Gradient Computation Statistics"));
        assert!(report.contains("Backend: CUDA"));
    }
}
