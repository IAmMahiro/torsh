//! Convolution operations for all backends
//!
//! This module provides a unified interface for convolution operations across all backends,
//! with optimized implementations for each platform including direct convolution,
//! Winograd algorithm, FFT-based convolution, and im2col-based approaches.

use crate::{BackendResult, Buffer, Device};
use torsh_core::dtype::DType;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

/// Convolution operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvolutionType {
    /// 1D convolution
    Conv1D,
    /// 2D convolution (most common)
    Conv2D,
    /// 3D convolution
    Conv3D,
    /// Depthwise convolution
    DepthwiseConv2D,
    /// Separable convolution
    SeparableConv2D,
    /// Transposed convolution (deconvolution)
    ConvTranspose2D,
    /// Dilated convolution
    DilatedConv2D,
    /// Grouped convolution
    GroupedConv2D,
}

/// Convolution algorithm implementation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvolutionAlgorithm {
    /// Auto-select best algorithm
    Auto,
    /// Direct convolution implementation
    Direct,
    /// Im2col + GEMM approach
    Im2col,
    /// Winograd algorithm for small kernels
    Winograd,
    /// FFT-based convolution for large kernels
    FftBased,
    /// Optimized backend-specific implementation
    Optimized,
}

/// Padding mode for convolution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingMode {
    /// No padding
    Valid,
    /// Zero padding to maintain output size
    Same,
    /// Custom padding
    Custom,
}

/// Convolution configuration
#[derive(Debug, Clone)]
pub struct ConvolutionConfig {
    /// Convolution type
    pub conv_type: ConvolutionType,
    /// Input dimensions [batch, channels, height, width] for 2D
    pub input_dims: Vec<usize>,
    /// Output dimensions [batch, channels, height, width] for 2D
    pub output_dims: Vec<usize>,
    /// Kernel dimensions [out_channels, in_channels, height, width] for 2D
    pub kernel_dims: Vec<usize>,
    /// Stride in each dimension
    pub strides: Vec<usize>,
    /// Padding in each dimension
    pub padding: Vec<usize>,
    /// Dilation in each dimension
    pub dilation: Vec<usize>,
    /// Number of groups for grouped convolution
    pub groups: usize,
    /// Padding mode
    pub padding_mode: PaddingMode,
    /// Data type
    pub dtype: DType,
    /// Preferred algorithm
    pub algorithm: ConvolutionAlgorithm,
}

impl ConvolutionConfig {
    /// Create a new 2D convolution configuration
    pub fn conv2d(
        batch_size: usize,
        in_channels: usize,
        out_channels: usize,
        input_size: (usize, usize),
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        let (in_h, in_w) = input_size;
        let (k_h, k_w) = kernel_size;
        let (s_h, s_w) = stride;
        let (p_h, p_w) = padding;

        // Calculate output dimensions
        let out_h = (in_h + 2 * p_h - k_h) / s_h + 1;
        let out_w = (in_w + 2 * p_w - k_w) / s_w + 1;

        Self {
            conv_type: ConvolutionType::Conv2D,
            input_dims: vec![batch_size, in_channels, in_h, in_w],
            output_dims: vec![batch_size, out_channels, out_h, out_w],
            kernel_dims: vec![out_channels, in_channels, k_h, k_w],
            strides: vec![s_h, s_w],
            padding: vec![p_h, p_w],
            dilation: vec![1, 1],
            groups: 1,
            padding_mode: PaddingMode::Custom,
            dtype: DType::F32,
            algorithm: ConvolutionAlgorithm::Auto,
        }
    }

    /// Create a depthwise convolution configuration
    pub fn depthwise_conv2d(
        batch_size: usize,
        channels: usize,
        input_size: (usize, usize),
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        let mut config = Self::conv2d(
            batch_size,
            channels,
            channels,
            input_size,
            kernel_size,
            stride,
            padding,
        );
        config.conv_type = ConvolutionType::DepthwiseConv2D;
        config.groups = channels;
        config.kernel_dims = vec![channels, 1, kernel_size.0, kernel_size.1];
        config
    }

    /// Set the algorithm preference
    pub fn with_algorithm(mut self, algorithm: ConvolutionAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set the data type
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Set dilation
    pub fn with_dilation(mut self, dilation: Vec<usize>) -> Self {
        self.dilation = dilation;
        self
    }

    /// Calculate total input elements
    pub fn input_elements(&self) -> usize {
        self.input_dims.iter().product()
    }

    /// Calculate total output elements
    pub fn output_elements(&self) -> usize {
        self.output_dims.iter().product()
    }

    /// Calculate total kernel elements
    pub fn kernel_elements(&self) -> usize {
        self.kernel_dims.iter().product()
    }

    /// Get input buffer size in bytes
    pub fn input_buffer_size(&self) -> usize {
        let element_size = match self.dtype {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::F16 => 2,
            _ => 4,
        };
        self.input_elements() * element_size
    }

    /// Get output buffer size in bytes
    pub fn output_buffer_size(&self) -> usize {
        let element_size = match self.dtype {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::F16 => 2,
            _ => 4,
        };
        self.output_elements() * element_size
    }

    /// Get kernel buffer size in bytes
    pub fn kernel_buffer_size(&self) -> usize {
        let element_size = match self.dtype {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::F16 => 2,
            _ => 4,
        };
        self.kernel_elements() * element_size
    }

    /// Check if the configuration is valid
    pub fn is_valid(&self) -> bool {
        !self.input_dims.is_empty()
            && !self.output_dims.is_empty()
            && !self.kernel_dims.is_empty()
            && self.input_dims.iter().all(|&d| d > 0)
            && self.output_dims.iter().all(|&d| d > 0)
            && self.kernel_dims.iter().all(|&d| d > 0)
            && self.groups > 0
    }
}

/// Convolution operations trait
#[async_trait::async_trait]
pub trait ConvolutionOps: Send + Sync {
    /// Execute a convolution operation
    async fn convolution(
        &self,
        device: &Device,
        input: &Buffer,
        kernel: &Buffer,
        bias: Option<&Buffer>,
        output: &Buffer,
        config: &ConvolutionConfig,
    ) -> BackendResult<()>;

    /// Execute a 2D convolution.
    ///
    /// Shapes are given explicitly in NCHW layout — buffer byte-lengths alone
    /// cannot recover a tensor's shape, so callers must supply them:
    /// `input_shape = [N, C_in, H, W]`, `kernel_shape = [C_out, C_in, kH, kW]`.
    #[allow(clippy::too_many_arguments)]
    async fn conv2d(
        &self,
        device: &Device,
        input: &Buffer,
        kernel: &Buffer,
        bias: Option<&Buffer>,
        output: &Buffer,
        input_shape: [usize; 4],
        kernel_shape: [usize; 4],
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> BackendResult<()>;

    /// Execute a depthwise convolution.
    ///
    /// `input_shape = [N, C, H, W]`, `kernel_shape = [C_out, 1, kH, kW]` where
    /// `C_out = C * channel_multiplier` (groups are implicitly `C`).
    #[allow(clippy::too_many_arguments)]
    async fn depthwise_conv2d(
        &self,
        device: &Device,
        input: &Buffer,
        kernel: &Buffer,
        bias: Option<&Buffer>,
        output: &Buffer,
        input_shape: [usize; 4],
        kernel_shape: [usize; 4],
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> BackendResult<()>;

    /// Execute a transposed convolution
    async fn conv_transpose2d(
        &self,
        device: &Device,
        input: &Buffer,
        kernel: &Buffer,
        bias: Option<&Buffer>,
        output: &Buffer,
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
    ) -> BackendResult<()>;

    /// Execute a grouped convolution.
    ///
    /// `input_shape = [N, C_in, H, W]`, `kernel_shape = [C_out, C_in/groups, kH, kW]`.
    #[allow(clippy::too_many_arguments)]
    async fn grouped_conv2d(
        &self,
        device: &Device,
        input: &Buffer,
        kernel: &Buffer,
        bias: Option<&Buffer>,
        output: &Buffer,
        input_shape: [usize; 4],
        kernel_shape: [usize; 4],
        groups: usize,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> BackendResult<()>;

    /// Get the best algorithm for given configuration
    fn select_algorithm(&self, config: &ConvolutionConfig) -> ConvolutionAlgorithm;

    /// Check if convolution operations are supported
    fn supports_convolution(&self) -> bool;

    /// Get supported convolution types
    fn supported_conv_types(&self) -> Vec<ConvolutionType>;

    /// Get supported algorithms
    fn supported_algorithms(&self) -> Vec<ConvolutionAlgorithm>;
}

/// Performance characteristics for algorithm selection
#[derive(Debug, Clone)]
pub struct ConvolutionPerformanceHints {
    /// Optimal algorithm for small kernels (3x3, 5x5)
    pub small_kernel_algorithm: ConvolutionAlgorithm,
    /// Optimal algorithm for large kernels (7x7, 9x9+)
    pub large_kernel_algorithm: ConvolutionAlgorithm,
    /// Threshold for switching to FFT-based convolution
    pub fft_threshold: usize,
    /// Threshold for using Winograd algorithm
    pub winograd_threshold: usize,
    /// Preferred tile size for tiled algorithms
    pub tile_size: (usize, usize),
    /// Memory bandwidth in GB/s
    pub memory_bandwidth: f32,
    /// Compute throughput in GOPS
    pub compute_throughput: f32,
}

impl Default for ConvolutionPerformanceHints {
    fn default() -> Self {
        Self {
            small_kernel_algorithm: ConvolutionAlgorithm::Winograd,
            large_kernel_algorithm: ConvolutionAlgorithm::FftBased,
            fft_threshold: 7,
            winograd_threshold: 6,
            tile_size: (16, 16),
            memory_bandwidth: 50.0,
            compute_throughput: 100.0,
        }
    }
}

/// Default convolution operations implementation
pub struct DefaultConvolutionOps {
    performance_hints: ConvolutionPerformanceHints,
}

impl DefaultConvolutionOps {
    pub fn new() -> Self {
        Self {
            performance_hints: ConvolutionPerformanceHints::default(),
        }
    }

    pub fn with_performance_hints(mut self, hints: ConvolutionPerformanceHints) -> Self {
        self.performance_hints = hints;
        self
    }
}

#[async_trait::async_trait]
impl ConvolutionOps for DefaultConvolutionOps {
    async fn convolution(
        &self,
        _device: &Device,
        _input: &Buffer,
        _kernel: &Buffer,
        _bias: Option<&Buffer>,
        _output: &Buffer,
        _config: &ConvolutionConfig,
    ) -> BackendResult<()> {
        Err(torsh_core::error::TorshError::BackendError(
            "Convolution operations not implemented for this backend".to_string(),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    async fn conv2d(
        &self,
        _device: &Device,
        _input: &Buffer,
        _kernel: &Buffer,
        _bias: Option<&Buffer>,
        _output: &Buffer,
        _input_shape: [usize; 4],
        _kernel_shape: [usize; 4],
        _stride: (usize, usize),
        _padding: (usize, usize),
        _dilation: (usize, usize),
    ) -> BackendResult<()> {
        Err(torsh_core::error::TorshError::BackendError(
            "Conv2D operations not implemented for this backend".to_string(),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    async fn depthwise_conv2d(
        &self,
        _device: &Device,
        _input: &Buffer,
        _kernel: &Buffer,
        _bias: Option<&Buffer>,
        _output: &Buffer,
        _input_shape: [usize; 4],
        _kernel_shape: [usize; 4],
        _stride: (usize, usize),
        _padding: (usize, usize),
        _dilation: (usize, usize),
    ) -> BackendResult<()> {
        Err(torsh_core::error::TorshError::BackendError(
            "Depthwise convolution not implemented for this backend".to_string(),
        ))
    }

    async fn conv_transpose2d(
        &self,
        _device: &Device,
        _input: &Buffer,
        _kernel: &Buffer,
        _bias: Option<&Buffer>,
        _output: &Buffer,
        _stride: (usize, usize),
        _padding: (usize, usize),
        _output_padding: (usize, usize),
    ) -> BackendResult<()> {
        Err(torsh_core::error::TorshError::BackendError(
            "Transposed convolution not implemented for this backend".to_string(),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    async fn grouped_conv2d(
        &self,
        _device: &Device,
        _input: &Buffer,
        _kernel: &Buffer,
        _bias: Option<&Buffer>,
        _output: &Buffer,
        _input_shape: [usize; 4],
        _kernel_shape: [usize; 4],
        _groups: usize,
        _stride: (usize, usize),
        _padding: (usize, usize),
        _dilation: (usize, usize),
    ) -> BackendResult<()> {
        Err(torsh_core::error::TorshError::BackendError(
            "Grouped convolution not implemented for this backend".to_string(),
        ))
    }

    fn select_algorithm(&self, config: &ConvolutionConfig) -> ConvolutionAlgorithm {
        if config.algorithm != ConvolutionAlgorithm::Auto {
            return config.algorithm;
        }

        // Auto-select based on kernel size and configuration
        match config.conv_type {
            ConvolutionType::Conv2D => {
                if config.kernel_dims.len() >= 4 {
                    let kernel_h = config.kernel_dims[2];
                    let kernel_w = config.kernel_dims[3];
                    let kernel_size = kernel_h.max(kernel_w);

                    if kernel_size <= self.performance_hints.winograd_threshold {
                        ConvolutionAlgorithm::Winograd
                    } else if kernel_size >= self.performance_hints.fft_threshold {
                        ConvolutionAlgorithm::FftBased
                    } else {
                        ConvolutionAlgorithm::Im2col
                    }
                } else {
                    ConvolutionAlgorithm::Direct
                }
            }
            ConvolutionType::DepthwiseConv2D => ConvolutionAlgorithm::Direct,
            ConvolutionType::SeparableConv2D => ConvolutionAlgorithm::Direct,
            _ => ConvolutionAlgorithm::Im2col,
        }
    }

    fn supports_convolution(&self) -> bool {
        false
    }

    fn supported_conv_types(&self) -> Vec<ConvolutionType> {
        vec![]
    }

    fn supported_algorithms(&self) -> Vec<ConvolutionAlgorithm> {
        vec![ConvolutionAlgorithm::Direct]
    }
}

impl Default for DefaultConvolutionOps {
    fn default() -> Self {
        Self::new()
    }
}

/// Convolution algorithm implementations
pub mod algorithms {
    use super::*;

    /// Direct convolution implementation
    pub struct DirectConvolution;

    impl DirectConvolution {
        /// Perform 2D convolution using a direct approach.
        ///
        /// Supports dilation and grouped convolution (which also subsumes
        /// depthwise convolution, expressed as `groups == in_channels`).
        ///
        /// Layout conventions (NCHW): `input_dims = [N, C_in, H, W]`,
        /// `kernel_dims = [C_out, C_in / groups, kH, kW]`,
        /// `output_dims = [N, C_out, out_H, out_W]`. The output spatial size
        /// must satisfy `out = (in + 2*pad - dilation*(k-1) - 1) / stride + 1`.
        #[allow(clippy::too_many_arguments)]
        pub fn conv2d_direct(
            input: &[f32],
            kernel: &[f32],
            output: &mut [f32],
            input_dims: &[usize],
            kernel_dims: &[usize],
            output_dims: &[usize],
            stride: (usize, usize),
            padding: (usize, usize),
            dilation: (usize, usize),
            groups: usize,
        ) -> BackendResult<()> {
            let (batch, in_channels, in_h, in_w) =
                (input_dims[0], input_dims[1], input_dims[2], input_dims[3]);
            let (out_channels, kernel_in_per_group, k_h, k_w) = (
                kernel_dims[0],
                kernel_dims[1],
                kernel_dims[2],
                kernel_dims[3],
            );
            let (_, _, out_h, out_w) = (
                output_dims[0],
                output_dims[1],
                output_dims[2],
                output_dims[3],
            );
            let (s_h, s_w) = stride;
            let (p_h, p_w) = padding;
            let (d_h, d_w) = dilation;

            if groups == 0 || in_channels % groups != 0 || out_channels % groups != 0 {
                return Err(torsh_core::error::TorshError::BackendError(format!(
                    "grouped convolution requires groups ({}) to divide both in_channels ({}) and out_channels ({})",
                    groups, in_channels, out_channels
                )));
            }
            if d_h == 0 || d_w == 0 || s_h == 0 || s_w == 0 {
                return Err(torsh_core::error::TorshError::BackendError(
                    "convolution stride and dilation must be non-zero".to_string(),
                ));
            }

            let in_per_group = in_channels / groups;
            let out_per_group = out_channels / groups;

            if kernel_in_per_group != in_per_group {
                return Err(torsh_core::error::TorshError::BackendError(format!(
                    "kernel in-channels per group ({}) does not match in_channels/groups ({})",
                    kernel_in_per_group, in_per_group
                )));
            }

            // Validate slice lengths up front so the indexed inner loop can never
            // go out of bounds (a bounds violation must be a returned error, not a
            // panic from deep inside the kernel).
            if input.len() < batch * in_channels * in_h * in_w {
                return Err(torsh_core::error::TorshError::BackendError(format!(
                    "input buffer ({} elems) too small for [{}, {}, {}, {}]",
                    input.len(),
                    batch,
                    in_channels,
                    in_h,
                    in_w
                )));
            }
            if kernel.len() < out_channels * in_per_group * k_h * k_w {
                return Err(torsh_core::error::TorshError::BackendError(format!(
                    "kernel buffer ({} elems) too small for [{}, {}, {}, {}]",
                    kernel.len(),
                    out_channels,
                    in_per_group,
                    k_h,
                    k_w
                )));
            }
            if output.len() < batch * out_channels * out_h * out_w {
                return Err(torsh_core::error::TorshError::BackendError(format!(
                    "output buffer ({} elems) too small for [{}, {}, {}, {}]",
                    output.len(),
                    batch,
                    out_channels,
                    out_h,
                    out_w
                )));
            }

            for b in 0..batch {
                for oc in 0..out_channels {
                    // Each output channel only sees the input channels of its group.
                    let group = oc / out_per_group;
                    let ic_base = group * in_per_group;

                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let mut sum = 0.0;

                            for icg in 0..in_per_group {
                                let ic = ic_base + icg;
                                for kh in 0..k_h {
                                    for kw in 0..k_w {
                                        // Dilated tap position in the padded input.
                                        let ih = oh * s_h + kh * d_h;
                                        let iw = ow * s_w + kw * d_w;

                                        if ih >= p_h
                                            && iw >= p_w
                                            && ih < in_h + p_h
                                            && iw < in_w + p_w
                                        {
                                            let input_h = ih - p_h;
                                            let input_w = iw - p_w;

                                            if input_h < in_h && input_w < in_w {
                                                let input_idx = b * in_channels * in_h * in_w
                                                    + ic * in_h * in_w
                                                    + input_h * in_w
                                                    + input_w;
                                                // Kernel is indexed by the in-group channel
                                                // offset, not the absolute input channel.
                                                let kernel_idx = oc * in_per_group * k_h * k_w
                                                    + icg * k_h * k_w
                                                    + kh * k_w
                                                    + kw;

                                                sum += input[input_idx] * kernel[kernel_idx];
                                            }
                                        }
                                    }
                                }
                            }

                            let output_idx = b * out_channels * out_h * out_w
                                + oc * out_h * out_w
                                + oh * out_w
                                + ow;
                            output[output_idx] = sum;
                        }
                    }
                }
            }

            Ok(())
        }
    }

    /// Im2col convolution implementation
    pub struct Im2colConvolution;

    impl Im2colConvolution {
        /// Convert input to column matrix for GEMM-based convolution
        pub fn im2col(
            input: &[f32],
            output: &mut [f32],
            input_dims: &[usize],
            kernel_size: (usize, usize),
            stride: (usize, usize),
            padding: (usize, usize),
        ) -> BackendResult<()> {
            let (batch, channels, height, width) =
                (input_dims[0], input_dims[1], input_dims[2], input_dims[3]);
            let (k_h, k_w) = kernel_size;
            let (s_h, s_w) = stride;
            let (p_h, p_w) = padding;

            let out_h = (height + 2 * p_h - k_h) / s_h + 1;
            let out_w = (width + 2 * p_w - k_w) / s_w + 1;

            for b in 0..batch {
                for c in 0..channels {
                    for kh in 0..k_h {
                        for kw in 0..k_w {
                            for oh in 0..out_h {
                                for ow in 0..out_w {
                                    let ih = oh * s_h + kh;
                                    let iw = ow * s_w + kw;

                                    let value = if ih >= p_h
                                        && iw >= p_w
                                        && ih < height + p_h
                                        && iw < width + p_w
                                    {
                                        let input_h = ih - p_h;
                                        let input_w = iw - p_w;

                                        if input_h < height && input_w < width {
                                            let input_idx = b * channels * height * width
                                                + c * height * width
                                                + input_h * width
                                                + input_w;
                                            input[input_idx]
                                        } else {
                                            0.0
                                        }
                                    } else {
                                        0.0
                                    };

                                    let col_idx =
                                        (b * channels * k_h * k_w + c * k_h * k_w + kh * k_w + kw)
                                            * out_h
                                            * out_w
                                            + oh * out_w
                                            + ow;

                                    if col_idx < output.len() {
                                        output[col_idx] = value;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            Ok(())
        }
    }

    /// Winograd convolution implementation
    pub struct WinogradConvolution;

    impl WinogradConvolution {
        /// Check if Winograd can be applied
        pub fn can_apply(kernel_size: (usize, usize), stride: (usize, usize)) -> bool {
            let (k_h, k_w) = kernel_size;
            let (s_h, s_w) = stride;

            // Winograd is most effective for 3x3 kernels with stride 1
            k_h == 3 && k_w == 3 && s_h == 1 && s_w == 1
        }

        /// Perform Winograd convolution (simplified F(2,3) implementation)
        pub fn conv2d_winograd(
            input: &[f32],
            kernel: &[f32],
            output: &mut [f32],
            input_dims: &[usize],
            kernel_dims: &[usize],
            output_dims: &[usize],
        ) -> BackendResult<()> {
            // For now, fall back to direct convolution
            // A full Winograd implementation would involve complex matrix transformations
            DirectConvolution::conv2d_direct(
                input,
                kernel,
                output,
                input_dims,
                kernel_dims,
                output_dims,
                (1, 1),
                (1, 1),
                (1, 1),
                1,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convolution_config_creation() {
        let config = ConvolutionConfig::conv2d(1, 3, 16, (32, 32), (3, 3), (1, 1), (1, 1));

        assert_eq!(config.conv_type, ConvolutionType::Conv2D);
        assert_eq!(config.input_dims, vec![1, 3, 32, 32]);
        assert_eq!(config.output_dims, vec![1, 16, 32, 32]);
        assert_eq!(config.kernel_dims, vec![16, 3, 3, 3]);
        assert!(config.is_valid());
    }

    #[test]
    fn test_depthwise_config_creation() {
        let config = ConvolutionConfig::depthwise_conv2d(1, 16, (32, 32), (3, 3), (1, 1), (1, 1));

        assert_eq!(config.conv_type, ConvolutionType::DepthwiseConv2D);
        assert_eq!(config.groups, 16);
        assert_eq!(config.kernel_dims, vec![16, 1, 3, 3]);
        assert!(config.is_valid());
    }

    #[test]
    fn test_algorithm_selection() {
        let ops = DefaultConvolutionOps::new();

        // Small kernel should prefer Winograd
        let small_kernel_config =
            ConvolutionConfig::conv2d(1, 3, 16, (32, 32), (3, 3), (1, 1), (1, 1));
        assert_eq!(
            ops.select_algorithm(&small_kernel_config),
            ConvolutionAlgorithm::Winograd
        );

        // Large kernel should prefer FFT
        let large_kernel_config =
            ConvolutionConfig::conv2d(1, 3, 16, (32, 32), (9, 9), (1, 1), (4, 4));
        assert_eq!(
            ops.select_algorithm(&large_kernel_config),
            ConvolutionAlgorithm::FftBased
        );
    }

    #[test]
    fn test_buffer_size_calculations() {
        let config = ConvolutionConfig::conv2d(2, 3, 16, (32, 32), (3, 3), (1, 1), (1, 1));

        assert_eq!(config.input_elements(), 2 * 3 * 32 * 32);
        assert_eq!(config.output_elements(), 2 * 16 * 32 * 32);
        assert_eq!(config.kernel_elements(), 16 * 3 * 3 * 3);

        assert_eq!(config.input_buffer_size(), 2 * 3 * 32 * 32 * 4); // F32 = 4 bytes
        assert_eq!(config.output_buffer_size(), 2 * 16 * 32 * 32 * 4);
        assert_eq!(config.kernel_buffer_size(), 16 * 3 * 3 * 3 * 4);
    }

    #[test]
    fn test_winograd_applicability() {
        assert!(algorithms::WinogradConvolution::can_apply((3, 3), (1, 1)));
        assert!(!algorithms::WinogradConvolution::can_apply((5, 5), (1, 1)));
        assert!(!algorithms::WinogradConvolution::can_apply((3, 3), (2, 2)));
    }
}
