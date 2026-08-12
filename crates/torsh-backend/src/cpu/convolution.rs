//! CPU convolution implementation with optimized algorithms

use crate::convolution::{
    algorithms, ConvolutionAlgorithm, ConvolutionOps, ConvolutionPerformanceHints, ConvolutionType,
    PaddingMode,
};

// Re-export for benchmarks
pub use crate::convolution::ConvolutionConfig;
use crate::cpu::buffer::BufferCpuExt;
use crate::{BackendResult, Buffer, Device};

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

/// CPU convolution operations implementation
#[derive(Clone, Debug)]
pub struct CpuConvolutionOps {
    /// Performance hints for algorithm selection
    performance_hints: ConvolutionPerformanceHints,
    /// Number of threads for parallel processing
    #[allow(dead_code)]
    num_threads: usize,
}

impl CpuConvolutionOps {
    /// Create a new CPU convolution operations instance
    pub fn new(num_threads: Option<usize>) -> Self {
        let num_threads = num_threads.unwrap_or_else(|| rayon::current_num_threads());

        Self {
            performance_hints: ConvolutionPerformanceHints {
                small_kernel_algorithm: ConvolutionAlgorithm::Direct,
                large_kernel_algorithm: ConvolutionAlgorithm::Im2col,
                fft_threshold: 7,
                winograd_threshold: 6,
                tile_size: (16, 16),
                memory_bandwidth: 100.0, // CPU memory bandwidth
                compute_throughput: num_threads as f32 * 50.0, // Estimated GOPS
            },
            num_threads,
        }
    }

    /// Copy buffer data safely for CPU
    #[allow(dead_code)]
    fn copy_buffer_data(&self, src: &Buffer, dst: &Buffer, size: usize) -> BackendResult<()> {
        if !src.is_cpu() || !dst.is_cpu() {
            return Err(torsh_core::error::TorshError::BackendError(
                "Both buffers must be CPU buffers".to_string(),
            ));
        }

        let src_ptr = src.as_cpu_ptr().ok_or_else(|| {
            torsh_core::error::TorshError::BackendError(
                "Failed to get source buffer pointer".to_string(),
            )
        })?;

        let dst_ptr = dst.as_cpu_ptr().ok_or_else(|| {
            torsh_core::error::TorshError::BackendError(
                "Failed to get destination buffer pointer".to_string(),
            )
        })?;

        if size > src.size.min(dst.size) {
            return Err(torsh_core::error::TorshError::BackendError(format!(
                "Copy size {} exceeds buffer capacity",
                size
            )));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, size);
        }

        Ok(())
    }

    /// Execute direct convolution on CPU
    fn direct_convolution(
        &self,
        input: &Buffer,
        kernel: &Buffer,
        bias: Option<&Buffer>,
        output: &Buffer,
        config: &ConvolutionConfig,
    ) -> BackendResult<()> {
        // Get buffer pointers
        let input_ptr = input.as_cpu_ptr().ok_or_else(|| {
            torsh_core::error::TorshError::BackendError(
                "Failed to get input buffer pointer".to_string(),
            )
        })?;

        let kernel_ptr = kernel.as_cpu_ptr().ok_or_else(|| {
            torsh_core::error::TorshError::BackendError(
                "Failed to get kernel buffer pointer".to_string(),
            )
        })?;

        let output_ptr = output.as_cpu_ptr().ok_or_else(|| {
            torsh_core::error::TorshError::BackendError(
                "Failed to get output buffer pointer".to_string(),
            )
        })?;

        unsafe {
            let input_data = std::slice::from_raw_parts(input_ptr as *const f32, input.size / 4);
            let kernel_data = std::slice::from_raw_parts(kernel_ptr as *const f32, kernel.size / 4);
            let output_data =
                std::slice::from_raw_parts_mut(output_ptr as *mut f32, output.size / 4);

            match config.conv_type {
                // Plain, dilated, grouped and depthwise 2D convolution are all
                // handled by the one generalized direct kernel: depthwise is
                // simply `groups == in_channels`, dilation and groups are read
                // straight from the config so nothing is silently ignored.
                ConvolutionType::Conv2D
                | ConvolutionType::DilatedConv2D
                | ConvolutionType::GroupedConv2D
                | ConvolutionType::DepthwiseConv2D => {
                    algorithms::DirectConvolution::conv2d_direct(
                        input_data,
                        kernel_data,
                        output_data,
                        &config.input_dims,
                        &config.kernel_dims,
                        &config.output_dims,
                        (config.strides[0], config.strides[1]),
                        (config.padding[0], config.padding[1]),
                        (config.dilation[0], config.dilation[1]),
                        config.groups,
                    )?;
                }
                _ => {
                    return Err(torsh_core::error::TorshError::BackendError(format!(
                        "Convolution type {:?} not implemented yet",
                        config.conv_type
                    )));
                }
            }

            // Add bias if provided
            if let Some(bias_buffer) = bias {
                let bias_ptr = bias_buffer.as_cpu_ptr().ok_or_else(|| {
                    torsh_core::error::TorshError::BackendError(
                        "Failed to get bias buffer pointer".to_string(),
                    )
                })?;
                let bias_data =
                    std::slice::from_raw_parts(bias_ptr as *const f32, bias_buffer.size / 4);

                self.add_bias(output_data, bias_data, &config.output_dims)?;
            }
        }

        Ok(())
    }

    /// Add bias to output
    fn add_bias(
        &self,
        output: &mut [f32],
        bias: &[f32],
        output_dims: &[usize],
    ) -> BackendResult<()> {
        if output_dims.len() < 4 {
            return Ok(());
        }

        let (batch, channels, height, width) = (
            output_dims[0],
            output_dims[1],
            output_dims[2],
            output_dims[3],
        );

        for b in 0..batch {
            for c in 0..channels {
                let bias_value = bias.get(c).copied().unwrap_or(0.0);
                for h in 0..height {
                    for w in 0..width {
                        let idx =
                            b * channels * height * width + c * height * width + h * width + w;
                        if idx < output.len() {
                            output[idx] += bias_value;
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Compute the NCHW output spatial size for a dilated convolution and assemble
/// a [`ConvolutionConfig`] from caller-supplied shapes.
///
/// This replaces the previous placeholder-dimension approach: buffer byte
/// lengths cannot recover a tensor shape, so the shapes are passed explicitly
/// and the output size is derived from them together with the stride, padding
/// and dilation actually requested.
fn build_conv_config(
    conv_type: ConvolutionType,
    input_shape: [usize; 4],
    kernel_shape: [usize; 4],
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
) -> BackendResult<ConvolutionConfig> {
    let [n, c_in, in_h, in_w] = input_shape;
    let [c_out, _k_in_per_group, k_h, k_w] = kernel_shape;
    let (s_h, s_w) = stride;
    let (p_h, p_w) = padding;
    let (d_h, d_w) = dilation;

    if s_h == 0 || s_w == 0 || d_h == 0 || d_w == 0 {
        return Err(torsh_core::error::TorshError::BackendError(
            "convolution stride and dilation must be non-zero".to_string(),
        ));
    }

    // Effective (dilated) kernel extent.
    let eff_h = d_h * (k_h - 1) + 1;
    let eff_w = d_w * (k_w - 1) + 1;
    if in_h + 2 * p_h < eff_h || in_w + 2 * p_w < eff_w {
        return Err(torsh_core::error::TorshError::BackendError(format!(
            "convolution kernel ({}x{} dilated to {}x{}) does not fit padded input ({}x{})",
            k_h,
            k_w,
            eff_h,
            eff_w,
            in_h + 2 * p_h,
            in_w + 2 * p_w
        )));
    }

    let out_h = (in_h + 2 * p_h - eff_h) / s_h + 1;
    let out_w = (in_w + 2 * p_w - eff_w) / s_w + 1;

    Ok(ConvolutionConfig {
        conv_type,
        input_dims: vec![n, c_in, in_h, in_w],
        output_dims: vec![n, c_out, out_h, out_w],
        kernel_dims: vec![kernel_shape[0], kernel_shape[1], k_h, k_w],
        strides: vec![s_h, s_w],
        padding: vec![p_h, p_w],
        dilation: vec![d_h, d_w],
        groups,
        padding_mode: PaddingMode::Custom,
        dtype: torsh_core::dtype::DType::F32,
        algorithm: ConvolutionAlgorithm::Direct,
    })
}

#[async_trait::async_trait]
impl ConvolutionOps for CpuConvolutionOps {
    async fn convolution(
        &self,
        _device: &Device,
        input: &Buffer,
        kernel: &Buffer,
        bias: Option<&Buffer>,
        output: &Buffer,
        config: &ConvolutionConfig,
    ) -> BackendResult<()> {
        if !config.is_valid() {
            return Err(torsh_core::error::TorshError::BackendError(
                "Invalid convolution configuration".to_string(),
            ));
        }

        let algorithm = self.select_algorithm(config);

        match algorithm {
            ConvolutionAlgorithm::Direct => {
                self.direct_convolution(input, kernel, bias, output, config)
            }
            ConvolutionAlgorithm::Im2col => {
                // For now, fall back to direct convolution
                // A full im2col implementation would require GEMM operations
                self.direct_convolution(input, kernel, bias, output, config)
            }
            ConvolutionAlgorithm::Winograd => {
                // For now, fall back to direct convolution
                // A full Winograd implementation would require specialized transforms
                self.direct_convolution(input, kernel, bias, output, config)
            }
            ConvolutionAlgorithm::FftBased => {
                // For now, fall back to direct convolution
                // FFT-based convolution would use our FFT operations module
                self.direct_convolution(input, kernel, bias, output, config)
            }
            _ => self.direct_convolution(input, kernel, bias, output, config),
        }
    }

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
    ) -> BackendResult<()> {
        let config = build_conv_config(
            ConvolutionType::Conv2D,
            input_shape,
            kernel_shape,
            stride,
            padding,
            dilation,
            1,
        )?;

        self.convolution(device, input, kernel, bias, output, &config)
            .await
    }

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
    ) -> BackendResult<()> {
        // Depthwise convolution is grouped convolution with one group per input
        // channel; the kernel is laid out as [C_out, 1, kH, kW].
        let groups = input_shape[1];
        let config = build_conv_config(
            ConvolutionType::DepthwiseConv2D,
            input_shape,
            kernel_shape,
            stride,
            padding,
            dilation,
            groups,
        )?;

        self.convolution(device, input, kernel, bias, output, &config)
            .await
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
            "Transposed convolution not implemented for CPU backend yet".to_string(),
        ))
    }

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
    ) -> BackendResult<()> {
        let config = build_conv_config(
            ConvolutionType::GroupedConv2D,
            input_shape,
            kernel_shape,
            stride,
            padding,
            dilation,
            groups,
        )?;

        self.convolution(device, input, kernel, bias, output, &config)
            .await
    }

    fn select_algorithm(&self, config: &ConvolutionConfig) -> ConvolutionAlgorithm {
        if config.algorithm != ConvolutionAlgorithm::Auto {
            return config.algorithm;
        }

        // Auto-select based on configuration and performance hints
        match config.conv_type {
            ConvolutionType::Conv2D => {
                if config.kernel_dims.len() >= 4 {
                    let kernel_h = config.kernel_dims[2];
                    let kernel_w = config.kernel_dims[3];
                    let kernel_size = kernel_h.max(kernel_w);

                    if kernel_size <= 3 {
                        // Small kernels work well with direct convolution on CPU
                        ConvolutionAlgorithm::Direct
                    } else if kernel_size <= self.performance_hints.winograd_threshold {
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
            ConvolutionType::GroupedConv2D => ConvolutionAlgorithm::Direct,
            _ => ConvolutionAlgorithm::Im2col,
        }
    }

    fn supports_convolution(&self) -> bool {
        true
    }

    fn supported_conv_types(&self) -> Vec<ConvolutionType> {
        vec![
            ConvolutionType::Conv1D,
            ConvolutionType::Conv2D,
            ConvolutionType::Conv3D,
            ConvolutionType::DepthwiseConv2D,
            ConvolutionType::SeparableConv2D,
            ConvolutionType::GroupedConv2D,
            // ConvolutionType::ConvTranspose2D, // Not implemented yet
            ConvolutionType::DilatedConv2D,
        ]
    }

    fn supported_algorithms(&self) -> Vec<ConvolutionAlgorithm> {
        vec![
            ConvolutionAlgorithm::Auto,
            ConvolutionAlgorithm::Direct,
            ConvolutionAlgorithm::Im2col,
            ConvolutionAlgorithm::Winograd,
            ConvolutionAlgorithm::FftBased,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convolution::ConvolutionConfig;

    #[test]
    fn test_cpu_convolution_ops_creation() {
        let conv_ops = CpuConvolutionOps::new(Some(2));
        assert!(conv_ops.supports_convolution());
        assert!(!conv_ops.supported_conv_types().is_empty());
        assert!(!conv_ops.supported_algorithms().is_empty());
    }

    #[test]
    fn test_algorithm_selection() {
        let conv_ops = CpuConvolutionOps::new(Some(1));

        // Small kernel should use direct convolution on CPU
        let small_config = ConvolutionConfig::conv2d(1, 3, 16, (32, 32), (3, 3), (1, 1), (1, 1));
        assert_eq!(
            conv_ops.select_algorithm(&small_config),
            ConvolutionAlgorithm::Direct
        );

        // Large kernel should use FFT-based convolution
        let large_config = ConvolutionConfig::conv2d(1, 3, 16, (32, 32), (9, 9), (1, 1), (4, 4));
        assert_eq!(
            conv_ops.select_algorithm(&large_config),
            ConvolutionAlgorithm::FftBased
        );

        // Depthwise should always use direct
        let depthwise_config =
            ConvolutionConfig::depthwise_conv2d(1, 16, (32, 32), (3, 3), (1, 1), (1, 1));
        assert_eq!(
            conv_ops.select_algorithm(&depthwise_config),
            ConvolutionAlgorithm::Direct
        );
    }
}
