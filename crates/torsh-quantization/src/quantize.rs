//! Quantization operations

use crate::{QScheme, TorshResult};
use scirs2_core::parallel_ops::*;
use torsh_core::{DType, TorshError};
use torsh_tensor::Tensor;

/// Calculate strides for tensor dimensions (optimized, reusable function)
#[inline]
fn calculate_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Integer range of a quantization dtype.
fn dtype_range(dtype: DType) -> TorshResult<(i32, i32)> {
    match dtype {
        DType::I8 => Ok((-128, 127)),
        DType::U8 => Ok((0, 255)),
        DType::I16 => Ok((-32768, 32767)),
        _ => Err(TorshError::InvalidArgument(format!(
            "Unsupported quantization dtype: {dtype:?}"
        ))),
    }
}

/// Fallback scalar quantization kernel clamped to an explicit `[qmin, qmax]` range.
#[inline]
fn quantize_scalar_f32(
    data: &[f32],
    scale: f32,
    zero_point: i32,
    qmin: i32,
    qmax: i32,
    output: &mut [f32],
) {
    let inv_scale = 1.0 / scale;
    let zero_point_f32 = zero_point as f32;
    let (lo, hi) = (qmin as f32, qmax as f32);

    for (out, &val) in output.iter_mut().zip(data.iter()) {
        *out = ((val * inv_scale).round() + zero_point_f32).clamp(lo, hi);
    }
}

/// AVX2 + FMA quantization kernel.
///
/// Ties are rounded to even (the hardware rounding mode), which may differ from
/// the scalar kernel's round-half-away-from-zero for values exactly on `.5`.
///
/// # Safety
///
/// The caller must ensure the `avx2` and `fma` CPU features are available and
/// that `output.len() == data.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn quantize_avx2_f32(
    data: &[f32],
    scale: f32,
    zero_point: i32,
    qmin: i32,
    qmax: i32,
    output: &mut [f32],
) {
    use std::arch::x86_64::*;

    let inv_scale_vec = _mm256_set1_ps(1.0 / scale);
    let zero_point_vec = _mm256_set1_ps(zero_point as f32);
    let min_vec = _mm256_set1_ps(qmin as f32);
    let max_vec = _mm256_set1_ps(qmax as f32);

    let n_chunks = data.len() / 8;
    for i in 0..n_chunks {
        let base = i * 8;
        let input = _mm256_loadu_ps(data.as_ptr().add(base));
        let scaled = _mm256_fmadd_ps(input, inv_scale_vec, zero_point_vec);
        let rounded = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(scaled);
        let clamped = _mm256_max_ps(_mm256_min_ps(rounded, max_vec), min_vec);
        _mm256_storeu_ps(output.as_mut_ptr().add(base), clamped);
    }

    // Process the tail with the scalar kernel
    let processed = n_chunks * 8;
    if processed < data.len() {
        quantize_scalar_f32(
            &data[processed..],
            scale,
            zero_point,
            qmin,
            qmax,
            &mut output[processed..],
        );
    }
}

/// AVX-512F quantization kernel processing 16 lanes per iteration.
///
/// # Safety
///
/// The caller must ensure the `avx512f` CPU feature is available and that
/// `output.len() == data.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn quantize_avx512_f32(
    data: &[f32],
    scale: f32,
    zero_point: i32,
    qmin: i32,
    qmax: i32,
    output: &mut [f32],
) {
    use std::arch::x86_64::*;

    let inv_scale_vec = _mm512_set1_ps(1.0 / scale);
    let zero_point_vec = _mm512_set1_ps(zero_point as f32);
    let min_vec = _mm512_set1_ps(qmin as f32);
    let max_vec = _mm512_set1_ps(qmax as f32);

    let n_chunks = data.len() / 16;
    for i in 0..n_chunks {
        let base = i * 16;
        let input = _mm512_loadu_ps(data.as_ptr().add(base));
        let scaled = _mm512_fmadd_ps(input, inv_scale_vec, zero_point_vec);
        // imm8 = 0x00: round to nearest (even), no scaling, suppress exceptions
        let rounded = _mm512_roundscale_ps::<0x00>(scaled);
        let clamped = _mm512_max_ps(_mm512_min_ps(rounded, max_vec), min_vec);
        _mm512_storeu_ps(output.as_mut_ptr().add(base), clamped);
    }

    // Process the tail with the scalar kernel
    let processed = n_chunks * 16;
    if processed < data.len() {
        quantize_scalar_f32(
            &data[processed..],
            scale,
            zero_point,
            qmin,
            qmax,
            &mut output[processed..],
        );
    }
}

/// Optimized quantization with runtime SIMD feature detection.
///
/// The SIMD kernels are compiled unconditionally on `x86_64` (they use
/// `#[target_feature]` rather than compile-time `#[cfg(target_feature)]` gates),
/// so they are always type-checked and are actually selected on default builds
/// whenever the running CPU supports them.
#[inline]
fn quantize_optimized(data: &[f32], scale: f32, zero_point: i32, qmin: i32, qmax: i32) -> Vec<f32> {
    let mut output = vec![0.0f32; data.len()];

    #[cfg(target_arch = "x86_64")]
    {
        if data.len() >= 16 && std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: the avx512f feature was just verified at runtime and the
            // output buffer has exactly the same length as the input.
            unsafe { quantize_avx512_f32(data, scale, zero_point, qmin, qmax, &mut output) };
            return output;
        }
        if data.len() >= 8
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: the avx2/fma features were just verified at runtime and
            // the output buffer has exactly the same length as the input.
            unsafe { quantize_avx2_f32(data, scale, zero_point, qmin, qmax, &mut output) };
            return output;
        }
    }

    quantize_scalar_f32(data, scale, zero_point, qmin, qmax, &mut output);
    output
}

/// Quantize a tensor to INT8 using per-tensor affine quantization
///
/// This is the `DType::I8` specialisation of
/// [`quantize_per_tensor_affine_dtype`].
pub fn quantize_per_tensor_affine(
    tensor: &Tensor,
    scale: f32,
    zero_point: i32,
) -> TorshResult<(Tensor, f32, i32)> {
    quantize_per_tensor_affine_dtype(tensor, scale, zero_point, DType::I8)
}

/// Quantize a tensor using per-tensor affine quantization for a target dtype
///
/// The integer codes are clamped to the dtype's own range (`[0, 255]` for `U8`,
/// `[-128, 127]` for `I8`, `[-32768, 32767]` for `I16`) and `zero_point` is
/// validated against that same range.
pub fn quantize_per_tensor_affine_dtype(
    tensor: &Tensor,
    scale: f32,
    zero_point: i32,
    dtype: DType,
) -> TorshResult<(Tensor, f32, i32)> {
    let data = tensor.data()?;

    // Validate inputs
    if scale <= 0.0 {
        return Err(TorshError::InvalidArgument(
            "Quantization scale must be positive".to_string(),
        ));
    }

    let (qmin, qmax) = dtype_range(dtype)?;
    if !(qmin..=qmax).contains(&zero_point) {
        return Err(TorshError::InvalidArgument(format!(
            "Zero point {zero_point} must be in range [{qmin}, {qmax}] for {dtype:?}"
        )));
    }

    // Use optimized quantization with SIMD when available
    let quantized_f32: Vec<f32> = if data.len() > 1000 {
        // Use parallel processing for large tensors
        data.par_chunks(4096) // Process in cache-friendly chunks
            .flat_map(|chunk| quantize_optimized(chunk, scale, zero_point, qmin, qmax))
            .collect()
    } else {
        quantize_optimized(&data, scale, zero_point, qmin, qmax)
    };

    let quantized_tensor = Tensor::from_data(
        quantized_f32,
        tensor.shape().dims().to_vec(),
        tensor.device(),
    );

    Ok((quantized_tensor?, scale, zero_point))
}

/// Quantize a tensor using symmetric quantization (INT8, zero_point = 0)
pub fn quantize_per_tensor_symmetric(tensor: &Tensor, scale: f32) -> TorshResult<(Tensor, f32)> {
    let (quantized_tensor, computed_scale, _) = quantize_per_tensor_affine(tensor, scale, 0)?;
    Ok((quantized_tensor, computed_scale))
}

/// Quantize a tensor using symmetric quantization for a target dtype
///
/// `zero_point` is the symmetric zero of `dtype` (`0` for signed types, the
/// midpoint of the range for unsigned ones), as produced by
/// [`calculate_symmetric_qparams`].
pub fn quantize_per_tensor_symmetric_dtype(
    tensor: &Tensor,
    scale: f32,
    zero_point: i32,
    dtype: DType,
) -> TorshResult<(Tensor, f32, i32)> {
    quantize_per_tensor_affine_dtype(tensor, scale, zero_point, dtype)
}

/// Calculate symmetric quantization parameters from tensor statistics
///
/// Unlike [`calculate_qparams`] (affine), the scale is derived from the maximum
/// absolute value: `scale = max_abs / (qmax - zero_point)`. Using the affine
/// scale for a symmetric quantizer clips every value above half the range —
/// for an input spanning `[-1, 3]` the tensor maximum would be reduced by a
/// third.
///
/// The zero point is `0` for signed ranges and the midpoint of the range for
/// unsigned ones (`128` for `U8`), because a `U8` quantizer with zero point `0`
/// cannot represent negative values at all.
pub fn calculate_symmetric_qparams(
    tensor: &Tensor,
    qmin: i32,
    qmax: i32,
) -> TorshResult<(f32, i32)> {
    let data = tensor.data()?;

    if data.is_empty() {
        return Err(TorshError::InvalidArgument(
            "Cannot calculate quantization parameters for empty tensor".to_string(),
        ));
    }

    if qmin >= qmax {
        return Err(TorshError::InvalidArgument(
            "qmin must be less than qmax".to_string(),
        ));
    }

    // `f32::max` ignores NaN, so a fold alone would silently accept NaN inputs
    // and emit NaN codes; check every element explicitly, as the affine
    // `calculate_qparams` does.
    if data.iter().any(|val| !val.is_finite()) {
        return Err(TorshError::InvalidArgument(
            "Tensor contains non-finite values (NaN or infinity)".to_string(),
        ));
    }

    let max_abs = data.iter().fold(0.0f32, |acc, &val| acc.max(val.abs()));

    let zero_point = if qmin >= 0 {
        (((qmin as i64) + (qmax as i64) + 1) / 2) as i32
    } else {
        0
    };

    let headroom = (qmax - zero_point).max(1) as f32;
    let scale = if max_abs < 1e-7 {
        1e-7 / headroom
    } else {
        max_abs / headroom
    };

    Ok((scale, zero_point))
}

/// Calculate quantization parameters (scale and zero_point) from tensor statistics
pub fn calculate_qparams(
    tensor: &Tensor,
    qmin: i32,
    qmax: i32,
    _dtype: DType,
) -> TorshResult<(f32, i32)> {
    let data = tensor.data()?;

    if data.is_empty() {
        return Err(TorshError::InvalidArgument(
            "Cannot calculate quantization parameters for empty tensor".to_string(),
        ));
    }

    if qmin >= qmax {
        return Err(TorshError::InvalidArgument(
            "qmin must be less than qmax".to_string(),
        ));
    }

    // Optimized min/max calculation with better numerical stability
    let (min_val, max_val) = if data.len() > 10000 {
        // Use parallel processing for very large tensors
        data.par_iter().map(|&val| (val, val)).reduce(
            || (f32::INFINITY, f32::NEG_INFINITY),
            |(min1, max1), (min2, max2)| (min1.min(min2), max1.max(max2)),
        )
    } else {
        data.iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &val| {
                (min.min(val), max.max(val))
            })
    };

    // Handle edge cases
    if !min_val.is_finite() || !max_val.is_finite() {
        return Err(TorshError::InvalidArgument(
            "Tensor contains non-finite values (NaN or infinity)".to_string(),
        ));
    }

    // Ensure the range includes zero for better numerical stability
    let min_val = min_val.min(0.0);
    let max_val = max_val.max(0.0);

    // Add small epsilon to prevent zero range
    let range = max_val - min_val;
    let adjusted_range = if range < 1e-7 {
        1e-7 // Minimum meaningful range
    } else {
        range
    };

    // Calculate scale with improved numerical stability
    let scale = adjusted_range / (qmax - qmin) as f32;

    // Use more precise zero point calculation
    let zero_point_exact = qmin as f64 - (min_val as f64) / (scale as f64);
    let zero_point = zero_point_exact.round().max(qmin as f64).min(qmax as f64) as i32;

    Ok((scale, zero_point))
}

/// Quantize using per-channel affine quantization into the INT8 range
pub fn quantize_per_channel_affine(
    tensor: &Tensor,
    scales: &[f32],
    zero_points: &[i32],
    axis: usize,
) -> TorshResult<(Tensor, Vec<f32>, Vec<i32>)> {
    quantize_per_channel_affine_dtype(tensor, scales, zero_points, axis, DType::I8)
}

/// Quantize using per-channel affine quantization for a target dtype
///
/// Codes are clamped to the dtype's range rather than always to the INT8 range.
pub fn quantize_per_channel_affine_dtype(
    tensor: &Tensor,
    scales: &[f32],
    zero_points: &[i32],
    axis: usize,
    dtype: DType,
) -> TorshResult<(Tensor, Vec<f32>, Vec<i32>)> {
    let data = tensor.data()?;
    let binding = tensor.shape();
    let shape = binding.dims();

    if axis >= shape.len() {
        return Err(TorshError::InvalidArgument(
            "Axis out of bounds".to_string(),
        ));
    }

    let channel_size = shape[axis];
    if scales.len() != channel_size || zero_points.len() != channel_size {
        return Err(TorshError::InvalidArgument(
            "Scales and zero_points length must match channel size".to_string(),
        ));
    }

    let (qmin, qmax) = dtype_range(dtype)?;
    let (qmin_f, qmax_f) = (qmin as f32, qmax as f32);

    for (channel, (&scale, &zero_point)) in scales.iter().zip(zero_points.iter()).enumerate() {
        if scale <= 0.0 {
            return Err(TorshError::InvalidArgument(format!(
                "Scale for channel {channel} must be positive"
            )));
        }
        if !(qmin..=qmax).contains(&zero_point) {
            return Err(TorshError::InvalidArgument(format!(
                "Zero point {zero_point} for channel {channel} must be in range \
                 [{qmin}, {qmax}] for {dtype:?}"
            )));
        }
    }

    // Calculate strides for the given axis (using optimized helper)
    let strides = calculate_strides(shape);

    let quantized_f32: Vec<f32> = data
        .iter()
        .enumerate()
        .map(|(idx, &x)| {
            // Calculate which channel this element belongs to
            let channel_idx = (idx / strides[axis]) % shape[axis];
            let scale = scales[channel_idx];
            let zero_point = zero_points[channel_idx];

            // Quantize: q = round(x / scale) + zero_point
            ((x / scale).round() + zero_point as f32).clamp(qmin_f, qmax_f)
        })
        .collect();

    let quantized_tensor = Tensor::from_data(quantized_f32, shape.to_vec(), tensor.device());

    Ok((quantized_tensor?, scales.to_vec(), zero_points.to_vec()))
}

/// Calculate per-channel symmetric quantization parameters
///
/// Returns `scales[c] = max_abs(channel c) / (qmax - zero_point)` and the
/// symmetric zero point of `dtype` for every channel. See
/// [`calculate_symmetric_qparams`] for why the affine scale must not be reused.
pub fn calculate_per_channel_symmetric_qparams(
    tensor: &Tensor,
    axis: usize,
    dtype: DType,
) -> TorshResult<(Vec<f32>, Vec<i32>)> {
    let data = tensor.data()?;
    let binding = tensor.shape();
    let shape = binding.dims();

    if axis >= shape.len() {
        return Err(TorshError::InvalidArgument(
            "Axis out of bounds".to_string(),
        ));
    }

    let (qmin, qmax) = dtype_range(dtype)?;
    let zero_point = if qmin >= 0 {
        (((qmin as i64) + (qmax as i64) + 1) / 2) as i32
    } else {
        0
    };
    let headroom = (qmax - zero_point).max(1) as f32;

    if data.iter().any(|val| !val.is_finite()) {
        return Err(TorshError::InvalidArgument(
            "Tensor contains non-finite values (NaN or infinity)".to_string(),
        ));
    }

    let channel_size = shape[axis];
    let strides = calculate_strides(shape);
    let mut channel_abs_max = vec![0.0f32; channel_size];

    for (idx, &val) in data.iter().enumerate() {
        let channel_idx = (idx / strides[axis]) % shape[axis];
        channel_abs_max[channel_idx] = channel_abs_max[channel_idx].max(val.abs());
    }

    let scales = channel_abs_max
        .iter()
        .map(|&max_abs| {
            if max_abs < 1e-7 {
                1e-7 / headroom
            } else {
                max_abs / headroom
            }
        })
        .collect();

    Ok((scales, vec![zero_point; channel_size]))
}

/// Calculate per-channel quantization parameters
pub fn calculate_per_channel_qparams(
    tensor: &Tensor,
    axis: usize,
    dtype: DType,
) -> TorshResult<(Vec<f32>, Vec<i32>)> {
    let data = tensor.data()?;
    let binding = tensor.shape();
    let shape = binding.dims();

    if axis >= shape.len() {
        return Err(TorshError::InvalidArgument(
            "Axis out of bounds".to_string(),
        ));
    }

    let (qmin, qmax) = dtype_range(dtype)?;

    let channel_size = shape[axis];
    let mut channel_mins = vec![f32::INFINITY; channel_size];
    let mut channel_maxs = vec![f32::NEG_INFINITY; channel_size];

    // Calculate strides for the given axis
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    // Find min/max for each channel
    for (idx, &val) in data.iter().enumerate() {
        let channel_idx = (idx / strides[axis]) % shape[axis];
        channel_mins[channel_idx] = channel_mins[channel_idx].min(val);
        channel_maxs[channel_idx] = channel_maxs[channel_idx].max(val);
    }

    let mut scales = Vec::with_capacity(channel_size);
    let mut zero_points = Vec::with_capacity(channel_size);

    for ch in 0..channel_size {
        let min_val = channel_mins[ch].min(0.0);
        let max_val = channel_maxs[ch].max(0.0);

        let scale = (max_val - min_val) / (qmax - qmin) as f32;
        let scale = if scale == 0.0 { 1.0 } else { scale };

        let zero_point = (qmin as f32 - min_val / scale)
            .round()
            .max(qmin as f32)
            .min(qmax as f32) as i32;

        scales.push(scale);
        zero_points.push(zero_point);
    }

    Ok((scales, zero_points))
}

/// Quantize tensor with automatic parameter calculation
///
/// Per-channel schemes are rejected here because the scalar `(scale,
/// zero_point)` return type cannot carry their parameters; use
/// [`quantize_per_channel_auto`], which returns the full vectors.
pub fn quantize_tensor_auto(
    tensor: &Tensor,
    dtype: DType,
    scheme: QScheme,
) -> TorshResult<(Tensor, f32, i32)> {
    let (qmin, qmax) = dtype_range(dtype)?;

    match scheme {
        QScheme::PerTensorAffine => {
            let (scale, zero_point) = calculate_qparams(tensor, qmin, qmax, dtype)?;
            quantize_per_tensor_affine_dtype(tensor, scale, zero_point, dtype)
        }
        QScheme::PerTensorSymmetric => {
            let (scale, zero_point) = calculate_symmetric_qparams(tensor, qmin, qmax)?;
            quantize_per_tensor_symmetric_dtype(tensor, scale, zero_point, dtype)
        }
        QScheme::PerChannelAffine | QScheme::PerChannelSymmetric => {
            Err(TorshError::InvalidArgument(format!(
                "{scheme:?} produces one scale/zero-point pair per channel which \
                 cannot be returned through this scalar API; call \
                 quantize_per_channel_auto instead"
            )))
        }
        QScheme::Int4PerTensor => {
            // Use the quantize_int4_per_tensor function from lib.rs
            crate::quantize_int4_per_tensor(tensor, &crate::QuantConfig::int4())
        }
        QScheme::Int4PerChannel => {
            // Use the quantize_int4_per_channel function from lib.rs
            let axis = 0;
            crate::quantize_int4_per_channel(tensor, axis, &crate::QuantConfig::int4())
        }
        QScheme::Binary => {
            // Use the quantize_binary function from lib.rs
            crate::quantize_binary(tensor)
        }
        QScheme::Ternary => {
            // Use the quantize_ternary function from lib.rs
            crate::quantize_ternary(tensor)
        }
        QScheme::GroupWise => {
            // Use the quantize_group_wise function from lib.rs with default parameters
            let axis = 0;
            let group_size = 32;
            crate::quantize_group_wise(
                tensor,
                axis,
                group_size,
                &crate::QuantConfig::group_wise(axis, group_size),
            )
        }
        QScheme::MixedPrecision => {
            // Mixed precision requires different handling
            Err(TorshError::InvalidArgument(
                "Mixed precision quantization requires specialized API".to_string(),
            ))
        }
    }
}

/// Quantize tensor with per-channel scheme
pub fn quantize_per_channel_auto(
    tensor: &Tensor,
    axis: usize,
    dtype: DType,
    scheme: QScheme,
) -> TorshResult<(Tensor, Vec<f32>, Vec<i32>)> {
    match scheme {
        QScheme::PerChannelAffine => {
            let (scales, zero_points) = calculate_per_channel_qparams(tensor, axis, dtype)?;
            quantize_per_channel_affine_dtype(tensor, &scales, &zero_points, axis, dtype)
        }
        QScheme::PerChannelSymmetric => {
            let (scales, zero_points) =
                calculate_per_channel_symmetric_qparams(tensor, axis, dtype)?;
            quantize_per_channel_affine_dtype(tensor, &scales, &zero_points, axis, dtype)
        }
        _ => Err(TorshError::InvalidArgument(
            "Scheme not supported for per-channel quantization".to_string(),
        )),
    }
}

/// Convenience function for automatic quantization based on configuration
/// This function takes a tensor and a QuantConfig and returns a quantized tensor
/// along with the scale and zero point parameters
pub fn quantize_auto(
    tensor: &Tensor,
    config: &crate::QuantConfig,
) -> TorshResult<(Tensor, f32, i32)> {
    quantize_tensor_auto(tensor, config.dtype, config.scheme)
}

/// Dynamic quantization for modules
// Temporarily disabled: pub fn quantize_dynamic(_module: &mut dyn torsh_nn::Module) -> TorshResult<()> {
#[allow(dead_code)]
pub fn quantize_dynamic(module: &mut dyn crate::qat::Module) -> TorshResult<()> {
    // Iterate through module parameters and quantize them dynamically
    let mut quantized_params = Vec::new();

    // Get mutable parameters
    let mut_params = module.parameters_mut();

    for param in mut_params {
        // Use INT8 quantization for each parameter
        let config = crate::QuantConfig::int8();
        let (quantized, _scale, _zero_point) = quantize_auto(param, &config)?;

        // Store quantized parameter (in practice, this would update the module's parameters)
        quantized_params.push(quantized);
    }

    // Note: In a real implementation, we would update the module's parameters
    // with the quantized versions, but this requires more complex module handling

    Ok(())
}

/// Static quantization preparation
// Temporarily disabled: pub fn prepare_qat(_module: &mut dyn torsh_nn::Module) -> TorshResult<()> {
#[allow(dead_code)]
pub fn prepare_qat(module: &mut dyn crate::qat::Module) -> TorshResult<()> {
    // Insert fake quantization operations into the module for QAT
    // This is a simplified implementation that sets up the module for QAT

    // Switch module to training mode for QAT
    module.train(true);

    // Get named parameters to track which parameters need fake quantization
    let named_params = module.named_parameters();

    // For each parameter, we would typically insert fake quantization nodes
    // In this simplified implementation, we just validate that parameters exist
    for (name, param) in named_params {
        // Validate parameter can be quantized
        if param.numel() == 0 {
            return Err(TorshError::InvalidArgument(format!(
                "Parameter {} is empty and cannot be quantized",
                name
            )));
        }

        // In a real implementation, we would insert fake quantization observers
        // and prepare the parameter for quantization-aware training
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::{tensor_1d, tensor_2d};

    /// F249: the runtime-dispatched kernel must agree with the scalar
    /// reference for both the signed and the unsigned code range.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_simd_dispatch_matches_scalar_kernel() {
        let data: Vec<f32> = (0..133).map(|i| (i as f32 - 66.0) * 0.37).collect();
        let scale = 0.05;

        for (qmin, qmax, zero_point) in [(-128, 127, -3), (0, 255, 170)] {
            let mut expected = vec![0.0f32; data.len()];
            quantize_scalar_f32(&data, scale, zero_point, qmin, qmax, &mut expected);
            let actual = quantize_optimized(&data, scale, zero_point, qmin, qmax);

            for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                // SIMD rounds ties to even, the scalar kernel rounds ties away
                // from zero, so a one-code difference is admissible.
                assert!(
                    (a - e).abs() <= 1.0,
                    "lane {i}: simd {a} vs scalar {e} (range [{qmin}, {qmax}])"
                );
                assert!((qmin as f32..=qmax as f32).contains(a));
            }
        }
    }

    /// F249: the AVX2 kernel itself must compile and produce correct codes.
    /// Skipped at runtime on CPUs without AVX2/FMA, but always type-checked.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_kernel_matches_scalar_kernel() {
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }

        let data: Vec<f32> = (0..37).map(|i| (i as f32 - 18.0) * 0.31).collect();
        let (scale, zero_point, qmin, qmax) = (0.05f32, 12, -128, 127);

        let mut expected = vec![0.0f32; data.len()];
        quantize_scalar_f32(&data, scale, zero_point, qmin, qmax, &mut expected);

        let mut actual = vec![0.0f32; data.len()];
        // SAFETY: avx2 and fma were verified above; buffers have equal length.
        unsafe {
            quantize_avx2_f32(&data, scale, zero_point, qmin, qmax, &mut actual);
        }

        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() <= 1.0, "lane {i}: avx2 {a} vs scalar {e}");
        }
    }

    #[test]
    fn test_calculate_qparams() {
        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let tensor = tensor_1d(&data).unwrap();

        let (scale, zero_point) = calculate_qparams(&tensor, -128, 127, DType::I8).unwrap();

        // Scale should be approximately (2.0 - (-2.0)) / (127 - (-128)) = 4.0 / 255
        assert!(scale > 0.0);
        assert!((-128..=127).contains(&zero_point));
    }

    #[test]
    fn test_quantize_per_tensor_affine() {
        let data = vec![0.0, 1.0, 2.0, 3.0];
        let tensor = tensor_1d(&data).unwrap();

        let scale = 0.1;
        let zero_point = 0;

        let (quantized, ret_scale, ret_zero_point) =
            quantize_per_tensor_affine(&tensor, scale, zero_point).unwrap();

        assert_eq!(ret_scale, scale);
        assert_eq!(ret_zero_point, zero_point);
        assert_eq!(quantized.shape().dims(), tensor.shape().dims());
    }

    #[test]
    fn test_quantize_tensor_auto() {
        let data = vec![-1.0, 0.0, 1.0, 2.0];
        let tensor = tensor_1d(&data).unwrap();

        let (quantized, scale, zero_point) =
            quantize_tensor_auto(&tensor, DType::I8, QScheme::PerTensorAffine).unwrap();

        assert!(scale > 0.0);
        assert!((-128..=127).contains(&zero_point));
        assert_eq!(quantized.shape().dims(), tensor.shape().dims());
    }

    #[test]
    fn test_per_channel_quantization() {
        // Create a 2x3 tensor where each row has different scales
        let tensor = tensor_2d(&[
            &[0.0, 1.0, 2.0],  // Channel 0: range [0, 2]
            &[0.0, 5.0, 10.0], // Channel 1: range [0, 10]
        ])
        .unwrap();

        let axis = 0; // Quantize along the first dimension (channels)
        let (scales, zero_points) =
            calculate_per_channel_qparams(&tensor, axis, DType::I8).unwrap();

        assert_eq!(scales.len(), 2);
        assert_eq!(zero_points.len(), 2);

        // Channel 1 should have a larger scale than channel 0
        assert!(scales[1] > scales[0]);

        let (quantized, ret_scales, ret_zero_points) =
            quantize_per_channel_affine(&tensor, &scales, &zero_points, axis).unwrap();

        assert_eq!(ret_scales, scales);
        assert_eq!(ret_zero_points, zero_points);
        assert_eq!(quantized.shape().dims(), tensor.shape().dims());
    }

    #[test]
    fn test_per_channel_auto() {
        let tensor = tensor_2d(&[&[-2.0, 0.0, 2.0], &[-10.0, 0.0, 10.0]]).unwrap();

        let (quantized, scales, zero_points) =
            quantize_per_channel_auto(&tensor, 0, DType::I8, QScheme::PerChannelAffine).unwrap();

        assert_eq!(scales.len(), 2);
        assert_eq!(zero_points.len(), 2);
        assert!(scales[1] > scales[0]); // Channel 1 has larger range
        assert_eq!(quantized.shape().dims(), tensor.shape().dims());
    }

    #[test]
    fn test_per_channel_symmetric() {
        let tensor = tensor_2d(&[&[-1.0, 0.0, 1.0], &[-5.0, 0.0, 5.0]]).unwrap();

        let (_quantized, scales, zero_points) =
            quantize_per_channel_auto(&tensor, 0, DType::I8, QScheme::PerChannelSymmetric).unwrap();

        assert_eq!(scales.len(), 2);
        assert_eq!(zero_points.len(), 2);

        // All zero points should be 0 for symmetric quantization
        for &zp in &zero_points {
            assert_eq!(zp, 0);
        }

        assert!(scales[1] > scales[0]); // Channel 1 has larger range
    }
}
