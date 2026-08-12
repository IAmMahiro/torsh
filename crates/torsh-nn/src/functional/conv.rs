//! Convolution operations for neural networks
//!
//! This module provides 1D, 2D, and 3D convolution operations with comprehensive
//! parameter support including stride, padding, dilation, and groups.

use torsh_core::error::Result;
use torsh_tensor::Tensor;

// =============================================================================
// 1D CONVOLUTION
// =============================================================================

/// 1D convolution function
pub fn conv1d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<Tensor> {
    // Input shape: [batch_size, in_channels, length]
    // Weight shape: [out_channels, in_channels/groups, kernel_size]
    // Output shape: [batch_size, out_channels, out_length]

    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();
    let weight_shape_obj = weight.shape();
    let weight_shape = weight_shape_obj.dims();

    if input_shape.len() != 3 {
        return Err(torsh_core::error::TorshError::InvalidShape(format!(
            "Input must be 3D [batch, in_channels, length], got {}D",
            input_shape.len()
        )));
    }

    if weight_shape.len() != 3 {
        return Err(torsh_core::error::TorshError::InvalidShape(format!(
            "Weight must be 3D [out_channels, in_channels/groups, kernel_size], got {}D",
            weight_shape.len()
        )));
    }

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let in_length = input_shape[2];

    let out_channels = weight_shape[0];
    let kernel_size = weight_shape[2];

    // Validate groups
    if in_channels % groups != 0 || out_channels % groups != 0 {
        return Err(torsh_core::error::TorshError::InvalidShape(format!(
            "in_channels ({}) and out_channels ({}) must be divisible by groups ({})",
            in_channels, out_channels, groups
        )));
    }

    // Calculate output dimensions
    let out_length = (in_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Extract data once for efficiency
    let input_vec = input.to_vec()?;
    let weight_vec = weight.to_vec()?;

    // Build output data directly
    let mut output_data = vec![0.0f32; batch_size * out_channels * out_length];

    // Implement 1D convolution
    for b in 0..batch_size {
        for g in 0..groups {
            let channels_per_group = in_channels / groups;
            let out_channels_per_group = out_channels / groups;
            let group_start = g * channels_per_group;
            let out_start = g * out_channels_per_group;

            for oc in 0..out_channels_per_group {
                let global_oc = out_start + oc;

                for ol in 0..out_length {
                    let mut sum = 0.0f32;

                    // Convolve over kernel
                    for ic in 0..channels_per_group {
                        let global_ic = group_start + ic;

                        for k in 0..kernel_size {
                            // Calculate input position
                            let il = (ol * stride + k * dilation) as i32 - padding as i32;

                            // Check bounds
                            if il >= 0 && il < in_length as i32 {
                                // Compute flat indices
                                let input_idx = b * in_channels * in_length
                                    + global_ic * in_length
                                    + il as usize;

                                let weight_idx = global_oc * (in_channels / groups) * kernel_size
                                    + ic * kernel_size
                                    + k;

                                sum += input_vec[input_idx] * weight_vec[weight_idx];
                            }
                        }
                    }

                    // Store result
                    let output_idx = b * out_channels * out_length + global_oc * out_length + ol;
                    output_data[output_idx] = sum;
                }
            }
        }
    }

    // Create output tensor from data
    let mut output = Tensor::from_vec(output_data, &[batch_size, out_channels, out_length])?;

    // Apply bias if provided
    if let Some(bias_tensor) = bias {
        let bias_reshaped = bias_tensor.reshape(&[1, out_channels as i32, 1])?;
        output = output.add(&bias_reshaped)?;
    }

    Ok(output)
}

// =============================================================================
// 2D CONVOLUTION
// =============================================================================

/// 2D convolution function
pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
) -> Result<Tensor> {
    // Input shape: [batch_size, in_channels, height, width]
    // Weight shape: [out_channels, in_channels/groups, kernel_height, kernel_width]
    // Output shape: [batch_size, out_channels, out_height, out_width]

    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();
    let weight_shape_obj = weight.shape();
    let weight_shape = weight_shape_obj.dims();

    if input_shape.len() != 4 {
        return Err(torsh_core::error::TorshError::InvalidShape(format!(
            "Input must be 4D [batch, in_channels, height, width], got {}D",
            input_shape.len()
        )));
    }

    if weight_shape.len() != 4 {
        return Err(torsh_core::error::TorshError::InvalidShape(format!(
            "Weight must be 4D [out_channels, in_channels/groups, kernel_h, kernel_w], got {}D",
            weight_shape.len()
        )));
    }

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let in_height = input_shape[2];
    let in_width = input_shape[3];

    let out_channels = weight_shape[0];
    let kernel_height = weight_shape[2];
    let kernel_width = weight_shape[3];

    // Validate groups
    if in_channels % groups != 0 || out_channels % groups != 0 {
        return Err(torsh_core::error::TorshError::InvalidShape(format!(
            "in_channels ({}) and out_channels ({}) must be divisible by groups ({})",
            in_channels, out_channels, groups
        )));
    }

    // Calculate output dimensions
    let out_height =
        (in_height + 2 * padding.0 - dilation.0 * (kernel_height - 1) - 1) / stride.0 + 1;
    let out_width = (in_width + 2 * padding.1 - dilation.1 * (kernel_width - 1) - 1) / stride.1 + 1;

    // Extract data once for efficiency
    let input_vec = input.to_vec()?;
    let weight_vec = weight.to_vec()?;

    // Build output data directly
    let mut output_data = vec![0.0f32; batch_size * out_channels * out_height * out_width];

    // Implement 2D convolution using direct approach
    for b in 0..batch_size {
        for g in 0..groups {
            let channels_per_group = in_channels / groups;
            let out_channels_per_group = out_channels / groups;
            let group_start = g * channels_per_group;
            let out_start = g * out_channels_per_group;

            for oc in 0..out_channels_per_group {
                let global_oc = out_start + oc;

                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let mut sum = 0.0f32;

                        // Convolve over kernel
                        for ic in 0..channels_per_group {
                            let global_ic = group_start + ic;

                            for kh in 0..kernel_height {
                                for kw in 0..kernel_width {
                                    // Calculate input position with stride, padding, and dilation
                                    let ih =
                                        (oh * stride.0 + kh * dilation.0) as i32 - padding.0 as i32;
                                    let iw =
                                        (ow * stride.1 + kw * dilation.1) as i32 - padding.1 as i32;

                                    // Check bounds
                                    if ih >= 0
                                        && ih < in_height as i32
                                        && iw >= 0
                                        && iw < in_width as i32
                                    {
                                        // Compute flat indices
                                        let input_idx = b * in_channels * in_height * in_width
                                            + global_ic * in_height * in_width
                                            + ih as usize * in_width
                                            + iw as usize;

                                        let weight_idx = global_oc
                                            * (in_channels / groups)
                                            * kernel_height
                                            * kernel_width
                                            + ic * kernel_height * kernel_width
                                            + kh * kernel_width
                                            + kw;

                                        sum += input_vec[input_idx] * weight_vec[weight_idx];
                                    }
                                }
                            }
                        }

                        // Store result
                        let output_idx = b * out_channels * out_height * out_width
                            + global_oc * out_height * out_width
                            + oh * out_width
                            + ow;
                        output_data[output_idx] = sum;
                    }
                }
            }
        }
    }

    // Create output tensor from data
    let mut output = Tensor::from_vec(
        output_data,
        &[batch_size, out_channels, out_height, out_width],
    )?;

    // Apply bias if provided
    if let Some(bias_tensor) = bias {
        let bias_reshaped = bias_tensor.reshape(&[1, out_channels as i32, 1, 1])?;
        output = output.add(&bias_reshaped)?;
    }

    Ok(output)
}

// =============================================================================
// 3D CONVOLUTION
// =============================================================================

/// 3D convolution function
pub fn conv3d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    dilation: (usize, usize, usize),
    groups: usize,
) -> Result<Tensor> {
    // Input shape: [batch_size, in_channels, depth, height, width]
    // Weight shape: [out_channels, in_channels/groups, kernel_depth, kernel_height, kernel_width]
    // Output shape: [batch_size, out_channels, out_depth, out_height, out_width]

    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();
    let weight_shape_obj = weight.shape();
    let weight_shape = weight_shape_obj.dims();

    if input_shape.len() != 5 {
        return Err(torsh_core::error::TorshError::InvalidShape(format!(
            "Input must be 5D [batch, in_channels, depth, height, width], got {}D",
            input_shape.len()
        )));
    }

    if weight_shape.len() != 5 {
        return Err(torsh_core::error::TorshError::InvalidShape(format!(
            "Weight must be 5D [out_channels, in_channels/groups, kernel_d, kernel_h, kernel_w], got {}D", weight_shape.len()
        )));
    }

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let in_depth = input_shape[2];
    let in_height = input_shape[3];
    let in_width = input_shape[4];

    let out_channels = weight_shape[0];
    let kernel_depth = weight_shape[2];
    let kernel_height = weight_shape[3];
    let kernel_width = weight_shape[4];

    // Validate groups
    if in_channels % groups != 0 || out_channels % groups != 0 {
        return Err(torsh_core::error::TorshError::InvalidShape(format!(
            "in_channels ({}) and out_channels ({}) must be divisible by groups ({})",
            in_channels, out_channels, groups
        )));
    }

    // Calculate output dimensions
    let out_depth = (in_depth + 2 * padding.0 - dilation.0 * (kernel_depth - 1) - 1) / stride.0 + 1;
    let out_height =
        (in_height + 2 * padding.1 - dilation.1 * (kernel_height - 1) - 1) / stride.1 + 1;
    let out_width = (in_width + 2 * padding.2 - dilation.2 * (kernel_width - 1) - 1) / stride.2 + 1;

    // Extract data once for efficiency
    let input_vec = input.to_vec()?;
    let weight_vec = weight.to_vec()?;

    // Build output data directly
    let mut output_data =
        vec![0.0f32; batch_size * out_channels * out_depth * out_height * out_width];

    // Implement 3D convolution
    for b in 0..batch_size {
        for g in 0..groups {
            let channels_per_group = in_channels / groups;
            let out_channels_per_group = out_channels / groups;
            let group_start = g * channels_per_group;
            let out_start = g * out_channels_per_group;

            for oc in 0..out_channels_per_group {
                let global_oc = out_start + oc;

                for od in 0..out_depth {
                    for oh in 0..out_height {
                        for ow in 0..out_width {
                            let mut sum = 0.0f32;

                            // Convolve over kernel
                            for ic in 0..channels_per_group {
                                let global_ic = group_start + ic;

                                for kd in 0..kernel_depth {
                                    for kh in 0..kernel_height {
                                        for kw in 0..kernel_width {
                                            // Calculate input position
                                            let id = (od * stride.0 + kd * dilation.0) as i32
                                                - padding.0 as i32;
                                            let ih = (oh * stride.1 + kh * dilation.1) as i32
                                                - padding.1 as i32;
                                            let iw = (ow * stride.2 + kw * dilation.2) as i32
                                                - padding.2 as i32;

                                            // Check bounds
                                            if id >= 0
                                                && id < in_depth as i32
                                                && ih >= 0
                                                && ih < in_height as i32
                                                && iw >= 0
                                                && iw < in_width as i32
                                            {
                                                // Compute flat indices
                                                let input_idx = b
                                                    * in_channels
                                                    * in_depth
                                                    * in_height
                                                    * in_width
                                                    + global_ic * in_depth * in_height * in_width
                                                    + id as usize * in_height * in_width
                                                    + ih as usize * in_width
                                                    + iw as usize;

                                                let weight_idx = global_oc
                                                    * (in_channels / groups)
                                                    * kernel_depth
                                                    * kernel_height
                                                    * kernel_width
                                                    + ic * kernel_depth
                                                        * kernel_height
                                                        * kernel_width
                                                    + kd * kernel_height * kernel_width
                                                    + kh * kernel_width
                                                    + kw;

                                                sum +=
                                                    input_vec[input_idx] * weight_vec[weight_idx];
                                            }
                                        }
                                    }
                                }
                            }

                            // Store result
                            let output_idx = b * out_channels * out_depth * out_height * out_width
                                + global_oc * out_depth * out_height * out_width
                                + od * out_height * out_width
                                + oh * out_width
                                + ow;
                            output_data[output_idx] = sum;
                        }
                    }
                }
            }
        }
    }

    // Create output tensor from data
    let mut output = Tensor::from_vec(
        output_data,
        &[batch_size, out_channels, out_depth, out_height, out_width],
    )?;

    // Apply bias if provided
    if let Some(bias_tensor) = bias {
        let bias_reshaped = bias_tensor.reshape(&[1, out_channels as i32, 1, 1, 1])?;
        output = output.add(&bias_reshaped)?;
    }

    Ok(output)
}

// =============================================================================
// TRANSPOSED (DECONVOLUTION) OPERATIONS
// =============================================================================

/// Scatter-add kernel shared by the 1-D/2-D/3-D transposed convolutions.
///
/// Transposed convolution is the gradient of a convolution with respect to its
/// input: every input element is multiplied by the whole kernel and accumulated
/// into the output at `pos * stride + k * dilation - padding`. Positions that
/// fall outside the (cropped) output are dropped, which is exactly what the
/// `padding` argument means here.
///
/// Shapes use the generic N-D layout `[batch, channels, spatial...]` with the
/// weight laid out as `[in_channels, out_channels / groups, kernel...]`.
struct TransposedConvSpec<'a> {
    input_spatial: &'a [usize],
    kernel: &'a [usize],
    stride: &'a [usize],
    padding: &'a [usize],
    output_padding: &'a [usize],
    dilation: &'a [usize],
    groups: usize,
}

impl TransposedConvSpec<'_> {
    /// Output extent of each spatial axis.
    ///
    /// Over-cropping (a `padding` larger than the uncropped extent) is reported
    /// as an error instead of wrapping around in unsigned arithmetic.
    fn output_spatial(&self) -> Result<Vec<usize>> {
        let mut extents = Vec::with_capacity(self.input_spatial.len());
        for (axis, &size) in self.input_spatial.iter().enumerate() {
            if size == 0 {
                return Err(torsh_core::error::TorshError::InvalidShape(
                    "conv_transpose does not accept an empty spatial dimension".to_string(),
                ));
            }
            let uncropped = (size - 1) * self.stride[axis]
                + self.dilation[axis] * (self.kernel[axis] - 1)
                + self.output_padding[axis]
                + 1;
            let cropped = uncropped
                .checked_sub(2 * self.padding[axis])
                .filter(|&extent| extent > 0)
                .ok_or_else(|| {
                    torsh_core::error::TorshError::InvalidArgument(format!(
                        "padding {} crops the whole output of axis {axis} (extent {uncropped})",
                        self.padding[axis]
                    ))
                })?;
            extents.push(cropped);
        }
        Ok(extents)
    }
}

/// Validate the transposed-convolution arguments and report the output extents.
fn transposed_conv_dims(
    input_shape: &[usize],
    weight_shape: &[usize],
    spec: &TransposedConvSpec<'_>,
) -> Result<(usize, usize, Vec<usize>)> {
    let batch_size = input_shape[0];
    let in_channels = input_shape[1];

    if spec.groups == 0 {
        return Err(torsh_core::error::TorshError::InvalidArgument(
            "groups must be positive".to_string(),
        ));
    }
    if in_channels != weight_shape[0] {
        return Err(torsh_core::error::TorshError::InvalidShape(format!(
            "weight has {} input channels but the input has {}",
            weight_shape[0], in_channels
        )));
    }
    if in_channels % spec.groups != 0 {
        return Err(torsh_core::error::TorshError::InvalidShape(format!(
            "in_channels ({}) must be divisible by groups ({})",
            in_channels, spec.groups
        )));
    }
    for (axis, &size) in spec.kernel.iter().enumerate() {
        if size == 0 || spec.stride[axis] == 0 || spec.dilation[axis] == 0 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "kernel size, stride and dilation must all be positive".to_string(),
            ));
        }
        if spec.output_padding[axis] >= spec.stride[axis].max(spec.dilation[axis]) {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "output_padding ({}) must be smaller than stride ({}) or dilation ({})",
                spec.output_padding[axis], spec.stride[axis], spec.dilation[axis]
            )));
        }
    }

    let out_channels = weight_shape[1] * spec.groups;
    let output_spatial = spec.output_spatial()?;
    Ok((batch_size, out_channels, output_spatial))
}

/// Generic N-D transposed convolution over flat row-major buffers.
fn transposed_conv_nd(
    input: &Tensor,
    weight: &Tensor,
    spec: &TransposedConvSpec<'_>,
) -> Result<(Vec<f32>, usize, usize, Vec<usize>)> {
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();
    let weight_shape_obj = weight.shape();
    let weight_shape = weight_shape_obj.dims();

    let (batch_size, out_channels, out_spatial) =
        transposed_conv_dims(input_shape, weight_shape, spec)?;

    let in_channels = input_shape[1];
    let axes = spec.input_spatial.len();
    let in_spatial_size: usize = spec.input_spatial.iter().product();
    let out_spatial_size: usize = out_spatial.iter().product();
    let kernel_size: usize = spec.kernel.iter().product();
    let out_per_group = weight_shape[1];
    let in_per_group = in_channels / spec.groups;

    let input_data = input.to_vec()?;
    let weight_data = weight.to_vec()?;
    let mut output_data = vec![0.0f32; batch_size * out_channels * out_spatial_size];

    // Row-major coordinate decoding helpers.
    let mut in_coords = vec![0usize; axes];
    let mut kernel_coords = vec![0usize; axes];

    for batch in 0..batch_size {
        for in_channel in 0..in_channels {
            let group = in_channel / in_per_group;
            let input_base = (batch * in_channels + in_channel) * in_spatial_size;
            let weight_base = in_channel * out_per_group * kernel_size;

            for flat_in in 0..in_spatial_size {
                let value = input_data[input_base + flat_in];
                if value == 0.0 {
                    continue;
                }

                let mut remaining = flat_in;
                for axis in (0..axes).rev() {
                    in_coords[axis] = remaining % spec.input_spatial[axis];
                    remaining /= spec.input_spatial[axis];
                }

                for flat_k in 0..kernel_size {
                    let mut remaining = flat_k;
                    for axis in (0..axes).rev() {
                        kernel_coords[axis] = remaining % spec.kernel[axis];
                        remaining /= spec.kernel[axis];
                    }

                    // Map the (input position, kernel tap) pair to an output cell.
                    let mut flat_out = 0usize;
                    let mut inside = true;
                    for axis in 0..axes {
                        let position = in_coords[axis] * spec.stride[axis]
                            + kernel_coords[axis] * spec.dilation[axis];
                        if position < spec.padding[axis] {
                            inside = false;
                            break;
                        }
                        let position = position - spec.padding[axis];
                        if position >= out_spatial[axis] {
                            inside = false;
                            break;
                        }
                        flat_out = flat_out * out_spatial[axis] + position;
                    }
                    if !inside {
                        continue;
                    }

                    for out_channel in 0..out_per_group {
                        let weight_value =
                            weight_data[weight_base + out_channel * kernel_size + flat_k];
                        let global_out = group * out_per_group + out_channel;
                        let output_index =
                            (batch * out_channels + global_out) * out_spatial_size + flat_out;
                        output_data[output_index] += value * weight_value;
                    }
                }
            }
        }
    }

    Ok((output_data, batch_size, out_channels, out_spatial))
}

/// 1D transposed convolution (deconvolution)
///
/// Input `[batch, in_channels, length]`, weight
/// `[in_channels, out_channels / groups, kernel]`, output
/// `[batch, out_channels, (length - 1) * stride - 2 * padding + dilation *
/// (kernel - 1) + output_padding + 1]`.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose1d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    groups: usize,
    dilation: usize,
) -> Result<Tensor> {
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();
    let weight_shape_obj = weight.shape();
    let weight_shape = weight_shape_obj.dims();

    if input_shape.len() != 3 || weight_shape.len() != 3 {
        return Err(torsh_core::error::TorshError::InvalidArgument(
            "Input and weight must be 3D tensors for conv_transpose1d".to_string(),
        ));
    }

    let spec = TransposedConvSpec {
        input_spatial: &input_shape[2..],
        kernel: &weight_shape[2..],
        stride: &[stride],
        padding: &[padding],
        output_padding: &[output_padding],
        dilation: &[dilation],
        groups,
    };

    let (data, batch_size, out_channels, out_spatial) = transposed_conv_nd(input, weight, &spec)?;
    let mut result = Tensor::from_vec(data, &[batch_size, out_channels, out_spatial[0]])?;

    if let Some(bias_tensor) = bias {
        let bias_reshaped = bias_tensor.reshape(&[1, out_channels as i32, 1])?;
        result = result.add_op(&bias_reshaped)?;
    }

    Ok(result)
}

/// 2D transposed convolution (deconvolution)
///
/// Input `[batch, in_channels, height, width]`, weight
/// `[in_channels, out_channels / groups, kernel_h, kernel_w]`.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    output_padding: (usize, usize),
    groups: usize,
    dilation: (usize, usize),
) -> Result<Tensor> {
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();
    let weight_shape_obj = weight.shape();
    let weight_shape = weight_shape_obj.dims();

    if input_shape.len() != 4 || weight_shape.len() != 4 {
        return Err(torsh_core::error::TorshError::InvalidArgument(
            "Input and weight must be 4D tensors for conv_transpose2d".to_string(),
        ));
    }

    let spec = TransposedConvSpec {
        input_spatial: &input_shape[2..],
        kernel: &weight_shape[2..],
        stride: &[stride.0, stride.1],
        padding: &[padding.0, padding.1],
        output_padding: &[output_padding.0, output_padding.1],
        dilation: &[dilation.0, dilation.1],
        groups,
    };

    let (data, batch_size, out_channels, out_spatial) = transposed_conv_nd(input, weight, &spec)?;
    let mut result = Tensor::from_vec(
        data,
        &[batch_size, out_channels, out_spatial[0], out_spatial[1]],
    )?;

    if let Some(bias_tensor) = bias {
        let bias_reshaped = bias_tensor.reshape(&[1, out_channels as i32, 1, 1])?;
        result = result.add_op(&bias_reshaped)?;
    }

    Ok(result)
}

/// 3D transposed convolution (deconvolution)
///
/// Input `[batch, in_channels, depth, height, width]`, weight
/// `[in_channels, out_channels / groups, kernel_d, kernel_h, kernel_w]`.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose3d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    output_padding: (usize, usize, usize),
    groups: usize,
    dilation: (usize, usize, usize),
) -> Result<Tensor> {
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();
    let weight_shape_obj = weight.shape();
    let weight_shape = weight_shape_obj.dims();

    if input_shape.len() != 5 || weight_shape.len() != 5 {
        return Err(torsh_core::error::TorshError::InvalidArgument(
            "Input and weight must be 5D tensors for conv_transpose3d".to_string(),
        ));
    }

    let spec = TransposedConvSpec {
        input_spatial: &input_shape[2..],
        kernel: &weight_shape[2..],
        stride: &[stride.0, stride.1, stride.2],
        padding: &[padding.0, padding.1, padding.2],
        output_padding: &[output_padding.0, output_padding.1, output_padding.2],
        dilation: &[dilation.0, dilation.1, dilation.2],
        groups,
    };

    let (data, batch_size, out_channels, out_spatial) = transposed_conv_nd(input, weight, &spec)?;
    let mut result = Tensor::from_vec(
        data,
        &[
            batch_size,
            out_channels,
            out_spatial[0],
            out_spatial[1],
            out_spatial[2],
        ],
    )?;

    if let Some(bias_tensor) = bias {
        let bias_reshaped = bias_tensor.reshape(&[1, out_channels as i32, 1, 1, 1])?;
        result = result.add_op(&bias_reshaped)?;
    }

    Ok(result)
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/// Calculate output size for convolution operation
pub fn conv_output_size(
    input_size: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> usize {
    (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
}

/// Calculate output size for transposed convolution operation
pub fn conv_transpose_output_size(
    input_size: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
) -> usize {
    (input_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
}

/// Validate convolution parameters
pub fn validate_conv_params(
    in_channels: usize,
    out_channels: usize,
    groups: usize,
    kernel_size: &[usize],
) -> Result<()> {
    if in_channels % groups != 0 {
        return Err(torsh_core::error::TorshError::InvalidArgument(format!(
            "in_channels ({}) must be divisible by groups ({})",
            in_channels, groups
        )));
    }

    if out_channels % groups != 0 {
        return Err(torsh_core::error::TorshError::InvalidArgument(format!(
            "out_channels ({}) must be divisible by groups ({})",
            out_channels, groups
        )));
    }

    for &size in kernel_size {
        if size == 0 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "Kernel size must be positive".to_string(),
            ));
        }
    }

    Ok(())
}
