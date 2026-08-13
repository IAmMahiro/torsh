//! Functional operations and activations for graph neural networks
// Framework infrastructure - components designed for future use
#![allow(dead_code)]
/// Crate-local result alias: the error type defaults to [`TorshError`],
/// so both `Result<T>` and `Result<T, OtherError>` stay valid.
type Result<T, E = torsh_core::error::TorshError> = std::result::Result<T, E>;

use torsh_tensor::Tensor;

/// Leaky ReLU activation function
/// LeakyReLU(x) = max(0, x) + negative_slope * min(0, x)
pub fn leaky_relu(input: &Tensor, negative_slope: f64) -> Result<Tensor> {
    let zero_tensor = torsh_tensor::creation::zeros_like(input)?;
    let positive_part = input.maximum(&zero_tensor)?;
    let negative_part = input
        .minimum(&zero_tensor)?
        .mul_scalar(negative_slope as f32)?;
    Ok(positive_part.add(&negative_part)?)
}

/// ELU (Exponential Linear Unit) activation function
/// ELU(x) = x if x > 0, alpha * (exp(x) - 1) if x <= 0
pub fn elu(input: &Tensor, alpha: f64) -> Result<Tensor> {
    let zero_tensor = torsh_tensor::creation::zeros_like(input)?;
    let positive_part = input.maximum(&zero_tensor)?;
    let negative_part = input.minimum(&zero_tensor)?;
    let exp_part = negative_part
        .exp()?
        .sub_scalar(1.0)?
        .mul_scalar(alpha as f32)?;
    Ok(positive_part.add(&exp_part)?)
}

/// Swish activation function (also known as SiLU)
/// Swish(x) = x * sigmoid(x)
pub fn swish(input: &Tensor) -> Result<Tensor> {
    let one_tensor = torsh_tensor::creation::ones_like(input)?;
    let neg_input = input.neg()?;
    let exp_neg = neg_input.exp()?;
    let denominator = one_tensor.add(&exp_neg)?;
    let sigmoid = one_tensor.div(&denominator)?;
    Ok(input.mul(&sigmoid)?)
}

/// GELU activation function (Gaussian Error Linear Unit)
/// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
pub fn gelu(input: &Tensor) -> Result<Tensor> {
    let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt() as f32;
    let coefficient = 0.044715_f32;

    let x_cubed = input.pow_scalar(3.0)?;
    let coeff_term = x_cubed.mul_scalar(coefficient)?;
    let sum_term = input.add(&coeff_term)?;
    let inner = sum_term.mul_scalar(sqrt_2_over_pi)?;
    let tanh_part = inner.tanh()?;

    let one_tensor = torsh_tensor::creation::ones_like(input)?;
    let one_plus_tanh = one_tensor.add(&tanh_part)?;
    let half_tensor = torsh_tensor::creation::ones_like(input)?.mul_scalar(0.5)?;

    Ok(half_tensor.mul(input)?.mul(&one_plus_tanh)?)
}

/// Mish activation function
/// Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
pub fn mish(input: &Tensor) -> Result<Tensor> {
    let one_tensor = torsh_tensor::creation::ones_like(input)?;
    let exp_input = input.exp()?;
    let one_plus_exp = one_tensor.add(&exp_input)?;
    let softplus = one_plus_exp.ln()?;
    let tanh_softplus = softplus.tanh()?;
    Ok(input.mul(&tanh_softplus)?)
}

/// Graph-specific normalization functions
pub mod normalization {
    use super::*;

    /// Layer normalization for graph features
    /// Normalizes features across the feature dimension for each node
    pub fn layer_norm(input: &Tensor, eps: f64) -> Result<Tensor> {
        let shape = input.shape();
        let dims = shape.dims();
        let last_dim = dims.len() - 1;
        let mean = input.mean(Some(&[last_dim]), false)?;

        // Create broadcast-compatible mean tensor
        let input_shape = dims.to_vec();
        let mut mean_shape = input_shape.clone();
        mean_shape[last_dim] = 1;

        let mean_expanded = mean.unsqueeze(-1)?;
        let mean_broadcasted = mean_expanded.expand(&input_shape)?;
        let diff = input.sub(&mean_broadcasted)?;

        let variance = diff.pow_scalar(2.0)?.mean(Some(&[last_dim]), false)?;

        let variance_expanded = variance.unsqueeze(-1)?;
        let variance_broadcasted = variance_expanded.expand(&input_shape)?;
        let eps_tensor =
            torsh_tensor::creation::ones_like(&variance_broadcasted)?.mul_scalar(eps as f32)?;
        let variance_plus_eps = variance_broadcasted.add(&eps_tensor)?;
        let std = variance_plus_eps.sqrt()?;
        Ok(diff.div(&std)?)
    }

    /// Graph normalization
    /// Normalizes node features by the square root of node degree
    pub fn graph_norm(input: &Tensor, edge_index: &Tensor, num_nodes: usize) -> Result<Tensor> {
        // Convert f32 edge_index to i64 for processing
        let edge_index_i32 = edge_index.to_i32_simd()?;
        let edge_index_i64 = edge_index_i32.to_i64_simd()?;
        let edge_data = crate::utils::tensor_to_vec2::<i64>(&edge_index_i64)?;
        let mut degrees = vec![0.0_f32; num_nodes];

        for j in 0..edge_data[0].len() {
            let src = edge_data[0][j] as usize;
            let dst = edge_data[1][j] as usize;
            if src < num_nodes {
                degrees[src] += 1.0;
            }
            if dst < num_nodes {
                degrees[dst] += 1.0;
            }
        }

        // Create degree tensor and normalize
        let degree_tensor = torsh_tensor::creation::from_vec(
            degrees
                .iter()
                .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
                .collect(),
            &[num_nodes],
            torsh_core::device::DeviceType::Cpu,
        )?;

        // Apply normalization - expand degree tensor to match input shape
        let input_shape_binding = input.shape();
        let input_shape = input_shape_binding.dims();
        let degree_expanded = degree_tensor.unsqueeze(-1)?.expand(input_shape)?;
        Ok(input.mul(&degree_expanded)?)
    }

    /// Batch normalization for graphs
    /// Normalizes features across the batch dimension
    pub fn batch_norm(input: &Tensor, eps: f64) -> Result<Tensor> {
        let mean = input.mean(Some(&[0]), true)?;
        let input_shape_binding = input.shape();
        let input_shape = input_shape_binding.dims();
        let mean_expanded = mean.expand(input_shape)?;
        let diff = input.sub(&mean_expanded)?;
        let variance = diff.pow_scalar(2.0)?.mean(Some(&[0]), true)?;
        let eps_tensor = torsh_tensor::creation::ones_like(&variance)?.mul_scalar(eps as f32)?;
        let variance_plus_eps = variance.add(&eps_tensor)?;
        let std = variance_plus_eps.sqrt()?;
        let std_expanded = std.expand(input_shape)?;
        Ok(diff.div(&std_expanded)?)
    }
}

/// Dropout function for regularization
pub fn dropout(input: &Tensor, p: f64, training: bool) -> Result<Tensor> {
    if !training || p == 0.0 {
        return Ok(input.clone());
    }

    if p == 1.0 {
        return Ok(torsh_tensor::creation::zeros_like(input)?);
    }

    // Create dropout mask using simple thresholding
    let keep_prob = 1.0 - p;
    let random_tensor = torsh_tensor::creation::rand_like(input)?;
    let keep_prob_tensor =
        torsh_tensor::creation::ones_like(input)?.mul_scalar(keep_prob as f32)?;

    // Create a mask by subtracting threshold and applying relu (values > 0 become positive, <= 0 become 0)
    let diff_tensor = random_tensor.sub(&keep_prob_tensor)?;
    let mask_raw = diff_tensor.relu()?;

    // Normalize mask to 0 or 1 by checking if > 0, then convert to binary mask
    let zero_tensor = torsh_tensor::creation::zeros_like(input)?;
    let _mask_binary = mask_raw.gt(&zero_tensor)?;

    // For now, let's use a simplified approach - just threshold the random tensor directly
    // This creates a binary effect where values above threshold are kept
    let inverted_prob = random_tensor.gt(&keep_prob_tensor)?;
    let _ones_f32 = torsh_tensor::creation::ones_like(input)?;
    let _zeros_f32 = torsh_tensor::creation::zeros_like(input)?;

    // Manual conversion from boolean to f32: if inverted_prob is true, use 0.0 (drop), else use 1.0 (keep)
    let inverted_data = inverted_prob.to_vec()?;
    let keep_data: Vec<f32> = inverted_data
        .iter()
        .map(|&drop| if drop { 0.0 } else { 1.0 })
        .collect();
    let keep_mask =
        torsh_tensor::creation::from_vec(keep_data, input.shape().dims(), input.device())?;

    // Apply dropout with proper scaling
    let masked = input.mul(&keep_mask)?;
    Ok(masked.div_scalar(keep_prob as f32)?)
}

/// Attention mechanism functions
pub mod attention {
    use super::*;

    /// Scaled dot-product attention
    pub fn scaled_dot_product_attention(
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Compute attention scores: Q @ K^T / sqrt(d_k)
        let binding = key.shape();
        let d_k = binding.dims().last().ok_or_else(|| {
            torsh_core::error::TorshError::InvalidArgument(
                "key tensor must have at least one dimension".to_string(),
            )
        })?;
        let scale = 1.0 / (*d_k as f64).sqrt();
        let key_transposed = key.transpose(-2, -1)?;
        let scores = query.matmul(&key_transposed)?.mul_scalar(scale as f32)?;

        // Apply mask if provided
        let masked_scores = if let Some(mask) = mask {
            let large_neg = torsh_tensor::creation::ones_like(&scores)?.mul_scalar(-1e9_f32)?;
            let mask_effect = mask.mul(&large_neg)?;
            scores.add(&mask_effect)?
        } else {
            scores
        };

        // Apply softmax and compute attention output
        let attention_weights = masked_scores.softmax(-1)?;
        Ok(attention_weights.matmul(value)?)
    }

    /// Multi-head attention mechanism
    pub fn multi_head_attention(
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        num_heads: usize,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let batch_size = query.shape().dims()[0];
        let seq_len = query.shape().dims()[1];
        let d_model = query.shape().dims()[2];
        let d_k = d_model / num_heads;

        // Reshape for multi-head attention
        let q = query
            .view(&[
                batch_size as i32,
                seq_len as i32,
                num_heads as i32,
                d_k as i32,
            ])?
            .transpose(1, 2)?;
        let k = key
            .view(&[
                batch_size as i32,
                seq_len as i32,
                num_heads as i32,
                d_k as i32,
            ])?
            .transpose(1, 2)?;
        let v = value
            .view(&[
                batch_size as i32,
                seq_len as i32,
                num_heads as i32,
                d_k as i32,
            ])?
            .transpose(1, 2)?;

        // Apply scaled dot-product attention
        let attention_output = scaled_dot_product_attention(&q, &k, &v, mask);

        // Reshape back to original dimensions
        Ok(attention_output?.transpose(1, 2)?.contiguous()?.view(&[
            batch_size as i32,
            seq_len as i32,
            d_model as i32,
        ])?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_leaky_relu() {
        let input = torsh_tensor::creation::from_vec(
            vec![-2.0, -1.0, 0.0, 1.0, 2.0],
            &[5],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap();

        let output = leaky_relu(&input, 0.01);
        let expected = vec![-0.02, -0.01, 0.0, 1.0, 2.0];

        let output_vec = output
            .expect("operation should succeed")
            .to_vec()
            .expect("conversion should succeed");
        for (actual, expected) in output_vec.iter().zip(expected.iter()) {
            assert_relative_eq!(actual, expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_elu() {
        let input = torsh_tensor::creation::from_vec(
            vec![-1.0, 0.0, 1.0],
            &[3],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap();

        let output = elu(&input, 1.0);
        let output_vec = output
            .expect("operation should succeed")
            .to_vec()
            .expect("conversion should succeed");

        // For x > 0, ELU(x) = x
        assert_relative_eq!(output_vec[2], 1.0, epsilon = 1e-6);

        // For x = 0, ELU(x) = 0
        assert_relative_eq!(output_vec[1], 0.0, epsilon = 1e-6);

        // For x < 0, ELU(x) = alpha * (exp(x) - 1)
        let expected_negative = 1.0 * ((-1.0_f32).exp() - 1.0);
        assert_relative_eq!(output_vec[0], expected_negative, epsilon = 1e-6);
    }

    #[test]
    fn test_swish() {
        let input = torsh_tensor::creation::from_vec(
            vec![0.0, 1.0, -1.0],
            &[3],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap();

        let output = swish(&input);
        let output_vec = output
            .expect("operation should succeed")
            .to_vec()
            .expect("conversion should succeed");

        // Swish(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        assert_relative_eq!(output_vec[0], 0.0, epsilon = 1e-6);

        // Swish(1) = 1 * sigmoid(1) ≈ 1 * 0.731 ≈ 0.731
        let sigmoid_1 = 1.0 / (1.0 + (-1.0_f32).exp());
        assert_relative_eq!(output_vec[1], sigmoid_1, epsilon = 1e-6);
    }

    #[test]
    fn test_layer_norm() {
        let input = torsh_tensor::creation::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap();

        let output = normalization::layer_norm(&input, 1e-8).expect("operation should succeed");

        // Check that each row is normalized (mean ≈ 0, std ≈ 1)
        let output_2d = crate::utils::tensor_to_vec2::<f32>(&output).unwrap();

        for row in output_2d {
            let mean: f32 = row.iter().sum::<f32>() / row.len() as f32;
            let var: f32 = row.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / row.len() as f32;

            assert_relative_eq!(mean, 0.0, epsilon = 1e-6);
            assert_relative_eq!(var.sqrt(), 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_dropout() {
        let input = torsh_tensor::creation::ones(&[100]).unwrap();

        // Test training mode with 0.5 dropout
        let output_training = dropout(&input, 0.5, true);
        let output_vec = output_training
            .expect("operation should succeed")
            .to_vec()
            .expect("conversion should succeed");

        // Should have approximately 50% zeros and 50% values scaled by 2.0
        let num_zeros = output_vec.iter().filter(|&&x| x == 0.0).count();
        let num_nonzeros = output_vec.iter().filter(|&&x| x != 0.0).count();

        assert!(num_zeros > 30 && num_zeros < 70); // Roughly 50% with some variance
        assert!(num_nonzeros > 30 && num_nonzeros < 70);

        // Test eval mode (no dropout)
        let output_eval = dropout(&input, 0.5, false);
        let output_eval_vec = output_eval
            .expect("operation should succeed")
            .to_vec()
            .expect("conversion should succeed");
        assert!(output_eval_vec.iter().all(|&x| x == 1.0));
    }
}
