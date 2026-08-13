//! Regularization functions for training stability
//!
//! This module provides regularization techniques commonly used in deep learning,
//! particularly for adversarial training and GAN stabilization.

use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::{creation::rand, Tensor};

/// Differentiate `sum(discriminator_fn(x))` with respect to `x`.
///
/// The sample is copied into a *fresh* leaf tensor before the forward pass: a
/// clone or `detach()` shares the caller's gradient slot, so the backward pass
/// would silently accumulate into the caller's tensors as a side effect.
///
/// Summing the discriminator output is exact for a batched penalty because the
/// discriminator treats samples independently, so `d(sum_b f(x_b))/dx_b` is the
/// per-sample gradient.
fn input_gradient<F>(sample: &Tensor, discriminator_fn: &F, context: &str) -> TorshResult<Tensor>
where
    F: Fn(&Tensor) -> TorshResult<Tensor>,
{
    let leaf = Tensor::from_data(
        sample.data()?,
        sample.shape().dims().to_vec(),
        sample.device(),
    )?
    .requires_grad_(true);

    let output = discriminator_fn(&leaf)?;
    if !output.requires_grad() {
        return Err(TorshError::invalid_argument_with_context(
            "the discriminator output is not connected to its input in the autograd \
             graph, so no gradient penalty can be computed; build the discriminator \
             from differentiable tensor operations",
            context,
        ));
    }

    output.sum()?.backward()?;

    leaf.grad().ok_or_else(|| {
        TorshError::invalid_argument_with_context(
            "the backward pass produced no gradient for the discriminator input; \
             one of the operations in the discriminator does not register a backward \
             rule",
            context,
        )
    })
}

/// Per-sample squared L2 norm of a gradient tensor shaped `[batch, ...]`.
fn per_sample_squared_norms(gradient: &Tensor, batch_size: usize) -> TorshResult<Vec<f32>> {
    let data = gradient.data()?;
    if batch_size == 0 || data.len() % batch_size != 0 {
        return Err(TorshError::InvalidArgument(format!(
            "gradient with {} elements cannot be split into {batch_size} samples",
            data.len()
        )));
    }
    let per_sample = data.len() / batch_size;
    Ok((0..batch_size)
        .map(|b| {
            data[b * per_sample..(b + 1) * per_sample]
                .iter()
                .map(|&value| value * value)
                .sum::<f32>()
        })
        .collect())
}

/// Fill `probe` with a deterministic pseudo-random unit vector.
///
/// Used to restart the power iteration when the current probe happens to lie in
/// the null space of the matrix (a uniform start does exactly that for, e.g.,
/// `[[1, -1], [1, -1]]`). The generator is a fixed LCG so results stay reproducible.
fn fill_probe_vector(probe: &mut [f32], seed: u32) {
    let mut state = seed | 1;
    let mut norm_squared = 0.0f32;
    for value in probe.iter_mut() {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let unit = (state >> 8) as f32 / (1u32 << 24) as f32; // [0, 1)
        *value = unit - 0.5;
        norm_squared += *value * *value;
    }
    let norm = norm_squared.sqrt();
    if norm > 0.0 {
        for value in probe.iter_mut() {
            *value /= norm;
        }
    }
}

/// Largest singular value of a `[rows, cols]` matrix, by power iteration on `MᵀM`.
///
/// Deterministic: the iteration starts from a uniform vector and, whenever a probe
/// is annihilated by the matrix, restarts from a fixed pseudo-random vector rather
/// than reporting a spectral norm of zero for a non-zero matrix.
fn spectral_norm_power_iteration(
    matrix: &[f32],
    rows: usize,
    cols: usize,
    iterations: usize,
) -> f32 {
    if rows == 0 || cols == 0 || matrix.iter().all(|&value| value == 0.0) {
        return 0.0;
    }
    let norm_of = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();

    let mut v = vec![1.0f32 / (cols as f32).sqrt(); cols];
    let mut u = vec![0.0f32; rows];
    let mut sigma = 0.0f32;
    let mut restarts = 0u32;
    const MAX_RESTARTS: u32 = 8;

    let mut step = 0usize;
    while step < iterations {
        // u = M v
        for (r, u_r) in u.iter_mut().enumerate() {
            *u_r = (0..cols).map(|c| matrix[r * cols + c] * v[c]).sum();
        }
        let u_norm = norm_of(&u);
        if u_norm < 1e-12 {
            // The probe lies in the null space; try a different one.
            if restarts >= MAX_RESTARTS {
                return sigma;
            }
            restarts += 1;
            fill_probe_vector(&mut v, 0x2545_F491u32.wrapping_mul(restarts));
            continue;
        }
        for u_r in u.iter_mut() {
            *u_r /= u_norm;
        }

        // v = Mᵀ u
        for (c, v_c) in v.iter_mut().enumerate() {
            *v_c = (0..rows).map(|r| matrix[r * cols + c] * u[r]).sum();
        }
        let v_norm = norm_of(&v);
        if v_norm < 1e-12 {
            if restarts >= MAX_RESTARTS {
                return sigma;
            }
            restarts += 1;
            fill_probe_vector(&mut v, 0x9E37_79B9u32.wrapping_mul(restarts));
            continue;
        }
        for v_c in v.iter_mut() {
            *v_c /= v_norm;
        }
        sigma = v_norm; // ||Mᵀ u|| converges to the largest singular value
        step += 1;
    }

    sigma
}

/// Apply the reduction shared by every penalty in this module.
fn apply_reduction(penalty: Tensor, reduction: &str, context: &str) -> TorshResult<Tensor> {
    match reduction {
        "none" => Ok(penalty),
        "mean" => penalty.mean(None, false),
        "sum" => penalty.sum(),
        _ => Err(TorshError::invalid_argument_with_context(
            &format!(
                "Invalid reduction: {}, expected 'none', 'mean', or 'sum'",
                reduction
            ),
            context,
        )),
    }
}

/// Gradient penalty for WGAN-GP (Wasserstein GAN with Gradient Penalty)
///
/// Computes the gradient penalty term used in WGAN-GP to enforce
/// Lipschitz constraint on the discriminator/critic.
///
/// # Arguments
/// * `real_samples` - Real data samples
/// * `fake_samples` - Generated/fake data samples  
/// * `discriminator_fn` - Function that computes discriminator output
/// * `lambda` - Gradient penalty coefficient (typically 10.0)
/// * `reduction` - Reduction method: "mean", "sum", or "none"
///
/// # Returns
/// Per-sample gradient penalty `lambda * (||∇_x D(x)||_2 - 1)^2`, reduced as requested.
///
/// # Note
/// The penalty is a *value*: the gradient it is built from is obtained by running
/// a backward pass, and the engine has no `create_graph` mode, so the returned
/// tensor cannot be differentiated a second time with respect to the
/// discriminator parameters.
pub fn gradient_penalty<F>(
    real_samples: &Tensor,
    fake_samples: &Tensor,
    discriminator_fn: F,
    lambda: f64,
    reduction: &str,
) -> TorshResult<Tensor>
where
    F: Fn(&Tensor) -> TorshResult<Tensor>,
{
    let context = "gradient_penalty";
    let dims = real_samples.shape().dims().to_vec();
    if dims.is_empty() {
        return Err(TorshError::invalid_argument_with_context(
            "real_samples must have a batch dimension",
            context,
        ));
    }
    if fake_samples.shape().dims() != dims.as_slice() {
        return Err(TorshError::ShapeMismatch {
            expected: dims.clone(),
            got: fake_samples.shape().dims().to_vec(),
        });
    }
    let batch_size = dims[0];
    let per_sample = real_samples.numel() / batch_size.max(1);

    // Random per-sample interpolation coefficient, as in the WGAN-GP paper.
    let epsilon: Tensor = rand(&[batch_size])?;
    let epsilon_data = epsilon.data()?;
    let real_data = real_samples.data()?;
    let fake_data = fake_samples.data()?;
    let mut interpolated_data = Vec::with_capacity(real_data.len());
    for b in 0..batch_size {
        let eps = epsilon_data[b];
        for i in 0..per_sample {
            let index = b * per_sample + i;
            interpolated_data.push(eps * real_data[index] + (1.0 - eps) * fake_data[index]);
        }
    }
    let interpolated = Tensor::from_data(interpolated_data, dims.clone(), real_samples.device())?;

    // Real gradient of the critic with respect to the interpolated samples
    let gradient = input_gradient(&interpolated, &discriminator_fn, context)?;
    let squared_norms = per_sample_squared_norms(&gradient, batch_size)?;

    // (||grad||_2 - 1)^2, scaled by lambda
    let penalty_data: Vec<f32> = squared_norms
        .iter()
        .map(|&sq| {
            let deviation = sq.sqrt() - 1.0;
            lambda as f32 * deviation * deviation
        })
        .collect();
    let penalty = Tensor::from_data(penalty_data, vec![batch_size], real_samples.device())?;

    apply_reduction(penalty, reduction, context)
}

/// Spectral normalization gradient penalty
///
/// Computes gradient penalty specifically designed for spectral normalization,
/// enforcing spectral norm constraints on network weights.
///
/// # Arguments
/// * `network_fn` - Function that computes the network output for an input batch
/// * `input_tensor` - Network input tensor `[batch, ...]`
/// * `lambda` - Penalty coefficient
/// * `reduction` - Reduction method
/// * `iterations` - Number of power iterations (20 is a good default)
///
/// # Note
/// The network is evaluated and differentiated with respect to `input_tensor`; the
/// per-sample gradients form the rows of the Jacobian-like matrix whose spectral
/// norm is estimated. The result is a value, not a doubly-differentiable term.
pub fn spectral_gradient_penalty<F>(
    network_fn: F,
    input_tensor: &Tensor,
    lambda: f64,
    reduction: &str,
    iterations: usize,
) -> TorshResult<Tensor>
where
    F: Fn(&Tensor) -> TorshResult<Tensor>,
{
    let context = "spectral_gradient_penalty";
    let dims = input_tensor.shape().dims().to_vec();
    if dims.is_empty() {
        return Err(TorshError::invalid_argument_with_context(
            "input_tensor must have a batch dimension",
            context,
        ));
    }
    let batch_size = dims[0];
    if batch_size == 0 {
        return Err(TorshError::invalid_argument_with_context(
            "input_tensor must contain at least one sample",
            context,
        ));
    }
    let per_sample = input_tensor.numel() / batch_size;

    // Real gradient of the network with respect to its input
    let gradient = input_gradient(input_tensor, &network_fn, context)?;
    let gradient_data = gradient.data()?;

    // Largest singular value of the [batch, features] gradient matrix
    let sigma_max =
        spectral_norm_power_iteration(&gradient_data, batch_size, per_sample, iterations.max(1));

    // Spectral gradient penalty: lambda * (sigma_max - 1)^2
    let deviation = sigma_max - 1.0;
    let penalty = Tensor::from_data(
        vec![lambda as f32 * deviation * deviation],
        vec![1],
        input_tensor.device(),
    )?;

    apply_reduction(penalty, reduction, context)
}

/// R1 gradient penalty used in StyleGAN
///
/// Implements the R1 regularization term from "Which Training Methods for GANs
/// do actually Converge?" This penalty encourages the discriminator to have
/// zero gradients on real data.
///
/// # Arguments
/// * `real_samples` - Real data samples
/// * `discriminator_fn` - Function that computes discriminator output
/// * `lambda` - Gradient penalty coefficient
/// * `reduction` - Reduction method
pub fn r1_gradient_penalty<F>(
    real_samples: &Tensor,
    discriminator_fn: F,
    lambda: f64,
    reduction: &str,
) -> TorshResult<Tensor>
where
    F: Fn(&Tensor) -> TorshResult<Tensor>,
{
    penalty_from_squared_gradient_norm(
        real_samples,
        discriminator_fn,
        lambda,
        reduction,
        "r1_gradient_penalty",
    )
}

/// Shared body of the R1/R2 penalties: `0.5 * lambda * ||∇_x D(x)||²` per sample.
fn penalty_from_squared_gradient_norm<F>(
    samples: &Tensor,
    discriminator_fn: F,
    lambda: f64,
    reduction: &str,
    context: &'static str,
) -> TorshResult<Tensor>
where
    F: Fn(&Tensor) -> TorshResult<Tensor>,
{
    let dims = samples.shape().dims().to_vec();
    if dims.is_empty() {
        return Err(TorshError::invalid_argument_with_context(
            "samples must have a batch dimension",
            context,
        ));
    }
    let batch_size = dims[0];

    // Real gradient of the discriminator with respect to the samples
    let gradient = input_gradient(samples, &discriminator_fn, context)?;
    let squared_norms = per_sample_squared_norms(&gradient, batch_size)?;

    // 0.5 factor from the R1/R2 paper
    let scale = lambda as f32 * 0.5;
    let penalty_data: Vec<f32> = squared_norms.iter().map(|&sq| scale * sq).collect();
    let penalty = Tensor::from_data(penalty_data, vec![batch_size], samples.device())?;

    apply_reduction(penalty, reduction, context)
}

/// R2 gradient penalty for generator regularization
///
/// Implements R2 regularization for generators, encouraging zero gradients
/// on fake data. Less commonly used than R1 but useful in some settings.
///
/// # Arguments
/// * `fake_samples` - Generated/fake data samples
/// * `discriminator_fn` - Function that computes discriminator output
/// * `lambda` - Gradient penalty coefficient
/// * `reduction` - Reduction method
pub fn r2_gradient_penalty<F>(
    fake_samples: &Tensor,
    discriminator_fn: F,
    lambda: f64,
    reduction: &str,
) -> TorshResult<Tensor>
where
    F: Fn(&Tensor) -> TorshResult<Tensor>,
{
    penalty_from_squared_gradient_norm(
        fake_samples,
        discriminator_fn,
        lambda,
        reduction,
        "r2_gradient_penalty",
    )
}

/// Consistency regularization penalty
///
/// Enforces consistency between network outputs for slightly perturbed inputs.
/// Commonly used in semi-supervised learning and domain adaptation.
///
/// # Arguments
/// * `model_fn` - Function that computes model output
/// * `input` - Original input tensor
/// * `perturbed_input` - Perturbed version of input
/// * `lambda` - Consistency penalty coefficient
/// * `reduction` - Reduction method
pub fn consistency_penalty<F>(
    model_fn: F,
    input: &Tensor,
    perturbed_input: &Tensor,
    lambda: f64,
    reduction: &str,
) -> TorshResult<Tensor>
where
    F: Fn(&Tensor) -> TorshResult<Tensor>,
{
    let output_original = model_fn(input)?;
    let output_perturbed = model_fn(perturbed_input)?;

    // Compute MSE between outputs
    let diff = output_original.sub(&output_perturbed)?;
    let penalty = diff.pow_scalar(2.0)?.mul_scalar(lambda as f32)?;

    match reduction {
        "none" => Ok(penalty),
        "mean" => penalty.mean(None, false),
        "sum" => penalty.sum(),
        _ => Err(TorshError::invalid_argument_with_context(
            &format!(
                "Invalid reduction: {}, expected 'none', 'mean', or 'sum'",
                reduction
            ),
            "consistency_penalty",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random_ops::randn;

    #[test]
    fn power_iteration_recovers_the_spectral_norm() {
        // [[3, 0], [0, 4]] -> singular values 3 and 4
        let diagonal = [3.0, 0.0, 0.0, 4.0];
        let sigma = spectral_norm_power_iteration(&diagonal, 2, 2, 100);
        assert!((sigma - 4.0).abs() < 1e-4, "expected 4.0, got {sigma}");
    }

    #[test]
    fn power_iteration_survives_a_null_space_start() {
        // A uniform probe is annihilated by this matrix, but sigma_max is 2.
        let degenerate = [1.0, -1.0, 1.0, -1.0];
        let sigma = spectral_norm_power_iteration(&degenerate, 2, 2, 100);
        assert!((sigma - 2.0).abs() < 1e-4, "expected 2.0, got {sigma}");
    }

    #[test]
    fn power_iteration_reports_zero_for_a_zero_matrix() {
        let zeros = [0.0; 4];
        assert_eq!(spectral_norm_power_iteration(&zeros, 2, 2, 10), 0.0);
    }

    #[test]
    fn test_gradient_penalty_shapes() {
        let real_samples = randn(&[4, 3, 32, 32], None, None, None).unwrap();
        let fake_samples = randn(&[4, 3, 32, 32], None, None, None).unwrap();

        // Simple discriminator function for testing
        let discriminator_fn = |x: &Tensor| -> TorshResult<Tensor> {
            // Simple linear transformation for testing
            let flattened = x.view(&[x.shape().dims()[0] as i32, -1])?;
            let weight = randn(&[flattened.shape().dims()[1], 1], None, None, None)?;
            flattened.matmul(&weight)
        };

        let penalty =
            gradient_penalty(&real_samples, &fake_samples, discriminator_fn, 10.0, "mean");

        assert!(penalty.is_ok());
        let penalty = penalty.unwrap();
        assert_eq!(penalty.shape().dims(), &[] as &[usize]); // Scalar for mean reduction
    }

    #[test]
    fn test_r1_penalty_shapes() {
        let real_samples = randn(&[4, 3, 32, 32], None, None, None).unwrap();

        let discriminator_fn = |x: &Tensor| -> TorshResult<Tensor> {
            let flattened = x.view(&[x.shape().dims()[0] as i32, -1])?;
            let weight = randn(&[flattened.shape().dims()[1], 1], None, None, None)?;
            flattened.matmul(&weight)
        };

        let penalty = r1_gradient_penalty(&real_samples, discriminator_fn, 10.0, "mean");

        assert!(penalty.is_ok());
        let penalty = penalty.unwrap();
        assert_eq!(penalty.shape().dims(), &[] as &[usize]); // Scalar for mean reduction
    }

    #[test]
    fn test_consistency_penalty() {
        let input = randn(&[4, 10], None, None, None).unwrap();
        let perturbed_input = input.add_scalar(0.1).unwrap();

        let model_fn = |x: &Tensor| -> TorshResult<Tensor> {
            let weight = randn(&[10, 5], None, None, None)?;
            x.matmul(&weight)
        };

        let penalty = consistency_penalty(model_fn, &input, &perturbed_input, 1.0, "mean");

        assert!(penalty.is_ok());
        let penalty = penalty.unwrap();
        assert_eq!(penalty.shape().dims(), &[] as &[usize]); // Scalar for mean reduction
    }
}
