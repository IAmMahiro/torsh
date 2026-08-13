//! Real training engine backed by torsh-nn / torsh-optim / torsh-autograd.
//!
//! This module performs *genuine* optimisation: it builds a real neural network
//! from [`torsh_nn`], runs real forward passes, computes a real loss, back-props
//! through [`torsh_autograd`], and updates parameters with a real
//! [`torsh_optim`] optimiser. Nothing here fabricates losses or gradients.
//!
//! It intentionally supports a bounded, honest surface (a multi-layer perceptron
//! trained on an explicitly-synthetic regression task). Callers that need a real
//! dataset or a real architecture that this engine cannot build must surface an
//! honest error rather than fall back to fabricated numbers.

use anyhow::{anyhow, Result};

use torsh::core::device::DeviceType;
use torsh::nn::container::Sequential;
use torsh::nn::layers::{Linear, ReLU};
use torsh::nn::Module;
use torsh::optim::sgd::SGD;
use torsh::optim::Optimizer;
use torsh::tensor::Tensor;

/// Shape of the multi-layer perceptron built by [`build_mlp`].
#[derive(Debug, Clone, Copy)]
pub struct MlpConfig {
    /// Number of input features.
    pub input_dim: usize,
    /// Width of the single hidden layer.
    pub hidden_dim: usize,
    /// Number of regression outputs.
    pub output_dim: usize,
}

/// An in-memory regression dataset made of real tensors.
///
/// Both tensors live on the CPU. `inputs` has shape `[n, input_dim]` and
/// `targets` has shape `[n, output_dim]`.
#[derive(Debug)]
pub struct RegressionData {
    /// Feature matrix, shape `[n, input_dim]`.
    pub inputs: Tensor,
    /// Target matrix, shape `[n, output_dim]`.
    pub targets: Tensor,
    /// Number of samples.
    pub n: usize,
    /// Input dimensionality.
    pub input_dim: usize,
    /// Output dimensionality.
    pub output_dim: usize,
}

/// Build a real MLP (`Linear -> ReLU -> Linear`) with the requested shape.
pub fn build_mlp(cfg: &MlpConfig) -> Result<Sequential> {
    if cfg.input_dim == 0 || cfg.hidden_dim == 0 || cfg.output_dim == 0 {
        return Err(anyhow!(
            "MLP dimensions must all be non-zero (got input={}, hidden={}, output={})",
            cfg.input_dim,
            cfg.hidden_dim,
            cfg.output_dim
        ));
    }
    let model = Sequential::new()
        .add(Linear::new(cfg.input_dim, cfg.hidden_dim, true))
        .add(ReLU::new())
        .add(Linear::new(cfg.hidden_dim, cfg.output_dim, true));
    Ok(model)
}

/// Deterministically generate an **explicitly synthetic** regression dataset.
///
/// Targets are a fixed affine function of the inputs plus a small deterministic
/// perturbation, so a correctly-wired trainer must be able to drive the loss
/// down. The data is generated from a simple LCG seeded by `seed` — it is *not*
/// read from disk and callers must label it as synthetic in any user output.
pub fn synthetic_regression(
    n: usize,
    input_dim: usize,
    output_dim: usize,
    seed: u64,
) -> Result<RegressionData> {
    if n == 0 || input_dim == 0 || output_dim == 0 {
        return Err(anyhow!("synthetic dataset dimensions must be non-zero"));
    }

    // Simple deterministic LCG (numerical-recipes constants) for reproducibility
    // without pulling randomness into the training result.
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut next_unit = || -> f32 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Top 24 bits -> [0, 1)
        ((state >> 40) as f32) / ((1u64 << 24) as f32)
    };

    // Fixed ground-truth weights so the mapping is learnable.
    let true_w: Vec<f32> = (0..input_dim * output_dim)
        .map(|k| 0.5 - ((k % 7) as f32) * 0.1)
        .collect();
    let true_b: Vec<f32> = (0..output_dim).map(|j| 0.05 * (j as f32 + 1.0)).collect();

    let mut inputs = Vec::with_capacity(n * input_dim);
    let mut targets = Vec::with_capacity(n * output_dim);

    for _ in 0..n {
        let row: Vec<f32> = (0..input_dim).map(|_| next_unit() - 0.5).collect();
        for j in 0..output_dim {
            let mut acc = true_b[j];
            for (i, &x) in row.iter().enumerate() {
                acc += x * true_w[i * output_dim + j];
            }
            // Small deterministic perturbation.
            acc += (next_unit() - 0.5) * 0.01;
            targets.push(acc);
        }
        inputs.extend_from_slice(&row);
    }

    let inputs = Tensor::from_data(inputs, vec![n, input_dim], DeviceType::Cpu)?;
    let targets = Tensor::from_data(targets, vec![n, output_dim], DeviceType::Cpu)?;

    Ok(RegressionData {
        inputs,
        targets,
        n,
        input_dim,
        output_dim,
    })
}

/// Compute the real mean-squared-error loss tensor between `pred` and `target`.
///
/// Returns a scalar tensor still attached to the autograd graph so that
/// [`Tensor::backward`] populates real parameter gradients.
pub fn mse_loss(pred: &Tensor, target: &Tensor, count: usize) -> Result<Tensor> {
    let diff = pred.sub(target)?;
    let sq = diff.mul(&diff)?;
    let sse = sq.sum()?;
    Ok(sse.mul_scalar(1.0 / count as f32)?)
}

/// Run one real forward pass and return the scalar MSE loss value.
pub fn evaluate_loss(model: &Sequential, data: &RegressionData) -> Result<f64> {
    let pred = model.forward(&data.inputs)?;
    let loss = mse_loss(&pred, &data.targets, data.n * data.output_dim)?;
    let value = loss
        .to_vec()?
        .first()
        .copied()
        .ok_or_else(|| anyhow!("loss tensor was empty"))?;
    Ok(value as f64)
}

/// Train `model` on `data` for `steps` full-batch SGD steps.
///
/// Returns the real, measured loss after every step (length `steps`). This
/// performs genuine backpropagation and in-place parameter updates.
pub fn train_regression(
    model: &Sequential,
    data: &RegressionData,
    learning_rate: f32,
    steps: usize,
) -> Result<Vec<f64>> {
    if data.input_dim == 0 || data.output_dim == 0 {
        return Err(anyhow!("dataset has zero-sized dimensions"));
    }
    let params: Vec<_> = model.parameters().values().map(|p| p.tensor()).collect();
    if params.is_empty() {
        return Err(anyhow!("model exposes no trainable parameters"));
    }

    let mut optimizer = SGD::new(params, learning_rate, Some(0.9), None, None, false);
    let denom = data.n * data.output_dim;
    let mut history = Vec::with_capacity(steps);

    for _ in 0..steps {
        optimizer.zero_grad();
        let pred = model.forward(&data.inputs)?;
        let loss = mse_loss(&pred, &data.targets, denom)?;
        loss.backward()?;
        optimizer
            .step()
            .map_err(|e| anyhow!("optimizer step failed: {e}"))?;

        let value = loss
            .to_vec()?
            .first()
            .copied()
            .ok_or_else(|| anyhow!("loss tensor was empty"))? as f64;
        history.push(value);
    }

    Ok(history)
}
