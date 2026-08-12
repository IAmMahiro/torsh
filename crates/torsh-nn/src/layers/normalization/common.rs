//! Common utilities and types for normalization layers
//!
//! This module provides shared functionality used across different normalization
//! implementations including configuration types, utility functions, and common patterns.

use parking_lot::RwLock;
use std::sync::Arc;
use torsh_core::error::Result;
use torsh_tensor::{creation::*, Tensor};

// Conditional imports for std/no_std compatibility

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

/// Configuration for normalization layers
#[derive(Debug, Clone)]
pub struct NormalizationConfig {
    /// Small constant added to variance for numerical stability
    pub eps: f32,
    /// Momentum for running statistics update
    pub momentum: f32,
    /// Whether to use learnable affine parameters (weight and bias)
    pub affine: bool,
    /// Whether to track running statistics for batch norm
    pub track_running_stats: bool,
}

impl Default for NormalizationConfig {
    fn default() -> Self {
        Self {
            eps: 1e-5,
            momentum: 0.1,
            affine: true,
            track_running_stats: true,
        }
    }
}

impl NormalizationConfig {
    /// Create configuration for training mode with tracking
    pub fn training() -> Self {
        Self::default()
    }

    /// Create configuration for inference mode without tracking
    pub fn inference() -> Self {
        Self {
            track_running_stats: false,
            ..Self::default()
        }
    }

    /// Create configuration without learnable parameters
    pub fn non_affine() -> Self {
        Self {
            affine: false,
            ..Self::default()
        }
    }

    /// Create configuration with custom epsilon for numerical stability
    pub fn with_eps(eps: f32) -> Self {
        Self {
            eps,
            ..Self::default()
        }
    }

    /// Create configuration with custom momentum for running stats
    pub fn with_momentum(momentum: f32) -> Self {
        Self {
            momentum,
            ..Self::default()
        }
    }
}

/// Normalization statistics for tracking and analysis
#[derive(Debug, Clone)]
pub struct NormalizationStats {
    pub mean: Tensor,
    pub var: Tensor,
    pub running_mean: Option<Tensor>,
    pub running_var: Option<Tensor>,
    pub num_batches_tracked: Option<Tensor>,
}

impl NormalizationStats {
    /// Create new normalization statistics
    pub fn new(num_features: usize, track_running: bool) -> Result<Self> {
        let mean = zeros(&[num_features])?;
        let var = ones(&[num_features])?;

        let (running_mean, running_var, num_batches_tracked) = if track_running {
            (
                Some(zeros(&[num_features])?),
                Some(ones(&[num_features])?),
                Some(zeros(&[1])?),
            )
        } else {
            (None, None, None)
        };

        Ok(Self {
            mean,
            var,
            running_mean,
            running_var,
            num_batches_tracked,
        })
    }

    /// Update running statistics
    pub fn update_running_stats(
        &mut self,
        batch_mean: &Tensor,
        batch_var: &Tensor,
        momentum: f32,
    ) -> Result<()> {
        if let (Some(ref mut running_mean), Some(ref mut running_var)) =
            (&mut self.running_mean, &mut self.running_var)
        {
            // running_mean = (1 - momentum) * running_mean + momentum * batch_mean
            let one_minus_momentum = 1.0 - momentum;
            *running_mean = running_mean
                .mul_scalar(one_minus_momentum)?
                .add(&batch_mean.mul_scalar(momentum)?)?;

            // running_var = (1 - momentum) * running_var + momentum * batch_var
            *running_var = running_var
                .mul_scalar(one_minus_momentum)?
                .add(&batch_var.mul_scalar(momentum)?)?;

            // Increment batch counter
            if let Some(ref mut num_batches) = self.num_batches_tracked {
                *num_batches = num_batches.add_scalar(1.0)?;
            }
        }
        Ok(())
    }
}

/// Running statistics buffers for batch-normalization layers.
///
/// The buffers live behind `Arc<RwLock<Tensor>>` so a `forward(&self, ...)` can
/// update them, and so the very same handles can be published through
/// [`crate::Module::named_buffers`] — there is exactly one copy of the state,
/// not a registered buffer plus a disconnected shadow.
#[derive(Debug, Clone)]
pub struct RunningStats {
    running_mean: Arc<RwLock<Tensor>>,
    running_var: Arc<RwLock<Tensor>>,
    num_batches_tracked: Arc<RwLock<Tensor>>,
}

impl RunningStats {
    /// Create zero-mean / unit-variance running statistics for `num_features`.
    pub fn new(num_features: usize) -> Result<Self> {
        Ok(Self {
            running_mean: Arc::new(RwLock::new(zeros(&[num_features])?)),
            running_var: Arc::new(RwLock::new(ones(&[num_features])?)),
            num_batches_tracked: Arc::new(RwLock::new(zeros(&[1])?)),
        })
    }

    /// Shared handle to the running mean buffer.
    pub fn running_mean_handle(&self) -> Arc<RwLock<Tensor>> {
        Arc::clone(&self.running_mean)
    }

    /// Shared handle to the running variance buffer.
    pub fn running_var_handle(&self) -> Arc<RwLock<Tensor>> {
        Arc::clone(&self.running_var)
    }

    /// Shared handle to the batch counter buffer.
    pub fn num_batches_tracked_handle(&self) -> Arc<RwLock<Tensor>> {
        Arc::clone(&self.num_batches_tracked)
    }

    /// Snapshot of the current running mean.
    pub fn running_mean(&self) -> Tensor {
        self.running_mean.read().clone()
    }

    /// Snapshot of the current running variance.
    pub fn running_var(&self) -> Tensor {
        self.running_var.read().clone()
    }

    /// Number of batches folded into the running statistics so far.
    pub fn num_batches_tracked(&self) -> Result<f32> {
        let counter = self.num_batches_tracked.read();
        Ok(counter.to_vec()?.first().copied().unwrap_or(0.0))
    }

    /// Fold a batch into the running statistics.
    ///
    /// `batch_var` must be the **unbiased** batch variance, matching PyTorch,
    /// which normalizes with the biased variance but tracks the unbiased one.
    pub fn update(&self, batch_mean: &Tensor, batch_var: &Tensor, momentum: f32) -> Result<()> {
        let one_minus_momentum = 1.0 - momentum;

        {
            let mut running_mean = self.running_mean.write();
            *running_mean = running_mean
                .mul_scalar(one_minus_momentum)?
                .add(&batch_mean.mul_scalar(momentum)?)?;
        }
        {
            let mut running_var = self.running_var.write();
            *running_var = running_var
                .mul_scalar(one_minus_momentum)?
                .add(&batch_var.mul_scalar(momentum)?)?;
        }
        {
            let mut counter = self.num_batches_tracked.write();
            *counter = counter.add_scalar(1.0)?;
        }

        Ok(())
    }
}

/// Convert a biased (population) variance into the unbiased (sample) variance.
///
/// `count` is the number of elements that contributed to each channel
/// statistic. With a single sample the correction is undefined, so the biased
/// value is returned unchanged.
pub fn unbiased_variance(biased: &Tensor, count: usize) -> Result<Tensor> {
    if count > 1 {
        biased.mul_scalar(count as f32 / (count - 1) as f32)
    } else {
        Ok(biased.clone())
    }
}

/// Common utility functions for normalization implementations
pub mod utils {
    use super::*;

    /// Manually compute channel-wise mean for batch normalization
    /// This is a fallback when mean_dim operations are not available
    pub fn compute_channel_mean(input: &Tensor) -> Result<Tensor> {
        let input_shape = input.shape();
        let dims = input_shape.dims();

        match dims.len() {
            2 => compute_channel_mean_1d(input, dims),
            4 => compute_channel_mean_2d(input, dims),
            5 => compute_channel_mean_3d(input, dims),
            _ => Err(torsh_core::error::TorshError::InvalidShape(format!(
                "Unsupported input dimensions: {}",
                dims.len()
            ))),
        }
    }

    /// Compute channel-wise variance for batch normalization
    pub fn compute_channel_variance(input: &Tensor, mean: &Tensor) -> Result<Tensor> {
        let input_shape = input.shape();
        let dims = input_shape.dims();

        match dims.len() {
            2 => compute_channel_var_1d(input, mean, dims),
            4 => compute_channel_var_2d(input, mean, dims),
            5 => compute_channel_var_3d(input, mean, dims),
            _ => Err(torsh_core::error::TorshError::InvalidShape(format!(
                "Unsupported input dimensions: {}",
                dims.len()
            ))),
        }
    }

    fn compute_channel_mean_1d(input: &Tensor, dims: &[usize]) -> Result<Tensor> {
        let batch_size = dims[0];
        let channels = dims[1];

        let input_data = input.to_vec()?;
        let mut channel_means = vec![0.0f32; channels];

        for batch in 0..batch_size {
            for c in 0..channels {
                let idx = batch * channels + c;
                channel_means[c] += input_data[idx];
            }
        }

        for mean in &mut channel_means {
            *mean /= batch_size as f32;
        }

        Tensor::from_data(channel_means, vec![channels], input.device())
    }

    fn compute_channel_mean_2d(input: &Tensor, dims: &[usize]) -> Result<Tensor> {
        let batch_size = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];

        let input_data = input.to_vec()?;
        let mut channel_means = vec![0.0f32; channels];
        let elements_per_channel = (batch_size * height * width) as f32;

        for batch in 0..batch_size {
            for c in 0..channels {
                for h in 0..height {
                    for w in 0..width {
                        let idx = batch * (channels * height * width)
                            + c * (height * width)
                            + h * width
                            + w;
                        channel_means[c] += input_data[idx];
                    }
                }
            }
        }

        for mean in &mut channel_means {
            *mean /= elements_per_channel;
        }

        Tensor::from_data(channel_means, vec![channels], input.device())
    }

    fn compute_channel_mean_3d(input: &Tensor, dims: &[usize]) -> Result<Tensor> {
        let batch_size = dims[0];
        let channels = dims[1];
        let depth = dims[2];
        let height = dims[3];
        let width = dims[4];

        let input_data = input.to_vec()?;
        let mut channel_means = vec![0.0f32; channels];
        let elements_per_channel = (batch_size * depth * height * width) as f32;

        for batch in 0..batch_size {
            for c in 0..channels {
                for d in 0..depth {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = batch * (channels * depth * height * width)
                                + c * (depth * height * width)
                                + d * (height * width)
                                + h * width
                                + w;
                            channel_means[c] += input_data[idx];
                        }
                    }
                }
            }
        }

        for mean in &mut channel_means {
            *mean /= elements_per_channel;
        }

        Tensor::from_data(channel_means, vec![channels], input.device())
    }

    fn compute_channel_var_1d(input: &Tensor, mean: &Tensor, dims: &[usize]) -> Result<Tensor> {
        let batch_size = dims[0];
        let channels = dims[1];

        let input_data = input.to_vec()?;
        let mean_data = mean.to_vec()?;
        let mut channel_vars = vec![0.0f32; channels];

        for batch in 0..batch_size {
            for c in 0..channels {
                let idx = batch * channels + c;
                let diff = input_data[idx] - mean_data[c];
                channel_vars[c] += diff * diff;
            }
        }

        for var in &mut channel_vars {
            *var /= batch_size as f32;
        }

        Tensor::from_data(channel_vars, vec![channels], input.device())
    }

    fn compute_channel_var_2d(input: &Tensor, mean: &Tensor, dims: &[usize]) -> Result<Tensor> {
        let batch_size = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];

        let input_data = input.to_vec()?;
        let mean_data = mean.to_vec()?;
        let mut channel_vars = vec![0.0f32; channels];
        let elements_per_channel = (batch_size * height * width) as f32;

        for batch in 0..batch_size {
            for c in 0..channels {
                for h in 0..height {
                    for w in 0..width {
                        let idx = batch * (channels * height * width)
                            + c * (height * width)
                            + h * width
                            + w;
                        let diff = input_data[idx] - mean_data[c];
                        channel_vars[c] += diff * diff;
                    }
                }
            }
        }

        for var in &mut channel_vars {
            *var /= elements_per_channel;
        }

        Tensor::from_data(channel_vars, vec![channels], input.device())
    }

    fn compute_channel_var_3d(input: &Tensor, mean: &Tensor, dims: &[usize]) -> Result<Tensor> {
        let batch_size = dims[0];
        let channels = dims[1];
        let depth = dims[2];
        let height = dims[3];
        let width = dims[4];

        let input_data = input.to_vec()?;
        let mean_data = mean.to_vec()?;
        let mut channel_vars = vec![0.0f32; channels];
        let elements_per_channel = (batch_size * depth * height * width) as f32;

        for batch in 0..batch_size {
            for c in 0..channels {
                for d in 0..depth {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = batch * (channels * depth * height * width)
                                + c * (depth * height * width)
                                + d * (height * width)
                                + h * width
                                + w;
                            let diff = input_data[idx] - mean_data[c];
                            channel_vars[c] += diff * diff;
                        }
                    }
                }
            }
        }

        for var in &mut channel_vars {
            *var /= elements_per_channel;
        }

        Tensor::from_data(channel_vars, vec![channels], input.device())
    }

    /// Broadcast shape that lines a per-channel vector up with an `NC...` tensor.
    fn channel_broadcast_shape(rank: usize, channels: usize) -> Vec<i32> {
        let mut shape = vec![1i32; rank];
        if rank >= 2 {
            shape[1] = channels as i32;
        }
        shape
    }

    /// Apply a *per-channel* normalization: `(x - mean) / sqrt(var + eps) * weight + bias`.
    ///
    /// `mean`, `var`, `weight` and `bias` are 1-D tensors of length `C` and are
    /// explicitly reshaped to `[1, C, 1, ...]` before broadcasting. This is what
    /// distinguishes it from [`apply_normalization`], whose generic trailing-axis
    /// broadcast silently aligns `C` with the *last* axis whenever the two happen
    /// to have the same length (e.g. an `[N, 2, 2, 2]` input).
    pub fn apply_channel_normalization(
        input: &Tensor,
        mean: &Tensor,
        var: &Tensor,
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor> {
        let input_shape = input.shape();
        let dims = input_shape.dims();

        if dims.len() < 2 {
            return Err(torsh_core::error::TorshError::InvalidShape(format!(
                "channel normalization expects an (N, C, ...) tensor, got {dims:?}"
            )));
        }

        let channels = dims[1];
        let broadcast = channel_broadcast_shape(dims.len(), channels);

        let inv_std = var.add_scalar(eps)?.sqrt()?;
        let centered = input.sub(&mean.reshape(&broadcast)?)?;
        let mut normalized = centered.div(&inv_std.reshape(&broadcast)?)?;

        if let Some(w) = weight {
            normalized = normalized.mul(&w.reshape(&broadcast)?)?;
        }
        if let Some(b) = bias {
            normalized = normalized.add(&b.reshape(&broadcast)?)?;
        }

        Ok(normalized)
    }

    /// Apply a per-channel affine map `x * scale + shift`.
    ///
    /// Used by batch renormalization, where the correction terms `r`/`d` fold
    /// into a single scale/shift pair per channel.
    pub fn apply_channel_affine(input: &Tensor, scale: &[f32], shift: &[f32]) -> Result<Tensor> {
        let input_shape = input.shape();
        let dims = input_shape.dims();

        if dims.len() < 2 {
            return Err(torsh_core::error::TorshError::InvalidShape(format!(
                "channel affine expects an (N, C, ...) tensor, got {dims:?}"
            )));
        }

        let channels = dims[1];
        if scale.len() != channels || shift.len() != channels {
            return Err(torsh_core::error::TorshError::InvalidShape(format!(
                "expected {channels} scale/shift values, got {} and {}",
                scale.len(),
                shift.len()
            )));
        }

        let broadcast: Vec<usize> = channel_broadcast_shape(dims.len(), channels)
            .into_iter()
            .map(|d| d as usize)
            .collect();
        let scale_tensor = Tensor::from_data(scale.to_vec(), broadcast.clone(), input.device())?;
        let shift_tensor = Tensor::from_data(shift.to_vec(), broadcast, input.device())?;

        input.mul(&scale_tensor)?.add(&shift_tensor)
    }

    /// Apply normalization transformation: (x - mean) / sqrt(var + eps) * weight + bias
    pub fn apply_normalization(
        input: &Tensor,
        mean: &Tensor,
        var: &Tensor,
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor> {
        // Try to use the tensor's built-in broadcasting first
        // If that fails, we can implement manual broadcasting
        match try_apply_normalization_simple(input, mean, var, weight, bias, eps) {
            Ok(result) => Ok(result),
            Err(_) => {
                // Fall back to manual broadcasting if simple approach fails
                apply_normalization_with_broadcasting(input, mean, var, weight, bias, eps)
            }
        }
    }

    /// Simple approach that relies on built-in broadcasting
    fn try_apply_normalization_simple(
        input: &Tensor,
        mean: &Tensor,
        var: &Tensor,
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor> {
        // Subtract mean
        let centered = input.sub(mean)?;

        // Compute standard deviation
        let std = var.add_scalar(eps)?.sqrt()?;

        // Normalize
        let mut normalized = centered.div(&std)?;

        // Apply learnable parameters if provided
        if let Some(w) = weight {
            normalized = normalized.mul(w)?;
        }

        if let Some(b) = bias {
            normalized = normalized.add(b)?;
        }

        Ok(normalized)
    }

    /// Manual broadcasting approach for when simple broadcasting doesn't work
    fn apply_normalization_with_broadcasting(
        input: &Tensor,
        mean: &Tensor,
        var: &Tensor,
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor> {
        let input_shape = input.shape();
        let input_dims = input_shape.dims();
        let mean_shape = mean.shape();
        let mean_dims = mean_shape.dims();

        // For broadcasting: if mean/var are 1D [C] and input is 4D [N,C,H,W],
        // we need to make mean/var into [1,C,1,1] for proper broadcasting
        let (broadcast_mean, broadcast_var) = if input_dims.len() == 4 && mean_dims.len() == 1 {
            let channels = mean_dims[0];
            let mean_broadcast = mean.reshape(&[1i32, channels as i32, 1i32, 1i32])?;
            let var_broadcast = var.reshape(&[1i32, channels as i32, 1i32, 1i32])?;
            (mean_broadcast, var_broadcast)
        } else if input_dims.len() == 2 && mean_dims.len() == 1 {
            let channels = mean_dims[0];
            let mean_broadcast = mean.reshape(&[1i32, channels as i32])?;
            let var_broadcast = var.reshape(&[1i32, channels as i32])?;
            (mean_broadcast, var_broadcast)
        } else {
            // Already compatible shapes
            (mean.clone(), var.clone())
        };

        // Subtract mean
        let centered = input.sub(&broadcast_mean)?;

        // Compute standard deviation
        let std = broadcast_var.add_scalar(eps)?.sqrt()?;

        // Normalize
        let mut normalized = centered.div(&std)?;

        // Apply learnable parameters if provided with proper broadcasting
        if let Some(w) = weight {
            let weight_shape = w.shape();
            let weight_dims = weight_shape.dims();
            let broadcast_weight = if input_dims.len() == 4 && weight_dims.len() == 1 {
                let channels = weight_dims[0];
                w.reshape(&[1i32, channels as i32, 1i32, 1i32])?
            } else if input_dims.len() == 2 && weight_dims.len() == 1 {
                let channels = weight_dims[0];
                w.reshape(&[1i32, channels as i32])?
            } else {
                w.clone()
            };
            normalized = normalized.mul(&broadcast_weight)?;
        }

        if let Some(b) = bias {
            let bias_shape = b.shape();
            let bias_dims = bias_shape.dims();
            let broadcast_bias = if input_dims.len() == 4 && bias_dims.len() == 1 {
                let channels = bias_dims[0];
                b.reshape(&[1i32, channels as i32, 1i32, 1i32])?
            } else if input_dims.len() == 2 && bias_dims.len() == 1 {
                let channels = bias_dims[0];
                b.reshape(&[1i32, channels as i32])?
            } else {
                b.clone()
            };
            normalized = normalized.add(&broadcast_bias)?;
        }

        Ok(normalized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalization_config() {
        let config = NormalizationConfig::default();
        assert_eq!(config.eps, 1e-5);
        assert_eq!(config.momentum, 0.1);
        assert!(config.affine);
        assert!(config.track_running_stats);

        let inference_config = NormalizationConfig::inference();
        assert!(!inference_config.track_running_stats);

        let non_affine_config = NormalizationConfig::non_affine();
        assert!(!non_affine_config.affine);
    }

    #[test]
    fn test_normalization_stats_creation() {
        let stats = NormalizationStats::new(10, true).expect("Normalization Stats should succeed");
        assert!(stats.running_mean.is_some());
        assert!(stats.running_var.is_some());
        assert!(stats.num_batches_tracked.is_some());

        let stats_no_tracking =
            NormalizationStats::new(10, false).expect("Normalization Stats should succeed");
        assert!(stats_no_tracking.running_mean.is_none());
        assert!(stats_no_tracking.running_var.is_none());
        assert!(stats_no_tracking.num_batches_tracked.is_none());
    }

    #[test]
    fn test_channel_mean_computation() {
        // Test 2D case (batch_size=2, channels=3)
        let input = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            torsh_core::device::DeviceType::Cpu,
        )
        .expect("operation should succeed");
        let mean = utils::compute_channel_mean(&input).expect("utils should succeed");
        let expected_mean = vec![2.5, 3.5, 4.5]; // Channel-wise means
        let mean_data = mean
            .to_vec()
            .expect("tensor to vec conversion should succeed");

        for (i, &expected) in expected_mean.iter().enumerate() {
            assert!((mean_data[i] - expected).abs() < 1e-6);
        }
    }
}
