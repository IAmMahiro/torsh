//! Layer and Group normalization implementations
//!
//! This module provides normalization techniques that operate on different dimensions:
//! - Layer normalization: normalizes across the feature dimension
//! - Group normalization: normalizes across grouped features

use crate::{Module, ModuleBase, Parameter};
use torsh_core::device::DeviceType;
use torsh_core::error::Result;
use torsh_tensor::{creation::*, Tensor};

use super::common::{utils, NormalizationConfig};

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

/// Layer normalization
pub struct LayerNorm {
    base: ModuleBase,
    normalized_shape: Vec<usize>,
    config: NormalizationConfig,
}

impl LayerNorm {
    pub fn new(normalized_shape: Vec<usize>) -> Result<Self> {
        Self::with_config(normalized_shape, NormalizationConfig::default())
    }

    pub fn with_config(normalized_shape: Vec<usize>, config: NormalizationConfig) -> Result<Self> {
        let mut base = ModuleBase::new();

        // Initialize parameters if affine
        if config.affine {
            let weight = ones(&normalized_shape)?;
            let bias = zeros(&normalized_shape)?;
            base.register_parameter("weight".to_string(), Parameter::new(weight));
            base.register_parameter("bias".to_string(), Parameter::new(bias));
        }

        Ok(Self {
            base,
            normalized_shape,
            config,
        })
    }

    pub fn normalized_shape(&self) -> &[usize] {
        &self.normalized_shape
    }

    pub fn eps(&self) -> f32 {
        self.config.eps
    }

    /// Axes the layer normalizes over: the trailing `normalized_shape.len()`
    /// dimensions, validated against `normalized_shape`.
    fn normalized_axes(&self, input: &Tensor) -> Result<Vec<usize>> {
        let input_shape = input.shape();
        let dims = input_shape.dims();

        let normalized_dims = self.normalized_shape.len();
        let input_dims = dims.len();

        if input_dims < normalized_dims {
            return Err(torsh_core::error::TorshError::InvalidShape(format!(
                "Input has {} dims but normalized_shape has {} dims",
                input_dims, normalized_dims
            )));
        }

        let start_idx = input_dims - normalized_dims;
        for (i, &norm_dim) in self.normalized_shape.iter().enumerate() {
            if dims[start_idx + i] != norm_dim {
                return Err(torsh_core::error::TorshError::InvalidShape(format!(
                    "Expected dimension {} to be {}, got {}",
                    start_idx + i,
                    norm_dim,
                    dims[start_idx + i]
                )));
            }
        }

        Ok((start_idx..input_dims).collect())
    }

    /// Per-sample mean and *biased* variance over the normalized axes, kept on
    /// the autograd graph.
    ///
    /// Both come back with `keepdim`, i.e. shaped `[..batch.., 1, ..., 1]`,
    /// which is exactly what the pre-rewrite `to_vec()` loop reshaped its
    /// detached leaves to — the difference is only that these carry a backward
    /// rule, so `d loss / d input` now includes the mean-subtraction and
    /// variance terms of the true layer-norm Jacobian.
    fn compute_layer_stats(&self, input: &Tensor) -> Result<(Tensor, Tensor)> {
        let axes = self.normalized_axes(input)?;
        let mean = input.mean(Some(&axes), true)?;
        let centered = input.sub(&mean)?;
        let variance = centered.pow_scalar(2.0)?.mean(Some(&axes), true)?;
        Ok((mean, variance))
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Compute layer statistics
        let (mean, var) = self.compute_layer_stats(input)?;

        // Get learnable parameters
        let weight = if self.config.affine {
            self.base.parameters.get("weight")
        } else {
            None
        };

        let bias = if self.config.affine {
            self.base.parameters.get("bias")
        } else {
            None
        };

        // `weight`/`bias` have exactly `normalized_shape`, so trailing-axis
        // broadcasting lines them up with the normalized axes by construction.
        let weight_tensor = weight.as_ref().map(|p| p.tensor().read().clone());
        let bias_tensor = bias.as_ref().map(|p| p.tensor().read().clone());

        let std = var.add_scalar(self.config.eps)?.sqrt()?;
        let mut normalized = input.sub(&mean)?.div(&std)?;

        if let Some(w) = weight_tensor.as_ref() {
            normalized = normalized.mul(w)?;
        }
        if let Some(b) = bias_tensor.as_ref() {
            normalized = normalized.add(b)?;
        }

        Ok(normalized)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }
}

/// Group normalization
pub struct GroupNorm {
    base: ModuleBase,
    num_groups: usize,
    num_channels: usize,
    config: NormalizationConfig,
}

impl GroupNorm {
    pub fn new(num_groups: usize, num_channels: usize) -> Result<Self> {
        Self::with_config(num_groups, num_channels, NormalizationConfig::default())
    }

    pub fn with_config(
        num_groups: usize,
        num_channels: usize,
        config: NormalizationConfig,
    ) -> Result<Self> {
        if num_channels % num_groups != 0 {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "num_channels ({}) must be divisible by num_groups ({})",
                num_channels, num_groups
            )));
        }

        let mut base = ModuleBase::new();

        // Initialize parameters if affine
        if config.affine {
            let weight = ones(&[num_channels])?;
            let bias = zeros(&[num_channels])?;
            base.register_parameter("weight".to_string(), Parameter::new(weight));
            base.register_parameter("bias".to_string(), Parameter::new(bias));
        }

        Ok(Self {
            base,
            num_groups,
            num_channels,
            config,
        })
    }

    pub fn num_groups(&self) -> usize {
        self.num_groups
    }

    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    pub fn eps(&self) -> f32 {
        self.config.eps
    }

    /// Validate the input's rank and channel count.
    fn validate(&self, input: &Tensor) -> Result<()> {
        let input_shape = input.shape();
        let dims = input_shape.dims();

        if dims.len() < 2 {
            return Err(torsh_core::error::TorshError::InvalidShape(format!(
                "GroupNorm expects at least 2D input, got {:?}",
                dims
            )));
        }

        if dims[1] != self.num_channels {
            return Err(torsh_core::error::TorshError::InvalidShape(format!(
                "Expected {} channels, got {}",
                self.num_channels, dims[1]
            )));
        }

        Ok(())
    }
}

impl Module for GroupNorm {
    /// Normalize each `(sample, group)` slice over its channels and spatial
    /// extent.
    ///
    /// The statistics used to come from a `to_vec()` loop, i.e. from detached
    /// leaves, so `backward()` saw a constant mean and variance. Here the whole
    /// expression is a recording composition evaluated in a
    /// `[N * G, (C / G) * spatial]` view, which is the same arithmetic the loop
    /// performed and gives the real Jacobian.
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.validate(input)?;

        let input_shape = input.shape();
        let dims = input_shape.dims();
        let batch_size = dims[0];
        let channels_per_group = self.num_channels / self.num_groups;
        let spatial_size: usize = if dims.len() > 2 {
            dims[2..].iter().product()
        } else {
            1
        };

        let grouped = input.reshape(&[
            (batch_size * self.num_groups) as i32,
            (channels_per_group * spatial_size) as i32,
        ])?;
        let mean = grouped.mean(Some(&[1]), true)?;
        let centered = grouped.sub(&mean)?;
        let variance = centered.pow_scalar(2.0)?.mean(Some(&[1]), true)?;
        let std = variance.add_scalar(self.config.eps)?.sqrt()?;

        let original: Vec<i32> = dims.iter().map(|&d| d as i32).collect();
        let mut normalized = centered.div(&std)?.reshape(&original)?;

        // Per-channel affine parameters, reshaped to `[1, C, 1, ...]` so they
        // can never be aligned with the trailing axis by accident.
        let broadcast = utils::channel_broadcast_shape(dims.len(), self.num_channels);
        if self.config.affine {
            if let Some(w) = self.base.parameters.get("weight") {
                let weight = w.tensor().read().reshape(&broadcast)?;
                normalized = normalized.mul(&weight)?;
            }
            if let Some(b) = self.base.parameters.get("bias") {
                let bias = b.tensor().read().reshape(&broadcast)?;
                normalized = normalized.add(&bias)?;
            }
        }

        Ok(normalized)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }
}

/// Root Mean Square Layer Normalization (RMSNorm)
///
/// RMSNorm is a simpler and faster alternative to LayerNorm that only normalizes
/// by the root mean square (without mean centering). Used in modern transformers
/// like LLaMA, Gopher, and others.
///
/// # Mathematical Formulation
///
/// ```text
/// RMS(x) = sqrt(mean(x^2) + eps)
/// y = (x / RMS(x)) * weight
/// ```
///
/// # Performance Benefits
///
/// - Faster than LayerNorm (no mean calculation/subtraction)
/// - Simpler gradient computation
/// - Similar or better performance in many applications
///
/// # PyTorch Compatibility
///
/// Compatible with PyTorch's RMSNorm implementations from various libraries.
/// Weight parameter is applied after normalization.
///
/// # Examples
///
/// ```rust
/// # use torsh_nn::layers::normalization::RMSNorm;
/// # use torsh_nn::Module;
/// # use torsh_tensor::creation::randn;
/// # use torsh_core::error::Result;
/// # fn main() -> Result<()> {
/// // Create RMSNorm for transformer hidden dimension
/// let rms_norm = RMSNorm::new(vec![768])?;
///
/// // Apply to transformer hidden states [batch, seq_len, hidden_dim]
/// let hidden_states = randn(&[2, 128, 768])?;
/// let normalized = rms_norm.forward(&hidden_states)?;
/// # Ok(())
/// # }
/// ```
pub struct RMSNorm {
    base: ModuleBase,
    normalized_shape: Vec<usize>,
    eps: f32,
    affine: bool,
}

impl RMSNorm {
    /// Create a new RMSNorm layer with default epsilon (1e-6)
    ///
    /// # Arguments
    ///
    /// * `normalized_shape` - Shape of the features to normalize over
    ///
    /// # Returns
    ///
    /// Result containing the layer or an error
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use torsh_nn::layers::normalization::RMSNorm;
    /// # use torsh_core::error::Result;
    /// # fn main() -> Result<()> {
    /// // For transformer models (normalize last dimension)
    /// let rms_norm = RMSNorm::new(vec![768])?;
    ///
    /// // For multi-dimensional normalization
    /// let rms_norm_2d = RMSNorm::new(vec![64, 64])?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(normalized_shape: Vec<usize>) -> Result<Self> {
        Self::with_config(normalized_shape, 1e-6, true)
    }

    /// Create RMSNorm with custom configuration
    ///
    /// # Arguments
    ///
    /// * `normalized_shape` - Shape of features to normalize
    /// * `eps` - Small constant for numerical stability (default: 1e-6)
    /// * `affine` - Whether to include learnable scale parameter (default: true)
    ///
    /// # Returns
    ///
    /// Result containing the configured layer
    pub fn with_config(normalized_shape: Vec<usize>, eps: f32, affine: bool) -> Result<Self> {
        let mut base = ModuleBase::new();

        // Initialize scale parameter if affine
        if affine {
            let weight = ones(&normalized_shape)?;
            base.register_parameter("weight".to_string(), Parameter::new(weight));
        }

        Ok(Self {
            base,
            normalized_shape,
            eps,
            affine,
        })
    }

    /// Get the normalized shape
    pub fn normalized_shape(&self) -> &[usize] {
        &self.normalized_shape
    }

    /// Get the epsilon value
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Get whether affine transformation is enabled
    pub fn affine(&self) -> bool {
        self.affine
    }

    /// Compute RMS (Root Mean Square) for the input.
    ///
    /// `sqrt(mean(x²) + eps)` as a recording composition: the pre-rewrite
    /// `to_vec()` loop handed back a detached leaf, so `x / RMS(x)` was
    /// differentiated as if `RMS(x)` were a constant and the `-x·(x/RMS³)/n`
    /// term of the true Jacobian went missing.
    fn compute_rms(&self, input: &Tensor) -> Result<Tensor> {
        let input_shape = input.shape();
        let dims = input_shape.dims();

        // Determine which dimensions to normalize over
        let normalized_dims = self.normalized_shape.len();
        let input_dims = dims.len();

        if input_dims < normalized_dims {
            return Err(torsh_core::error::TorshError::InvalidShape(format!(
                "Input has {} dims but normalized_shape has {} dims",
                input_dims, normalized_dims
            )));
        }

        // Check that the last dimensions match normalized_shape
        let start_idx = input_dims - normalized_dims;
        for (i, &norm_dim) in self.normalized_shape.iter().enumerate() {
            if dims[start_idx + i] != norm_dim {
                return Err(torsh_core::error::TorshError::InvalidShape(format!(
                    "Expected dimension {} to be {}, got {}",
                    start_idx + i,
                    norm_dim,
                    dims[start_idx + i]
                )));
            }
        }

        let axes: Vec<usize> = (start_idx..input_dims).collect();
        input
            .pow_scalar(2.0)?
            .mean(Some(&axes), true)?
            .add_scalar(self.eps)?
            .sqrt()
    }
}

impl Module for RMSNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Compute RMS
        let rms = self.compute_rms(input)?;

        // Normalize: x / RMS(x)
        let normalized = input.div(&rms)?;

        // Apply learnable scale if affine
        if self.affine {
            if let Some(weight) = self.base.parameters.get("weight") {
                let weight_tensor = weight.tensor().read().clone();
                normalized.mul(&weight_tensor)
            } else {
                Ok(normalized)
            }
        } else {
            Ok(normalized)
        }
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }
}

// Re-export the layer and group normalization components (already defined in this module)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm_creation() {
        let layer_norm = LayerNorm::new(vec![128]).expect("Layer Norm should succeed");
        assert_eq!(layer_norm.normalized_shape(), &[128]);
        assert_eq!(layer_norm.eps(), 1e-5);

        let layer_norm_2d = LayerNorm::new(vec![64, 64]).expect("Layer Norm should succeed");
        assert_eq!(layer_norm_2d.normalized_shape(), &[64, 64]);
    }

    #[test]
    fn test_group_norm_creation() {
        let group_norm = GroupNorm::new(8, 32).expect("Group Norm should succeed");
        assert_eq!(group_norm.num_groups(), 8);
        assert_eq!(group_norm.num_channels(), 32);
        assert_eq!(group_norm.eps(), 1e-5);

        // Test invalid configuration
        assert!(GroupNorm::new(8, 30).is_err()); // 30 not divisible by 8
    }

    #[test]
    fn test_group_norm_shape_validation() {
        let group_norm = GroupNorm::new(4, 8).expect("Group Norm should succeed");

        // Valid input
        let input = zeros(&[2, 8, 16, 16]).expect("zeros should succeed");
        assert!(group_norm.forward(&input).is_ok());

        // Wrong number of channels
        let input_wrong_channels = zeros(&[2, 16, 16, 16]).expect("zeros should succeed");
        assert!(group_norm.forward(&input_wrong_channels).is_err());
    }

    #[test]
    fn test_rms_norm_creation() {
        let rms_norm = RMSNorm::new(vec![768]).expect("RMSNorm should succeed");
        assert_eq!(rms_norm.normalized_shape(), &[768]);
        assert_eq!(rms_norm.eps(), 1e-6);
        assert!(rms_norm.affine());

        // Non-affine variant
        let rms_norm_no_affine =
            RMSNorm::with_config(vec![512], 1e-8, false).expect("RMSNorm should succeed");
        assert!(!rms_norm_no_affine.affine());
        assert_eq!(rms_norm_no_affine.eps(), 1e-8);
    }

    #[test]
    fn test_rms_norm_forward() {
        use torsh_tensor::creation::ones;

        let rms_norm = RMSNorm::new(vec![4]).expect("RMSNorm should succeed");

        // Test with ones - RMS of ones is 1.0
        let input = ones(&[2, 4]).expect("ones should succeed");
        let output = rms_norm.forward(&input);
        assert!(output.is_ok(), "RMSNorm forward failed: {:?}", output.err());

        if let Ok(result) = output {
            let result_shape = result.shape();
            assert_eq!(result_shape.dims(), &[2, 4]);
        }
    }

    #[test]
    fn test_rms_norm_3d_input() {
        use torsh_tensor::creation::randn;

        // Typical transformer use case: [batch, seq_len, hidden_dim]
        let rms_norm = RMSNorm::new(vec![768]).expect("RMSNorm should succeed");
        let input = randn(&[2, 128, 768]).expect("randn should succeed");

        let output = rms_norm.forward(&input);
        assert!(output.is_ok(), "3D RMSNorm forward failed");

        if let Ok(result) = output {
            assert_eq!(result.shape().dims(), &[2, 128, 768]);
        }
    }

    #[test]
    fn test_rms_norm_no_affine() {
        use torsh_tensor::creation::ones;

        let rms_norm = RMSNorm::with_config(vec![4], 1e-6, false).expect("RMSNorm should succeed");

        // Should have no parameters
        assert!(rms_norm.parameters().is_empty());

        // Should still normalize
        let input = ones(&[2, 4]).expect("ones should succeed");
        let output = rms_norm.forward(&input);
        assert!(output.is_ok());
    }

    #[test]
    fn test_rms_norm_multi_dimensional() {
        use torsh_tensor::creation::randn;

        // Test normalization over last 2 dimensions
        let rms_norm = RMSNorm::new(vec![8, 8]).expect("RMSNorm should succeed");
        let input = randn(&[4, 8, 8]).expect("randn should succeed");

        let output = rms_norm.forward(&input);
        assert!(output.is_ok());

        if let Ok(result) = output {
            assert_eq!(result.shape().dims(), &[4, 8, 8]);
        }
    }

    #[test]
    fn test_rms_norm_shape_mismatch() {
        let rms_norm = RMSNorm::new(vec![768]).expect("RMSNorm should succeed");

        // Input with wrong feature dimension
        let input = zeros(&[2, 128, 512]).expect("zeros should succeed");
        let result = rms_norm.forward(&input);
        assert!(result.is_err(), "Should error on shape mismatch");
    }

    #[test]
    fn test_rms_norm_training_modes() {
        let mut rms_norm = RMSNorm::new(vec![64]).expect("RMSNorm should succeed");

        // Test training mode switching (default is training mode)
        assert!(rms_norm.training());

        rms_norm.eval();
        assert!(!rms_norm.training());

        rms_norm.train();
        assert!(rms_norm.training());
    }
}
