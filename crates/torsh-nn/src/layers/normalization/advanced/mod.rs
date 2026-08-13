//! Advanced normalization techniques
//!
//! This module provides sophisticated normalization methods that combine or adapt
//! multiple normalization strategies for improved performance.

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

/// Switchable normalization that learns to combine different normalization techniques
///
/// This layer learns weights to combine batch normalization, instance normalization,
/// and layer normalization adaptively for each channel.
pub struct SwitchableNorm2d {
    base: ModuleBase,
    num_features: usize,
    config: NormalizationConfig,
    #[allow(dead_code)]
    using_movavg: bool,
}

impl SwitchableNorm2d {
    pub fn new(num_features: usize) -> Result<Self> {
        Self::with_config(num_features, NormalizationConfig::default())
    }

    pub fn with_config(num_features: usize, config: NormalizationConfig) -> Result<Self> {
        let mut base = ModuleBase::new();

        // Initialize switchable weights for combining different normalizations
        let switch_weight = ones(&[3, num_features])?; // 3 normalization types
        base.register_parameter("switch_weight".to_string(), Parameter::new(switch_weight));

        // Initialize parameters if affine
        if config.affine {
            let weight = ones(&[num_features])?;
            let bias = zeros(&[num_features])?;
            base.register_parameter("weight".to_string(), Parameter::new(weight));
            base.register_parameter("bias".to_string(), Parameter::new(bias));
        }

        // Initialize running statistics for batch norm component
        if config.track_running_stats {
            let running_mean = zeros(&[num_features])?;
            let running_var = ones(&[num_features])?;
            base.register_buffer("running_mean".to_string(), running_mean);
            base.register_buffer("running_var".to_string(), running_var);
            base.register_buffer("num_batches_tracked".to_string(), zeros(&[1])?);
        }

        let using_movavg = config.track_running_stats;

        Ok(Self {
            base,
            num_features,
            config,
            using_movavg,
        })
    }

    pub fn num_features(&self) -> usize {
        self.num_features
    }

    pub fn eps(&self) -> f32 {
        self.config.eps
    }

    /// Per-channel batch statistics, shaped `[1, C, 1, 1]`.
    fn batch_norm_stats(&self, input: &Tensor) -> Result<(Tensor, Tensor)> {
        let mean = input.mean(Some(&[0, 2, 3]), true)?;
        let centered = input.sub(&mean)?;
        let variance = centered.pow_scalar(2.0)?.mean(Some(&[0, 2, 3]), true)?;
        Ok((mean, variance))
    }

    /// Per-instance statistics, shaped `[N, C, 1, 1]`.
    fn instance_norm_stats(&self, input: &Tensor) -> Result<(Tensor, Tensor)> {
        let mean = input.mean(Some(&[2, 3]), true)?;
        let centered = input.sub(&mean)?;
        let variance = centered.pow_scalar(2.0)?.mean(Some(&[2, 3]), true)?;
        Ok((mean, variance))
    }

    /// Per-sample statistics over channels and space, shaped `[N, 1, 1, 1]`.
    fn layer_norm_stats(&self, input: &Tensor) -> Result<(Tensor, Tensor)> {
        let mean = input.mean(Some(&[1, 2, 3]), true)?;
        let centered = input.sub(&mean)?;
        let variance = centered.pow_scalar(2.0)?.mean(Some(&[1, 2, 3]), true)?;
        Ok((mean, variance))
    }

    /// Per-channel softmax over the three normalization types.
    ///
    /// The max subtraction is the same numerical stabilisation the scalar
    /// implementation performed, but the shift is a *constant* tensor, which
    /// leaves the softmax exactly shift-invariant and therefore leaves the
    /// gradient with respect to `switch_weight` intact. Before the rewrite the
    /// whole softmax ran on `to_vec()` values, so `switch_weight` never reached
    /// the autograd graph at all and could not be trained.
    fn switch_probabilities(&self) -> Result<Tensor> {
        let switch_weight = self.base.parameters.get("switch_weight").ok_or_else(|| {
            torsh_core::error::TorshError::InvalidOperation(
                "Switch weight parameter not found".to_string(),
            )
        })?;
        let logits = switch_weight.tensor().read().clone();

        let values = logits.to_vec()?;
        let channels = self.num_features;
        let mut maxima = vec![f32::NEG_INFINITY; channels];
        for (index, value) in values.iter().enumerate() {
            let channel = index % channels;
            if *value > maxima[channel] {
                maxima[channel] = *value;
            }
        }
        let shift = Tensor::from_data(maxima, vec![1, channels], logits.device())?;

        let exponentials = logits.sub(&shift)?.exp()?;
        let total = exponentials.sum_dim(&[0], true)?;
        exponentials.div(&total)
    }

    /// Apply switchable normalization
    fn apply_switchable_norm(&self, input: &Tensor) -> Result<Tensor> {
        let (bn_mean, bn_var) = self.batch_norm_stats(input)?;
        let (in_mean, in_var) = self.instance_norm_stats(input)?;
        let (ln_mean, ln_var) = self.layer_norm_stats(input)?;

        // `[3, C]` probabilities sliced into three `[1, C, 1, 1]` weights.
        let probabilities = self.switch_probabilities()?;
        let channels = self.num_features as i32;
        let weight_for = |row: i64| -> Result<Tensor> {
            probabilities
                .narrow(0, row, 1)?
                .reshape(&[1, channels, 1, 1])
        };
        let bn_weight = weight_for(0)?;
        let in_weight = weight_for(1)?;
        let ln_weight = weight_for(2)?;

        let combined_mean = bn_mean
            .mul(&bn_weight)?
            .add(&in_mean.mul(&in_weight)?)?
            .add(&ln_mean.mul(&ln_weight)?)?;
        let combined_var = bn_var
            .mul(&bn_weight)?
            .add(&in_var.mul(&in_weight)?)?
            .add(&ln_var.mul(&ln_weight)?)?;

        let std = combined_var.add_scalar(self.config.eps)?.sqrt()?;
        let mut normalized = input.sub(&combined_mean)?.div(&std)?;

        if self.config.affine {
            let broadcast = utils::channel_broadcast_shape(4, self.num_features);
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
}

impl Module for SwitchableNorm2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input_shape = input.shape();
        let dims = input_shape.dims();

        if dims.len() != 4 {
            return Err(torsh_core::error::TorshError::InvalidShape(format!(
                "SwitchableNorm2d expects 4D input (N, C, H, W), got shape {:?}",
                dims
            )));
        }

        if dims[1] != self.num_features {
            return Err(torsh_core::error::TorshError::InvalidShape(format!(
                "Expected {} features, got {}",
                self.num_features, dims[1]
            )));
        }

        self.apply_switchable_norm(input)
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

// Re-export the advanced normalization components (already defined in this module)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_switchable_norm_creation() {
        let switchable_norm = SwitchableNorm2d::new(64).expect("Switchable Norm2d should succeed");
        assert_eq!(switchable_norm.num_features(), 64);
        assert_eq!(switchable_norm.eps(), 1e-5);
    }

    #[test]
    fn test_switchable_norm_shape_validation() {
        let switchable_norm = SwitchableNorm2d::new(3).expect("Switchable Norm2d should succeed");

        // Valid input
        let input = zeros(&[2, 3, 32, 32]).expect("zeros should succeed");
        assert!(switchable_norm.forward(&input).is_ok());

        // Invalid dimensions
        let input_3d = zeros(&[2, 3, 32]).expect("zeros should succeed");
        assert!(switchable_norm.forward(&input_3d).is_err());

        // Wrong number of channels
        let input_wrong_channels = zeros(&[2, 4, 32, 32]).expect("zeros should succeed");
        assert!(switchable_norm.forward(&input_wrong_channels).is_err());
    }
}
