//! Instance normalization layers
//!
//! Instance normalization normalizes each sample independently across spatial dimensions.
//! This is particularly useful for style transfer and generative models where batch
//! statistics may not be meaningful.

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

/// Pull the affine parameters out of a module base, if the layer is affine.
///
/// Returns clones of the parameter tensors, which share their storage *and*
/// their gradient slot with the registered `Parameter`, so `backward()`
/// accumulates straight into the module's gamma/beta.
fn affine_tensors(base: &ModuleBase, affine: bool) -> (Option<Tensor>, Option<Tensor>) {
    if !affine {
        return (None, None);
    }
    let weight = base
        .parameters
        .get("weight")
        .map(|p| p.tensor().read().clone());
    let bias = base
        .parameters
        .get("bias")
        .map(|p| p.tensor().read().clone());
    (weight, bias)
}

/// 1D instance normalization layer, for `(N, C)` inputs.
///
/// # Degenerate by construction
///
/// This layer's rank contract is `(N, C)`, not PyTorch's `(N, C, L)`: there is
/// no spatial axis left to normalize over, so each `(sample, channel)` statistic
/// is taken over a *single* element. The mean is the element itself, the
/// variance is zero, and the output is therefore exactly `bias`, constant in the
/// input — `d output / d input` is identically zero.
///
/// That is what the arithmetic says, and it is now what `backward()` reports.
/// While the mean was a detached constant the layer instead claimed a gradient
/// of `gamma / sqrt(eps)` (about `316 * gamma` at the default epsilon), which
/// was pure noise. Callers that want a real 1-D instance norm should feed an
/// `(N, C, L)` tensor to [`crate::functional::instance_norm`].
pub struct InstanceNorm1d {
    base: ModuleBase,
    num_features: usize,
    config: NormalizationConfig,
}

impl InstanceNorm1d {
    pub fn new(num_features: usize) -> Result<Self> {
        Self::with_config(num_features, NormalizationConfig::default())
    }

    pub fn with_config(num_features: usize, config: NormalizationConfig) -> Result<Self> {
        let mut base = ModuleBase::new();

        // Initialize parameters if affine
        if config.affine {
            let weight = ones(&[num_features])?;
            let bias = zeros(&[num_features])?;
            base.register_parameter("weight".to_string(), Parameter::new(weight));
            base.register_parameter("bias".to_string(), Parameter::new(bias));
        }

        Ok(Self {
            base,
            num_features,
            config,
        })
    }

    pub fn num_features(&self) -> usize {
        self.num_features
    }

    pub fn eps(&self) -> f32 {
        self.config.eps
    }
}

impl Module for InstanceNorm1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input_shape = input.shape();
        let dims = input_shape.dims();

        if dims.len() != 2 {
            return Err(torsh_core::error::TorshError::InvalidShape(format!(
                "InstanceNorm1d expects 2D input (N, C), got shape {:?}",
                dims
            )));
        }

        if dims[1] != self.num_features {
            return Err(torsh_core::error::TorshError::InvalidShape(format!(
                "Expected {} features, got {}",
                self.num_features, dims[1]
            )));
        }

        let (weight, bias) = affine_tensors(&self.base, self.config.affine);
        utils::instance_normalize(input, weight.as_ref(), bias.as_ref(), self.config.eps)
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

/// 2D instance normalization layer
pub struct InstanceNorm2d {
    base: ModuleBase,
    num_features: usize,
    config: NormalizationConfig,
}

impl InstanceNorm2d {
    pub fn new(num_features: usize) -> Result<Self> {
        Self::with_config(num_features, NormalizationConfig::default())
    }

    pub fn with_config(num_features: usize, config: NormalizationConfig) -> Result<Self> {
        let mut base = ModuleBase::new();

        // Initialize parameters if affine
        if config.affine {
            let weight = ones(&[num_features])?;
            let bias = zeros(&[num_features])?;
            base.register_parameter("weight".to_string(), Parameter::new(weight));
            base.register_parameter("bias".to_string(), Parameter::new(bias));
        }

        Ok(Self {
            base,
            num_features,
            config,
        })
    }

    pub fn num_features(&self) -> usize {
        self.num_features
    }

    pub fn eps(&self) -> f32 {
        self.config.eps
    }
}

impl Module for InstanceNorm2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input_shape = input.shape();
        let dims = input_shape.dims();

        if dims.len() != 4 {
            return Err(torsh_core::error::TorshError::InvalidShape(format!(
                "InstanceNorm2d expects 4D input (N, C, H, W), got shape {:?}",
                dims
            )));
        }

        if dims[1] != self.num_features {
            return Err(torsh_core::error::TorshError::InvalidShape(format!(
                "Expected {} features, got {}",
                self.num_features, dims[1]
            )));
        }

        let (weight, bias) = affine_tensors(&self.base, self.config.affine);
        utils::instance_normalize(input, weight.as_ref(), bias.as_ref(), self.config.eps)
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

/// 3D instance normalization layer
pub struct InstanceNorm3d {
    base: ModuleBase,
    num_features: usize,
    config: NormalizationConfig,
}

impl InstanceNorm3d {
    pub fn new(num_features: usize) -> Result<Self> {
        Self::with_config(num_features, NormalizationConfig::default())
    }

    pub fn with_config(num_features: usize, config: NormalizationConfig) -> Result<Self> {
        let mut base = ModuleBase::new();

        // Initialize parameters if affine
        if config.affine {
            let weight = ones(&[num_features])?;
            let bias = zeros(&[num_features])?;
            base.register_parameter("weight".to_string(), Parameter::new(weight));
            base.register_parameter("bias".to_string(), Parameter::new(bias));
        }

        Ok(Self {
            base,
            num_features,
            config,
        })
    }

    pub fn num_features(&self) -> usize {
        self.num_features
    }

    pub fn eps(&self) -> f32 {
        self.config.eps
    }
}

impl Module for InstanceNorm3d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input_shape = input.shape();
        let dims = input_shape.dims();

        if dims.len() != 5 {
            return Err(torsh_core::error::TorshError::InvalidShape(format!(
                "InstanceNorm3d expects 5D input (N, C, D, H, W), got shape {:?}",
                dims
            )));
        }

        if dims[1] != self.num_features {
            return Err(torsh_core::error::TorshError::InvalidShape(format!(
                "Expected {} features, got {}",
                self.num_features, dims[1]
            )));
        }

        let (weight, bias) = affine_tensors(&self.base, self.config.affine);
        utils::instance_normalize(input, weight.as_ref(), bias.as_ref(), self.config.eps)
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

// Re-export the instance normalization components (already defined in this module)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instance_norm_2d_creation() {
        let instance_norm = InstanceNorm2d::new(64).expect("Instance Norm2d should succeed");
        assert_eq!(instance_norm.num_features(), 64);
        assert_eq!(instance_norm.eps(), 1e-5);
    }

    #[test]
    fn test_instance_norm_2d_shape_validation() {
        let instance_norm = InstanceNorm2d::new(3).expect("Instance Norm2d should succeed");

        // Valid input
        let input = zeros(&[2, 3, 32, 32]).expect("zeros should succeed");
        assert!(instance_norm.forward(&input).is_ok());

        // Invalid dimensions
        let input_3d = zeros(&[2, 3, 32]).expect("zeros should succeed");
        assert!(instance_norm.forward(&input_3d).is_err());

        // Wrong number of channels
        let input_wrong_channels = zeros(&[2, 4, 32, 32]).expect("zeros should succeed");
        assert!(instance_norm.forward(&input_wrong_channels).is_err());
    }
}
