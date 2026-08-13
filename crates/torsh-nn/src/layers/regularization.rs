//! Regularization layers
//!
//! Every stochastic layer in this module samples its mask through
//! `scirs2_core::random`, so training-mode behaviour is genuinely random rather
//! than a constant rescaling.

use crate::{Module, ModuleBase, Parameter};
use scirs2_core::random::quick::{random_f32, random_usize};
use torsh_core::device::DeviceType;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;
use torsh_core::error::Result;
use torsh_tensor::{creation::*, Tensor};

/// Sample an inverted-dropout mask of `len` independent Bernoulli draws.
///
/// Kept entries carry the `1/(1-p)` rescaling so the expected value of the
/// output matches the input; dropped entries are exactly `0.0`.
fn bernoulli_keep_mask(len: usize, p: f32) -> Vec<f32> {
    let keep_prob = 1.0 - p;
    let scale = 1.0 / keep_prob;
    (0..len)
        .map(|_| if random_f32() < p { 0.0 } else { scale })
        .collect()
}

/// Build a tensor from a mask vector, matching `reference`'s shape and device.
fn mask_tensor(mask: Vec<f32>, reference: &Tensor) -> Result<Tensor> {
    Tensor::from_data(mask, reference.shape().dims().to_vec(), reference.device())
}

/// Dropout layer for regularization
pub struct Dropout {
    base: ModuleBase,
    p: f32,
    inplace: bool,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        Self {
            base: ModuleBase::new(),
            p: p.clamp(0.0, 1.0),
            inplace: false,
        }
    }

    pub fn with_inplace(p: f32, inplace: bool) -> Self {
        Self {
            base: ModuleBase::new(),
            p: p.clamp(0.0, 1.0),
            inplace,
        }
    }
}

impl Default for Dropout {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !self.base.training() {
            // During evaluation, just return the input
            return Ok(input.clone());
        }

        if self.p == 0.0 {
            return Ok(input.clone());
        }

        if self.p == 1.0 {
            return zeros(input.shape().dims());
        }

        // Inverted dropout: sample an independent Bernoulli draw per element and
        // rescale the survivors by 1/(1-p) so the expectation is preserved.
        let mask = mask_tensor(bernoulli_keep_mask(input.shape().numel(), self.p), input)?;
        input.mul_op(&mask)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
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

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

impl std::fmt::Debug for Dropout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dropout")
            .field("p", &self.p)
            .field("inplace", &self.inplace)
            .finish()
    }
}

// =============================================================================
// MODERN REGULARIZATION TECHNIQUES
// =============================================================================

/// DropConnect layer - drops individual weights instead of activations
///
/// DropConnect is a generalization of Dropout that randomly drops connections
/// (weights) instead of activations. This provides more thorough regularization.
///
/// # Reference
/// Wan et al., "Regularization of Neural Networks using DropConnect", ICML 2013
pub struct DropConnect {
    base: ModuleBase,
    p: f32,
}

impl DropConnect {
    pub fn new(p: f32) -> Self {
        Self {
            base: ModuleBase::new(),
            p: p.clamp(0.0, 1.0),
        }
    }

    /// Apply DropConnect to a weight matrix.
    ///
    /// This is the formulation from the paper: every *connection* is dropped
    /// independently. Call it on the weight tensor before the matmul, e.g.
    /// `let w = drop_connect.drop_weights(&weight)?;`.
    ///
    /// Outside training mode the weights are returned unchanged.
    pub fn drop_weights(&self, weight: &Tensor) -> Result<Tensor> {
        if !self.base.training() || self.p == 0.0 {
            return Ok(weight.clone());
        }
        if self.p == 1.0 {
            return zeros(weight.shape().dims());
        }
        let mask = mask_tensor(bernoulli_keep_mask(weight.shape().numel(), self.p), weight)?;
        weight.mul_op(&mask)
    }
}

impl Module for DropConnect {
    /// Activation-level DropConnect.
    ///
    /// A `Module` only sees activations, so `forward` applies an independent
    /// Bernoulli mask per activation. Use [`DropConnect::drop_weights`] for the
    /// true connection-level formulation.
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !self.base.training() || self.p == 0.0 {
            return Ok(input.clone());
        }

        if self.p == 1.0 {
            return zeros(input.shape().dims());
        }

        let mask = mask_tensor(bernoulli_keep_mask(input.shape().numel(), self.p), input)?;
        input.mul_op(&mask)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
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

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// Dropout2d - Spatial dropout for convolutional layers
///
/// Drops entire feature maps instead of individual elements, which is more
/// effective for convolutional layers where nearby activations are strongly correlated.
///
/// # Reference
/// Tompson et al., "Efficient Object Localization Using Convolutional Networks", CVPR 2015
pub struct Dropout2d {
    base: ModuleBase,
    p: f32,
}

impl Dropout2d {
    pub fn new(p: f32) -> Self {
        Self {
            base: ModuleBase::new(),
            p: p.clamp(0.0, 1.0),
        }
    }
}

impl Module for Dropout2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !self.base.training() || self.p == 0.0 {
            return Ok(input.clone());
        }

        // Input shape: (batch_size, channels, height, width)
        // Drop entire channels (feature maps)
        let shape = input.shape();
        let dims = shape.dims();

        if dims.len() != 4 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "Dropout2d expects 4D input (batch, channels, height, width)".to_string(),
            ));
        }

        if self.p == 1.0 {
            return zeros(dims);
        }

        // One Bernoulli draw per (batch, channel) feature map, broadcast across
        // the spatial extent so whole feature maps are dropped together.
        let batch_size = dims[0];
        let channels = dims[1];
        let spatial = dims[2] * dims[3];
        let per_channel = bernoulli_keep_mask(batch_size * channels, self.p);

        let mut mask_data = Vec::with_capacity(batch_size * channels * spatial);
        for value in per_channel {
            mask_data.extend(std::iter::repeat(value).take(spatial));
        }

        let mask = mask_tensor(mask_data, input)?;
        input.mul_op(&mask)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
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

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// DropBlock - Structured dropout for convolutional networks
///
/// DropBlock drops contiguous regions instead of individual elements,
/// which is more effective than standard dropout for conv networks.
///
/// # Reference
/// Ghiasi et al., "DropBlock: A regularization method for convolutional networks", NeurIPS 2018
pub struct DropBlock2d {
    base: ModuleBase,
    drop_prob: f32,
    block_size: usize,
}

impl DropBlock2d {
    pub fn new(drop_prob: f32, block_size: usize) -> Self {
        Self {
            base: ModuleBase::new(),
            drop_prob: drop_prob.clamp(0.0, 1.0),
            block_size,
        }
    }
}

impl Module for DropBlock2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !self.base.training() || self.drop_prob == 0.0 {
            return Ok(input.clone());
        }

        // Input shape: (batch_size, channels, height, width)
        let shape = input.shape();
        let dims = shape.dims();

        if dims.len() != 4 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "DropBlock2d expects 4D input (batch, channels, height, width)".to_string(),
            ));
        }

        let (batch, channels, height, width) = (dims[0], dims[1], dims[2], dims[3]);
        let block_size = self.block_size.clamp(1, height.min(width));

        // Ghiasi et al. eq. (1): the seed probability that yields the requested
        // drop rate once each seed is expanded into a block_size^2 region.
        let valid_h = height - block_size + 1;
        let valid_w = width - block_size + 1;
        let gamma = self.drop_prob / (block_size * block_size) as f32 * (height * width) as f32
            / (valid_h * valid_w) as f32;

        let plane = height * width;
        let mut mask_data = vec![1.0f32; batch * channels * plane];

        for n in 0..batch {
            for c in 0..channels {
                let base = n * channels * plane + c * plane;
                for seed_h in 0..valid_h {
                    for seed_w in 0..valid_w {
                        if random_f32() >= gamma {
                            continue;
                        }
                        for dh in 0..block_size {
                            for dw in 0..block_size {
                                let idx = base + (seed_h + dh) * width + (seed_w + dw);
                                mask_data[idx] = 0.0;
                            }
                        }
                    }
                }

                // Per-plane renormalization: count / kept, as in the reference
                // implementation, so the activation scale is preserved.
                let kept: f32 = mask_data[base..base + plane].iter().sum();
                if kept > 0.0 {
                    let scale = plane as f32 / kept;
                    for value in &mut mask_data[base..base + plane] {
                        *value *= scale;
                    }
                }
            }
        }

        let mask = mask_tensor(mask_data, input)?;
        input.mul_op(&mask)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
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

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// Stochastic Depth (Drop Path) - randomly drops entire layers during training
///
/// Stochastic Depth improves training of very deep networks by randomly
/// dropping entire layers during training, effectively training an ensemble.
///
/// # Reference
/// Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016
pub struct StochasticDepth {
    base: ModuleBase,
    drop_prob: f32,
    scale_by_keep: bool,
}

impl StochasticDepth {
    pub fn new(drop_prob: f32) -> Self {
        Self {
            base: ModuleBase::new(),
            drop_prob: drop_prob.clamp(0.0, 1.0),
            scale_by_keep: true,
        }
    }

    pub fn with_scaling(drop_prob: f32, scale_by_keep: bool) -> Self {
        Self {
            base: ModuleBase::new(),
            drop_prob: drop_prob.clamp(0.0, 1.0),
            scale_by_keep,
        }
    }
}

impl Module for StochasticDepth {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !self.base.training() || self.drop_prob == 0.0 {
            return Ok(input.clone());
        }

        if self.drop_prob == 1.0 {
            return zeros(input.shape().dims());
        }

        // Drop each sample of the batch independently ("drop path"): the whole
        // residual branch either survives (optionally rescaled by 1/keep_prob)
        // or is zeroed for that sample.
        let shape = input.shape();
        let dims = shape.dims();
        let batch = if dims.is_empty() { 1 } else { dims[0] };
        let per_sample = if batch == 0 { 0 } else { shape.numel() / batch };

        let keep_prob = 1.0 - self.drop_prob;
        let kept_scale = if self.scale_by_keep {
            1.0 / keep_prob
        } else {
            1.0
        };

        let mut mask_data = Vec::with_capacity(shape.numel());
        for _ in 0..batch {
            let value = if random_f32() < self.drop_prob {
                0.0
            } else {
                kept_scale
            };
            mask_data.extend(std::iter::repeat(value).take(per_sample));
        }

        let mask = mask_tensor(mask_data, input)?;
        input.mul_op(&mask)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
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

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// AlphaDropout - Dropout variant for SELU activation
///
/// AlphaDropout is designed to work with SELU activations, maintaining
/// self-normalizing properties during dropout.
///
/// # Reference
/// Klambauer et al., "Self-Normalizing Neural Networks", NeurIPS 2017
pub struct AlphaDropout {
    base: ModuleBase,
    p: f32,
    alpha: f32,
    #[allow(dead_code)]
    scale: f32,
}

impl AlphaDropout {
    pub fn new(p: f32) -> Self {
        // SELU-specific constants
        let alpha = -1.7580993408473766; // SELU alpha
        let scale = 1.0507009873554804; // SELU lambda

        Self {
            base: ModuleBase::new(),
            p: p.clamp(0.0, 1.0),
            alpha,
            scale,
        }
    }
}

impl Module for AlphaDropout {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !self.base.training() || self.p == 0.0 {
            return Ok(input.clone());
        }

        let keep_prob = 1.0 - self.p;

        // Klambauer et al. eq. (6)-(8): dropped units are set to the SELU
        // saturation value `alpha` (not to zero) and the result is affinely
        // rescaled so mean and variance are preserved.
        let a = (keep_prob + self.alpha * self.alpha * keep_prob * (1.0 - keep_prob)).powf(-0.5);
        let b = -a * self.alpha * (1.0 - keep_prob);

        let data = input.to_vec()?;
        let result: Vec<f32> = data
            .iter()
            .map(|&x| {
                let kept = random_f32() >= self.p;
                let value = if kept { x } else { self.alpha };
                a * value + b
            })
            .collect();

        Tensor::from_data(result, input.shape().dims().to_vec(), input.device())
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
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

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// Cutout - randomly masks out square regions of input
///
/// Cutout is a simple regularization technique that randomly masks out
/// square regions of the input during training. Particularly effective for images.
///
/// # Reference
/// DeVries & Taylor, "Improved Regularization of Convolutional Neural Networks with Cutout", arXiv 2017
pub struct Cutout {
    base: ModuleBase,
    n_holes: usize,
    length: usize,
}

impl Cutout {
    pub fn new(n_holes: usize, length: usize) -> Self {
        Self {
            base: ModuleBase::new(),
            n_holes,
            length,
        }
    }
}

impl Module for Cutout {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !self.base.training() {
            return Ok(input.clone());
        }

        // Input shape: (batch, channels, height, width)
        let shape = input.shape();
        let dims = shape.dims();

        if dims.len() != 4 {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Cutout expects 4D input (batch, channels, height, width), got {dims:?}"
            )));
        }

        if self.n_holes == 0 || self.length == 0 {
            return Ok(input.clone());
        }

        let (batch, channels, height, width) = (dims[0], dims[1], dims[2], dims[3]);
        let plane = height * width;
        let mut mask_data = vec![1.0f32; batch * channels * plane];
        let half = self.length / 2;

        for n in 0..batch {
            for _ in 0..self.n_holes {
                // Hole centres are sampled uniformly over the image, and the
                // square is clipped at the borders (DeVries & Taylor, sec. 3).
                let centre_y = random_usize(0, height.saturating_sub(1));
                let centre_x = random_usize(0, width.saturating_sub(1));
                let y0 = centre_y.saturating_sub(half);
                let x0 = centre_x.saturating_sub(half);
                let y1 = (centre_y + half + 1).min(height);
                let x1 = (centre_x + half + 1).min(width);

                for c in 0..channels {
                    let base = n * channels * plane + c * plane;
                    for y in y0..y1 {
                        for x in x0..x1 {
                            mask_data[base + y * width + x] = 0.0;
                        }
                    }
                }
            }
        }

        let mask = mask_tensor(mask_data, input)?;
        input.mul_op(&mask)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
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

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// Mixup - linearly interpolates between random pairs of training examples
///
/// Mixup creates virtual training examples by mixing pairs of examples and their labels.
/// This helps improve generalization and robustness.
///
/// # Reference
/// Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018
pub struct Mixup {
    alpha: f32,
}

/// Draw `lambda ~ Beta(alpha, alpha)` through `scirs2_core::random`.
fn sample_symmetric_beta(alpha: f32) -> Result<f32> {
    use scirs2_core::random::distributions::Beta;
    use scirs2_core::random::thread_rng;

    let distribution = Beta::new(alpha as f64, alpha as f64).map_err(|e| {
        torsh_core::error::TorshError::InvalidArgument(format!("invalid mixing alpha {alpha}: {e}"))
    })?;
    let mut rng = thread_rng();
    Ok(distribution.sample(&mut rng) as f32)
}

/// Random permutation of `0..len` used to pick the partner sample of the batch.
fn batch_permutation(len: usize) -> Vec<usize> {
    use scirs2_core::random::quick::shuffle;

    let mut indices: Vec<usize> = (0..len).collect();
    shuffle(&mut indices);
    indices
}

impl Mixup {
    pub fn new(alpha: f32) -> Self {
        Self { alpha }
    }

    /// Apply mixup to a batch of data.
    ///
    /// Returns the mixed inputs and the lambda used, so labels can be mixed as
    /// `lambda * y + (1 - lambda) * y[permutation]`. Use
    /// [`Mixup::apply_with_permutation`] when the permutation itself is needed.
    pub fn apply(&self, input: &Tensor, training: bool) -> Result<(Tensor, f32)> {
        let (mixed, _permutation, lambda) = self.apply_with_permutation(input, training)?;
        Ok((mixed, lambda))
    }

    /// Apply mixup and also return the batch permutation used for mixing.
    pub fn apply_with_permutation(
        &self,
        input: &Tensor,
        training: bool,
    ) -> Result<(Tensor, Vec<usize>, f32)> {
        let shape = input.shape();
        let dims = shape.dims();
        let batch = dims.first().copied().unwrap_or(0);

        if !training || self.alpha <= 0.0 || batch < 2 {
            return Ok((input.clone(), (0..batch).collect(), 1.0));
        }

        let lambda = sample_symmetric_beta(self.alpha)?;
        let permutation = batch_permutation(batch);
        let per_sample = shape.numel() / batch;

        let data = input.to_vec()?;
        let mut mixed = vec![0.0f32; data.len()];
        for n in 0..batch {
            let partner = permutation[n];
            for i in 0..per_sample {
                mixed[n * per_sample + i] = lambda * data[n * per_sample + i]
                    + (1.0 - lambda) * data[partner * per_sample + i];
            }
        }

        let output = Tensor::from_data(mixed, dims.to_vec(), input.device())?;
        Ok((output, permutation, lambda))
    }
}

/// CutMix - replaces regions of images with patches from other images
///
/// CutMix combines regions from different images and mixes their labels
/// proportionally to the area of the patches.
///
/// # Reference
/// Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features", ICCV 2019
pub struct CutMix {
    alpha: f32,
}

impl CutMix {
    pub fn new(alpha: f32) -> Self {
        Self { alpha }
    }

    /// Apply CutMix to a batch of data.
    ///
    /// Returns the mixed inputs and the *area-corrected* lambda, which is what
    /// the paper uses for label mixing.
    pub fn apply(&self, input: &Tensor, training: bool) -> Result<(Tensor, f32)> {
        let (mixed, _permutation, lambda) = self.apply_with_permutation(input, training)?;
        Ok((mixed, lambda))
    }

    /// Apply CutMix and also return the batch permutation used for mixing.
    ///
    /// Expects 4D `(batch, channels, height, width)` input.
    pub fn apply_with_permutation(
        &self,
        input: &Tensor,
        training: bool,
    ) -> Result<(Tensor, Vec<usize>, f32)> {
        let shape = input.shape();
        let dims = shape.dims();

        if dims.len() != 4 {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "CutMix expects 4D input (batch, channels, height, width), got {dims:?}"
            )));
        }

        let (batch, channels, height, width) = (dims[0], dims[1], dims[2], dims[3]);
        if !training || self.alpha <= 0.0 || batch < 2 {
            return Ok((input.clone(), (0..batch).collect(), 1.0));
        }

        let lambda = sample_symmetric_beta(self.alpha)?;
        let permutation = batch_permutation(batch);

        // Bounding box whose area is (1 - lambda) of the image, clipped to the
        // image borders (Yun et al., sec. 3.1).
        let ratio = (1.0 - lambda).max(0.0).sqrt();
        let cut_h = ((height as f32) * ratio).round() as usize;
        let cut_w = ((width as f32) * ratio).round() as usize;
        let centre_y = random_usize(0, height.saturating_sub(1));
        let centre_x = random_usize(0, width.saturating_sub(1));
        let y0 = centre_y.saturating_sub(cut_h / 2);
        let x0 = centre_x.saturating_sub(cut_w / 2);
        let y1 = (centre_y + cut_h / 2).min(height);
        let x1 = (centre_x + cut_w / 2).min(width);

        let plane = height * width;
        let data = input.to_vec()?;
        let mut mixed = data.clone();
        for n in 0..batch {
            let partner = permutation[n];
            for c in 0..channels {
                let dst = n * channels * plane + c * plane;
                let src = partner * channels * plane + c * plane;
                for y in y0..y1 {
                    for x in x0..x1 {
                        mixed[dst + y * width + x] = data[src + y * width + x];
                    }
                }
            }
        }

        // Correct lambda for the actually pasted (clipped) area.
        let pasted = (y1.saturating_sub(y0) * x1.saturating_sub(x0)) as f32;
        let corrected_lambda = 1.0 - pasted / plane as f32;

        let output = Tensor::from_data(mixed, dims.to_vec(), input.device())?;
        Ok((output, permutation, corrected_lambda))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dropout_creation() {
        let dropout = Dropout::new(0.5);
        assert!((dropout.p - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_dropconnect_creation() {
        let dropconnect = DropConnect::new(0.3);
        assert!((dropconnect.p - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_dropout2d_creation() {
        let dropout2d = Dropout2d::new(0.5);
        assert!((dropout2d.p - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_dropblock_creation() {
        let dropblock = DropBlock2d::new(0.1, 7);
        assert!((dropblock.drop_prob - 0.1).abs() < 1e-6);
        assert_eq!(dropblock.block_size, 7);
    }

    #[test]
    fn test_stochastic_depth_creation() {
        let stoch_depth = StochasticDepth::new(0.2);
        assert!((stoch_depth.drop_prob - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_alpha_dropout_creation() {
        let alpha_dropout = AlphaDropout::new(0.1);
        assert!((alpha_dropout.p - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_cutout_creation() {
        let cutout = Cutout::new(1, 16);
        assert_eq!(cutout.n_holes, 1);
        assert_eq!(cutout.length, 16);
    }

    #[test]
    fn test_mixup_creation() {
        let mixup = Mixup::new(1.0);
        assert!((mixup.alpha - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cutmix_creation() {
        let cutmix = CutMix::new(1.0);
        assert!((cutmix.alpha - 1.0).abs() < 1e-6);
    }
}
