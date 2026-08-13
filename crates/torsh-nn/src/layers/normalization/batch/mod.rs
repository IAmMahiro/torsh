//! Batch normalization layers and variants
//!
//! This module implements various batch normalization techniques:
//! - Standard batch normalization (1D, 2D, 3D)
//! - Synchronized batch normalization for distributed training
//! - Virtual batch normalization for stable training
//! - Batch renormalization for improved stability
//!
//! All of them keep their running statistics behind shared, interior-mutable
//! handles, so `forward(&self, ...)` can fold each training batch into them and
//! evaluation mode really does normalize with the tracked statistics.

use crate::{Module, ModuleBase, Parameter};
use parking_lot::RwLock;
use std::sync::Arc;
use torsh_core::device::DeviceType;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::{creation::*, Tensor};

use super::common::{unbiased_variance, utils, NormalizationConfig, RunningStats};

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

/// Number of elements folded into each channel statistic.
fn samples_per_channel(dims: &[usize]) -> usize {
    dims.iter()
        .enumerate()
        .filter(|(axis, _)| *axis != 1)
        .map(|(_, size)| *size)
        .product()
}

/// Validate the rank and channel count of a batch-norm input.
fn validate_input(
    input: &Tensor,
    expected_rank: usize,
    num_features: usize,
    name: &str,
) -> Result<()> {
    let shape = input.shape();
    let dims = shape.dims();

    if dims.len() != expected_rank {
        return Err(TorshError::InvalidShape(format!(
            "{name} expects {expected_rank}D input, got shape {dims:?}"
        )));
    }

    if dims[1] != num_features {
        return Err(TorshError::InvalidShape(format!(
            "Expected {} features, got {}",
            num_features, dims[1]
        )));
    }

    Ok(())
}

/// Pull the affine parameters out of a module base, if the layer is affine.
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

/// Register the affine parameters of a batch-norm layer.
fn register_affine(base: &mut ModuleBase, num_features: usize) -> Result<()> {
    base.register_parameter("weight".to_string(), Parameter::new(ones(&[num_features])?));
    base.register_parameter("bias".to_string(), Parameter::new(zeros(&[num_features])?));
    Ok(())
}

/// The statistics-selection and normalization core shared by BatchNorm{1,2,3}d.
///
/// In training mode the batch statistics are used *and* folded into the running
/// statistics (with the unbiased batch variance, matching PyTorch). In
/// evaluation mode the tracked running statistics are used instead.
///
/// # Autograd
///
/// The training-mode statistics are recording tensor compositions, so
/// `backward()` propagates through the batch mean and variance exactly as
/// PyTorch does. What is folded into the running buffers is a *detached* copy:
/// those buffers are state, not graph nodes, and splicing them into the graph
/// would both mark them as requiring gradients and pin every past batch's graph
/// in memory. Evaluation mode likewise consumes the buffers detached, so the
/// tracked statistics are constants there.
fn batch_norm_forward(
    input: &Tensor,
    base: &ModuleBase,
    config: &NormalizationConfig,
    stats: Option<&RunningStats>,
    training: bool,
) -> Result<Tensor> {
    let (mean, var) = match (training, stats) {
        (false, Some(tracked)) => (
            tracked.running_mean().detach(),
            tracked.running_var().detach(),
        ),
        (_, tracked) => {
            let batch_mean = utils::compute_channel_mean(input)?;
            let batch_var = utils::compute_channel_variance(input, &batch_mean)?;

            if training {
                if let Some(tracked) = tracked {
                    let shape = input.shape();
                    let count = samples_per_channel(shape.dims());
                    let buffered_mean = utils::detached_statistic(&batch_mean)?;
                    let buffered_var = utils::detached_statistic(&batch_var)?;
                    let unbiased = unbiased_variance(&buffered_var, count)?;
                    tracked.update(&buffered_mean, &unbiased, config.momentum)?;
                }
            }

            (batch_mean, batch_var)
        }
    };

    let (weight, bias) = affine_tensors(base, config.affine);

    utils::apply_channel_normalization(
        input,
        &mean,
        &var,
        weight.as_ref(),
        bias.as_ref(),
        config.eps,
    )
}

/// Publish the running-statistics handles as module buffers.
fn stats_buffers(stats: Option<&RunningStats>) -> HashMap<String, Arc<RwLock<Tensor>>> {
    let mut buffers = HashMap::new();
    if let Some(stats) = stats {
        buffers.insert("running_mean".to_string(), stats.running_mean_handle());
        buffers.insert("running_var".to_string(), stats.running_var_handle());
        buffers.insert(
            "num_batches_tracked".to_string(),
            stats.num_batches_tracked_handle(),
        );
    }
    buffers
}

macro_rules! impl_batch_norm {
    ($name:ident, $rank:literal, $label:literal) => {
        impl $name {
            /// Create the layer with the default normalization configuration.
            pub fn new(num_features: usize) -> Result<Self> {
                Self::with_config(num_features, NormalizationConfig::default())
            }

            /// Create the layer with an explicit configuration.
            pub fn with_config(num_features: usize, config: NormalizationConfig) -> Result<Self> {
                let mut base = ModuleBase::new();

                if config.affine {
                    register_affine(&mut base, num_features)?;
                }

                let stats = if config.track_running_stats {
                    Some(RunningStats::new(num_features)?)
                } else {
                    None
                };

                Ok(Self {
                    base,
                    num_features,
                    config,
                    stats,
                })
            }

            /// Number of normalized channels.
            pub fn num_features(&self) -> usize {
                self.num_features
            }

            /// Numerical-stability epsilon.
            pub fn eps(&self) -> f32 {
                self.config.eps
            }

            /// Momentum used for the running-statistics update.
            pub fn momentum(&self) -> f32 {
                self.config.momentum
            }

            /// Snapshot of the tracked running mean, if statistics are tracked.
            pub fn running_mean(&self) -> Option<Tensor> {
                self.stats.as_ref().map(|s| s.running_mean())
            }

            /// Snapshot of the tracked running variance, if statistics are tracked.
            pub fn running_var(&self) -> Option<Tensor> {
                self.stats.as_ref().map(|s| s.running_var())
            }

            /// Number of batches folded into the running statistics.
            pub fn num_batches_tracked(&self) -> Result<Option<f32>> {
                match self.stats.as_ref() {
                    Some(stats) => Ok(Some(stats.num_batches_tracked()?)),
                    None => Ok(None),
                }
            }
        }

        impl Module for $name {
            fn forward(&self, input: &Tensor) -> Result<Tensor> {
                validate_input(input, $rank, self.num_features, $label)?;
                batch_norm_forward(
                    input,
                    &self.base,
                    &self.config,
                    self.stats.as_ref(),
                    self.training(),
                )
            }

            fn parameters(&self) -> HashMap<String, Parameter> {
                self.base.named_parameters()
            }

            fn named_parameters(&self) -> HashMap<String, Parameter> {
                self.base.named_parameters()
            }

            fn buffers(&self) -> Vec<Arc<RwLock<Tensor>>> {
                stats_buffers(self.stats.as_ref()).into_values().collect()
            }

            fn named_buffers(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
                stats_buffers(self.stats.as_ref())
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
        }

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct($label)
                    .field("num_features", &self.num_features)
                    .field("training", &self.base.training())
                    .finish()
            }
        }
    };
}

/// 1D batch normalization layer, for `(N, C)` inputs.
pub struct BatchNorm1d {
    base: ModuleBase,
    num_features: usize,
    config: NormalizationConfig,
    stats: Option<RunningStats>,
}

/// 2D batch normalization layer, for `(N, C, H, W)` inputs.
pub struct BatchNorm2d {
    base: ModuleBase,
    num_features: usize,
    config: NormalizationConfig,
    stats: Option<RunningStats>,
}

/// 3D batch normalization layer, for `(N, C, D, H, W)` inputs.
pub struct BatchNorm3d {
    base: ModuleBase,
    num_features: usize,
    config: NormalizationConfig,
    stats: Option<RunningStats>,
}

impl_batch_norm!(BatchNorm1d, 2, "BatchNorm1d");
impl_batch_norm!(BatchNorm2d, 4, "BatchNorm2d");
impl_batch_norm!(BatchNorm3d, 5, "BatchNorm3d");

// =============================================================================
// SPECIALIZED BATCH NORMALIZATION VARIANTS
// =============================================================================

/// Synchronized batch normalization for 2D inputs.
///
/// `SyncBatchNorm` differs from [`BatchNorm2d`] only in *where* the batch
/// statistics come from: with a distributed process group the per-rank sums are
/// all-reduced so every rank normalizes with the global statistics.
///
/// ToRSh's `torsh-nn` has no process-group handle, so this layer runs with a
/// world size of one: the statistics are computed from the local batch, which
/// is mathematically identical to [`BatchNorm2d`]. The type exists so models
/// converted from PyTorch keep their structure, and so a future distributed
/// build can swap the reduction in without changing user code. Check
/// [`SyncBatchNorm2d::world_size`] if you need to know whether synchronization
/// is actually happening.
pub struct SyncBatchNorm2d {
    inner: BatchNorm2d,
}

impl SyncBatchNorm2d {
    /// Create the layer with the default normalization configuration.
    pub fn new(num_features: usize) -> Result<Self> {
        Ok(Self {
            inner: BatchNorm2d::new(num_features)?,
        })
    }

    /// Create the layer with an explicit configuration.
    pub fn with_config(num_features: usize, config: NormalizationConfig) -> Result<Self> {
        Ok(Self {
            inner: BatchNorm2d::with_config(num_features, config)?,
        })
    }

    /// Number of ranks whose statistics are combined.
    ///
    /// Always `1` in this build: no process group is available, so no
    /// all-reduce takes place.
    pub fn world_size(&self) -> usize {
        1
    }

    /// Number of normalized channels.
    pub fn num_features(&self) -> usize {
        self.inner.num_features()
    }

    /// Snapshot of the tracked running mean.
    pub fn running_mean(&self) -> Option<Tensor> {
        self.inner.running_mean()
    }

    /// Snapshot of the tracked running variance.
    pub fn running_var(&self) -> Option<Tensor> {
        self.inner.running_var()
    }
}

impl Module for SyncBatchNorm2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.inner.forward(input)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.inner.parameters()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.inner.named_parameters()
    }

    fn buffers(&self) -> Vec<Arc<RwLock<Tensor>>> {
        self.inner.buffers()
    }

    fn named_buffers(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
        self.inner.named_buffers()
    }

    fn training(&self) -> bool {
        self.inner.training()
    }

    fn train(&mut self) {
        self.inner.train();
    }

    fn eval(&mut self) {
        self.inner.eval();
    }

    fn set_training(&mut self, training: bool) {
        self.inner.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.inner.to_device(device)
    }
}

impl std::fmt::Debug for SyncBatchNorm2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyncBatchNorm2d")
            .field("num_features", &self.inner.num_features())
            .field("world_size", &self.world_size())
            .finish()
    }
}

/// Reference-batch statistics captured by [`VirtualBatchNorm2d`].
#[derive(Debug, Clone)]
struct ReferenceStatistics {
    /// Per-channel mean of the reference batch.
    mean: Vec<f32>,
    /// Per-channel mean of the *squared* reference activations.
    mean_square: Vec<f32>,
    /// Number of activations per channel in the reference batch.
    count: usize,
}

/// Virtual batch normalization for 2D inputs.
///
/// Normalizes every example against statistics computed from a *fixed*
/// reference batch pooled with the current batch, which removes the strong
/// dependence of plain batch norm on the other examples in the minibatch
/// (Salimans et al., "Improved Techniques for Training GANs", NeurIPS 2016).
///
/// A reference batch must be installed with
/// [`VirtualBatchNorm2d::set_reference_batch`] before the first forward pass;
/// calling `forward` without one is an error rather than a silent fallback.
pub struct VirtualBatchNorm2d {
    base: ModuleBase,
    num_features: usize,
    config: NormalizationConfig,
    reference: Option<ReferenceStatistics>,
}

impl VirtualBatchNorm2d {
    /// Create the layer with the default normalization configuration.
    pub fn new(num_features: usize) -> Result<Self> {
        Self::with_config(num_features, NormalizationConfig::default())
    }

    /// Create the layer with an explicit configuration.
    pub fn with_config(num_features: usize, config: NormalizationConfig) -> Result<Self> {
        let mut base = ModuleBase::new();
        if config.affine {
            register_affine(&mut base, num_features)?;
        }

        Ok(Self {
            base,
            num_features,
            config,
            reference: None,
        })
    }

    /// Number of normalized channels.
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// Whether a reference batch has been installed.
    pub fn has_reference_batch(&self) -> bool {
        self.reference.is_some()
    }

    /// Install (or replace) the reference batch.
    ///
    /// The reference batch is summarised once into per-channel first and second
    /// moments; the batch itself is not retained.
    pub fn set_reference_batch(&mut self, reference: &Tensor) -> Result<()> {
        validate_input(reference, 4, self.num_features, "VirtualBatchNorm2d")?;

        let (mean, mean_square, count) = channel_moments(reference)?;
        self.reference = Some(ReferenceStatistics {
            mean,
            mean_square,
            count,
        });
        Ok(())
    }
}

/// Per-channel first and second moments of an `(N, C, H, W)` tensor.
///
/// Returns `(E[x], E[x^2], elements_per_channel)`.
fn channel_moments(input: &Tensor) -> Result<(Vec<f32>, Vec<f32>, usize)> {
    let shape = input.shape();
    let dims = shape.dims();
    let (batch, channels, height, width) = (dims[0], dims[1], dims[2], dims[3]);
    let plane = height * width;
    let count = batch * plane;

    let data = input.to_vec()?;
    let mut sums = vec![0.0f64; channels];
    let mut square_sums = vec![0.0f64; channels];

    for n in 0..batch {
        for c in 0..channels {
            let base = n * channels * plane + c * plane;
            for offset in 0..plane {
                let value = data[base + offset] as f64;
                sums[c] += value;
                square_sums[c] += value * value;
            }
        }
    }

    if count == 0 {
        return Err(TorshError::InvalidShape(
            "cannot compute channel moments of an empty batch".to_string(),
        ));
    }

    let divisor = count as f64;
    let mean = sums.iter().map(|s| (s / divisor) as f32).collect();
    let mean_square = square_sums.iter().map(|s| (s / divisor) as f32).collect();

    Ok((mean, mean_square, count))
}

impl Module for VirtualBatchNorm2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        validate_input(input, 4, self.num_features, "VirtualBatchNorm2d")?;

        let reference = self.reference.as_ref().ok_or_else(|| {
            TorshError::InvalidOperation(
                "VirtualBatchNorm2d requires a reference batch; call set_reference_batch() first"
                    .to_string(),
            )
        })?;

        let shape = input.shape();
        let dims = shape.dims();
        let batch_count = dims[0] * dims[2] * dims[3];

        // The *reference* moments are frozen constants; the current batch's
        // moments are recording tensor compositions, so `backward()` reaches the
        // input through the pooled statistics as well as through `x` itself.
        let batch_mean = utils::compute_channel_mean(input)?;
        let batch_mean_square = utils::compute_channel_mean(&input.pow_scalar(2.0)?)?;

        // Pool the reference batch with the current batch, weighted by size.
        let total = (reference.count + batch_count) as f32;
        let reference_weight = reference.count as f32 / total;
        let batch_weight = batch_count as f32 / total;

        let channels = self.num_features;
        let reference_mean = Tensor::from_data(
            reference
                .mean
                .iter()
                .map(|m| m * reference_weight)
                .collect::<Vec<f32>>(),
            vec![channels],
            input.device(),
        )?;
        let reference_mean_square = Tensor::from_data(
            reference
                .mean_square
                .iter()
                .map(|m| m * reference_weight)
                .collect::<Vec<f32>>(),
            vec![channels],
            input.device(),
        )?;

        let pooled_mean = reference_mean.add(&batch_mean.mul_scalar(batch_weight)?)?;
        let pooled_mean_square =
            reference_mean_square.add(&batch_mean_square.mul_scalar(batch_weight)?)?;
        // `relu` is the differentiable spelling of the `max(0.0)` clamp the
        // scalar implementation used: identical values, zero gradient where the
        // clamp bites.
        let pooled_variance = pooled_mean_square
            .sub(&pooled_mean.pow_scalar(2.0)?)?
            .relu()?;

        let (weight, bias) = affine_tensors(&self.base, self.config.affine);

        utils::apply_channel_normalization(
            input,
            &pooled_mean,
            &pooled_variance,
            weight.as_ref(),
            bias.as_ref(),
            self.config.eps,
        )
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

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }
}

impl std::fmt::Debug for VirtualBatchNorm2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VirtualBatchNorm2d")
            .field("num_features", &self.num_features)
            .field("has_reference_batch", &self.has_reference_batch())
            .finish()
    }
}

/// Clipping schedule for the batch-renormalization correction terms.
///
/// `r` is clipped to `[1/r_max, r_max]` and `d` to `[-d_max, d_max]`. Both
/// limits stay at their identity values (`r_max = 1`, `d_max = 0`) for
/// `warmup_steps` updates — during which the layer behaves exactly like batch
/// norm — and then ramp linearly to their final values over `ramp_steps`
/// further updates, as prescribed in Ioffe (2017).
#[derive(Debug, Clone, Copy)]
pub struct BatchRenormSchedule {
    /// Final upper bound on `r`.
    pub r_max: f32,
    /// Final absolute bound on `d`.
    pub d_max: f32,
    /// Number of updates before the bounds start to relax.
    pub warmup_steps: u64,
    /// Number of updates over which the bounds reach their final values.
    pub ramp_steps: u64,
}

impl Default for BatchRenormSchedule {
    fn default() -> Self {
        Self {
            r_max: 3.0,
            d_max: 5.0,
            warmup_steps: 5_000,
            ramp_steps: 35_000,
        }
    }
}

impl BatchRenormSchedule {
    /// Bounds `(r_max, d_max)` in force at `step`.
    pub fn limits_at(&self, step: u64) -> (f32, f32) {
        if step < self.warmup_steps {
            return (1.0, 0.0);
        }
        if self.ramp_steps == 0 {
            return (self.r_max, self.d_max);
        }
        let progress = ((step - self.warmup_steps) as f32 / self.ramp_steps as f32).clamp(0.0, 1.0);
        (1.0 + progress * (self.r_max - 1.0), progress * self.d_max)
    }
}

/// Batch renormalization for 2D inputs.
///
/// Corrects the training/inference mismatch of batch norm by rescaling the
/// batch-normalized activations with per-channel correction terms `r` and `d`
/// derived from the running statistics (Ioffe, "Batch Renormalization", NeurIPS
/// 2017):
///
/// ```text
/// r = clip(sigma_batch / sigma_running, 1/r_max, r_max)
/// d = clip((mu_batch - mu_running) / sigma_running, -d_max, d_max)
/// x_hat = (x - mu_batch) / sigma_batch * r + d
/// ```
///
/// `r` and `d` are treated as constants (no gradient flows through them), and
/// their bounds follow [`BatchRenormSchedule`]. In evaluation mode the layer
/// normalizes with the running statistics, exactly like batch norm.
pub struct BatchRenorm2d {
    base: ModuleBase,
    num_features: usize,
    config: NormalizationConfig,
    schedule: BatchRenormSchedule,
    stats: RunningStats,
    step: Arc<RwLock<u64>>,
}

impl BatchRenorm2d {
    /// Create the layer with the default configuration and schedule.
    pub fn new(num_features: usize) -> Result<Self> {
        Self::with_config(
            num_features,
            NormalizationConfig::default(),
            BatchRenormSchedule::default(),
        )
    }

    /// Create the layer with an explicit configuration and schedule.
    pub fn with_config(
        num_features: usize,
        config: NormalizationConfig,
        schedule: BatchRenormSchedule,
    ) -> Result<Self> {
        let mut base = ModuleBase::new();
        if config.affine {
            register_affine(&mut base, num_features)?;
        }

        Ok(Self {
            base,
            num_features,
            config,
            schedule,
            stats: RunningStats::new(num_features)?,
            step: Arc::new(RwLock::new(0)),
        })
    }

    /// Number of normalized channels.
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// Clipping schedule in use.
    pub fn schedule(&self) -> BatchRenormSchedule {
        self.schedule
    }

    /// Number of training steps performed so far.
    pub fn step(&self) -> u64 {
        *self.step.read()
    }

    /// Snapshot of the tracked running mean.
    pub fn running_mean(&self) -> Tensor {
        self.stats.running_mean()
    }

    /// Snapshot of the tracked running variance.
    pub fn running_var(&self) -> Tensor {
        self.stats.running_var()
    }
}

impl Module for BatchRenorm2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        validate_input(input, 4, self.num_features, "BatchRenorm2d")?;

        let (weight, bias) = affine_tensors(&self.base, self.config.affine);

        if !self.training() {
            // The running statistics are buffers, not graph nodes.
            return utils::apply_channel_normalization(
                input,
                &self.stats.running_mean().detach(),
                &self.stats.running_var().detach(),
                weight.as_ref(),
                bias.as_ref(),
                self.config.eps,
            );
        }

        // Batch statistics stay on the autograd graph: Ioffe's backward pass
        // propagates through `mu_B` and `sigma_B` and only stops the gradient at
        // the correction terms `r` and `d`.
        let batch_mean = utils::compute_channel_mean(input)?;
        let batch_var = utils::compute_channel_variance(input, &batch_mean)?;

        let mean_values = batch_mean.to_vec()?;
        let var_values = batch_var.to_vec()?;
        let running_mean_values = self.stats.running_mean().to_vec()?;
        let running_var_values = self.stats.running_var().to_vec()?;

        let (r_max, d_max) = self.schedule.limits_at(self.step());
        let eps = self.config.eps;

        // `r` and `d` are constants (stop-gradient), so they are built from the
        // raw values as detached per-channel tensors.
        let mut r_values = Vec::with_capacity(self.num_features);
        let mut d_values = Vec::with_capacity(self.num_features);
        for c in 0..self.num_features {
            let sigma_batch = (var_values[c] + eps).sqrt();
            let sigma_running = (running_var_values[c] + eps).sqrt();

            r_values.push((sigma_batch / sigma_running).clamp(1.0 / r_max, r_max));
            d_values.push(
                ((mean_values[c] - running_mean_values[c]) / sigma_running).clamp(-d_max, d_max),
            );
        }

        let shape = input.shape();
        let dims = shape.dims();
        let broadcast_usize: Vec<usize> = utils::channel_broadcast_shape(dims.len(), dims[1])
            .into_iter()
            .map(|d| d as usize)
            .collect();
        let broadcast = utils::channel_broadcast_shape(dims.len(), dims[1]);
        let r_tensor = Tensor::from_data(r_values, broadcast_usize.clone(), input.device())?;
        let d_tensor = Tensor::from_data(d_values, broadcast_usize, input.device())?;

        // x_hat = (x - mu_B) / sigma_B * r + d, then the affine parameters.
        let sigma = batch_var.add_scalar(eps)?.sqrt()?;
        let centered = input.sub(&batch_mean.reshape(&broadcast)?)?;
        let mut output = centered
            .div(&sigma.reshape(&broadcast)?)?
            .mul(&r_tensor)?
            .add(&d_tensor)?;

        if let Some(w) = weight.as_ref() {
            output = output.mul(&w.reshape(&broadcast)?)?;
        }
        if let Some(b) = bias.as_ref() {
            output = output.add(&b.reshape(&broadcast)?)?;
        }

        let count = samples_per_channel(dims);
        let buffered_mean = utils::detached_statistic(&batch_mean)?;
        let buffered_var = utils::detached_statistic(&batch_var)?;
        let unbiased = unbiased_variance(&buffered_var, count)?;
        self.stats
            .update(&buffered_mean, &unbiased, self.config.momentum)?;
        *self.step.write() += 1;

        Ok(output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }

    fn buffers(&self) -> Vec<Arc<RwLock<Tensor>>> {
        stats_buffers(Some(&self.stats)).into_values().collect()
    }

    fn named_buffers(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
        stats_buffers(Some(&self.stats))
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
}

impl std::fmt::Debug for BatchRenorm2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchRenorm2d")
            .field("num_features", &self.num_features)
            .field("step", &self.step())
            .finish()
    }
}
