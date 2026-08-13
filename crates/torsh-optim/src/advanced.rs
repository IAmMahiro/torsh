//! Advanced optimizers using SciRS2 optimization algorithms
//!
//! The optimizers here own their parameter groups, so `step()` performs a real
//! update, `add_param_group` extends the set of optimised tensors, and
//! `state_dict` / `load_state_dict` round-trip both the parameter groups and the
//! per-parameter moment buffers.

use crate::{
    Optimizer, OptimizerError, OptimizerResult, OptimizerState, ParamGroup, ParamGroupState,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use torsh_core::error::TorshError;
use torsh_tensor::Tensor;

/// Stable per-parameter key derived from the handle's identity.
fn param_key(param: &Arc<RwLock<Tensor>>) -> String {
    format!("param_{:p}", Arc::as_ptr(param))
}

/// Deep-copy a tensor's data into a fresh, gradient-free tensor.
///
/// Used for snapshots (slow weights, restored state) that must not observe later
/// in-place updates to the source tensor.
fn deep_copy(tensor: &Tensor) -> OptimizerResult<Tensor> {
    let data = tensor.to_vec().map_err(OptimizerError::TensorError)?;
    Tensor::from_data(data, tensor.shape().dims().to_vec(), tensor.device())
        .map_err(OptimizerError::TensorError)
}

/// Advanced Adam optimizer with SciRS2 enhancements
pub struct AdvancedAdam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub amsgrad: bool,

    /// Parameter groups optimised by this instance
    pub param_groups: Vec<ParamGroup>,

    // State variables
    pub state: HashMap<String, AdamState>,
    pub step_count: u64,

    // SciRS2 enhancements
    pub adaptive_lr: bool,
    pub gradient_clipping: Option<f64>,
    pub warmup_steps: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct AdamState {
    pub exp_avg: Tensor,
    pub exp_avg_sq: Tensor,
    pub max_exp_avg_sq: Option<Tensor>,
}

impl AdvancedAdam {
    /// Create a new advanced Adam optimizer
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
            param_groups: Vec::new(),
            state: HashMap::new(),
            step_count: 0,
            adaptive_lr: false,
            gradient_clipping: None,
            warmup_steps: None,
        }
    }

    /// Create an optimizer that already owns `params`
    pub fn with_params(lr: f64, params: Vec<Arc<RwLock<Tensor>>>) -> Self {
        let mut optimizer = Self::new(lr);
        optimizer
            .param_groups
            .push(ParamGroup::new(params, lr as f32));
        optimizer
    }

    /// Enable AMSGrad variant
    pub fn with_amsgrad(mut self) -> Self {
        self.amsgrad = true;
        self
    }

    /// Add weight decay (L2 regularization)
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Enable the adaptive (inverse square-root) learning rate schedule
    ///
    /// With this enabled the learning rate decays as `sqrt(t_ref / t)` once the
    /// step count passes `t_ref` (the warmup length, or 1 when no warmup is
    /// configured) — the "Noam" schedule used for transformer training.
    pub fn with_adaptive_lr(mut self) -> Self {
        self.adaptive_lr = true;
        self
    }

    /// Add gradient clipping
    pub fn with_gradient_clipping(mut self, max_norm: f64) -> Self {
        self.gradient_clipping = Some(max_norm);
        self
    }

    /// Add learning rate warmup
    pub fn with_warmup(mut self, warmup_steps: u64) -> Self {
        self.warmup_steps = Some(warmup_steps);
        self
    }

    /// Multiplier applied to every group's learning rate at the current step.
    ///
    /// Combines linear warmup (`step / warmup_steps`, capped at 1) with the
    /// optional inverse square-root decay.
    fn schedule_scale(&self) -> f64 {
        let step = self.step_count.max(1) as f64;
        let warmup = self.warmup_steps.unwrap_or(0);

        let warmup_scale = if warmup > 0 {
            (step / warmup as f64).min(1.0)
        } else {
            1.0
        };

        let decay_scale = if self.adaptive_lr {
            let reference = warmup.max(1) as f64;
            if step > reference {
                (reference / step).sqrt()
            } else {
                1.0
            }
        } else {
            1.0
        };

        warmup_scale * decay_scale
    }
}

impl Optimizer for AdvancedAdam {
    fn step(&mut self) -> OptimizerResult<()> {
        self.step_count += 1;
        let step = self.step_count as i32;
        let scale = self.schedule_scale();
        let bias_correction1 = 1.0 - self.beta1.powi(step);
        let bias_correction2 = 1.0 - self.beta2.powi(step);

        // Snapshot the (Arc) handles so the per-parameter state map can be
        // borrowed mutably while iterating.
        let groups: Vec<(f32, Vec<Arc<RwLock<Tensor>>>)> = self
            .param_groups
            .iter()
            .map(|group| (group.lr, group.params.clone()))
            .collect();

        for (group_lr, params) in groups {
            let effective_lr = (group_lr as f64 * scale) as f32;

            for param_arc in params {
                let mut param = param_arc.write();
                let Some(mut grad) = param.grad() else {
                    continue;
                };

                // Gradient clipping by global norm of this parameter's gradient.
                if let Some(max_norm) = self.gradient_clipping {
                    let norm =
                        grad.norm()
                            .map_err(OptimizerError::TensorError)?
                            .item()
                            .map_err(OptimizerError::TensorError)? as f64;
                    if norm > max_norm && norm > 0.0 {
                        grad = grad
                            .mul_scalar((max_norm / norm) as f32)
                            .map_err(OptimizerError::TensorError)?;
                    }
                }

                // L2 regularisation folded into the gradient.
                if self.weight_decay != 0.0 {
                    let decay = param
                        .mul_scalar(self.weight_decay as f32)
                        .map_err(OptimizerError::TensorError)?;
                    grad = grad.add(&decay).map_err(OptimizerError::TensorError)?;
                }

                let key = param_key(&param_arc);
                if !self.state.contains_key(&key) {
                    let zeros = torsh_tensor::creation::zeros_like(&param)
                        .map_err(OptimizerError::TensorError)?;
                    self.state.insert(
                        key.clone(),
                        AdamState {
                            exp_avg: zeros.clone(),
                            exp_avg_sq: zeros.clone(),
                            max_exp_avg_sq: if self.amsgrad { Some(zeros) } else { None },
                        },
                    );
                }
                let state = self
                    .state
                    .get_mut(&key)
                    .expect("state was just inserted for this key");

                // m_t = b1 * m_{t-1} + (1 - b1) * g
                let grad_term = grad
                    .mul_scalar(1.0 - self.beta1 as f32)
                    .map_err(OptimizerError::TensorError)?;
                state
                    .exp_avg
                    .mul_scalar_(self.beta1 as f32)
                    .map_err(OptimizerError::TensorError)?;
                state
                    .exp_avg
                    .add_(&grad_term)
                    .map_err(OptimizerError::TensorError)?;

                // v_t = b2 * v_{t-1} + (1 - b2) * g^2
                let grad_sq = grad.mul_op(&grad).map_err(OptimizerError::TensorError)?;
                let grad_sq_term = grad_sq
                    .mul_scalar(1.0 - self.beta2 as f32)
                    .map_err(OptimizerError::TensorError)?;
                state
                    .exp_avg_sq
                    .mul_scalar_(self.beta2 as f32)
                    .map_err(OptimizerError::TensorError)?;
                state
                    .exp_avg_sq
                    .add_(&grad_sq_term)
                    .map_err(OptimizerError::TensorError)?;

                let corrected_exp_avg = state
                    .exp_avg
                    .div_scalar(bias_correction1 as f32)
                    .map_err(OptimizerError::TensorError)?;
                let corrected_exp_avg_sq = state
                    .exp_avg_sq
                    .div_scalar(bias_correction2 as f32)
                    .map_err(OptimizerError::TensorError)?;

                // AMSGrad maxes over the bias-corrected second moments.
                let denom_source = if let Some(max_exp_avg_sq) = state.max_exp_avg_sq.as_mut() {
                    let new_max = max_exp_avg_sq
                        .maximum(&corrected_exp_avg_sq)
                        .map_err(OptimizerError::TensorError)?;
                    *max_exp_avg_sq = new_max;
                    max_exp_avg_sq.clone()
                } else {
                    corrected_exp_avg_sq
                };

                let denom = denom_source
                    .sqrt()
                    .map_err(OptimizerError::TensorError)?
                    .add_scalar(self.eps as f32)
                    .map_err(OptimizerError::TensorError)?;

                let update = corrected_exp_avg
                    .div(&denom)
                    .map_err(OptimizerError::TensorError)?
                    .mul_scalar(effective_lr)
                    .map_err(OptimizerError::TensorError)?;

                crate::param_update::sub_assign(&mut param, &update)
                    .map_err(OptimizerError::TensorError)?;
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        for group in &self.param_groups {
            for param in &group.params {
                param.write().zero_grad();
            }
        }
    }

    fn get_lr(&self) -> Vec<f32> {
        if self.param_groups.is_empty() {
            vec![self.lr as f32]
        } else {
            self.param_groups.iter().map(|group| group.lr).collect()
        }
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr as f64;
        for group in &mut self.param_groups {
            group.lr = lr;
        }
    }

    fn set_lrs(&mut self, lrs: &[f32]) {
        if let Some(&lr) = lrs.first() {
            self.lr = lr as f64;
        }
        for (group, &lr) in self.param_groups.iter_mut().zip(lrs.iter()) {
            group.lr = lr;
        }
    }

    fn add_param_group(&mut self, params: Vec<Arc<RwLock<Tensor>>>, options: HashMap<String, f32>) {
        let mut options = options;
        let lr = options.remove("lr").unwrap_or(self.lr as f32);
        let mut group = ParamGroup::new(params, lr);
        group.options = options;
        self.param_groups.push(group);
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        self.param_groups
            .iter()
            .flat_map(|group| group.params.iter().cloned())
            .collect()
    }

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        let mut global_state = HashMap::new();
        global_state.insert("lr".to_string(), self.lr as f32);
        global_state.insert("beta1".to_string(), self.beta1 as f32);
        global_state.insert("beta2".to_string(), self.beta2 as f32);
        global_state.insert("eps".to_string(), self.eps as f32);
        global_state.insert("weight_decay".to_string(), self.weight_decay as f32);
        global_state.insert("step_count".to_string(), self.step_count as f32);
        global_state.insert("amsgrad".to_string(), if self.amsgrad { 1.0 } else { 0.0 });

        let param_groups = self
            .param_groups
            .iter()
            .map(|group| ParamGroupState {
                lr: group.lr,
                options: group.options.clone(),
                param_count: group.params.len(),
            })
            .collect();

        // Per-parameter moment buffers, keyed the same way `step` keys them.
        let mut state = HashMap::new();
        for (key, adam_state) in &self.state {
            let mut entry = HashMap::new();
            entry.insert("exp_avg".to_string(), adam_state.exp_avg.clone());
            entry.insert("exp_avg_sq".to_string(), adam_state.exp_avg_sq.clone());
            if let Some(max_exp_avg_sq) = &adam_state.max_exp_avg_sq {
                entry.insert("max_exp_avg_sq".to_string(), max_exp_avg_sq.clone());
            }
            state.insert(key.clone(), entry);
        }

        Ok(OptimizerState {
            optimizer_type: "AdvancedAdam".to_string(),
            version: "1.0".to_string(),
            param_groups,
            state,
            global_state,
        })
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        if state.optimizer_type != "AdvancedAdam" {
            return Err(OptimizerError::InvalidParameter(format!(
                "Expected AdvancedAdam, got {}",
                state.optimizer_type
            )));
        }

        if let Some(lr) = state.global_state.get("lr") {
            self.lr = *lr as f64;
        }
        if let Some(beta1) = state.global_state.get("beta1") {
            self.beta1 = *beta1 as f64;
        }
        if let Some(beta2) = state.global_state.get("beta2") {
            self.beta2 = *beta2 as f64;
        }
        if let Some(eps) = state.global_state.get("eps") {
            self.eps = *eps as f64;
        }
        if let Some(weight_decay) = state.global_state.get("weight_decay") {
            self.weight_decay = *weight_decay as f64;
        }
        if let Some(step_count) = state.global_state.get("step_count") {
            self.step_count = *step_count as u64;
        }
        if let Some(amsgrad) = state.global_state.get("amsgrad") {
            self.amsgrad = *amsgrad != 0.0;
        }

        // Restore per-group learning rates for the groups this optimizer owns.
        for (group, saved) in self.param_groups.iter_mut().zip(state.param_groups.iter()) {
            group.lr = saved.lr;
            group.options = saved.options.clone();
        }

        // Restore per-parameter moment buffers.
        self.state.clear();
        for (key, entry) in state.state {
            let exp_avg = entry.get("exp_avg").ok_or_else(|| {
                OptimizerError::StateError(format!("AdvancedAdam state for {key} has no exp_avg"))
            })?;
            let exp_avg_sq = entry.get("exp_avg_sq").ok_or_else(|| {
                OptimizerError::StateError(format!(
                    "AdvancedAdam state for {key} has no exp_avg_sq"
                ))
            })?;
            let max_exp_avg_sq = entry.get("max_exp_avg_sq").map(deep_copy).transpose()?;
            self.state.insert(
                key,
                AdamState {
                    exp_avg: deep_copy(exp_avg)?,
                    exp_avg_sq: deep_copy(exp_avg_sq)?,
                    max_exp_avg_sq,
                },
            );
        }

        Ok(())
    }
}

/// LAMB (Layer-wise Adaptive Moments optimizer for Batch training)
/// Particularly effective for large batch training
pub struct LAMB {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub bias_correction: bool,

    /// Parameter groups optimised by this instance
    pub param_groups: Vec<ParamGroup>,

    pub state: HashMap<String, LambState>,
    pub step_count: u64,
}

#[derive(Debug, Clone)]
pub struct LambState {
    pub exp_avg: Tensor,
    pub exp_avg_sq: Tensor,
}

impl LAMB {
    /// Create a new LAMB optimizer
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-6,
            weight_decay: 0.01,
            bias_correction: true,
            param_groups: Vec::new(),
            state: HashMap::new(),
            step_count: 0,
        }
    }

    /// Create an optimizer that already owns `params`
    pub fn with_params(lr: f64, params: Vec<Arc<RwLock<Tensor>>>) -> Self {
        let mut optimizer = Self::new(lr);
        optimizer
            .param_groups
            .push(ParamGroup::new(params, lr as f32));
        optimizer
    }
}

impl Optimizer for LAMB {
    fn step(&mut self) -> OptimizerResult<()> {
        self.step_count += 1;
        let step = self.step_count as i32;
        let (bias_correction1, bias_correction2) = if self.bias_correction {
            (1.0 - self.beta1.powi(step), 1.0 - self.beta2.powi(step))
        } else {
            (1.0, 1.0)
        };

        let groups: Vec<(f32, Vec<Arc<RwLock<Tensor>>>)> = self
            .param_groups
            .iter()
            .map(|group| (group.lr, group.params.clone()))
            .collect();

        for (group_lr, params) in groups {
            for param_arc in params {
                let mut param = param_arc.write();
                let Some(grad) = param.grad() else {
                    continue;
                };

                let key = param_key(&param_arc);
                if !self.state.contains_key(&key) {
                    let zeros = torsh_tensor::creation::zeros_like(&param)
                        .map_err(OptimizerError::TensorError)?;
                    self.state.insert(
                        key.clone(),
                        LambState {
                            exp_avg: zeros.clone(),
                            exp_avg_sq: zeros,
                        },
                    );
                }
                let state = self
                    .state
                    .get_mut(&key)
                    .expect("state was just inserted for this key");

                let grad_term = grad
                    .mul_scalar(1.0 - self.beta1 as f32)
                    .map_err(OptimizerError::TensorError)?;
                state
                    .exp_avg
                    .mul_scalar_(self.beta1 as f32)
                    .map_err(OptimizerError::TensorError)?;
                state
                    .exp_avg
                    .add_(&grad_term)
                    .map_err(OptimizerError::TensorError)?;

                let grad_sq = grad.mul_op(&grad).map_err(OptimizerError::TensorError)?;
                let grad_sq_term = grad_sq
                    .mul_scalar(1.0 - self.beta2 as f32)
                    .map_err(OptimizerError::TensorError)?;
                state
                    .exp_avg_sq
                    .mul_scalar_(self.beta2 as f32)
                    .map_err(OptimizerError::TensorError)?;
                state
                    .exp_avg_sq
                    .add_(&grad_sq_term)
                    .map_err(OptimizerError::TensorError)?;

                let corrected_exp_avg = state
                    .exp_avg
                    .div_scalar(bias_correction1 as f32)
                    .map_err(OptimizerError::TensorError)?;
                let corrected_exp_avg_sq = state
                    .exp_avg_sq
                    .div_scalar(bias_correction2 as f32)
                    .map_err(OptimizerError::TensorError)?;

                let denom = corrected_exp_avg_sq
                    .sqrt()
                    .map_err(OptimizerError::TensorError)?
                    .add_scalar(self.eps as f32)
                    .map_err(OptimizerError::TensorError)?;

                // Adam direction plus decoupled weight decay.
                let mut direction = corrected_exp_avg
                    .div(&denom)
                    .map_err(OptimizerError::TensorError)?;
                if self.weight_decay != 0.0 {
                    let decay = param
                        .mul_scalar(self.weight_decay as f32)
                        .map_err(OptimizerError::TensorError)?;
                    direction = direction.add(&decay).map_err(OptimizerError::TensorError)?;
                }

                // Layer-wise trust ratio: ||w|| / ||r||, defaulting to 1 when
                // either norm vanishes (as in the LAMB paper).
                let param_norm = param
                    .norm()
                    .map_err(OptimizerError::TensorError)?
                    .item()
                    .map_err(OptimizerError::TensorError)?;
                let direction_norm = direction
                    .norm()
                    .map_err(OptimizerError::TensorError)?
                    .item()
                    .map_err(OptimizerError::TensorError)?;
                let trust_ratio = if param_norm > 0.0 && direction_norm > 0.0 {
                    param_norm / direction_norm
                } else {
                    1.0
                };

                let update = direction
                    .mul_scalar(group_lr * trust_ratio)
                    .map_err(OptimizerError::TensorError)?;
                crate::param_update::sub_assign(&mut param, &update)
                    .map_err(OptimizerError::TensorError)?;
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        for group in &self.param_groups {
            for param in &group.params {
                param.write().zero_grad();
            }
        }
    }

    fn get_lr(&self) -> Vec<f32> {
        if self.param_groups.is_empty() {
            vec![self.lr as f32]
        } else {
            self.param_groups.iter().map(|group| group.lr).collect()
        }
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr as f64;
        for group in &mut self.param_groups {
            group.lr = lr;
        }
    }

    fn set_lrs(&mut self, lrs: &[f32]) {
        if let Some(&lr) = lrs.first() {
            self.lr = lr as f64;
        }
        for (group, &lr) in self.param_groups.iter_mut().zip(lrs.iter()) {
            group.lr = lr;
        }
    }

    fn add_param_group(&mut self, params: Vec<Arc<RwLock<Tensor>>>, options: HashMap<String, f32>) {
        let mut options = options;
        let lr = options.remove("lr").unwrap_or(self.lr as f32);
        let mut group = ParamGroup::new(params, lr);
        group.options = options;
        self.param_groups.push(group);
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        self.param_groups
            .iter()
            .flat_map(|group| group.params.iter().cloned())
            .collect()
    }

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        let mut global_state = HashMap::new();
        global_state.insert("lr".to_string(), self.lr as f32);
        global_state.insert("beta1".to_string(), self.beta1 as f32);
        global_state.insert("beta2".to_string(), self.beta2 as f32);
        global_state.insert("eps".to_string(), self.eps as f32);
        global_state.insert("weight_decay".to_string(), self.weight_decay as f32);
        global_state.insert("step_count".to_string(), self.step_count as f32);
        global_state.insert(
            "bias_correction".to_string(),
            if self.bias_correction { 1.0 } else { 0.0 },
        );

        let param_groups = self
            .param_groups
            .iter()
            .map(|group| ParamGroupState {
                lr: group.lr,
                options: group.options.clone(),
                param_count: group.params.len(),
            })
            .collect();

        let mut state = HashMap::new();
        for (key, lamb_state) in &self.state {
            let mut entry = HashMap::new();
            entry.insert("exp_avg".to_string(), lamb_state.exp_avg.clone());
            entry.insert("exp_avg_sq".to_string(), lamb_state.exp_avg_sq.clone());
            state.insert(key.clone(), entry);
        }

        Ok(OptimizerState {
            optimizer_type: "LAMB".to_string(),
            version: "1.0".to_string(),
            param_groups,
            state,
            global_state,
        })
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        if state.optimizer_type != "LAMB" {
            return Err(OptimizerError::InvalidParameter(format!(
                "Expected LAMB, got {}",
                state.optimizer_type
            )));
        }

        if let Some(lr) = state.global_state.get("lr") {
            self.lr = *lr as f64;
        }
        if let Some(beta1) = state.global_state.get("beta1") {
            self.beta1 = *beta1 as f64;
        }
        if let Some(beta2) = state.global_state.get("beta2") {
            self.beta2 = *beta2 as f64;
        }
        if let Some(eps) = state.global_state.get("eps") {
            self.eps = *eps as f64;
        }
        if let Some(weight_decay) = state.global_state.get("weight_decay") {
            self.weight_decay = *weight_decay as f64;
        }
        if let Some(step_count) = state.global_state.get("step_count") {
            self.step_count = *step_count as u64;
        }
        if let Some(bias_correction) = state.global_state.get("bias_correction") {
            self.bias_correction = *bias_correction != 0.0;
        }

        for (group, saved) in self.param_groups.iter_mut().zip(state.param_groups.iter()) {
            group.lr = saved.lr;
            group.options = saved.options.clone();
        }

        self.state.clear();
        for (key, entry) in state.state {
            let exp_avg = entry.get("exp_avg").ok_or_else(|| {
                OptimizerError::StateError(format!("LAMB state for {key} has no exp_avg"))
            })?;
            let exp_avg_sq = entry.get("exp_avg_sq").ok_or_else(|| {
                OptimizerError::StateError(format!("LAMB state for {key} has no exp_avg_sq"))
            })?;
            self.state.insert(
                key,
                LambState {
                    exp_avg: deep_copy(exp_avg)?,
                    exp_avg_sq: deep_copy(exp_avg_sq)?,
                },
            );
        }

        Ok(())
    }
}

/// Lookahead optimizer wrapper
///
/// Wraps any optimizer that exposes its parameters via
/// [`Optimizer::parameters`]. Every `k` fast-weight steps the slow weights are
/// pulled a fraction `alpha` towards the fast weights and the fast weights are
/// reset onto them: `phi <- phi + alpha * (theta - phi)`, `theta <- phi`.
pub struct Lookahead<T: Optimizer> {
    pub base_optimizer: T,
    pub alpha: f64,
    pub k: u64,

    pub slow_weights: HashMap<String, Tensor>,
    pub step_count: u64,
}

impl<T: Optimizer> Lookahead<T> {
    /// Create a new Lookahead optimizer
    pub fn new(base_optimizer: T, alpha: f64, k: u64) -> Self {
        Self {
            base_optimizer,
            alpha,
            k,
            slow_weights: HashMap::new(),
            step_count: 0,
        }
    }

    /// Snapshot the current fast weights as slow weights, for parameters that do
    /// not have a snapshot yet.
    fn initialize_slow_weights(&mut self) -> OptimizerResult<()> {
        for param in self.base_optimizer.parameters() {
            let key = param_key(&param);
            if !self.slow_weights.contains_key(&key) {
                let snapshot = deep_copy(&param.read())?;
                self.slow_weights.insert(key, snapshot);
            }
        }
        Ok(())
    }

    /// Perform the slow/fast synchronisation.
    fn synchronize(&mut self) -> OptimizerResult<()> {
        for param in self.base_optimizer.parameters() {
            let key = param_key(&param);
            let Some(slow) = self.slow_weights.get_mut(&key) else {
                continue;
            };

            let mut fast = param.write();
            // phi <- phi + alpha * (theta - phi)
            let diff = fast
                .detach()
                .sub(slow)
                .map_err(OptimizerError::TensorError)?
                .mul_scalar(self.alpha as f32)
                .map_err(OptimizerError::TensorError)?;
            slow.add_(&diff).map_err(OptimizerError::TensorError)?;

            // theta <- phi
            crate::param_update::assign(&mut fast, slow).map_err(OptimizerError::TensorError)?;
        }
        Ok(())
    }
}

impl<T: Optimizer> Optimizer for Lookahead<T> {
    fn step(&mut self) -> OptimizerResult<()> {
        self.initialize_slow_weights()?;

        // Perform base optimizer step
        self.base_optimizer.step()?;
        self.step_count += 1;

        if self.k > 0 && self.step_count % self.k == 0 {
            self.synchronize()?;
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        self.base_optimizer.zero_grad();
    }

    fn get_lr(&self) -> Vec<f32> {
        self.base_optimizer.get_lr()
    }

    fn set_lr(&mut self, lr: f32) {
        self.base_optimizer.set_lr(lr);
    }

    fn set_lrs(&mut self, lrs: &[f32]) {
        self.base_optimizer.set_lrs(lrs);
    }

    fn add_param_group(&mut self, params: Vec<Arc<RwLock<Tensor>>>, options: HashMap<String, f32>) {
        self.base_optimizer.add_param_group(params, options);
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        self.base_optimizer.parameters()
    }

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        let mut base_state = self.base_optimizer.state_dict()?;

        // Add Lookahead-specific state
        base_state
            .global_state
            .insert("alpha".to_string(), self.alpha as f32);
        base_state
            .global_state
            .insert("k".to_string(), self.k as f32);
        base_state
            .global_state
            .insert("step_count".to_string(), self.step_count as f32);

        // Slow weights live alongside the base optimizer's per-parameter state,
        // under a reserved key so they round-trip through the same state dict.
        for (key, slow) in &self.slow_weights {
            base_state
                .state
                .entry(key.clone())
                .or_default()
                .insert("lookahead_slow_weight".to_string(), slow.clone());
        }

        base_state.optimizer_type = format!("Lookahead<{}>", base_state.optimizer_type);

        Ok(base_state)
    }

    fn load_state_dict(&mut self, mut state: OptimizerState) -> OptimizerResult<()> {
        // Extract Lookahead-specific state
        if let Some(alpha) = state.global_state.remove("alpha") {
            self.alpha = alpha as f64;
        }
        if let Some(k) = state.global_state.remove("k") {
            self.k = k as u64;
        }
        if let Some(step_count) = state.global_state.remove("step_count") {
            self.step_count = step_count as u64;
        }

        // Pull the slow weights back out before handing the rest to the base.
        //
        // The lookup is driven by the *parameters*, never by iteration order over
        // the state map: `HashMap` iteration is unordered, so pairing entries
        // positionally would attach each slow weight to an arbitrary parameter.
        // A parameter whose key is absent from the checkpoint simply gets no slow
        // weight and is re-snapshotted by `initialize_slow_weights` on the next
        // step. (Keys are derived from handle addresses, so — as for every
        // optimizer in this crate — they identify parameters only within the
        // process that produced them.)
        self.slow_weights.clear();
        for param in self.base_optimizer.parameters() {
            let key = param_key(&param);
            if let Some(entry) = state.state.get_mut(&key) {
                if let Some(slow) = entry.remove("lookahead_slow_weight") {
                    self.slow_weights.insert(key, slow);
                }
            }
        }
        // Drop any slow weights belonging to parameters this optimizer no longer
        // holds, so they cannot leak into the base optimizer's state.
        for entry in state.state.values_mut() {
            entry.remove("lookahead_slow_weight");
        }

        // Restore base optimizer type
        if state.optimizer_type.starts_with("Lookahead<") && state.optimizer_type.ends_with(">") {
            let base_type = &state.optimizer_type[10..state.optimizer_type.len() - 1];
            state.optimizer_type = base_type.to_string();
        }

        // Load base optimizer state
        self.base_optimizer.load_state_dict(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    fn make_param(data: Vec<f32>) -> Arc<RwLock<Tensor>> {
        let len = data.len();
        let tensor = Tensor::from_data(data, vec![len], DeviceType::Cpu)
            .expect("parameter creation")
            .requires_grad_(true);
        Arc::new(RwLock::new(tensor))
    }

    fn set_grad(param: &Arc<RwLock<Tensor>>, data: Vec<f32>) {
        let len = data.len();
        let grad = Tensor::from_data(data, vec![len], DeviceType::Cpu).expect("gradient creation");
        param.read().set_grad(Some(grad));
    }

    #[test]
    fn test_advanced_adam() {
        let mut optimizer = AdvancedAdam::new(0.001)
            .with_amsgrad()
            .with_weight_decay(0.01)
            .with_gradient_clipping(1.0);

        // Test interface
        assert_eq!(optimizer.get_lr(), vec![0.001]);
        assert!(optimizer.step().is_ok());
    }

    #[test]
    fn test_advanced_adam_updates_parameters() {
        let param = make_param(vec![1.0, 1.0]);
        set_grad(&param, vec![1.0, 1.0]);

        let mut optimizer = AdvancedAdam::with_params(0.1, vec![Arc::clone(&param)]);
        optimizer.step().expect("step");
        let after_first = param.read().to_vec().expect("to_vec");
        assert!(
            after_first[0] < 1.0,
            "parameter must move, got {after_first:?}"
        );

        optimizer.step().expect("step");
        let after_second = param.read().to_vec().expect("to_vec");
        assert!(
            after_second[0] < after_first[0],
            "second step must move further: {after_first:?} -> {after_second:?}"
        );
    }

    #[test]
    fn test_advanced_adam_add_param_group() {
        let first = make_param(vec![1.0]);
        let second = make_param(vec![1.0]);
        set_grad(&first, vec![1.0]);
        set_grad(&second, vec![1.0]);

        let mut optimizer = AdvancedAdam::with_params(0.1, vec![Arc::clone(&first)]);
        let mut options = HashMap::new();
        options.insert("lr".to_string(), 0.5);
        optimizer.add_param_group(vec![Arc::clone(&second)], options);

        assert_eq!(optimizer.parameters().len(), 2);
        assert_eq!(optimizer.get_lr(), vec![0.1, 0.5]);

        optimizer.step().expect("step");
        assert!(first.read().to_vec().expect("to_vec")[0] < 1.0);
        assert!(second.read().to_vec().expect("to_vec")[0] < 1.0);
    }

    #[test]
    fn test_advanced_adam_state_dict_round_trip() {
        let param = make_param(vec![1.0, 2.0]);
        set_grad(&param, vec![0.5, 0.5]);

        let mut optimizer = AdvancedAdam::with_params(0.1, vec![Arc::clone(&param)]);
        optimizer.step().expect("step");

        let dict = optimizer.state_dict().expect("state_dict");
        assert_eq!(dict.param_groups.len(), 1);
        assert_eq!(dict.param_groups[0].param_count, 1);
        assert_eq!(dict.state.len(), 1);
        assert!(dict
            .state
            .values()
            .next()
            .expect("state entry")
            .contains_key("exp_avg"));

        let mut restored = AdvancedAdam::with_params(0.0, vec![Arc::clone(&param)]);
        restored.load_state_dict(dict).expect("load_state_dict");
        assert_eq!(restored.step_count, 1);
        assert_eq!(restored.state.len(), 1);
    }

    #[test]
    fn test_lamb_optimizer() {
        let mut optimizer = LAMB::new(0.001);

        // Test interface
        assert_eq!(optimizer.get_lr(), vec![0.001]);
        assert!(optimizer.step().is_ok());
    }

    #[test]
    fn test_lamb_updates_parameters_with_trust_ratio() {
        let param = make_param(vec![1.0, 1.0]);
        set_grad(&param, vec![1.0, 1.0]);

        let mut optimizer = LAMB::with_params(0.01, vec![Arc::clone(&param)]);
        optimizer.step().expect("step");

        let after = param.read().to_vec().expect("to_vec");
        assert!(
            after[0] < 1.0,
            "LAMB must move the parameter, got {after:?}"
        );
        assert!(after[0] > 0.0, "trust ratio must keep the step bounded");
    }

    #[test]
    fn test_lookahead_wrapper() {
        let base_optimizer = AdvancedAdam::new(0.001);
        let mut lookahead = Lookahead::new(base_optimizer, 0.5, 5);

        // Test interface
        assert_eq!(lookahead.get_lr(), vec![0.001]);
        assert!(lookahead.step().is_ok());
    }

    #[test]
    fn test_lookahead_state_dict_round_trip_keeps_slow_weights_per_parameter() {
        let first = make_param(vec![10.0]);
        let second = make_param(vec![-10.0]);
        set_grad(&first, vec![1.0]);
        set_grad(&second, vec![1.0]);

        let base = AdvancedAdam::with_params(0.1, vec![Arc::clone(&first), Arc::clone(&second)]);
        let mut lookahead = Lookahead::new(base, 0.5, 1);
        lookahead.step().expect("step");

        let expected: HashMap<String, f32> = lookahead
            .slow_weights
            .iter()
            .map(|(key, tensor)| (key.clone(), tensor.to_vec().expect("to_vec")[0]))
            .collect();
        assert_eq!(expected.len(), 2);

        let dict = lookahead.state_dict().expect("state_dict");
        let restored_base =
            AdvancedAdam::with_params(0.1, vec![Arc::clone(&first), Arc::clone(&second)]);
        let mut restored = Lookahead::new(restored_base, 0.0, 1);
        restored.load_state_dict(dict).expect("load_state_dict");

        assert_eq!(restored.slow_weights.len(), 2);
        for (key, value) in expected {
            let got = restored
                .slow_weights
                .get(&key)
                .unwrap_or_else(|| panic!("slow weight for {key} must be restored"))
                .to_vec()
                .expect("to_vec")[0];
            assert!(
                (got - value).abs() < 1e-6,
                "slow weight for {key} must land on the same parameter: {value} vs {got}"
            );
        }
    }

    #[test]
    fn test_lookahead_updates_slow_weights_every_k_steps() {
        let param = make_param(vec![0.0]);
        set_grad(&param, vec![1.0]);

        let base = AdvancedAdam::with_params(0.1, vec![Arc::clone(&param)]);
        let mut lookahead = Lookahead::new(base, 0.5, 2);

        lookahead.step().expect("step 1");
        let fast_after_one = param.read().to_vec().expect("to_vec")[0];
        // Slow weights still hold the initial value after a non-sync step.
        let slow = lookahead
            .slow_weights
            .values()
            .next()
            .expect("slow weight")
            .to_vec()
            .expect("to_vec")[0];
        assert!((slow - 0.0).abs() < 1e-6, "slow weight must not move yet");

        lookahead.step().expect("step 2");
        let slow = lookahead
            .slow_weights
            .values()
            .next()
            .expect("slow weight")
            .to_vec()
            .expect("to_vec")[0];
        let fast = param.read().to_vec().expect("to_vec")[0];
        assert!(
            slow < 0.0,
            "slow weight must be pulled towards the fast weights at step k"
        );
        assert!(
            (fast - slow).abs() < 1e-6,
            "fast weights must be reset onto the slow weights: {fast} vs {slow}"
        );
        // Two Adam steps at lr = 0.1 move the fast weight to about -0.2, and
        // alpha = 0.5 places the slow weight halfway there: strictly between the
        // starting point and the fast trajectory, never beyond it.
        assert!(
            slow > 2.0 * fast_after_one && slow < 0.0,
            "the interpolated slow weight must lag the fast trajectory, got {slow}"
        );
    }
}
