//! Learning rate schedulers - PyTorch-compatible `torsh.optim.lr_scheduler` family
//!
//! Each scheduler holds a `Py<PyAny>` handle to any `Optimizer` subclass
//! instance (SGD, Adam, AdamW, Adagrad, RMSprop, ...) and drives its `lr`
//! property directly -- exactly the way a Python user would
//! (`optimizer.lr = new_lr`). The Rust `torsh_optim::lr_scheduler` types are
//! generic over `O: torsh_optim::Optimizer` and own their optimizer by
//! value, so they cannot wrap a `Py<PyAny>` handle; the formulas below are
//! therefore reimplemented directly against the optimizer's Python-visible
//! `lr` attribute, matching `crates/torsh-optim/src/lr_scheduler.rs` and
//! `lr_scheduler_additional.rs` exactly (consult those first if a formula
//! below ever needs to change).
//!
//! Every scheduler starts with `last_epoch = 0` and `last_lr` equal to the
//! optimizer's learning rate at construction time; the schedule formula only
//! takes effect starting from the first `step()` call. This mirrors
//! `torsh_optim::lr_scheduler::BaseScheduler::new`, which seeds `last_lr`
//! from `optimizer.get_lr()` unmodified.

use super::base::get_field;
use crate::error::PyResult;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule, PyModuleMethods};
use std::collections::HashMap;

/// Read the current learning rate from any `PyOptimizer` subclass via its
/// `lr` property.
fn get_optimizer_lr(py: Python<'_>, optimizer: &Py<PyAny>) -> PyResult<f32> {
    optimizer.bind(py).getattr("lr")?.extract()
}

/// Write a new learning rate onto any `PyOptimizer` subclass via its `lr`
/// property. This invokes the real `#[setter]`, which (for every concrete
/// optimizer in this crate) also updates the wrapped optimizer's own
/// internal state used by `step()`, not just the Python-visible attribute.
fn set_optimizer_lr(py: Python<'_>, optimizer: &Py<PyAny>, lr: f32) -> PyResult<()> {
    optimizer.bind(py).setattr("lr", lr)
}

/// Extract an `f32` field from a scheduler's `load_state_dict` input.
fn extract_f32(py: Python<'_>, state: &HashMap<String, Py<PyAny>>, key: &str) -> PyResult<f32> {
    get_field(state, key)?.extract(py)
}

/// Extract an `i32` field from a scheduler's `load_state_dict` input.
fn extract_i32(py: Python<'_>, state: &HashMap<String, Py<PyAny>>, key: &str) -> PyResult<i32> {
    get_field(state, key)?.extract(py)
}

/// Insert `$val` into the `HashMap<String, Py<PyAny>>` `$map`, converting it
/// via `IntoPyObject` and matching the `.expect(...)` house style used
/// throughout `optim/*.rs`. Cuts the boilerplate of the (very repetitive)
/// `state_dict()` bodies below.
macro_rules! py_insert {
    ($map:expr, $py:expr, $key:expr, $val:expr) => {
        $map.insert(
            $key.to_string(),
            $val.into_pyobject($py)
                .expect("Python object conversion should succeed")
                .into_any()
                .unbind(),
        );
    };
}

// ===========================================================================
// StepLR
// ===========================================================================

/// Decays the learning rate by `gamma` every `step_size` epochs:
/// `lr = base_lr * gamma ** (last_epoch // step_size)`.
#[pyclass(name = "StepLR")]
pub struct PyStepLR {
    optimizer: Py<PyAny>,
    base_lr: f32,
    last_lr: f32,
    last_epoch: i32,
    step_size: i32,
    gamma: f32,
}

#[pymethods]
impl PyStepLR {
    #[new]
    fn new(optimizer: Py<PyAny>, step_size: i32, gamma: f32) -> PyResult<Self> {
        let base_lr = Python::attach(|py| get_optimizer_lr(py, &optimizer))?;
        Ok(Self {
            optimizer,
            base_lr,
            last_lr: base_lr,
            last_epoch: 0,
            step_size,
            gamma,
        })
    }

    /// Advance the schedule by one epoch and update the optimizer's `lr`.
    fn step(&mut self) -> PyResult<()> {
        self.last_epoch += 1;
        let num_steps = self.last_epoch / self.step_size;
        let new_lr = self.base_lr * self.gamma.powi(num_steps);
        Python::attach(|py| set_optimizer_lr(py, &self.optimizer, new_lr))?;
        self.last_lr = new_lr;
        Ok(())
    }

    /// Learning rate(s) as of the most recent `step()` call (or the base
    /// learning rate if `step()` has not been called yet).
    fn get_last_lr(&self) -> Vec<f32> {
        vec![self.last_lr]
    }

    #[getter]
    fn last_epoch(&self) -> i32 {
        self.last_epoch
    }

    fn state_dict(&self) -> PyResult<HashMap<String, Py<PyAny>>> {
        Python::attach(|py| {
            let mut state: HashMap<String, Py<PyAny>> = HashMap::new();
            py_insert!(state, py, "last_epoch", self.last_epoch);
            py_insert!(state, py, "base_lr", self.base_lr);
            py_insert!(state, py, "last_lr", self.last_lr);
            py_insert!(state, py, "step_size", self.step_size);
            py_insert!(state, py, "gamma", self.gamma);
            Ok(state)
        })
    }

    fn load_state_dict(&mut self, state: HashMap<String, Py<PyAny>>) -> PyResult<()> {
        Python::attach(|py| {
            self.last_epoch = extract_i32(py, &state, "last_epoch")?;
            self.base_lr = extract_f32(py, &state, "base_lr")?;
            self.last_lr = extract_f32(py, &state, "last_lr")?;
            self.step_size = extract_i32(py, &state, "step_size")?;
            self.gamma = extract_f32(py, &state, "gamma")?;
            set_optimizer_lr(py, &self.optimizer, self.last_lr)
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "StepLR(step_size={}, gamma={}, last_epoch={})",
            self.step_size, self.gamma, self.last_epoch
        )
    }
}

// ===========================================================================
// MultiStepLR
// ===========================================================================

/// Decays the learning rate by `gamma` once the epoch count reaches each
/// milestone: `lr = base_lr * gamma ** (number of milestones <=
/// last_epoch)`.
#[pyclass(name = "MultiStepLR")]
pub struct PyMultiStepLR {
    optimizer: Py<PyAny>,
    base_lr: f32,
    last_lr: f32,
    last_epoch: i32,
    milestones: Vec<i32>,
    gamma: f32,
}

#[pymethods]
impl PyMultiStepLR {
    #[new]
    fn new(optimizer: Py<PyAny>, milestones: Vec<i32>, gamma: f32) -> PyResult<Self> {
        let base_lr = Python::attach(|py| get_optimizer_lr(py, &optimizer))?;
        let mut milestones = milestones;
        milestones.sort_unstable();
        Ok(Self {
            optimizer,
            base_lr,
            last_lr: base_lr,
            last_epoch: 0,
            milestones,
            gamma,
        })
    }

    fn step(&mut self) -> PyResult<()> {
        self.last_epoch += 1;
        let num_milestones_passed = self
            .milestones
            .iter()
            .filter(|&&milestone| self.last_epoch >= milestone)
            .count() as i32;
        let new_lr = self.base_lr * self.gamma.powi(num_milestones_passed);
        Python::attach(|py| set_optimizer_lr(py, &self.optimizer, new_lr))?;
        self.last_lr = new_lr;
        Ok(())
    }

    fn get_last_lr(&self) -> Vec<f32> {
        vec![self.last_lr]
    }

    #[getter]
    fn last_epoch(&self) -> i32 {
        self.last_epoch
    }

    fn state_dict(&self) -> PyResult<HashMap<String, Py<PyAny>>> {
        Python::attach(|py| {
            let mut state: HashMap<String, Py<PyAny>> = HashMap::new();
            py_insert!(state, py, "last_epoch", self.last_epoch);
            py_insert!(state, py, "base_lr", self.base_lr);
            py_insert!(state, py, "last_lr", self.last_lr);
            py_insert!(state, py, "milestones", self.milestones.clone());
            py_insert!(state, py, "gamma", self.gamma);
            Ok(state)
        })
    }

    fn load_state_dict(&mut self, state: HashMap<String, Py<PyAny>>) -> PyResult<()> {
        Python::attach(|py| {
            self.last_epoch = extract_i32(py, &state, "last_epoch")?;
            self.base_lr = extract_f32(py, &state, "base_lr")?;
            self.last_lr = extract_f32(py, &state, "last_lr")?;
            self.milestones = get_field(&state, "milestones")?.extract(py)?;
            self.gamma = extract_f32(py, &state, "gamma")?;
            set_optimizer_lr(py, &self.optimizer, self.last_lr)
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "MultiStepLR(milestones={:?}, gamma={}, last_epoch={})",
            self.milestones, self.gamma, self.last_epoch
        )
    }
}

// ===========================================================================
// ExponentialLR
// ===========================================================================

/// Decays the learning rate by `gamma` every epoch:
/// `lr = base_lr * gamma ** last_epoch`.
#[pyclass(name = "ExponentialLR")]
pub struct PyExponentialLR {
    optimizer: Py<PyAny>,
    base_lr: f32,
    last_lr: f32,
    last_epoch: i32,
    gamma: f32,
}

#[pymethods]
impl PyExponentialLR {
    #[new]
    fn new(optimizer: Py<PyAny>, gamma: f32) -> PyResult<Self> {
        let base_lr = Python::attach(|py| get_optimizer_lr(py, &optimizer))?;
        Ok(Self {
            optimizer,
            base_lr,
            last_lr: base_lr,
            last_epoch: 0,
            gamma,
        })
    }

    fn step(&mut self) -> PyResult<()> {
        self.last_epoch += 1;
        let new_lr = self.base_lr * self.gamma.powi(self.last_epoch);
        Python::attach(|py| set_optimizer_lr(py, &self.optimizer, new_lr))?;
        self.last_lr = new_lr;
        Ok(())
    }

    fn get_last_lr(&self) -> Vec<f32> {
        vec![self.last_lr]
    }

    #[getter]
    fn last_epoch(&self) -> i32 {
        self.last_epoch
    }

    fn state_dict(&self) -> PyResult<HashMap<String, Py<PyAny>>> {
        Python::attach(|py| {
            let mut state: HashMap<String, Py<PyAny>> = HashMap::new();
            py_insert!(state, py, "last_epoch", self.last_epoch);
            py_insert!(state, py, "base_lr", self.base_lr);
            py_insert!(state, py, "last_lr", self.last_lr);
            py_insert!(state, py, "gamma", self.gamma);
            Ok(state)
        })
    }

    fn load_state_dict(&mut self, state: HashMap<String, Py<PyAny>>) -> PyResult<()> {
        Python::attach(|py| {
            self.last_epoch = extract_i32(py, &state, "last_epoch")?;
            self.base_lr = extract_f32(py, &state, "base_lr")?;
            self.last_lr = extract_f32(py, &state, "last_lr")?;
            self.gamma = extract_f32(py, &state, "gamma")?;
            set_optimizer_lr(py, &self.optimizer, self.last_lr)
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ExponentialLR(gamma={}, last_epoch={})",
            self.gamma, self.last_epoch
        )
    }
}

// ===========================================================================
// CosineAnnealingLR
// ===========================================================================

/// Anneals the learning rate following a half-cosine curve down to
/// `eta_min` over `t_max` epochs:
/// `lr = eta_min + (base_lr - eta_min) * (1 + cos(pi * last_epoch / t_max)) / 2`.
#[pyclass(name = "CosineAnnealingLR")]
pub struct PyCosineAnnealingLR {
    optimizer: Py<PyAny>,
    base_lr: f32,
    last_lr: f32,
    last_epoch: i32,
    t_max: i32,
    eta_min: f32,
}

#[pymethods]
impl PyCosineAnnealingLR {
    #[new]
    fn new(optimizer: Py<PyAny>, t_max: i32, eta_min: f32) -> PyResult<Self> {
        let base_lr = Python::attach(|py| get_optimizer_lr(py, &optimizer))?;
        Ok(Self {
            optimizer,
            base_lr,
            last_lr: base_lr,
            last_epoch: 0,
            t_max,
            eta_min,
        })
    }

    fn step(&mut self) -> PyResult<()> {
        self.last_epoch += 1;
        let new_lr = self.eta_min
            + (self.base_lr - self.eta_min)
                * (1.0 + (std::f32::consts::PI * self.last_epoch as f32 / self.t_max as f32).cos())
                / 2.0;
        Python::attach(|py| set_optimizer_lr(py, &self.optimizer, new_lr))?;
        self.last_lr = new_lr;
        Ok(())
    }

    fn get_last_lr(&self) -> Vec<f32> {
        vec![self.last_lr]
    }

    #[getter]
    fn last_epoch(&self) -> i32 {
        self.last_epoch
    }

    fn state_dict(&self) -> PyResult<HashMap<String, Py<PyAny>>> {
        Python::attach(|py| {
            let mut state: HashMap<String, Py<PyAny>> = HashMap::new();
            py_insert!(state, py, "last_epoch", self.last_epoch);
            py_insert!(state, py, "base_lr", self.base_lr);
            py_insert!(state, py, "last_lr", self.last_lr);
            py_insert!(state, py, "t_max", self.t_max);
            py_insert!(state, py, "eta_min", self.eta_min);
            Ok(state)
        })
    }

    fn load_state_dict(&mut self, state: HashMap<String, Py<PyAny>>) -> PyResult<()> {
        Python::attach(|py| {
            self.last_epoch = extract_i32(py, &state, "last_epoch")?;
            self.base_lr = extract_f32(py, &state, "base_lr")?;
            self.last_lr = extract_f32(py, &state, "last_lr")?;
            self.t_max = extract_i32(py, &state, "t_max")?;
            self.eta_min = extract_f32(py, &state, "eta_min")?;
            set_optimizer_lr(py, &self.optimizer, self.last_lr)
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "CosineAnnealingLR(t_max={}, eta_min={}, last_epoch={})",
            self.t_max, self.eta_min, self.last_epoch
        )
    }
}

// ===========================================================================
// LinearLR
// ===========================================================================

/// Linearly interpolates the learning rate multiplier from `start_factor`
/// to `end_factor` over `total_iters` epochs, then holds at `end_factor`:
/// `lr = base_lr * factor(last_epoch)`.
#[pyclass(name = "LinearLR")]
pub struct PyLinearLR {
    optimizer: Py<PyAny>,
    base_lr: f32,
    last_lr: f32,
    last_epoch: i32,
    start_factor: f32,
    end_factor: f32,
    total_iters: i32,
}

#[pymethods]
impl PyLinearLR {
    #[new]
    fn new(
        optimizer: Py<PyAny>,
        start_factor: f32,
        end_factor: f32,
        total_iters: i32,
    ) -> PyResult<Self> {
        let base_lr = Python::attach(|py| get_optimizer_lr(py, &optimizer))?;
        Ok(Self {
            optimizer,
            base_lr,
            last_lr: base_lr,
            last_epoch: 0,
            start_factor,
            end_factor,
            total_iters,
        })
    }

    fn step(&mut self) -> PyResult<()> {
        self.last_epoch += 1;
        let factor = if self.last_epoch >= self.total_iters {
            self.end_factor
        } else {
            self.start_factor
                + (self.end_factor - self.start_factor)
                    * (self.last_epoch as f32 / self.total_iters as f32)
        };
        let new_lr = self.base_lr * factor;
        Python::attach(|py| set_optimizer_lr(py, &self.optimizer, new_lr))?;
        self.last_lr = new_lr;
        Ok(())
    }

    fn get_last_lr(&self) -> Vec<f32> {
        vec![self.last_lr]
    }

    #[getter]
    fn last_epoch(&self) -> i32 {
        self.last_epoch
    }

    fn state_dict(&self) -> PyResult<HashMap<String, Py<PyAny>>> {
        Python::attach(|py| {
            let mut state: HashMap<String, Py<PyAny>> = HashMap::new();
            py_insert!(state, py, "last_epoch", self.last_epoch);
            py_insert!(state, py, "base_lr", self.base_lr);
            py_insert!(state, py, "last_lr", self.last_lr);
            py_insert!(state, py, "start_factor", self.start_factor);
            py_insert!(state, py, "end_factor", self.end_factor);
            py_insert!(state, py, "total_iters", self.total_iters);
            Ok(state)
        })
    }

    fn load_state_dict(&mut self, state: HashMap<String, Py<PyAny>>) -> PyResult<()> {
        Python::attach(|py| {
            self.last_epoch = extract_i32(py, &state, "last_epoch")?;
            self.base_lr = extract_f32(py, &state, "base_lr")?;
            self.last_lr = extract_f32(py, &state, "last_lr")?;
            self.start_factor = extract_f32(py, &state, "start_factor")?;
            self.end_factor = extract_f32(py, &state, "end_factor")?;
            self.total_iters = extract_i32(py, &state, "total_iters")?;
            set_optimizer_lr(py, &self.optimizer, self.last_lr)
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "LinearLR(start_factor={}, end_factor={}, total_iters={}, last_epoch={})",
            self.start_factor, self.end_factor, self.total_iters, self.last_epoch
        )
    }
}

// ===========================================================================
// ReduceLROnPlateau
// ===========================================================================

/// Reduces the learning rate by `factor` once a monitored metric has
/// stopped improving for `patience` consecutive `step()` calls.
///
/// Matches `torch.optim.lr_scheduler.ReduceLROnPlateau` semantics (and
/// `torsh_optim::lr_scheduler::ReduceLROnPlateau`): the FIRST `step()` call
/// always just establishes the baseline (`best`) without counting as a bad
/// epoch. After that, a call counts as "bad" unless the metric improves by
/// more than `threshold` (relative: `best * (1 - threshold)` for
/// `mode="min"`, `best * (1 + threshold)` for `mode="max"`). The learning
/// rate is reduced once `num_bad_epochs > patience` (i.e. after `patience +
/// 1` consecutive non-improving calls following the baseline), never
/// before. `threshold_mode="rel"`, `cooldown=0`, `min_lr=0.0`, and
/// `eps=1e-8` match PyTorch's defaults and are not currently exposed as
/// constructor parameters.
#[pyclass(name = "ReduceLROnPlateau")]
pub struct PyReduceLROnPlateau {
    optimizer: Py<PyAny>,
    mode: String,
    factor: f32,
    patience: i32,
    threshold: f32,
    last_lr: f32,
    last_epoch: i32,
    best: Option<f32>,
    num_bad_epochs: i32,
    cooldown_counter: i32,
}

#[pymethods]
impl PyReduceLROnPlateau {
    #[new]
    fn new(
        optimizer: Py<PyAny>,
        mode: &str,
        factor: f32,
        patience: i32,
        threshold: f32,
    ) -> PyResult<Self> {
        if !(0.0..1.0).contains(&factor) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "factor must be in [0.0, 1.0)",
            ));
        }
        if mode != "min" && mode != "max" {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "mode must be 'min' or 'max'",
            ));
        }
        let last_lr = Python::attach(|py| get_optimizer_lr(py, &optimizer))?;
        Ok(Self {
            optimizer,
            mode: mode.to_string(),
            factor,
            patience,
            threshold,
            last_lr,
            last_epoch: 0,
            best: None,
            num_bad_epochs: 0,
            cooldown_counter: 0,
        })
    }

    /// Feed a new metric value (matches PyTorch's `.step(metrics)`). `None`
    /// skips this call entirely: no baseline/bad-epoch bookkeeping happens,
    /// useful for epochs where the monitored metric wasn't computed.
    fn step(&mut self, metric: Option<f32>) -> PyResult<()> {
        let Some(current) = metric else {
            return Ok(());
        };
        self.last_epoch += 1;

        match self.best {
            None => {
                self.best = Some(current);
            }
            Some(best_value) => {
                let is_better = match self.mode.as_str() {
                    "min" => current < best_value * (1.0 - self.threshold),
                    "max" => current > best_value * (1.0 + self.threshold),
                    _ => false,
                };

                if is_better {
                    self.best = Some(current);
                    self.num_bad_epochs = 0;
                } else {
                    self.num_bad_epochs += 1;
                }

                if self.cooldown_counter > 0 {
                    self.cooldown_counter -= 1;
                    self.num_bad_epochs = 0;
                }

                if self.num_bad_epochs > self.patience {
                    self.reduce_lr()?;
                    self.cooldown_counter = 0;
                    self.num_bad_epochs = 0;
                }
            }
        }

        Ok(())
    }

    fn get_last_lr(&self) -> Vec<f32> {
        vec![self.last_lr]
    }

    #[getter]
    fn last_epoch(&self) -> i32 {
        self.last_epoch
    }

    #[getter]
    fn num_bad_epochs(&self) -> i32 {
        self.num_bad_epochs
    }

    #[getter]
    fn best(&self) -> Option<f32> {
        self.best
    }

    fn state_dict(&self) -> PyResult<HashMap<String, Py<PyAny>>> {
        Python::attach(|py| {
            let mut state: HashMap<String, Py<PyAny>> = HashMap::new();
            py_insert!(state, py, "mode", self.mode.clone());
            py_insert!(state, py, "factor", self.factor);
            py_insert!(state, py, "patience", self.patience);
            py_insert!(state, py, "threshold", self.threshold);
            py_insert!(state, py, "last_lr", self.last_lr);
            py_insert!(state, py, "last_epoch", self.last_epoch);
            py_insert!(state, py, "best", self.best);
            py_insert!(state, py, "num_bad_epochs", self.num_bad_epochs);
            py_insert!(state, py, "cooldown_counter", self.cooldown_counter);
            Ok(state)
        })
    }

    fn load_state_dict(&mut self, state: HashMap<String, Py<PyAny>>) -> PyResult<()> {
        Python::attach(|py| {
            self.mode = get_field(&state, "mode")?.extract(py)?;
            self.factor = extract_f32(py, &state, "factor")?;
            self.patience = extract_i32(py, &state, "patience")?;
            self.threshold = extract_f32(py, &state, "threshold")?;
            self.last_lr = extract_f32(py, &state, "last_lr")?;
            self.last_epoch = extract_i32(py, &state, "last_epoch")?;
            self.best = get_field(&state, "best")?.extract(py)?;
            self.num_bad_epochs = extract_i32(py, &state, "num_bad_epochs")?;
            self.cooldown_counter = extract_i32(py, &state, "cooldown_counter")?;
            set_optimizer_lr(py, &self.optimizer, self.last_lr)
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ReduceLROnPlateau(mode='{}', factor={}, patience={}, threshold={})",
            self.mode, self.factor, self.patience, self.threshold
        )
    }
}

impl PyReduceLROnPlateau {
    /// Reduce `last_lr` by `factor` and push it onto the optimizer, unless
    /// the change would be negligible (matches
    /// `torsh_optim::lr_scheduler::ReduceLROnPlateau::reduce_lr`'s `eps`
    /// guard against redundant no-op reductions).
    fn reduce_lr(&mut self) -> PyResult<()> {
        const EPS: f32 = 1e-8;
        let new_lr = self.last_lr * self.factor;
        if self.last_lr - new_lr > EPS {
            Python::attach(|py| set_optimizer_lr(py, &self.optimizer, new_lr))?;
            self.last_lr = new_lr;
        }
        Ok(())
    }
}

/// Register the `lr_scheduler` submodule with Python, mirroring
/// `torch.optim.lr_scheduler`.
pub fn register_lr_scheduler_module(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStepLR>()?;
    m.add_class::<PyMultiStepLR>()?;
    m.add_class::<PyExponentialLR>()?;
    m.add_class::<PyCosineAnnealingLR>()?;
    m.add_class::<PyLinearLR>()?;
    m.add_class::<PyReduceLROnPlateau>()?;
    Ok(())
}
