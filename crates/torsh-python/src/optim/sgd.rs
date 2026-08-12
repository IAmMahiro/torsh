//! SGD (Stochastic Gradient Descent) optimizer

use super::base::{
    create_param_group, get_field, param_index_error, state_entry_to_tensor, tensor_to_state_entry,
    PyOptimizer,
};
use crate::{error::PyResult, py_result, tensor::PyTensor};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool};
use std::collections::HashMap;
use torsh_tensor::Tensor;

/// SGD optimizer - Stochastic Gradient Descent
#[pyclass(name = "SGD", extends = PyOptimizer)]
pub struct PySGD {
    parameters: Vec<Tensor<f32>>,
    momentum_buffers: Vec<Option<Tensor<f32>>>,
    param_groups: Vec<HashMap<String, Py<PyAny>>>,
    lr: f32,
    momentum: f32,
    dampening: f32,
    weight_decay: f32,
    nesterov: bool,
}

#[pymethods]
impl PySGD {
    #[new]
    #[pyo3(signature = (params, lr, momentum=None, dampening=None, weight_decay=None, nesterov=None))]
    fn new(
        params: Vec<PyTensor>,
        lr: f32,
        momentum: Option<f32>,
        dampening: Option<f32>,
        weight_decay: Option<f32>,
        nesterov: Option<bool>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let momentum = momentum.unwrap_or(0.0);
        let dampening = dampening.unwrap_or(0.0);
        let weight_decay = weight_decay.unwrap_or(0.0);
        let nesterov = nesterov.unwrap_or(false);

        // Extract tensor parameters
        let parameters: Vec<Tensor<f32>> = params.iter().map(|p| p.tensor.clone()).collect();
        let momentum_buffers = vec![None; parameters.len()];

        // Create parameter groups
        let mut param_group_data = HashMap::new();
        Python::attach(|py| {
            param_group_data.insert(
                "momentum".to_string(),
                momentum
                    .into_pyobject(py)
                    .expect("Python object conversion should succeed")
                    .into_any()
                    .unbind(),
            );
            param_group_data.insert(
                "dampening".to_string(),
                dampening
                    .into_pyobject(py)
                    .expect("Python object conversion should succeed")
                    .into_any()
                    .unbind(),
            );
            param_group_data.insert(
                "weight_decay".to_string(),
                weight_decay
                    .into_pyobject(py)
                    .expect("Python object conversion should succeed")
                    .into_any()
                    .unbind(),
            );
            param_group_data.insert(
                "nesterov".to_string(),
                PyBool::new(py, nesterov).to_owned().into(),
            );
        });

        let param_groups = vec![create_param_group(params, lr, param_group_data)?];

        Ok((
            Self {
                parameters,
                momentum_buffers,
                param_groups,
                lr,
                momentum,
                dampening,
                weight_decay,
                nesterov,
            },
            PyOptimizer {},
        )
            .into())
    }

    /// Perform a single optimization step
    fn step(&mut self) -> PyResult<()> {
        for (i, param) in self.parameters.iter_mut().enumerate() {
            if let Some(grad) = param.grad() {
                let mut d_p = grad.clone();

                // Apply weight decay if specified
                if self.weight_decay != 0.0 {
                    let weight_decay_term = py_result!(param.mul_scalar(self.weight_decay))?;
                    d_p = py_result!(d_p.add(&weight_decay_term))?;
                }

                // Apply momentum if specified
                if self.momentum != 0.0 {
                    if let Some(ref mut buf) = self.momentum_buffers[i] {
                        // buf = momentum * buf + d_p
                        let momentum_buf = py_result!(buf.mul_scalar(self.momentum))?;
                        *buf = py_result!(momentum_buf.add(&d_p))?;

                        if self.nesterov {
                            let momentum_term = py_result!(buf.mul_scalar(self.momentum))?;
                            d_p = py_result!(d_p.add(&momentum_term))?;
                        } else {
                            d_p = buf.clone();
                        }
                    } else {
                        // Initialize momentum buffer
                        self.momentum_buffers[i] = Some(d_p.clone());
                        if self.nesterov {
                            let momentum_term = py_result!(d_p.mul_scalar(self.momentum))?;
                            d_p = py_result!(d_p.add(&momentum_term))?;
                        }
                    }
                }

                // Update parameter: param = param - lr * d_p
                let update = py_result!(d_p.mul_scalar(self.lr))?;
                *param = py_result!(param.sub(&update))?;
            }
        }
        Ok(())
    }

    /// Zero out gradients of all parameters
    fn zero_grad(&mut self, set_to_none: Option<bool>) {
        let _set_to_none = set_to_none.unwrap_or(false);
        for param in &mut self.parameters {
            let _ = param.zero_grad();
        }
    }

    /// Get parameter groups
    fn param_groups(&self) -> PyResult<Vec<HashMap<String, Py<PyAny>>>> {
        // Manual clone since Py<PyAny> doesn't implement Clone
        Python::attach(|py| {
            let cloned_groups = self
                .param_groups
                .iter()
                .map(|group| {
                    group
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone_ref(py)))
                        .collect()
                })
                .collect();
            Ok(cloned_groups)
        })
    }

    /// Get current state: the real per-parameter momentum buffers, keyed by
    /// ordinal parameter index. Only parameters that have accumulated a
    /// momentum buffer so far are present, mirroring PyTorch's lazy `state`.
    fn state(&self) -> PyResult<HashMap<String, Py<PyAny>>> {
        Python::attach(|py| self.build_state_pydict(py))
    }

    /// Get state dictionary: `{"state": {...}, "param_groups": [...]}`,
    /// matching PyTorch's `Optimizer.state_dict()` shape. `"state"` holds the
    /// real momentum buffers (as `{"data": [...], "shape": [...]}` per
    /// tensor) so a full checkpoint/resume round-trip is possible.
    fn state_dict(&self) -> PyResult<HashMap<String, Py<PyAny>>> {
        Python::attach(|py| {
            let mut state_dict: HashMap<String, Py<PyAny>> = HashMap::new();

            let state = self.build_state_pydict(py)?;
            state_dict.insert(
                "state".to_string(),
                state
                    .into_pyobject(py)
                    .expect("Python object conversion should succeed")
                    .into_any()
                    .unbind(),
            );

            let mut group: HashMap<String, Py<PyAny>> = HashMap::new();
            group.insert(
                "lr".to_string(),
                self.lr
                    .into_pyobject(py)
                    .expect("Python object conversion should succeed")
                    .into_any()
                    .unbind(),
            );
            group.insert(
                "momentum".to_string(),
                self.momentum
                    .into_pyobject(py)
                    .expect("Python object conversion should succeed")
                    .into_any()
                    .unbind(),
            );
            group.insert(
                "dampening".to_string(),
                self.dampening
                    .into_pyobject(py)
                    .expect("Python object conversion should succeed")
                    .into_any()
                    .unbind(),
            );
            group.insert(
                "weight_decay".to_string(),
                self.weight_decay
                    .into_pyobject(py)
                    .expect("Python object conversion should succeed")
                    .into_any()
                    .unbind(),
            );
            group.insert(
                "nesterov".to_string(),
                PyBool::new(py, self.nesterov).to_owned().into(),
            );
            let param_indices: Vec<i64> = (0..self.parameters.len() as i64).collect();
            group.insert(
                "params".to_string(),
                param_indices
                    .into_pyobject(py)
                    .expect("Python object conversion should succeed")
                    .into_any()
                    .unbind(),
            );

            state_dict.insert(
                "param_groups".to_string(),
                vec![group]
                    .into_pyobject(py)
                    .expect("Python object conversion should succeed")
                    .into_any()
                    .unbind(),
            );

            Ok(state_dict)
        })
    }

    /// Load a state dictionary produced by [`Self::state_dict`], restoring
    /// momentum buffers, hyperparameters, and learning rate exactly.
    fn load_state_dict(&mut self, state_dict: HashMap<String, Py<PyAny>>) -> PyResult<()> {
        Python::attach(|py| {
            let groups: Vec<HashMap<String, Py<PyAny>>> =
                get_field(&state_dict, "param_groups")?.extract(py)?;
            let group0 = groups.first().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "state_dict 'param_groups' must contain at least one group",
                )
            })?;

            if let Some(params_obj) = group0.get("params") {
                let params: Vec<i64> = params_obj.extract(py)?;
                if params.len() != self.parameters.len() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "loaded state dict has {} parameters, but this optimizer has {}",
                        params.len(),
                        self.parameters.len()
                    )));
                }
            }

            let lr: f32 = get_field(group0, "lr")?.extract(py)?;
            let momentum: f32 = get_field(group0, "momentum")?.extract(py)?;
            let dampening: f32 = get_field(group0, "dampening")?.extract(py)?;
            let weight_decay: f32 = get_field(group0, "weight_decay")?.extract(py)?;
            let nesterov: bool = get_field(group0, "nesterov")?.extract(py)?;

            self.lr = lr;
            self.momentum = momentum;
            self.dampening = dampening;
            self.weight_decay = weight_decay;
            self.nesterov = nesterov;
            for param_group in &mut self.param_groups {
                param_group.insert(
                    "lr".to_string(),
                    lr.into_pyobject(py)
                        .expect("Python object conversion should succeed")
                        .into_any()
                        .unbind(),
                );
                param_group.insert(
                    "momentum".to_string(),
                    momentum
                        .into_pyobject(py)
                        .expect("Python object conversion should succeed")
                        .into_any()
                        .unbind(),
                );
                param_group.insert(
                    "dampening".to_string(),
                    dampening
                        .into_pyobject(py)
                        .expect("Python object conversion should succeed")
                        .into_any()
                        .unbind(),
                );
                param_group.insert(
                    "weight_decay".to_string(),
                    weight_decay
                        .into_pyobject(py)
                        .expect("Python object conversion should succeed")
                        .into_any()
                        .unbind(),
                );
                param_group.insert(
                    "nesterov".to_string(),
                    PyBool::new(py, nesterov).to_owned().into(),
                );
            }

            for buf in &mut self.momentum_buffers {
                *buf = None;
            }
            if let Some(state_obj) = state_dict.get("state") {
                let per_param: HashMap<String, Py<PyAny>> = state_obj.extract(py)?;
                for (idx, buf) in self.momentum_buffers.iter_mut().enumerate() {
                    if let Some(entry_obj) = per_param.get(&idx.to_string()) {
                        let entry_map: HashMap<String, Py<PyAny>> = entry_obj.extract(py)?;
                        if let Some(mb) = entry_map.get("momentum_buffer") {
                            *buf = Some(state_entry_to_tensor(py, mb)?);
                        }
                    }
                }
            }

            Ok(())
        })
    }

    /// Manually set the gradient for the parameter at `index`.
    ///
    /// Useful for driving `step()` without a full autograd pass (e.g. tests,
    /// or custom training loops that compute gradients out-of-band).
    fn set_param_grad(&mut self, index: usize, grad: PyTensor) -> PyResult<()> {
        let len = self.parameters.len();
        let param = self
            .parameters
            .get_mut(index)
            .ok_or_else(|| param_index_error(index, len))?;
        param.set_grad(Some(grad.tensor));
        Ok(())
    }

    /// Read the current value of the parameter at `index`.
    fn get_param_data(&self, index: usize) -> PyResult<PyTensor> {
        let param = self
            .parameters
            .get(index)
            .ok_or_else(|| param_index_error(index, self.parameters.len()))?;
        Ok(PyTensor {
            tensor: param.clone(),
        })
    }

    /// Add a new parameter group
    fn add_param_group(&mut self, mut param_group: HashMap<String, Py<PyAny>>) -> PyResult<()> {
        // Set default values if not provided
        Python::attach(|py| {
            if !param_group.contains_key("lr") {
                param_group.insert(
                    "lr".to_string(),
                    self.lr
                        .into_pyobject(py)
                        .expect("Python object conversion should succeed")
                        .into_any()
                        .unbind(),
                );
            }
            if !param_group.contains_key("momentum") {
                param_group.insert(
                    "momentum".to_string(),
                    self.momentum
                        .into_pyobject(py)
                        .expect("Python object conversion should succeed")
                        .into_any()
                        .unbind(),
                );
            }
            if !param_group.contains_key("dampening") {
                param_group.insert(
                    "dampening".to_string(),
                    self.dampening
                        .into_pyobject(py)
                        .expect("Python object conversion should succeed")
                        .into_any()
                        .unbind(),
                );
            }
            if !param_group.contains_key("weight_decay") {
                param_group.insert(
                    "weight_decay".to_string(),
                    self.weight_decay
                        .into_pyobject(py)
                        .expect("Python object conversion should succeed")
                        .into_any()
                        .unbind(),
                );
            }
            if !param_group.contains_key("nesterov") {
                param_group.insert(
                    "nesterov".to_string(),
                    PyBool::new(py, self.nesterov).to_owned().into(),
                );
            }
        });

        self.param_groups.push(param_group);
        Ok(())
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "SGD(lr={}, momentum={}, dampening={}, weight_decay={}, nesterov={})",
            self.lr, self.momentum, self.dampening, self.weight_decay, self.nesterov
        )
    }

    /// Get defaults (default hyperparameters)
    fn defaults(&self) -> PyResult<HashMap<String, Py<PyAny>>> {
        let mut defaults = HashMap::new();
        Python::attach(|py| {
            defaults.insert(
                "lr".to_string(),
                self.lr
                    .into_pyobject(py)
                    .expect("Python object conversion should succeed")
                    .into_any()
                    .unbind(),
            );
            defaults.insert(
                "momentum".to_string(),
                self.momentum
                    .into_pyobject(py)
                    .expect("Python object conversion should succeed")
                    .into_any()
                    .unbind(),
            );
            defaults.insert(
                "dampening".to_string(),
                self.dampening
                    .into_pyobject(py)
                    .expect("Python object conversion should succeed")
                    .into_any()
                    .unbind(),
            );
            defaults.insert(
                "weight_decay".to_string(),
                self.weight_decay
                    .into_pyobject(py)
                    .expect("Python object conversion should succeed")
                    .into_any()
                    .unbind(),
            );
            defaults.insert(
                "nesterov".to_string(),
                PyBool::new(py, self.nesterov).to_owned().into(),
            );
        });
        Ok(defaults)
    }

    /// Get learning rate
    #[getter]
    fn lr(&self) -> f32 {
        self.lr
    }

    /// Set learning rate
    #[setter]
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
        // Update all parameter groups
        Python::attach(|py| {
            for param_group in &mut self.param_groups {
                param_group.insert(
                    "lr".to_string(),
                    lr.into_pyobject(py)
                        .expect("Python object conversion should succeed")
                        .into_any()
                        .unbind(),
                );
            }
        });
    }

    /// Get momentum
    #[getter]
    fn momentum(&self) -> f32 {
        self.momentum
    }

    /// Get dampening
    #[getter]
    fn dampening(&self) -> f32 {
        self.dampening
    }

    /// Get weight decay
    #[getter]
    fn weight_decay(&self) -> f32 {
        self.weight_decay
    }

    /// Get nesterov flag
    #[getter]
    fn nesterov(&self) -> bool {
        self.nesterov
    }
}

impl PySGD {
    /// Build the real per-parameter state dict:
    /// `{"<idx>": {"momentum_buffer": {"data": [...], "shape": [...]}}, ...}`,
    /// including only parameters that have actually accumulated a momentum
    /// buffer so far (mirrors PyTorch, which populates `state` lazily).
    fn build_state_pydict(&self, py: Python<'_>) -> PyResult<HashMap<String, Py<PyAny>>> {
        let mut state: HashMap<String, Py<PyAny>> = HashMap::new();
        for (idx, buf) in self.momentum_buffers.iter().enumerate() {
            if let Some(tensor) = buf {
                let mut entry: HashMap<String, Py<PyAny>> = HashMap::new();
                entry.insert(
                    "momentum_buffer".to_string(),
                    tensor_to_state_entry(py, tensor)?,
                );
                state.insert(
                    idx.to_string(),
                    entry
                        .into_pyobject(py)
                        .expect("Python object conversion should succeed")
                        .into_any()
                        .unbind(),
                );
            }
        }
        Ok(state)
    }
}
