//! RMSprop optimizer

use super::base::{
    create_param_group, extract_parameters, get_field, optim_result_to_py, optim_state_to_pydict,
    param_index_error, pydict_to_optim_state, PyOptimizer,
};
use crate::{error::PyResult, tensor::PyTensor};
use parking_lot::RwLock;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool};
use std::collections::HashMap;
use std::sync::Arc;
use torsh_optim::{rmsprop::RMSprop, Optimizer};

/// RMSprop optimizer - Root Mean Square Propagation
#[pyclass(name = "RMSprop", extends = PyOptimizer)]
pub struct PyRMSprop {
    rmsprop: RMSprop,
    param_groups: Vec<HashMap<String, Py<PyAny>>>,
    lr: f32,
    alpha: f32,
    eps: f32,
    weight_decay: f32,
    momentum: f32,
    centered: bool,
}

#[pymethods]
impl PyRMSprop {
    #[new]
    #[pyo3(signature = (params, lr=None, alpha=None, eps=None, weight_decay=None, momentum=None, centered=None))]
    fn new(
        params: Vec<PyTensor>,
        lr: Option<f32>,
        alpha: Option<f32>,
        eps: Option<f32>,
        weight_decay: Option<f32>,
        momentum: Option<f32>,
        centered: Option<bool>,
    ) -> PyClassInitializer<Self> {
        let lr = lr.unwrap_or(0.01);
        let alpha = alpha.unwrap_or(0.99);
        let eps = eps.unwrap_or(1e-8);
        let weight_decay = weight_decay.unwrap_or(0.0);
        let momentum = momentum.unwrap_or(0.0);
        let centered = centered.unwrap_or(false);

        // Extract tensor parameters and wrap in Arc<RwLock>
        let tensor_params =
            extract_parameters(params.clone()).expect("parameter extraction should succeed");
        let wrapped_params: Vec<Arc<RwLock<_>>> = tensor_params
            .into_iter()
            .map(|tensor| Arc::new(RwLock::new(tensor)))
            .collect();
        let rmsprop = RMSprop::new(
            wrapped_params,
            Some(lr),
            Some(alpha),
            Some(eps),
            Some(weight_decay),
            Some(momentum),
            centered,
        );

        // Create parameter groups
        let mut param_group_data = HashMap::new();
        Python::attach(|py| {
            param_group_data.insert(
                "alpha".to_string(),
                alpha
                    .into_pyobject(py)
                    .expect("Python object conversion should succeed")
                    .into_any()
                    .unbind(),
            );
            param_group_data.insert(
                "eps".to_string(),
                eps.into_pyobject(py)
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
                "momentum".to_string(),
                momentum
                    .into_pyobject(py)
                    .expect("Python object conversion should succeed")
                    .into_any()
                    .unbind(),
            );
            param_group_data.insert(
                "centered".to_string(),
                PyBool::new(py, centered).to_owned().into(),
            );
        });

        let param_groups = vec![create_param_group(params, lr, param_group_data)
            .expect("param group creation should succeed")];

        (
            Self {
                rmsprop,
                param_groups,
                lr,
                alpha,
                eps,
                weight_decay,
                momentum,
                centered,
            },
            PyOptimizer {},
        )
            .into()
    }

    /// Perform a single optimization step
    fn step(&mut self) -> PyResult<()> {
        self.rmsprop.step().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Optimizer step failed: {}",
                e
            ))
        })?;
        Ok(())
    }

    /// Zero out gradients of all parameters
    fn zero_grad(&mut self, set_to_none: Option<bool>) {
        let _set_to_none = set_to_none.unwrap_or(false);
        self.rmsprop.zero_grad();
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

    /// Get current state: the real per-parameter RMSprop buffers
    /// (`square_avg`, and `momentum_buffer`/`grad_avg` when enabled), keyed
    /// by ordinal parameter index.
    fn state(&self) -> PyResult<HashMap<String, Py<PyAny>>> {
        Python::attach(|py| self.build_state_pydict(py))
    }

    /// Get state dictionary: `{"state": {...}, "param_groups": [...]}`,
    /// matching PyTorch's `Optimizer.state_dict()` shape. `"state"` holds the
    /// real RMSprop buffers so a full checkpoint/resume round-trip is
    /// possible.
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

            let group = self.build_param_group_pydict(py)?;
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

    /// Load a state dictionary produced by [`Self::state_dict`].
    ///
    /// `torsh_optim::rmsprop::RMSprop` has no in-place setter for
    /// `alpha`/`eps`/`weight_decay`/`momentum`/`centered`, so restoring them
    /// faithfully (as PyTorch does: `load_state_dict` fully replaces
    /// param-group hyperparameters) requires reconstructing the wrapped
    /// optimizer. The SAME parameter handles are reused (only the
    /// hyperparameters and per-parameter buffers change), so gradient/data
    /// identity is preserved.
    fn load_state_dict(&mut self, state_dict: HashMap<String, Py<PyAny>>) -> PyResult<()> {
        Python::attach(|py| {
            let groups: Vec<HashMap<String, Py<PyAny>>> =
                get_field(&state_dict, "param_groups")?.extract(py)?;
            let group0 = groups.first().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "state_dict 'param_groups' must contain at least one group",
                )
            })?;

            let parameters = self.rmsprop.parameters();
            if let Some(params_obj) = group0.get("params") {
                let params: Vec<i64> = params_obj.extract(py)?;
                if params.len() != parameters.len() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "loaded state dict has {} parameters, but this optimizer has {}",
                        params.len(),
                        parameters.len()
                    )));
                }
            }

            let lr: f32 = get_field(group0, "lr")?.extract(py)?;
            let alpha: f32 = get_field(group0, "alpha")?.extract(py)?;
            let eps: f32 = get_field(group0, "eps")?.extract(py)?;
            let weight_decay: f32 = get_field(group0, "weight_decay")?.extract(py)?;
            let momentum: f32 = get_field(group0, "momentum")?.extract(py)?;
            let centered: bool = get_field(group0, "centered")?.extract(py)?;

            let mut new_rmsprop = RMSprop::new(
                parameters.clone(),
                Some(lr),
                Some(alpha),
                Some(eps),
                Some(weight_decay),
                Some(momentum),
                centered,
            );

            if let Some(state_obj) = state_dict.get("state") {
                let per_param: HashMap<String, Py<PyAny>> = state_obj.extract(py)?;
                let restored = pydict_to_optim_state(py, &parameters, &per_param)?;
                let mut rust_state =
                    optim_result_to_py("RMSprop state_dict", new_rmsprop.state_dict())?;
                rust_state.state = restored;
                optim_result_to_py(
                    "RMSprop load_state_dict",
                    new_rmsprop.load_state_dict(rust_state),
                )?;
            }

            self.rmsprop = new_rmsprop;
            self.lr = lr;
            self.alpha = alpha;
            self.eps = eps;
            self.weight_decay = weight_decay;
            self.momentum = momentum;
            self.centered = centered;

            let group = self.build_param_group_pydict(py)?;
            self.param_groups = vec![group];

            Ok(())
        })
    }

    /// Manually set the gradient for the parameter at `index`.
    ///
    /// Useful for driving `step()` without a full autograd pass (e.g. tests,
    /// or custom training loops that compute gradients out-of-band).
    fn set_param_grad(&mut self, index: usize, grad: PyTensor) -> PyResult<()> {
        let parameters = self.rmsprop.parameters();
        let param = parameters
            .get(index)
            .ok_or_else(|| param_index_error(index, parameters.len()))?;
        param.read().set_grad(Some(grad.tensor));
        Ok(())
    }

    /// Read the current value of the parameter at `index`.
    fn get_param_data(&self, index: usize) -> PyResult<PyTensor> {
        let parameters = self.rmsprop.parameters();
        let param = parameters
            .get(index)
            .ok_or_else(|| param_index_error(index, parameters.len()))?;
        let tensor = param.read().clone();
        Ok(PyTensor { tensor })
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "RMSprop(lr={}, alpha={}, eps={}, weight_decay={}, momentum={}, centered={})",
            self.lr, self.alpha, self.eps, self.weight_decay, self.momentum, self.centered
        )
    }

    /// Get defaults
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
                "alpha".to_string(),
                self.alpha
                    .into_pyobject(py)
                    .expect("Python object conversion should succeed")
                    .into_any()
                    .unbind(),
            );
            defaults.insert(
                "eps".to_string(),
                self.eps
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
                "momentum".to_string(),
                self.momentum
                    .into_pyobject(py)
                    .expect("Python object conversion should succeed")
                    .into_any()
                    .unbind(),
            );
            defaults.insert(
                "centered".to_string(),
                PyBool::new(py, self.centered).to_owned().into(),
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
        // Propagate to the wrapped real optimizer too -- without this, `step()`
        // keeps using whatever `lr` it was constructed with, since it reads its
        // OWN internal param-group `lr`, not this pyclass's `self.lr` field.
        self.rmsprop.set_lr(lr);
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

    /// Get alpha (smoothing constant)
    #[getter]
    fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Get epsilon
    #[getter]
    fn eps(&self) -> f32 {
        self.eps
    }

    /// Get weight decay
    #[getter]
    fn weight_decay(&self) -> f32 {
        self.weight_decay
    }

    /// Get momentum
    #[getter]
    fn momentum(&self) -> f32 {
        self.momentum
    }

    /// Get centered flag
    #[getter]
    fn centered(&self) -> bool {
        self.centered
    }
}

impl PyRMSprop {
    /// Build the real per-parameter state dict (`square_avg`, and
    /// `momentum_buffer`/`grad_avg` when enabled) keyed by ordinal parameter
    /// index.
    fn build_state_pydict(&self, py: Python<'_>) -> PyResult<HashMap<String, Py<PyAny>>> {
        let parameters = self.rmsprop.parameters();
        let rust_state = optim_result_to_py("RMSprop state_dict", self.rmsprop.state_dict())?;
        optim_state_to_pydict(py, &parameters, &rust_state)
    }

    /// Build the `param_groups[0]` dict: lr + every RMSprop hyperparameter,
    /// plus `"params"` (ordinal indices, for shape-fidelity/validation).
    fn build_param_group_pydict(&self, py: Python<'_>) -> PyResult<HashMap<String, Py<PyAny>>> {
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
            "alpha".to_string(),
            self.alpha
                .into_pyobject(py)
                .expect("Python object conversion should succeed")
                .into_any()
                .unbind(),
        );
        group.insert(
            "eps".to_string(),
            self.eps
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
            "momentum".to_string(),
            self.momentum
                .into_pyobject(py)
                .expect("Python object conversion should succeed")
                .into_any()
                .unbind(),
        );
        group.insert(
            "centered".to_string(),
            PyBool::new(py, self.centered).to_owned().into(),
        );
        let param_indices: Vec<i64> = (0..self.rmsprop.parameters().len() as i64).collect();
        group.insert(
            "params".to_string(),
            param_indices
                .into_pyobject(py)
                .expect("Python object conversion should succeed")
                .into_any()
                .unbind(),
        );
        Ok(group)
    }
}
