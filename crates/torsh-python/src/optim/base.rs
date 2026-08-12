//! Base optimizer implementation - Foundation for all PyTorch-compatible optimizers

use crate::{error::PyResult, py_result, tensor::PyTensor};
use parking_lot::RwLock;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::collections::HashMap;
use std::sync::Arc;
use torsh_core::device::DeviceType;
use torsh_tensor::Tensor;

/// Base optimizer class - foundation for all optimizers
#[pyclass(name = "Optimizer", subclass)]
pub struct PyOptimizer {
    // This will be overridden by subclasses
}

#[pymethods]
impl PyOptimizer {
    #[new]
    #[pyo3(signature = ())]
    fn new() -> Self {
        Self {}
    }

    /// Perform a single optimization step - must be implemented by subclasses
    fn step(&mut self) -> PyResult<()> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Subclasses must implement step method",
        ))
    }

    /// Zero out gradients of all parameters
    fn zero_grad(&mut self, set_to_none: Option<bool>) {
        // Default implementation - subclasses should override
        let _set_to_none = set_to_none.unwrap_or(false);
        // Subclasses should implement actual gradient zeroing
    }

    /// Get the state dictionary (optimizer state and hyperparameters)
    fn state_dict(&self) -> PyResult<HashMap<String, Py<PyAny>>> {
        // Default implementation - subclasses should override
        Ok(HashMap::new())
    }

    /// Load state dictionary
    fn load_state_dict(&mut self, state_dict: HashMap<String, Py<PyAny>>) -> PyResult<()> {
        // Default implementation - subclasses should override
        let _state_dict = state_dict;
        Ok(())
    }

    /// Get parameter groups
    fn param_groups(&self) -> PyResult<Vec<HashMap<String, Py<PyAny>>>> {
        // Default implementation - subclasses should override
        Ok(Vec::new())
    }

    /// Get current state
    fn state(&self) -> PyResult<HashMap<String, Py<PyAny>>> {
        // Default implementation - subclasses should override
        Ok(HashMap::new())
    }

    /// Add a new parameter group
    fn add_param_group(&mut self, param_group: HashMap<String, Py<PyAny>>) -> PyResult<()> {
        // Default implementation - subclasses should override
        let _param_group = param_group;
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Subclasses must implement add_param_group method",
        ))
    }

    /// String representation
    fn __repr__(&self) -> String {
        "Optimizer()".to_string()
    }

    /// Get defaults (default hyperparameters)
    fn defaults(&self) -> PyResult<HashMap<String, Py<PyAny>>> {
        // Default implementation - subclasses should override
        Ok(HashMap::new())
    }
}

/// Helper function to extract parameters from Python objects
pub fn extract_parameters(params: Vec<PyTensor>) -> PyResult<Vec<torsh_tensor::Tensor<f32>>> {
    params.into_iter().map(|p| Ok(p.tensor)).collect()
}

/// Helper function to create parameter group
pub fn create_param_group(
    params: Vec<PyTensor>,
    lr: f32,
    extra_params: HashMap<String, Py<PyAny>>,
) -> PyResult<HashMap<String, Py<PyAny>>> {
    let mut param_group = HashMap::new();

    Python::attach(|py| {
        // Add parameters
        let py_params: Vec<Py<PyAny>> = params
            .into_iter()
            .map(|p| {
                p.into_pyobject(py)
                    .expect("Python object conversion should succeed")
                    .into()
            })
            .collect();
        param_group.insert(
            "params".to_string(),
            py_params
                .into_pyobject(py)
                .expect("Python object conversion should succeed")
                .into(),
        );

        // Add learning rate
        param_group.insert(
            "lr".to_string(),
            lr.into_pyobject(py)
                .expect("Python object conversion should succeed")
                .into(),
        );

        // Add extra parameters
        for (key, value) in extra_params {
            param_group.insert(key, value);
        }

        Ok(param_group)
    })
}

// ===========================================================================
// state_dict() / load_state_dict() serialization helpers
// ===========================================================================
//
// Every concrete optimizer (SGD, Adam, AdamW, Adagrad, RMSprop) serializes
// its real per-parameter buffers (momentum, exp_avg, exp_avg_sq, step count,
// ...) through the helpers below, so the wire format is identical across all
// of them: `{"data": [f32, ...], "shape": [usize, ...]}` per tensor, nested
// under an ordinal parameter index ("0", "1", ...) rather than the raw
// pointer strings `torsh_optim` uses internally. Ordinal position is stable
// across serialization / fresh-instance reconstruction; raw pointers are not.

/// Encode a tensor's raw contents into a Python-reconstructable snapshot:
/// `{"data": [f32, ...], "shape": [usize, ...]}`.
pub fn tensor_to_state_entry(py: Python<'_>, tensor: &Tensor<f32>) -> PyResult<Py<PyAny>> {
    let data = py_result!(tensor.to_vec())?;
    let shape: Vec<usize> = tensor.shape().dims().to_vec();

    let mut entry: HashMap<String, Py<PyAny>> = HashMap::new();
    entry.insert(
        "data".to_string(),
        data.into_pyobject(py)
            .expect("Python object conversion should succeed")
            .into_any()
            .unbind(),
    );
    entry.insert(
        "shape".to_string(),
        shape
            .into_pyobject(py)
            .expect("Python object conversion should succeed")
            .into_any()
            .unbind(),
    );

    Ok(entry
        .into_pyobject(py)
        .expect("Python object conversion should succeed")
        .into_any()
        .unbind())
}

/// Decode a tensor snapshot previously produced by [`tensor_to_state_entry`].
pub fn state_entry_to_tensor(py: Python<'_>, entry: &Py<PyAny>) -> PyResult<Tensor<f32>> {
    let map: HashMap<String, Py<PyAny>> = entry.extract(py)?;
    let data: Vec<f32> = get_field(&map, "data")?.extract(py)?;
    let shape: Vec<usize> = get_field(&map, "shape")?.extract(py)?;
    py_result!(Tensor::from_data(data, shape, DeviceType::Cpu))
}

/// Look up `key` in a Python-dict-shaped state map, converting a missing key
/// into a catchable `ValueError` instead of panicking.
pub fn get_field<'a>(map: &'a HashMap<String, Py<PyAny>>, key: &str) -> PyResult<&'a Py<PyAny>> {
    map.get(key).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "state dict is missing required key '{key}'"
        ))
    })
}

/// Convert a `torsh_optim::OptimizerResult` into a `PyResult`, matching the
/// error-message style already used for `step()` failures.
pub fn optim_result_to_py<T>(
    context: &str,
    result: torsh_optim::OptimizerResult<T>,
) -> PyResult<T> {
    result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{context}: {e}")))
}

/// Build a bounds-check error for `set_param_grad`, shared across every
/// optimizer so the message stays consistent.
pub fn param_index_error(index: usize, len: usize) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
        "parameter index {index} out of range (have {len})"
    ))
}

/// Build the PyTorch-shaped, ordinal-indexed `"state"` sub-dict from a
/// `torsh_optim::OptimizerState`.
///
/// `torsh_optim` optimizers key their internal per-parameter state by a raw
/// pointer string (`format!("{:p}", ...)`), which is only stable for the
/// lifetime of a single optimizer instance. Ordinal position (0, 1, 2, ...)
/// within `parameters` is stable across serialization / reconstruction, so
/// state is re-keyed by ordinal index here -- this is what makes
/// `load_state_dict` on a freshly constructed optimizer (with brand-new
/// parameter pointers) able to restore the exact per-parameter buffers.
pub fn optim_state_to_pydict(
    py: Python<'_>,
    parameters: &[Arc<RwLock<Tensor<f32>>>],
    rust_state: &torsh_optim::OptimizerState,
) -> PyResult<HashMap<String, Py<PyAny>>> {
    let mut per_param: HashMap<String, Py<PyAny>> = HashMap::new();

    for (idx, param) in parameters.iter().enumerate() {
        let param_id = format!("{:p}", Arc::as_ptr(param));
        let Some(named) = rust_state.state.get(&param_id) else {
            continue;
        };

        let mut entry: HashMap<String, Py<PyAny>> = HashMap::new();
        for (name, tensor) in named {
            entry.insert(name.clone(), tensor_to_state_entry(py, tensor)?);
        }

        per_param.insert(
            idx.to_string(),
            entry
                .into_pyobject(py)
                .expect("Python object conversion should succeed")
                .into_any()
                .unbind(),
        );
    }

    Ok(per_param)
}

/// Reverse of [`optim_state_to_pydict`]: rebuild a pointer-keyed `state` map
/// (suitable for `torsh_optim::Optimizer::load_state_dict`) from the
/// ordinal-indexed per-parameter dict, using the CURRENT parameter handles.
/// Because it re-derives pointer keys from `parameters` at call time, this
/// is correct even when `parameters` belongs to a freshly reconstructed
/// optimizer whose pointers differ from whatever was live when the snapshot
/// was captured.
pub fn pydict_to_optim_state(
    py: Python<'_>,
    parameters: &[Arc<RwLock<Tensor<f32>>>],
    per_param: &HashMap<String, Py<PyAny>>,
) -> PyResult<HashMap<String, HashMap<String, Tensor<f32>>>> {
    let mut state: HashMap<String, HashMap<String, Tensor<f32>>> = HashMap::new();

    for (idx, param) in parameters.iter().enumerate() {
        let Some(entry_obj) = per_param.get(&idx.to_string()) else {
            continue;
        };
        let entry_map: HashMap<String, Py<PyAny>> = entry_obj.extract(py)?;

        let mut named: HashMap<String, Tensor<f32>> = HashMap::new();
        for (name, tensor_entry) in &entry_map {
            named.insert(name.clone(), state_entry_to_tensor(py, tensor_entry)?);
        }

        let param_id = format!("{:p}", Arc::as_ptr(param));
        state.insert(param_id, named);
    }

    Ok(state)
}
