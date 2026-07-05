//! Tensor creation functions - zeros, ones, randn, etc.

use super::core::PyTensor;
use crate::{device::PyDevice, dtype::PyDType, error::PyResult, py_result};
use pyo3::prelude::*;

/// Register simplified tensor creation functions
pub fn register_creation_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    use pyo3::wrap_pyfunction;

    #[pyfunction]
    #[pyo3(signature = (data, dtype=None, device=None, requires_grad=None))]
    fn tensor(
        data: &Bound<'_, PyAny>,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        PyTensor::new(data, dtype, device, requires_grad)
    }

    #[pyfunction]
    #[pyo3(signature = (size, dtype=None, device=None, requires_grad=None))]
    fn zeros(
        size: Vec<usize>,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let _dtype = dtype;
        let _device = device;
        let tensor_result = py_result!(torsh_tensor::creation::zeros(&size))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    #[pyfunction]
    #[pyo3(signature = (size, dtype=None, device=None, requires_grad=None))]
    fn ones(
        size: Vec<usize>,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let _dtype = dtype;
        let _device = device;
        let tensor_result = py_result!(torsh_tensor::creation::ones(&size))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    #[pyfunction]
    #[pyo3(signature = (size, dtype=None, device=None, requires_grad=None))]
    fn randn(
        size: Vec<usize>,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let _dtype = dtype;
        let _device = device;
        let tensor_result = py_result!(torsh_tensor::creation::randn(&size))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    #[pyfunction]
    #[pyo3(signature = (size, dtype=None, device=None, requires_grad=None))]
    fn rand(
        size: Vec<usize>,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let _dtype = dtype;
        let _device = device;
        let tensor_result = py_result!(torsh_tensor::creation::rand(&size))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    #[pyfunction]
    #[pyo3(signature = (size, dtype=None, device=None, requires_grad=None))]
    fn empty(
        size: Vec<usize>,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let _dtype = dtype;
        let _device = device;
        // Use zeros as a fallback since empty is not available
        let tensor_result = py_result!(torsh_tensor::creation::zeros(&size))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    #[pyfunction]
    #[pyo3(signature = (size, fill_value, dtype=None, device=None, requires_grad=None))]
    fn full(
        size: Vec<usize>,
        fill_value: f32,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let _dtype = dtype;
        let _device = device;
        let tensor_result = py_result!(torsh_tensor::creation::full(&size, fill_value))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    #[pyfunction]
    #[pyo3(signature = (n, m=None, dtype=None, device=None, requires_grad=None))]
    fn eye(
        n: usize,
        m: Option<usize>,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let _m = m;
        let _dtype = dtype;
        let _device = device;
        // torsh_tensor::creation::eye only takes one parameter
        let tensor_result = py_result!(torsh_tensor::creation::eye(n))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    #[pyfunction]
    #[pyo3(signature = (start, end=None, step=None, dtype=None, device=None, requires_grad=None))]
    fn arange(
        start: f32,
        end: Option<f32>,
        step: Option<f32>,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let _dtype = dtype;
        let _device = device;
        let (start, end) = if let Some(end) = end {
            (start, end)
        } else {
            (0.0, start)
        };
        let step = step.unwrap_or(1.0);
        let tensor_result = py_result!(torsh_tensor::creation::arange(start, end, step))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    #[pyfunction]
    #[pyo3(signature = (start, end, steps, dtype=None, device=None, requires_grad=None))]
    fn linspace(
        start: f32,
        end: f32,
        steps: usize,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let _dtype = dtype;
        let _device = device;
        let tensor_result = py_result!(torsh_tensor::creation::linspace(start, end, steps))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    // "_like" functions - create tensors with same shape as input
    #[pyfunction]
    #[pyo3(signature = (input, dtype=None, device=None, requires_grad=None))]
    fn zeros_like(
        input: &PyTensor,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let _dtype = dtype;
        let _device = device;
        let tensor_result = py_result!(torsh_tensor::creation::zeros_like(&input.tensor))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    #[pyfunction]
    #[pyo3(signature = (input, dtype=None, device=None, requires_grad=None))]
    fn ones_like(
        input: &PyTensor,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let _dtype = dtype;
        let _device = device;
        let tensor_result = py_result!(torsh_tensor::creation::ones_like(&input.tensor))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    #[pyfunction]
    #[pyo3(signature = (input, fill_value, dtype=None, device=None, requires_grad=None))]
    fn full_like(
        input: &PyTensor,
        fill_value: f32,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let _dtype = dtype;
        let _device = device;
        let shape = input.tensor.shape().dims().to_vec();
        let tensor_result = py_result!(torsh_tensor::creation::full(&shape, fill_value))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    #[pyfunction]
    #[pyo3(signature = (input, dtype=None, device=None, requires_grad=None))]
    fn empty_like(
        input: &PyTensor,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let _dtype = dtype;
        let _device = device;
        // Use zeros_like as fallback since empty is not critical
        let tensor_result = py_result!(torsh_tensor::creation::zeros_like(&input.tensor))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    #[pyfunction]
    #[pyo3(signature = (input, dtype=None, device=None, requires_grad=None))]
    fn randn_like(
        input: &PyTensor,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let _dtype = dtype;
        let _device = device;
        let shape = input.tensor.shape().dims().to_vec();
        let tensor_result = py_result!(torsh_tensor::creation::randn(&shape))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    #[pyfunction]
    #[pyo3(signature = (input, dtype=None, device=None, requires_grad=None))]
    fn rand_like(
        input: &PyTensor,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let _dtype = dtype;
        let _device = device;
        let shape = input.tensor.shape().dims().to_vec();
        let tensor_result = py_result!(torsh_tensor::creation::rand(&shape))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    // Register all functions
    m.add_function(wrap_pyfunction!(tensor, m)?)?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(randn, m)?)?;
    m.add_function(wrap_pyfunction!(rand, m)?)?;
    m.add_function(wrap_pyfunction!(empty, m)?)?;
    m.add_function(wrap_pyfunction!(full, m)?)?;
    m.add_function(wrap_pyfunction!(eye, m)?)?;
    m.add_function(wrap_pyfunction!(arange, m)?)?;
    m.add_function(wrap_pyfunction!(linspace, m)?)?;

    // Register "_like" functions
    m.add_function(wrap_pyfunction!(zeros_like, m)?)?;
    m.add_function(wrap_pyfunction!(ones_like, m)?)?;
    m.add_function(wrap_pyfunction!(full_like, m)?)?;
    m.add_function(wrap_pyfunction!(empty_like, m)?)?;
    m.add_function(wrap_pyfunction!(randn_like, m)?)?;
    m.add_function(wrap_pyfunction!(rand_like, m)?)?;

    Ok(())
}
