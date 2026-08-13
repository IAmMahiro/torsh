//! Functional API bindings - PyO3 0.25 compatible

use crate::{error::PyResult, py_result, tensor::PyTensor};
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyModuleMethods};
use pyo3::wrap_pyfunction;

// ===============================
// Activation Functions
// ===============================

#[pyfunction]
#[pyo3(signature = (input, inplace=None))]
fn relu(input: &PyTensor, inplace: Option<bool>) -> PyResult<PyTensor> {
    let _ = inplace;
    let result = py_result!(input.tensor.relu())?;
    Ok(PyTensor { tensor: result })
}

#[pyfunction]
#[pyo3(signature = (input, inplace=None))]
fn relu6(input: &PyTensor, inplace: Option<bool>) -> PyResult<PyTensor> {
    let _ = inplace;
    // ReLU6: clamp(0, 6)
    let result = py_result!(input.tensor.clamp(0.0, 6.0))?;
    Ok(PyTensor { tensor: result })
}

#[pyfunction]
#[pyo3(signature = (input, negative_slope=None, inplace=None))]
fn leaky_relu(
    input: &PyTensor,
    negative_slope: Option<f32>,
    inplace: Option<bool>,
) -> PyResult<PyTensor> {
    let _ = inplace;
    // Leaky ReLU: x if x >= 0 else negative_slope * x
    let slope = negative_slope.unwrap_or(0.01) as f64;
    let result = py_result!(torsh_functional::leaky_relu(&input.tensor, slope, false))?;
    Ok(PyTensor { tensor: result })
}

#[pyfunction]
#[pyo3(signature = (input, alpha=None, inplace=None))]
fn elu(input: &PyTensor, alpha: Option<f32>, inplace: Option<bool>) -> PyResult<PyTensor> {
    let _ = inplace;
    let alpha = alpha.unwrap_or(1.0);
    // ELU: x if x > 0 else alpha * (exp(x) - 1)
    let result = py_result!(torsh_functional::elu(&input.tensor, alpha as f64, false))?;
    Ok(PyTensor { tensor: result })
}

#[pyfunction]
#[pyo3(signature = (input, inplace=None))]
fn selu(input: &PyTensor, inplace: Option<bool>) -> PyResult<PyTensor> {
    let _ = inplace;
    // SELU: scale * (x if x > 0 else alpha * (exp(x) - 1))
    let result = py_result!(torsh_functional::selu(&input.tensor, false))?;
    Ok(PyTensor { tensor: result })
}

#[pyfunction]
#[pyo3(signature = (input, approximate=None))]
fn gelu(input: &PyTensor, approximate: Option<String>) -> PyResult<PyTensor> {
    // GELU: 0.5 * x * (1 + erf(x / sqrt(2))). Only the exact ('none') variant
    // is implemented; refuse the 'tanh' approximation rather than silently
    // returning the exact result under a different name.
    match approximate.as_deref() {
        None | Some("none") => {}
        Some(other) => {
            return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                format!(
                    "gelu(approximate={:?}) is not implemented; only 'none' is supported",
                    other
                ),
            ))
        }
    }
    let result = py_result!(torsh_functional::gelu(&input.tensor))?;
    Ok(PyTensor { tensor: result })
}

#[pyfunction]
#[pyo3(signature = (input, inplace=None))]
fn silu(input: &PyTensor, inplace: Option<bool>) -> PyResult<PyTensor> {
    let _ = inplace;
    // SiLU: x * sigmoid(x)
    let sigmoid_result = py_result!(input.tensor.sigmoid())?;
    let result = py_result!(input.tensor.mul(&sigmoid_result))?;
    Ok(PyTensor { tensor: result })
}

#[pyfunction]
#[pyo3(signature = (input, inplace=None))]
fn mish(input: &PyTensor, inplace: Option<bool>) -> PyResult<PyTensor> {
    let _ = inplace;
    // Mish: x * tanh(softplus(x))
    let result = py_result!(torsh_functional::mish(&input.tensor, false))?;
    Ok(PyTensor { tensor: result })
}

#[pyfunction]
fn sigmoid(input: &PyTensor) -> PyResult<PyTensor> {
    let result = py_result!(input.tensor.sigmoid())?;
    Ok(PyTensor { tensor: result })
}

#[pyfunction]
fn tanh(input: &PyTensor) -> PyResult<PyTensor> {
    let result = py_result!(input.tensor.tanh())?;
    Ok(PyTensor { tensor: result })
}

#[pyfunction]
#[pyo3(signature = (input, dim, dtype=None))]
fn softmax(input: &PyTensor, dim: i32, dtype: Option<String>) -> PyResult<PyTensor> {
    let _ = dtype;
    let result = py_result!(input.tensor.softmax(dim))?;
    Ok(PyTensor { tensor: result })
}

#[pyfunction]
#[pyo3(signature = (input, dim, dtype=None))]
fn log_softmax(input: &PyTensor, dim: i32, dtype: Option<String>) -> PyResult<PyTensor> {
    let _ = dtype;
    // log_softmax = log(softmax(x))
    let softmax_result = py_result!(input.tensor.softmax(dim))?;
    let result = py_result!(softmax_result.log())?;
    Ok(PyTensor { tensor: result })
}

#[pyfunction]
#[pyo3(signature = (input, beta=None, threshold=None))]
fn softplus(input: &PyTensor, beta: Option<f32>, threshold: Option<f32>) -> PyResult<PyTensor> {
    // Softplus: (1/beta) * ln(1 + exp(beta * x)); linear above `threshold`.
    let beta = beta.unwrap_or(1.0) as f64;
    let threshold = threshold.unwrap_or(20.0) as f64;
    let result = py_result!(torsh_functional::softplus(&input.tensor, beta, threshold))?;
    Ok(PyTensor { tensor: result })
}

#[pyfunction]
fn softsign(input: &PyTensor) -> PyResult<PyTensor> {
    // Softsign: x / (1 + |x|)
    let result = py_result!(torsh_functional::softsign(&input.tensor))?;
    Ok(PyTensor { tensor: result })
}

// ===============================
// Loss Functions
// ===============================

#[pyfunction]
#[pyo3(signature = (input, target, reduction=None))]
fn mse_loss(input: &PyTensor, target: &PyTensor, reduction: Option<String>) -> PyResult<PyTensor> {
    // MSE: (input - target)^2
    let diff = py_result!(input.tensor.sub(&target.tensor))?;
    let squared = py_result!(diff.mul(&diff))?;
    let result = match reduction.as_deref() {
        Some("mean") | None => py_result!(squared.mean(None, false))?,
        Some("sum") => py_result!(squared.sum())?,
        Some("none") => squared,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Invalid reduction",
            ))
        }
    };
    Ok(PyTensor { tensor: result })
}

#[pyfunction]
#[pyo3(signature = (input, target, weight=None, size_average=None, ignore_index=None, reduce=None, reduction=None, label_smoothing=None))]
fn cross_entropy(
    input: &PyTensor,
    target: &PyTensor,
    weight: Option<&PyTensor>,
    size_average: Option<bool>,
    ignore_index: Option<i64>,
    reduce: Option<bool>,
    reduction: Option<String>,
    label_smoothing: Option<f32>,
) -> PyResult<PyTensor> {
    let _ = size_average;
    let _ = reduce;
    let reduction = reduction.as_deref().unwrap_or("mean");
    let result = py_result!(torsh_functional::cross_entropy(
        &input.tensor,
        &target.tensor,
        weight.map(|w| &w.tensor),
        reduction,
        ignore_index,
        label_smoothing.unwrap_or(0.0) as f64,
    ))?;
    Ok(PyTensor { tensor: result })
}

#[pyfunction]
#[pyo3(signature = (input, target, reduction=None))]
fn l1_loss(input: &PyTensor, target: &PyTensor, reduction: Option<String>) -> PyResult<PyTensor> {
    // L1: |input - target|
    let diff = py_result!(input.tensor.sub(&target.tensor))?;
    let abs_diff = py_result!(diff.abs())?;
    let result = match reduction.as_deref() {
        Some("mean") | None => py_result!(abs_diff.mean(None, false))?,
        Some("sum") => py_result!(abs_diff.sum())?,
        Some("none") => abs_diff,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Invalid reduction",
            ))
        }
    };
    Ok(PyTensor { tensor: result })
}

#[pyfunction]
#[pyo3(signature = (input, target, weight=None, size_average=None, reduce=None, reduction=None))]
fn binary_cross_entropy(
    input: &PyTensor,
    target: &PyTensor,
    weight: Option<&PyTensor>,
    size_average: Option<bool>,
    reduce: Option<bool>,
    reduction: Option<String>,
) -> PyResult<PyTensor> {
    let _ = size_average;
    let _ = reduce;
    use torsh_functional::loss::ReductionType;
    let reduction = match reduction.as_deref().unwrap_or("mean") {
        "mean" => ReductionType::Mean,
        "sum" => ReductionType::Sum,
        "none" => ReductionType::None,
        other => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid reduction: {}",
                other
            )))
        }
    };
    // BCE: -[target * log(input) + (1 - target) * log(1 - input)]
    let result = py_result!(torsh_functional::binary_cross_entropy(
        &input.tensor,
        &target.tensor,
        weight.map(|w| &w.tensor),
        reduction,
    ))?;
    Ok(PyTensor { tensor: result })
}

// ===============================
// Pooling Functions
// ===============================

#[pyfunction]
#[pyo3(signature = (input, kernel_size, stride=None, padding=None, dilation=None, ceil_mode=None, return_indices=None))]
fn max_pool2d(
    input: &PyTensor,
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: Option<(usize, usize)>,
    dilation: Option<(usize, usize)>,
    ceil_mode: Option<bool>,
    return_indices: Option<bool>,
) -> PyResult<PyTensor> {
    // PyTorch returns (output, indices) when return_indices=True. This binding
    // returns only the pooled tensor, so refuse the tuple form rather than
    // dropping the indices silently.
    if return_indices == Some(true) {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "max_pool2d(return_indices=True) is not supported; indices are not returned",
        ));
    }
    let (result, _indices) = py_result!(torsh_functional::max_pool2d(
        &input.tensor,
        kernel_size,
        stride,
        padding.unwrap_or((0, 0)),
        dilation.unwrap_or((1, 1)),
        ceil_mode.unwrap_or(false),
        false,
    ))?;
    Ok(PyTensor { tensor: result })
}

#[pyfunction]
#[pyo3(signature = (input, kernel_size, stride=None, padding=None, ceil_mode=None, count_include_pad=None, divisor_override=None))]
fn avg_pool2d(
    input: &PyTensor,
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: Option<(usize, usize)>,
    ceil_mode: Option<bool>,
    count_include_pad: Option<bool>,
    divisor_override: Option<usize>,
) -> PyResult<PyTensor> {
    let result = py_result!(torsh_functional::avg_pool2d(
        &input.tensor,
        kernel_size,
        stride,
        padding.unwrap_or((0, 0)),
        ceil_mode.unwrap_or(false),
        count_include_pad.unwrap_or(true),
        divisor_override,
    ))?;
    Ok(PyTensor { tensor: result })
}

#[pyfunction]
fn adaptive_avg_pool2d(input: &PyTensor, output_size: (usize, usize)) -> PyResult<PyTensor> {
    let result = py_result!(torsh_functional::adaptive_avg_pool2d(
        &input.tensor,
        output_size
    ))?;
    Ok(PyTensor { tensor: result })
}

#[pyfunction]
#[pyo3(signature = (input, output_size, return_indices=None))]
fn adaptive_max_pool2d(
    input: &PyTensor,
    output_size: (usize, usize),
    return_indices: Option<bool>,
) -> PyResult<PyTensor> {
    if return_indices == Some(true) {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "adaptive_max_pool2d(return_indices=True) is not supported; indices are not returned",
        ));
    }
    let result = py_result!(torsh_functional::adaptive_max_pool2d(
        &input.tensor,
        output_size
    ))?;
    Ok(PyTensor { tensor: result })
}

// ===============================
// Normalization Functions
// ===============================

#[pyfunction]
#[pyo3(signature = (input, running_mean=None, running_var=None, weight=None, bias=None, training=None, momentum=None, eps=None))]
fn batch_norm(
    input: &PyTensor,
    running_mean: Option<&PyTensor>,
    running_var: Option<&PyTensor>,
    weight: Option<&PyTensor>,
    bias: Option<&PyTensor>,
    training: Option<bool>,
    momentum: Option<f32>,
    eps: Option<f32>,
) -> PyResult<PyTensor> {
    let result = py_result!(torsh_functional::batch_norm(
        &input.tensor,
        running_mean.map(|t| &t.tensor),
        running_var.map(|t| &t.tensor),
        weight.map(|t| &t.tensor),
        bias.map(|t| &t.tensor),
        training.unwrap_or(false),
        momentum.unwrap_or(0.1) as f64,
        eps.unwrap_or(1e-5) as f64,
    ))?;
    Ok(PyTensor { tensor: result })
}

#[pyfunction]
#[pyo3(signature = (input, normalized_shape, weight=None, bias=None, eps=None))]
fn layer_norm(
    input: &PyTensor,
    normalized_shape: Vec<usize>,
    weight: Option<&PyTensor>,
    bias: Option<&PyTensor>,
    eps: Option<f32>,
) -> PyResult<PyTensor> {
    let result = py_result!(torsh_functional::layer_norm(
        &input.tensor,
        &normalized_shape,
        weight.map(|t| &t.tensor),
        bias.map(|t| &t.tensor),
        eps.unwrap_or(1e-5) as f64,
    ))?;
    Ok(PyTensor { tensor: result })
}

// ===============================
// Dropout Functions
// ===============================

#[pyfunction]
#[pyo3(signature = (input, p=None, training=None, inplace=None))]
fn dropout(
    input: &PyTensor,
    p: Option<f32>,
    training: Option<bool>,
    inplace: Option<bool>,
) -> PyResult<PyTensor> {
    let _ = inplace;
    // Inverted dropout with a real Bernoulli mask (identity in eval mode).
    let p = p.unwrap_or(0.5) as f64;
    let training = training.unwrap_or(true);
    let result = py_result!(torsh_functional::dropout(&input.tensor, p, training, false))?;
    Ok(PyTensor { tensor: result })
}

// ===============================
// Linear Algebra Functions
// ===============================

#[pyfunction]
#[pyo3(signature = (input, weight, bias=None))]
fn linear(input: &PyTensor, weight: &PyTensor, bias: Option<&PyTensor>) -> PyResult<PyTensor> {
    let result = py_result!(input.tensor.matmul(&weight.tensor))?;
    if let Some(b) = bias {
        let result = py_result!(result.add(&b.tensor))?;
        Ok(PyTensor { tensor: result })
    } else {
        Ok(PyTensor { tensor: result })
    }
}

#[pyfunction]
#[pyo3(signature = (input, weight, bias=None, stride=None, padding=None, dilation=None, groups=None))]
fn conv2d(
    input: &PyTensor,
    weight: &PyTensor,
    bias: Option<&PyTensor>,
    stride: Option<(usize, usize)>,
    padding: Option<(usize, usize)>,
    dilation: Option<(usize, usize)>,
    groups: Option<usize>,
) -> PyResult<PyTensor> {
    let stride = stride.unwrap_or((1, 1));
    let padding = padding.unwrap_or((0, 0));
    let dilation = dilation.unwrap_or((1, 1));
    let groups = groups.unwrap_or(1);

    let result = py_result!(input.tensor.conv2d(
        &weight.tensor,
        bias.map(|b| &b.tensor),
        padding,
        stride,
        dilation,
        groups
    ))?;
    Ok(PyTensor { tensor: result })
}

/// Register functional module
pub fn register_functional_module(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add activation functions
    m.add_function(wrap_pyfunction!(relu, m)?)?;
    m.add_function(wrap_pyfunction!(relu6, m)?)?;
    m.add_function(wrap_pyfunction!(leaky_relu, m)?)?;
    m.add_function(wrap_pyfunction!(elu, m)?)?;
    m.add_function(wrap_pyfunction!(selu, m)?)?;
    m.add_function(wrap_pyfunction!(gelu, m)?)?;
    m.add_function(wrap_pyfunction!(silu, m)?)?;
    m.add_function(wrap_pyfunction!(mish, m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(tanh, m)?)?;
    m.add_function(wrap_pyfunction!(softmax, m)?)?;
    m.add_function(wrap_pyfunction!(log_softmax, m)?)?;
    m.add_function(wrap_pyfunction!(softplus, m)?)?;
    m.add_function(wrap_pyfunction!(softsign, m)?)?;

    // Add loss functions
    m.add_function(wrap_pyfunction!(mse_loss, m)?)?;
    m.add_function(wrap_pyfunction!(cross_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(l1_loss, m)?)?;
    m.add_function(wrap_pyfunction!(binary_cross_entropy, m)?)?;

    // Add pooling functions
    m.add_function(wrap_pyfunction!(max_pool2d, m)?)?;
    m.add_function(wrap_pyfunction!(avg_pool2d, m)?)?;
    m.add_function(wrap_pyfunction!(adaptive_avg_pool2d, m)?)?;
    m.add_function(wrap_pyfunction!(adaptive_max_pool2d, m)?)?;

    // Add normalization functions
    m.add_function(wrap_pyfunction!(batch_norm, m)?)?;
    m.add_function(wrap_pyfunction!(layer_norm, m)?)?;

    // Add dropout functions
    m.add_function(wrap_pyfunction!(dropout, m)?)?;

    // Add linear algebra functions
    m.add_function(wrap_pyfunction!(linear, m)?)?;
    m.add_function(wrap_pyfunction!(conv2d, m)?)?;

    Ok(())
}

/// Register the subset of functional operations that the pure-Python
/// `rstorch.nn` shim imports directly from `_C.nn`
/// (`relu`, `sigmoid`, `tanh`, `softmax`, `log_softmax`, `mse_loss`,
/// `cross_entropy`). Kept in sync with `python/rstorch/nn.py`.
pub fn register_nn_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(relu, m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(tanh, m)?)?;
    m.add_function(wrap_pyfunction!(softmax, m)?)?;
    m.add_function(wrap_pyfunction!(log_softmax, m)?)?;
    m.add_function(wrap_pyfunction!(mse_loss, m)?)?;
    m.add_function(wrap_pyfunction!(cross_entropy, m)?)?;
    Ok(())
}
