//! Python bindings for ToRSh - PyTorch-compatible deep learning in Rust
//!
//! This crate provides Python bindings for the ToRSh deep learning framework,
//! enabling PyTorch-compatible APIs to be used from Python.
//!
//! # Modular Structure
//!
//! The crate is organized into focused modules:
//! - `tensor` - Tensor operations and creation functions
//! - `nn` - Neural network layers and containers
//! - `optim` - Optimization algorithms
//! - `device` - Device management and utilities
//! - `dtype` - Data type definitions and conversions
//! - `error` - Error handling and conversions
//! - `utils` - Common utilities and helpers

use pyo3::prelude::*;

// Core modules - modular structure
pub mod device;
pub mod dtype;
pub mod error;
pub mod nn;
pub mod optim;
pub mod tensor;
pub mod utils;

pub mod autograd;
pub mod data;
pub mod distributed;
pub mod functional;

// Re-export main types
pub use device::PyDevice;
pub use dtype::PyDType;
pub use error::TorshPyError;
pub use tensor::PyTensor;

/// ToRSh compiled extension module.
///
/// This is exposed to Python as `rstorch._C` (the `#[pyo3(name = "_C")]`
/// produces the `PyInit__C` symbol that maturin's `module-name =
/// "rstorch._C"` expects). The pure-Python `rstorch` package wraps it.
#[pymodule]
#[pyo3(name = "_C")]
fn rstorch(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();

    // Register main classes
    m.add_class::<PyTensor>()?;
    m.add_class::<PyDevice>()?;
    m.add_class::<PyDType>()?;

    // Build real child submodules so that `rstorch._C.nn` / `rstorch._C.optim`
    // exist as their own modules (rather than classes landing on the root).
    let nn_module = PyModule::new(py, "nn")?;
    nn::register_nn_module(py, &nn_module)?;
    // The Python `rstorch.nn` shim also imports free functions (relu, sigmoid,
    // ...) from `_C.nn`; register them alongside the layer classes.
    functional::register_nn_functions(&nn_module)?;
    m.add_submodule(&nn_module)?;

    let optim_module = PyModule::new(py, "optim")?;
    optim::register_optim_module(py, &optim_module)?;
    m.add_submodule(&optim_module)?;

    let data_module = PyModule::new(py, "data")?;
    data::register_data_module(py, &data_module)?;
    m.add_submodule(&data_module)?;

    let autograd_module = PyModule::new(py, "autograd")?;
    autograd::register_autograd_module(py, &autograd_module)?;
    m.add_submodule(&autograd_module)?;

    let distributed_module = PyModule::new(py, "distributed")?;
    distributed::register_distributed_module(py, &distributed_module)?;
    m.add_submodule(&distributed_module)?;

    let functional_module = PyModule::new(py, "functional")?;
    functional::register_functional_module(py, &functional_module)?;
    m.add_submodule(&functional_module)?;

    // Add tensor creation functions
    tensor::register_creation_functions(m)?;

    // Add device and dtype constants
    device::register_device_constants(m)?;
    dtype::register_dtype_constants(m)?;

    // Register error types
    error::register_error_types(m)?;

    // Set version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // PyO3's `add_submodule` only sets an attribute on the parent; it does NOT
    // register the child in `sys.modules`, so `import rstorch._C.autograd`
    // would otherwise fail. Register every submodule under its fully-qualified
    // dotted name so the pure-Python package can import from them.
    let sys_modules = py.import("sys")?.getattr("modules")?;
    for (name, module) in [
        ("rstorch._C.nn", &nn_module),
        ("rstorch._C.optim", &optim_module),
        ("rstorch._C.data", &data_module),
        ("rstorch._C.autograd", &autograd_module),
        ("rstorch._C.distributed", &distributed_module),
        ("rstorch._C.functional", &functional_module),
    ] {
        module.setattr("__name__", name)?;
        sys_modules.set_item(name, module)?;
    }
    // `lr_scheduler` is a nested submodule created inside `register_optim_module`.
    if let Ok(lr_scheduler) = optim_module.getattr("lr_scheduler") {
        lr_scheduler.setattr("__name__", "rstorch._C.optim.lr_scheduler")?;
        sys_modules.set_item("rstorch._C.optim.lr_scheduler", &lr_scheduler)?;
    }

    Ok(())
}
