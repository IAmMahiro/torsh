//! Optimization algorithms module - PyTorch-compatible optimizers
//!
//! This module provides a modular structure for optimization algorithms:
//! - `base` - Base PyOptimizer class and common functionality
//! - `sgd` - Stochastic Gradient Descent optimizer
//! - `adam` - Adam and AdamW optimizers
//! - `adagrad` - Adagrad optimizer
//! - `rmsprop` - RMSprop optimizer
//! - `lr_scheduler` - PyTorch-compatible `torsh.optim.lr_scheduler` family
//!   (StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, LinearLR,
//!   ReduceLROnPlateau), registered as a nested Python submodule mirroring
//!   `torch.optim.lr_scheduler` (same pattern as the `data`/`autograd`/
//!   `distributed` submodules registered in `lib.rs`).

pub mod adagrad;
pub mod adam;
pub mod base;
pub mod lr_scheduler;
pub mod rmsprop;
pub mod sgd;

// Re-export the main types
pub use adagrad::PyAdaGrad;
pub use adam::{PyAdam, PyAdamW};
pub use base::PyOptimizer;
pub use lr_scheduler::{
    PyCosineAnnealingLR, PyExponentialLR, PyLinearLR, PyMultiStepLR, PyReduceLROnPlateau, PyStepLR,
};
pub use rmsprop::PyRMSprop;
pub use sgd::PySGD;

use pyo3::prelude::*;
use pyo3::types::{PyModule, PyModuleMethods};

/// Register the optim module with Python
pub fn register_optim_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register base optimizer
    m.add_class::<PyOptimizer>()?;

    // Register specific optimizers
    m.add_class::<PySGD>()?;
    m.add_class::<PyAdam>()?;
    m.add_class::<PyAdamW>()?;
    m.add_class::<PyAdaGrad>()?;
    m.add_class::<PyRMSprop>()?;

    // Register the `torsh.optim.lr_scheduler` submodule (PyTorch-shaped).
    let lr_scheduler_module = PyModule::new(py, "lr_scheduler")?;
    lr_scheduler::register_lr_scheduler_module(py, &lr_scheduler_module)?;
    m.add_submodule(&lr_scheduler_module)?;

    Ok(())
}
