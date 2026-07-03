//! Integration tests for `state_dict()` / `load_state_dict()` on every
//! concrete Python optimizer (SGD, Adam, AdamW, Adagrad, RMSprop).
//!
//! Each test: constructs an optimizer over a tiny parameter, drives several
//! real `step()` calls with synthetic gradients (via `set_param_grad`,
//! since neither this crate's Python-visible `Tensor` nor these optimizers
//! currently expose a way to attach a gradient other than that method),
//! captures `state_dict()`, builds a FRESH optimizer instance (deliberately
//! constructed with *different* hyperparameters, to prove `load_state_dict`
//! actually restores them rather than just cosmetically matching) over a
//! parameter seeded with `opt.get_param_data(0)` (the ORIGINAL optimizer's
//! current, already-evolved parameter value -- optimizer `state_dict()`
//! only covers optimizer state, not parameter data, exactly like PyTorch,
//! so a real checkpoint/resume restores both separately), loads the
//! snapshot into it, and asserts the fresh optimizer's state/hyperparameters
//! match the original exactly. It then continues stepping *both* optimizers
//! with the same new gradient and asserts their state still matches --
//! proving the restored buffers are not just structurally present but
//! numerically correct enough to keep training in lock-step (this also
//! exercises weight-decay terms, which depend on the *current* parameter
//! value, not just the gradient).

mod common;

use common::{init_py, run_script};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rstorch_python::optim::{PyAdaGrad, PyAdam, PyAdamW, PyRMSprop, PySGD};
use rstorch_python::PyTensor;

#[test]
fn test_sgd_state_dict_round_trip() {
    init_py();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals
            .set_item("SGD", py.get_type::<PySGD>())
            .expect("set SGD");
        locals
            .set_item("Tensor", py.get_type::<PyTensor>())
            .expect("set Tensor");

        let script = r#"
p1 = [Tensor([1.0, -2.0, 3.0], None, None, None)]
opt = SGD(p1, 0.1, 0.9, 0.0, 0.01, False)
assert opt.state_dict()["state"] == {}, "state must start empty before any step"

for _ in range(5):
    opt.set_param_grad(0, Tensor([0.5, -0.3, 0.2], None, None, None))
    opt.step()

snap = opt.state_dict()
assert "0" in snap["state"], f"expected a momentum buffer for param 0, got {snap['state']}"
assert "momentum_buffer" in snap["state"]["0"]

# Fresh instance, deliberately constructed with DIFFERENT hyperparameters,
# to prove load_state_dict really restores lr/momentum/etc., not just the
# buffers.
p2 = [opt.get_param_data(0)]  # sync the evolved parameter value, not the stale initial one
opt2 = SGD(p2, 0.5, 0.1, 0.2, 0.0, True)
opt2.load_state_dict(snap)

assert abs(opt.lr - opt2.lr) < 1e-9, f"lr: {opt.lr} vs {opt2.lr}"
assert abs(opt.momentum - opt2.momentum) < 1e-9
assert abs(opt.dampening - opt2.dampening) < 1e-9
assert abs(opt.weight_decay - opt2.weight_decay) < 1e-9
assert opt.nesterov == opt2.nesterov
assert_close(opt.state_dict(), opt2.state_dict())

# Continue training both in lock-step with the same new gradient.
opt.set_param_grad(0, Tensor([0.1, 0.4, -0.2], None, None, None))
opt.step()
opt2.set_param_grad(0, Tensor([0.1, 0.4, -0.2], None, None, None))
opt2.step()
assert_close(opt.state_dict(), opt2.state_dict())
"#;

        run_script(py, &locals, script).expect("SGD state_dict round-trip script should succeed");
    });
}

#[test]
fn test_adam_state_dict_round_trip() {
    init_py();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals
            .set_item("Adam", py.get_type::<PyAdam>())
            .expect("set Adam");
        locals
            .set_item("Tensor", py.get_type::<PyTensor>())
            .expect("set Tensor");

        let script = r#"
p1 = [Tensor([1.0, -2.0, 3.0], None, None, None)]
opt = Adam(p1, 0.01, (0.9, 0.999), 1e-8, 0.0, True)
assert opt.state_dict()["state"] == {}, "state must start empty before any step"

for _ in range(5):
    opt.set_param_grad(0, Tensor([0.5, -0.3, 0.2], None, None, None))
    opt.step()

snap = opt.state_dict()
assert "0" in snap["state"]
buf0 = snap["state"]["0"]
assert "exp_avg" in buf0
assert "exp_avg_sq" in buf0
assert "step" in buf0
assert "max_exp_avg_sq" in buf0  # amsgrad=True was requested

# Fresh instance, deliberately WRONG hyperparameters.
p2 = [opt.get_param_data(0)]  # sync the evolved parameter value, not the stale initial one
opt2 = Adam(p2, 0.999, (0.5, 0.5), 1e-4, 0.5, False)
opt2.load_state_dict(snap)

assert abs(opt.lr - opt2.lr) < 1e-9, f"lr: {opt.lr} vs {opt2.lr}"
assert opt.betas == opt2.betas, f"betas: {opt.betas} vs {opt2.betas}"
assert abs(opt.eps - opt2.eps) < 1e-12
assert abs(opt.weight_decay - opt2.weight_decay) < 1e-9
assert opt.amsgrad == opt2.amsgrad
assert_close(opt.state_dict(), opt2.state_dict())

opt.set_param_grad(0, Tensor([0.1, 0.4, -0.2], None, None, None))
opt.step()
opt2.set_param_grad(0, Tensor([0.1, 0.4, -0.2], None, None, None))
opt2.step()
assert_close(opt.state_dict(), opt2.state_dict())
"#;

        run_script(py, &locals, script).expect("Adam state_dict round-trip script should succeed");
    });
}

#[test]
fn test_adamw_state_dict_round_trip() {
    init_py();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals
            .set_item("AdamW", py.get_type::<PyAdamW>())
            .expect("set AdamW");
        locals
            .set_item("Tensor", py.get_type::<PyTensor>())
            .expect("set Tensor");

        let script = r#"
p1 = [Tensor([1.0, -2.0, 3.0], None, None, None)]
opt = AdamW(p1, 0.01, (0.9, 0.999), 1e-8, 0.05, False)
assert opt.state_dict()["state"] == {}

for _ in range(5):
    opt.set_param_grad(0, Tensor([0.5, -0.3, 0.2], None, None, None))
    opt.step()

snap = opt.state_dict()
assert "0" in snap["state"]
assert "exp_avg" in snap["state"]["0"]
assert "exp_avg_sq" in snap["state"]["0"]
assert "step" in snap["state"]["0"]

p2 = [opt.get_param_data(0)]  # sync the evolved parameter value, not the stale initial one
opt2 = AdamW(p2, 0.999, (0.5, 0.5), 1e-4, 0.0, True)
opt2.load_state_dict(snap)

assert abs(opt.lr - opt2.lr) < 1e-9
assert opt.betas == opt2.betas
assert abs(opt.eps - opt2.eps) < 1e-12
assert abs(opt.weight_decay - opt2.weight_decay) < 1e-9
assert opt.amsgrad == opt2.amsgrad
assert_close(opt.state_dict(), opt2.state_dict())

opt.set_param_grad(0, Tensor([0.1, 0.4, -0.2], None, None, None))
opt.step()
opt2.set_param_grad(0, Tensor([0.1, 0.4, -0.2], None, None, None))
opt2.step()
assert_close(opt.state_dict(), opt2.state_dict())
"#;

        run_script(py, &locals, script).expect("AdamW state_dict round-trip script should succeed");
    });
}

#[test]
fn test_adagrad_state_dict_round_trip() {
    init_py();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals
            .set_item("Adagrad", py.get_type::<PyAdaGrad>())
            .expect("set Adagrad");
        locals
            .set_item("Tensor", py.get_type::<PyTensor>())
            .expect("set Tensor");

        let script = r#"
p1 = [Tensor([1.0, -2.0, 3.0], None, None, None)]
opt = Adagrad(p1, 0.1, 0.01, 0.0, 1e-10)
assert opt.state_dict()["state"] == {}

for _ in range(5):
    opt.set_param_grad(0, Tensor([0.5, -0.3, 0.2], None, None, None))
    opt.step()

snap = opt.state_dict()
assert "0" in snap["state"]
assert "sum_of_squares" in snap["state"]["0"]

p2 = [opt.get_param_data(0)]  # sync the evolved parameter value, not the stale initial one
opt2 = Adagrad(p2, 0.999, 0.5, 0.5, 1e-4)
opt2.load_state_dict(snap)

assert abs(opt.lr - opt2.lr) < 1e-9
assert abs(opt.lr_decay - opt2.lr_decay) < 1e-9
assert abs(opt.weight_decay - opt2.weight_decay) < 1e-9
assert abs(opt.eps - opt2.eps) < 1e-12
assert_close(opt.state_dict(), opt2.state_dict())

opt.set_param_grad(0, Tensor([0.1, 0.4, -0.2], None, None, None))
opt.step()
opt2.set_param_grad(0, Tensor([0.1, 0.4, -0.2], None, None, None))
opt2.step()
assert_close(opt.state_dict(), opt2.state_dict())
"#;

        run_script(py, &locals, script)
            .expect("Adagrad state_dict round-trip script should succeed");
    });
}

#[test]
fn test_rmsprop_state_dict_round_trip() {
    init_py();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals
            .set_item("RMSprop", py.get_type::<PyRMSprop>())
            .expect("set RMSprop");
        locals
            .set_item("Tensor", py.get_type::<PyTensor>())
            .expect("set Tensor");

        let script = r#"
p1 = [Tensor([1.0, -2.0, 3.0], None, None, None)]
opt = RMSprop(p1, 0.1, 0.99, 1e-8, 0.0, 0.9, True)
assert opt.state_dict()["state"] == {}

for _ in range(5):
    opt.set_param_grad(0, Tensor([0.5, -0.3, 0.2], None, None, None))
    opt.step()

snap = opt.state_dict()
assert "0" in snap["state"]
buf0 = snap["state"]["0"]
assert "square_avg" in buf0
assert "momentum_buffer" in buf0  # momentum=0.9 was requested
assert "grad_avg" in buf0  # centered=True was requested

p2 = [opt.get_param_data(0)]  # sync the evolved parameter value, not the stale initial one
opt2 = RMSprop(p2, 0.999, 0.5, 1e-4, 0.5, 0.0, False)
opt2.load_state_dict(snap)

assert abs(opt.lr - opt2.lr) < 1e-9
assert abs(opt.alpha - opt2.alpha) < 1e-9
assert abs(opt.eps - opt2.eps) < 1e-12
assert abs(opt.weight_decay - opt2.weight_decay) < 1e-9
assert abs(opt.momentum - opt2.momentum) < 1e-9
assert opt.centered == opt2.centered
assert_close(opt.state_dict(), opt2.state_dict())

opt.set_param_grad(0, Tensor([0.1, 0.4, -0.2], None, None, None))
opt.step()
opt2.set_param_grad(0, Tensor([0.1, 0.4, -0.2], None, None, None))
opt2.step()
assert_close(opt.state_dict(), opt2.state_dict())
"#;

        run_script(py, &locals, script)
            .expect("RMSprop state_dict round-trip script should succeed");
    });
}
