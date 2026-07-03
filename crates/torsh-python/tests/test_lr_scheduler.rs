//! Integration tests for `torsh.optim.lr_scheduler`
//! (StepLR/MultiStepLR/ExponentialLR/CosineAnnealingLR/LinearLR/
//! ReduceLROnPlateau).
//!
//! Every trajectory test computes the expected learning rate at each epoch
//! from a hand-written closed-form formula in the test script itself (not
//! by re-invoking the scheduler's own internals), so a bug in the
//! scheduler's `step()` cannot "agree with itself". `ReduceLROnPlateau`'s
//! patience/reduction logic is likewise checked against manually-counted
//! call numbers.

mod common;

use common::{init_py, run_script};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rstorch_python::optim::{
    PyAdam, PyCosineAnnealingLR, PyExponentialLR, PyLinearLR, PyMultiStepLR, PyReduceLROnPlateau,
    PySGD, PyStepLR,
};
use rstorch_python::PyTensor;

#[test]
fn test_step_lr_trajectory() {
    init_py();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals
            .set_item("SGD", py.get_type::<PySGD>())
            .expect("set SGD");
        locals
            .set_item("Tensor", py.get_type::<PyTensor>())
            .expect("set Tensor");
        locals
            .set_item("StepLR", py.get_type::<PyStepLR>())
            .expect("set StepLR");

        let script = r#"
params = [Tensor([1.0], None, None, None)]
opt = SGD(params, 0.1, 0.0, 0.0, 0.0, False)
scheduler = StepLR(opt, 3, 0.5)

base_lr = 0.1
step_size = 3
gamma = 0.5
for epoch in range(1, 13):
    scheduler.step()
    expected = base_lr * (gamma ** (epoch // step_size))
    got = scheduler.get_last_lr()[0]
    assert abs(got - expected) < 1e-6, f"epoch {epoch}: got {got}, expected {expected}"
    assert abs(opt.lr - expected) < 1e-6, f"epoch {epoch}: optimizer.lr {opt.lr} != {expected}"
    assert scheduler.last_epoch == epoch
"#;

        run_script(py, &locals, script).expect("StepLR trajectory script should succeed");
    });
}

#[test]
fn test_multi_step_lr_trajectory() {
    init_py();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals
            .set_item("SGD", py.get_type::<PySGD>())
            .expect("set SGD");
        locals
            .set_item("Tensor", py.get_type::<PyTensor>())
            .expect("set Tensor");
        locals
            .set_item("MultiStepLR", py.get_type::<PyMultiStepLR>())
            .expect("set MultiStepLR");

        let script = r#"
params = [Tensor([1.0], None, None, None)]
opt = SGD(params, 0.1, 0.0, 0.0, 0.0, False)
milestones = [3, 6, 9]
scheduler = MultiStepLR(opt, milestones, 0.1)

base_lr = 0.1
gamma = 0.1
for epoch in range(1, 13):
    scheduler.step()
    passed = sum(1 for m in milestones if epoch >= m)
    expected = base_lr * (gamma ** passed)
    got = scheduler.get_last_lr()[0]
    assert abs(got - expected) < 1e-6, f"epoch {epoch}: got {got}, expected {expected}"
    assert abs(opt.lr - expected) < 1e-6
"#;

        run_script(py, &locals, script).expect("MultiStepLR trajectory script should succeed");
    });
}

#[test]
fn test_exponential_lr_trajectory() {
    init_py();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals
            .set_item("SGD", py.get_type::<PySGD>())
            .expect("set SGD");
        locals
            .set_item("Tensor", py.get_type::<PyTensor>())
            .expect("set Tensor");
        locals
            .set_item("ExponentialLR", py.get_type::<PyExponentialLR>())
            .expect("set ExponentialLR");

        let script = r#"
params = [Tensor([1.0], None, None, None)]
opt = SGD(params, 0.1, 0.0, 0.0, 0.0, False)
scheduler = ExponentialLR(opt, 0.9)

base_lr = 0.1
gamma = 0.9
for epoch in range(1, 9):
    scheduler.step()
    expected = base_lr * (gamma ** epoch)
    got = scheduler.get_last_lr()[0]
    assert abs(got - expected) < 1e-6, f"epoch {epoch}: got {got}, expected {expected}"
    assert abs(opt.lr - expected) < 1e-6
"#;

        run_script(py, &locals, script).expect("ExponentialLR trajectory script should succeed");
    });
}

#[test]
fn test_cosine_annealing_lr_trajectory() {
    init_py();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals
            .set_item("SGD", py.get_type::<PySGD>())
            .expect("set SGD");
        locals
            .set_item("Tensor", py.get_type::<PyTensor>())
            .expect("set Tensor");
        locals
            .set_item("CosineAnnealingLR", py.get_type::<PyCosineAnnealingLR>())
            .expect("set CosineAnnealingLR");

        let script = r#"
import math

params = [Tensor([1.0], None, None, None)]
opt = SGD(params, 0.1, 0.0, 0.0, 0.0, False)
scheduler = CosineAnnealingLR(opt, 10, 0.001)

base_lr = 0.1
eta_min = 0.001
t_max = 10
for epoch in range(1, 21):
    scheduler.step()
    expected = eta_min + (base_lr - eta_min) * (1.0 + math.cos(math.pi * epoch / t_max)) / 2.0
    got = scheduler.get_last_lr()[0]
    assert abs(got - expected) < 1e-5, f"epoch {epoch}: got {got}, expected {expected}"
    assert abs(opt.lr - expected) < 1e-5
"#;

        run_script(py, &locals, script)
            .expect("CosineAnnealingLR trajectory script should succeed");
    });
}

#[test]
fn test_linear_lr_trajectory() {
    init_py();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals
            .set_item("SGD", py.get_type::<PySGD>())
            .expect("set SGD");
        locals
            .set_item("Tensor", py.get_type::<PyTensor>())
            .expect("set Tensor");
        locals
            .set_item("LinearLR", py.get_type::<PyLinearLR>())
            .expect("set LinearLR");

        let script = r#"
params = [Tensor([1.0], None, None, None)]
opt = SGD(params, 0.2, 0.0, 0.0, 0.0, False)
scheduler = LinearLR(opt, 0.1, 1.0, 10)

base_lr = 0.2
start_factor = 0.1
end_factor = 1.0
total_iters = 10
for epoch in range(1, 16):
    scheduler.step()
    if epoch >= total_iters:
        factor = end_factor
    else:
        factor = start_factor + (end_factor - start_factor) * (epoch / total_iters)
    expected = base_lr * factor
    got = scheduler.get_last_lr()[0]
    assert abs(got - expected) < 1e-6, f"epoch {epoch}: got {got}, expected {expected}"
    assert abs(opt.lr - expected) < 1e-6
"#;

        run_script(py, &locals, script).expect("LinearLR trajectory script should succeed");
    });
}

#[test]
fn test_reduce_lr_on_plateau_drops_exactly_after_patience() {
    init_py();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals
            .set_item("SGD", py.get_type::<PySGD>())
            .expect("set SGD");
        locals
            .set_item("Tensor", py.get_type::<PyTensor>())
            .expect("set Tensor");
        locals
            .set_item("ReduceLROnPlateau", py.get_type::<PyReduceLROnPlateau>())
            .expect("set ReduceLROnPlateau");

        // With a CONSTANT (never-improving) metric: call #1 only establishes
        // the baseline (never counts as a bad epoch). Calls #2..#(patience+1)
        // accumulate `num_bad_epochs` up to `patience`, which is NOT enough
        // to trigger a reduction (`num_bad_epochs > patience` is required,
        // matching torsh_optim's / PyTorch's semantics). Only the very next
        // call (#(patience+2)) pushes `num_bad_epochs` to `patience + 1` and
        // triggers exactly one reduction by `factor`.
        let script = r#"
params = [Tensor([1.0], None, None, None)]
opt = SGD(params, 1.0, 0.0, 0.0, 0.0, False)
patience = 3
factor = 0.5
scheduler = ReduceLROnPlateau(opt, "min", factor, patience, 1e-4)

base_lr = 1.0
scheduler.step(5.0)  # call #1: baseline only
assert abs(opt.lr - base_lr) < 1e-9, "must not reduce on the very first call"
assert scheduler.num_bad_epochs == 0

for i in range(patience):
    scheduler.step(5.0)  # calls #2..#(patience+1)
    assert abs(opt.lr - base_lr) < 1e-9, f"must not reduce before bad_epochs > patience (call {i})"
    assert scheduler.num_bad_epochs == i + 1

scheduler.step(5.0)  # call #(patience+2): num_bad_epochs == patience+1 > patience
expected = base_lr * factor
assert abs(opt.lr - expected) < 1e-9, f"lr should have dropped to {expected}, got {opt.lr}"
assert abs(scheduler.get_last_lr()[0] - expected) < 1e-9
assert scheduler.num_bad_epochs == 0, "bad-epoch counter resets after a reduction"
"#;

        run_script(py, &locals, script).expect("ReduceLROnPlateau patience script should succeed");
    });
}

#[test]
fn test_step_lr_state_dict_resume() {
    init_py();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals
            .set_item("SGD", py.get_type::<PySGD>())
            .expect("set SGD");
        locals
            .set_item("Tensor", py.get_type::<PyTensor>())
            .expect("set Tensor");
        locals
            .set_item("StepLR", py.get_type::<PyStepLR>())
            .expect("set StepLR");

        let script = r#"
params = [Tensor([1.0], None, None, None)]
opt = SGD(params, 0.1, 0.0, 0.0, 0.0, False)
scheduler = StepLR(opt, 3, 0.5)
for _ in range(5):
    scheduler.step()
snap = scheduler.state_dict()
lr_at_5 = scheduler.get_last_lr()[0]

# Continue the ORIGINAL scheduler to know what the trajectory "would have
# been" past the snapshot point.
for _ in range(3):
    scheduler.step()
expected_lr_at_8 = scheduler.get_last_lr()[0]

# Fresh optimizer + scheduler that never took a step, then resume from snap.
params2 = [Tensor([1.0], None, None, None)]
opt2 = SGD(params2, 0.1, 0.0, 0.0, 0.0, False)
scheduler2 = StepLR(opt2, 3, 0.5)
scheduler2.load_state_dict(snap)

assert abs(scheduler2.get_last_lr()[0] - lr_at_5) < 1e-6
assert abs(opt2.lr - lr_at_5) < 1e-6
assert scheduler2.last_epoch == 5

for _ in range(3):
    scheduler2.step()

assert abs(scheduler2.get_last_lr()[0] - expected_lr_at_8) < 1e-6, (
    f"resumed trajectory diverged: {scheduler2.get_last_lr()[0]} vs {expected_lr_at_8}"
)
assert abs(opt2.lr - expected_lr_at_8) < 1e-6
"#;

        run_script(py, &locals, script).expect("StepLR resume script should succeed");
    });
}

#[test]
fn test_reduce_lr_on_plateau_state_dict_resume() {
    init_py();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals
            .set_item("SGD", py.get_type::<PySGD>())
            .expect("set SGD");
        locals
            .set_item("Tensor", py.get_type::<PyTensor>())
            .expect("set Tensor");
        locals
            .set_item("ReduceLROnPlateau", py.get_type::<PyReduceLROnPlateau>())
            .expect("set ReduceLROnPlateau");

        let script = r#"
params = [Tensor([1.0], None, None, None)]
opt = SGD(params, 1.0, 0.0, 0.0, 0.0, False)
scheduler = ReduceLROnPlateau(opt, "min", 0.5, 1, 1e-4)

scheduler.step(10.0)  # call #1: baseline
scheduler.step(10.0)  # call #2: bad epoch #1 (num_bad_epochs=1, not yet > patience=1)
snap = scheduler.state_dict()
lr_before_resume = opt.lr
assert abs(lr_before_resume - 1.0) < 1e-9, "must not have reduced yet"
assert scheduler.num_bad_epochs == 1

params2 = [Tensor([1.0], None, None, None)]
opt2 = SGD(params2, 1.0, 0.0, 0.0, 0.0, False)
scheduler2 = ReduceLROnPlateau(opt2, "min", 0.5, 1, 1e-4)
scheduler2.load_state_dict(snap)
assert abs(opt2.lr - lr_before_resume) < 1e-9
assert scheduler2.num_bad_epochs == 1

# One more non-improving call on BOTH: both must cross
# num_bad_epochs(=2) > patience(=1) at the same point and drop by `factor`.
scheduler.step(10.0)
scheduler2.step(10.0)
assert abs(opt.lr - 0.5) < 1e-9
assert abs(opt2.lr - 0.5) < 1e-9
assert abs(opt.lr - opt2.lr) < 1e-9
"#;

        run_script(py, &locals, script).expect("ReduceLROnPlateau resume script should succeed");
    });
}

/// Regression test for the `set_lr` cascade fix made alongside the
/// scheduler work: `Adam`/`AdamW`/`Adagrad`/`RMSprop` wrap an internal
/// `torsh_optim` optimizer that reads its OWN copy of `lr`. Before the fix,
/// `set_lr` only updated the pyclass's `self.lr` field (and the cosmetic
/// `param_groups` dict) -- `step()` kept using whatever `lr` the wrapped
/// optimizer was originally constructed with, so an LR scheduler driving
/// `optimizer.lr` would have had **zero** effect on actual parameter
/// updates. This drives `lr` to (approximately) zero via a scheduler, then
/// proves a subsequent `step()` with a large gradient leaves the parameter
/// unchanged -- which only holds if `set_lr` really reached the wrapped
/// optimizer.
#[test]
fn test_scheduler_lr_cascades_into_wrapped_optimizer_step() {
    init_py();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals
            .set_item("Adam", py.get_type::<PyAdam>())
            .expect("set Adam");
        locals
            .set_item("Tensor", py.get_type::<PyTensor>())
            .expect("set Tensor");
        locals
            .set_item("ExponentialLR", py.get_type::<PyExponentialLR>())
            .expect("set ExponentialLR");

        let script = r#"
params = [Tensor([1.0, 1.0, 1.0], None, None, None)]
opt = Adam(params, 1.0, (0.9, 0.999), 1e-8, 0.0, False)
scheduler = ExponentialLR(opt, 0.0)  # gamma=0 -> lr collapses to ~0 after one step()

before = opt.get_param_data(0).tolist()

scheduler.step()
assert abs(opt.lr) < 1e-9, f"lr should have collapsed to ~0, got {opt.lr}"

opt.set_param_grad(0, Tensor([10.0, -10.0, 5.0], None, None, None))
opt.step()

after = opt.get_param_data(0).tolist()
assert_close(before, after, tol=1e-6)
"#;

        run_script(py, &locals, script).expect(
            "lr-cascade regression script should succeed: if `set_lr` did not \
             propagate into the wrapped torsh_optim::Adam, the parameter would \
             have moved despite lr being driven to ~0 by the scheduler",
        );
    });
}
