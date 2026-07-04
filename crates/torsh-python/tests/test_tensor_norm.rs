//! Integration tests for `Tensor.norm(p, dim, keepdim)`.
//!
//! `PyTensor::norm` used to silently ignore `p`, `dim`, and `keepdim`,
//! always returning the whole-tensor L2 norm no matter what the caller
//! asked for -- a silent wrong-result bug (e.g. `t.norm(p=1)` returned an
//! L1 norm to nobody: it actually returned the L2 norm). These tests
//! exercise the real Lp-norm behavior: `p` actually selects the norm order
//! (L1 vs L2 give different, hand-computed results), and `dim`/`keepdim`
//! actually perform a per-dimension reduction instead of being dropped on
//! the floor. Every assertion below except the "default still equals p=2"
//! one fails against the old code.

mod common;

use common::{init_py, run_script};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rstorch_python::PyTensor;

#[test]
fn test_norm_p1_vs_p2_global() {
    init_py();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals
            .set_item("Tensor", py.get_type::<PyTensor>())
            .expect("set Tensor");

        let script = r#"
t = Tensor([3.0, -4.0], None, None, None)

l1 = t.norm(p=1.0)
l2 = t.norm(p=2.0)
default = t.norm()

# Hand-computed: L1 = |3| + |-4| = 7; L2 = sqrt(3^2 + (-4)^2) = sqrt(25) = 5.
assert abs(l1.item() - 7.0) < 1e-5, f"L1 norm: expected 7.0, got {l1.item()}"
assert abs(l2.item() - 5.0) < 1e-5, f"L2 norm: expected 5.0, got {l2.item()}"

# p must actually change the result (this is the core of the bug: the old
# code always computed the same thing regardless of p).
assert abs(l1.item() - l2.item()) > 1e-3, (
    f"p=1 and p=2 norms must differ, got {l1.item()} and {l2.item()}"
)

# Omitting p must still default to L2, preserving old call sites.
assert abs(default.item() - l2.item()) < 1e-5, (
    f"default norm() must match p=2, got {default.item()} vs {l2.item()}"
)
"#;

        run_script(py, &locals, script).expect("p=1 vs p=2 norm script should succeed");
    });
}

#[test]
fn test_norm_dim_keepdim_true() {
    init_py();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals
            .set_item("Tensor", py.get_type::<PyTensor>())
            .expect("set Tensor");

        let script = r#"
# Row-major 2x3: [[3, 0, 0], [4, 0, 5]]
t = Tensor([3.0, 0.0, 0.0, 4.0, 0.0, 5.0], None, None, None).reshape([2, 3])

result = t.norm(dim=[0], keepdim=True)
assert result.shape == [1, 3], f"expected shape [1, 3], got {result.shape}"

# Column 0: sqrt(3^2+4^2)=5, column 1: sqrt(0^2+0^2)=0, column 2: sqrt(0^2+5^2)=5
expected = [[5.0, 0.0, 5.0]]
assert_close(result.tolist(), expected, tol=1e-5)
"#;

        run_script(py, &locals, script).expect("dim=[0], keepdim=True norm script should succeed");
    });
}

#[test]
fn test_norm_dim_keepdim_false() {
    init_py();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals
            .set_item("Tensor", py.get_type::<PyTensor>())
            .expect("set Tensor");

        let script = r#"
# Row-major 2x3: [[3, 0, 0], [4, 0, 5]]
t = Tensor([3.0, 0.0, 0.0, 4.0, 0.0, 5.0], None, None, None).reshape([2, 3])

result = t.norm(dim=[0], keepdim=False)
assert result.shape == [3], f"expected shape [3], got {result.shape}"

# Same per-column L2 values as the keepdim=True case, but dim 0 is fully
# removed from the shape rather than kept as size 1.
expected = [5.0, 0.0, 5.0]
assert_close(result.tolist(), expected, tol=1e-5)
"#;

        run_script(py, &locals, script).expect("dim=[0], keepdim=False norm script should succeed");
    });
}
