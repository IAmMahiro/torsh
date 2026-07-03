//! Shared test harness for embedded-Python integration tests.
//!
//! These tests construct pyo3 objects directly via `Python::run` with a
//! pre-populated locals dict (`py.get_type::<T>()` for every `#[pyclass]`
//! type they need), entirely bypassing `import rstorch_python`. No compiled
//! `.so` needs to be discoverable on `sys.path` for this to work: the
//! integration test binary links `torsh-python` as an ordinary Rust `rlib`
//! dependency, and `py.get_type::<T>()` reaches any `#[pyclass]` type's
//! Python type object directly (lazily created/cached by pyo3), regardless
//! of whether that type was ever registered into a `#[pymodule]`.

#![allow(dead_code)] // Not every helper is used by every test binary.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::ffi::CString;

/// Initialise the embedded Python interpreter. Idempotent -- safe to call
/// from every test (mirrors `tests/test_error.rs`'s `init_py`).
pub fn init_py() {
    Python::initialize();
}

/// Run a Python script with [`ASSERT_HELPERS`] prepended, against the given
/// `namespace` dict (which callers pre-populate with `py.get_type::<T>()`
/// entries for whichever `#[pyclass]` types the script needs). Any `assert`
/// failure inside `script` surfaces as an `Err(PyErr)` carrying the Python
/// traceback / assertion message.
///
/// `namespace` is passed as BOTH `globals` and `locals` to `Python::run`.
/// This matters: `exec(code, globals, locals)` with two *different* dicts
/// makes top-level `def`s store themselves in `locals` while their
/// `__globals__` stays pointing at `globals`, so a helper function calling
/// itself recursively (or a generator expression/comprehension referencing
/// an outer loop variable) raises `NameError` even though the name is
/// plainly "in scope" when read top to bottom. Using one shared dict makes
/// the exec behave like ordinary module-level code.
pub fn run_script(py: Python<'_>, namespace: &Bound<'_, PyDict>, script: &str) -> PyResult<()> {
    let full_source = format!("{ASSERT_HELPERS}\n{script}");
    let code = CString::new(full_source).expect("script must not contain NUL bytes");
    py.run(&code, Some(namespace), None)
}

/// Python helper functions available to every test's code string:
/// `assert_close(a, b, tol=1e-5)` recursively walks dicts/lists/tuples
/// (exactly the shape `state_dict()` produces) and compares leaf numbers
/// with a numeric tolerance, raising `AssertionError` with a path-annotated
/// message on the first mismatch. `assert` failures inside `py.run` show up
/// as a Rust-side `PyErr` (with the Python traceback embedded), so tests
/// simply `.expect(...)` the `py.run` call.
pub const ASSERT_HELPERS: &str = r#"
def assert_close(a, b, tol=1e-5, path="root"):
    if isinstance(a, dict) and isinstance(b, dict):
        assert set(a.keys()) == set(b.keys()), (
            f"key mismatch at {path}: {sorted(a.keys())} vs {sorted(b.keys())}"
        )
        for k in a:
            assert_close(a[k], b[k], tol, f"{path}.{k}")
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        assert len(a) == len(b), f"length mismatch at {path}: {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            assert_close(x, y, tol, f"{path}[{i}]")
    elif isinstance(a, bool) or isinstance(b, bool):
        assert a == b, f"bool mismatch at {path}: {a} vs {b}"
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        assert abs(a - b) <= tol, f"value mismatch at {path}: {a} vs {b}"
    else:
        assert a == b, f"value mismatch at {path}: {a!r} vs {b!r}"
"#;
