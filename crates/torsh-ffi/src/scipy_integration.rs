//! SciPy integration for ToRSh tensors
//!
//! This module provides comprehensive integration with SciPy, enabling seamless conversion
//! between ToRSh tensors and SciPy arrays, as well as access to SciPy's scientific computing
//! functionality including optimization, linear algebra, signal processing, and statistics.

use crate::numpy_compatibility::NumpyCompat;
use crate::tensor::PyTensor;
use numpy::PyArrayDyn;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyAny, PyDict, PyModule, PyTuple};
use pyo3::Bound;
use std::collections::HashMap;
use torsh_core::DType;

/// SciPy integration layer providing scientific computing capabilities
#[pyclass(name = "SciPyIntegration")]
#[derive(Debug)]
pub struct SciPyIntegration {
    /// NumPy compatibility layer for array operations
    numpy_compat: NumpyCompat,
    /// Cached SciPy module references
    scipy_modules: HashMap<String, Py<PyModule>>,
    /// Default tolerances for numerical operations
    default_tolerances: ScipyTolerances,
    /// Integration configuration
    config: ScipyConfig,
}

/// Configuration for SciPy integration
#[derive(Debug, Clone)]
pub struct ScipyConfig {
    /// Enable automatic gradient computation for optimization
    pub enable_autodiff: bool,
    /// Default solver method for optimization
    pub default_solver: String,
    /// Maximum iterations for iterative algorithms
    pub max_iterations: usize,
    /// Enable sparse matrix optimizations
    pub enable_sparse: bool,
    /// Memory limit for large array operations (in bytes)
    pub memory_limit: Option<usize>,
}

impl Default for ScipyConfig {
    fn default() -> Self {
        Self {
            enable_autodiff: true,
            default_solver: "BFGS".to_string(),
            max_iterations: 1000,
            enable_sparse: true,
            memory_limit: Some(2 * 1024 * 1024 * 1024), // 2GB default
        }
    }
}

/// Numerical tolerances for SciPy operations
#[derive(Debug, Clone)]
pub struct ScipyTolerances {
    /// Relative tolerance for optimization
    pub rtol: f64,
    /// Absolute tolerance for optimization
    pub atol: f64,
    /// Function tolerance for optimization
    pub ftol: f64,
    /// Gradient tolerance for optimization
    pub gtol: f64,
}

impl Default for ScipyTolerances {
    fn default() -> Self {
        Self {
            rtol: 1e-8,
            atol: 1e-12,
            ftol: 1e-9,
            gtol: 1e-5,
        }
    }
}

/// Result of optimization operations
#[pyclass(name = "OptimizationResult", from_py_object)]
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Final parameter values
    #[pyo3(get)]
    pub x: PyTensor,
    /// Final function value
    #[pyo3(get)]
    pub fun: f64,
    /// Number of iterations
    #[pyo3(get)]
    pub nit: usize,
    /// Number of function evaluations
    #[pyo3(get)]
    pub nfev: usize,
    /// Success flag
    #[pyo3(get)]
    pub success: bool,
    /// Status message
    #[pyo3(get)]
    pub message: String,
}

/// Result of linear algebra operations
#[pyclass(name = "LinalgResult", from_py_object)]
#[derive(Debug, Clone)]
pub struct LinalgResult {
    /// Primary result tensor
    #[pyo3(get)]
    pub result: PyTensor,
    /// Secondary result (e.g., singular values)
    #[pyo3(get)]
    pub secondary: Option<PyTensor>,
    /// Condition number (if applicable)
    #[pyo3(get)]
    pub condition_number: Option<f64>,
    /// Rank (if applicable)
    #[pyo3(get)]
    pub rank: Option<usize>,
}

/// Signal processing result
#[pyclass(name = "SignalResult", from_py_object)]
#[derive(Debug, Clone)]
pub struct SignalResult {
    /// Processed signal
    #[pyo3(get)]
    pub signal: PyTensor,
    /// Frequencies (for frequency domain operations)
    #[pyo3(get)]
    pub frequencies: Option<PyTensor>,
    /// Time values (for time domain operations)
    #[pyo3(get)]
    pub time: Option<PyTensor>,
    /// Metadata
    #[pyo3(get)]
    pub metadata: HashMap<String, f64>,
}

#[pymethods]
impl SciPyIntegration {
    /// Create a new SciPy integration instance
    #[new]
    pub fn new() -> PyResult<Self> {
        Ok(Self {
            numpy_compat: NumpyCompat::new(),
            scipy_modules: HashMap::new(),
            default_tolerances: ScipyTolerances::default(),
            config: ScipyConfig::default(),
        })
    }

    /// Configure SciPy integration settings
    pub fn configure(&mut self, _py: Python, config: &Bound<'_, PyDict>) -> PyResult<()> {
        if let Some(enable_autodiff) = config.get_item("enable_autodiff")? {
            self.config.enable_autodiff = enable_autodiff.extract()?;
        }
        if let Some(default_solver) = config.get_item("default_solver")? {
            self.config.default_solver = default_solver.extract()?;
        }
        if let Some(max_iterations) = config.get_item("max_iterations")? {
            self.config.max_iterations = max_iterations.extract()?;
        }
        if let Some(enable_sparse) = config.get_item("enable_sparse")? {
            self.config.enable_sparse = enable_sparse.extract()?;
        }
        if let Some(memory_limit) = config.get_item("memory_limit")? {
            self.config.memory_limit = Some(memory_limit.extract()?);
        }
        Ok(())
    }

    /// Convert ToRSh tensor to SciPy sparse matrix
    pub fn to_sparse_matrix(
        &self,
        py: Python,
        tensor: &PyTensor,
        format: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let scipy_sparse = self.get_scipy_module(py, "sparse")?;
        let numpy_array = self
            .numpy_compat
            .to_numpy_array(&tensor.data, &tensor.shape)
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(e))?;

        let format = format.unwrap_or("csr");
        let sparse_constructor = scipy_sparse.getattr(py, format)?;
        let sparse_matrix = sparse_constructor.call1(py, (numpy_array,))?;

        Ok(sparse_matrix)
    }

    /// Convert SciPy sparse matrix to ToRSh tensor
    pub fn from_sparse_matrix(
        &self,
        _py: Python,
        sparse_matrix: Bound<'_, PyAny>,
    ) -> PyResult<PyTensor> {
        // Convert sparse matrix to dense array first
        let dense_array = sparse_matrix.call_method("toarray", (), None)?;
        let shape = self.numpy_shape(&dense_array)?;
        let data = self.extract_real_f32_array(&dense_array, "sparse matrix")?;

        Ok(PyTensor::from_raw(data, shape, DType::F32, false))
    }

    /// Solve linear system using SciPy
    pub fn solve_linear_system(
        &self,
        py: Python,
        a: &PyTensor,
        b: &PyTensor,
        method: Option<&str>,
    ) -> PyResult<LinalgResult> {
        let scipy_linalg = self.get_scipy_module(py, "linalg")?;
        let a_np = self
            .numpy_compat
            .to_numpy_array(&a.data, &a.shape)
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(e))?;
        let b_np = self
            .numpy_compat
            .to_numpy_array(&b.data, &b.shape)
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(e))?;

        let solve_method = method.unwrap_or("solve");
        let x_obj: Py<PyAny> = match solve_method {
            "solve" => scipy_linalg.call_method1(py, "solve", (a_np, b_np))?,
            "lstsq" => {
                // lstsq returns (x, residues, rank, singular_values); only
                // the solution vector/matrix `x` is needed here. Note its
                // shape is *not* generally `b.shape` when `a` is
                // rectangular, so the shape is read back from the result
                // itself via `numpy_shape` below rather than assumed.
                let result = scipy_linalg.call_method1(py, "lstsq", (a_np, b_np))?;
                let result = result.bind(py);
                let result_tuple = result.cast::<PyTuple>()?;
                result_tuple.get_item(0)?.unbind()
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown solve method: {}",
                    solve_method
                )))
            }
        };

        let x_bound = x_obj.bind(py);
        let x_shape = self.numpy_shape(x_bound)?;
        let x_data = self.extract_real_f32_array(x_bound, "solve result")?;

        Ok(LinalgResult {
            result: PyTensor::from_raw(x_data, x_shape, DType::F32, false),
            secondary: None,
            condition_number: None,
            rank: None,
        })
    }

    /// Compute eigenvalues and eigenvectors
    ///
    /// Design choice: `scipy.linalg.eig`/`eigvals` always return
    /// `complex128` arrays, even for inputs whose spectrum is mathematically
    /// real (e.g. symmetric matrices) -- unlike `numpy.linalg.eig`, SciPy
    /// does not special-case this. Since [`PyTensor`] only stores real
    /// `f32` data, [`Self::extract_real_f32_array`] is used to drop a
    /// negligible imaginary part (within a small tolerance) or return a
    /// clear error when the imaginary component is too large to discard.
    pub fn eigendecomposition(
        &self,
        py: Python,
        tensor: &PyTensor,
        compute_eigenvectors: bool,
    ) -> PyResult<LinalgResult> {
        if tensor.shape.len() != 2 || tensor.shape[0] != tensor.shape[1] {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "eigendecomposition requires a square 2D matrix, got shape {:?}",
                tensor.shape
            )));
        }
        let n = tensor.shape[0];

        let scipy_linalg = self.get_scipy_module(py, "linalg")?;
        let numpy_array = self
            .numpy_compat
            .to_numpy_array(&tensor.data, &tensor.shape)
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(e))?;

        let (eigenvalues, eigenvectors) = if compute_eigenvectors {
            let result = scipy_linalg.call_method1(py, "eig", (numpy_array,))?;
            let result = result.bind(py);
            let result_tuple = result.cast::<PyTuple>()?;
            let eigenvals_obj = result_tuple.get_item(0)?;
            let eigenvecs_obj = result_tuple.get_item(1)?;
            (
                self.extract_real_f32_array(&eigenvals_obj, "eigenvalues")?,
                Some(self.extract_real_f32_array(&eigenvecs_obj, "eigenvectors")?),
            )
        } else {
            let eigenvals_obj = scipy_linalg.call_method1(py, "eigvals", (numpy_array,))?;
            let eigenvals_obj = eigenvals_obj.bind(py);
            (
                self.extract_real_f32_array(eigenvals_obj, "eigenvalues")?,
                None,
            )
        };

        Ok(LinalgResult {
            result: PyTensor::from_raw(eigenvalues, vec![n], DType::F32, false),
            secondary: eigenvectors
                .map(|data| PyTensor::from_raw(data, vec![n, n], DType::F32, false)),
            condition_number: None,
            rank: None,
        })
    }

    /// Singular Value Decomposition
    pub fn svd(
        &self,
        py: Python,
        tensor: &PyTensor,
        full_matrices: bool,
    ) -> PyResult<(PyTensor, PyTensor, PyTensor)> {
        let scipy_linalg = self.get_scipy_module(py, "linalg")?;
        let numpy_array = self
            .numpy_compat
            .to_numpy_array(&tensor.data, &tensor.shape)
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(e))?;

        let kwargs = [("full_matrices", full_matrices)].into_py_dict(py)?;
        let result = scipy_linalg.call_method(py, "svd", (numpy_array,), Some(&kwargs))?;
        let result = result.bind(py);
        let result_tuple = result.cast::<PyTuple>()?;

        let u_obj = result_tuple.get_item(0)?;
        let s_obj = result_tuple.get_item(1)?;
        let vt_obj = result_tuple.get_item(2)?;

        let u_shape = self.numpy_shape(&u_obj)?;
        let s_shape = self.numpy_shape(&s_obj)?;
        let vt_shape = self.numpy_shape(&vt_obj)?;

        let u = self.extract_real_f32_array(&u_obj, "SVD U")?;
        let s = self.extract_real_f32_array(&s_obj, "SVD singular values")?;
        let vt = self.extract_real_f32_array(&vt_obj, "SVD Vt")?;

        Ok((
            PyTensor::from_raw(u, u_shape, DType::F32, false),
            PyTensor::from_raw(s, s_shape, DType::F32, false),
            PyTensor::from_raw(vt, vt_shape, DType::F32, false),
        ))
    }

    /// Optimize function using SciPy optimizers
    pub fn minimize(
        &self,
        py: Python,
        objective: Bound<'_, PyAny>,
        initial_guess: &PyTensor,
        method: Option<&str>,
        bounds: Option<Bound<'_, PyAny>>,
        constraints: Option<Bound<'_, PyAny>>,
    ) -> PyResult<OptimizationResult> {
        let scipy_optimize = self.get_scipy_module(py, "optimize")?;
        let x0 = self
            .numpy_compat
            .to_numpy_array(&initial_guess.data, &initial_guess.shape)
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(e))?;
        // SciPy's gradient-based methods (the default here is BFGS) estimate
        // the gradient via finite differences with a step size calibrated
        // for float64 precision (~1.49e-8). With a float32 `x0`, that step
        // is below float32's representable resolution near typical values,
        // so `f(x + eps) == f(x)` after rounding and the estimated gradient
        // comes out as exactly zero -- the optimizer then reports immediate
        // "success" without ever moving from the initial guess. Upcasting
        // to float64 for the call avoids this; the result is downcast back
        // to float32 by `extract_real_f32_array` below as usual.
        let x0 = x0.bind(py).call_method1("astype", ("float64",))?;

        let method = method.unwrap_or(&self.config.default_solver);
        let kwargs = PyDict::new(py);
        kwargs.set_item("method", method)?;

        let options = PyDict::new(py);
        options.set_item("maxiter", self.config.max_iterations)?;
        // `gtol` (gradient-norm convergence tolerance) is broadly supported
        // by the gradient-based methods this crate defaults to (BFGS, CG,
        // trust-constr). `ftol` -- what an earlier version of this option
        // set used -- is *not* one of BFGS's recognized options and
        // triggers a spurious `OptimizeWarning: Unknown solver options`
        // there.
        options.set_item("gtol", self.default_tolerances.gtol)?;
        kwargs.set_item("options", options)?;

        if let Some(bounds) = bounds {
            kwargs.set_item("bounds", bounds)?;
        }
        if let Some(constraints) = constraints {
            kwargs.set_item("constraints", constraints)?;
        }

        let result = scipy_optimize.call_method(py, "minimize", (objective, x0), Some(&kwargs))?;
        let result = result.bind(py);

        // Extract results. Using the `Bound` form uniformly (rather than
        // `Py::getattr(py, ...)`) means every field access below carries the
        // GIL token implicitly, so none of them can accidentally omit it.
        let x_obj = result.getattr("x")?;
        let x_shape = self.numpy_shape(&x_obj)?;
        let x_data = self.extract_real_f32_array(&x_obj, "optimization result")?;
        let fun: f64 = result.getattr("fun")?.extract()?;
        let nit: usize = result.getattr("nit")?.extract()?;
        let nfev: usize = result.getattr("nfev")?.extract()?;
        let success: bool = result.getattr("success")?.extract()?;
        let message: String = result.getattr("message")?.extract()?;

        Ok(OptimizationResult {
            x: PyTensor::from_raw(x_data, x_shape, DType::F32, false),
            fun,
            nit,
            nfev,
            success,
            message,
        })
    }

    /// Apply digital filter to signal
    ///
    /// Note: `cutoff` is a single scalar, which fully parameterizes
    /// `"lowpass"`/`"highpass"` filters. `"bandpass"`/`"bandstop"` normally
    /// need a `(low, high)` pair for SciPy's `Wn` argument; passing a single
    /// value for those two filter types will raise a clear error from
    /// `scipy.signal.butter` itself rather than silently doing the wrong
    /// thing.
    pub fn filter_signal(
        &self,
        py: Python,
        signal: &PyTensor,
        filter_type: &str,
        cutoff: f64,
        sample_rate: f64,
        order: Option<usize>,
    ) -> PyResult<SignalResult> {
        let scipy_signal = self.get_scipy_module(py, "signal")?;
        let signal_np = self
            .numpy_compat
            .to_numpy_array(&signal.data, &signal.shape)
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(e))?;

        let order = order.unwrap_or(4);
        let nyquist = sample_rate / 2.0;
        let normalized_cutoff = cutoff / nyquist;

        // Design filter
        let filter_result = match filter_type {
            "lowpass" | "highpass" | "bandpass" | "bandstop" => scipy_signal.call_method1(
                py,
                "butter",
                (order, normalized_cutoff, filter_type),
            )?,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown filter type: {}",
                    filter_type
                )))
            }
        };

        let filter_result = filter_result.bind(py);
        let filter_tuple = filter_result.cast::<PyTuple>()?;
        let b = filter_tuple.get_item(0)?;
        let a = filter_tuple.get_item(1)?;

        // Apply filter
        let filtered = scipy_signal.call_method1(py, "filtfilt", (b, a, signal_np))?;
        let filtered = filtered.bind(py);
        let filtered_data = self.extract_real_f32_array(filtered, "filtered signal")?;

        Ok(SignalResult {
            signal: PyTensor::from_raw(filtered_data, signal.shape.clone(), DType::F32, false),
            frequencies: None,
            time: None,
            metadata: [
                ("cutoff".to_string(), cutoff),
                ("sample_rate".to_string(), sample_rate),
            ]
            .into(),
        })
    }

    /// Compute Fast Fourier Transform
    ///
    /// Design choice: `scipy.fft.fft` always returns a complex-valued array
    /// (both magnitude and phase), but [`PyTensor`] only stores real `f32`
    /// data. Rather than erroring unconditionally or silently truncating to
    /// the real component (which is not physically meaningful for a
    /// spectrum), this returns the magnitude spectrum `|fft(x)|`: the most
    /// common real-valued representation callers want (e.g. for spectrum
    /// plots), and well-defined for every input.
    pub fn fft(&self, py: Python, signal: &PyTensor, axis: Option<i32>) -> PyResult<PyTensor> {
        let scipy_fft = self.get_scipy_module(py, "fft")?;
        let signal_np = self
            .numpy_compat
            .to_numpy_array(&signal.data, &signal.shape)
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(e))?;

        let result = if let Some(axis) = axis {
            let kwargs = [("axis", axis)].into_py_dict(py)?;
            scipy_fft.call_method(py, "fft", (signal_np,), Some(&kwargs))?
        } else {
            scipy_fft.call_method1(py, "fft", (signal_np,))?
        };
        let result = result.bind(py);

        // `fft` preserves the input's shape (no `n` override is exposed
        // here), and `abs()` of a complex array is always real, so this
        // routes through the non-complex fast path of `extract_real_f32_array`.
        let magnitude = result.call_method("__abs__", (), None)?;
        let data = self.extract_real_f32_array(&magnitude, "FFT magnitude")?;

        Ok(PyTensor::from_raw(
            data,
            signal.shape.clone(),
            DType::F32,
            false,
        ))
    }

    /// Compute statistical tests
    pub fn statistical_test(
        &self,
        py: Python,
        data1: &PyTensor,
        data2: Option<&PyTensor>,
        test_type: &str,
    ) -> PyResult<(f64, f64)> {
        let scipy_stats = self.get_scipy_module(py, "stats")?;
        let data1_np = self
            .numpy_compat
            .to_numpy_array(&data1.data, &data1.shape)
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(e))?;

        let result = match test_type {
            "ttest_1samp" => {
                if let Some(data2) = data2 {
                    let popmean = self
                        .numpy_compat
                        .to_numpy_array(&data2.data, &data2.shape)
                        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e))?;
                    scipy_stats.call_method1(py, "ttest_1samp", (data1_np, popmean))?
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "ttest_1samp requires population mean",
                    ));
                }
            }
            "ttest_ind" => {
                if let Some(data2) = data2 {
                    let data2_np = self
                        .numpy_compat
                        .to_numpy_array(&data2.data, &data2.shape)
                        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e))?;
                    scipy_stats.call_method1(py, "ttest_ind", (data1_np, data2_np))?
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "ttest_ind requires two samples",
                    ));
                }
            }
            "normaltest" => scipy_stats.call_method1(py, "normaltest", (data1_np,))?,
            "kstest" => scipy_stats.call_method1(py, "kstest", (data1_np, "norm"))?,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown test type: {}",
                    test_type
                )))
            }
        };

        let result = result.bind(py);
        let result_tuple = result.cast::<PyTuple>()?;
        let statistic: f64 = result_tuple.get_item(0)?.extract()?;
        let p_value: f64 = result_tuple.get_item(1)?.extract()?;

        Ok((statistic, p_value))
    }

    /// Interpolate data points
    pub fn interpolate(
        &self,
        py: Python,
        x: &PyTensor,
        y: &PyTensor,
        x_new: &PyTensor,
        method: Option<&str>,
    ) -> PyResult<PyTensor> {
        let scipy_interpolate = self.get_scipy_module(py, "interpolate")?;
        let x_np = self
            .numpy_compat
            .to_numpy_array(&x.data, &x.shape)
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(e))?;
        let y_np = self
            .numpy_compat
            .to_numpy_array(&y.data, &y.shape)
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(e))?;
        let x_new_np = self
            .numpy_compat
            .to_numpy_array(&x_new.data, &x_new.shape)
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(e))?;

        let method = method.unwrap_or("linear");
        // The original sketch never actually passed `method` through to
        // `interp1d` (it always used SciPy's own "linear" default); wire it
        // up via the `kind` kwarg so the parameter has an effect.
        let kwargs = [("kind", method)].into_py_dict(py)?;
        let interpolator =
            scipy_interpolate.call_method(py, "interp1d", (x_np, y_np), Some(&kwargs))?;
        let y_new = interpolator.call1(py, (x_new_np,))?;
        let y_new = y_new.bind(py);

        let data = self.extract_real_f32_array(y_new, "interpolation result")?;
        Ok(PyTensor::from_raw(
            data,
            x_new.shape.clone(),
            DType::F32,
            false,
        ))
    }

    /// Get available SciPy modules
    pub fn get_available_modules(&self, py: Python) -> PyResult<Vec<String>> {
        let modules = vec![
            "cluster",
            "constants",
            "fft",
            "integrate",
            "interpolate",
            "io",
            "linalg",
            "ndimage",
            "optimize",
            "signal",
            "sparse",
            "spatial",
            "special",
            "stats",
        ];

        let mut available = Vec::new();
        for module in modules {
            if self.get_scipy_module(py, module).is_ok() {
                available.push(module.to_string());
            }
        }

        Ok(available)
    }

    /// Get SciPy version information
    pub fn get_scipy_version(&self, py: Python) -> PyResult<String> {
        let scipy = py.import("scipy")?;
        let version: String = scipy.getattr("__version__")?.extract()?;
        Ok(version)
    }

    /// Benchmark SciPy operations performance
    ///
    /// `tensor_size` must be a square 2D shape (e.g. `[n, n]`) since the
    /// eigendecomposition benchmark below requires a square matrix; this
    /// mirrors the shape that `eigendecomposition`/`minimize`/`fft`
    /// (benchmarked below) already require or accept.
    pub fn benchmark_operations(
        &self,
        py: Python,
        tensor_size: Vec<usize>,
        num_iterations: usize,
    ) -> PyResult<HashMap<String, f64>> {
        use std::time::Instant;

        if num_iterations == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "num_iterations must be greater than zero",
            ));
        }

        let mut results = HashMap::new();

        // Create test tensor. The original sketch called `PyTensor::zeros`
        // as a free function, but `zeros` is a `#[classmethod]` requiring a
        // `&Bound<PyType>` `cls` token, so it cannot be called this way;
        // build the zero-filled tensor directly instead.
        let total_elements: usize = tensor_size.iter().product();
        let test_tensor =
            PyTensor::from_raw(vec![0.0f32; total_elements], tensor_size, DType::F32, false);

        // Benchmark matrix operations
        let start = Instant::now();
        for _ in 0..num_iterations {
            let _ = self.eigendecomposition(py, &test_tensor, false)?;
        }
        results.insert(
            "eigenvalues".to_string(),
            start.elapsed().as_secs_f64() / num_iterations as f64,
        );

        // Benchmark optimization. `Python::eval` requires a `&CStr` in
        // pyo3 0.29, not a plain `&str`. Unlike `eigendecomposition` (which
        // needs the square 2D `test_tensor`), `scipy.optimize.minimize`
        // rejects any `x0` that isn't 1-D (`ValueError: 'x0' must only have
        // one dimension.`), so a separate flat tensor with the same element
        // count is used here instead of reusing `test_tensor` directly.
        let flat_tensor = PyTensor::from_raw(
            vec![0.0f32; total_elements],
            vec![total_elements],
            DType::F32,
            false,
        );
        let objective = py.eval(c"lambda x: sum(x**2)", None, None)?;
        let start = Instant::now();
        for _ in 0..num_iterations {
            let _ = self.minimize(py, objective.clone(), &flat_tensor, None, None, None)?;
        }
        results.insert(
            "optimization".to_string(),
            start.elapsed().as_secs_f64() / num_iterations as f64,
        );

        // Benchmark FFT
        let start = Instant::now();
        for _ in 0..num_iterations {
            let _ = self.fft(py, &test_tensor, None)?;
        }
        results.insert(
            "fft".to_string(),
            start.elapsed().as_secs_f64() / num_iterations as f64,
        );

        Ok(results)
    }
}

impl SciPyIntegration {
    /// Get or cache SciPy module
    fn get_scipy_module(&self, py: Python, module_name: &str) -> PyResult<Py<PyModule>> {
        if let Some(module) = self.scipy_modules.get(module_name) {
            return Ok(module.clone_ref(py));
        }

        let module_path = format!("scipy.{}", module_name);
        let module = py.import(&module_path)?;
        Ok(module.into())
    }

    /// Read a NumPy array's `.shape` attribute into a `Vec<usize>`.
    ///
    /// Used whenever a SciPy call's result shape cannot simply be assumed
    /// to match one of the input tensors (e.g. `lstsq` on a rectangular
    /// system, or `svd`'s three differently-shaped outputs).
    fn numpy_shape(&self, array: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
        let shape_py = array.getattr("shape")?;
        let shape_tuple = shape_py.cast::<PyTuple>()?;
        shape_tuple
            .iter()
            .map(|item| item.extract::<usize>())
            .collect()
    }

    /// Extract a real-valued `Vec<f32>` from a NumPy array that may be
    /// complex-typed, normalizing dtype along the way.
    ///
    /// Two SciPy quirks motivate this single choke point for every
    /// `from_numpy_array` call site in this file:
    ///
    /// 1. SciPy numeric routines mostly return `float64`, so an
    ///    `.astype("float32")` normalization is applied unconditionally
    ///    before casting to `PyArrayDyn<f32>`.
    /// 2. Some linear-algebra routines (`eig`, `eigvals`, ...) return
    ///    `complex128` arrays even when the mathematical result is real
    ///    (e.g. symmetric matrices with a real spectrum) -- unlike
    ///    `numpy.linalg.eig`, `scipy.linalg.eig` does not special-case this.
    ///    Since [`PyTensor`] only stores real `f32` data, this checks
    ///    whether the imaginary component is within `COMPLEX_TOLERANCE` of
    ///    zero and, if so, drops it; otherwise it surfaces a clear error
    ///    rather than silently discarding a meaningful imaginary part.
    fn extract_real_f32_array(&self, array: &Bound<'_, PyAny>, label: &str) -> PyResult<Vec<f32>> {
        const COMPLEX_TOLERANCE: f64 = 1e-6;

        let dtype_name: String = array.getattr("dtype")?.str()?.extract()?;
        let real_array = if dtype_name.starts_with("complex") {
            let imag = array.getattr("imag")?;
            let max_abs_imag: f64 = imag
                .call_method("__abs__", (), None)?
                .call_method("max", (), None)?
                .extract()?;
            if max_abs_imag > COMPLEX_TOLERANCE {
                return Err(PyErr::new::<PyRuntimeError, _>(format!(
                    "SciPy {} contain non-negligible imaginary components (max |Im| = {:.3e}); \
                     PyTensor only supports real f32 data",
                    label, max_abs_imag
                )));
            }
            array.getattr("real")?
        } else {
            array.clone()
        };

        let float32_array = real_array.call_method1("astype", ("float32",))?;
        let py_array = float32_array
            .cast::<PyArrayDyn<f32>>()
            .map_err(|e| {
                PyErr::new::<PyRuntimeError, _>(format!(
                    "Failed to interpret {} as a NumPy array: {}",
                    label, e
                ))
            })?;

        self.numpy_compat
            .from_numpy_array(py_array)
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(e))
    }
}

/// Create SciPy integration utilities
pub fn create_scipy_utilities(py: Python) -> PyResult<Bound<PyDict>> {
    let utils = PyDict::new(py);

    // Add utility functions
    utils.set_item("create_integration", py.get_type::<SciPyIntegration>())?;
    utils.set_item("OptimizationResult", py.get_type::<OptimizationResult>())?;
    utils.set_item("LinalgResult", py.get_type::<LinalgResult>())?;
    utils.set_item("SignalResult", py.get_type::<SignalResult>())?;

    Ok(utils)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scipy_config() {
        let config = ScipyConfig::default();
        assert_eq!(config.default_solver, "BFGS");
        assert_eq!(config.max_iterations, 1000);
        assert!(config.enable_autodiff);
    }

    #[test]
    fn test_tolerances() {
        let tol = ScipyTolerances::default();
        assert_eq!(tol.rtol, 1e-8);
        assert_eq!(tol.atol, 1e-12);
    }

    /// Skip a scipy-dependent test cleanly if scipy isn't importable in this
    /// environment, rather than failing the whole suite.
    fn require_scipy(py: Python<'_>) -> bool {
        py.import("scipy").is_ok()
    }

    /// Compare two `f32` slices elementwise within a small tolerance, rather
    /// than via `assert_eq!` -- avoids relying on bit-exact float equality
    /// after a round trip through Python/SciPy/NumPy.
    fn assert_f32_slice_approx_eq(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-4, "{:?} vs {:?}", actual, expected);
        }
    }

    #[test]
    fn test_fft_magnitude_of_impulse() {
        Python::initialize();
        Python::attach(|py| {
            if !require_scipy(py) {
                eprintln!("skipping test_fft_magnitude_of_impulse: scipy not installed");
                return;
            }
            let integration = SciPyIntegration::new().expect("SciPyIntegration::new");
            // The DFT of a unit impulse [1, 0, 0, 0] is exactly [1, 1, 1, 1]
            // (magnitude 1 at every bin), giving an exact assertion with no
            // floating-point tolerance games.
            let signal = PyTensor::from_raw(vec![1.0, 0.0, 0.0, 0.0], vec![4], DType::F32, false);

            let result = integration
                .fft(py, &signal, None)
                .expect("fft should succeed on a real, non-empty signal");

            assert_eq!(result.shape, vec![4]);
            for magnitude in &result.data {
                assert!((magnitude - 1.0).abs() < 1e-5, "got {magnitude}");
            }
        });
    }

    #[test]
    fn test_solve_linear_system_identity() {
        Python::initialize();
        Python::attach(|py| {
            if !require_scipy(py) {
                eprintln!("skipping test_solve_linear_system_identity: scipy not installed");
                return;
            }
            let integration = SciPyIntegration::new().expect("SciPyIntegration::new");
            let a = PyTensor::from_raw(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], DType::F32, false);
            let b = PyTensor::from_raw(vec![3.0, 4.0], vec![2], DType::F32, false);

            let result = integration
                .solve_linear_system(py, &a, &b, Some("solve"))
                .expect("solving I*x = b should succeed");

            assert_eq!(result.result.shape, vec![2]);
            assert!((result.result.data[0] - 3.0).abs() < 1e-4);
            assert!((result.result.data[1] - 4.0).abs() < 1e-4);
        });
    }

    #[test]
    fn test_eigendecomposition_symmetric() {
        Python::initialize();
        Python::attach(|py| {
            if !require_scipy(py) {
                eprintln!("skipping test_eigendecomposition_symmetric: scipy not installed");
                return;
            }
            let integration = SciPyIntegration::new().expect("SciPyIntegration::new");
            // diag(2, 3) is symmetric with an exactly real spectrum {2, 3},
            // exercising the "drop negligible imaginary part" path.
            let tensor =
                PyTensor::from_raw(vec![2.0, 0.0, 0.0, 3.0], vec![2, 2], DType::F32, false);

            let result = integration
                .eigendecomposition(py, &tensor, false)
                .expect("eigendecomposition of a real-spectrum matrix should succeed");

            let mut eigenvalues = result.result.data.clone();
            eigenvalues.sort_by(|a, b| a.partial_cmp(b).expect("no NaNs expected"));
            assert_eq!(eigenvalues.len(), 2);
            assert!((eigenvalues[0] - 2.0).abs() < 1e-4);
            assert!((eigenvalues[1] - 3.0).abs() < 1e-4);
            assert!(result.secondary.is_none());
        });
    }

    #[test]
    fn test_eigendecomposition_rejects_non_square() {
        Python::initialize();
        Python::attach(|py| {
            if !require_scipy(py) {
                eprintln!("skipping test_eigendecomposition_rejects_non_square: scipy not installed");
                return;
            }
            let integration = SciPyIntegration::new().expect("SciPyIntegration::new");
            let tensor = PyTensor::from_raw(vec![1.0, 2.0, 3.0], vec![1, 3], DType::F32, false);

            let result = integration.eigendecomposition(py, &tensor, false);
            assert!(result.is_err(), "non-square input must be rejected");
        });
    }

    #[test]
    fn test_svd_diagonal() {
        Python::initialize();
        Python::attach(|py| {
            if !require_scipy(py) {
                eprintln!("skipping test_svd_diagonal: scipy not installed");
                return;
            }
            let integration = SciPyIntegration::new().expect("SciPyIntegration::new");
            let tensor =
                PyTensor::from_raw(vec![3.0, 0.0, 0.0, 1.0], vec![2, 2], DType::F32, false);

            let (u, s, vt) = integration
                .svd(py, &tensor, true)
                .expect("SVD of a real, non-empty matrix should succeed");

            assert_eq!(u.shape, vec![2, 2]);
            assert_eq!(s.shape, vec![2]);
            assert_eq!(vt.shape, vec![2, 2]);
            // Singular values are uniquely determined (descending order is
            // LAPACK's convention); U/Vt signs are not, so only `s` is
            // checked exactly here.
            assert!((s.data[0] - 3.0).abs() < 1e-4);
            assert!((s.data[1] - 1.0).abs() < 1e-4);
        });
    }

    #[test]
    fn test_minimize_quadratic() {
        Python::initialize();
        Python::attach(|py| {
            if !require_scipy(py) {
                eprintln!("skipping test_minimize_quadratic: scipy not installed");
                return;
            }
            let integration = SciPyIntegration::new().expect("SciPyIntegration::new");
            let objective = py
                .eval(c"lambda x: (x[0] - 3.0) ** 2", None, None)
                .expect("failed to build objective");
            let x0 = PyTensor::from_raw(vec![0.0], vec![1], DType::F32, false);

            let result = integration
                .minimize(py, objective, &x0, None, None, None)
                .expect("minimize should succeed on a simple convex quadratic");

            assert!(result.success, "message: {}", result.message);
            assert_eq!(result.x.shape, vec![1]);
            assert!(
                (result.x.data[0] - 3.0).abs() < 1e-2,
                "expected x ~= 3.0, got {}",
                result.x.data[0]
            );
        });
    }

    #[test]
    fn test_interpolate_linear() {
        Python::initialize();
        Python::attach(|py| {
            if !require_scipy(py) {
                eprintln!("skipping test_interpolate_linear: scipy not installed");
                return;
            }
            let integration = SciPyIntegration::new().expect("SciPyIntegration::new");
            let x = PyTensor::from_raw(vec![0.0, 1.0, 2.0], vec![3], DType::F32, false);
            let y = PyTensor::from_raw(vec![0.0, 1.0, 4.0], vec![3], DType::F32, false);
            let x_new = PyTensor::from_raw(vec![0.5, 1.5], vec![2], DType::F32, false);

            let result = integration
                .interpolate(py, &x, &y, &x_new, Some("linear"))
                .expect("linear interpolation should succeed");

            assert_eq!(result.shape, vec![2]);
            assert!((result.data[0] - 0.5).abs() < 1e-4);
            assert!((result.data[1] - 2.5).abs() < 1e-4);
        });
    }

    #[test]
    fn test_statistical_test_normaltest() {
        Python::initialize();
        Python::attach(|py| {
            if !require_scipy(py) {
                eprintln!("skipping test_statistical_test_normaltest: scipy not installed");
                return;
            }
            let integration = SciPyIntegration::new().expect("SciPyIntegration::new");
            let data: Vec<f32> = (0..50).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
            let tensor = PyTensor::from_raw(data.clone(), vec![data.len()], DType::F32, false);

            let (statistic, p_value) = integration
                .statistical_test(py, &tensor, None, "normaltest")
                .expect("normaltest should succeed on a real, non-empty sample");

            assert!(statistic.is_finite());
            assert!((0.0..=1.0).contains(&p_value));
        });
    }

    #[test]
    fn test_filter_signal_lowpass_preserves_shape() {
        Python::initialize();
        Python::attach(|py| {
            if !require_scipy(py) {
                eprintln!("skipping test_filter_signal_lowpass_preserves_shape: scipy not installed");
                return;
            }
            let integration = SciPyIntegration::new().expect("SciPyIntegration::new");
            let data: Vec<f32> = (0..64)
                .map(|i| (i as f32 * 0.2).sin() + (i as f32 * 2.5).sin() * 0.3)
                .collect();
            let signal = PyTensor::from_raw(data.clone(), vec![data.len()], DType::F32, false);

            let result = integration
                .filter_signal(py, &signal, "lowpass", 5.0, 50.0, Some(4))
                .expect("lowpass filtering should succeed");

            assert_eq!(result.signal.shape, signal.shape);
            assert!(result.signal.data.iter().all(|v| v.is_finite()));
        });
    }

    #[test]
    fn test_from_sparse_matrix() {
        Python::initialize();
        Python::attach(|py| {
            if !require_scipy(py) {
                eprintln!("skipping test_from_sparse_matrix: scipy not installed");
                return;
            }
            let integration = SciPyIntegration::new().expect("SciPyIntegration::new");
            let scipy_sparse = py
                .import("scipy.sparse")
                .expect("scipy.sparse should be importable");
            let dense_nested: Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![0.0, 2.0]];
            let sparse = scipy_sparse
                .call_method1("csr_matrix", (dense_nested,))
                .expect("failed to build test sparse matrix");

            let tensor = integration
                .from_sparse_matrix(py, sparse)
                .expect("from_sparse_matrix should succeed on a real, non-empty matrix");

            assert_eq!(tensor.shape, vec![2, 2]);
            assert_f32_slice_approx_eq(&tensor.data, &[1.0, 0.0, 0.0, 2.0]);
        });
    }

    #[test]
    fn test_benchmark_operations_smoke() {
        Python::initialize();
        Python::attach(|py| {
            if !require_scipy(py) {
                eprintln!("skipping test_benchmark_operations_smoke: scipy not installed");
                return;
            }
            let integration = SciPyIntegration::new().expect("SciPyIntegration::new");

            let results = integration
                .benchmark_operations(py, vec![2, 2], 1)
                .expect("benchmark_operations should succeed end-to-end");

            for key in ["eigenvalues", "optimization", "fft"] {
                let elapsed = results.get(key).unwrap_or_else(|| panic!("missing '{key}' result"));
                assert!(*elapsed >= 0.0);
            }
        });
    }
}
