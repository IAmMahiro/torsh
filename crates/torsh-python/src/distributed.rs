//! Distributed training bindings

use crate::{error::PyResult, tensor::PyTensor};
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyModuleMethods, PyTuple};

/// Process group for distributed training
#[pyclass(name = "ProcessGroup")]
pub struct PyProcessGroup {
    rank: u32,
    world_size: u32,
}

#[pymethods]
impl PyProcessGroup {
    #[new]
    fn new(rank: u32, world_size: u32) -> Self {
        Self { rank, world_size }
    }

    #[getter]
    fn rank(&self) -> u32 {
        self.rank
    }

    #[getter]
    fn world_size(&self) -> u32 {
        self.world_size
    }

    #[pyo3(signature = (_tensor, _op=None))]
    fn all_reduce(&self, _tensor: &PyTensor, _op: Option<String>) -> PyResult<()> {
        Err(not_implemented("ProcessGroup.all_reduce"))
    }

    fn all_gather(&self, _tensors: Vec<PyTensor>, _tensor: &PyTensor) -> PyResult<()> {
        Err(not_implemented("ProcessGroup.all_gather"))
    }

    fn broadcast(&self, _tensor: &PyTensor, _src: u32) -> PyResult<()> {
        Err(not_implemented("ProcessGroup.broadcast"))
    }

    fn barrier(&self) -> PyResult<()> {
        Err(not_implemented("ProcessGroup.barrier"))
    }
}

/// Build an honest `NotImplementedError` for a collective that is not yet wired
/// to a real multi-process backend.
///
/// The previous implementations returned `Ok(())` while ignoring every
/// argument, so a data-parallel run silently trained with unsynchronised
/// gradients. Raising here makes the gap loud instead of corrupting results.
/// A real backend (TCP transport + initialised process group) is future work;
/// `torsh-distributed` currently only ships a mock backend.
fn not_implemented(op: &str) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(format!(
        "rstorch.distributed.{op} is not implemented: the Python bindings are \
         not yet wired to a real distributed backend. Track this in the ToRSh \
         distributed roadmap.",
    ))
}

/// Distributed Data Parallel wrapper
#[pyclass(name = "DistributedDataParallel")]
pub struct PyDDP {
    module: Py<PyAny>,
    process_group: Option<Py<PyProcessGroup>>,
}

#[pymethods]
impl PyDDP {
    #[new]
    #[pyo3(signature = (module, _device_ids=None, _output_device=None, _broadcast_buffers=None, process_group=None, _bucket_cap_mb=None, _find_unused_parameters=None, _check_reduction=None, _gradient_as_bucket_view=None))]
    fn new(
        module: Py<PyAny>,
        _device_ids: Option<Vec<u32>>,
        _output_device: Option<u32>,
        _broadcast_buffers: Option<bool>,
        process_group: Option<Py<PyProcessGroup>>,
        _bucket_cap_mb: Option<f32>,
        _find_unused_parameters: Option<bool>,
        _check_reduction: Option<bool>,
        _gradient_as_bucket_view: Option<bool>,
    ) -> Self {
        Self {
            module,
            process_group,
        }
    }

    fn forward(&self, py: Python<'_>, inputs: Vec<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        let forward_method = self.module.getattr(py, "forward")?;
        let tuple = PyTuple::new(py, inputs)?;
        forward_method.call1(py, tuple)
    }

    fn __call__(&self, py: Python<'_>, inputs: Vec<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        self.forward(py, inputs)
    }

    fn parameters(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let method = self.module.getattr(py, "parameters")?;
        method.call0(py)
    }

    fn named_parameters(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let method = self.module.getattr(py, "named_parameters")?;
        method.call0(py)
    }

    #[pyo3(signature = (mode=None))]
    fn train(&mut self, py: Python<'_>, mode: Option<bool>) -> PyResult<()> {
        let method = self.module.getattr(py, "train")?;
        method.call1(py, (mode.unwrap_or(true),))?;
        Ok(())
    }

    fn eval(&mut self, py: Python<'_>) -> PyResult<()> {
        self.train(py, Some(false))
    }

    #[getter]
    fn process_group(&self) -> Option<&Py<PyProcessGroup>> {
        self.process_group.as_ref()
    }
}

/// Register distributed module
pub fn register_distributed_module(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProcessGroup>()?;
    m.add_class::<PyDDP>()?;

    #[pyfunction]
    #[pyo3(signature = (backend, init_method=None, world_size=None, rank=None, store=None, timeout=None, group_name=None, pg_options=None))]
    fn init_process_group(
        backend: String,
        init_method: Option<String>,
        world_size: Option<u32>,
        rank: Option<u32>,
        store: Option<Py<PyAny>>,
        timeout: Option<f64>,
        group_name: Option<String>,
        pg_options: Option<Py<PyAny>>,
    ) -> PyProcessGroup {
        let _ = (backend, init_method, store, timeout, group_name, pg_options);
        PyProcessGroup::new(rank.unwrap_or(0), world_size.unwrap_or(1))
    }

    #[pyfunction]
    #[pyo3(signature = (_group=None))]
    fn destroy_process_group(_group: Option<Py<PyAny>>) -> PyResult<()> {
        Ok(())
    }

    #[pyfunction]
    #[pyo3(signature = (_group=None))]
    fn get_rank(_group: Option<Py<PyAny>>) -> u32 {
        0
    }

    #[pyfunction]
    #[pyo3(signature = (_group=None))]
    fn get_world_size(_group: Option<Py<PyAny>>) -> u32 {
        1
    }

    #[pyfunction]
    fn is_initialized() -> bool {
        false
    }

    #[pyfunction]
    fn is_available() -> bool {
        // Honest: the Python bindings are not wired to a working distributed
        // backend yet, so collective communication is not available here.
        false
    }

    #[pyfunction]
    #[pyo3(signature = (_group=None))]
    fn barrier(_group: Option<Py<PyAny>>) -> PyResult<()> {
        Err(not_implemented("barrier"))
    }

    #[pyfunction]
    #[pyo3(signature = (_tensor, _op=None, _group=None))]
    fn all_reduce(
        _tensor: &PyTensor,
        _op: Option<String>,
        _group: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        Err(not_implemented("all_reduce"))
    }

    #[pyfunction]
    #[pyo3(signature = (_tensor_list, _tensor, _group=None))]
    fn all_gather(
        _tensor_list: Vec<PyTensor>,
        _tensor: &PyTensor,
        _group: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        Err(not_implemented("all_gather"))
    }

    #[pyfunction]
    #[pyo3(signature = (_tensor, _src, _group=None))]
    fn broadcast(_tensor: &PyTensor, _src: u32, _group: Option<Py<PyAny>>) -> PyResult<()> {
        Err(not_implemented("broadcast"))
    }

    #[pyfunction]
    #[pyo3(signature = (_tensor, _dst, _op=None, _group=None))]
    fn reduce(
        _tensor: &PyTensor,
        _dst: u32,
        _op: Option<String>,
        _group: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        Err(not_implemented("reduce"))
    }

    #[pyfunction]
    #[pyo3(signature = (_tensor, _scatter_list=None, _src=0, _group=None))]
    fn scatter(
        _tensor: &PyTensor,
        _scatter_list: Option<Vec<PyTensor>>,
        _src: u32,
        _group: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        Err(not_implemented("scatter"))
    }

    #[pyfunction]
    #[pyo3(signature = (_tensor, _gather_list=None, _dst=0, _group=None))]
    fn gather(
        _tensor: &PyTensor,
        _gather_list: Option<Vec<PyTensor>>,
        _dst: u32,
        _group: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        Err(not_implemented("gather"))
    }

    m.add_function(wrap_pyfunction!(init_process_group, m)?)?;
    m.add_function(wrap_pyfunction!(destroy_process_group, m)?)?;
    m.add_function(wrap_pyfunction!(get_rank, m)?)?;
    m.add_function(wrap_pyfunction!(get_world_size, m)?)?;
    m.add_function(wrap_pyfunction!(is_initialized, m)?)?;
    m.add_function(wrap_pyfunction!(is_available, m)?)?;
    m.add_function(wrap_pyfunction!(barrier, m)?)?;
    m.add_function(wrap_pyfunction!(all_reduce, m)?)?;
    m.add_function(wrap_pyfunction!(all_gather, m)?)?;
    m.add_function(wrap_pyfunction!(broadcast, m)?)?;
    m.add_function(wrap_pyfunction!(reduce, m)?)?;
    m.add_function(wrap_pyfunction!(scatter, m)?)?;
    m.add_function(wrap_pyfunction!(gather, m)?)?;

    Ok(())
}
