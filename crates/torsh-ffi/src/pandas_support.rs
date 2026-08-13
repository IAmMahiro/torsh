//! Pandas support for ToRSh tensors
//!
//! This module provides comprehensive integration with Pandas, enabling seamless conversion
//! between ToRSh tensors and Pandas DataFrames/Series, as well as access to Pandas'
//! data manipulation and analysis functionality.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::error::FfiError;
use crate::numpy_compatibility::NumpyCompat;
use crate::tensor::PyTensor;
use numpy::PyArrayDyn;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyAny, PyDict, PyList, PyModule, PyTuple};
use pyo3::Bound;
use std::collections::HashMap;
use torsh_core::DType;

/// Pandas integration layer providing data manipulation capabilities
#[pyclass(name = "PandasSupport")]
#[derive(Debug)]
pub struct PandasSupport {
    /// NumPy compatibility layer for array operations
    numpy_compat: NumpyCompat,
    /// Cached Pandas module reference
    pandas_module: Option<Py<PyModule>>,
    /// Configuration for Pandas operations
    config: PandasConfig,
    /// Type mappings between ToRSh and Pandas
    #[allow(dead_code)]
    type_mappings: HashMap<String, String>,
}

/// Configuration for Pandas integration
#[derive(Debug, Clone)]
pub struct PandasConfig {
    /// Default index type for DataFrames
    pub default_index_type: String,
    /// Handling of missing values
    pub missing_value_strategy: MissingValueStrategy,
    /// Memory optimization settings
    pub optimize_memory: bool,
    /// Maximum rows to display
    pub max_display_rows: usize,
    /// Precision for floating point display
    pub float_precision: usize,
}

impl Default for PandasConfig {
    fn default() -> Self {
        Self {
            default_index_type: "range".to_string(),
            missing_value_strategy: MissingValueStrategy::DropNA,
            optimize_memory: true,
            max_display_rows: 100,
            float_precision: 4,
        }
    }
}

/// Strategy for handling missing values
#[derive(Debug, Clone)]
pub enum MissingValueStrategy {
    /// Drop rows/columns with missing values
    DropNA,
    /// Fill missing values with a constant
    FillValue(f64),
    /// Forward fill missing values
    ForwardFill,
    /// Backward fill missing values
    BackwardFill,
    /// Interpolate missing values
    Interpolate,
}

/// Result of data analysis operations
#[pyclass(name = "DataAnalysisResult", from_py_object)]
#[derive(Debug, Clone)]
pub struct DataAnalysisResult {
    /// Primary result tensor/data
    #[pyo3(get)]
    pub data: PyTensor,
    /// Statistical summary
    #[pyo3(get)]
    pub statistics: HashMap<String, f64>,
    /// Metadata about the analysis
    #[pyo3(get)]
    pub metadata: HashMap<String, String>,
    /// Column information
    #[pyo3(get)]
    pub columns: Vec<String>,
}

/// DataFrame representation compatible with Pandas
#[pyclass(name = "TorshDataFrame", from_py_object)]
#[derive(Debug, Clone)]
pub struct TorshDataFrame {
    /// Underlying tensor data
    #[pyo3(get)]
    pub data: PyTensor,
    /// Column names
    #[pyo3(get)]
    pub columns: Vec<String>,
    /// Index information
    #[pyo3(get)]
    pub index: Vec<String>,
    /// Data types for each column
    #[pyo3(get)]
    pub dtypes: HashMap<String, String>,
    /// Shape information
    #[pyo3(get)]
    pub shape: (usize, usize),
}

/// Series representation compatible with Pandas
#[pyclass(name = "TorshSeries", from_py_object)]
#[derive(Debug, Clone)]
pub struct TorshSeries {
    /// Underlying tensor data
    #[pyo3(get)]
    pub data: PyTensor,
    /// Series name
    #[pyo3(get)]
    pub name: Option<String>,
    /// Index information
    #[pyo3(get)]
    pub index: Vec<String>,
    /// Data type
    #[pyo3(get)]
    pub dtype: String,
    /// Length
    #[pyo3(get)]
    pub length: usize,
}

#[pymethods]
impl PandasSupport {
    /// Create a new Pandas support instance
    #[new]
    pub fn new() -> PyResult<Self> {
        let mut type_mappings = HashMap::new();

        // Set up type mappings between ToRSh and Pandas
        type_mappings.insert("f32".to_string(), "float32".to_string());
        type_mappings.insert("f64".to_string(), "float64".to_string());
        type_mappings.insert("i32".to_string(), "int32".to_string());
        type_mappings.insert("i64".to_string(), "int64".to_string());
        type_mappings.insert("bool".to_string(), "bool".to_string());

        Ok(Self {
            numpy_compat: NumpyCompat::new(),
            pandas_module: None,
            config: PandasConfig::default(),
            type_mappings,
        })
    }

    /// Configure Pandas support settings
    pub fn configure(&mut self, _py: Python, config: &Bound<'_, PyDict>) -> PyResult<()> {
        if let Some(index_type) = config.get_item("default_index_type")? {
            self.config.default_index_type = index_type.extract()?;
        }
        if let Some(optimize) = config.get_item("optimize_memory")? {
            self.config.optimize_memory = optimize.extract()?;
        }
        if let Some(max_rows) = config.get_item("max_display_rows")? {
            self.config.max_display_rows = max_rows.extract()?;
        }
        if let Some(precision) = config.get_item("float_precision")? {
            self.config.float_precision = precision.extract()?;
        }
        Ok(())
    }

    /// Convert ToRSh tensor to Pandas DataFrame
    pub fn to_dataframe(
        &self,
        py: Python,
        tensor: &PyTensor,
        columns: Option<Vec<String>>,
        index: Option<Vec<String>>,
    ) -> PyResult<Py<PyAny>> {
        let pandas = self.get_pandas_module(py)?;
        let numpy_array = self
            .numpy_compat
            .to_numpy_array(&tensor.data, &tensor.shape)
            .map_err(|e| FfiError::InvalidConversion { message: e })?;

        // Create DataFrame constructor arguments
        let kwargs = PyDict::new(py);
        kwargs.set_item("data", numpy_array)?;

        if let Some(cols) = columns {
            kwargs.set_item("columns", cols)?;
        }

        if let Some(idx) = index {
            kwargs.set_item("index", idx)?;
        }

        let dataframe = pandas.call_method(py, "DataFrame", (), Some(&kwargs))?;
        Ok(dataframe)
    }

    /// Convert Pandas DataFrame to ToRSh tensor
    pub fn from_dataframe(
        &self,
        _py: Python,
        dataframe: Bound<'_, PyAny>,
    ) -> PyResult<TorshDataFrame> {
        // Get the underlying numpy array and convert it (see `extract_f32_array`
        // for why an `.astype("float32")` normalization is required here).
        let values = dataframe.getattr("_values")?;
        let data = self.extract_f32_array(&values)?;

        // Column labels. Read as raw Python objects (not yet stringified) so
        // that the `dtypes` lookup below can index positionally via `.iloc`
        // rather than assuming labels are string keys (a plain
        // `pd.DataFrame(np.zeros((r, c)))` has an integer `RangeIndex` for
        // its columns, not strings).
        let columns_py = dataframe.getattr("columns")?;
        let raw_columns = columns_py.call_method("tolist", (), None)?;
        let raw_columns = raw_columns.cast::<PyList>()?;

        let dtypes_iloc = dataframe.getattr("dtypes")?.getattr("iloc")?;
        let mut columns = Vec::with_capacity(raw_columns.len());
        let mut dtypes = HashMap::new();
        for (i, item) in raw_columns.iter().enumerate() {
            let col_name: String = item.str()?.extract()?;
            let dtype_str: String = dtypes_iloc.get_item(i)?.str()?.extract()?;
            dtypes.insert(col_name.clone(), dtype_str);
            columns.push(col_name);
        }

        // Row index. Stringify defensively: the default `RangeIndex` is
        // integer-valued, and arbitrary index dtypes (datetime, etc.) are
        // not directly extractable as Rust `String`.
        let index_py = dataframe.getattr("index")?;
        let index: Vec<String> = index_py
            .call_method("tolist", (), None)?
            .cast::<PyList>()?
            .iter()
            .map(|item| item.str().and_then(|s| s.extract::<String>()))
            .collect::<PyResult<Vec<String>>>()?;

        let shape_py = dataframe.getattr("shape")?;
        let shape_tuple = shape_py.cast::<PyTuple>()?;
        let shape: (usize, usize) = (
            shape_tuple.get_item(0)?.extract()?,
            shape_tuple.get_item(1)?.extract()?,
        );

        let tensor = PyTensor::from_raw(data, vec![shape.0, shape.1], DType::F32, false);

        Ok(TorshDataFrame {
            data: tensor,
            columns,
            index,
            dtypes,
            shape,
        })
    }

    /// Convert ToRSh tensor to Pandas Series
    pub fn to_series(
        &self,
        py: Python,
        tensor: &PyTensor,
        name: Option<String>,
        index: Option<Vec<String>>,
    ) -> PyResult<Py<PyAny>> {
        let pandas = self.get_pandas_module(py)?;
        let numpy_array = self
            .numpy_compat
            .to_numpy_array(&tensor.data, &tensor.shape)
            .map_err(|e| FfiError::InvalidConversion { message: e })?;

        let kwargs = PyDict::new(py);
        kwargs.set_item("data", numpy_array)?;

        if let Some(name_str) = name {
            kwargs.set_item("name", name_str)?;
        }

        if let Some(idx) = index {
            kwargs.set_item("index", idx)?;
        }

        let series = pandas.call_method(py, "Series", (), Some(&kwargs))?;
        Ok(series)
    }

    /// Convert Pandas Series to ToRSh tensor
    pub fn from_series(&self, _py: Python, series: Bound<'_, PyAny>) -> PyResult<TorshSeries> {
        // Get the underlying numpy array and convert it
        let values = series.getattr("values")?;
        let data = self.extract_f32_array(&values)?;

        // Extract metadata
        let name_py = series.getattr("name")?;
        let name = if name_py.is_none() {
            None
        } else {
            Some(name_py.str()?.extract()?)
        };

        let index_py = series.getattr("index")?;
        let index: Vec<String> = index_py
            .call_method("tolist", (), None)?
            .cast::<PyList>()?
            .iter()
            .map(|item| item.str().and_then(|s| s.extract::<String>()))
            .collect::<PyResult<Vec<String>>>()?;

        let dtype_py = series.getattr("dtype")?;
        let dtype: String = dtype_py.str()?.extract()?;

        let length: usize = series.len()?;

        let tensor = PyTensor::from_raw(data, vec![length], DType::F32, false);

        Ok(TorshSeries {
            data: tensor,
            name,
            index,
            dtype,
            length,
        })
    }

    /// Perform data grouping operations
    pub fn groupby_analysis(
        &self,
        py: Python,
        dataframe: Bound<'_, PyAny>,
        group_by: Vec<String>,
        aggregation: &str,
    ) -> PyResult<DataAnalysisResult> {
        let grouped = dataframe.call_method1("groupby", (group_by.clone(),))?;

        let result = match aggregation {
            "mean" => grouped.call_method("mean", (), None)?,
            "sum" => grouped.call_method("sum", (), None)?,
            "count" => grouped.call_method("count", (), None)?,
            "std" => grouped.call_method("std", (), None)?,
            "var" => grouped.call_method("var", (), None)?,
            "min" => grouped.call_method("min", (), None)?,
            "max" => grouped.call_method("max", (), None)?,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown aggregation: {}",
                    aggregation
                )))
            }
        };

        // Convert result back to tensor
        let torsh_df = self.from_dataframe(py, result)?;

        // Generate statistics
        let mut statistics = HashMap::new();
        statistics.insert("num_groups".to_string(), group_by.len() as f64);
        statistics.insert("aggregation_type".to_string(), aggregation.len() as f64);

        // Generate metadata
        let mut metadata = HashMap::new();
        metadata.insert("operation".to_string(), "groupby".to_string());
        metadata.insert("aggregation".to_string(), aggregation.to_string());
        metadata.insert("group_columns".to_string(), group_by.join(","));

        Ok(DataAnalysisResult {
            data: torsh_df.data,
            statistics,
            metadata,
            columns: torsh_df.columns,
        })
    }

    /// Perform statistical analysis
    pub fn statistical_analysis(
        &self,
        py: Python,
        dataframe: Bound<'_, PyAny>,
    ) -> PyResult<DataAnalysisResult> {
        let describe_result = dataframe.call_method("describe", (), None)?;
        let corr_result = dataframe.call_method("corr", (), None)?;

        // Convert correlation matrix to tensor
        let corr_df = self.from_dataframe(py, corr_result)?;

        // Extract statistical summaries
        let mut statistics = HashMap::new();
        let _describe_df = self.from_dataframe(py, describe_result)?;

        // Get basic stats
        let mean_values = dataframe.call_method("mean", (), None)?;
        let std_values = dataframe.call_method("std", (), None)?;
        let min_values = dataframe.call_method("min", (), None)?;
        let max_values = dataframe.call_method("max", (), None)?;

        statistics.insert(
            "mean_of_means".to_string(),
            mean_values.call_method("mean", (), None)?.extract()?,
        );
        statistics.insert(
            "mean_of_stds".to_string(),
            std_values.call_method("mean", (), None)?.extract()?,
        );
        statistics.insert(
            "global_min".to_string(),
            min_values.call_method("min", (), None)?.extract()?,
        );
        statistics.insert(
            "global_max".to_string(),
            max_values.call_method("max", (), None)?.extract()?,
        );

        let mut metadata = HashMap::new();
        metadata.insert("operation".to_string(), "statistical_analysis".to_string());
        metadata.insert(
            "includes".to_string(),
            "correlation,describe,summary".to_string(),
        );

        Ok(DataAnalysisResult {
            data: corr_df.data,
            statistics,
            metadata,
            columns: corr_df.columns,
        })
    }

    /// Handle missing values according to strategy
    pub fn handle_missing_values(
        &self,
        py: Python,
        dataframe: Bound<'_, PyAny>,
        strategy: &str,
        value: Option<f64>,
    ) -> PyResult<Py<PyAny>> {
        match strategy {
            "dropna" => Ok(dataframe.call_method("dropna", (), None)?.into()),
            "fillna" => {
                let fill_value = value.unwrap_or(0.0);
                Ok(dataframe.call_method1("fillna", (fill_value,))?.into())
            }
            "ffill" => {
                let kwargs = [("method", "ffill")].into_py_dict(py)?;
                Ok(dataframe.call_method("fillna", (), Some(&kwargs))?.into())
            }
            "bfill" => {
                let kwargs = [("method", "bfill")].into_py_dict(py)?;
                Ok(dataframe.call_method("fillna", (), Some(&kwargs))?.into())
            }
            "interpolate" => Ok(dataframe.call_method("interpolate", (), None)?.into()),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown missing value strategy: {}",
                strategy
            ))),
        }
    }

    /// Perform data filtering and selection
    pub fn filter_data(
        &self,
        _py: Python,
        dataframe: Bound<'_, PyAny>,
        query: &str,
    ) -> PyResult<Py<PyAny>> {
        Ok(dataframe.call_method1("query", (query,))?.into())
    }

    /// Merge/join DataFrames
    pub fn merge_dataframes(
        &self,
        py: Python,
        left: Bound<'_, PyAny>,
        right: Bound<'_, PyAny>,
        on: Vec<String>,
        how: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let _pandas = self.get_pandas_module(py)?;
        let how_str = how.unwrap_or("inner");

        let kwargs = PyDict::new(py);
        // An empty `on` means "let pandas infer the common columns" (its
        // `on=None` default); passing an empty list explicitly would instead
        // make pandas raise `MergeError: No common columns`.
        if !on.is_empty() {
            kwargs.set_item("on", on)?;
        }
        kwargs.set_item("how", how_str)?;

        Ok(left.call_method("merge", (right,), Some(&kwargs))?.into())
    }

    /// Pivot table operations
    pub fn pivot_table(
        &self,
        py: Python,
        dataframe: Bound<'_, PyAny>,
        values: Vec<String>,
        index: Vec<String>,
        columns: Vec<String>,
        aggfunc: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let kwargs = PyDict::new(py);
        // Empty vectors mean "unspecified" for each of these -- pandas'
        // `pivot_table` treats `values`/`index`/`columns=None` as "use all
        // remaining columns", which is a more useful default than forcing
        // an empty list (which pandas would otherwise reject).
        if !values.is_empty() {
            kwargs.set_item("values", values)?;
        }
        if !index.is_empty() {
            kwargs.set_item("index", index)?;
        }
        if !columns.is_empty() {
            kwargs.set_item("columns", columns)?;
        }
        kwargs.set_item("aggfunc", aggfunc.unwrap_or("mean"))?;

        Ok(dataframe
            .call_method("pivot_table", (), Some(&kwargs))?
            .into())
    }

    /// Time series operations
    ///
    /// Design choice: `resample(freq)` and `rolling(window)` are mutually
    /// exclusive pandas time-series transforms, so exactly one selects the
    /// operation to perform (preferring `freq`/resample if both are given).
    /// Plain descriptive statistics without either are already covered by
    /// [`Self::statistical_analysis`], so neither being present is an error.
    pub fn time_series_analysis(
        &self,
        py: Python,
        series: Bound<'_, PyAny>,
        freq: Option<&str>,
        window: Option<usize>,
    ) -> PyResult<DataAnalysisResult> {
        let (result, operation) = if let Some(freq) = freq {
            let resampled = series.call_method1("resample", (freq,))?;
            (resampled.call_method("mean", (), None)?, "resample")
        } else if let Some(window) = window {
            let rolling = series.call_method1("rolling", (window,))?;
            (rolling.call_method("mean", (), None)?, "rolling")
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "time_series_analysis requires either `freq` (for resample) or `window` (for rolling)",
            ));
        };

        // Normalize a Series result (the common case for a single input
        // Series) into a single-column DataFrame so it can flow through the
        // shared `from_dataframe` extraction path used elsewhere in this file.
        let result_df = if result.hasattr("columns")? {
            result
        } else {
            result.call_method("to_frame", (), None)?
        };

        let torsh_df = self.from_dataframe(py, result_df)?;

        let mut statistics = HashMap::new();
        statistics.insert(
            "num_observations".to_string(),
            torsh_df.data.data.len() as f64,
        );
        if let Some(window) = window {
            statistics.insert("window".to_string(), window as f64);
        }

        let mut metadata = HashMap::new();
        metadata.insert("operation".to_string(), operation.to_string());
        if let Some(freq) = freq {
            metadata.insert("freq".to_string(), freq.to_string());
        }

        Ok(DataAnalysisResult {
            data: torsh_df.data,
            statistics,
            metadata,
            columns: torsh_df.columns,
        })
    }

    /// Get Pandas version information
    pub fn get_pandas_version(&self, py: Python) -> PyResult<String> {
        let pandas = self.get_pandas_module(py)?;
        let version: String = pandas.getattr(py, "__version__")?.extract(py)?;
        Ok(version)
    }

    /// Export DataFrame to various formats
    pub fn export_dataframe(
        &self,
        py: Python,
        dataframe: Bound<'_, PyAny>,
        format: &str,
        path: &str,
    ) -> PyResult<()> {
        match format.to_lowercase().as_str() {
            "csv" => {
                dataframe.call_method1("to_csv", (path,))?;
            }
            "json" => {
                dataframe.call_method1("to_json", (path,))?;
            }
            "excel" => {
                dataframe.call_method1("to_excel", (path,))?;
            }
            "parquet" => {
                dataframe.call_method1("to_parquet", (path,))?;
            }
            "hdf5" | "hdf" => {
                let kwargs = [("mode", "w")].into_py_dict(py)?;
                dataframe.call_method("to_hdf", (path, "data"), Some(&kwargs))?;
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unsupported export format: {}",
                    format
                )))
            }
        }
        Ok(())
    }

    /// Import DataFrame from various formats
    pub fn import_dataframe(&self, py: Python, format: &str, path: &str) -> PyResult<Py<PyAny>> {
        let pandas = self.get_pandas_module(py)?;

        match format.to_lowercase().as_str() {
            "csv" => pandas.call_method1(py, "read_csv", (path,)),
            "json" => pandas.call_method1(py, "read_json", (path,)),
            "excel" => pandas.call_method1(py, "read_excel", (path,)),
            "parquet" => pandas.call_method1(py, "read_parquet", (path,)),
            "hdf5" | "hdf" => pandas.call_method(py, "read_hdf", (path, "data"), None),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unsupported import format: {}",
                format
            ))),
        }
    }
}

impl PandasSupport {
    /// Get or cache Pandas module
    fn get_pandas_module(&self, py: Python) -> PyResult<Py<PyModule>> {
        if let Some(module) = &self.pandas_module {
            return Ok(module.clone_ref(py));
        }

        let module = py.import("pandas")?;
        Ok(module.into())
    }

    /// Coerce an arbitrary NumPy-array-like Python object (e.g. the result
    /// of `DataFrame._values` / `Series.values`) into a `Vec<f32>` via
    /// [`NumpyCompat::from_numpy_array`].
    ///
    /// Pandas defaults essentially all numeric columns to `float64`, so an
    /// unconditional `.astype("float32")` normalization is applied before
    /// casting to `PyArrayDyn<f32>`. Without it, this conversion would only
    /// work for the rare caller that already has float32-typed data, making
    /// DataFrame/Series ingestion effectively unusable for ordinary pandas
    /// output.
    fn extract_f32_array(&self, array: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
        let float32_array = array.call_method1("astype", ("float32",))?;
        let py_array =
            float32_array
                .cast::<PyArrayDyn<f32>>()
                .map_err(|e| FfiError::InvalidConversion {
                    message: format!("Expected a NumPy-array-convertible value: {}", e),
                })?;
        self.numpy_compat
            .from_numpy_array(py_array)
            .map_err(|e| FfiError::InvalidConversion { message: e }.into())
    }
}

/// Create Pandas support utilities
pub fn create_pandas_utilities(py: Python) -> PyResult<Bound<PyDict>> {
    let utils = PyDict::new(py);

    // Add utility functions
    utils.set_item("create_support", py.get_type::<PandasSupport>())?;
    utils.set_item("TorshDataFrame", py.get_type::<TorshDataFrame>())?;
    utils.set_item("TorshSeries", py.get_type::<TorshSeries>())?;
    utils.set_item("DataAnalysisResult", py.get_type::<DataAnalysisResult>())?;

    Ok(utils)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pandas_config() {
        let config = PandasConfig::default();
        assert_eq!(config.default_index_type, "range");
        assert_eq!(config.max_display_rows, 100);
        assert!(config.optimize_memory);
    }

    #[test]
    fn test_missing_value_strategy() {
        let strategy = MissingValueStrategy::FillValue(42.0);
        match strategy {
            MissingValueStrategy::FillValue(val) => assert_eq!(val, 42.0),
            _ => panic!("Expected FillValue strategy"),
        }
    }

    /// Skip a pandas-dependent test cleanly if pandas isn't importable in
    /// this environment, rather than failing the whole suite.
    fn require_pandas(py: Python<'_>) -> bool {
        py.import("pandas").is_ok()
    }

    /// Compare two `f32` slices elementwise within a small tolerance, rather
    /// than via `assert_eq!` -- avoids relying on bit-exact float equality
    /// after a round trip through Python/pandas/NumPy.
    fn assert_f32_slice_approx_eq(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-5, "{:?} vs {:?}", actual, expected);
        }
    }

    #[test]
    fn test_from_dataframe_round_trip() {
        Python::initialize();
        Python::attach(|py| {
            if !require_pandas(py) {
                eprintln!("skipping test_from_dataframe_round_trip: pandas not installed");
                return;
            }
            let support = PandasSupport::new().expect("PandasSupport::new should succeed");
            let df = py
                .eval(
                    c"__import__('pandas').DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})",
                    None,
                    None,
                )
                .expect("failed to build test dataframe");

            let torsh_df = support
                .from_dataframe(py, df)
                .expect("from_dataframe should succeed on a real, non-empty numeric DataFrame");

            assert_eq!(torsh_df.shape, (3, 2));
            assert_eq!(torsh_df.columns, vec!["a".to_string(), "b".to_string()]);
            assert_eq!(
                torsh_df.index,
                vec!["0".to_string(), "1".to_string(), "2".to_string()]
            );
            assert_eq!(torsh_df.data.shape, vec![3, 2]);
            // `_values` is row-major: row i is [a[i], b[i]].
            assert_f32_slice_approx_eq(&torsh_df.data.data, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
            assert_eq!(
                torsh_df.dtypes.get("a").map(String::as_str),
                Some("float64")
            );
        });
    }

    #[test]
    fn test_from_series_round_trip() {
        Python::initialize();
        Python::attach(|py| {
            if !require_pandas(py) {
                eprintln!("skipping test_from_series_round_trip: pandas not installed");
                return;
            }
            let support = PandasSupport::new().expect("PandasSupport::new should succeed");
            let series = py
                .eval(
                    c"__import__('pandas').Series([1.5, 2.5, 3.5, 4.5], name='s')",
                    None,
                    None,
                )
                .expect("failed to build test series");

            let torsh_series = support
                .from_series(py, series)
                .expect("from_series should succeed on a real, non-empty numeric Series");

            assert_eq!(torsh_series.length, 4);
            assert_eq!(torsh_series.name, Some("s".to_string()));
            assert_f32_slice_approx_eq(&torsh_series.data.data, &[1.5, 2.5, 3.5, 4.5]);
            assert_eq!(torsh_series.data.shape, vec![4]);
        });
    }

    #[test]
    fn test_merge_dataframes() {
        Python::initialize();
        Python::attach(|py| {
            if !require_pandas(py) {
                eprintln!("skipping test_merge_dataframes: pandas not installed");
                return;
            }
            let support = PandasSupport::new().expect("PandasSupport::new should succeed");
            let left = py
                .eval(
                    c"__import__('pandas').DataFrame({'key': ['a', 'b', 'c'], 'left_val': [1, 2, 3]})",
                    None,
                    None,
                )
                .expect("failed to build left dataframe");
            let right = py
                .eval(
                    c"__import__('pandas').DataFrame({'key': ['a', 'b', 'd'], 'right_val': [10, 20, 30]})",
                    None,
                    None,
                )
                .expect("failed to build right dataframe");

            let merged = support
                .merge_dataframes(py, left, right, vec!["key".to_string()], Some("inner"))
                .expect("merge_dataframes should succeed");
            let merged = merged.bind(py);

            // Inner join on 'key' only matches 'a' and 'b' ('c' vs 'd' don't).
            assert_eq!(merged.len().expect("merged result should have a length"), 2);
            let columns: Vec<String> = merged
                .getattr("columns")
                .expect("columns")
                .call_method("tolist", (), None)
                .expect("tolist")
                .extract()
                .expect("extract columns");
            assert!(columns.contains(&"left_val".to_string()));
            assert!(columns.contains(&"right_val".to_string()));
        });
    }

    #[test]
    fn test_pivot_table() {
        Python::initialize();
        Python::attach(|py| {
            if !require_pandas(py) {
                eprintln!("skipping test_pivot_table: pandas not installed");
                return;
            }
            let support = PandasSupport::new().expect("PandasSupport::new should succeed");
            let df = py
                .eval(
                    c"__import__('pandas').DataFrame({'cat': ['x', 'x', 'y', 'y'], 'val': [1.0, 3.0, 2.0, 4.0]})",
                    None,
                    None,
                )
                .expect("failed to build test dataframe");

            let pivoted = support
                .pivot_table(
                    py,
                    df,
                    vec!["val".to_string()],
                    vec!["cat".to_string()],
                    vec![],
                    Some("mean"),
                )
                .expect("pivot_table should succeed");
            let pivoted = pivoted.bind(py);

            // Two distinct categories ('x', 'y') become two index rows.
            assert_eq!(
                pivoted.len().expect("pivoted result should have a length"),
                2
            );
        });
    }

    #[test]
    fn test_time_series_analysis_rolling() {
        Python::initialize();
        Python::attach(|py| {
            if !require_pandas(py) {
                eprintln!("skipping test_time_series_analysis_rolling: pandas not installed");
                return;
            }
            let support = PandasSupport::new().expect("PandasSupport::new should succeed");
            let series = py
                .eval(
                    c"__import__('pandas').Series([1.0, 2.0, 3.0, 4.0, 5.0])",
                    None,
                    None,
                )
                .expect("failed to build test series");

            let result = support
                .time_series_analysis(py, series, None, Some(2))
                .expect("time_series_analysis with a rolling window should succeed");

            assert_eq!(
                result.metadata.get("operation").map(String::as_str),
                Some("rolling")
            );
            assert_eq!(result.data.data.len(), 5);
            // First rolling(2).mean() observation is NaN (not enough history).
            assert!(result.data.data[0].is_nan());
            assert!((result.data.data[1] - 1.5).abs() < 1e-6);
            assert!((result.data.data[4] - 4.5).abs() < 1e-6);
        });
    }

    #[test]
    fn test_time_series_analysis_requires_freq_or_window() {
        Python::initialize();
        Python::attach(|py| {
            if !require_pandas(py) {
                eprintln!(
                    "skipping test_time_series_analysis_requires_freq_or_window: pandas not installed"
                );
                return;
            }
            let support = PandasSupport::new().expect("PandasSupport::new should succeed");
            let series = py
                .eval(c"__import__('pandas').Series([1.0, 2.0, 3.0])", None, None)
                .expect("failed to build test series");

            let result = support.time_series_analysis(py, series, None, None);
            assert!(
                result.is_err(),
                "expected an error when neither freq nor window is given"
            );
        });
    }
}
