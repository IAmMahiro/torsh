//! NumPy compatibility layer for seamless integration with ToRSh
//!
//! This module provides comprehensive compatibility with NumPy arrays,
//! enabling zero-copy conversion, broadcasting compatibility, and familiar
//! NumPy-style operations on ToRSh tensors.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use numpy::{IxDyn as NpIxDyn, PyArray1, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;

/// NumPy data type mapping
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NumpyDType {
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float16,
    Float32,
    Float64,
    Complex64,
    Complex128,
}

impl NumpyDType {
    /// Convert from NumPy dtype string
    pub fn from_numpy_str(dtype_str: &str) -> Option<Self> {
        match dtype_str {
            "bool" | "bool_" => Some(Self::Bool),
            "int8" => Some(Self::Int8),
            "int16" => Some(Self::Int16),
            "int32" => Some(Self::Int32),
            "int64" => Some(Self::Int64),
            "uint8" => Some(Self::UInt8),
            "uint16" => Some(Self::UInt16),
            "uint32" => Some(Self::UInt32),
            "uint64" => Some(Self::UInt64),
            "float16" => Some(Self::Float16),
            "float32" => Some(Self::Float32),
            "float64" => Some(Self::Float64),
            "complex64" => Some(Self::Complex64),
            "complex128" => Some(Self::Complex128),
            _ => None,
        }
    }

    /// Convert to NumPy dtype string
    pub fn to_numpy_str(&self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::Int8 => "int8",
            Self::Int16 => "int16",
            Self::Int32 => "int32",
            Self::Int64 => "int64",
            Self::UInt8 => "uint8",
            Self::UInt16 => "uint16",
            Self::UInt32 => "uint32",
            Self::UInt64 => "uint64",
            Self::Float16 => "float16",
            Self::Float32 => "float32",
            Self::Float64 => "float64",
            Self::Complex64 => "complex64",
            Self::Complex128 => "complex128",
        }
    }

    /// Get the size in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::Bool | Self::Int8 | Self::UInt8 => 1,
            Self::Int16 | Self::UInt16 | Self::Float16 => 2,
            Self::Int32 | Self::UInt32 | Self::Float32 => 4,
            Self::Int64 | Self::UInt64 | Self::Float64 | Self::Complex64 => 8,
            Self::Complex128 => 16,
        }
    }
}

/// NumPy array metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumpyArrayInfo {
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
    pub dtype: NumpyDType,
    pub fortran_order: bool,
    pub contiguous: bool,
    pub writeable: bool,
    pub aligned: bool,
}

/// NumPy-style broadcasting rules
#[derive(Debug, Clone)]
pub struct BroadcastingRules {
    /// Enable automatic broadcasting
    pub auto_broadcast: bool,
    /// Maximum number of dimensions for broadcasting
    pub max_dims: usize,
    /// Strict NumPy compatibility mode
    pub strict_numpy_compat: bool,
}

impl Default for BroadcastingRules {
    fn default() -> Self {
        Self {
            auto_broadcast: true,
            max_dims: 32,
            strict_numpy_compat: true,
        }
    }
}

/// NumPy compatibility layer
#[derive(Debug)]
pub struct NumpyCompat {
    broadcasting_rules: BroadcastingRules,
    type_promotions: HashMap<(NumpyDType, NumpyDType), NumpyDType>,
    #[allow(dead_code)]
    conversion_cache: Arc<RwLock<HashMap<String, Vec<u8>>>>,
}

impl NumpyCompat {
    /// Create a new NumPy compatibility layer
    pub fn new() -> Self {
        let mut compat = Self {
            broadcasting_rules: BroadcastingRules::default(),
            type_promotions: HashMap::new(),
            conversion_cache: Arc::new(RwLock::new(HashMap::new())),
        };
        compat.init_type_promotions();
        compat
    }

    /// Initialize NumPy-style type promotion rules
    fn init_type_promotions(&mut self) {
        // NumPy type promotion hierarchy
        let promotions = vec![
            // Boolean promotions
            ((NumpyDType::Bool, NumpyDType::Int8), NumpyDType::Int8),
            ((NumpyDType::Bool, NumpyDType::Float32), NumpyDType::Float32),
            ((NumpyDType::Bool, NumpyDType::Float64), NumpyDType::Float64),
            // Integer promotions
            ((NumpyDType::Int8, NumpyDType::Int16), NumpyDType::Int16),
            ((NumpyDType::Int8, NumpyDType::Int32), NumpyDType::Int32),
            ((NumpyDType::Int8, NumpyDType::Int64), NumpyDType::Int64),
            ((NumpyDType::Int16, NumpyDType::Int32), NumpyDType::Int32),
            ((NumpyDType::Int16, NumpyDType::Int64), NumpyDType::Int64),
            ((NumpyDType::Int32, NumpyDType::Int64), NumpyDType::Int64),
            // Unsigned integer promotions
            ((NumpyDType::UInt8, NumpyDType::UInt16), NumpyDType::UInt16),
            ((NumpyDType::UInt8, NumpyDType::UInt32), NumpyDType::UInt32),
            ((NumpyDType::UInt8, NumpyDType::UInt64), NumpyDType::UInt64),
            ((NumpyDType::UInt16, NumpyDType::UInt32), NumpyDType::UInt32),
            ((NumpyDType::UInt16, NumpyDType::UInt64), NumpyDType::UInt64),
            ((NumpyDType::UInt32, NumpyDType::UInt64), NumpyDType::UInt64),
            // Mixed signed/unsigned promotions
            ((NumpyDType::Int8, NumpyDType::UInt8), NumpyDType::Int16),
            ((NumpyDType::Int16, NumpyDType::UInt16), NumpyDType::Int32),
            ((NumpyDType::Int32, NumpyDType::UInt32), NumpyDType::Int64),
            // Float promotions
            (
                (NumpyDType::Float16, NumpyDType::Float32),
                NumpyDType::Float32,
            ),
            (
                (NumpyDType::Float16, NumpyDType::Float64),
                NumpyDType::Float64,
            ),
            (
                (NumpyDType::Float32, NumpyDType::Float64),
                NumpyDType::Float64,
            ),
            // Integer to float promotions
            ((NumpyDType::Int8, NumpyDType::Float16), NumpyDType::Float16),
            ((NumpyDType::Int8, NumpyDType::Float32), NumpyDType::Float32),
            ((NumpyDType::Int8, NumpyDType::Float64), NumpyDType::Float64),
            (
                (NumpyDType::Int16, NumpyDType::Float32),
                NumpyDType::Float32,
            ),
            (
                (NumpyDType::Int16, NumpyDType::Float64),
                NumpyDType::Float64,
            ),
            (
                (NumpyDType::Int32, NumpyDType::Float64),
                NumpyDType::Float64,
            ),
            // Complex promotions
            (
                (NumpyDType::Float32, NumpyDType::Complex64),
                NumpyDType::Complex64,
            ),
            (
                (NumpyDType::Float64, NumpyDType::Complex128),
                NumpyDType::Complex128,
            ),
            (
                (NumpyDType::Complex64, NumpyDType::Complex128),
                NumpyDType::Complex128,
            ),
        ];

        for ((a, b), result) in promotions {
            self.type_promotions
                .insert((a.clone(), b.clone()), result.clone());
            self.type_promotions.insert((b, a), result); // Commutative
        }
    }

    /// Promote two data types according to NumPy rules
    pub fn promote_types(&self, a: &NumpyDType, b: &NumpyDType) -> NumpyDType {
        if a == b {
            return a.clone();
        }

        if let Some(promoted) = self.type_promotions.get(&(a.clone(), b.clone())) {
            promoted.clone()
        } else {
            // Default to the "larger" type
            match (a, b) {
                (NumpyDType::Float64, _) | (_, NumpyDType::Float64) => NumpyDType::Float64,
                (NumpyDType::Float32, _) | (_, NumpyDType::Float32) => NumpyDType::Float32,
                (NumpyDType::Int64, _) | (_, NumpyDType::Int64) => NumpyDType::Int64,
                (NumpyDType::Int32, _) | (_, NumpyDType::Int32) => NumpyDType::Int32,
                _ => a.clone(),
            }
        }
    }

    /// Check if two shapes can be broadcast together
    pub fn can_broadcast(&self, shape1: &[usize], shape2: &[usize]) -> bool {
        if !self.broadcasting_rules.auto_broadcast {
            return shape1 == shape2;
        }

        let max_len = shape1.len().max(shape2.len());
        if max_len > self.broadcasting_rules.max_dims {
            return false;
        }

        // Pad with 1s on the left
        let padded1: Vec<usize> = std::iter::repeat(1)
            .take(max_len.saturating_sub(shape1.len()))
            .chain(shape1.iter().cloned())
            .collect();

        let padded2: Vec<usize> = std::iter::repeat(1)
            .take(max_len.saturating_sub(shape2.len()))
            .chain(shape2.iter().cloned())
            .collect();

        // Check broadcasting rules
        for (d1, d2) in padded1.iter().zip(padded2.iter()) {
            if *d1 != *d2 && *d1 != 1 && *d2 != 1 {
                return false;
            }
        }

        true
    }

    /// Compute the broadcast shape for two shapes
    pub fn broadcast_shapes(&self, shape1: &[usize], shape2: &[usize]) -> Option<Vec<usize>> {
        if !self.can_broadcast(shape1, shape2) {
            return None;
        }

        let max_len = shape1.len().max(shape2.len());

        // Pad with 1s on the left
        let padded1: Vec<usize> = std::iter::repeat(1)
            .take(max_len.saturating_sub(shape1.len()))
            .chain(shape1.iter().cloned())
            .collect();

        let padded2: Vec<usize> = std::iter::repeat(1)
            .take(max_len.saturating_sub(shape2.len()))
            .chain(shape2.iter().cloned())
            .collect();

        let result: Vec<usize> = padded1
            .iter()
            .zip(padded2.iter())
            .map(|(d1, d2)| (*d1).max(*d2))
            .collect();

        Some(result)
    }

    /// Convert strides from NumPy format to ToRSh format
    pub fn convert_strides(&self, strides: &[isize], dtype: &NumpyDType) -> Vec<usize> {
        let element_size = dtype.size_bytes() as isize;
        strides
            .iter()
            .map(|&stride| (stride / element_size) as usize)
            .collect()
    }

    /// Check if array is C-contiguous (NumPy default)
    pub fn is_c_contiguous(&self, shape: &[usize], strides: &[isize], dtype: &NumpyDType) -> bool {
        if shape.is_empty() {
            return true;
        }

        let element_size = dtype.size_bytes() as isize;
        let mut expected_stride = element_size;

        for i in (0..shape.len()).rev() {
            if strides[i] != expected_stride {
                return false;
            }
            expected_stride *= shape[i] as isize;
        }

        true
    }

    /// Check if array is Fortran-contiguous
    pub fn is_fortran_contiguous(
        &self,
        shape: &[usize],
        strides: &[isize],
        dtype: &NumpyDType,
    ) -> bool {
        if shape.is_empty() {
            return true;
        }

        let element_size = dtype.size_bytes() as isize;
        let mut expected_stride = element_size;

        for i in 0..shape.len() {
            if strides[i] != expected_stride {
                return false;
            }
            expected_stride *= shape[i] as isize;
        }

        true
    }

    /// Create NumPy-compatible array info
    pub fn create_array_info(
        &self,
        shape: Vec<usize>,
        dtype: NumpyDType,
        order: Option<&str>,
    ) -> NumpyArrayInfo {
        let fortran_order = order == Some("F");
        let strides = self.compute_strides(&shape, &dtype, fortran_order);
        let contiguous = self.is_c_contiguous(&shape, &strides, &dtype)
            || self.is_fortran_contiguous(&shape, &strides, &dtype);

        NumpyArrayInfo {
            shape,
            strides,
            dtype,
            fortran_order,
            contiguous,
            writeable: true,
            aligned: true,
        }
    }

    /// Compute strides for given shape and memory order
    fn compute_strides(
        &self,
        shape: &[usize],
        dtype: &NumpyDType,
        fortran_order: bool,
    ) -> Vec<isize> {
        if shape.is_empty() {
            return vec![];
        }

        let element_size = dtype.size_bytes() as isize;
        let mut strides = vec![0; shape.len()];

        if fortran_order {
            // Fortran order: stride increases from first to last dimension
            let mut stride = element_size;
            for i in 0..shape.len() {
                strides[i] = stride;
                stride *= shape[i] as isize;
            }
        } else {
            // C order: stride decreases from last to first dimension
            let mut stride = element_size;
            for i in (0..shape.len()).rev() {
                strides[i] = stride;
                stride *= shape[i] as isize;
            }
        }

        strides
    }

    /// NumPy-style array slicing
    pub fn slice_array(&self, shape: &[usize], slice_spec: &[SliceSpec]) -> SliceResult {
        let mut new_shape = Vec::new();
        let mut new_strides = Vec::new();
        let mut offset = 0;

        for (i, &dim_size) in shape.iter().enumerate() {
            let slice = if i < slice_spec.len() {
                &slice_spec[i]
            } else {
                &SliceSpec::Full
            };

            match slice {
                SliceSpec::Full => {
                    new_shape.push(dim_size);
                    new_strides.push(1);
                }
                SliceSpec::Index(idx) => {
                    // Single index, dimension is removed
                    let actual_idx = if *idx < 0 {
                        (dim_size as isize + idx) as usize
                    } else {
                        *idx as usize
                    };
                    offset += actual_idx;
                }
                SliceSpec::Range { start, end, step } => {
                    let actual_start = start.unwrap_or(0);
                    let actual_end = end.unwrap_or(dim_size);
                    let actual_step = step.unwrap_or(1);

                    let slice_size = if actual_step > 0 {
                        ((actual_end.saturating_sub(actual_start)) + actual_step - 1) / actual_step
                    } else {
                        0
                    };

                    new_shape.push(slice_size);
                    new_strides.push(actual_step);
                    offset += actual_start;
                }
            }
        }

        SliceResult {
            shape: new_shape,
            strides: new_strides,
            offset,
        }
    }

    /// Generate NumPy-compatible operation mapping
    pub fn generate_operation_mapping(&self) -> HashMap<String, String> {
        let mut mapping = HashMap::new();

        // Basic operations
        mapping.insert("np.add".to_string(), "tensor.add".to_string());
        mapping.insert("np.subtract".to_string(), "tensor.sub".to_string());
        mapping.insert("np.multiply".to_string(), "tensor.mul".to_string());
        mapping.insert("np.divide".to_string(), "tensor.div".to_string());
        mapping.insert("np.power".to_string(), "tensor.pow".to_string());
        mapping.insert("np.sqrt".to_string(), "tensor.sqrt".to_string());
        mapping.insert("np.exp".to_string(), "tensor.exp".to_string());
        mapping.insert("np.log".to_string(), "tensor.log".to_string());

        // Reduction operations
        mapping.insert("np.sum".to_string(), "tensor.sum".to_string());
        mapping.insert("np.mean".to_string(), "tensor.mean".to_string());
        mapping.insert("np.std".to_string(), "tensor.std".to_string());
        mapping.insert("np.var".to_string(), "tensor.var".to_string());
        mapping.insert("np.min".to_string(), "tensor.min".to_string());
        mapping.insert("np.max".to_string(), "tensor.max".to_string());

        // Shape operations
        mapping.insert("np.reshape".to_string(), "tensor.reshape".to_string());
        mapping.insert("np.transpose".to_string(), "tensor.transpose".to_string());
        mapping.insert("np.flatten".to_string(), "tensor.flatten".to_string());
        mapping.insert("np.squeeze".to_string(), "tensor.squeeze".to_string());
        mapping.insert("np.expand_dims".to_string(), "tensor.unsqueeze".to_string());

        // Linear algebra
        mapping.insert("np.dot".to_string(), "tensor.mm".to_string());
        mapping.insert("np.matmul".to_string(), "tensor.matmul".to_string());
        mapping.insert("np.linalg.norm".to_string(), "tensor.norm".to_string());

        // Indexing and slicing
        mapping.insert("np.take".to_string(), "tensor.index_select".to_string());
        mapping.insert("np.where".to_string(), "tensor.where".to_string());

        mapping
    }

    /// Convert NumPy array to ToRSh tensor (Python integration)
    ///
    /// Mirrors the zero-copy-when-possible strategy used by
    /// [`crate::tensor::PyTensor::new`]: contiguous arrays are read out via a
    /// single slice copy, while non-contiguous arrays fall back to iterating
    /// the stride-aware `ArrayView` (which transparently handles arbitrary
    /// strides, including negative ones, without any hand-rolled offset
    /// arithmetic).
    pub fn from_numpy_array(
        &self,
        py_array: &Bound<'_, PyArrayDyn<f32>>,
    ) -> Result<Vec<f32>, String> {
        let readonly = py_array.readonly();
        if readonly.is_c_contiguous() {
            // Zero-copy path for contiguous arrays
            readonly
                .as_slice()
                .map(|slice| slice.to_vec())
                .map_err(|e| format!("Array conversion error: {}", e))
        } else {
            // Fallback for non-contiguous arrays: ArrayView iteration handles
            // arbitrary strides for free.
            Ok(readonly.as_array().iter().copied().collect())
        }
    }

    /// Convert ToRSh tensor to NumPy array (Python integration)
    ///
    /// Mirrors [`crate::tensor::PyTensor::to_numpy`]: build a flat 1-D array
    /// and reshape it to the target dimensionality, which sidesteps the
    /// `ndarray` version mismatch between this crate and the `numpy` crate's
    /// vendored version.
    pub fn to_numpy_array(
        &self,
        data: &[f32],
        shape: &[usize],
    ) -> Result<Py<PyArrayDyn<f32>>, String> {
        Python::attach(|py| {
            let array_1d = PyArray1::from_vec(py, data.to_vec());
            let array_nd = array_1d
                .reshape(NpIxDyn(shape))
                .map_err(|e| format!("Array creation error: {}", e))?;
            Ok(array_nd.unbind())
        })
    }
}

impl Default for NumpyCompat {
    fn default() -> Self {
        Self::new()
    }
}

/// NumPy-style slice specification
#[derive(Debug, Clone)]
pub enum SliceSpec {
    /// Full slice (:)
    Full,
    /// Single index
    Index(isize),
    /// Range slice (start:end:step)
    Range {
        start: Option<usize>,
        end: Option<usize>,
        step: Option<usize>,
    },
}

/// Result of array slicing operation
#[derive(Debug, Clone)]
pub struct SliceResult {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
}

/// NumPy-style universal function (ufunc) implementation
pub struct UniversalFunction {
    pub name: String,
    pub input_count: usize,
    pub output_count: usize,
    pub supports_broadcasting: bool,
    pub supports_reduction: bool,
}

impl UniversalFunction {
    /// Create a new universal function
    pub fn new(name: String, input_count: usize, output_count: usize) -> Self {
        Self {
            name,
            input_count,
            output_count,
            supports_broadcasting: true,
            supports_reduction: false,
        }
    }

    /// Apply the universal function with broadcasting
    pub fn apply_with_broadcasting(
        &self,
        inputs: &[&NumpyArrayInfo],
        compat: &NumpyCompat,
    ) -> Result<Vec<usize>, String> {
        if inputs.len() != self.input_count {
            return Err(format!(
                "Expected {} inputs, got {}",
                self.input_count,
                inputs.len()
            ));
        }

        if !self.supports_broadcasting {
            // All inputs must have the same shape
            let first_shape = &inputs[0].shape;
            for input in inputs.iter().skip(1) {
                if input.shape != *first_shape {
                    return Err("Shape mismatch for non-broadcasting function".to_string());
                }
            }
            return Ok(first_shape.clone());
        }

        // Compute broadcast shape
        let mut result_shape = inputs[0].shape.clone();
        for input in inputs.iter().skip(1) {
            if let Some(broadcast_shape) = compat.broadcast_shapes(&result_shape, &input.shape) {
                result_shape = broadcast_shape;
            } else {
                return Err("Cannot broadcast input shapes".to_string());
            }
        }

        Ok(result_shape)
    }
}

/// Common NumPy universal functions
pub struct NumpyUFuncs {
    pub add: UniversalFunction,
    pub subtract: UniversalFunction,
    pub multiply: UniversalFunction,
    pub divide: UniversalFunction,
    pub power: UniversalFunction,
    pub sqrt: UniversalFunction,
    pub exp: UniversalFunction,
    pub log: UniversalFunction,
    pub sin: UniversalFunction,
    pub cos: UniversalFunction,
    pub tan: UniversalFunction,
}

impl Default for NumpyUFuncs {
    fn default() -> Self {
        Self {
            add: UniversalFunction::new("add".to_string(), 2, 1),
            subtract: UniversalFunction::new("subtract".to_string(), 2, 1),
            multiply: UniversalFunction::new("multiply".to_string(), 2, 1),
            divide: UniversalFunction::new("divide".to_string(), 2, 1),
            power: UniversalFunction::new("power".to_string(), 2, 1),
            sqrt: UniversalFunction::new("sqrt".to_string(), 1, 1),
            exp: UniversalFunction::new("exp".to_string(), 1, 1),
            log: UniversalFunction::new("log".to_string(), 1, 1),
            sin: UniversalFunction::new("sin".to_string(), 1, 1),
            cos: UniversalFunction::new("cos".to_string(), 1, 1),
            tan: UniversalFunction::new("tan".to_string(), 1, 1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numpy_dtype_conversion() {
        assert_eq!(
            NumpyDType::from_numpy_str("float32"),
            Some(NumpyDType::Float32)
        );
        assert_eq!(NumpyDType::from_numpy_str("int64"), Some(NumpyDType::Int64));
        assert_eq!(NumpyDType::from_numpy_str("bool"), Some(NumpyDType::Bool));

        assert_eq!(NumpyDType::Float32.to_numpy_str(), "float32");
        assert_eq!(NumpyDType::Int64.to_numpy_str(), "int64");
        assert_eq!(NumpyDType::Bool.to_numpy_str(), "bool");
    }

    #[test]
    fn test_type_promotion() {
        let compat = NumpyCompat::new();

        // Test basic promotions
        assert_eq!(
            compat.promote_types(&NumpyDType::Int32, &NumpyDType::Int64),
            NumpyDType::Int64
        );
        assert_eq!(
            compat.promote_types(&NumpyDType::Float32, &NumpyDType::Float64),
            NumpyDType::Float64
        );
        assert_eq!(
            compat.promote_types(&NumpyDType::Int32, &NumpyDType::Float32),
            NumpyDType::Float32
        );
    }

    #[test]
    fn test_broadcasting() {
        let compat = NumpyCompat::new();

        // Test compatible shapes
        assert!(compat.can_broadcast(&[3, 4], &[4]));
        assert!(compat.can_broadcast(&[2, 1, 4], &[3, 4]));
        assert!(compat.can_broadcast(&[1], &[8, 4, 5]));

        // Test incompatible shapes
        assert!(!compat.can_broadcast(&[3, 4], &[3, 5]));
        assert!(!compat.can_broadcast(&[2, 4], &[3, 4]));

        // Test broadcast shape computation
        assert_eq!(compat.broadcast_shapes(&[3, 4], &[4]), Some(vec![3, 4]));
        assert_eq!(
            compat.broadcast_shapes(&[2, 1, 4], &[3, 4]),
            Some(vec![2, 3, 4])
        );
    }

    #[test]
    fn test_contiguity() {
        let compat = NumpyCompat::new();
        let dtype = NumpyDType::Float32;

        // C-contiguous array
        let shape = vec![2, 3, 4];
        let c_strides = vec![48, 16, 4]; // 3*4*4, 4*4, 4 bytes
        assert!(compat.is_c_contiguous(&shape, &c_strides, &dtype));
        assert!(!compat.is_fortran_contiguous(&shape, &c_strides, &dtype));

        // Fortran-contiguous array
        let f_strides = vec![4, 8, 24]; // 4, 2*4, 2*3*4 bytes
        assert!(!compat.is_c_contiguous(&shape, &f_strides, &dtype));
        assert!(compat.is_fortran_contiguous(&shape, &f_strides, &dtype));
    }

    #[test]
    fn test_array_slicing() {
        let compat = NumpyCompat::new();
        let shape = vec![4, 6, 8];

        // Full slice
        let slice_spec = vec![SliceSpec::Full, SliceSpec::Full, SliceSpec::Full];
        let result = compat.slice_array(&shape, &slice_spec);
        assert_eq!(result.shape, vec![4, 6, 8]);
        assert_eq!(result.offset, 0);

        // Index slice (removes dimension)
        let slice_spec = vec![SliceSpec::Index(1), SliceSpec::Full, SliceSpec::Full];
        let result = compat.slice_array(&shape, &slice_spec);
        assert_eq!(result.shape, vec![6, 8]);
        assert_eq!(result.offset, 1);

        // Range slice
        let slice_spec = vec![
            SliceSpec::Range {
                start: Some(1),
                end: Some(3),
                step: Some(1),
            },
            SliceSpec::Full,
            SliceSpec::Range {
                start: Some(0),
                end: Some(8),
                step: Some(2),
            },
        ];
        let result = compat.slice_array(&shape, &slice_spec);
        assert_eq!(result.shape, vec![2, 6, 4]); // (3-1)/1, 6, 8/2
    }

    #[test]
    fn test_universal_function() {
        let compat = NumpyCompat::new();
        let ufunc = UniversalFunction::new("add".to_string(), 2, 1);

        let array1 = NumpyArrayInfo {
            shape: vec![3, 4],
            strides: vec![16, 4],
            dtype: NumpyDType::Float32,
            fortran_order: false,
            contiguous: true,
            writeable: true,
            aligned: true,
        };

        let array2 = NumpyArrayInfo {
            shape: vec![4],
            strides: vec![4],
            dtype: NumpyDType::Float32,
            fortran_order: false,
            contiguous: true,
            writeable: true,
            aligned: true,
        };

        let result = ufunc.apply_with_broadcasting(&[&array1, &array2], &compat);
        assert_eq!(result.unwrap(), vec![3, 4]);
    }

    #[test]
    fn test_operation_mapping() {
        let compat = NumpyCompat::new();
        let mapping = compat.generate_operation_mapping();

        assert_eq!(mapping.get("np.add"), Some(&"tensor.add".to_string()));
        assert_eq!(mapping.get("np.matmul"), Some(&"tensor.matmul".to_string()));
        assert_eq!(
            mapping.get("np.reshape"),
            Some(&"tensor.reshape".to_string())
        );
    }

    /// Compare two `f32` slices elementwise within a small tolerance, rather
    /// than via `assert_eq!` -- avoids relying on bit-exact float equality
    /// after a round trip through Python/NumPy.
    fn assert_f32_slice_approx_eq(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-5, "{:?} vs {:?}", actual, expected);
        }
    }

    #[test]
    fn test_numpy_array_round_trip_contiguous() {
        Python::initialize();
        let compat = NumpyCompat::new();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2usize, 3usize];

        let py_array = compat
            .to_numpy_array(&data, &shape)
            .expect("to_numpy_array should succeed for a real, non-empty tensor");

        Python::attach(|py| {
            let bound = py_array.bind(py);
            assert_eq!(bound.readonly().shape(), shape.as_slice());

            let round_tripped = compat
                .from_numpy_array(bound)
                .expect("from_numpy_array should succeed on a contiguous array");
            assert_f32_slice_approx_eq(&round_tripped, &data);
        });
    }

    #[test]
    fn test_numpy_array_round_trip_non_contiguous() {
        Python::initialize();
        let compat = NumpyCompat::new();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2usize, 3usize];

        let py_array = compat
            .to_numpy_array(&data, &shape)
            .expect("to_numpy_array should succeed for a real, non-empty tensor");

        Python::attach(|py| {
            let bound = py_array.bind(py);

            // `.T` yields a non-contiguous view over the same underlying
            // buffer (Fortran-ordered relative to the original), exercising
            // the strided `ArrayView` fallback path in `from_numpy_array`.
            let transposed = bound
                .getattr("T")
                .expect("transpose attribute access should succeed");
            let transposed = transposed
                .cast::<PyArrayDyn<f32>>()
                .expect("transposed view should still be a float32 array");
            assert!(!transposed.readonly().is_c_contiguous());

            let round_tripped = compat
                .from_numpy_array(transposed)
                .expect("from_numpy_array should succeed on non-contiguous input");
            // Logical (C-order) traversal of the transpose of
            // [[1, 2, 3], [4, 5, 6]] is [1, 4, 2, 5, 3, 6].
            assert_f32_slice_approx_eq(&round_tripped, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        });
    }

    #[test]
    fn test_numpy_array_round_trip_empty_data_still_errors_cleanly() {
        // A dtype mismatch (e.g. handing in a non-float32 array) must not
        // silently succeed with wrong/empty data the way the old stub did;
        // it should fail the `.cast::<PyArrayDyn<f32>>()` step instead.
        Python::initialize();
        Python::attach(|py| {
            let int_array = PyArray1::<i64>::from_vec(py, vec![1, 2, 3]);
            let cast_result = int_array.cast::<PyArrayDyn<f32>>();
            assert!(
                cast_result.is_err(),
                "an int64 array must not be castable to a float32 array view"
            );
        });
    }
}
