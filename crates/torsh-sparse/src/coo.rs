//! COO (Coordinate) sparse tensor format

use crate::{CsrTensor, SparseFormat, SparseTensor, TorshResult};
use torsh_core::{device::DeviceType, DType, Shape, TorshError};
use torsh_tensor::Tensor;

/// COO (Coordinate) format sparse tensor
#[derive(Debug, Clone)]
pub struct CooTensor {
    /// Row indices
    row_indices: Vec<usize>,
    /// Column indices  
    col_indices: Vec<usize>,
    /// Non-zero values
    values: Vec<f32>,
    /// Shape of the tensor
    shape: Shape,
    /// Data type
    dtype: DType,
    /// Device
    device: DeviceType,
    /// Whether the entries are sorted by (row, col) and free of duplicates
    is_coalesced: bool,
}

impl CooTensor {
    /// Create a new COO tensor
    pub fn new(
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<f32>,
        shape: Shape,
    ) -> TorshResult<Self> {
        // Validate inputs
        if row_indices.len() != col_indices.len() || row_indices.len() != values.len() {
            return Err(TorshError::InvalidArgument(
                "Indices and values must have the same length".to_string(),
            ));
        }

        if shape.ndim() != 2 {
            return Err(TorshError::InvalidArgument(
                "COO format currently only supports 2D tensors".to_string(),
            ));
        }

        // Validate indices are within bounds
        let (rows, cols) = (shape.dims()[0], shape.dims()[1]);
        for (&row, &col) in row_indices.iter().zip(&col_indices) {
            if row >= rows || col >= cols {
                return Err(TorshError::InvalidArgument(format!(
                    "Index ({row}, {col}) out of bounds for shape {shape:?}"
                )));
            }
        }

        let is_coalesced = Self::entries_are_coalesced(&row_indices, &col_indices);

        Ok(Self {
            row_indices,
            col_indices,
            values,
            shape,
            dtype: DType::F32,
            device: DeviceType::Cpu,
            is_coalesced,
        })
    }

    /// True when the entries are strictly increasing in (row, col) order, which
    /// implies there are no duplicate coordinates.
    fn entries_are_coalesced(row_indices: &[usize], col_indices: &[usize]) -> bool {
        row_indices
            .windows(2)
            .zip(col_indices.windows(2))
            .all(|(rows, cols)| (rows[0], cols[0]) < (rows[1], cols[1]))
    }

    /// Whether this tensor is coalesced: entries sorted by `(row, col)` with no
    /// duplicate coordinates.
    ///
    /// PyTorch semantics: duplicate coordinates are *summed*, so a tensor that
    /// is not coalesced still denotes the sum of its entries.
    pub fn is_coalesced(&self) -> bool {
        self.is_coalesced
    }

    /// Sort the entries by `(row, col)`, sum duplicate coordinates and drop
    /// entries that become exactly zero.
    ///
    /// This is what makes `matvec`, `to_dense` and the CSR/CSC conversions agree
    /// on the value of a tensor built with repeated coordinates.
    pub fn coalesce(&mut self) {
        if self.is_coalesced {
            return;
        }

        let mut entries: Vec<(usize, usize, f32)> = self.triplets();
        entries.sort_by_key(|&(row, col, _)| (row, col));

        let mut row_indices = Vec::with_capacity(entries.len());
        let mut col_indices = Vec::with_capacity(entries.len());
        let mut values = Vec::with_capacity(entries.len());

        for (row, col, value) in entries {
            if let (Some(&last_row), Some(&last_col)) = (row_indices.last(), col_indices.last()) {
                if last_row == row && last_col == col {
                    if let Some(last) = values.last_mut() {
                        *last += value;
                    }
                    continue;
                }
            }
            row_indices.push(row);
            col_indices.push(col);
            values.push(value);
        }

        // Drop coordinates whose accumulated value cancelled out.
        let mut kept_rows = Vec::with_capacity(values.len());
        let mut kept_cols = Vec::with_capacity(values.len());
        let mut kept_values = Vec::with_capacity(values.len());
        for i in 0..values.len() {
            if values[i] != 0.0 {
                kept_rows.push(row_indices[i]);
                kept_cols.push(col_indices[i]);
                kept_values.push(values[i]);
            }
        }

        self.row_indices = kept_rows;
        self.col_indices = kept_cols;
        self.values = kept_values;
        self.is_coalesced = true;
    }

    /// Return a coalesced copy, leaving `self` untouched.
    pub fn coalesced(&self) -> Self {
        let mut result = self.clone();
        result.coalesce();
        result
    }

    /// Create from dense tensor
    pub fn from_dense(dense: &Tensor, threshold: f32) -> TorshResult<Self> {
        if dense.shape().ndim() != 2 {
            return Err(TorshError::InvalidArgument(
                "COO format currently only supports 2D tensors".to_string(),
            ));
        }

        let shape = dense.shape().clone();
        let (rows, cols) = (shape.dims()[0], shape.dims()[1]);

        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        // Extract non-zero values
        for i in 0..rows {
            for j in 0..cols {
                let value = dense.get(&[i, j])?;
                if value.abs() > threshold {
                    row_indices.push(i);
                    col_indices.push(j);
                    values.push(value);
                }
            }
        }

        Self::new(row_indices, col_indices, values, shape)
    }

    /// Create an empty COO tensor with given shape
    pub fn empty(shape: Shape, dtype: DType) -> TorshResult<Self> {
        if shape.ndim() != 2 {
            return Err(TorshError::InvalidArgument(
                "COO format currently only supports 2D tensors".to_string(),
            ));
        }

        Ok(Self {
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
            shape,
            dtype,
            device: DeviceType::Cpu,
            is_coalesced: true,
        })
    }

    /// Create COO tensor from triplets (row, col, value)
    pub fn from_triplets(
        triplets: Vec<(usize, usize, f32)>,
        shape: (usize, usize),
    ) -> TorshResult<Self> {
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for (row, col, value) in triplets {
            row_indices.push(row);
            col_indices.push(col);
            values.push(value);
        }

        let shape = Shape::new(vec![shape.0, shape.1]);
        Self::new(row_indices, col_indices, values, shape)
    }

    /// Insert a value at the given position
    pub fn insert(&mut self, row: usize, col: usize, value: f32) -> TorshResult<()> {
        if row >= self.shape.dims()[0] || col >= self.shape.dims()[1] {
            return Err(TorshError::InvalidArgument(format!(
                "Index ({}, {}) out of bounds for shape {:?}",
                row,
                col,
                self.shape.dims()
            )));
        }

        // Appending keeps the coalesced property only if the new coordinate is
        // strictly after the current last one.
        let still_coalesced = match (self.row_indices.last(), self.col_indices.last()) {
            (Some(&last_row), Some(&last_col)) => {
                self.is_coalesced && (last_row, last_col) < (row, col)
            }
            _ => true,
        };

        self.row_indices.push(row);
        self.col_indices.push(col);
        self.values.push(value);
        self.is_coalesced = still_coalesced;

        Ok(())
    }

    /// Get triplets (row, col, value)
    pub fn triplets(&self) -> Vec<(usize, usize, f32)> {
        self.row_indices
            .iter()
            .zip(&self.col_indices)
            .zip(&self.values)
            .map(|((&row, &col), &val)| (row, col, val))
            .collect()
    }

    /// Get row indices
    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    /// Get column indices
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    /// Get values
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Sort by row then column
    pub fn sort_indices(&mut self) {
        // Create index permutation
        let mut indices: Vec<_> = (0..self.nnz()).collect();
        indices.sort_by_key(|&i| (self.row_indices[i], self.col_indices[i]));

        // Apply permutation
        let row_indices: Vec<_> = indices.iter().map(|&i| self.row_indices[i]).collect();
        let col_indices: Vec<_> = indices.iter().map(|&i| self.col_indices[i]).collect();
        let values: Vec<_> = indices.iter().map(|&i| self.values[i]).collect();

        self.row_indices = row_indices;
        self.col_indices = col_indices;
        self.values = values;
        self.is_coalesced = Self::entries_are_coalesced(&self.row_indices, &self.col_indices);
    }

    /// Transpose the sparse tensor
    pub fn transpose(&self) -> Self {
        let row_indices = self.col_indices.clone();
        let col_indices = self.row_indices.clone();
        let is_coalesced = Self::entries_are_coalesced(&row_indices, &col_indices);
        Self {
            row_indices,
            col_indices,
            values: self.values.clone(),
            shape: Shape::new(vec![self.shape.dims()[1], self.shape.dims()[0]]),
            dtype: self.dtype,
            device: self.device,
            is_coalesced,
        }
    }
}

impl SparseTensor for CooTensor {
    fn format(&self) -> SparseFormat {
        SparseFormat::Coo
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> DeviceType {
        self.device
    }

    fn nnz(&self) -> usize {
        self.values.len()
    }

    fn to_dense(&self) -> TorshResult<Tensor> {
        // Duplicate coordinates are summed (PyTorch semantics), which is what
        // `CsrTensor::matvec` and the sparse kernels do as well.
        let (rows, cols) = (self.shape.dims()[0], self.shape.dims()[1]);
        let mut data = vec![0.0f32; rows * cols];

        for i in 0..self.nnz() {
            data[self.row_indices[i] * cols + self.col_indices[i]] += self.values[i];
        }

        Tensor::from_data(data, vec![rows, cols], self.device)
    }

    fn to_coo(&self) -> TorshResult<CooTensor> {
        Ok(self.clone())
    }

    fn to_csr(&self) -> TorshResult<CsrTensor> {
        CsrTensor::from_coo(self)
    }

    fn to_csc(&self) -> TorshResult<crate::CscTensor> {
        crate::CscTensor::from_coo(self)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coo_creation() {
        let row_indices = vec![0, 1, 2];
        let col_indices = vec![1, 2, 0];
        let values = vec![1.0, 2.0, 3.0];
        let shape = Shape::new(vec![3, 3]);

        let coo = CooTensor::new(row_indices, col_indices, values, shape).unwrap();
        assert_eq!(coo.nnz(), 3);
        assert_eq!(coo.shape().dims(), &[3, 3]);
    }

    #[test]
    fn test_coo_to_dense() {
        let row_indices = vec![0, 1, 2];
        let col_indices = vec![1, 2, 0];
        let values = vec![1.0, 2.0, 3.0];
        let shape = Shape::new(vec![3, 3]);

        let coo = CooTensor::new(row_indices, col_indices, values, shape).unwrap();
        let dense = coo.to_dense().unwrap();

        assert_eq!(dense.get(&[0, 1]).unwrap(), 1.0);
        assert_eq!(dense.get(&[1, 2]).unwrap(), 2.0);
        assert_eq!(dense.get(&[2, 0]).unwrap(), 3.0);
    }
}
