//! Tensor stacking utilities

use torsh_core::{
    dtype::TensorElement,
    error::{Result, TorshError},
};
use torsh_tensor::Tensor;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ✅ SciRS2 POLICY: Use scirs2_core::parallel_ops instead of rayon::prelude
#[cfg(feature = "std")]
use scirs2_core::parallel_ops::*;

/// Consolidated tensor stacking utility that reduces code duplication
pub struct TensorStacker {
    use_parallel: bool,
    parallel_threshold: usize,
    memory_mapped: bool,
}

impl Default for TensorStacker {
    fn default() -> Self {
        Self {
            use_parallel: cfg!(feature = "std"),
            parallel_threshold: 1000,
            memory_mapped: false,
        }
    }
}

impl TensorStacker {
    /// Create a new tensor stacker with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable/disable parallel processing
    pub fn with_parallel(mut self, enabled: bool) -> Self {
        self.use_parallel = enabled && cfg!(feature = "std");
        self
    }

    /// Set threshold for parallel processing
    pub fn parallel_threshold(mut self, threshold: usize) -> Self {
        self.parallel_threshold = threshold;
        self
    }

    /// Enable memory-mapped stacking for very large batches
    pub fn with_memory_mapping(mut self, enabled: bool) -> Self {
        self.memory_mapped = enabled && cfg!(all(feature = "std", feature = "mmap-support"));
        self
    }

    /// Stack tensors along the specified dimension
    pub fn stack<T: TensorElement + Copy>(
        &self,
        tensors: &[Tensor<T>],
        dim: usize,
    ) -> Result<Tensor<T>> {
        if tensors.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Cannot stack empty tensor list".to_string(),
            ));
        }

        // Validate shapes
        self.validate_shapes(tensors)?;

        // Choose stacking strategy based on configuration and batch size
        if self.memory_mapped && tensors.len() > 100 {
            self.stack_with_mmap(tensors, dim)
        } else if self.use_parallel && self.should_use_parallel(tensors) {
            self.stack_parallel(tensors, dim)
        } else {
            self.stack_sequential(tensors, dim)
        }
    }

    /// Check if all tensors have the same shape
    fn validate_shapes<T: TensorElement>(&self, tensors: &[Tensor<T>]) -> Result<()> {
        let first_shape = tensors[0].shape();
        for tensor in tensors[1..].iter() {
            if tensor.shape() != first_shape {
                return Err(TorshError::ShapeMismatch {
                    expected: first_shape.dims().to_vec(),
                    got: tensor.shape().dims().to_vec(),
                });
            }
        }
        Ok(())
    }

    /// Determine if parallel processing should be used
    fn should_use_parallel<T: TensorElement>(&self, tensors: &[Tensor<T>]) -> bool {
        tensors.len() > 4 && tensors[0].numel() > self.parallel_threshold
    }

    /// Create new shape with additional dimension
    fn create_new_shape(
        &self,
        original_dims: &[usize],
        batch_size: usize,
        dim: usize,
    ) -> Vec<usize> {
        let mut new_dims = Vec::with_capacity(original_dims.len() + 1);

        if dim == 0 {
            new_dims.push(batch_size);
            new_dims.extend_from_slice(original_dims);
        } else {
            new_dims.extend_from_slice(&original_dims[..dim.min(original_dims.len())]);
            new_dims.push(batch_size);
            if dim < original_dims.len() {
                new_dims.extend_from_slice(&original_dims[dim..]);
            }
        }

        new_dims
    }

    /// Split each source tensor's flat buffer into an `outer_size` (product of
    /// dims before `dim`) by `inner_size` (product of dims from `dim` onward)
    /// view. Stacking a new batch axis at `dim` means each of the
    /// `outer_size` blocks gets one `inner_size`-long chunk copied in from
    /// every source tensor in turn, which is exactly `dim == 0` (`outer_size
    /// == 1`) generalized to any `dim`.
    fn outer_inner_split(&self, dims: &[usize], dim: usize) -> (usize, usize) {
        let dim = dim.min(dims.len());
        let outer_size: usize = dims[..dim].iter().product();
        let inner_size: usize = dims[dim..].iter().product();
        (outer_size, inner_size)
    }

    /// Sequential tensor stacking
    fn stack_sequential<T: TensorElement + Copy>(
        &self,
        tensors: &[Tensor<T>],
        dim: usize,
    ) -> Result<Tensor<T>> {
        let original_dims = tensors[0].shape().dims().to_vec();
        let new_dims = self.create_new_shape(&original_dims, tensors.len(), dim);
        let (outer_size, inner_size) = self.outer_inner_split(&original_dims, dim);
        let total_elements = new_dims.iter().product::<usize>();

        // Gather every source tensor's data BEFORE allocating the output
        // buffer, so a mid-batch conversion error surfaces via `?` before any
        // partially-filled buffer exists (avoids the set_len-before-fully-
        // initialized UB the old implementation had).
        let all_data: Vec<Vec<T>> = tensors
            .iter()
            .map(|tensor| tensor.to_vec())
            .collect::<Result<Vec<_>>>()?;

        let mut new_data = Vec::with_capacity(total_elements);
        for outer in 0..outer_size {
            let offset = outer * inner_size;
            for data in &all_data {
                new_data.extend_from_slice(&data[offset..offset + inner_size]);
            }
        }

        torsh_tensor::Tensor::from_data(new_data, new_dims, tensors[0].device())
    }

    /// Parallel tensor stacking
    #[cfg(feature = "std")]
    fn stack_parallel<T: TensorElement + Copy>(
        &self,
        tensors: &[Tensor<T>],
        dim: usize,
    ) -> Result<Tensor<T>> {
        let original_dims = tensors[0].shape().dims().to_vec();
        let new_dims = self.create_new_shape(&original_dims, tensors.len(), dim);
        let (outer_size, inner_size) = self.outer_inner_split(&original_dims, dim);

        // Parallel data collection - gathered fully before any output buffer
        // exists, so a conversion error surfaces via `?` before any
        // partially-filled buffer would need to be dropped.
        let parallel_data: std::result::Result<Vec<Vec<T>>, TorshError> =
            tensors.par_iter().map(|tensor| tensor.to_vec()).collect();
        let all_data = parallel_data?;

        // Stack each of the `outer_size` interleaved blocks in parallel, then
        // flatten the per-block chunks back together in order.
        let chunks: Vec<Vec<T>> = (0..outer_size)
            .into_par_iter()
            .map(|outer| {
                let offset = outer * inner_size;
                let mut chunk = Vec::with_capacity(all_data.len() * inner_size);
                for data in &all_data {
                    chunk.extend_from_slice(&data[offset..offset + inner_size]);
                }
                chunk
            })
            .collect();

        let new_data: Vec<T> = chunks.into_iter().flatten().collect();
        torsh_tensor::Tensor::from_data(new_data, new_dims, tensors[0].device())
    }

    /// Fallback for no_std
    #[cfg(not(feature = "std"))]
    fn stack_parallel<T: TensorElement + Copy>(
        &self,
        tensors: &[Tensor<T>],
        dim: usize,
    ) -> Result<Tensor<T>> {
        self.stack_sequential(tensors, dim)
    }

    /// Memory-mapped stacking for very large batches
    #[cfg(all(feature = "std", feature = "mmap-support"))]
    fn stack_with_mmap<T: TensorElement + Copy>(
        &self,
        tensors: &[Tensor<T>],
        dim: usize,
    ) -> Result<Tensor<T>> {
        // Placeholder implementation - in practice would use memory mapping
        // For now, fallback to parallel stacking
        self.stack_parallel(tensors, dim)
    }

    /// Fallback when mmap is not available
    #[cfg(not(all(feature = "std", feature = "mmap-support")))]
    fn stack_with_mmap<T: TensorElement + Copy>(
        &self,
        tensors: &[Tensor<T>],
        dim: usize,
    ) -> Result<Tensor<T>> {
        if cfg!(feature = "std") {
            self.stack_parallel(tensors, dim)
        } else {
            self.stack_sequential(tensors, dim)
        }
    }
}
