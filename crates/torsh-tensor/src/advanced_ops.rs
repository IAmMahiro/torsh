//! Advanced tensor operations including reductions, linear algebra, and backend integration
//!
//! This module provides high-level tensor operations including reductions, linear algebra
//! operations, SciRS2 backend integration, and advanced data manipulation functions.
//!
//! # Features
//!
//! - **Reductions**: max, norm, sum, mean operations
//! - **Linear algebra**: Matrix multiplication and vector operations
//! - **SciRS2 integration**: Optimized backend operations for performance
//! - **Activation functions**: ReLU, sigmoid, tanh through SciRS2 backend
//! - **Functional programming**: Apply operations and data transformations
//! - **Memory management**: Copy-on-write semantics and unique data operations

use std::sync::Arc;
use torsh_core::{
    device::DeviceType,
    dtype::{FloatElement, TensorElement},
    error::{Result, TorshError},
};

use crate::{
    core_ops::{Tensor, UnaryKind},
    storage::TensorStorage,
};

/// Element count above which a full-tensor sum is folded in parallel chunks.
const PARALLEL_SUM_THRESHOLD: usize = 65_536;

/// Element count above which the `f32` sum uses the SIMD reduction.
#[cfg(feature = "simd")]
const SIMD_SUM_THRESHOLD: usize = 1_024;

/// Sum a contiguous slice, dispatching to SIMD (`f32`) and to threaded chunks
/// for large inputs, with a scalar fold as the fallback.
fn sum_slice<T>(data: &[T]) -> T
where
    T: TensorElement + Copy + std::ops::Add<Output = T> + num_traits::Zero,
{
    let zero = <T as num_traits::Zero>::zero();

    #[cfg(feature = "simd")]
    {
        if data.len() >= SIMD_SUM_THRESHOLD
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        {
            use scirs2_core::simd_ops::SimdUnifiedOps;
            // Safety: TypeId confirmed T == f32, so the slices have identical layout.
            let as_f32: &[f32] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len()) };
            let total =
                <f32 as SimdUnifiedOps>::simd_sum(&scirs2_core::ndarray::ArrayView1::from(as_f32));
            // f32 -> f64 -> T is exact for T == f32; fall through if a type ever
            // reports f32's TypeId without being convertible.
            if let Some(value) = <T as TensorElement>::from_f64(f64::from(total)) {
                return value;
            }
        }
    }

    #[cfg(feature = "parallel")]
    {
        if data.len() >= PARALLEL_SUM_THRESHOLD {
            use scirs2_core::parallel_ops::*;
            return data
                .par_chunks(16_384)
                .map(|chunk| chunk.iter().fold(zero, |acc, &x| acc + x))
                .reduce(|| zero, |a, b| a + b);
        }
    }

    data.iter().fold(zero, |acc, &x| acc + x)
}

// Float-specific operations
impl<T: FloatElement + Copy> Tensor<T> {
    /// Create a 0-dimensional tensor (scalar) from a single value
    pub fn scalar(value: T) -> Result<Self> {
        Self::from_data(vec![value], vec![], DeviceType::Cpu)
    }

    /// Convert tensor to ndarray
    pub fn as_ndarray(&self) -> Result<scirs2_core::ndarray::ArrayD<T>> {
        use scirs2_core::ndarray::ArrayD;
        let data = self.data()?;
        let shape_obj = self.shape().clone();
        let shape = shape_obj.dims();
        ArrayD::from_shape_vec(shape, data.to_vec())
            .map_err(|e| TorshError::InvalidShape(format!("ndarray conversion failed: {}", e)))
    }

    /// Create tensor from ndarray
    pub fn from_ndarray(
        array: scirs2_core::ndarray::ArrayD<T>,
        device: DeviceType,
    ) -> Result<Self> {
        let shape = array.shape().to_vec();
        let (data, _offset) = array.into_raw_vec_and_offset();
        Self::from_data(data, shape, device)
    }
}

/// Boolean reduction operations for tensors
impl<T: TensorElement + Copy> Tensor<T>
where
    T: PartialEq + num_traits::Zero,
{
    /// Check if all elements are non-zero (true)
    pub fn all(&self) -> Result<Tensor<bool>> {
        let data = self.to_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let all_true = data.iter().all(|&x| x != zero);
        Tensor::from_data(vec![all_true], vec![], self.device())
    }

    /// Check if any element is non-zero (true)
    pub fn any(&self) -> Result<Tensor<bool>> {
        let data = self.to_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let any_true = data.iter().any(|&x| x != zero);
        Tensor::from_data(vec![any_true], vec![], self.device())
    }

    /// Check if all elements along dimension are non-zero (true)
    pub fn all_dim(&self, dim: i32, keepdim: bool) -> Result<Tensor<bool>> {
        let shape_binding = self.shape();
        let input_shape = shape_binding.dims();

        let normalized_dim = if dim < 0 {
            (input_shape.len() as i32 + dim) as usize
        } else {
            dim as usize
        };

        if normalized_dim >= input_shape.len() {
            return Err(torsh_core::error::TorshError::InvalidDimension {
                dim: normalized_dim,
                ndim: input_shape.len(),
            });
        }

        let data = self.data()?;
        let zero = <T as num_traits::Zero>::zero();

        let outer_size: usize = input_shape[..normalized_dim].iter().product();
        let dim_size = input_shape[normalized_dim];
        let inner_size: usize = input_shape[normalized_dim + 1..].iter().product();

        let output_size = outer_size * inner_size;
        let mut result_data = vec![true; output_size];

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let all_nonzero = (0..dim_size).all(|d| {
                    let idx = outer * dim_size * inner_size + d * inner_size + inner;
                    data[idx] != zero
                });
                let out_idx = outer * inner_size + inner;
                result_data[out_idx] = all_nonzero;
            }
        }

        let mut output_shape = input_shape.to_vec();
        if keepdim {
            output_shape[normalized_dim] = 1;
        } else {
            output_shape.remove(normalized_dim);
        }

        Tensor::<bool>::from_data(result_data, output_shape, self.device())
    }

    /// Check if any element along dimension is non-zero (true)
    pub fn any_dim(&self, dim: i32, keepdim: bool) -> Result<Tensor<bool>> {
        let shape_binding = self.shape();
        let input_shape = shape_binding.dims();

        let normalized_dim = if dim < 0 {
            (input_shape.len() as i32 + dim) as usize
        } else {
            dim as usize
        };

        if normalized_dim >= input_shape.len() {
            return Err(torsh_core::error::TorshError::InvalidDimension {
                dim: normalized_dim,
                ndim: input_shape.len(),
            });
        }

        let data = self.data()?;
        let zero = <T as num_traits::Zero>::zero();

        let outer_size: usize = input_shape[..normalized_dim].iter().product();
        let dim_size = input_shape[normalized_dim];
        let inner_size: usize = input_shape[normalized_dim + 1..].iter().product();

        let output_size = outer_size * inner_size;
        let mut result_data = vec![false; output_size];

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let any_nonzero = (0..dim_size).any(|d| {
                    let idx = outer * dim_size * inner_size + d * inner_size + inner;
                    data[idx] != zero
                });
                let out_idx = outer * inner_size + inner;
                result_data[out_idx] = any_nonzero;
            }
        }

        let mut output_shape = input_shape.to_vec();
        if keepdim {
            output_shape[normalized_dim] = 1;
        } else {
            output_shape.remove(normalized_dim);
        }

        Tensor::<bool>::from_data(result_data, output_shape, self.device())
    }
}

// General tensor operations
impl<T: TensorElement + Copy> Tensor<T> {
    /// Compute sum of all elements
    ///
    /// Reads the storage in place (no intermediate copy) and, for large
    /// tensors, folds chunks in parallel through `scirs2_core::parallel_ops`.
    pub fn sum(&self) -> Result<Self>
    where
        T: std::ops::Add<Output = T> + num_traits::Zero,
    {
        let sum_value = self.with_contiguous_data(|data| Ok(sum_slice(data)))?;
        let mut result = Tensor::from_data(vec![sum_value], vec![], self.device())?;

        // Record the sum operation for autograd: d(sum)/dx_i = 1 for every element.
        if crate::should_record_grad(self.requires_grad) {
            result.requires_grad = true;
            result.operation = crate::core_ops::Operation::Sum {
                input: Arc::new(self.clone()),
            };
        }

        Ok(result)
    }

    /// Global minimum of the tensor (see [`Tensor::amin`] for the dimension-aware
    /// form, and [`Tensor::min_dim`] for a single axis).
    pub fn min(&self) -> Result<Self>
    where
        T: std::cmp::PartialOrd + Copy,
    {
        let min_val = self.with_contiguous_data(|data| {
            let mut iter = data.iter().copied();
            let first = iter.next().ok_or_else(|| {
                TorshError::InvalidOperation("Cannot compute min of empty tensor".to_string())
            })?;
            Ok(iter.fold(first, |acc, x| if x < acc { x } else { acc }))
        })?;
        Self::from_data(vec![min_val], vec![], self.device)
    }

    /// Transpose operation (2D tensor)
    ///
    /// Produces exactly what [`Tensor::transpose(0, 1)`](Tensor::transpose)
    /// produces, including its autograd node: a transpose is the permutation
    /// that swaps the two axes, recorded as
    /// [`ViewKind::Permute`](crate::core_ops::ViewKind::Permute) so the backward
    /// pass permutes the gradient straight back. Building the result with a
    /// private `from_data` copy instead — which is what this used to do — made
    /// `t()` an honestly-detached leaf: `x.t()` came back with
    /// `requires_grad == false` even from a `requires_grad` input, so any loss
    /// routed through a transpose silently lost its gradient path.
    ///
    /// The rank check is kept: unlike `transpose`, `t()` is defined only for
    /// matrices and still rejects any other rank.
    ///
    /// # Layout
    /// The result is a *strided view* of `self`, exactly like `transpose`'s (no
    /// data is copied). Callers that need a packed buffer call `.contiguous()`;
    /// callers that read it through [`Tensor::to_vec`] or
    /// `with_contiguous_data` see the transposed order either way.
    pub fn t(&self) -> Result<Self>
    where
        T: Copy + num_traits::Zero,
    {
        let shape = self.shape();
        let dims = shape.dims();

        if dims.len() != 2 {
            return Err(TorshError::InvalidOperation(
                "Transpose operation only supported for 2D tensors".to_string(),
            ));
        }

        self.transpose(0, 1)
    }

    /// Check if two tensors share the same underlying storage
    pub fn shares_storage(&self, other: &Self) -> bool {
        // For storage abstraction, we need to check the underlying storage
        match (&self.storage, &other.storage) {
            (TensorStorage::InMemory(a), TensorStorage::InMemory(b)) => Arc::ptr_eq(a, b),
            (TensorStorage::MemoryMapped(a), TensorStorage::MemoryMapped(b)) => Arc::ptr_eq(a, b),
            // Without this arm the catch-all below would report two aliases of
            // one device allocation as unshared, and `make_unique` would skip a
            // copy it needs.
            #[cfg(feature = "gpu")]
            (TensorStorage::Device { buffer: a, .. }, TensorStorage::Device { buffer: b, .. }) => {
                Arc::ptr_eq(a, b)
            }
            _ => false,
        }
    }

    /// Get data as a vector (backward compatibility method)
    pub fn data(&self) -> Result<Vec<T>>
    where
        T: Copy,
    {
        self.to_vec()
    }

    /// Run `f` against a contiguous, view-ordered slice of this tensor's data.
    ///
    /// Base tensors hand the storage slice straight to the closure (zero copy).
    /// Strided views are materialised in view order first, so callers can always
    /// index the slice with plain row-major arithmetic.
    pub(crate) fn with_contiguous_data<R, F>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&[T]) -> Result<R>,
        T: Copy,
    {
        let numel = self.numel();
        if self.is_view() || self.strides.is_some() || self.storage_offset != 0 {
            let data = self.to_vec()?;
            return f(&data);
        }

        self.storage.with_slice(|slice| {
            if slice.len() < numel {
                return Err(TorshError::InvalidOperation(format!(
                    "storage holds {} elements but the tensor shape needs {}",
                    slice.len(),
                    numel
                )));
            }
            f(&slice[..numel])
        })
    }

    /// Materialise this tensor as a contiguous, uniquely owned, *mutable* base
    /// tensor.
    ///
    /// This is the single entry point used by every in-place operation. It
    /// compacts strided views into view order **and** resets the view metadata
    /// (`strides`, `storage_offset`, `base_tensor`), which is what keeps later
    /// reads from applying the permutation a second time.
    ///
    /// Note that a view detaches from its base here: subsequent writes are no
    /// longer visible through the original tensor.
    pub(crate) fn materialize_contiguous(&mut self) -> Result<()>
    where
        T: Copy,
    {
        let data_vec = self.to_vec()?;
        self.storage = Self::mutable_storage(data_vec)?;
        self.strides = None;
        self.storage_offset = 0;
        self.base_tensor = None;
        Ok(())
    }

    /// Build storage that supports `with_slice_mut` (i.e. never `SimdOptimized`).
    fn mutable_storage(data: Vec<T>) -> Result<TensorStorage<T>> {
        #[cfg(feature = "simd")]
        {
            // Aligned storage keeps SIMD-friendly alignment while staying mutable.
            const MEMORY_MAPPING_BYTES: usize = 1024 * 1024 * 1024;
            if data.len() * std::mem::size_of::<T>() < MEMORY_MAPPING_BYTES {
                return TensorStorage::aligned(data);
            }
        }
        TensorStorage::create_optimal(data)
    }

    /// Apply a function to all elements in-place using direct storage access
    pub fn data_mut_apply<F>(&mut self, mut func: F) -> Result<()>
    where
        F: FnMut(&mut T),
        T: Copy,
    {
        self.prepare_for_inplace()?;

        match &self.storage {
            TensorStorage::MemoryMapped(_) => {
                // Memory-mapped storage has no mutable slice access: read, modify, write back.
                let mut new_data = self.to_vec()?;
                for item in new_data.iter_mut() {
                    func(item);
                }
                self.storage = TensorStorage::create_optimal(new_data)?;
                Ok(())
            }
            _ => {
                let numel = self.numel();
                self.storage.with_slice_mut(|slice| {
                    for item in slice.iter_mut().take(numel) {
                        func(item);
                    }
                    Ok(())
                })
            }
        }
    }

    /// Clone the tensor with independent data (deep copy)
    pub fn clone_data(&self) -> Self
    where
        T: Copy,
    {
        let data = self
            .to_vec()
            .expect("tensor to vec conversion should succeed");
        Self::from_data(data, self.shape().dims().to_vec(), self.device)
            .expect("tensor creation should succeed")
    }

    /// Ensure tensor has unique data (copy-on-write semantics)
    ///
    /// Strided views are compacted *and* detached (see
    /// [`Tensor::materialize_contiguous`]); base tensors are copied only when
    /// their storage is actually shared — including `SimdOptimized` storage,
    /// which implements copy-on-write internally and therefore needs no eager
    /// promotion when this tensor is its only owner.
    pub fn make_unique(&mut self) -> Result<()> {
        if self.is_view() || self.strides.is_some() || self.storage_offset != 0 {
            return self.materialize_contiguous();
        }

        match &self.storage {
            TensorStorage::InMemory(data) => {
                if Arc::strong_count(data) > 1 {
                    let data_vec = self.to_vec()?;
                    self.storage = Self::mutable_storage(data_vec)?;
                }
            }
            TensorStorage::MemoryMapped(storage) => {
                if Arc::strong_count(storage) > 1 {
                    let data_vec = self.to_vec()?;
                    self.storage = TensorStorage::create_optimal(data_vec)?;
                }
            }
            #[cfg(feature = "simd")]
            TensorStorage::Aligned(data) => {
                if Arc::strong_count(data) > 1 {
                    let data_vec = self.to_vec()?;
                    self.storage = Self::mutable_storage(data_vec)?;
                }
            }
            #[cfg(feature = "simd")]
            TensorStorage::SimdOptimized(storage) => {
                // `SimdStorage` is read-optimised: it never releases its
                // `original` buffer, so writing through its copy-on-write path
                // would leave every mutated tensor holding two full buffers for
                // the rest of its life. Promote to `Aligned` instead, which is a
                // single resident buffer that `with_slice_mut` can mutate in
                // place from then on.
                //
                // The promotion copies the data straight out of the SIMD
                // storage: the previous `to_vec()` + `aligned()` pair allocated
                // and copied the whole tensor twice.
                let promoted = storage.with_slice(TensorStorage::aligned_from_slice)?;
                self.storage = promoted;
            }
            #[cfg(feature = "gpu")]
            TensorStorage::Device { .. } => {
                // A device buffer is immutable and has no offset-capable write
                // primitive, so making a tensor writable *always* means moving
                // it back to the host. The download is the storage's cached one,
                // so this costs at most a single transfer.
                let data_vec = self.to_vec()?;
                self.storage = Self::mutable_storage(data_vec)?;
            }
        }
        Ok(())
    }

    /// Prepare this tensor for an in-place write: contiguous, uniquely owned and
    /// backed by mutable storage.
    pub(crate) fn prepare_for_inplace(&mut self) -> Result<()>
    where
        T: Copy,
    {
        self.make_unique()
    }

    /// Apply function in-place
    ///
    /// Mutates the storage buffer directly - no temporary buffer is allocated
    /// for base tensors that are already uniquely owned.
    pub fn apply_<F>(&mut self, func: F) -> Result<()>
    where
        F: Fn(T) -> T,
        T: Copy,
    {
        self.prepare_for_inplace()?;

        if matches!(self.storage, TensorStorage::MemoryMapped(_)) {
            // Memory-mapped storage cannot expose a mutable slice.
            let data = self.to_vec()?;
            let new_data: Vec<T> = data.into_iter().map(func).collect();
            self.storage = TensorStorage::create_optimal(new_data)?;
            return Ok(());
        }

        let numel = self.numel();
        self.storage.with_slice_mut(|slice| {
            for value in slice.iter_mut().take(numel) {
                *value = func(*value);
            }
            Ok(())
        })
    }

    /// Apply function element-wise to create new tensor
    ///
    /// Allocates exactly one output buffer and fills it from a borrowed view of
    /// the input.
    ///
    /// **Forward-only: the result is a detached leaf.** A closure carries no
    /// derivative, so propagating `requires_grad` here would manufacture a
    /// `requires_grad` node whose operation is still
    /// [`Operation::Leaf`](crate::core_ops::Operation::Leaf) — and the backward
    /// pass stops at every leaf and accumulates into that leaf's own
    /// (unreachable) gradient slot, which makes `backward()` succeed with
    /// silently wrong numbers. Callers that *are* differentiable must therefore
    /// record their own [`Operation`](crate::core_ops::Operation) on the result
    /// (see `Tensor::record_unary` and the `MulScalar`/`AddScalar` idiom in
    /// `math_ops.rs`).
    pub fn map<F>(&self, func: F) -> Result<Self>
    where
        F: Fn(T) -> T,
        T: Copy,
    {
        let new_data = self.with_contiguous_data(|data| {
            let mut out = Vec::with_capacity(data.len());
            out.extend(data.iter().map(|&x| func(x)));
            Ok(out)
        })?;
        Self::from_data(new_data, self.shape().dims().to_vec(), self.device)
    }

    /// Extract a scalar value from a single-element tensor
    pub fn item(&self) -> Result<T>
    where
        T: Copy,
    {
        let data = self.data()?;
        if data.len() != 1 {
            return Err(TorshError::InvalidArgument(format!(
                "item() can only be called on single-element tensors, got {} elements",
                data.len()
            )));
        }
        Ok(data[0])
    }

    /// Concatenate tensors along a dimension
    pub fn cat(tensors: &[&Self], dim: i32) -> Result<Self>
    where
        T: Copy,
    {
        if tensors.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Cannot concatenate empty tensor list".to_string(),
            ));
        }

        let first_shape_binding = tensors[0].shape();
        let first_shape = first_shape_binding.dims();
        let ndim = first_shape.len();

        // Normalize dim (allow negative indexing)
        let actual_dim = if dim < 0 {
            (ndim as i32 + dim) as usize
        } else {
            dim as usize
        };

        if actual_dim >= ndim {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of range for {}-dimensional tensor",
                dim, ndim
            )));
        }

        // Validate all tensors have compatible shapes (same on all dims except actual_dim)
        for (i, tensor) in tensors.iter().enumerate().skip(1) {
            let shape_binding = tensor.shape();
            let shape = shape_binding.dims();
            if shape.len() != ndim {
                return Err(TorshError::InvalidArgument(format!(
                    "Tensor {} has {} dimensions but first tensor has {}",
                    i,
                    shape.len(),
                    ndim
                )));
            }
            for (d, (&s1, &s2)) in first_shape.iter().zip(shape.iter()).enumerate() {
                if d != actual_dim && s1 != s2 {
                    return Err(TorshError::ShapeMismatch {
                        expected: first_shape.to_vec(),
                        got: shape.to_vec(),
                    });
                }
            }
        }

        // Compute output shape: same as input except actual_dim is sum of all cat dims
        let cat_dim_total: usize = tensors.iter().map(|t| t.shape().dims()[actual_dim]).sum();
        let mut result_shape = first_shape.to_vec();
        result_shape[actual_dim] = cat_dim_total;

        // Gather all data in order, interleaving elements for proper layout
        // Outer = product of dims before actual_dim
        // Cat stride = product of dims after actual_dim (inner)
        let outer_size: usize = first_shape[..actual_dim].iter().product();
        let inner_size: usize = first_shape[actual_dim + 1..].iter().product();

        let total_numel: usize = result_shape.iter().product();
        let mut result_data = Vec::with_capacity(total_numel);

        // Materialise every input exactly once (previously this happened once per
        // (outer, tensor) pair, i.e. `outer_size` full copies of every input).
        let sources: Vec<Vec<T>> = tensors
            .iter()
            .map(|tensor| tensor.to_vec())
            .collect::<Result<Vec<_>>>()?;
        let cat_sizes: Vec<usize> = tensors
            .iter()
            .map(|tensor| tensor.shape().dims()[actual_dim])
            .collect();

        for outer in 0..outer_size {
            for (source, &cat_size) in sources.iter().zip(cat_sizes.iter()) {
                // One contiguous run per (outer, tensor) pair.
                let run = cat_size * inner_size;
                let start = outer * run;
                result_data.extend_from_slice(&source[start..start + run]);
            }
        }

        let mut result = Self::from_data(result_data, result_shape, tensors[0].device)?;

        // Record the concatenation so gradients split back to each input along
        // `actual_dim`. Only recorded when at least one input tracks gradients.
        let any_requires_grad = tensors.iter().any(|t| t.requires_grad);
        if crate::should_record_grad(any_requires_grad) {
            result.requires_grad = true;
            result.operation = crate::core_ops::Operation::Concat {
                inputs: tensors.iter().map(|t| Arc::new((*t).clone())).collect(),
                dim: actual_dim,
            };
        }

        Ok(result)
    }
}

// Numeric operations
impl<T: TensorElement + Copy> Tensor<T>
where
    T: num_traits::Float,
{
    /// Compute the L2 norm of the tensor
    pub fn norm(&self) -> Result<Self> {
        let data = self.data()?;
        let sum_squares: T = data
            .iter()
            .map(|&x| x * x)
            .fold(num_traits::Zero::zero(), |acc, x| acc + x);
        let norm_value = sum_squares.sqrt();

        // Return scalar tensor (1-element tensor with shape [])
        Tensor::from_data(vec![norm_value], vec![], self.device())
    }

    /// Computes the p-norm (Lp norm) of the tensor, optionally reduced along
    /// specific dimensions.
    ///
    /// Mirrors PyTorch's `torch.norm(p, dim, keepdim)` semantics:
    /// - `p == 1.0` -> L1 ("Manhattan") norm: `sum(|x|)`
    /// - `p == 2.0` -> L2 (Euclidean) norm: `sqrt(sum(x^2))`, matching [`Tensor::norm`]
    /// - `p == 0.0` -> count of non-zero elements
    /// - `p == f64::INFINITY` -> maximum absolute value
    /// - `p == f64::NEG_INFINITY` -> minimum absolute value
    /// - any other finite `p` -> general Lp norm: `(sum(|x|^p))^(1/p)`
    ///
    /// `dims == None` reduces over every element, producing a scalar tensor
    /// (or an all-ones-shaped tensor when `keepdim` is true). `dims ==
    /// Some(&[...])` reduces only the given dimensions, which must already be
    /// normalized (non-negative and in range); duplicates are ignored.
    pub fn norm_lp(&self, p: f64, dims: Option<&[usize]>, keepdim: bool) -> Result<Self>
    where
        T: num_traits::FromPrimitive,
    {
        let shape_binding = self.shape();
        let input_shape = shape_binding.dims().to_vec();
        let ndim = input_shape.len();

        let reduce_dims: Vec<usize> = match dims {
            Some(requested) => {
                for &dim in requested {
                    if dim >= ndim {
                        return Err(TorshError::InvalidOperation(format!(
                            "Dimension {} out of range for {}-dimensional tensor",
                            dim, ndim
                        )));
                    }
                }
                let mut normalized = requested.to_vec();
                normalized.sort_unstable();
                normalized.dedup();
                normalized
            }
            None => (0..ndim).collect(),
        };

        let convert = |value: f64| -> Result<T> {
            <T as num_traits::FromPrimitive>::from_f64(value).ok_or_else(|| {
                TorshError::InvalidOperation(format!(
                    "norm: p={} cannot be represented in this tensor's element type",
                    p
                ))
            })
        };

        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();

        #[derive(Clone, Copy, PartialEq)]
        enum NormKind {
            L0,
            L1,
            L2,
            MaxAbs,
            MinAbs,
            General,
        }

        let kind = if p == 1.0 {
            NormKind::L1
        } else if p == 2.0 {
            NormKind::L2
        } else if p == 0.0 {
            NormKind::L0
        } else if p == f64::INFINITY {
            NormKind::MaxAbs
        } else if p == f64::NEG_INFINITY {
            NormKind::MinAbs
        } else {
            NormKind::General
        };

        let (p_t, inv_p_t) = if kind == NormKind::General {
            (convert(p)?, convert(1.0 / p)?)
        } else {
            (zero, zero)
        };

        // Identity element for the combining operation (sum -> 0, min -> +inf).
        let init: T = if kind == NormKind::MinAbs {
            <T as num_traits::Float>::infinity()
        } else {
            zero
        };

        let elem = |x: T| -> T {
            match kind {
                NormKind::L1 | NormKind::MaxAbs | NormKind::MinAbs => x.abs(),
                NormKind::L2 => x * x,
                NormKind::L0 => {
                    if x == zero {
                        zero
                    } else {
                        one
                    }
                }
                NormKind::General => x.abs().powf(p_t),
            }
        };

        let combine = |a: T, b: T| -> T {
            match kind {
                NormKind::MaxAbs => a.max(b),
                NormKind::MinAbs => a.min(b),
                _ => a + b,
            }
        };

        let finalize = |s: T| -> T {
            match kind {
                NormKind::L2 => s.sqrt(),
                NormKind::General => s.powf(inv_p_t),
                _ => s,
            }
        };

        let data = self.data()?;

        // Fully reduced (global norm) fast path: every dimension collapses to a scalar.
        if ndim == 0 || reduce_dims.len() == ndim {
            let acc = data.iter().fold(init, |acc, &x| combine(acc, elem(x)));
            let value = finalize(acc);
            let out_shape = if keepdim { vec![1; ndim] } else { vec![] };
            return Self::from_data(vec![value], out_shape, self.device());
        }

        // Partial reduction over an arbitrary subset of dimensions: walk every
        // element once, mapping its flat input index to the flat index of the
        // (keepdim-shaped) output it accumulates into.
        let mut is_reduced = vec![false; ndim];
        for &d in &reduce_dims {
            is_reduced[d] = true;
        }

        let mut input_strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
        }

        let mut output_shape_keepdim = input_shape.clone();
        for &d in &reduce_dims {
            output_shape_keepdim[d] = 1;
        }
        let mut output_strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            output_strides[i] = output_strides[i + 1] * output_shape_keepdim[i + 1];
        }
        let output_size: usize = output_shape_keepdim.iter().product();

        let mut acc = vec![init; output_size];
        for (flat_idx, &x) in data.iter().enumerate() {
            let mut remaining = flat_idx;
            let mut out_flat = 0usize;
            for (dim, &reduced) in is_reduced.iter().enumerate() {
                let coord = remaining / input_strides[dim];
                remaining %= input_strides[dim];
                if !reduced {
                    out_flat += coord * output_strides[dim];
                }
            }
            acc[out_flat] = combine(acc[out_flat], elem(x));
        }

        let result_data: Vec<T> = acc.into_iter().map(finalize).collect();

        let final_shape = if keepdim {
            output_shape_keepdim
        } else {
            input_shape
                .into_iter()
                .zip(is_reduced.iter())
                .filter(|(_, &reduced)| !reduced)
                .map(|(size, _)| size)
                .collect::<Vec<_>>()
        };

        Self::from_data(result_data, final_shape, self.device())
    }
}

// SciRS2 backend integration (placeholder implementations)
impl<T: TensorElement + Copy> Tensor<T> {
    /// Use SciRS2 backend for optimized matrix multiplication
    pub fn matmul_scirs2(&self, other: &Self) -> Result<Self>
    where
        T: num_traits::Float + num_traits::Zero + num_traits::One + std::iter::Sum,
    {
        // TODO: Integrate with actual SciRS2 backend
        // For now, implement basic matrix multiplication
        self.basic_matmul(other)
    }

    /// Use SciRS2 backend for optimized sum reduction
    pub fn sum_scirs2(&self) -> Result<Self>
    where
        T: std::ops::Add<Output = T> + num_traits::Zero,
    {
        // TODO: Integrate with actual SciRS2 backend
        let data = self.data()?;
        let sum_value = data
            .iter()
            .fold(<T as num_traits::Zero>::zero(), |acc, &x| acc + x);
        Tensor::from_data(vec![sum_value], vec![], self.device())
    }

    /// Use SciRS2 backend for optimized mean reduction
    pub fn mean_scirs2(&self) -> Result<Self>
    where
        T: std::ops::Add<Output = T>
            + std::ops::Div<Output = T>
            + num_traits::Zero
            + From<usize>
            + num_traits::FromPrimitive,
    {
        // TODO: Integrate with actual SciRS2 backend
        let data = self.data()?;
        if data.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Cannot compute mean of empty tensor".to_string(),
            ));
        }
        let sum_value = data
            .iter()
            .fold(<T as num_traits::Zero>::zero(), |acc, &x| acc + x);
        let mean_value = sum_value / T::from(data.len());
        Tensor::from_data(vec![mean_value], vec![], self.device())
    }

    /// Use SciRS2 backend for optimized ReLU activation
    ///
    /// Records the same [`UnaryKind::Relu`] as [`Tensor::relu`]: the forward
    /// predicate (`x > 0`) is identical, just computed through a bare `map`
    /// instead of `relu`'s SIMD/parallel dispatch, so the two must share one
    /// recorded derivative rather than one silently detaching.
    pub fn relu_scirs2(&self) -> Result<Self>
    where
        T: PartialOrd + num_traits::Zero,
    {
        // TODO: Integrate with actual SciRS2 backend
        let zero = <T as num_traits::Zero>::zero();
        let result = self.map(|x| if x > zero { x } else { zero })?;
        Ok(self.record_unary(result, UnaryKind::Relu))
    }

    /// Use SciRS2 backend for optimized sigmoid activation
    ///
    /// Records the same [`UnaryKind::Sigmoid`] as [`Tensor::sigmoid`]: this is
    /// the exact closed form (not an approximation), so it is numerically
    /// consistent with `Sigmoid`'s recorded derivative on every input.
    pub fn sigmoid_scirs2(&self) -> Result<Self>
    where
        T: num_traits::Float,
    {
        // TODO: Integrate with actual SciRS2 backend
        let result = self.map(|x| {
            let one = <T as num_traits::One>::one();
            one / (one + (-x).exp())
        })?;
        Ok(self.record_unary(result, UnaryKind::Sigmoid))
    }

    /// Use SciRS2 backend for optimized tanh activation
    ///
    /// Records the same [`UnaryKind::Tanh`] as [`Tensor::tanh`].
    pub fn tanh_scirs2(&self) -> Result<Self>
    where
        T: num_traits::Float,
    {
        // TODO: Integrate with actual SciRS2 backend
        let result = self.map(|x| x.tanh())?;
        Ok(self.record_unary(result, UnaryKind::Tanh))
    }

    /// Softmax activation along specified dimension
    /// Computes softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
    pub fn softmax(&self, dim: i32) -> Result<Self>
    where
        T: torsh_core::dtype::FloatElement
            + Copy
            + std::ops::Sub<Output = T>
            + std::ops::Div<Output = T>,
    {
        let data = self.data()?;
        let shape_binding = self.shape();
        let shape = shape_binding.dims();

        // Validate tensor has data
        if data.is_empty() || shape.is_empty() {
            return Err(TorshError::InvalidOperation(
                "Cannot compute softmax on empty tensor".to_string(),
            ));
        }

        // Handle negative dimension
        let actual_dim = if dim < 0 {
            (shape.len() as i32 + dim) as usize
        } else {
            dim as usize
        };

        if actual_dim >= shape.len() {
            return Err(TorshError::InvalidOperation(format!(
                "Dimension {} out of range for {}-dimensional tensor",
                actual_dim,
                shape.len()
            )));
        }

        // For numerical stability, subtract max before exp
        let max_tensor = self.max(Some(actual_dim), true)?;

        // Expand max_tensor to match input shape for broadcasting
        let expanded_max = max_tensor.expand(shape)?;
        let shifted = self.sub(&expanded_max)?;
        let exp_tensor = shifted.exp()?;
        let sum_tensor = exp_tensor.sum_dim(&[actual_dim as i32], true)?;

        // Expand sum_tensor to match exp_tensor shape for broadcasting
        let expanded_sum = sum_tensor.expand(shape)?;
        exp_tensor.div(&expanded_sum)
    }

    /// Log softmax activation along specified dimension.
    ///
    /// Computed with the log-sum-exp identity
    /// `log_softmax(x) = (x - max) - log(sum(exp(x - max)))` rather than as
    /// `log(softmax(x))`. The naive form underflows: once a logit sits about 90
    /// (`f32`) below the row maximum, `exp` rounds it to exactly `0` and the
    /// following `log` returns `-inf`, which then poisons every downstream loss
    /// (this is why cross-entropy over sharply scaled logits used to be `inf`).
    /// Here the sum is at least `1` — the maximum contributes `exp(0)` — so the
    /// logarithm never sees zero and every output stays finite.
    pub fn log_softmax(&self, dim: i32) -> Result<Self>
    where
        T: torsh_core::dtype::FloatElement + Copy + std::ops::Sub<Output = T>,
    {
        let shape_binding = self.shape();
        let shape = shape_binding.dims().to_vec();

        if shape.is_empty() {
            return Err(TorshError::InvalidOperation(
                "Cannot compute log_softmax on empty tensor".to_string(),
            ));
        }

        let actual_dim = if dim < 0 {
            (shape.len() as i32 + dim) as usize
        } else {
            dim as usize
        };

        if actual_dim >= shape.len() {
            return Err(TorshError::InvalidOperation(format!(
                "Dimension {} out of range for {}-dimensional tensor",
                actual_dim,
                shape.len()
            )));
        }

        // Shift by the per-slice maximum so the largest exponent is exp(0) = 1.
        let max_tensor = self.max(Some(actual_dim), true)?;
        let expanded_max = max_tensor.expand(&shape)?;
        let shifted = self.sub(&expanded_max)?;

        // log(sum(exp(shifted))) is finite: the sum is bounded below by 1.
        let sum_exp = shifted.exp()?.sum_dim(&[actual_dim as i32], true)?;
        let log_sum_exp = sum_exp.log()?;
        let expanded_log_sum = log_sum_exp.expand(&shape)?;

        let mut result = shifted.sub(&expanded_log_sum)?;

        // Record the exact, stable log-softmax Jacobian instead of the composed
        // (partly detaching) sub/exp/sum/log graph. Overwriting the operation
        // discards those throwaway intermediates. This is what makes
        // cross-entropy differentiable end-to-end.
        if crate::should_record_grad(self.requires_grad) {
            result.requires_grad = true;
            result.operation = crate::core_ops::Operation::LogSoftmax {
                input: Arc::new(self.clone()),
                dim: actual_dim,
            };
        }

        Ok(result)
    }

    /// Returns the k largest elements along a dimension
    pub fn topk(
        &self,
        k: usize,
        dim: Option<i32>,
        largest: bool,
        sorted: bool,
    ) -> Result<(Self, Tensor<i64>)>
    where
        T: std::cmp::PartialOrd + Copy + num_traits::Zero,
    {
        let data = self.data()?;
        let shape_binding = self.shape();
        let shape = shape_binding.dims();

        if shape.is_empty() {
            return Err(TorshError::InvalidOperation(
                "Cannot compute topk on empty tensor".to_string(),
            ));
        }

        if k == 0 {
            return Err(TorshError::InvalidArgument(
                "k must be greater than 0".to_string(),
            ));
        }

        // Determine actual dimension to operate on (default: last dim)
        let actual_dim = match dim {
            Some(d) => {
                let norm = if d < 0 {
                    (shape.len() as i32 + d) as usize
                } else {
                    d as usize
                };
                if norm >= shape.len() {
                    return Err(TorshError::InvalidArgument(format!(
                        "Dimension {} out of range for {}-dimensional tensor",
                        d,
                        shape.len()
                    )));
                }
                norm
            }
            None => shape.len() - 1,
        };

        let dim_size = shape[actual_dim];
        let effective_k = k.min(dim_size);

        let outer_size: usize = shape[..actual_dim].iter().product();
        let inner_size: usize = shape[actual_dim + 1..].iter().product();

        // Output shape: same as input but actual_dim replaced with k
        let mut result_shape = shape.to_vec();
        result_shape[actual_dim] = effective_k;

        let mut values_data = Vec::with_capacity(outer_size * effective_k * inner_size);
        let mut indices_data = Vec::with_capacity(outer_size * effective_k * inner_size);

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                // Gather (local_index, value) pairs along actual_dim for this (outer, inner) slice
                let mut slice: Vec<(usize, T)> = (0..dim_size)
                    .map(|d| {
                        let src = outer * dim_size * inner_size + d * inner_size + inner;
                        (d, data[src])
                    })
                    .collect();

                // Sort by value to find top-k candidates
                if largest {
                    slice
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                } else {
                    slice
                        .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                }

                let mut top_k: Vec<(usize, T)> = slice.into_iter().take(effective_k).collect();

                // When sorted=false, restore original (position) order
                if !sorted {
                    top_k.sort_by_key(|(idx, _)| *idx);
                }

                for (local_idx, val) in &top_k {
                    values_data.push(*val);
                    indices_data.push(*local_idx as i64);
                }
            }
        }

        // Re-arrange from (outer, k, inner) to match result_shape layout
        // Currently we have data as outer * inner * k interleaved; need outer * k * inner
        // Transpose inner and k dimensions
        let transposed_len = outer_size * effective_k * inner_size;
        let mut values_transposed = Vec::with_capacity(transposed_len);
        let mut indices_transposed = Vec::with_capacity(transposed_len);

        for outer in 0..outer_size {
            for k_idx in 0..effective_k {
                for inner in 0..inner_size {
                    let src = outer * inner_size * effective_k + inner * effective_k + k_idx;
                    values_transposed.push(values_data[src]);
                    indices_transposed.push(indices_data[src]);
                }
            }
        }

        let values_tensor = Self::from_data(values_transposed, result_shape.clone(), self.device)?;
        let indices_tensor =
            Tensor::<i64>::from_data(indices_transposed, result_shape, self.device)?;

        Ok((values_tensor, indices_tensor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    /// Promoting `SimdOptimized` storage must produce a private buffer, so a
    /// write through one handle is never visible through a sibling clone.
    ///
    /// The promotion now copies straight out of the SIMD storage instead of
    /// going through an intermediate `Vec`; this test pins the isolation that
    /// the shortcut must not lose.
    #[cfg(feature = "simd")]
    #[test]
    fn make_unique_isolates_shared_simd_storage() {
        // 4096 f32 = 16 KiB, above the 10 KiB SimdOptimized threshold.
        let mut tensor = Tensor::from_data(vec![1.0f32; 4096], vec![4096], DeviceType::Cpu)
            .expect("tensor creation should succeed");
        assert!(
            matches!(tensor.storage, TensorStorage::SimdOptimized(_)),
            "precondition: a 16 KiB tensor uses lock-free SIMD storage"
        );
        let sibling = tensor.clone();

        tensor.make_unique().expect("make_unique should succeed");
        assert!(
            !matches!(tensor.storage, TensorStorage::SimdOptimized(_)),
            "SimdOptimized storage must be promoted to single-buffer mutable storage"
        );

        tensor.apply_(|value| value + 1.0).expect("apply_");
        assert_eq!(
            tensor.to_vec().expect("to_vec")[0],
            2.0,
            "the writer must see its own update"
        );
        assert_eq!(
            sibling.to_vec().expect("to_vec")[0],
            1.0,
            "the sibling must not observe the write"
        );
    }

    /// The single-copy promotion must reproduce the data exactly, including for
    /// storage that had already been written through its copy-on-write path.
    #[cfg(feature = "simd")]
    #[test]
    fn make_unique_preserves_mutated_simd_contents() {
        let source = Tensor::from_data(
            (0..4096).map(|i| i as f32).collect::<Vec<_>>(),
            vec![4096],
            DeviceType::Cpu,
        )
        .expect("tensor creation should succeed");

        // Write through the SIMD storage's copy-on-write path first.
        source
            .storage
            .with_slice_mut(|slice| {
                slice[0] = -1.0;
                Ok(())
            })
            .expect("copy-on-write write should succeed");

        let mut promoted = source.clone();
        promoted.make_unique().expect("make_unique should succeed");
        let data = promoted.to_vec().expect("to_vec");
        assert_eq!(data[0], -1.0, "the promotion must copy the CoW buffer");
        assert_eq!(data[4095], 4095.0);
    }

    #[test]
    fn test_scalar_creation() {
        let scalar = Tensor::<f32>::scalar(42.0).expect("operation should succeed");
        assert_eq!(scalar.shape().dims(), &[] as &[usize]);
        assert_eq!(scalar.item().expect("item extraction should succeed"), 42.0);
    }

    #[test]
    fn test_max_reduction() {
        let data = vec![1.0f32, 5.0, 3.0, 2.0];
        let tensor =
            Tensor::from_data(data, vec![4], DeviceType::Cpu).expect("operation should succeed");
        let max_val = tensor.max(None, false).expect("operation should succeed");
        assert_eq!(max_val.item().expect("item extraction should succeed"), 5.0);
    }

    #[test]
    fn test_norm_computation() {
        let data = vec![3.0f32, 4.0]; // 3-4-5 triangle
        let tensor =
            Tensor::from_data(data, vec![2], DeviceType::Cpu).expect("operation should succeed");
        let norm = tensor.norm().expect("norm computation should succeed");
        assert!((norm.item().expect("item extraction should succeed") - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_operations() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut tensor =
            Tensor::from_data(data, vec![4], DeviceType::Cpu).expect("operation should succeed");

        // Test apply_
        tensor
            .apply_(|x| x * 2.0)
            .expect("operation should succeed");
        assert_eq!(
            tensor.data().expect("data retrieval should succeed"),
            vec![2.0, 4.0, 6.0, 8.0]
        );

        // Test map
        let original = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu)
            .expect("operation should succeed");
        let mapped = original.map(|x| x + 1.0).expect("operation should succeed");
        assert_eq!(
            mapped.data().expect("data retrieval should succeed"),
            vec![2.0, 3.0, 4.0]
        );
        assert_eq!(
            original.data().expect("data retrieval should succeed"),
            vec![1.0, 2.0, 3.0]
        ); // Original unchanged
    }

    #[test]
    fn test_activation_functions() {
        let data = vec![-1.0f32, 0.0, 1.0, 2.0];
        let tensor =
            Tensor::from_data(data, vec![4], DeviceType::Cpu).expect("operation should succeed");

        // Test ReLU
        let relu_result = tensor.relu().expect("relu should succeed");
        assert_eq!(
            relu_result.data().expect("data retrieval should succeed"),
            vec![0.0, 0.0, 1.0, 2.0]
        );

        // Test abs
        let abs_result = tensor.abs().expect("abs computation should succeed");
        assert_eq!(
            abs_result.data().expect("data retrieval should succeed"),
            vec![1.0, 0.0, 1.0, 2.0]
        );

        // Test clamp
        let clamped = tensor.clamp(-0.5, 1.5).expect("operation should succeed");
        assert_eq!(
            clamped.data().expect("data retrieval should succeed"),
            vec![-0.5, 0.0, 1.0, 1.5]
        );
    }

    #[test]
    fn test_storage_sharing() {
        let tensor1 =
            Tensor::<f32>::zeros(&[2, 2], DeviceType::Cpu).expect("operation should succeed");
        let tensor2 = tensor1.clone();
        let tensor3 = tensor1.clone_data();

        assert!(tensor1.shares_storage(&tensor2));
        assert!(!tensor1.shares_storage(&tensor3));
    }

    #[test]
    fn test_basic_matmul() {
        let a = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu)
            .expect("operation should succeed");
        let b = Tensor::from_data(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2], DeviceType::Cpu)
            .expect("operation should succeed");

        let result = a.basic_matmul(&b).expect("operation should succeed");
        assert_eq!(result.shape().dims(), &[2, 2]);

        // Expected: [1*5+2*7, 1*6+2*8] = [19, 22]
        //           [3*5+4*7, 3*6+4*8] = [43, 50]
        let expected = vec![19.0, 22.0, 43.0, 50.0];
        assert_eq!(
            result.data().expect("data retrieval should succeed"),
            expected
        );
    }

    #[test]
    fn test_reductions() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor =
            Tensor::from_data(data, vec![4], DeviceType::Cpu).expect("operation should succeed");

        let sum = tensor.sum().expect("sum should succeed");
        assert_eq!(sum.item().expect("item extraction should succeed"), 10.0);

        let mean = tensor.mean(None, false).expect("operation should succeed");
        assert_eq!(mean.item().expect("item extraction should succeed"), 2.5);
    }

    #[test]
    fn test_copy_on_write() {
        let mut tensor1 =
            Tensor::<f32>::ones(&[2], DeviceType::Cpu).expect("operation should succeed");
        let tensor2 = tensor1.clone();

        // Both should share storage initially
        assert!(tensor1.shares_storage(&tensor2));

        // After make_unique, they should not share storage
        tensor1.make_unique().expect("make_unique should succeed");
        assert!(!tensor1.shares_storage(&tensor2));
    }

    /// F171: `make_unique` on an unshared tensor must not keep reallocating.
    ///
    /// `create_optimal` hands every f32 tensor >= 2560 elements the *immutable*
    /// `SimdOptimized` storage, so the first call has to convert it once to
    /// mutable `Aligned` storage. Every later call must be free.
    #[test]
    fn test_make_unique_does_not_realloc_unshared_storage() {
        let mut tensor = Tensor::<f32>::from_data(vec![1.0; 4096], vec![4096], DeviceType::Cpu)
            .expect("tensor creation should succeed");
        assert_eq!(
            tensor.storage_type(),
            "simd_optimized",
            "precondition: large f32 tensors start out immutable"
        );

        tensor.make_unique().expect("make_unique should succeed");
        assert_eq!(tensor.storage_type(), "aligned_simd");
        let ptr_after_first = tensor
            .storage
            .with_slice(|s| Ok(s.as_ptr() as usize))
            .expect("with_slice should succeed");

        tensor.make_unique().expect("make_unique should succeed");
        let ptr_after_second = tensor
            .storage
            .with_slice(|s| Ok(s.as_ptr() as usize))
            .expect("with_slice should succeed");

        assert_eq!(
            ptr_after_first, ptr_after_second,
            "make_unique must not copy storage that is already unique and mutable"
        );
    }

    /// F171: the copy-on-write copy must stay *mutable*.
    ///
    /// `create_optimal` would hand a large f32 tensor immutable `SimdOptimized`
    /// storage again, so every in-place op after a CoW split would fail.
    #[test]
    fn test_make_unique_cow_copy_stays_mutable() {
        let mut tensor = Tensor::<f32>::from_data(vec![1.0; 4096], vec![4096], DeviceType::Cpu)
            .expect("tensor creation should succeed");
        tensor.make_unique().expect("make_unique should succeed");
        assert_eq!(tensor.storage_type(), "aligned_simd");

        let shared = tensor.clone();
        tensor.make_unique().expect("make_unique should succeed");
        assert_eq!(
            tensor.storage_type(),
            "aligned_simd",
            "a copy-on-write split must not fall back to immutable storage"
        );
        tensor.apply_(|x| x + 1.0).expect("apply_ should succeed");
        assert_eq!(tensor.to_vec().expect("to_vec")[0], 2.0);
        assert_eq!(
            shared.to_vec().expect("to_vec")[0],
            1.0,
            "the shared tensor must be unaffected"
        );
    }

    /// F058/F168: `apply_` mutates the existing buffer instead of replacing it.
    #[test]
    fn test_apply_mutates_storage_in_place() {
        let mut tensor = Tensor::<f32>::from_data(vec![2.0; 4096], vec![4096], DeviceType::Cpu)
            .expect("tensor creation should succeed");
        tensor.make_unique().expect("make_unique should succeed");
        let ptr_before = tensor
            .storage
            .with_slice(|s| Ok(s.as_ptr() as usize))
            .expect("with_slice should succeed");

        tensor.apply_(|x| x * 3.0).expect("apply_ should succeed");

        let ptr_after = tensor
            .storage
            .with_slice(|s| Ok(s.as_ptr() as usize))
            .expect("with_slice should succeed");
        assert_eq!(ptr_before, ptr_after, "apply_ must not reallocate storage");
        assert_eq!(tensor.to_vec().expect("to_vec")[0], 6.0);
    }

    #[test]
    fn test_item_extraction() {
        let scalar = Tensor::from_data(vec![42.0f32], vec![], DeviceType::Cpu)
            .expect("operation should succeed");
        assert_eq!(scalar.item().expect("item extraction should succeed"), 42.0);

        let vector = Tensor::from_data(vec![1.0f32, 2.0], vec![2], DeviceType::Cpu)
            .expect("operation should succeed");
        assert!(vector.item().is_err()); // Should fail for multi-element tensor
    }

    #[test]
    fn test_all_dim() {
        // Shape [2, 3]: [[1, 0, 1], [1, 1, 1]]
        let data = vec![1i32, 0, 1, 1, 1, 1];
        let tensor = Tensor::from_data(data, vec![2, 3], DeviceType::Cpu)
            .expect("tensor creation should succeed");

        // all along dim 0 (rows): per column check
        // col0: 1&&1=true, col1: 0&&1=false, col2: 1&&1=true
        let result = tensor.all_dim(0, false).expect("all_dim should succeed");
        assert_eq!(result.shape().dims(), &[3]);
        assert_eq!(
            result.to_vec().expect("to_vec should succeed"),
            vec![true, false, true]
        );

        // all along dim 1 (cols): per row check
        // row0: 1&&0&&1=false, row1: 1&&1&&1=true
        let result_row = tensor.all_dim(1, false).expect("all_dim should succeed");
        assert_eq!(result_row.shape().dims(), &[2]);
        assert_eq!(
            result_row.to_vec().expect("to_vec should succeed"),
            vec![false, true]
        );

        // keepdim=true preserves dimension
        let result_kd = tensor.all_dim(1, true).expect("all_dim should succeed");
        assert_eq!(result_kd.shape().dims(), &[2, 1]);
    }

    #[test]
    fn test_any_dim() {
        // Shape [2, 3]: [[0, 0, 0], [0, 1, 0]]
        let data = vec![0i32, 0, 0, 0, 1, 0];
        let tensor = Tensor::from_data(data, vec![2, 3], DeviceType::Cpu)
            .expect("tensor creation should succeed");

        // any along dim 0: col0: false, col1: true, col2: false
        let result = tensor.any_dim(0, false).expect("any_dim should succeed");
        assert_eq!(result.shape().dims(), &[3]);
        assert_eq!(
            result.to_vec().expect("to_vec should succeed"),
            vec![false, true, false]
        );

        // any along dim 1: row0: false, row1: true
        let result_row = tensor.any_dim(1, false).expect("any_dim should succeed");
        assert_eq!(result_row.shape().dims(), &[2]);
        assert_eq!(
            result_row.to_vec().expect("to_vec should succeed"),
            vec![false, true]
        );
    }

    #[test]
    fn test_cat_multidim() {
        // Test concatenation along dim 0
        let a = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu)
            .expect("tensor creation should succeed");
        let b = Tensor::from_data(vec![5.0f32, 6.0], vec![1, 2], DeviceType::Cpu)
            .expect("tensor creation should succeed");

        let cat0 = Tensor::<f32>::cat(&[&a, &b], 0).expect("cat should succeed");
        assert_eq!(cat0.shape().dims(), &[3, 2]);
        assert_eq!(
            cat0.to_vec().expect("to_vec should succeed"),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );

        // Test concatenation along dim 1
        let c = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu)
            .expect("tensor creation should succeed");
        let d = Tensor::from_data(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2], DeviceType::Cpu)
            .expect("tensor creation should succeed");

        let cat1 = Tensor::<f32>::cat(&[&c, &d], 1).expect("cat should succeed");
        assert_eq!(cat1.shape().dims(), &[2, 4]);
        assert_eq!(
            cat1.to_vec().expect("to_vec should succeed"),
            vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]
        );
    }

    #[test]
    fn test_topk_along_dim() {
        // 2x4 tensor, topk along dim 1
        let data = vec![3.0f32, 1.0, 4.0, 2.0, 5.0, 9.0, 2.0, 6.0];
        let tensor = Tensor::from_data(data, vec![2, 4], DeviceType::Cpu)
            .expect("tensor creation should succeed");

        let (vals, idxs) = tensor
            .topk(2, Some(1), true, true)
            .expect("topk should succeed");
        assert_eq!(vals.shape().dims(), &[2, 2]);
        assert_eq!(idxs.shape().dims(), &[2, 2]);

        // Row 0: [3, 1, 4, 2] -> top2 = [4, 3] at positions [2, 0]
        // Row 1: [5, 9, 2, 6] -> top2 = [9, 6] at positions [1, 3]
        let vals_data = vals.to_vec().expect("to_vec should succeed");
        let idxs_data = idxs.to_vec().expect("to_vec should succeed");
        assert_eq!(vals_data[0], 4.0);
        assert_eq!(vals_data[1], 3.0);
        assert_eq!(vals_data[2], 9.0);
        assert_eq!(vals_data[3], 6.0);
        assert_eq!(idxs_data[0], 2);
        assert_eq!(idxs_data[1], 0);
        assert_eq!(idxs_data[2], 1);
        assert_eq!(idxs_data[3], 3);
    }

    // --- Regression tests for issue #43: mean must propagate requires_grad ---

    #[test]
    fn test_issue_43_mean_propagates_requires_grad() {
        // A tensor with requires_grad=true; mean result must also require grad.
        let input = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![4], DeviceType::Cpu)
            .expect("tensor creation failed")
            .requires_grad_(true);

        let result = input.mean(None, false).expect("mean should succeed");
        assert!(
            result.requires_grad(),
            "mean result must have requires_grad=true when input does"
        );
    }

    #[test]
    fn test_issue_43_mean_no_requires_grad_when_input_has_none() {
        // When input does not require grad, result should not either.
        let input = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu)
            .expect("tensor creation failed");

        let result = input.mean(None, false).expect("mean should succeed");
        assert!(
            !result.requires_grad(),
            "mean result must not require grad when input does not"
        );
    }

    #[test]
    fn test_issue_43_mean_backward() {
        // For mean of n elements, backward with upstream grad=1 distributes 1/n to each element.
        // mean() already reduces to a scalar, so backward() can be called directly.
        let n = 4usize;
        let input = Tensor::from_data(vec![2.0f32, 4.0, 6.0, 8.0], vec![n], DeviceType::Cpu)
            .expect("tensor creation failed")
            .requires_grad_(true);

        let result = input.mean(None, false).expect("mean should succeed");
        assert!(result.requires_grad(), "mean result must track gradients");
        // mean(None) with keepdim=false produces a scalar (numel=1), so backward is valid
        result.backward().expect("backward should succeed");

        let grad = input
            .grad()
            .expect("input must have gradient after backward");
        let grad_data = grad.data().expect("gradient data");

        // Each element should receive 1.0 / n = 0.25
        let expected = 1.0f32 / n as f32;
        for &g in &grad_data {
            assert!(
                (g - expected).abs() < 1e-6,
                "each element grad should be 1/n={expected}, got {g}"
            );
        }
    }

    #[test]
    fn test_sum_backward() {
        // loss = sum(x); d(loss)/dx_i = 1 for every element.
        let x = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![4], DeviceType::Cpu)
            .expect("tensor creation failed")
            .requires_grad_(true);
        let loss = x.sum().expect("sum should succeed");
        assert!(loss.requires_grad(), "sum result must track gradients");
        loss.backward().expect("backward should succeed");
        let grad = x.grad().expect("x must have a gradient after backward");
        let grad_data = grad.data().expect("gradient data");
        assert_eq!(
            grad_data,
            vec![1.0f32, 1.0, 1.0, 1.0],
            "d(sum)/dx must be all ones"
        );
    }

    #[test]
    fn test_matmul_backward() {
        // C = A @ B, loss = sum(C). grad_C = ones, so:
        //   grad_A = ones @ Bᵀ = [[11,15],[11,15]]
        //   grad_B = Aᵀ @ ones = [[4,4],[6,6]]
        // for A = [[1,2],[3,4]], B = [[5,6],[7,8]].
        let a = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu)
            .expect("tensor creation failed")
            .requires_grad_(true);
        let b = Tensor::from_data(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2], DeviceType::Cpu)
            .expect("tensor creation failed")
            .requires_grad_(true);

        let c = a.matmul(&b).expect("matmul should succeed");
        assert!(c.requires_grad(), "matmul result must track gradients");
        let loss = c.sum().expect("sum should succeed");
        loss.backward().expect("backward should succeed");

        let grad_a = a
            .grad()
            .expect("A must have a gradient")
            .data()
            .expect("grad data");
        let grad_b = b
            .grad()
            .expect("B must have a gradient")
            .data()
            .expect("grad data");
        assert_eq!(
            grad_a,
            vec![11.0f32, 15.0, 11.0, 15.0],
            "grad_A = ones @ Bᵀ"
        );
        assert_eq!(grad_b, vec![4.0f32, 4.0, 6.0, 6.0], "grad_B = Aᵀ @ ones");
    }
}
