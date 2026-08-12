//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "gpu")]
use crate::backend_integration::GpuBackendType;
use crate::storage::TensorStorage;
use std::path::PathBuf;
use std::sync::{Arc, RwLock, Weak};
use torsh_core::sync::RwLockExt;
use torsh_core::{
    device::DeviceType,
    dtype::{DType, FloatElement, TensorElement},
    error::{Result, TorshError},
    shape::Shape,
};

/// The main Tensor type for ToRSh
///
/// A tensor implementation with automatic memory mapping for large tensors
/// and efficient views with reference counting
#[derive(Clone)]
pub struct Tensor<T = f32>
where
    T: TensorElement,
{
    /// The data storage (automatically uses memory mapping for large tensors)
    pub(crate) storage: TensorStorage<T>,
    /// Shape of the tensor
    pub(crate) shape: Shape,
    /// Device information
    pub(crate) device: DeviceType,
    /// Whether gradients are required
    pub(crate) requires_grad: bool,
    /// Gradient tensor if computed
    pub(crate) grad: Arc<RwLock<Option<Tensor<T>>>>,
    /// Operation that created this tensor
    pub(crate) operation: Operation<T>,
    /// Custom strides for views (None means contiguous layout)
    pub(crate) strides: Option<Vec<usize>>,
    /// Offset into the storage for views (0 for base tensors)
    pub(crate) storage_offset: usize,
    /// The tensor this one is a view of (`None` for base tensors).
    ///
    /// This is a *strong* handle on purpose: it keeps the source tensor's
    /// metadata reachable for as long as any view of it exists, so aliasing can
    /// actually be reasoned about. It costs only the metadata struct — the data
    /// buffer inside [`TensorStorage`] is `Arc`-shared with the view anyway.
    pub(crate) base_tensor: Option<Arc<Tensor<T>>>,
}
impl<T: TensorElement + Copy> Tensor<T> {
    /// Clean up dead weak references in custom operations to improve memory efficiency
    pub fn cleanup_operation_refs(&mut self) {
        if let Operation::Custom(_, inputs) = &mut self.operation {
            inputs.retain(|weak_ref| weak_ref.strong_count() > 0);
        }
    }
    /// Create from raw data
    pub fn from_data(data: Vec<T>, shape: Vec<usize>, device: DeviceType) -> Result<Self> {
        let storage = TensorStorage::create_optimal(data)?;
        Ok(Self {
            storage,
            shape: Shape::new(shape),
            device,
            requires_grad: false,
            grad: Arc::new(RwLock::new(None)),
            operation: Operation::Leaf,
            strides: None,
            storage_offset: 0,
            base_tensor: None,
        })
    }
    /// 🚀 **Phase 7**: Create tensor with fast result storage (skips alignment copy)
    ///
    /// For SIMD operation results, uses simple InMemory storage to avoid
    /// the ~10µs overhead of copying data to aligned memory.
    ///
    /// # Performance
    /// - Skips AlignedVec copy (saves ~10µs for 50K elements)
    /// - Best for intermediate/result tensors
    /// - Input tensors should still use from_data for optimal SIMD input access
    pub fn from_data_fast(data: Vec<T>, shape: Vec<usize>, device: DeviceType) -> Self {
        let storage = TensorStorage::fast_result(data);
        Self {
            storage,
            shape: Shape::new(shape),
            device,
            requires_grad: false,
            grad: Arc::new(RwLock::new(None)),
            operation: Operation::Leaf,
            strides: None,
            storage_offset: 0,
            base_tensor: None,
        }
    }
    /// Create from raw data with explicit storage type
    pub fn from_data_with_storage(
        data: Vec<T>,
        shape: Vec<usize>,
        device: DeviceType,
        use_memory_mapping: bool,
    ) -> Result<Self> {
        let storage = if use_memory_mapping {
            TensorStorage::memory_mapped(data, None)?
        } else {
            TensorStorage::in_memory(data)
        };
        Ok(Self {
            storage,
            shape: Shape::new(shape),
            device,
            requires_grad: false,
            grad: Arc::new(RwLock::new(None)),
            operation: Operation::Leaf,
            strides: None,
            storage_offset: 0,
            base_tensor: None,
        })
    }
    /// Create from raw data with specified memory-mapped file path
    pub fn from_data_memory_mapped(
        data: Vec<T>,
        shape: Vec<usize>,
        device: DeviceType,
        file_path: PathBuf,
    ) -> Result<Self> {
        let storage = TensorStorage::memory_mapped(data, Some(file_path))?;
        Ok(Self {
            storage,
            shape: Shape::new(shape),
            device,
            requires_grad: false,
            grad: Arc::new(RwLock::new(None)),
            operation: Operation::Leaf,
            strides: None,
            storage_offset: 0,
            base_tensor: None,
        })
    }
    /// Create a tensor filled with zeros
    pub fn zeros(shape: &[usize], device: DeviceType) -> Result<Self> {
        let numel = shape.iter().product();
        let data = vec![T::zero(); numel];
        Self::from_data(data, shape.to_vec(), device)
    }
    /// Create a tensor filled with ones
    pub fn ones(shape: &[usize], device: DeviceType) -> Result<Self> {
        let numel = shape.iter().product();
        let data = vec![T::one(); numel];
        Self::from_data(data, shape.to_vec(), device)
    }
    /// Get the shape of the tensor
    pub fn shape(&self) -> Shape {
        self.shape.clone()
    }
    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }
    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }
    /// Get the data type
    pub fn dtype(&self) -> DType {
        T::dtype()
    }
    /// Convert tensor to a different data type
    pub fn to_dtype(&self, dtype: DType) -> Result<Self> {
        if T::dtype() == dtype {
            Ok(self.clone())
        } else {
            Err(TorshError::UnsupportedOperation {
                op: "dtype conversion".to_string(),
                dtype: format!("{:?} to {:?}", T::dtype(), dtype),
            })
        }
    }
    /// Get the device
    pub fn device(&self) -> DeviceType {
        self.device
    }
    /// Get element at multi-dimensional index
    pub fn get(&self, indices: &[usize]) -> Result<T>
    where
        T: Copy,
    {
        let flat_index = self.compute_flat_index(indices)?;
        self.storage.get(flat_index)
    }
    /// Get element at single flat index.
    ///
    /// `index` is a *logical* index in this tensor's own row-major order, so a
    /// view resolves it through its strides and storage offset rather than
    /// reading the base tensor's buffer directly.
    pub fn get_flat(&self, index: usize) -> Result<T>
    where
        T: Copy,
    {
        self.storage.get(self.logical_to_physical(index)?)
    }
    /// Set element at index (requires multi-dimensional indices for views)
    pub fn set(&self, indices: &[usize], value: T) -> Result<()>
    where
        T: Copy,
    {
        let flat_index = self.compute_flat_index(indices)?;
        self.storage.set(flat_index, value)
    }
    /// Get `len` elements starting at the logical flat index `start`.
    ///
    /// The range is expressed in this tensor's own row-major order; views gather
    /// the elements through their strides instead of slicing the base buffer.
    pub fn get_slice(&self, start: usize, len: usize) -> Result<Vec<T>>
    where
        T: Copy,
    {
        self.check_logical_range(start, len)?;
        if self.has_default_layout() {
            return self.storage.get_slice(start + self.storage_offset, len);
        }
        let mut values = Vec::with_capacity(len);
        for offset in 0..len {
            values.push(
                self.storage
                    .get(self.logical_to_physical(start + offset)?)?,
            );
        }
        Ok(values)
    }
    /// Write `values` starting at the logical flat index `start`.
    ///
    /// The range is expressed in this tensor's own row-major order, so writing
    /// through a view updates the elements the view actually addresses.
    pub fn set_slice(&self, start: usize, values: &[T]) -> Result<()>
    where
        T: Copy,
    {
        self.check_logical_range(start, values.len())?;
        if self.has_default_layout() {
            return self.storage.set_slice(start + self.storage_offset, values);
        }
        for (offset, &value) in values.iter().enumerate() {
            self.storage
                .set(self.logical_to_physical(start + offset)?, value)?;
        }
        Ok(())
    }
    /// Whether this tensor addresses its storage as a plain row-major block
    /// (contiguous strides), so logical indices differ from physical ones only by
    /// the storage offset.
    pub(crate) fn has_default_layout(&self) -> bool {
        match self.strides {
            Some(ref strides) => *strides == self.compute_default_strides(),
            None => true,
        }
    }
    /// Validate that `[start, start + len)` lies inside this tensor.
    fn check_logical_range(&self, start: usize, len: usize) -> Result<()> {
        let numel = self.numel();
        let end = start.checked_add(len).ok_or(TorshError::IndexOutOfBounds {
            index: start,
            size: numel,
        })?;
        if end > numel {
            return Err(TorshError::IndexOutOfBounds {
                index: end,
                size: numel,
            });
        }
        Ok(())
    }
    /// Translate a logical (view-order) flat index into a storage index.
    fn logical_to_physical(&self, index: usize) -> Result<usize> {
        let numel = self.numel();
        if index >= numel {
            return Err(TorshError::IndexOutOfBounds { index, size: numel });
        }
        if self.has_default_layout() {
            return Ok(index + self.storage_offset);
        }
        // Decode the logical index into per-axis coordinates, then re-encode it
        // with this tensor's strides.
        let shape_binding = self.shape();
        let dims = shape_binding.dims();
        let strides = self.strides();
        let mut remaining = index;
        let mut physical = self.storage_offset;
        for axis in (0..dims.len()).rev() {
            let extent = dims[axis];
            if extent == 0 {
                return Err(TorshError::IndexOutOfBounds { index, size: numel });
            }
            physical += (remaining % extent) * strides[axis];
            remaining /= extent;
        }
        Ok(physical)
    }
    /// Get all data as a vector (may be expensive for large memory-mapped tensors)
    /// For views, extracts only the data visible by this view
    pub fn to_vec(&self) -> Result<Vec<T>>
    where
        T: Copy,
    {
        if self.is_view() {
            let mut result = Vec::with_capacity(self.numel());
            let shape = self.shape.dims();
            let mut indices = vec![0; shape.len()];
            fn extract_recursive<T: TensorElement + Copy>(
                tensor: &Tensor<T>,
                indices: &mut [usize],
                dim: usize,
                result: &mut Vec<T>,
            ) -> Result<()> {
                let shape = tensor.shape.dims();
                if dim == shape.len() {
                    let flat_index = tensor.compute_flat_index(indices)?;
                    let value = tensor.storage.get(flat_index)?;
                    result.push(value);
                } else {
                    for i in 0..shape[dim] {
                        indices[dim] = i;
                        extract_recursive(tensor, indices, dim + 1, result)?;
                    }
                }
                Ok(())
            }
            extract_recursive(self, &mut indices, 0, &mut result)?;
            Ok(result)
        } else {
            self.storage.to_vec()
        }
    }
    /// Get storage type information
    pub fn storage_type(&self) -> &'static str {
        self.storage.storage_type()
    }
    /// Get estimated memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.storage.memory_usage()
    }
    /// Check if tensor uses memory mapping
    pub fn is_memory_mapped(&self) -> bool {
        matches!(self.storage, TensorStorage::MemoryMapped(_))
    }
    /// Check if this tensor addresses its storage through view metadata.
    ///
    /// The test is the *layout*, not the presence of a base handle: a tensor
    /// with custom strides or a non-zero storage offset must be read through
    /// [`Tensor::to_vec`]'s stride-aware path, while a tensor that owns a plain
    /// row-major block can be read straight out of the buffer. A reshape of a
    /// whole base tensor shares storage but keeps the default layout, so it is
    /// deliberately *not* reported as a view.
    pub fn is_view(&self) -> bool {
        self.strides.is_some() || self.storage_offset != 0
    }

    /// The tensor this one is a view of, if any.
    ///
    /// Views of a view report the original base tensor, and the handle is alive
    /// for as long as the view is.
    pub fn base_tensor(&self) -> Option<Arc<Tensor<T>>> {
        self.base_tensor.as_ref().map(Arc::clone)
    }
    /// Get the strides for this tensor (either custom strides for views or default contiguous strides)
    pub fn strides(&self) -> Vec<usize> {
        if let Some(ref strides) = self.strides {
            strides.clone()
        } else {
            self.compute_default_strides()
        }
    }
    /// Compute default contiguous strides for the tensor's shape
    pub(crate) fn compute_default_strides(&self) -> Vec<usize> {
        let shape = self.shape.dims();
        let mut strides = vec![1; shape.len()];
        if shape.len() > 1 {
            for i in (0..shape.len() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
        strides
    }
    /// Compute flat index with custom strides and offset for views
    fn compute_flat_index(&self, indices: &[usize]) -> Result<usize> {
        if indices.len() != self.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Expected {} indices, got {}",
                self.ndim(),
                indices.len()
            )));
        }
        let shape = self.shape.dims();
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= shape[i] {
                return Err(TorshError::IndexOutOfBounds {
                    index: idx,
                    size: shape[i],
                });
            }
        }
        let strides = self.strides();
        let flat_index = indices
            .iter()
            .zip(&strides)
            .map(|(idx, stride)| idx * stride)
            .sum::<usize>()
            + self.storage_offset;
        Ok(flat_index)
    }
    /// Execute a function with zero-copy access to tensor data (immutable)
    ///
    /// This enables SIMD operations without memory copies by providing direct
    /// access to the underlying buffer within a scoped context.
    ///
    /// # Arguments
    /// * `f` - Closure that receives `&[T]` and returns `Result<R>`
    ///
    /// # Returns
    /// Result from the closure
    ///
    /// # Performance
    /// - Zero memory copies for InMemory and Aligned storage
    /// - One allocation for MemoryMapped storage
    /// - Enables 2-4x SIMD speedup (per SciRS2 docs)
    ///
    /// # Examples
    /// ```ignore
    /// // Direct SIMD operation without copies
    /// let result = tensor.with_data_slice(|data| {
    ///     other_tensor.with_data_slice(|other_data| {
    ///         // Zero-copy SIMD addition
    ///         f32::simd_add(&data, &other_data)
    ///     })
    /// })?;
    /// ```
    pub fn with_data_slice<R, F>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&[T]) -> Result<R>,
        T: Copy,
    {
        let (offset, len) = self.contiguous_window("with_data_slice")?;
        self.storage.with_slice(|slice| {
            let window = slice
                .get(offset..offset + len)
                .ok_or_else(|| Self::window_error(slice.len(), offset, len))?;
            f(window)
        })
    }
    /// Storage window this tensor occupies, or an error when its layout is not a
    /// plain row-major block.
    ///
    /// Handing a strided or offset view's *base* buffer to a closure that indexes
    /// it row-major silently reads the wrong elements, so those layouts are
    /// rejected and the caller is told to materialise a contiguous copy first.
    fn contiguous_window(&self, method: &str) -> Result<(usize, usize)> {
        if !self.has_default_layout() {
            return Err(TorshError::InvalidOperation(format!(
                "{method} requires a contiguous tensor (strides {:?} for shape {:?}); \
                 call .contiguous() first",
                self.strides(),
                self.shape().dims()
            )));
        }
        Ok((self.storage_offset, self.numel()))
    }
    /// Error raised when the backing storage is smaller than this tensor claims.
    fn window_error(available: usize, offset: usize, len: usize) -> TorshError {
        TorshError::InvalidOperation(format!(
            "tensor needs storage elements {}..{} but only {} are available",
            offset,
            offset + len,
            available
        ))
    }
    /// Execute a function with zero-copy access to tensor data (mutable)
    ///
    /// This enables in-place SIMD operations without memory copies.
    ///
    /// # Arguments
    /// * `f` - Closure that receives `&mut [T]` and returns `Result<R>`
    ///
    /// # Returns
    /// Result from the closure
    ///
    /// # Performance
    /// - Zero memory copies for InMemory storage
    /// - Not supported for MemoryMapped or Aligned storage (returns error)
    ///
    /// # Examples
    /// ```ignore
    /// // In-place SIMD operation without copies
    /// tensor.with_data_slice_mut(|data| {
    ///     other_tensor.with_data_slice(|other_data| {
    ///         // Zero-copy in-place SIMD addition
    ///         for (x, y) in data.iter_mut().zip(other_data.iter()) {
    ///             *x = *x + *y;
    ///         }
    ///         Ok(())
    ///     })
    /// })?;
    /// ```
    pub fn with_data_slice_mut<R, F>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&mut [T]) -> Result<R>,
        T: Copy,
    {
        let (offset, len) = self.contiguous_window("with_data_slice_mut")?;
        self.storage.with_slice_mut(|slice| {
            let available = slice.len();
            let window = slice
                .get_mut(offset..offset + len)
                .ok_or_else(|| Self::window_error(available, offset, len))?;
            f(window)
        })
    }
    /// Create a tensor of ones with the same shape as this tensor
    pub fn ones_like(&self) -> Result<Self> {
        let ones_data = vec![T::one(); self.numel()];
        Self::from_data(ones_data, self.shape().dims().to_vec(), self.device)
    }
    /// Create a tensor of zeros with the same shape as this tensor
    pub fn zeros_like(&self) -> Result<Self> {
        let zeros_data = vec![T::zero(); self.numel()];
        Self::from_data(zeros_data, self.shape().dims().to_vec(), self.device)
    }
    /// Enable or disable gradient tracking for this tensor.
    ///
    /// This method marks the tensor as requiring (or not requiring) gradient computation
    /// during the backward pass. It consumes `self` and returns the modified tensor,
    /// allowing for method chaining in the builder pattern.
    ///
    /// # Parameters
    ///
    /// - `requires_grad`: If `true`, gradients will be computed and stored for this tensor
    ///   during backward passes. If `false`, gradients will not be computed.
    ///
    /// # Important Notes
    ///
    /// - Only **leaf tensors** (tensors created directly, not from operations) store gradients
    /// - Intermediate tensors in the computation graph don't store gradients, only leaf tensors do
    /// - Setting `requires_grad=true` enables tracking for all subsequent operations
    /// - You should typically set this for model parameters and input data you want to optimize
    ///
    /// # Examples
    ///
    /// ## Basic Usage
    ///
    /// ```ignore
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// // Create tensor and enable gradient tracking
    /// let x = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], DeviceType::Cpu)?
    ///     .requires_grad_(true);
    ///
    /// assert!(x.requires_grad());
    /// # Ok::<(), torsh_core::error::TorshError>(())
    /// ```
    ///
    /// ## Neural Network Parameters
    ///
    /// ```ignore
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// // Create trainable weights
    /// let weights = Tensor::randn(&[784, 128], DeviceType::Cpu)?
    ///     .requires_grad_(true);
    ///
    /// let bias = Tensor::zeros(&[128], DeviceType::Cpu)?
    ///     .requires_grad_(true);
    ///
    /// // Use in forward pass
    /// let input = Tensor::randn(&[32, 784], DeviceType::Cpu)?;
    /// let output = input.matmul(&weights)?.add(&bias)?;
    /// # Ok::<(), torsh_core::error::TorshError>(())
    /// ```
    ///
    /// ## Freezing Layers (Transfer Learning)
    ///
    /// ```ignore
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// // Pretrained layer - don't compute gradients
    /// let frozen_weights = Tensor::randn(&[512, 256], DeviceType::Cpu)?
    ///     .requires_grad_(false);
    ///
    /// // New layer - compute gradients
    /// let trainable_weights = Tensor::randn(&[256, 10], DeviceType::Cpu)?
    ///     .requires_grad_(true);
    /// # Ok::<(), torsh_core::error::TorshError>(())
    /// ```
    ///
    /// ## Method Chaining
    ///
    /// ```ignore
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// // Chain multiple operations
    /// let x = Tensor::ones(&[5], DeviceType::Cpu)?
    ///     .requires_grad_(true)
    ///     .mul_scalar(2.0)?
    ///     .add_scalar(1.0)?;
    ///
    /// assert!(x.requires_grad());
    /// # Ok::<(), torsh_core::error::TorshError>(())
    /// ```
    ///
    /// # Gradient Propagation
    ///
    /// When a tensor has `requires_grad=true`, all operations on it will also
    /// have `requires_grad=true`, building the computation graph:
    ///
    /// ```ignore
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// let x = Tensor::from_data(vec![2.0], vec![1], DeviceType::Cpu)?
    ///     .requires_grad_(true);
    ///
    /// let y = x.pow(2.0)?;        // y also requires grad
    /// let z = y.mul_scalar(3.0)?;  // z also requires grad
    /// let w = z.add_scalar(1.0)?;  // w also requires grad
    ///
    /// assert!(y.requires_grad());
    /// assert!(z.requires_grad());
    /// assert!(w.requires_grad());
    /// # Ok::<(), torsh_core::error::TorshError>(())
    /// ```
    ///
    /// # See Also
    ///
    /// - [`backward()`](#method.backward) - Compute gradients
    /// - [`grad()`](#method.grad) - Access computed gradients
    /// - [`detach()`](#method.detach) - Create copy without gradient tracking
    pub fn requires_grad_(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        self
    }
    /// Check if this tensor requires gradient computation.
    ///
    /// Returns `true` if gradients will be computed for this tensor during
    /// backward passes, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// let x = Tensor::ones(&[3], DeviceType::Cpu)?.requires_grad_(true);
    /// assert!(x.requires_grad());
    ///
    /// let y = Tensor::ones(&[3], DeviceType::Cpu)?;
    /// assert!(!y.requires_grad());
    /// # Ok::<(), torsh_core::error::TorshError>(())
    /// ```
    ///
    /// # See Also
    ///
    /// - [`requires_grad_()`](#method.requires_grad_) - Set gradient tracking
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }
    /// Set gradient tensor
    #[allow(dead_code)]
    pub fn set_grad(&self, grad: Option<Tensor<T>>) {
        let mut grad_lock = self.grad.write_or_recover();
        *grad_lock = grad;
    }
    /// Get mutable access to gradient
    pub fn grad_mut(&mut self) -> Option<&mut Self> {
        None
    }
    /// 🚀 Enhanced device transfer with multi-backend GPU support
    /// Automatically selects optimal transfer strategy and backend
    pub fn to<D: Into<DeviceType>>(self, device: D) -> Result<Self> {
        let target_device = device.into();
        if target_device == self.device {
            return Ok(self);
        }
        self.to_device(target_device)
    }
    /// 🚀 Advanced multi-GPU distribution for parallel processing
    /// Automatically distributes tensor across multiple GPUs with optimal strategy
    /// Note: Actual implementation is in backend_integration.rs module
    #[cfg(feature = "gpu")]
    #[allow(dead_code)]
    pub fn distribute_multi_gpu_wrapper(&self, gpu_count: usize) -> Result<Vec<Self>> {
        let _ = gpu_count;
        Err(TorshError::InvalidArgument(
            "Use the distribute_multi_gpu method from backend_integration module directly"
                .to_string(),
        ))
    }
    #[cfg(not(feature = "gpu"))]
    #[allow(dead_code)]
    pub fn distribute_multi_gpu_wrapper(&self, gpu_count: usize) -> Result<Vec<Self>> {
        let _ = gpu_count;
        Err(TorshError::UnsupportedOperation {
            op: format!("multi-GPU distribution ({} GPUs)", gpu_count),
            dtype: "GPU feature not enabled".to_string(),
        })
    }
    /// 🚀 Get optimal GPU backend for current hardware
    #[cfg(feature = "gpu")]
    pub fn get_optimal_gpu_backend() -> Option<GpuBackendType> {
        let backends = [
            GpuBackendType::Cuda,
            GpuBackendType::Metal,
            GpuBackendType::Rocm,
            GpuBackendType::WebGpu,
            GpuBackendType::OpenCl,
        ];
        for backend in &backends {
            if Self::is_backend_available(backend.clone()) {
                return Some(backend.clone());
            }
        }
        None
    }
    /// Check if a specific GPU backend is available
    /// TODO: Temporarily disabled - backend types not yet available in scirs2_core
    #[cfg(feature = "gpu")]
    #[allow(dead_code)]
    fn is_backend_available(_backend: GpuBackendType) -> bool {
        false
    }
    /// 🚀 Create tensor on optimal GPU device
    #[cfg(feature = "gpu")]
    pub fn zeros_gpu(shape: &[usize]) -> Result<Self> {
        let optimal_backend =
            Self::get_optimal_gpu_backend().ok_or_else(|| TorshError::UnsupportedOperation {
                op: "GPU tensor creation".to_string(),
                dtype: "No GPU backend available".to_string(),
            })?;
        let device = match optimal_backend {
            GpuBackendType::Cuda => DeviceType::Cuda(0),
            GpuBackendType::Metal => DeviceType::Metal(0),
            GpuBackendType::WebGpu => DeviceType::Wgpu(0),
            _ => {
                return Err(TorshError::UnsupportedOperation {
                    op: "GPU tensor creation".to_string(),
                    dtype: "Backend not supported for tensor creation".to_string(),
                });
            }
        };
        Self::zeros(shape, device)
    }
    /// 🚀 Create tensor on optimal GPU device filled with ones
    #[cfg(feature = "gpu")]
    pub fn ones_gpu(shape: &[usize]) -> Result<Self> {
        let optimal_backend =
            Self::get_optimal_gpu_backend().ok_or_else(|| TorshError::UnsupportedOperation {
                op: "GPU tensor creation".to_string(),
                dtype: "No GPU backend available".to_string(),
            })?;
        let device = match optimal_backend {
            GpuBackendType::Cuda => DeviceType::Cuda(0),
            GpuBackendType::Metal => DeviceType::Metal(0),
            GpuBackendType::WebGpu => DeviceType::Wgpu(0),
            _ => {
                return Err(TorshError::UnsupportedOperation {
                    op: "GPU tensor creation".to_string(),
                    dtype: "Backend not supported for tensor creation".to_string(),
                });
            }
        };
        Self::ones(shape, device)
    }
    /// Create a detached copy of this tensor that doesn't track gradients.
    ///
    /// This method creates a new tensor with the same data as the original, but with
    /// `requires_grad=false`. The detached tensor is not part of the computation graph
    /// and will not participate in gradient computation, even if the original tensor did.
    ///
    /// # Use Cases
    ///
    /// - **Inference with trained parameters**: Use detached weights for forward-only computation
    /// - **Custom gradient logic**: Manually control which tensors participate in backprop
    /// - **Debugging**: Inspect intermediate values without affecting the gradient flow
    /// - **Mixed training**: Some computations need gradients, others don't
    /// - **Memory optimization**: Reduce memory usage by not tracking gradients for certain operations
    ///
    /// # Important Notes
    ///
    /// - The returned tensor shares no gradient history with the original
    /// - Operations on the detached tensor will also have `requires_grad=false`
    /// - This creates a copy of the data (not a view) - consider memory implications
    /// - The detached tensor is a "leaf" tensor with no computation history
    ///
    /// # Examples
    ///
    /// ## Basic Detachment
    ///
    /// ```ignore
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// let x = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], DeviceType::Cpu)?
    ///     .requires_grad_(true);
    ///
    /// // Create detached copy
    /// let x_detached = x.detach();
    ///
    /// // Original tracks gradients, detached does not
    /// assert!(x.requires_grad());
    /// assert!(!x_detached.requires_grad());
    ///
    /// // Operations on detached tensor don't track gradients
    /// let y_detached = x_detached.pow(2.0)?;
    /// assert!(!y_detached.requires_grad());
    /// # Ok::<(), torsh_core::error::TorshError>(())
    /// ```
    ///
    /// ## Stopping Gradient Flow
    ///
    /// ```ignore
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// // This pattern prevents gradients from flowing through certain operations
    /// let x = Tensor::from_data(vec![2.0], vec![1], DeviceType::Cpu)?
    ///     .requires_grad_(true);
    ///
    /// // Compute something we want to use but not backpropagate through
    /// let intermediate = x.pow(2.0)?;
    /// let intermediate_detached = intermediate.detach();
    ///
    /// // Use detached version in further computation
    /// let y = intermediate_detached.mul_scalar(3.0)?;
    /// let loss = y.sum()?;
    ///
    /// // Backward won't compute gradients for x
    /// // (because gradient flow is stopped at detach point)
    /// # Ok::<(), torsh_core::error::TorshError>(())
    /// ```
    ///
    /// ## Target Tensor Pattern (No Gradients for Labels)
    ///
    /// ```ignore
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// // Predictions need gradients
    /// let predictions = Tensor::randn(&[32, 10], DeviceType::Cpu)?
    ///     .requires_grad_(true);
    ///
    /// // Target labels should NOT have gradients
    /// let targets = Tensor::from_data(
    ///     vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ///     vec![1, 10],
    ///     DeviceType::Cpu
    /// )?.detach();  // Explicitly detach to ensure no gradients
    ///
    /// assert!(predictions.requires_grad());
    /// assert!(!targets.requires_grad());
    ///
    /// // Compute loss - only predictions will have gradients
    /// let diff = predictions.sub(&targets)?;
    /// let loss = diff.pow(2.0)?.mean()?;
    /// loss.backward()?;
    /// # Ok::<(), torsh_core::error::TorshError>(())
    /// ```
    ///
    /// ## Debugging Intermediate Values
    ///
    /// ```ignore
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// let x = Tensor::randn(&[100], DeviceType::Cpu)?
    ///     .requires_grad_(true);
    ///
    /// // Complex computation
    /// let y = x.pow(2.0)?.add_scalar(1.0)?;
    ///
    /// // Detach to inspect values without affecting gradient computation
    /// let y_values = y.detach();
    /// let max_val = y_values.to_vec()?.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    /// println!("Max value: {}", max_val);
    ///
    /// // Continue with original gradient-tracking tensor
    /// let loss = y.sum()?;
    /// loss.backward()?;
    /// # Ok::<(), torsh_core::error::TorshError>(())
    /// ```
    ///
    /// ## Feature Extraction (Transfer Learning)
    ///
    /// ```ignore
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// // Pretrained model features - don't need gradients
    /// fn pretrained_feature_extractor(input: &Tensor<f32>) -> Result<Tensor<f32>, torsh_core::error::TorshError> {
    ///     // Complex pretrained network...
    ///     let features = input.matmul(&Tensor::randn(&[784, 512], DeviceType::Cpu)?)?;
    ///     // Detach to stop gradient flow
    ///     Ok(features.detach())
    /// }
    ///
    /// // New trainable classifier head
    /// let input = Tensor::randn(&[32, 784], DeviceType::Cpu)?;
    /// let features = pretrained_feature_extractor(&input)?;
    ///
    /// // Only this part will have gradients
    /// let weights = Tensor::randn(&[512, 10], DeviceType::Cpu)?
    ///     .requires_grad_(true);
    /// let output = features.matmul(&weights)?;
    /// # Ok::<(), torsh_core::error::TorshError>(())
    /// ```
    ///
    /// ## Memory Optimization Example
    ///
    /// ```ignore
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// let x = Tensor::randn(&[1000, 1000], DeviceType::Cpu)?
    ///     .requires_grad_(true);
    ///
    /// // Large intermediate computation
    /// let intermediate = x.matmul(&x.transpose(&[1, 0])?)?;
    ///
    /// // If we only need the values, not gradients
    /// let result = intermediate.detach();
    ///
    /// // Now intermediate computation graph can be freed
    /// drop(intermediate);
    ///
    /// // Use result without gradient tracking
    /// let final_result = result.mul_scalar(0.5)?;
    /// # Ok::<(), torsh_core::error::TorshError>(())
    /// ```
    ///
    /// ## Comparison with no_grad Context
    ///
    /// ```ignore
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    /// use torsh_autograd::guards::no_grad;
    ///
    /// let x = Tensor::randn(&[5], DeviceType::Cpu)?
    ///     .requires_grad_(true);
    ///
    /// // Method 1: Detach - creates a copy
    /// let y1 = x.detach().pow(2.0)?;
    /// assert!(!y1.requires_grad());
    ///
    /// // Method 2: no_grad context - affects all operations in scope
    /// let y2 = {
    ///     let _guard = no_grad();
    ///     x.pow(2.0)?
    /// };
    /// assert!(!y2.requires_grad());
    ///
    /// // Note: detach() is for selective detachment of specific tensors
    /// // no_grad() is for disabling gradient tracking in a code block
    /// # Ok::<(), torsh_core::error::TorshError>(())
    /// ```
    ///
    /// # Detach vs Clone
    ///
    /// - **`detach()`**: Creates a copy with `requires_grad=false`
    /// - **`clone()`**: Creates a copy preserving `requires_grad` status
    ///
    /// ```ignore
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// let x = Tensor::ones(&[3], DeviceType::Cpu)?
    ///     .requires_grad_(true);
    ///
    /// let x_clone = x.clone();
    /// let x_detach = x.detach();
    ///
    /// assert!(x_clone.requires_grad());   // Preserves gradient tracking
    /// assert!(!x_detach.requires_grad()); // Disables gradient tracking
    /// # Ok::<(), torsh_core::error::TorshError>(())
    /// ```
    ///
    /// # See Also
    ///
    /// - [`requires_grad_()`](#method.requires_grad_) - Set gradient tracking
    /// - `torsh_autograd::guards::no_grad()` - Disable gradients in a scope
    /// - [`backward()`](#method.backward) - Compute gradients
    pub fn detach(&self) -> Self {
        let mut detached = self.clone();
        detached.requires_grad = false;
        detached
    }
    /// Access the computed gradient for this tensor.
    ///
    /// Returns `Some(gradient_tensor)` if gradients have been computed via
    /// [`backward()`](#method.backward), or `None` if no gradients exist yet.
    ///
    /// # Important Notes
    ///
    /// - Only **leaf tensors** store gradients (tensors created directly, not from operations)
    /// - You must call [`backward()`](#method.backward) on a scalar output before gradients exist
    /// - Gradients accumulate across multiple `backward()` calls unless cleared with [`zero_grad()`](#method.zero_grad)
    /// - The returned gradient tensor has the same shape as the original tensor
    ///
    /// # Returns
    ///
    /// - `Some(Tensor)` - The gradient tensor if computed
    /// - `None` - If backward has not been called or gradients were cleared
    ///
    /// # Examples
    ///
    /// ## Basic Gradient Access
    ///
    /// ```ignore
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// let x = Tensor::from_data(vec![2.0], vec![1], DeviceType::Cpu)?
    ///     .requires_grad_(true);
    ///
    /// // Before backward: no gradient
    /// assert!(x.grad().is_none());
    ///
    /// // Compute gradient
    /// let y = x.pow(2.0)?;
    /// y.backward()?;
    ///
    /// // After backward: gradient exists
    /// let grad = x.grad().expect("gradient should exist");
    /// assert_eq!(grad.item()?, 4.0);  // dy/dx = 2x = 4
    /// # Ok::<(), torsh_core::error::TorshError>(())
    /// ```
    ///
    /// ## Multi-Dimensional Gradients
    ///
    /// ```ignore
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// let x = Tensor::from_data(
    ///     vec![1.0, 2.0, 3.0, 4.0],
    ///     vec![2, 2],
    ///     DeviceType::Cpu
    /// )?.requires_grad_(true);
    ///
    /// // Loss function: sum of squares
    /// let y = x.pow(2.0)?;
    /// let loss = y.sum()?;
    /// loss.backward()?;
    ///
    /// // Gradient has same shape as input
    /// let grad = x.grad().expect("gradient should exist");
    /// assert_eq!(grad.shape().dims(), &[2, 2]);
    /// // Gradient values: [2, 4, 6, 8] (2 * each input)
    /// # Ok::<(), torsh_core::error::TorshError>(())
    /// ```
    ///
    /// ## Using Gradients for Optimization
    ///
    /// ```ignore
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// let mut weights = Tensor::randn(&[10], DeviceType::Cpu)?
    ///     .requires_grad_(true);
    ///
    /// let learning_rate = 0.01;
    ///
    /// // Training loop
    /// for step in 0..100 {
    ///     // Forward pass
    ///     let output = weights.mul_scalar(2.0)?;
    ///     let loss = output.sum()?;
    ///
    ///     // Backward pass
    ///     loss.backward()?;
    ///
    ///     // Get gradients and update weights
    ///     if let Some(grad) = weights.grad() {
    ///         // weights = weights - learning_rate * grad
    ///         let update = grad.mul_scalar(learning_rate)?;
    ///         weights = weights.sub(&update)?;
    ///     }
    ///
    ///     // Clear gradients for next iteration
    ///     weights.zero_grad();
    /// }
    /// # Ok::<(), torsh_core::error::TorshError>(())
    /// ```
    ///
    /// ## Gradient Accumulation Check
    ///
    /// ```ignore
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// let mut x = Tensor::from_data(vec![1.0], vec![1], DeviceType::Cpu)?
    ///     .requires_grad_(true);
    ///
    /// // First backward pass
    /// let y1 = x.pow(2.0)?;
    /// y1.backward()?;
    /// let grad1 = x.grad().expect("gradient should exist").item()?;
    /// assert_eq!(grad1, 2.0);
    ///
    /// // Second backward pass without zeroing - accumulates!
    /// let y2 = x.pow(2.0)?;
    /// y2.backward()?;
    /// let grad2 = x.grad().expect("gradient should exist").item()?;
    /// assert_eq!(grad2, 4.0);  // 2.0 + 2.0 = 4.0
    ///
    /// // Clear for next iteration
    /// x.zero_grad();
    /// assert!(x.grad().is_none());
    /// # Ok::<(), torsh_core::error::TorshError>(())
    /// ```
    ///
    /// # Common Patterns
    ///
    /// ## Safe Gradient Extraction
    ///
    /// ```ignore
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    /// # let x = Tensor::from_data(vec![1.0], vec![1], DeviceType::Cpu)?
    /// #     .requires_grad_(true);
    /// # let y = x.pow(2.0)?;
    /// # y.backward()?;
    ///
    /// // Option 1: Unwrap (panics if no gradient)
    /// let grad = x.grad().expect("gradient should exist");
    ///
    /// // Option 2: Pattern matching (safer)
    /// match x.grad() {
    ///     Some(grad) => {
    ///         // Use gradient
    ///         println!("Gradient: {:?}", grad);
    ///     }
    ///     None => {
    ///         println!("No gradient computed yet");
    ///     }
    /// }
    ///
    /// // Option 3: if-let pattern
    /// if let Some(grad) = x.grad() {
    ///     // Use gradient
    /// }
    /// # Ok::<(), torsh_core::error::TorshError>(())
    /// ```
    ///
    /// # See Also
    ///
    /// - [`backward()`](#method.backward) - Compute gradients
    /// - [`zero_grad()`](#method.zero_grad) - Clear gradients
    /// - [`has_grad()`](#method.has_grad) - Check if gradients exist
    /// - [`requires_grad_()`](#method.requires_grad_) - Enable gradient tracking
    pub fn grad(&self) -> Option<Self> {
        let grad_lock = self.grad.read_or_recover();
        grad_lock.as_ref().cloned()
    }
    /// Check if this tensor has a computed gradient.
    ///
    /// Returns `true` if gradients have been computed and stored, `false` otherwise.
    /// This is equivalent to `tensor.grad().is_some()` but more explicit.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// let x = Tensor::from_data(vec![2.0], vec![1], DeviceType::Cpu)?
    ///     .requires_grad_(true);
    ///
    /// assert!(!x.has_grad());  // No gradient yet
    ///
    /// let y = x.pow(2.0)?;
    /// y.backward()?;
    ///
    /// assert!(x.has_grad());   // Gradient now exists
    /// # Ok::<(), torsh_core::error::TorshError>(())
    /// ```
    ///
    /// # See Also
    ///
    /// - [`grad()`](#method.grad) - Access the gradient tensor
    pub fn has_grad(&self) -> bool {
        let grad_lock = self.grad.read_or_recover();
        grad_lock.is_some()
    }
    /// Clear the gradient for this tensor.
    ///
    /// This method sets the gradient to `None`, effectively resetting it. You should call
    /// this method between training iterations to prevent gradient accumulation when you
    /// don't want it.
    ///
    /// # When to Use
    ///
    /// - **After optimizer step**: Clear gradients before the next training iteration
    /// - **Between validation runs**: Ensure clean state for evaluation
    /// - **After gradient accumulation**: Clear after applying accumulated gradients
    ///
    /// # Important Notes
    ///
    /// - Gradients **accumulate** by default across multiple `backward()` calls
    /// - Always call `zero_grad()` between training iterations unless you explicitly
    ///   want gradient accumulation
    /// - This only clears the gradient; it doesn't affect `requires_grad` status
    ///
    /// # Examples
    ///
    /// ## Standard Training Loop
    ///
    /// ```ignore
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// let mut weights = Tensor::randn(&[10], DeviceType::Cpu)?
    ///     .requires_grad_(true);
    ///
    /// for epoch in 0..100 {
    ///     // Forward pass
    ///     let output = weights.mul_scalar(2.0)?;
    ///     let loss = output.sum()?;
    ///
    ///     // Backward pass
    ///     loss.backward()?;
    ///
    ///     // Update weights (simplified)
    ///     if let Some(grad) = weights.grad() {
    ///         weights = weights.sub(&grad.mul_scalar(0.01)?)?;
    ///     }
    ///
    ///     // CRITICAL: Clear gradients for next iteration
    ///     weights.zero_grad();
    /// }
    /// # Ok::<(), torsh_core::error::TorshError>(())
    /// ```
    ///
    /// ## Gradient Accumulation Pattern
    ///
    /// ```ignore
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// let mut model_params = Tensor::randn(&[100], DeviceType::Cpu)?
    ///     .requires_grad_(true);
    ///
    /// let accumulation_steps = 4;
    ///
    /// // Zero gradients at start
    /// model_params.zero_grad();
    ///
    /// for step in 0..accumulation_steps {
    ///     // Forward and backward for mini-batch
    ///     let output = model_params.mul_scalar(2.0)?;
    ///     let loss = output.sum()?;
    ///     loss.backward()?;
    ///
    ///     // DON'T zero gradients here - let them accumulate
    /// }
    ///
    /// // After accumulation, update weights
    /// if let Some(grad) = model_params.grad() {
    ///     // Scale by number of accumulation steps
    ///     let scaled_grad = grad.div_scalar(accumulation_steps as f32)?;
    ///     model_params = model_params.sub(&scaled_grad.mul_scalar(0.01)?)?;
    /// }
    ///
    /// // NOW clear gradients for next accumulation cycle
    /// model_params.zero_grad();
    /// # Ok::<(), torsh_core::error::TorshError>(())
    /// ```
    ///
    /// ## Multiple Parameter Update
    ///
    /// ```ignore
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// let mut weights = Tensor::randn(&[10, 5], DeviceType::Cpu)?
    ///     .requires_grad_(true);
    /// let mut bias = Tensor::zeros(&[5], DeviceType::Cpu)?
    ///     .requires_grad_(true);
    ///
    /// // Training step
    /// let input = Tensor::randn(&[32, 10], DeviceType::Cpu)?;
    /// let output = input.matmul(&weights)?.add(&bias)?;
    /// let loss = output.sum()?;
    ///
    /// loss.backward()?;
    ///
    /// // Update both parameters
    /// // ... (weight updates)
    ///
    /// // Clear gradients for both
    /// weights.zero_grad();
    /// bias.zero_grad();
    /// # Ok::<(), torsh_core::error::TorshError>(())
    /// ```
    ///
    /// ## Validation Without Gradient Tracking
    ///
    /// ```ignore
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    /// use torsh_autograd::guards::no_grad;
    ///
    /// let weights = Tensor::randn(&[10], DeviceType::Cpu)?
    ///     .requires_grad_(true);
    ///
    /// // During validation, use no_grad guard
    /// let validation_loss = {
    ///     let _guard = no_grad();
    ///     let output = weights.mul_scalar(2.0)?;
    ///     output.sum()?
    /// };
    ///
    /// // No gradients computed during validation, so nothing to zero
    /// // But good practice to ensure clean state
    /// // weights.zero_grad();
    /// # Ok::<(), torsh_core::error::TorshError>(())
    /// ```
    ///
    /// # Common Mistake
    ///
    /// ## Forgetting to Zero Gradients
    ///
    /// ```ignore
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// let mut x = Tensor::from_data(vec![1.0], vec![1], DeviceType::Cpu)?
    ///     .requires_grad_(true);
    ///
    /// // Iteration 1
    /// let y1 = x.pow(2.0)?;
    /// y1.backward()?;
    /// let grad1 = x.grad().expect("gradient should exist").item()?;
    /// println!("Grad 1: {}", grad1);  // 2.0
    ///
    /// // Iteration 2 - FORGOT TO ZERO!
    /// let y2 = x.pow(2.0)?;
    /// y2.backward()?;
    /// let grad2 = x.grad().expect("gradient should exist").item()?;
    /// println!("Grad 2: {}", grad2);  // 4.0 (WRONG! Should be 2.0)
    ///
    /// // Correct approach:
    /// x.zero_grad();  // Clear between iterations
    /// let y3 = x.pow(2.0)?;
    /// y3.backward()?;
    /// let grad3 = x.grad().expect("gradient should exist").item()?;
    /// println!("Grad 3: {}", grad3);  // 2.0 (CORRECT!)
    /// # Ok::<(), torsh_core::error::TorshError>(())
    /// ```
    ///
    /// # See Also
    ///
    /// - [`grad()`](#method.grad) - Access computed gradients
    /// - [`backward()`](#method.backward) - Compute gradients
    /// - [`has_grad()`](#method.has_grad) - Check if gradients exist
    pub fn zero_grad(&mut self) {
        let mut grad_lock = self.grad.write_or_recover();
        *grad_lock = None;
    }
    /// Computes gradients for all tensors in the computation graph.
    ///
    /// This method performs backpropagation through the computation graph, computing
    /// gradients for all tensors with `requires_grad=true`. The backward pass starts
    /// from this tensor (which must be a scalar) and propagates gradients back through
    /// the computational graph to all leaf tensors.
    ///
    /// # Requirements
    ///
    /// - This tensor **must be a scalar** (single element) - use `.sum()` or similar
    ///   reduction operations to create a scalar from multi-dimensional tensors
    /// - This tensor must have `requires_grad=true`
    /// - All tensors in the computation graph are retained during backward pass
    ///
    /// # How It Works
    ///
    /// The backward pass:
    /// 1. Starts with a gradient of 1.0 for the output (scalar) tensor
    /// 2. Traverses the computation graph in reverse topological order
    /// 3. Applies the chain rule at each operation node
    /// 4. Accumulates gradients at leaf nodes (input tensors)
    ///
    /// # Gradient Accumulation
    ///
    /// If you call `backward()` multiple times without zeroing gradients, the
    /// gradients will **accumulate** (add together). This is useful for:
    /// - Gradient accumulation across mini-batches
    /// - Computing gradients for multiple outputs
    ///
    /// Use [`zero_grad()`](#method.zero_grad) to clear gradients between iterations.
    ///
    /// # Examples
    ///
    /// ## Basic Gradient Computation
    ///
    /// ```ignore
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// // Create input tensor with gradient tracking enabled
    /// let x = Tensor::from_data(vec![2.0f32], vec![1], DeviceType::Cpu)?
    ///     .requires_grad_(true);
    ///
    /// // Forward pass: y = x^2
    /// let y = x.pow(2.0)?;
    ///
    /// // Backward pass: compute dy/dx = 2x
    /// y.backward()?;
    ///
    /// // Access gradient: should be 2 * 2.0 = 4.0
    /// let grad = x.grad().expect("gradient should exist");
    /// assert_eq!(grad.item()?, 4.0);
    /// # Ok::<(), torsh_core::error::TorshError>(())
    /// ```
    ///
    /// ## Multi-Variable Function
    ///
    /// ```ignore
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// // f(x, y) = x^2 + 2xy + y^2
    /// let x = Tensor::from_data(vec![3.0f32], vec![1], DeviceType::Cpu)?
    ///     .requires_grad_(true);
    /// let y = Tensor::from_data(vec![4.0f32], vec![1], DeviceType::Cpu)?
    ///     .requires_grad_(true);
    ///
    /// // Forward pass
    /// let x_squared = x.pow(2.0)?;
    /// let xy = x.mul(&y)?;
    /// let two_xy = xy.mul_scalar(2.0)?;
    /// let y_squared = y.pow(2.0)?;
    /// let result = x_squared.add(&two_xy)?.add(&y_squared)?;
    ///
    /// // Backward pass
    /// result.backward()?;
    ///
    /// // df/dx = 2x + 2y = 6 + 8 = 14
    /// // df/dy = 2x + 2y = 6 + 8 = 14
    /// let grad_x = x.grad().expect("gradient should exist");
    /// let grad_y = y.grad().expect("gradient should exist");
    /// # Ok::<(), torsh_core::error::TorshError>(())
    /// ```
    ///
    /// ## Vector to Scalar (Loss Function)
    ///
    /// ```ignore
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// // Create predictions and targets
    /// let predictions = Tensor::from_data(
    ///     vec![0.8, 0.6, 0.9],
    ///     vec![3],
    ///     DeviceType::Cpu
    /// )?.requires_grad_(true);
    ///
    /// let targets = Tensor::from_data(
    ///     vec![1.0, 0.0, 1.0],
    ///     vec![3],
    ///     DeviceType::Cpu
    /// )?;
    ///
    /// // Compute MSE loss: mean((pred - target)^2)
    /// let diff = predictions.sub(&targets)?;
    /// let squared = diff.pow(2.0)?;
    /// let loss = squared.mean()?; // Reduces to scalar
    ///
    /// // Compute gradients
    /// loss.backward()?;
    ///
    /// // Gradient shows direction to reduce loss
    /// let grad = predictions.grad().expect("gradient should exist");
    /// # Ok::<(), torsh_core::error::TorshError>(())
    /// ```
    ///
    /// ## Gradient Accumulation Pattern
    ///
    /// ```ignore
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// let mut weights = Tensor::from_data(
    ///     vec![0.5f32; 10],
    ///     vec![10],
    ///     DeviceType::Cpu
    /// )?.requires_grad_(true);
    ///
    /// // Accumulate gradients over multiple batches
    /// for batch in 0..4 {
    ///     // Forward pass for this batch
    ///     let output = weights.mul_scalar(2.0)?;
    ///     let loss = output.sum()?;
    ///
    ///     // Backward pass - gradients accumulate
    ///     loss.backward()?;
    ///
    ///     // Don't zero gradients yet
    /// }
    ///
    /// // After accumulation, update weights
    /// // optimizer.step();
    ///
    /// // Clear gradients for next iteration
    /// weights.zero_grad();
    /// # Ok::<(), torsh_core::error::TorshError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The tensor doesn't have `requires_grad=true`
    /// - The tensor is not a scalar (has more than 1 element)
    /// - An error occurs during gradient computation
    ///
    /// # Common Pitfalls
    ///
    /// ## 1. Forgetting to Create Scalar Output
    ///
    /// ```rust,should_panic
    /// # use torsh_tensor::Tensor;
    /// # use torsh_core::device::DeviceType;
    /// let x = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], DeviceType::Cpu)
    ///     .expect("tensor creation should succeed")
    ///     .requires_grad_(true);
    /// let y = x.pow(2.0).expect("pow should succeed");
    ///
    /// // ERROR: y has 3 elements, not a scalar!
    /// y.backward().expect("backward pass should succeed");  // This will panic
    /// ```
    ///
    /// **Solution**: Use reduction operations:
    /// ```ignore
    /// # use torsh_tensor::Tensor;
    /// # use torsh_core::device::DeviceType;
    /// # let x = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], DeviceType::Cpu)
    /// #     .expect("tensor creation should succeed")
    /// #     .requires_grad_(true);
    /// let y = x.pow(2.0)?.sum()?;  // Reduce to scalar
    /// y.backward()?;
    /// # Ok::<(), torsh_core::error::TorshError>(())
    /// ```
    ///
    /// ## 2. Forgetting to Enable Gradient Tracking
    ///
    /// ```rust,should_panic
    /// # use torsh_tensor::Tensor;
    /// # use torsh_core::device::DeviceType;
    /// let x = Tensor::from_data(vec![2.0], vec![1], DeviceType::Cpu).expect("tensor creation should succeed");
    /// // Forgot .requires_grad_(true)
    /// let y = x.pow(2.0).expect("pow should succeed");
    /// y.backward().expect("backward pass should succeed");  // ERROR: requires_grad not set
    /// ```
    ///
    /// ## 3. Not Zeroing Gradients Between Training Steps
    ///
    /// ```ignore
    /// # use torsh_tensor::Tensor;
    /// # use torsh_core::device::DeviceType;
    /// let mut x = Tensor::from_data(vec![1.0], vec![1], DeviceType::Cpu)?
    ///     .requires_grad_(true);
    ///
    /// for epoch in 0..3 {
    ///     let y = x.pow(2.0)?.sum()?;
    ///     y.backward()?;
    ///
    ///     // Gradients accumulate without this!
    ///     x.zero_grad();  // IMPORTANT: Clear gradients
    /// }
    /// # Ok::<(), torsh_core::error::TorshError>(())
    /// ```
    ///
    /// # See Also
    ///
    /// - [`requires_grad_()`](#method.requires_grad_) - Enable gradient tracking
    /// - [`grad()`](#method.grad) - Access computed gradients
    /// - [`zero_grad()`](#method.zero_grad) - Clear gradients
    /// - [`detach()`](#method.detach) - Create non-differentiable copy
    pub fn backward(&self) -> Result<()>
    where
        T: FloatElement
            + Copy
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + Clone
            + std::fmt::Debug,
        f32: From<T>,
    {
        self.backward_with_grad(None)
    }
    /// Backward pass seeded with an explicit gradient.
    ///
    /// This is the vector-Jacobian entry point: pass the gradient of the loss
    /// with respect to *this* tensor and it is propagated through the recorded
    /// graph, accumulating into every reachable leaf's `grad` slot.
    ///
    /// # Parameters
    ///
    /// - `gradient`: seed gradient, which must have exactly this tensor's shape.
    ///   `None` is only valid for scalar tensors, where a gradient of `1` is used
    ///   (this is what [`backward()`](#method.backward) does).
    ///
    /// # Errors
    ///
    /// - the tensor does not require gradients
    /// - `gradient` is `None` and the tensor is not a scalar
    /// - `gradient` has a different shape than this tensor
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// let x = Tensor::from_data(vec![1.0, 2.0], vec![2], DeviceType::Cpu)?
    ///     .requires_grad_(true);
    /// let y = x.mul(&x)?;                       // non-scalar output
    /// let seed = Tensor::ones(&[2], DeviceType::Cpu)?;
    /// y.backward_with_grad(Some(&seed))?;       // dy/dx = 2x
    /// # Ok::<(), torsh_core::error::TorshError>(())
    /// ```
    pub fn backward_with_grad(&self, gradient: Option<&Self>) -> Result<()>
    where
        T: FloatElement
            + Copy
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + Clone
            + std::fmt::Debug,
        f32: From<T>,
    {
        if !self.requires_grad {
            return Err(TorshError::AutogradError(
                "Called backward on tensor that doesn't require grad".to_string(),
            ));
        }
        let seed = match gradient {
            Some(provided) => {
                let provided_shape = provided.shape();
                let own_shape = self.shape();
                if provided_shape.dims() != own_shape.dims() {
                    return Err(TorshError::ShapeMismatch {
                        expected: own_shape.dims().to_vec(),
                        got: provided_shape.dims().to_vec(),
                    });
                }
                provided.detach()
            }
            None => {
                if self.shape().numel() != 1 {
                    return Err(TorshError::AutogradError(
                        "Gradient can only be computed for scalar outputs".to_string(),
                    ));
                }
                self.ones_like()?
            }
        };
        self.backward_impl(&seed)
    }
}
#[path = "types_advanced.rs"]
mod types_advanced;
pub use types_advanced::{Im2ColConfig, Operation, UnaryKind, ViewKind};
