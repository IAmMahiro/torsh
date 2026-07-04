//! WebGPU backend implementation for ToRSh

use crate::buffer::generate_buffer_id;
use crate::memory::MemoryPoolConfig;
use crate::profiler::SimpleProfiler;
#[cfg(feature = "webgpu")]
use crate::webgpu::wgpu;
use crate::webgpu::{
    WebGpuBackendConfig, WebGpuBuffer, WebGpuDevice, WebGpuError, WebGpuKernelExecutor,
    WebGpuMemoryManager,
};
use crate::{
    BackendCore, BackendResult, Buffer, BufferDescriptor, BufferHandle, BufferUsage, Device,
    Kernel, KernelDescriptor, KernelHandle, MemoryLocation, MemoryManager, MemoryStats, Profiler,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use torsh_core::{device::DeviceType, error::TorshError};

/// WebGPU backend implementation
#[derive(Debug)]
pub struct WebGpuBackend {
    config: WebGpuBackendConfig,
    devices: RwLock<HashMap<usize, Arc<WebGpuDevice>>>,
    memory_managers: RwLock<HashMap<usize, Arc<RwLock<WebGpuMemoryManager>>>>,
    kernel_executors: RwLock<HashMap<usize, Arc<WebGpuKernelExecutor>>>,
    profiler: Arc<SimpleProfiler>,
    initialized: RwLock<bool>,
}

impl WebGpuBackend {
    /// Create a new WebGPU backend
    pub fn new(config: WebGpuBackendConfig) -> Self {
        Self {
            config,
            devices: RwLock::new(HashMap::new()),
            memory_managers: RwLock::new(HashMap::new()),
            kernel_executors: RwLock::new(HashMap::new()),
            profiler: Arc::new(SimpleProfiler::new()),
            initialized: RwLock::new(false),
        }
    }

    /// Create WebGPU backend with default configuration
    pub fn with_default_config() -> Self {
        Self::new(WebGpuBackendConfig::default())
    }

    /// Create a builder for WebGPU backend
    pub fn builder() -> WebGpuBackendBuilder {
        WebGpuBackendBuilder::new()
    }

    /// Get the backend configuration
    pub fn config(&self) -> &WebGpuBackendConfig {
        &self.config
    }

    /// Get a specific device by ID
    pub fn get_device(&self, device_id: usize) -> BackendResult<Arc<WebGpuDevice>> {
        let devices = self.devices.read();
        devices
            .get(&device_id)
            .cloned()
            .ok_or_else(|| TorshError::BackendError(format!("Device {} not found", device_id)))
    }

    /// Get memory manager for a device
    pub fn get_memory_manager(
        &self,
        device_id: usize,
    ) -> BackendResult<Arc<RwLock<WebGpuMemoryManager>>> {
        let managers = self.memory_managers.read();
        managers.get(&device_id).cloned().ok_or_else(|| {
            TorshError::BackendError(format!("Memory manager for device {} not found", device_id))
        })
    }

    /// Get kernel executor for a device
    pub fn get_kernel_executor(
        &self,
        device_id: usize,
    ) -> BackendResult<Arc<WebGpuKernelExecutor>> {
        let executors = self.kernel_executors.read();
        executors.get(&device_id).cloned().ok_or_else(|| {
            TorshError::BackendError(format!(
                "Kernel executor for device {} not found",
                device_id
            ))
        })
    }

    /// Initialize a specific device
    async fn initialize_device(&self, device_id: usize) -> BackendResult<Arc<WebGpuDevice>> {
        let device = if let Some(adapter_index) = self.config.adapter_index {
            WebGpuDevice::from_adapter_index(adapter_index, device_id).await
        } else {
            WebGpuDevice::from_best_adapter(device_id).await
        }
        .map_err(|e| TorshError::BackendError(e.to_string()))?;

        let device = Arc::new(device);

        // Create memory manager
        let memory_config = MemoryPoolConfig::default();
        let memory_manager = Arc::new(RwLock::new(WebGpuMemoryManager::new(
            Arc::clone(&device),
            memory_config,
        )));

        // Create kernel executor
        let kernel_executor = Arc::new(WebGpuKernelExecutor::new(Arc::clone(&device)));

        // Store in maps
        {
            let mut devices = self.devices.write();
            devices.insert(device_id, Arc::clone(&device));
        }
        {
            let mut managers = self.memory_managers.write();
            managers.insert(device_id, memory_manager);
        }
        {
            let mut executors = self.kernel_executors.write();
            executors.insert(device_id, kernel_executor);
        }

        Ok(device)
    }

    /// Convert WebGPU error to TorshError
    fn convert_error(error: WebGpuError) -> TorshError {
        TorshError::BackendError(error.to_string())
    }

    /// Resolve the live `WebGpuBuffer` backing a generic `Buffer` handle.
    ///
    /// `Buffer` only carries a lightweight, backend-agnostic `BufferHandle`;
    /// the actual GPU-side `WebGpuBuffer` (and its underlying `wgpu::Buffer`)
    /// is owned and tracked by the `WebGpuMemoryManager` that allocated it,
    /// keyed by that same handle (see `WebGpuMemoryManager::find_active_buffer`).
    /// This looks the real buffer back up so operations like `copy_buffer`,
    /// `copy_to_device` and `copy_from_device` can reach the actual GPU
    /// resource instead of having nothing to operate on.
    ///
    /// Note: this deliberately replaces an earlier approach that cast the
    /// handle's opaque id directly to a `*const wgpu::Buffer` pointer. That
    /// id is a small sequential counter (see `WebGpuBufferPool`/
    /// `WebGpuMemoryManager`'s `next_handle`), not a real pointer, so
    /// dereferencing it was undefined behavior. Looking the handle up in the
    /// memory manager's active-buffer registry is the sound equivalent.
    fn resolve_webgpu_buffer(&self, buffer: &Buffer) -> BackendResult<Arc<WebGpuBuffer>> {
        let device_id = buffer.device().id();
        let memory_manager = self.get_memory_manager(device_id)?;
        let found = memory_manager.read().find_active_buffer(buffer.handle());
        found.ok_or_else(|| {
            TorshError::BackendError(format!(
                "WebGPU buffer not found for handle {:?} on device {}",
                buffer.handle(),
                device_id
            ))
        })
    }
}

impl BackendCore for WebGpuBackend {
    fn device_type(&self) -> DeviceType {
        DeviceType::Wgpu(0)
    }

    fn name(&self) -> &str {
        "WebGPU"
    }

    fn is_available(&self) -> BackendResult<bool> {
        Ok(crate::webgpu::is_available())
    }

    fn capabilities(&self) -> crate::backend::BackendCapabilities {
        crate::backend::BackendCapabilities {
            max_buffer_size: 2_147_483_648, // 2GB for WebGPU
            max_compute_units: 8,
            max_workgroup_size: (256, 256, 64),
            supported_dtypes: vec![
                torsh_core::dtype::DType::F32,
                torsh_core::dtype::DType::I32,
                torsh_core::dtype::DType::U32,
            ],
            supports_async: true,
            supports_unified_memory: false,
            supports_sub_buffers: true,
            supports_kernel_caching: true,
            memory_bandwidth_gbps: 100.0,    // Default WebGPU bandwidth
            compute_throughput_gflops: 50.0, // Default WebGPU compute throughput
            extended_capabilities: crate::backend::ExtendedCapabilities::default(),
        }
    }

    fn performance_hints(&self) -> crate::backend::PerformanceHints {
        crate::backend::PerformanceHints {
            preferred_workgroup_size: (64, 1, 1),
            memory_alignment: 256, // WebGPU requires 256-byte alignment for buffer offsets
            prefer_vectorized: true,
            prefer_async: true,
            optimal_batch_size: 256,
            cache_kernels: true,
        }
    }
}

#[async_trait::async_trait]
impl crate::backend::BackendLifecycle for WebGpuBackend {
    async fn initialize(&mut self) -> BackendResult<()> {
        if *self.initialized.read() {
            return Ok(());
        }

        // Initialize WebGPU
        crate::webgpu::init().await.map_err(Self::convert_error)?;

        // Initialize at least one device (device 0)
        self.initialize_device(0).await?;

        *self.initialized.write() = true;
        Ok(())
    }

    async fn shutdown(&mut self) -> BackendResult<()> {
        // Clear all devices and managers
        self.devices.write().clear();
        self.memory_managers.write().clear();
        self.kernel_executors.write().clear();

        *self.initialized.write() = false;
        Ok(())
    }

    fn is_initialized(&self) -> bool {
        *self.initialized.read()
    }
}

impl crate::backend::BackendDeviceManager for WebGpuBackend {
    fn devices(&self) -> BackendResult<Vec<Device>> {
        let devices = self.devices.read();
        Ok(devices
            .values()
            .map(|d| {
                let webgpu_device = d.as_ref();
                Device::new(
                    0, // Use 0 as default device index for WebGPU
                    webgpu_device.device_type(),
                    webgpu_device.name().to_string(),
                    webgpu_device.info().clone(),
                )
            })
            .collect())
    }

    fn default_device(&self) -> BackendResult<Device> {
        let webgpu_device = self.get_device(0)?;
        Ok(Device::new(
            0, // Use 0 as default device index for WebGPU
            webgpu_device.device_type(),
            webgpu_device.name().to_string(),
            webgpu_device.info().clone(),
        ))
    }

    fn create_device(&self, device_id: usize) -> BackendResult<Device> {
        // Check if device already exists
        if let Ok(webgpu_device) = self.get_device(device_id) {
            return Ok(Device::new(
                device_id, // Use the provided device_id
                webgpu_device.device_type(),
                webgpu_device.name().to_string(),
                webgpu_device.info().clone(),
            ));
        }

        // This is synchronous but we need async - use a runtime
        let runtime = tokio::runtime::Handle::try_current().or_else(|_| {
            tokio::runtime::Runtime::new()
                .map(|rt| rt.handle().clone())
                .map_err(|e| {
                    TorshError::BackendError(format!("Failed to create async runtime: {}", e))
                })
        })?;

        let webgpu_device = runtime.block_on(async { self.initialize_device(device_id).await })?;

        Ok(Device::new(
            device_id, // Use the provided device_id
            webgpu_device.device_type(),
            webgpu_device.name().to_string(),
            webgpu_device.info().clone(),
        ))
    }

    fn device_count(&self) -> BackendResult<usize> {
        Ok(self.devices.read().len())
    }

    fn is_device_available(&self, device_id: usize) -> bool {
        self.devices.read().contains_key(&device_id)
    }
}

impl crate::backend::BackendResourceManager for WebGpuBackend {
    fn create_buffer(
        &self,
        device: &Device,
        descriptor: &BufferDescriptor,
    ) -> BackendResult<Buffer> {
        let memory_manager = self.get_memory_manager(device.id())?;
        let buffer = memory_manager.write().allocate(descriptor)?;

        // This is a bit of a hack - we return the buffer directly
        // In a real implementation, you'd want a more sophisticated approach
        Ok(buffer)
    }

    fn create_kernel(
        &self,
        device: &Device,
        descriptor: &KernelDescriptor,
    ) -> BackendResult<Kernel> {
        let kernel_executor = self.get_kernel_executor(device.id())?;
        let _webgpu_kernel = kernel_executor
            .create_kernel(descriptor.clone())
            .map_err(Self::convert_error)?;

        // Create a proper Kernel instance
        let kernel_handle = KernelHandle::WebGpu {
            shader_module_id: format!("webgpu_shader_{}", descriptor.name),
            entry_point: "main".to_string(), // Default WebGPU entry point
        };
        let kernel_metadata = crate::kernel::KernelMetadata {
            compile_time_ms: 0.0,
            binary_size: 0,
            registers_per_thread: None,
            shared_memory_usage: None,
            max_workgroup_size: descriptor.workgroup_size_hint,
            compiler_version: "wgpu".to_string(),
            warnings: Vec::new(),
            performance_hints: Vec::new(),
        };

        Ok(Kernel::new(
            0, // kernel id
            device.clone(),
            descriptor.name.clone(),
            descriptor.clone(),
            kernel_handle,
            kernel_metadata,
        ))
    }

    fn memory_manager(
        &self,
        device: &Device,
    ) -> BackendResult<Box<dyn MemoryManager + Send + Sync>> {
        let manager = self.get_memory_manager(device.id())?;

        // Create a wrapper that implements the MemoryManager trait
        Ok(Box::new(WebGpuMemoryManagerWrapper { inner: manager })
            as Box<dyn MemoryManager + Send + Sync>)
    }

    fn profiler(&self) -> BackendResult<Box<dyn Profiler + Send + Sync>> {
        Ok(Box::new((*self.profiler).clone()) as Box<dyn Profiler + Send + Sync>)
    }

    fn create_scoped_buffer(
        &self,
        device: &Device,
        descriptor: &BufferDescriptor,
    ) -> BackendResult<Buffer> {
        // For WebGPU, scoped buffers are the same as regular buffers
        self.create_buffer(device, descriptor)
    }
}

#[async_trait::async_trait]
impl crate::backend::BackendExecutor for WebGpuBackend {
    async fn synchronize(&self, device: &Device) -> BackendResult<()> {
        let webgpu_device = self.get_device(device.id())?;
        webgpu_device
            .wait_for_completion()
            .await
            .map_err(Self::convert_error)
    }

    async fn copy_buffer(
        &self,
        src: &Buffer,
        dst: &Buffer,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> BackendResult<()> {
        if src_offset + size > src.size() {
            return Err(TorshError::InvalidArgument(
                "copy_buffer: source range exceeds buffer size".to_string(),
            ));
        }
        if dst_offset + size > dst.size() {
            return Err(TorshError::InvalidArgument(
                "copy_buffer: destination range exceeds buffer size".to_string(),
            ));
        }

        let device_id = src.device().id();
        let webgpu_device = self.get_device(device_id)?;

        let src_buf = self.resolve_webgpu_buffer(src)?;
        let dst_buf = self.resolve_webgpu_buffer(dst)?;

        // Real GPU-to-GPU copy via a command encoder, mirroring the
        // encode -> submit -> wait-for-completion pattern used elsewhere in
        // this backend (see `WebGpuDevice::benchmark_memory_bandwidth`).
        let mut encoder = webgpu_device.create_command_encoder(Some("Buffer Copy"));
        dst_buf
            .copy_from_buffer(
                &mut encoder,
                &src_buf,
                src_offset as u64,
                dst_offset as u64,
                size as u64,
            )
            .map_err(Self::convert_error)?;

        webgpu_device.submit([encoder.finish()]);
        webgpu_device
            .wait_for_completion()
            .await
            .map_err(Self::convert_error)
    }

    async fn copy_to_device(
        &self,
        src: &[u8],
        dst: &Buffer,
        dst_offset: usize,
    ) -> BackendResult<()> {
        if dst_offset + src.len() > dst.size() {
            return Err(TorshError::InvalidArgument(
                "copy_to_device: write range exceeds destination buffer size".to_string(),
            ));
        }

        let device_id = dst.device().id();
        let webgpu_device = self.get_device(device_id)?;
        let dst_buf = self.resolve_webgpu_buffer(dst)?;

        // WebGpuBuffer::write_data picks the right strategy itself (direct
        // mapping when MAP_WRITE is supported, otherwise a queued write).
        dst_buf
            .write_data(dst_offset as u64, src)
            .await
            .map_err(Self::convert_error)?;

        webgpu_device
            .wait_for_completion()
            .await
            .map_err(Self::convert_error)
    }

    async fn copy_from_device(
        &self,
        src: &Buffer,
        dst: &mut [u8],
        src_offset: usize,
    ) -> BackendResult<()> {
        if src_offset + dst.len() > src.size() {
            return Err(TorshError::InvalidArgument(
                "copy_from_device: read range exceeds source buffer size".to_string(),
            ));
        }

        let device_id = src.device().id();
        let webgpu_device = self.get_device(device_id)?;
        let src_buf = self.resolve_webgpu_buffer(src)?;

        let size = dst.len() as u64;

        // WebGPU storage buffers generally cannot be mapped directly (a
        // buffer's usage may only combine MAP_READ with COPY_DST), so
        // read-back goes through a host-visible staging buffer: copy the
        // device data into it on the GPU timeline, then map and read it.
        let staging_descriptor =
            BufferDescriptor::new(dst.len(), BufferUsage::MAP_READ | BufferUsage::COPY_DST)
                .with_location(MemoryLocation::Host);
        let staging_handle = BufferHandle::WebGpu {
            buffer_ptr: generate_buffer_id() as u64,
            size: dst.len(),
        };
        let staging_buffer = WebGpuBuffer::new(
            Arc::clone(&webgpu_device),
            staging_descriptor,
            staging_handle,
        )
        .map_err(Self::convert_error)?;

        let mut encoder = webgpu_device.create_command_encoder(Some("Buffer Readback"));
        staging_buffer
            .copy_from_buffer(&mut encoder, &src_buf, src_offset as u64, 0, size)
            .map_err(Self::convert_error)?;
        webgpu_device.submit([encoder.finish()]);
        webgpu_device
            .wait_for_completion()
            .await
            .map_err(Self::convert_error)?;

        let data: Vec<u8> = staging_buffer
            .read_data(0, dst.len())
            .await
            .map_err(Self::convert_error)?;
        dst.copy_from_slice(&data);

        Ok(())
    }

    async fn execute_kernel(
        &self,
        kernel: &Kernel,
        _buffers: &[&Buffer],
        uniform_data: &[u8],
        workgroup_size: (u32, u32, u32),
        workgroup_count: (u32, u32, u32),
    ) -> BackendResult<()> {
        // Extract device ID
        let device_id = 0; // Default device for now
        let kernel_executor = self.get_kernel_executor(device_id)?;

        // Execute kernel using the stored WebGPU kernel from handle
        match &kernel.handle {
            KernelHandle::WebGpu {
                shader_module_id: _,
                entry_point: _,
            } => {
                // For now, execute a simple kernel based on kernel name
                // In a full implementation, this would use a kernel cache
                kernel_executor
                    .execute_simple_kernel(
                        &kernel.name,
                        &[], // Simplified - would pass actual wgpu buffers
                        uniform_data,
                        workgroup_size,
                        workgroup_count,
                    )
                    .await
                    .map_err(Self::convert_error)
            }
            _ => Err(TorshError::BackendError(
                "Invalid kernel handle for WebGPU backend".to_string(),
            )),
        }
    }
}

impl crate::backend::BackendOperations for WebGpuBackend {
    fn fft_ops(&self) -> Box<dyn crate::fft::FftOps> {
        Box::new(crate::cpu::fft::CpuFftOps::new(None))
    }

    fn convolution_ops(&self) -> Box<dyn crate::convolution::ConvolutionOps> {
        Box::new(crate::cpu::convolution::CpuConvolutionOps::new(None))
    }

    fn rnn_ops(&self) -> Box<dyn crate::rnn::RnnOps> {
        Box::new(crate::cpu::rnn::CpuRnnOps::new(None))
    }

    fn sparse_ops(&self) -> Box<dyn crate::sparse_ops::SparseOps<f32>> {
        Box::new(crate::sparse_ops::DefaultSparseOps::new(
            crate::Device::new(
                0,
                torsh_core::device::DeviceType::Wgpu(0),
                "WebGPU Device".to_string(),
                crate::DeviceInfo::default(),
            ),
        ))
    }

    fn quantization_ops(&self) -> Box<dyn crate::quantization::QuantizationOps> {
        Box::new(crate::quantization::CpuQuantizationOps::new())
    }

    fn operations_bundle(&self) -> crate::backend::OperationsBundle {
        crate::backend::OperationsBundle {
            fft: self.fft_ops(),
            convolution: self.convolution_ops(),
            rnn: self.rnn_ops(),
            quantization: self.quantization_ops(),
            sparse: self.sparse_ops(),
        }
    }
}

impl crate::backend::BackendOps for WebGpuBackend {
    fn backend_type(&self) -> crate::backend::BackendType {
        crate::backend::BackendType::WebGpu
    }

    fn available_ops(&self) -> Vec<&str> {
        vec![
            "elementwise_add",
            "elementwise_mul",
            "elementwise_sub",
            "elementwise_div",
            "matmul",
            "conv2d",
            "relu",
            "softmax",
            "batch_norm",
            "reduction",
        ]
    }

    fn supports_op(&self, op_name: &str) -> bool {
        self.available_ops().contains(&op_name)
    }

    fn supports_fft(&self) -> bool {
        true
    }

    fn supports_convolution(&self) -> bool {
        true
    }

    fn supports_rnn(&self) -> bool {
        true
    }

    fn supports_sparse(&self) -> bool {
        false
    }

    fn supports_quantization(&self) -> bool {
        true
    }

    fn operation_capabilities(
        &self,
        _op_name: &str,
    ) -> Option<std::collections::HashMap<String, crate::backend::CapabilityValue>> {
        None
    }
}

impl crate::backend::Backend for WebGpuBackend {
    fn as_core(&self) -> &dyn crate::backend::BackendCore {
        self
    }

    fn as_lifecycle(&mut self) -> &mut dyn crate::backend::BackendLifecycle {
        self
    }

    fn as_device_manager(&self) -> &dyn crate::backend::BackendDeviceManager {
        self
    }

    fn as_resource_manager(&self) -> &dyn crate::backend::BackendResourceManager {
        self
    }

    fn as_executor(&self) -> &dyn crate::backend::BackendExecutor {
        self
    }

    fn as_operations(&self) -> &dyn crate::backend::BackendOperations {
        self
    }
}

/// WebGPU backend builder for convenient configuration
#[derive(Debug)]
pub struct WebGpuBackendBuilder {
    config: WebGpuBackendConfig,
}

impl WebGpuBackendBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: WebGpuBackendConfig::default(),
        }
    }

    /// Set adapter index
    pub fn adapter_index(mut self, index: usize) -> Self {
        self.config.adapter_index = Some(index);
        self
    }

    /// Set device ID (alias for adapter_index for API consistency)
    pub fn device_id(mut self, id: usize) -> Self {
        self.config.adapter_index = Some(id);
        self
    }

    /// Set power preference
    pub fn power_preference(mut self, preference: wgpu::PowerPreference) -> Self {
        self.config.power_preference = preference;
        self
    }

    /// Enable debug mode
    pub fn debug_mode(mut self, enable: bool) -> Self {
        self.config.debug_mode = enable;
        self
    }

    /// Set maximum buffer size
    pub fn max_buffer_size(mut self, size: u64) -> Self {
        self.config.max_buffer_size = size;
        self
    }

    /// Enable pipeline cache
    pub fn enable_pipeline_cache(mut self, enable: bool) -> Self {
        self.config.enable_pipeline_cache = enable;
        self
    }

    /// Set preferred workgroup size
    pub fn preferred_workgroup_size(mut self, size: (u32, u32, u32)) -> Self {
        self.config.preferred_workgroup_size = size;
        self
    }

    /// Build the backend
    pub fn build(self) -> WebGpuBackend {
        WebGpuBackend::new(self.config)
    }
}

/// Memory manager wrapper to implement the trait
#[derive(Debug)]
pub struct WebGpuMemoryManagerWrapper {
    inner: Arc<RwLock<WebGpuMemoryManager>>,
}

impl MemoryManager for WebGpuMemoryManagerWrapper {
    fn allocate(
        &mut self,
        descriptor: &BufferDescriptor,
    ) -> torsh_core::error::Result<crate::Buffer> {
        let webgpu_buffer = self
            .inner
            .read()
            .buffer_pool()
            .get_buffer(descriptor.clone())
            .map_err(|e| TorshError::BackendError(e.to_string()))?;

        let handle = webgpu_buffer.handle().clone();
        let buffer = crate::Buffer::new(
            generate_buffer_id(),
            crate::Device::new(
                0,
                torsh_core::device::DeviceType::Wgpu(0),
                "WebGPU Device".to_string(),
                crate::DeviceInfo::default(),
            ),
            webgpu_buffer.descriptor().size as usize,
            descriptor.usage.clone(),
            descriptor.clone(),
            handle,
        );

        Ok(buffer)
    }

    fn deallocate(&mut self, _buffer: &crate::Buffer) -> torsh_core::error::Result<()> {
        Ok(())
    }

    fn stats(&self) -> MemoryStats {
        self.inner.read().stats()
    }

    fn garbage_collect(&mut self) -> torsh_core::error::Result<usize> {
        Ok(0)
    }

    fn set_pool(
        &mut self,
        _pool: Box<dyn crate::memory::MemoryPool>,
    ) -> torsh_core::error::Result<()> {
        Err(TorshError::BackendError(
            "WebGPU memory pool cannot be replaced".to_string(),
        ))
    }

    fn device(&self) -> &crate::Device {
        static WEBGPU_DEVICE: std::sync::OnceLock<crate::Device> = std::sync::OnceLock::new();
        WEBGPU_DEVICE.get_or_init(|| {
            crate::Device::new(
                0,
                torsh_core::device::DeviceType::Wgpu(0),
                "WebGPU Device".to_string(),
                crate::DeviceInfo::default(),
            )
        })
    }

    fn allocate_raw(
        &mut self,
        _size: usize,
        _alignment: usize,
    ) -> torsh_core::error::Result<*mut u8> {
        Err(TorshError::BackendError(
            "WebGPU doesn't support raw memory allocation".to_string(),
        ))
    }

    fn deallocate_raw(&mut self, _ptr: *mut u8, _size: usize) -> torsh_core::error::Result<()> {
        Err(TorshError::BackendError(
            "WebGPU doesn't support raw memory deallocation".to_string(),
        ))
    }

    fn supports_unified_memory(&self) -> bool {
        false
    }

    fn allocate_unified(&mut self, _size: usize) -> torsh_core::error::Result<*mut u8> {
        Err(TorshError::BackendError(
            "WebGPU doesn't support unified memory allocation".to_string(),
        ))
    }

    fn deallocate_unified(&mut self, _ptr: *mut u8, _size: usize) -> torsh_core::error::Result<()> {
        Err(TorshError::BackendError(
            "WebGPU doesn't support unified memory deallocation".to_string(),
        ))
    }

    fn prefetch_to_device(&self, _ptr: *mut u8, _size: usize) -> torsh_core::error::Result<()> {
        Ok(())
    }

    fn prefetch_to_host(&self, _ptr: *mut u8, _size: usize) -> torsh_core::error::Result<()> {
        Ok(())
    }

    fn set_memory_advice(
        &self,
        _ptr: *mut u8,
        _size: usize,
        _advice: crate::memory::MemoryAdvice,
    ) -> torsh_core::error::Result<()> {
        Ok(())
    }

    fn available_memory(&self) -> torsh_core::error::Result<usize> {
        Ok(1024 * 1024 * 1024)
    }

    fn total_memory(&self) -> torsh_core::error::Result<usize> {
        Ok(4 * 1024 * 1024 * 1024)
    }

    fn synchronize(&self) -> torsh_core::error::Result<()> {
        Ok(())
    }

    fn defragment(&mut self) -> torsh_core::error::Result<crate::memory::DefragmentationResult> {
        Ok(crate::memory::DefragmentationResult {
            blocks_moved: 0,
            memory_compacted: 0,
            duration_ms: 0.0,
            fragmentation_before: 0.0,
            fragmentation_after: 0.0,
            efficiency_improvement: 0.0,
            success: true,
        })
    }

    fn needs_defragmentation(&self) -> bool {
        false
    }

    fn fragmentation_info(&self) -> crate::memory::FragmentationInfo {
        crate::memory::FragmentationInfo {
            overall_fragmentation: 0.0,
            external_fragmentation: 0.0,
            internal_fragmentation: 0.0,
            free_blocks: 1,
            allocated_blocks: 0,
            largest_free_block: 1024 * 1024 * 1024,
            smallest_free_block: 1024 * 1024 * 1024,
            average_free_block: 1024 * 1024 * 1024,
            total_free_memory: 1024 * 1024 * 1024,
            total_allocated_memory: 0,
            utilization_efficiency: 1.0,
            allocation_efficiency: 1.0,
        }
    }

    fn compact_memory(&mut self) -> torsh_core::error::Result<crate::memory::CompactionResult> {
        Ok(crate::memory::CompactionResult {
            allocations_moved: 0,
            bytes_moved: 0,
            duration_ms: 0.0,
            largest_free_before: 1024 * 1024 * 1024,
            largest_free_after: 1024 * 1024 * 1024,
            free_blocks_before: 1,
            free_blocks_after: 1,
            success: true,
        })
    }

    fn set_defragmentation_policy(&mut self, _policy: crate::memory::DefragmentationPolicy) {
        // WebGPU doesn't support custom defragmentation policies
    }
}

/// Stub implementation of WebGPU RNN operations
pub struct WebGpuRnnOps;

impl WebGpuRnnOps {
    pub fn new() -> Self {
        Self
    }
}

// TODO: Implement RnnOps trait when it becomes available
/*
#[async_trait::async_trait]
impl crate::rnn::RnnOps for WebGpuRnnOps {
    async fn lstm_forward(
        &self,
        _device: &crate::Device,
        _input: &crate::Buffer,
        _hidden: &crate::Buffer,
        _cell: &crate::Buffer,
        _weights: &[&crate::Buffer],
        _biases: &[&crate::Buffer],
        _output: &crate::Buffer,
        _new_hidden: &crate::Buffer,
        _new_cell: &crate::Buffer,
        _batch_size: usize,
        _input_size: usize,
        _hidden_size: usize,
        _num_layers: usize,
        _dropout: f32,
        _bidirectional: bool,
    ) -> crate::BackendResult<()> {
        Err(torsh_core::error::TorshError::BackendError(
            "WebGPU RNN operations not yet implemented".to_string(),
        ))
    }

    async fn gru_forward(
        &self,
        _device: &crate::Device,
        _input: &crate::Buffer,
        _hidden: &crate::Buffer,
        _weights: &[&crate::Buffer],
        _biases: &[&crate::Buffer],
        _output: &crate::Buffer,
        _new_hidden: &crate::Buffer,
        _batch_size: usize,
        _input_size: usize,
        _hidden_size: usize,
        _num_layers: usize,
        _dropout: f32,
        _bidirectional: bool,
    ) -> crate::BackendResult<()> {
        Err(torsh_core::error::TorshError::BackendError(
            "WebGPU RNN operations not yet implemented".to_string(),
        ))
    }

    async fn rnn_forward(
        &self,
        _device: &crate::Device,
        _input: &crate::Buffer,
        _hidden: &crate::Buffer,
        _weights: &[&crate::Buffer],
        _biases: &[&crate::Buffer],
        _output: &crate::Buffer,
        _new_hidden: &crate::Buffer,
        _batch_size: usize,
        _input_size: usize,
        _hidden_size: usize,
        _num_layers: usize,
        _activation: crate::rnn::RnnActivation,
        _dropout: f32,
        _bidirectional: bool,
    ) -> crate::BackendResult<()> {
        Err(torsh_core::error::TorshError::BackendError(
            "WebGPU RNN operations not yet implemented".to_string(),
        ))
    }

    fn supports_lstm(&self) -> bool {
        false
    }

    fn supports_gru(&self) -> bool {
        false
    }

    fn supports_bidirectional(&self) -> bool {
        false
    }

    fn supports_dropout(&self) -> bool {
        false
    }

    fn optimal_workgroup_size(&self) -> (u32, u32, u32) {
        (64, 1, 1)
    }
}
*/

/// Stub implementation of WebGPU quantization operations
pub struct WebGpuQuantizationOps;

impl WebGpuQuantizationOps {
    pub fn new() -> Self {
        Self
    }
}

// TODO: Implement QuantizationOps trait with correct method signatures
/*
#[async_trait::async_trait]
impl crate::quantization::QuantizationOps for WebGpuQuantizationOps {
    async fn quantize_int8(
        &self,
        _device: &crate::Device,
        _input: &crate::Buffer,
        _output: &crate::Buffer,
        _scale: f32,
        _zero_point: i8,
    ) -> crate::BackendResult<()> {
        Err(torsh_core::error::TorshError::BackendError(
            "WebGPU quantization operations not yet implemented".to_string(),
        ))
    }

    async fn dequantize_int8(
        &self,
        _device: &crate::Device,
        _input: &crate::Buffer,
        _output: &crate::Buffer,
        _scale: f32,
        _zero_point: i8,
    ) -> crate::BackendResult<()> {
        Err(torsh_core::error::TorshError::BackendError(
            "WebGPU quantization operations not yet implemented".to_string(),
        ))
    }

    async fn quantize_int4(
        &self,
        _device: &crate::Device,
        _input: &crate::Buffer,
        _output: &crate::Buffer,
        _scale: f32,
        _zero_point: i8,
    ) -> crate::BackendResult<()> {
        Err(torsh_core::error::TorshError::BackendError(
            "WebGPU quantization operations not yet implemented".to_string(),
        ))
    }

    async fn dequantize_int4(
        &self,
        _device: &crate::Device,
        _input: &crate::Buffer,
        _output: &crate::Buffer,
        _scale: f32,
        _zero_point: i8,
    ) -> crate::BackendResult<()> {
        Err(torsh_core::error::TorshError::BackendError(
            "WebGPU quantization operations not yet implemented".to_string(),
        ))
    }

    fn supports_int8(&self) -> bool {
        false
    }

    fn supports_int4(&self) -> bool {
        false
    }

    fn optimal_workgroup_size(&self) -> (u32, u32, u32) {
        (64, 1, 1)
    }
}
*/

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{
        BackendCore, BackendDeviceManager, BackendExecutor, BackendLifecycle, BackendOps,
        BackendResourceManager,
    };
    use crate::BackendType;
    use torsh_core::DType;

    #[test]
    fn test_backend_creation() {
        let backend = WebGpuBackend::with_default_config();
        assert_eq!(backend.name(), "WebGPU");
        assert_eq!(backend.device_type(), DeviceType::Wgpu(0));
    }

    #[test]
    fn test_backend_builder() {
        let backend = WebGpuBackendBuilder::new()
            .adapter_index(0)
            .power_preference(wgpu::PowerPreference::HighPerformance)
            .debug_mode(true)
            .max_buffer_size(2 * 1024 * 1024 * 1024) // 2GB
            .enable_pipeline_cache(true)
            .preferred_workgroup_size((128, 1, 1))
            .build();

        assert_eq!(backend.config().adapter_index, Some(0));
        assert_eq!(
            backend.config().power_preference,
            wgpu::PowerPreference::HighPerformance
        );
        assert!(backend.config().debug_mode);
        assert_eq!(backend.config().max_buffer_size, 2 * 1024 * 1024 * 1024);
        assert!(backend.config().enable_pipeline_cache);
        assert_eq!(backend.config().preferred_workgroup_size, (128, 1, 1));
    }

    #[tokio::test]
    async fn test_backend_availability() {
        let backend = WebGpuBackend::with_default_config();

        match backend.is_available() {
            Ok(available) => {
                if available {
                    println!("WebGPU backend is available");
                } else {
                    println!("WebGPU backend is not available");
                }
            }
            Err(e) => {
                println!("Error checking WebGPU availability: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_backend_initialization() {
        if cfg!(feature = "webgpu") && crate::webgpu::is_available() {
            let mut backend = WebGpuBackend::with_default_config();

            let result = backend.initialize().await;
            if result.is_ok() {
                assert!(*backend.initialized.read());

                // Test device creation
                let device_result = backend.default_device();
                if device_result.is_ok() {
                    let device = device_result.expect("operation should succeed");
                    assert_eq!(device.device_type(), DeviceType::Wgpu(0));
                }

                // Test shutdown
                let shutdown_result = backend.shutdown().await;
                assert!(shutdown_result.is_ok());
                assert!(!*backend.initialized.read());
            }
        }
    }

    #[test]
    fn test_backend_ops() {
        let backend = WebGpuBackend::with_default_config();

        assert_eq!(backend.backend_type(), BackendType::WebGpu);
        assert!(backend.supports_op("elementwise_add"));
        assert!(backend.supports_op("matmul"));
        assert!(backend.supports_op("conv2d"));
        assert!(!backend.supports_op("nonexistent_op"));

        let ops = backend.available_ops();
        assert!(!ops.is_empty());
        assert!(ops.contains(&"elementwise_add"));
    }

    #[test]
    fn test_capabilities() {
        let backend = WebGpuBackend::with_default_config();
        let capabilities = backend.capabilities();

        // Default capabilities when no device is available
        assert!(capabilities.supported_dtypes.contains(&DType::F32));
        assert!(capabilities.supports_async);
        assert!(capabilities.supports_kernel_caching);
    }

    #[test]
    fn test_performance_hints() {
        let backend = WebGpuBackend::with_default_config();
        let hints = backend.performance_hints();

        assert_eq!(hints.preferred_workgroup_size, (64, 1, 1));
        assert_eq!(hints.memory_alignment, 256);
        assert!(hints.prefer_vectorized);
        assert!(hints.prefer_async);
        assert!(hints.cache_kernels);
    }

    /// Regression test for a real correctness bug: `copy_buffer`,
    /// `copy_to_device` and `copy_from_device` used to unconditionally
    /// `return Ok(())` without moving any data, so every cross-backend
    /// transfer through the WebGPU backend silently no-op'd while reporting
    /// success. This round-trips a distinguishable, non-zero payload through
    /// `copy_to_device` -> `copy_from_device` and checks it survives
    /// byte-for-byte.
    ///
    /// Against the old (buggy) implementation, `copy_from_device` never
    /// touches `readback`, so it stays all-zero and the final `assert_eq!`
    /// fails. Against the fix, the payload is actually written to the GPU
    /// buffer and actually read back, so the assertion passes.
    ///
    /// Gracefully skipped (not failed) when no real WebGPU adapter is
    /// available, e.g. in headless CI.
    #[tokio::test]
    async fn test_copy_to_device_and_back_round_trips_real_data() {
        if !(cfg!(feature = "webgpu") && crate::webgpu::is_available()) {
            eprintln!("Skipping: no WebGPU adapter available in this environment");
            return;
        }

        let mut backend = WebGpuBackend::with_default_config();
        if backend.initialize().await.is_err() {
            eprintln!("Skipping: failed to initialize WebGPU backend");
            return;
        }

        let device = match backend.default_device() {
            Ok(device) => device,
            Err(_) => {
                eprintln!("Skipping: no default WebGPU device");
                return;
            }
        };

        const LEN: usize = 4096;
        let descriptor = BufferDescriptor::new(
            LEN,
            BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        );

        let buffer = match backend.create_buffer(&device, &descriptor) {
            Ok(buffer) => buffer,
            Err(e) => {
                eprintln!("Skipping: failed to create WebGPU buffer: {e}");
                return;
            }
        };

        // Distinguishable, non-zero, non-constant payload so a silent no-op
        // (old buggy behavior) is unambiguously distinguishable from a real
        // transfer: a zero-filled or unmodified `readback` cannot match it.
        let payload: Vec<u8> = (0..LEN as u32)
            .map(|i| ((i * 37 + 11) % 256) as u8)
            .collect();
        assert!(
            payload.iter().any(|&b| b != 0),
            "test payload must contain non-zero bytes"
        );

        backend
            .copy_to_device(&payload, &buffer, 0)
            .await
            .expect("copy_to_device should succeed");

        let mut readback = vec![0u8; LEN];
        backend
            .copy_from_device(&buffer, &mut readback, 0)
            .await
            .expect("copy_from_device should succeed");

        assert_eq!(
            payload, readback,
            "data must survive a copy_to_device -> copy_from_device round trip byte-for-byte"
        );

        // Also exercise copy_buffer (device-to-device), the third
        // previously-broken function: copy into a second buffer and verify
        // it independently reads back correctly.
        let buffer2 = backend
            .create_buffer(&device, &descriptor)
            .expect("second buffer allocation should succeed");

        backend
            .copy_buffer(&buffer, &buffer2, 0, 0, LEN)
            .await
            .expect("copy_buffer should succeed");

        let mut readback2 = vec![0u8; LEN];
        backend
            .copy_from_device(&buffer2, &mut readback2, 0)
            .await
            .expect("copy_from_device on the copy_buffer target should succeed");

        assert_eq!(
            payload, readback2,
            "data must survive a copy_buffer device-to-device copy byte-for-byte"
        );
    }
}
