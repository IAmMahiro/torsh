//! Distributed backend implementations
//!
//! This module provides a modern, async-first backend abstraction for distributed training
//! with support for multiple communication backends and advanced features.

use crate::{TorshDistributedError, TorshResult};
use async_trait::async_trait;
use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

/// Reduce operation types for collective operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    /// Sum all values across processes
    Sum,
    /// Multiply all values across processes
    Product,
    /// Find minimum value across processes
    Min,
    /// Find maximum value across processes
    Max,
    /// Bitwise AND across processes
    Band,
    /// Bitwise OR across processes
    Bor,
    /// Bitwise XOR across processes
    Bxor,
    /// Average values across processes
    Mean,
}

impl fmt::Display for ReduceOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReduceOp::Sum => write!(f, "sum"),
            ReduceOp::Product => write!(f, "product"),
            ReduceOp::Min => write!(f, "min"),
            ReduceOp::Max => write!(f, "max"),
            ReduceOp::Band => write!(f, "bitwise_and"),
            ReduceOp::Bor => write!(f, "bitwise_or"),
            ReduceOp::Bxor => write!(f, "bitwise_xor"),
            ReduceOp::Mean => write!(f, "mean"),
        }
    }
}

/// Backend types for distributed training
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendType {
    /// NVIDIA Collective Communication Library (GPU)
    Nccl,
    /// Message Passing Interface (CPU/GPU)
    Mpi,
    /// Facebook Gloo (CPU)
    Gloo,
    /// Custom backend implementation
    Custom(&'static str),
}

impl fmt::Display for BackendType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendType::Nccl => write!(f, "nccl"),
            BackendType::Mpi => write!(f, "mpi"),
            BackendType::Gloo => write!(f, "gloo"),
            BackendType::Custom(name) => write!(f, "custom:{}", name),
        }
    }
}

/// Backend capabilities and features
#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    /// Supports asynchronous operations
    pub async_operations: bool,
    /// Supports GPU tensors
    pub gpu_support: bool,
    /// Supports point-to-point communication
    pub p2p_communication: bool,
    /// Supports custom reduce operations
    pub custom_reduce_ops: bool,
    /// Maximum tensor size supported
    pub max_tensor_size: Option<usize>,
    /// Supported data types
    pub supported_dtypes: Vec<String>,
}

impl Default for BackendCapabilities {
    fn default() -> Self {
        Self {
            async_operations: true,
            gpu_support: false,
            p2p_communication: true,
            custom_reduce_ops: false,
            max_tensor_size: None,
            supported_dtypes: vec![
                "f32".to_string(),
                "f64".to_string(),
                "i32".to_string(),
                "i64".to_string(),
            ],
        }
    }
}

/// Backend configuration options
#[derive(Debug, Clone)]
pub struct BackendConfig {
    /// Network timeout for operations
    pub timeout: Duration,
    /// Enable compression for communication
    pub enable_compression: bool,
    /// Custom configuration options
    pub custom_options: HashMap<String, String>,
    /// Maximum retries for failed operations
    pub max_retries: u32,
    /// Backoff multiplier for retries
    pub retry_backoff: f64,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            enable_compression: false,
            custom_options: HashMap::new(),
            max_retries: 3,
            retry_backoff: 2.0,
        }
    }
}

/// Backend status information
#[derive(Debug, Clone)]
pub struct BackendStatus {
    /// Whether the backend is initialized
    pub initialized: bool,
    /// Whether the backend is healthy
    pub healthy: bool,
    /// Number of active operations
    pub active_operations: u32,
    /// Total operations performed
    pub total_operations: u64,
    /// Number of failed operations
    pub failed_operations: u64,
    /// Last error encountered
    pub last_error: Option<String>,
}

impl Default for BackendStatus {
    fn default() -> Self {
        Self {
            initialized: false,
            healthy: true,
            active_operations: 0,
            total_operations: 0,
            failed_operations: 0,
            last_error: None,
        }
    }
}

/// Modern async-first distributed backend trait
#[async_trait]
pub trait Backend: Send + Sync {
    /// Get the backend type
    fn backend_type(&self) -> BackendType;

    /// Get backend capabilities
    fn capabilities(&self) -> BackendCapabilities;

    /// Initialize the backend with configuration
    async fn init(&mut self, config: BackendConfig) -> TorshResult<()>;

    /// Cleanup the backend resources
    async fn cleanup(&mut self) -> TorshResult<()>;

    /// Get current backend status
    fn status(&self) -> BackendStatus;

    /// Check if backend is ready for operations
    fn is_ready(&self) -> bool {
        let status = self.status();
        status.initialized && status.healthy
    }

    /// Get rank of current process
    fn rank(&self) -> u32;

    /// Get world size (total number of processes)
    fn world_size(&self) -> u32;

    /// Barrier synchronization across all processes
    async fn barrier(&mut self) -> TorshResult<()>;

    /// Barrier synchronization with timeout
    async fn barrier_with_timeout(&mut self, timeout: Duration) -> TorshResult<()> {
        tokio::time::timeout(timeout, self.barrier())
            .await
            .map_err(|_| TorshDistributedError::operation_timeout("barrier", timeout.as_secs()))?
    }

    /// All-reduce operation on tensor
    async fn all_reduce(
        &mut self,
        tensor: &mut (dyn Any + Send + Sync),
        op: ReduceOp,
    ) -> TorshResult<()>;

    /// All-gather operation on tensor
    async fn all_gather(
        &mut self,
        tensor: &(dyn Any + Send + Sync),
    ) -> TorshResult<Box<dyn Any + Send>>;

    /// Broadcast operation on tensor
    async fn broadcast(
        &mut self,
        tensor: &mut (dyn Any + Send + Sync),
        root: u32,
    ) -> TorshResult<()>;

    /// Point-to-point send operation
    async fn send(
        &mut self,
        tensor: &(dyn Any + Send + Sync),
        dst: u32,
        tag: u32,
    ) -> TorshResult<()>;

    /// Point-to-point receive operation
    async fn recv(&mut self, src: u32, tag: u32) -> TorshResult<Box<dyn Any + Send>>;

    /// Health check for the backend
    async fn health_check(&mut self) -> TorshResult<bool> {
        // Default implementation: check if barrier works
        match tokio::time::timeout(Duration::from_secs(5), self.barrier()).await {
            Ok(Ok(())) => Ok(true),
            _ => Ok(false),
        }
    }

    /// Get backend-specific metrics
    fn get_metrics(&self) -> HashMap<String, f64> {
        HashMap::new() // Default: no metrics
    }

    /// Downcast to any type for backend-specific operations
    fn as_any(&self) -> &dyn std::any::Any;

    /// Downcast to mutable any type for backend-specific operations
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// Factory trait for creating backend instances
pub trait BackendFactory: Send + Sync {
    /// Create a new backend instance
    fn create_backend(
        &self,
        rank: u32,
        world_size: u32,
        master_addr: &str,
        master_port: u16,
    ) -> TorshResult<Box<dyn Backend>>;

    /// Get the backend type this factory creates
    fn backend_type(&self) -> BackendType;

    /// Check if this backend is available on the current system
    fn is_available(&self) -> bool;

    /// Get default configuration for this backend
    fn default_config(&self) -> BackendConfig {
        BackendConfig::default()
    }
}

/// Mock backend for testing and development
#[derive(Debug)]
pub struct MockBackend {
    rank: u32,
    world_size: u32,
    status: BackendStatus,
    config: Option<BackendConfig>,
    metrics: HashMap<String, f64>,
    /// The backend type this mock is pretending to be
    backend_type: BackendType,
}

impl MockBackend {
    pub fn new(rank: u32, world_size: u32) -> Self {
        Self {
            rank,
            world_size,
            status: BackendStatus::default(),
            config: None,
            metrics: HashMap::new(),
            backend_type: BackendType::Gloo,
        }
    }

    /// Create a mock backend that reports itself as the given backend type
    pub fn with_backend_type(rank: u32, world_size: u32, backend_type: BackendType) -> Self {
        Self {
            rank,
            world_size,
            status: BackendStatus::default(),
            config: None,
            metrics: HashMap::new(),
            backend_type,
        }
    }

    /// Simulate operation latency for realistic testing
    async fn simulate_latency(&self) {
        let latency_ms = 1 + (self.rank() % 5); // 1-5ms based on rank
        tokio::time::sleep(Duration::from_millis(latency_ms as u64)).await;
    }

    /// Update operation metrics
    fn update_metrics(&mut self, operation: &str, success: bool) {
        self.status.total_operations += 1;
        if success {
            let key = format!("{}_success_count", operation);
            *self.metrics.entry(key).or_insert(0.0) += 1.0;
        } else {
            self.status.failed_operations += 1;
            let key = format!("{}_failure_count", operation);
            *self.metrics.entry(key).or_insert(0.0) += 1.0;
        }
    }
}

#[async_trait]
impl Backend for MockBackend {
    fn backend_type(&self) -> BackendType {
        self.backend_type
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            async_operations: true,
            gpu_support: false,
            p2p_communication: true,
            custom_reduce_ops: false,
            max_tensor_size: Some(1_000_000_000), // 1GB
            supported_dtypes: vec![
                "f32".to_string(),
                "f64".to_string(),
                "i32".to_string(),
                "i64".to_string(),
                "u32".to_string(),
                "u64".to_string(),
            ],
        }
    }

    async fn init(&mut self, config: BackendConfig) -> TorshResult<()> {
        if self.status.initialized {
            return Ok(());
        }

        self.config = Some(config);
        self.status.initialized = true;
        self.status.healthy = true;

        // Simulate initialization time
        self.simulate_latency().await;

        self.update_metrics("init", true);
        Ok(())
    }

    async fn cleanup(&mut self) -> TorshResult<()> {
        if !self.status.initialized {
            return Ok(());
        }

        self.status.initialized = false;
        self.status.active_operations = 0;
        self.config = None;

        self.simulate_latency().await;
        self.update_metrics("cleanup", true);
        Ok(())
    }

    fn status(&self) -> BackendStatus {
        self.status.clone()
    }

    fn rank(&self) -> u32 {
        self.rank
    }

    fn world_size(&self) -> u32 {
        self.world_size
    }

    async fn barrier(&mut self) -> TorshResult<()> {
        if !self.status.initialized {
            return Err(TorshDistributedError::BackendNotInitialized);
        }

        self.status.active_operations += 1;

        // Simulate barrier synchronization time
        self.simulate_latency().await;

        self.status.active_operations -= 1;
        self.update_metrics("barrier", true);
        Ok(())
    }

    async fn all_reduce(
        &mut self,
        _tensor: &mut (dyn Any + Send + Sync),
        op: ReduceOp,
    ) -> TorshResult<()> {
        if !self.status.initialized {
            return Err(TorshDistributedError::BackendNotInitialized);
        }

        self.status.active_operations += 1;

        // Simulate all-reduce computation and communication time based on tensor type
        let base_latency = 1; // Base latency for mock operation
        tokio::time::sleep(Duration::from_millis(base_latency)).await;

        // Mock operation: For testing, just simulate processing
        // In a real implementation, this would perform actual reduction
        match op {
            ReduceOp::Sum
            | ReduceOp::Mean
            | ReduceOp::Product
            | ReduceOp::Min
            | ReduceOp::Max
            | ReduceOp::Band
            | ReduceOp::Bor
            | ReduceOp::Bxor => {
                // Simulate reduction operation processing time
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        }

        self.status.active_operations -= 1;
        self.update_metrics("all_reduce", true);
        Ok(())
    }

    async fn all_gather(
        &mut self,
        _tensor: &(dyn Any + Send + Sync),
    ) -> TorshResult<Box<dyn Any + Send>> {
        if !self.status.initialized {
            return Err(TorshDistributedError::BackendNotInitialized);
        }

        self.status.active_operations += 1;

        // Simulate all-gather time
        self.simulate_latency().await;

        // Mock implementation: return empty vector wrapped in Box<dyn Any>
        // In a real implementation, this would gather tensors from all ranks
        let result: Vec<u8> = Vec::new(); // Placeholder result

        self.status.active_operations -= 1;
        self.update_metrics("all_gather", true);
        Ok(Box::new(result))
    }

    async fn broadcast(
        &mut self,
        _tensor: &mut (dyn Any + Send + Sync),
        root: u32,
    ) -> TorshResult<()> {
        if !self.status.initialized {
            return Err(TorshDistributedError::BackendNotInitialized);
        }

        if root >= self.world_size() {
            return Err(TorshDistributedError::RankOutOfBounds {
                rank: root,
                world_size: self.world_size(),
            });
        }

        self.status.active_operations += 1;

        // Simulate broadcast time
        self.simulate_latency().await;

        // Mock implementation: tensor remains unchanged (assumes root sent its data)

        self.status.active_operations -= 1;
        self.update_metrics("broadcast", true);
        Ok(())
    }

    async fn send(
        &mut self,
        _tensor: &(dyn Any + Send + Sync),
        dst: u32,
        _tag: u32,
    ) -> TorshResult<()> {
        if !self.status.initialized {
            return Err(TorshDistributedError::BackendNotInitialized);
        }

        if dst >= self.world_size() {
            return Err(TorshDistributedError::RankOutOfBounds {
                rank: dst,
                world_size: self.world_size(),
            });
        }

        self.status.active_operations += 1;

        // Simulate send time
        self.simulate_latency().await;

        self.status.active_operations -= 1;
        self.update_metrics("send", true);
        Ok(())
    }

    async fn recv(&mut self, src: u32, _tag: u32) -> TorshResult<Box<dyn Any + Send>> {
        if !self.status.initialized {
            return Err(TorshDistributedError::BackendNotInitialized);
        }

        if src >= self.world_size() {
            return Err(TorshDistributedError::RankOutOfBounds {
                rank: src,
                world_size: self.world_size(),
            });
        }

        self.status.active_operations += 1;

        // Simulate receive time
        self.simulate_latency().await;

        // Mock implementation: create a dummy tensor
        // In real implementation, this would receive actual data
        let dummy_data: Vec<u8> = Vec::new(); // Placeholder received data

        self.status.active_operations -= 1;
        self.update_metrics("recv", true);
        Ok(Box::new(dummy_data))
    }

    fn get_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = self.metrics.clone();
        metrics.insert(
            "total_operations".to_string(),
            self.status.total_operations as f64,
        );
        metrics.insert(
            "failed_operations".to_string(),
            self.status.failed_operations as f64,
        );
        metrics.insert(
            "active_operations".to_string(),
            self.status.active_operations as f64,
        );

        if self.status.total_operations > 0 {
            let success_rate = (self.status.total_operations - self.status.failed_operations)
                as f64
                / self.status.total_operations as f64;
            metrics.insert("success_rate".to_string(), success_rate);
        }

        metrics
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Factory for creating MockBackend instances
pub struct MockBackendFactory;

impl BackendFactory for MockBackendFactory {
    fn create_backend(
        &self,
        rank: u32,
        world_size: u32,
        _master_addr: &str,
        _master_port: u16,
    ) -> TorshResult<Box<dyn Backend>> {
        Ok(Box::new(MockBackend::new(rank, world_size)))
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Gloo
    }

    fn is_available(&self) -> bool {
        true // Mock backend is always available
    }

    fn default_config(&self) -> BackendConfig {
        BackendConfig {
            timeout: Duration::from_secs(10),
            enable_compression: false,
            custom_options: HashMap::new(),
            max_retries: 2,
            retry_backoff: 1.5,
        }
    }
}

#[cfg(feature = "mpi")]
mod mpi_backend {
    use super::*;
    use mpi::topology::Communicator;
    use mpi::traits::CommunicatorCollectives;
    use tracing::info;

    pub struct MpiBackend {
        // Must be kept alive for as long as `world` is used: `Universe`'s
        // `Drop` impl calls `MPI_Finalize()`, so if this were dropped at the
        // end of `new()` (e.g. by not storing it here), every subsequent MPI
        // call on `world` would fail with "Attempting to use an MPI routine
        // ... before initializing or after finalizing MPICH". The leading
        // underscore is intentional: this field is held only for its `Drop`
        // side effect and is never otherwise read.
        _universe: mpi::environment::Universe,
        world: mpi::topology::SimpleCommunicator,
        initialized: bool,
    }

    // SAFETY: MPI communicators are not inherently thread-safe, but we ensure:
    // 1. All MPI operations are protected by async/await boundaries
    // 2. No concurrent access to the same communicator from multiple threads
    // 3. MPI_THREAD_SERIALIZED or higher thread support is assumed
    // Users must ensure MPI is initialized with appropriate thread support level
    unsafe impl Send for MpiBackend {}
    unsafe impl Sync for MpiBackend {}

    impl MpiBackend {
        pub fn new() -> TorshResult<Self> {
            let universe = mpi::initialize().ok_or_else(|| {
                TorshDistributedError::backend_error("MPI", "Failed to initialize MPI".to_string())
            })?;

            let world = universe.world();

            Ok(Self {
                _universe: universe,
                world,
                initialized: false,
            })
        }
    }

    #[async_trait]
    impl Backend for MpiBackend {
        fn backend_type(&self) -> BackendType {
            BackendType::Mpi
        }

        async fn init(&mut self, _config: BackendConfig) -> TorshResult<()> {
            self.initialized = true;
            Ok(())
        }

        async fn cleanup(&mut self) -> TorshResult<()> {
            self.initialized = false;
            Ok(())
        }

        fn is_ready(&self) -> bool {
            self.initialized
        }

        fn rank(&self) -> u32 {
            self.world.rank() as u32
        }

        fn world_size(&self) -> u32 {
            self.world.size() as u32
        }

        fn capabilities(&self) -> BackendCapabilities {
            BackendCapabilities {
                async_operations: true,
                gpu_support: false,
                p2p_communication: true,
                custom_reduce_ops: true,
                max_tensor_size: None,
                supported_dtypes: vec![
                    "f32".to_string(),
                    "f64".to_string(),
                    "i32".to_string(),
                    "i64".to_string(),
                ],
            }
        }

        fn status(&self) -> BackendStatus {
            BackendStatus {
                initialized: self.initialized,
                healthy: true,
                active_operations: 0,
                total_operations: 0,
                failed_operations: 0,
                last_error: None,
            }
        }

        async fn barrier(&mut self) -> TorshResult<()> {
            if !self.initialized {
                return Err(TorshDistributedError::backend_error(
                    "MPI",
                    "Backend not initialized",
                ));
            }

            self.world.barrier();
            info!("MPI barrier completed (rank {})", self.rank());
            Ok(())
        }

        async fn all_reduce(
            &mut self,
            _tensor: &mut (dyn Any + Send + Sync),
            _op: ReduceOp,
        ) -> TorshResult<()> {
            if !self.initialized {
                return Err(TorshDistributedError::backend_error(
                    "MPI",
                    "Backend not initialized",
                ));
            }

            // A correct MPI all-reduce must call MPI_Allreduce on the concrete
            // element slice via the linked libmpi. Fabricating a reduction here
            // would silently corrupt gradients (the very bug this crate is being
            // hardened against), so return an honest error until the typed
            // MPI_Allreduce path is wired. `barrier` works because it needs no
            // data buffer; buffered collectives do.
            Err(TorshDistributedError::backend_error(
                "MPI",
                "all_reduce not yet implemented against the linked MPI communicator \
                 (the previous timing-only simulation did not reduce any data)",
            ))
        }

        async fn all_gather(
            &mut self,
            _tensor: &(dyn Any + Send + Sync),
        ) -> TorshResult<Box<dyn Any + Send>> {
            if !self.initialized {
                return Err(TorshDistributedError::backend_error(
                    "MPI",
                    "Backend not initialized",
                ));
            }

            // MPI all-gather requires a real libmpi linked at build time.
            // Enable the `mpi` feature and link against an MPI installation to
            // get actual MPI_Allgather semantics.
            Err(TorshDistributedError::backend_error(
                "MPI",
                "MPI backend not linked; enable 'mpi' feature with real libmpi",
            ))
        }

        async fn broadcast(
            &mut self,
            _tensor: &mut (dyn Any + Send + Sync),
            _root: u32,
        ) -> TorshResult<()> {
            if !self.initialized {
                return Err(TorshDistributedError::backend_error(
                    "MPI",
                    "Backend not initialized",
                ));
            }

            // A correct MPI broadcast must call MPI_Bcast on the concrete element
            // slice via the linked libmpi; the previous implementation only slept
            // for a simulated duration and delivered no data. Return an honest
            // error rather than leave non-root ranks with stale buffers.
            Err(TorshDistributedError::backend_error(
                "MPI",
                "broadcast not yet implemented against the linked MPI communicator \
                 (the previous timing-only simulation delivered no data)",
            ))
        }

        async fn send(
            &mut self,
            _tensor: &(dyn Any + Send + Sync),
            _dst: u32,
            _tag: u32,
        ) -> TorshResult<()> {
            if !self.initialized {
                return Err(TorshDistributedError::backend_error(
                    "MPI",
                    "Backend not initialized",
                ));
            }

            // A correct MPI send must call MPI_Send on the concrete element slice
            // via the linked libmpi; the previous implementation only slept and
            // transferred nothing. Return an honest error.
            Err(TorshDistributedError::backend_error(
                "MPI",
                "send not yet implemented against the linked MPI communicator \
                 (the previous timing-only simulation transferred no data)",
            ))
        }

        async fn recv(&mut self, _src: u32, _tag: u32) -> TorshResult<Box<dyn Any + Send>> {
            if !self.initialized {
                return Err(TorshDistributedError::backend_error(
                    "MPI",
                    "Backend not initialized",
                ));
            }

            // MPI point-to-point recv requires a real libmpi linked at build time.
            // Enable the `mpi` feature and link against an MPI installation to
            // get actual MPI_Recv semantics.
            Err(TorshDistributedError::backend_error(
                "MPI",
                "MPI backend not linked; enable 'mpi' feature with real libmpi",
            ))
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        // NOTE: `mpi::initialize()` can only succeed once per OS process, so
        // this crate must contain exactly one test that constructs an
        // `MpiBackend`. MPICH's singleton init means this works even when
        // run directly (without `mpirun`), creating a size-1 world.
        #[tokio::test]
        async fn test_mpi_barrier_returns_ok() -> TorshResult<()> {
            let mut backend = MpiBackend::new()?;
            backend.init(BackendConfig::default()).await?;
            backend.barrier().await?;
            Ok(())
        }
    }
}

#[cfg(feature = "mpi")]
pub use mpi_backend::MpiBackend;

#[cfg(feature = "nccl")]
mod nccl_backend {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use tracing::info;

    /// NCCL backend for GPU distributed training
    ///
    /// This implementation provides the interface for NCCL-based distributed training.
    /// Currently uses mock implementations with TODOs for actual NCCL integration.
    /// Real NCCL integration would require:
    /// 1. Proper NCCL Rust bindings (currently not available on crates.io)
    /// 2. CUDA runtime integration
    /// 3. Process coordination for communicator initialization
    pub struct NcclBackend {
        rank: u32,
        world_size: u32,
        initialized: AtomicBool,
        device_id: i32,
        // TODO: Add actual NCCL communicator when bindings are available
        // comm: Option<NcclCommunicator>,
    }

    impl NcclBackend {
        pub fn new(rank: u32, world_size: u32, device_id: Option<i32>) -> TorshResult<Self> {
            let device_id = device_id.unwrap_or(rank as i32);

            // TODO: Validate CUDA device exists and is accessible

            Ok(Self {
                rank,
                world_size,
                initialized: AtomicBool::new(false),
                device_id,
            })
        }

        /// Initialize NCCL communicator
        fn init_communicator(&mut self) -> TorshResult<()> {
            // Enhanced mock NCCL initialization with realistic behavior
            // This simulates the actual NCCL initialization process:
            // 1. Setting CUDA device: cudaSetDevice(self.device_id)
            // 2. Getting unique ID from rank 0: ncclGetUniqueId()
            // 3. Broadcasting unique ID to all ranks
            // 4. Initializing communicator: ncclCommInitRank()

            info!(
                " Enhanced Mock NCCL: Initializing communicator for device {} (rank {}/{})",
                self.device_id,
                self.rank(),
                self.world_size()
            );

            // Mock validation with comprehensive checks
            if self.world_size() == 0 {
                return Err(TorshDistributedError::invalid_argument(
                    "world_size",
                    "World size must be greater than 0",
                    "world_size > 0",
                ));
            }

            if self.rank() >= self.world_size() {
                return Err(TorshDistributedError::RankOutOfBounds {
                    rank: self.rank(),
                    world_size: self.world_size(),
                });
            }

            // Simulate CUDA device setting
            info!("   📱 Mock CUDA: Setting device {}", self.device_id);

            // Simulate unique ID generation (rank 0) and broadcast
            if self.rank() == 0 {
                info!("   🔑 Mock NCCL: Generating unique communicator ID");
            }
            info!("    Mock NCCL: Broadcasting unique ID to all ranks");

            // Simulate communicator initialization
            info!(
                "   🔧 Mock NCCL: Initializing communicator for rank {}",
                self.rank()
            );

            // Simulate initialization time
            std::thread::sleep(std::time::Duration::from_millis(50));

            info!("    Mock NCCL: Communicator successfully initialized");

            Ok(())
        }

        /// Get the device ID this backend is using
        pub fn device_id(&self) -> i32 {
            self.device_id
        }

        /// Check if NCCL backend is initialized
        pub fn is_initialized(&self) -> bool {
            self.initialized.load(std::sync::atomic::Ordering::Acquire)
        }
    }

    #[async_trait]
    impl Backend for NcclBackend {
        fn backend_type(&self) -> BackendType {
            BackendType::Nccl
        }

        async fn init(&mut self, _config: BackendConfig) -> TorshResult<()> {
            if self.initialized.load(Ordering::Acquire) {
                return Ok(());
            }

            self.init_communicator()?;
            self.initialized.store(true, Ordering::Release);

            info!(
                " Mock NCCL: Backend initialized for rank {}/{} on device {}",
                self.rank(),
                self.world_size(),
                self.device_id
            );

            Ok(())
        }

        async fn cleanup(&mut self) -> TorshResult<()> {
            if !self.initialized.load(Ordering::Acquire) {
                return Ok(());
            }

            // Enhanced mock NCCL cleanup
            // This simulates: ncclCommDestroy(comm)

            info!(
                "🧹 Enhanced Mock NCCL: Cleaning up backend for rank {} on device {}",
                self.rank(),
                self.device_id
            );

            // Simulate cleanup operations
            info!("   🔧 Destroying NCCL communicator");
            info!("   📱 Releasing CUDA resources");
            info!("    Freeing memory pools");

            // Simulate cleanup time
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;

            self.initialized.store(false, Ordering::Release);

            info!("    NCCL backend cleanup completed");

            Ok(())
        }

        fn is_ready(&self) -> bool {
            self.initialized.load(Ordering::Acquire)
        }

        fn rank(&self) -> u32 {
            self.rank
        }

        fn world_size(&self) -> u32 {
            self.world_size
        }

        async fn barrier(&mut self) -> TorshResult<()> {
            if !self.is_ready() {
                return Err(TorshDistributedError::backend_error(
                    "NCCL",
                    "Backend not initialized",
                ));
            }

            // Enhanced mock NCCL barrier using all-reduce approach
            // NCCL doesn't have a direct barrier, so we typically use:
            // 1. Create dummy data
            // 2. Call ncclAllReduce with sum operation
            // 3. Synchronize CUDA stream

            let start_time = std::time::Instant::now();

            info!(
                "🚧 Enhanced Mock NCCL: Barrier sync for rank {} on device {} ({} total ranks)",
                self.rank(),
                self.device_id,
                self.world_size()
            );

            // NCCL barrier is simulated by measuring simulated latency.
            // Real NCCL barriers use ncclAllReduce on a single sentinel element
            // followed by cudaStreamSynchronize; that path requires cudarc+nccl.
            let latency_ms = (self.world_size() as f64 * 2.0).max(5.0);
            std::thread::sleep(std::time::Duration::from_millis(latency_ms as u64));

            let duration = start_time.elapsed();

            info!(
                "    Mock NCCL barrier across {} ranks completed in {:?} (simulated latency only)",
                self.world_size(),
                duration
            );

            Ok(())
        }

        fn capabilities(&self) -> BackendCapabilities {
            BackendCapabilities {
                async_operations: true,
                // Honest: this is a simulated backend with no real GPU/NCCL
                // transport, so it does not actually support GPU collectives.
                gpu_support: false,
                p2p_communication: false,
                custom_reduce_ops: false,
                max_tensor_size: None,
                supported_dtypes: vec!["f32".to_string(), "f64".to_string()],
            }
        }

        fn status(&self) -> BackendStatus {
            BackendStatus {
                initialized: self.initialized.load(Ordering::Acquire),
                healthy: true,
                active_operations: 0,
                total_operations: 0,
                failed_operations: 0,
                last_error: None,
            }
        }

        async fn all_reduce(
            &mut self,
            _tensor: &mut (dyn Any + Send + Sync),
            _op: ReduceOp,
        ) -> TorshResult<()> {
            if !self.is_ready() {
                return Err(TorshDistributedError::backend_error(
                    "NCCL",
                    "Backend not initialized",
                ));
            }
            // Honest simulated NCCL: there is no real GPU/NCCL transport. A
            // single-rank reduction is the identity (data unchanged); a
            // multi-rank reduction cannot be performed, so return an error
            // instead of fabricating a result (the previous implementation
            // multiplied every element by world_size, silently corrupting data).
            if self.world_size() <= 1 {
                return Ok(());
            }
            Err(TorshDistributedError::backend_error(
                "NCCL",
                "simulated NCCL backend cannot perform multi-rank all_reduce; a real \
                 cudarc+nccl / oxicuda-comm transport is required (no data fabricated)",
            ))
        }

        async fn all_gather(
            &mut self,
            tensor: &(dyn Any + Send + Sync),
        ) -> TorshResult<Box<dyn Any + Send>> {
            if !self.is_ready() {
                return Err(TorshDistributedError::backend_error(
                    "NCCL",
                    "Backend not initialized",
                ));
            }
            if self.world_size() <= 1 {
                // Single rank: the gather is just this rank's own buffer.
                if let Some(data) = tensor.downcast_ref::<Vec<f32>>() {
                    return Ok(Box::new(data.clone()));
                }
                return Err(TorshDistributedError::backend_error(
                    "NCCL all_gather",
                    "unsupported tensor type (expected Vec<f32>)",
                ));
            }
            Err(TorshDistributedError::backend_error(
                "NCCL",
                "simulated NCCL backend cannot perform multi-rank all_gather; a real \
                 cudarc+nccl / oxicuda-comm transport is required (no data fabricated)",
            ))
        }

        async fn broadcast(
            &mut self,
            _tensor: &mut (dyn Any + Send + Sync),
            root: u32,
        ) -> TorshResult<()> {
            if !self.is_ready() {
                return Err(TorshDistributedError::backend_error(
                    "NCCL",
                    "Backend not initialized",
                ));
            }
            if root >= self.world_size() {
                return Err(TorshDistributedError::RankOutOfBounds {
                    rank: root,
                    world_size: self.world_size(),
                });
            }
            if self.world_size() <= 1 {
                return Ok(());
            }
            Err(TorshDistributedError::backend_error(
                "NCCL",
                "simulated NCCL backend cannot perform multi-rank broadcast; a real \
                 cudarc+nccl / oxicuda-comm transport is required (no data fabricated)",
            ))
        }

        async fn send(
            &mut self,
            _tensor: &(dyn Any + Send + Sync),
            dst: u32,
            _tag: u32,
        ) -> TorshResult<()> {
            if !self.is_ready() {
                return Err(TorshDistributedError::backend_error(
                    "NCCL",
                    "Backend not initialized",
                ));
            }
            if dst >= self.world_size() {
                return Err(TorshDistributedError::RankOutOfBounds {
                    rank: dst,
                    world_size: self.world_size(),
                });
            }
            Err(TorshDistributedError::backend_error(
                "NCCL",
                "simulated NCCL backend cannot perform point-to-point send; a real \
                 cudarc+nccl / oxicuda-comm transport is required",
            ))
        }

        async fn recv(&mut self, src: u32, _tag: u32) -> TorshResult<Box<dyn Any + Send>> {
            if !self.is_ready() {
                return Err(TorshDistributedError::backend_error(
                    "NCCL",
                    "Backend not initialized",
                ));
            }
            if src >= self.world_size() {
                return Err(TorshDistributedError::RankOutOfBounds {
                    rank: src,
                    world_size: self.world_size(),
                });
            }
            Err(TorshDistributedError::backend_error(
                "NCCL",
                "simulated NCCL backend cannot perform point-to-point recv; a real \
                 cudarc+nccl / oxicuda-comm transport is required",
            ))
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
    }

    impl Drop for NcclBackend {
        fn drop(&mut self) {
            std::mem::drop(self.cleanup());
        }
    }
}

#[cfg(feature = "nccl")]
pub use nccl_backend::NcclBackend;
