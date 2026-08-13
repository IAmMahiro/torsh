//! Process group management for distributed training

#![allow(unexpected_cfgs)]

use crate::backend::{Backend, BackendConfig, BackendType};
use crate::tcp_backend::TcpBackend;
use crate::{TorshDistributedError, TorshResult};
use parking_lot::RwLock;
use std::sync::Arc;

/// Process rank type
pub type Rank = u32;

/// World size type
pub type WorldSize = u32;

/// Process group for distributed communication
pub struct ProcessGroup {
    backend: Arc<RwLock<Box<dyn Backend>>>,
    rank: Rank,
    world_size: WorldSize,
    master_addr: String,
    master_port: u16,
}

impl ProcessGroup {
    /// Create a new process group
    pub async fn new(
        backend_type: BackendType,
        rank: Rank,
        world_size: WorldSize,
        master_addr: &str,
        master_port: u16,
    ) -> TorshResult<Self> {
        let mut backend = create_backend(backend_type, rank, world_size, master_addr, master_port)?;

        // Initialize the backend with default config
        let config = BackendConfig::default();
        backend.init(config).await?;

        let pg = Self {
            backend: Arc::new(RwLock::new(backend)),
            rank,
            world_size,
            master_addr: master_addr.to_string(),
            master_port,
        };

        Ok(pg)
    }

    /// Get the rank of this process
    pub fn rank(&self) -> Rank {
        self.rank
    }

    /// Get the world size
    pub fn world_size(&self) -> WorldSize {
        self.world_size
    }

    /// Get the backend type
    pub fn backend_type(&self) -> BackendType {
        self.backend.read().backend_type()
    }

    /// Get the master (rendezvous) address this group was configured with
    pub fn master_addr(&self) -> &str {
        &self.master_addr
    }

    /// Get the master (rendezvous) port this group was configured with
    pub fn master_port(&self) -> u16 {
        self.master_port
    }

    /// Get a reference to the backend
    pub fn backend(&self) -> &Arc<RwLock<Box<dyn Backend>>> {
        &self.backend
    }
}

/// Create a backend based on the type.
///
/// Honesty policy: this only ever returns a backend that can actually perform
/// the collectives it advertises. There is NO mock substitution in production
/// paths.
///
/// - `Gloo` (default): the real pure-Rust [`TcpBackend`] (store-based TCP
///   collectives). Works on a single node across processes/threads.
/// - `Mpi` (feature `mpi`): the real `MpiBackend` bound to a live MPI
///   communicator. Requires a system MPI library.
/// - `Nccl`: an honest error — no real NCCL/oxicuda-comm transport exists yet,
///   so callers must not receive a backend that silently fabricates results.
/// - `Custom`: an honest error until a concrete implementation is registered.
fn create_backend(
    backend_type: BackendType,
    rank: Rank,
    world_size: WorldSize,
    master_addr: &str,
    master_port: u16,
) -> TorshResult<Box<dyn Backend>> {
    match backend_type {
        BackendType::Nccl => Err(TorshDistributedError::feature_not_available(
            "NCCL backend",
            "a real NCCL/oxicuda-comm implementation (not yet available); \
             the previous mock has been removed to avoid fabricated collectives",
        )),
        #[cfg(feature = "mpi")]
        BackendType::Mpi => {
            // Wire the real MPI backend (bound to a live MPI communicator).
            // rank/world_size come from MPI itself, not the caller's arguments.
            Ok(Box::new(crate::backend::MpiBackend::new()?))
        }
        #[cfg(not(feature = "mpi"))]
        BackendType::Mpi => Err(TorshDistributedError::feature_not_available(
            "MPI backend",
            "mpi",
        )),
        BackendType::Gloo => Ok(Box::new(TcpBackend::new(
            rank,
            world_size,
            master_addr,
            master_port,
        ))),
        BackendType::Custom(name) => Err(TorshDistributedError::feature_not_available(
            format!("Custom backend: {}", name),
            "custom backend implementation",
        )),
    }
}
