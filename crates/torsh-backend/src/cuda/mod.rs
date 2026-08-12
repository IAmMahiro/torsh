//! CUDA backend for ToRSh deep learning framework
//!
//! # Status: pure-Rust honest-fallback shim
//!
//! ToRSh's real GPU compute path lives in `torsh-tensor`'s oxicuda-based
//! `GpuDispatch` (see `torsh-tensor/src/cuda_backend/`), which loads the CUDA
//! driver at runtime via libloading and needs no CUDA SDK at build time.
//!
//! This module used to carry a second, independent CUDA stack built on the
//! abandoned `cust` / `cuda-sys` / `cudnn-sys` C-FFI crates plus ~73k lines of
//! parked scheduling/fusion scaffolding. That stack could only compile on an
//! x86_64 host with the CUDA SDK installed, was never reachable from any other
//! crate, and duplicated (incompatibly) the oxicuda path. It has been removed.
//!
//! What remains is the pure-Rust [`fallback`] implementation: every entry point
//! either delegates to the CPU backend or returns an honest error, so the crate
//! builds identically on every host and advertises no GPU capability it cannot
//! deliver. Callers wanting real CUDA acceleration go through `torsh-tensor`.

pub mod fallback;

// The fallback surface is the crate's only CUDA API. It is pure Rust and builds
// on every platform.
pub use fallback::*;

/// Re-export commonly used types.
pub mod prelude {
    pub use crate::prelude::*;
}

/// CUDA runtime entry points (pure-Rust fallback: no device is ever available
/// through this crate; use `torsh-tensor`'s oxicuda path for real GPU work).
mod cuda_impl {
    use super::*;

    /// Initialize CUDA backend (no CUDA available through this crate).
    pub fn init() -> Result<(), CudaError> {
        Err(CudaError::RuntimeError(
            "CUDA not available through torsh-backend; use torsh-tensor's GPU path".to_string(),
        ))
    }

    /// Check if CUDA is available (always false through this crate).
    pub fn is_available() -> bool {
        false
    }

    /// Get number of CUDA devices (none through this crate).
    pub fn device_count() -> Result<u32, CudaError> {
        Ok(0)
    }

    /// Get current CUDA device (none through this crate).
    pub fn current_device() -> Result<CudaDevice, CudaError> {
        Err(CudaError::RuntimeError(
            "CUDA not available through torsh-backend; use torsh-tensor's GPU path".to_string(),
        ))
    }

    /// Set current CUDA device (none through this crate).
    pub fn set_device(_device_id: usize) -> Result<(), CudaError> {
        Err(CudaError::RuntimeError(
            "CUDA not available through torsh-backend; use torsh-tensor's GPU path".to_string(),
        ))
    }

    /// Synchronize current device (no-op: nothing to synchronize).
    pub fn synchronize() -> Result<(), CudaError> {
        Ok(())
    }
}

// Re-export the implementation.
pub use cuda_impl::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_unavailable_through_backend() {
        // torsh-backend never exposes a real CUDA device; the real path is
        // torsh-tensor's oxicuda GpuDispatch.
        assert!(!is_available());
        assert_eq!(device_count().expect("device_count is infallible here"), 0);
        assert!(init().is_err());
        assert!(current_device().is_err());
        assert!(set_device(0).is_err());
        assert!(synchronize().is_ok());
    }
}
