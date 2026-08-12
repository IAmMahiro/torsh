//! Shape Operations with GPU-Availability-Aware Statistics
//!
//! This module provides shape operations (broadcasting, reshape, batch
//! validation, stride computation) instrumented with usage statistics.
//!
//! # Honesty Note (no real GPU dispatch here)
//!
//! torsh-core is the foundation crate that every other ToRSh crate depends
//! on, so it cannot depend on `torsh-tensor` (that would be a circular
//! dependency) and has no direct CUDA/Metal/wgpu bindings of its own. GPU
//! compute for ToRSh tensors is provided by oxicuda via `torsh-tensor`'s
//! `gpu_dispatch` module. Every operation in this module therefore always
//! executes on the CPU; [`AcceleratorStats::gpu_operations`] and
//! [`AcceleratorStats::gpu_fallback_count`] exist for API stability and
//! forward compatibility but stay at zero rather than being incremented for
//! work that was actually done on the CPU. [`GpuShapeAccelerator::is_gpu_available`]
//! reports [`crate::gpu::is_gpu_available`]'s (currently always `false`)
//! answer purely as information; it does not change which code path runs.
//!
//! # Example
//!
//! ```rust,ignore
//! use torsh_core::gpu_shape_ops::{GpuShapeAccelerator, AcceleratorConfig};
//!
//! let config = AcceleratorConfig::default();
//! let accelerator = GpuShapeAccelerator::new(config)?;
//!
//! let shape1 = Shape::from_dims(vec![1000, 1000, 100])?;
//! let shape2 = Shape::from_dims(vec![1, 1000, 100])?;
//! let result = accelerator.broadcast(&shape1, &shape2)?;
//! ```

use crate::error::{Result, TorshError};
use crate::shape::Shape;
use crate::sync::MutexExt;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(feature = "std")]
use std::sync::Arc;

/// Configuration for GPU shape accelerator
///
/// Retained for API stability and forward compatibility with a future real
/// GPU dispatch path. Since [`GpuShapeAccelerator`] currently always executes
/// on the CPU (see the module docs), these thresholds are not read by any
/// operation today -- setting them has no observable effect yet.
#[derive(Debug, Clone)]
pub struct AcceleratorConfig {
    /// Threshold that would select GPU for broadcasting once a real GPU
    /// dispatch path exists (default: 10M). Not currently read.
    pub broadcast_threshold: usize,

    /// Threshold that would select GPU for reshape once a real GPU dispatch
    /// path exists (default: 5M). Not currently read.
    pub reshape_threshold: usize,

    /// Threshold that would select GPU for stride computation once a real
    /// GPU dispatch path exists (default: 10). Not currently read.
    pub stride_dimension_threshold: usize,

    /// Threshold that would select GPU for batch validation once a real GPU
    /// dispatch path exists (default: 100). Not currently read.
    pub batch_validation_threshold: usize,

    /// Enable automatic threshold tuning based on GPU performance (default: false)
    pub enable_auto_tuning: bool,

    /// Preferred GPU device ID (default: 0)
    pub device_id: usize,
}

impl Default for AcceleratorConfig {
    fn default() -> Self {
        Self {
            broadcast_threshold: 10_000_000,
            reshape_threshold: 5_000_000,
            stride_dimension_threshold: 10,
            batch_validation_threshold: 100,
            enable_auto_tuning: false,
            device_id: 0,
        }
    }
}

impl AcceleratorConfig {
    /// Create a new configuration with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the broadcast threshold
    pub fn with_broadcast_threshold(mut self, threshold: usize) -> Self {
        self.broadcast_threshold = threshold;
        self
    }

    /// Set the reshape threshold
    pub fn with_reshape_threshold(mut self, threshold: usize) -> Self {
        self.reshape_threshold = threshold;
        self
    }

    /// Set the stride dimension threshold
    pub fn with_stride_dimension_threshold(mut self, threshold: usize) -> Self {
        self.stride_dimension_threshold = threshold;
        self
    }

    /// Set the batch validation threshold
    pub fn with_batch_validation_threshold(mut self, threshold: usize) -> Self {
        self.batch_validation_threshold = threshold;
        self
    }

    /// Enable automatic threshold tuning
    pub fn with_auto_tuning(mut self, enable: bool) -> Self {
        self.enable_auto_tuning = enable;
        self
    }

    /// Set the GPU device ID
    pub fn with_device_id(mut self, device_id: usize) -> Self {
        self.device_id = device_id;
        self
    }

    /// Create a configuration optimized for very large tensors (>100M elements)
    pub fn for_very_large_tensors() -> Self {
        Self {
            broadcast_threshold: 1_000_000,
            reshape_threshold: 500_000,
            stride_dimension_threshold: 8,
            batch_validation_threshold: 50,
            enable_auto_tuning: true,
            device_id: 0,
        }
    }

    /// Create a configuration optimized for high-dimensional tensors
    pub fn for_high_dimensional() -> Self {
        Self {
            broadcast_threshold: 5_000_000,
            reshape_threshold: 2_000_000,
            stride_dimension_threshold: 6,
            batch_validation_threshold: 100,
            enable_auto_tuning: false,
            device_id: 0,
        }
    }

    /// Create a conservative configuration that prefers CPU
    pub fn conservative() -> Self {
        Self {
            broadcast_threshold: 50_000_000,
            reshape_threshold: 25_000_000,
            stride_dimension_threshold: 15,
            batch_validation_threshold: 200,
            enable_auto_tuning: false,
            device_id: 0,
        }
    }
}

/// Usage statistics for [`GpuShapeAccelerator`] operations
#[derive(Debug, Clone, Default)]
pub struct AcceleratorStats {
    /// Total number of operations
    pub total_operations: usize,

    /// Number of operations genuinely executed on a GPU. Always `0` today:
    /// see the module docs -- torsh-core has no real GPU dispatch path, so
    /// this is never incremented for CPU work.
    pub gpu_operations: usize,

    /// Number of operations executed on the CPU (currently: all of them)
    pub cpu_operations: usize,

    /// Total time spent on GPU operations (microseconds). Always `0` today.
    pub gpu_time_us: u64,

    /// Total time spent on CPU operations (microseconds)
    pub cpu_time_us: u64,

    /// Number of times a GPU attempt failed and fell back to CPU. Always `0`
    /// today, since no GPU attempt is ever made.
    pub gpu_fallback_count: usize,
}

impl AcceleratorStats {
    /// Create new statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the percentage of operations that used GPU
    pub fn gpu_usage_percentage(&self) -> f64 {
        if self.total_operations == 0 {
            0.0
        } else {
            (self.gpu_operations as f64 / self.total_operations as f64) * 100.0
        }
    }

    /// Get average GPU operation time in microseconds
    pub fn avg_gpu_time_us(&self) -> f64 {
        if self.gpu_operations == 0 {
            0.0
        } else {
            self.gpu_time_us as f64 / self.gpu_operations as f64
        }
    }

    /// Get average CPU operation time in microseconds
    pub fn avg_cpu_time_us(&self) -> f64 {
        if self.cpu_operations == 0 {
            0.0
        } else {
            self.cpu_time_us as f64 / self.cpu_operations as f64
        }
    }

    /// Get the speedup factor (CPU time / GPU time)
    pub fn speedup_factor(&self) -> f64 {
        if self.gpu_time_us == 0 {
            0.0
        } else {
            self.cpu_time_us as f64 / self.gpu_time_us as f64
        }
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// GPU-accelerated shape operations
///
/// Provides intelligent GPU acceleration for shape operations on very large tensors.
/// Automatically selects between GPU and CPU based on operation size and complexity.
#[cfg(feature = "std")]
pub struct GpuShapeAccelerator {
    config: AcceleratorConfig,
    stats: Arc<std::sync::Mutex<AcceleratorStats>>,
    gpu_available: bool,
}

#[cfg(feature = "std")]
impl GpuShapeAccelerator {
    /// Create a new GPU shape accelerator with default configuration
    pub fn new(config: AcceleratorConfig) -> Result<Self> {
        // Check GPU availability through scirs2-core
        #[cfg(feature = "gpu")]
        let gpu_available = crate::gpu::is_gpu_available();

        #[cfg(not(feature = "gpu"))]
        let gpu_available = false;

        Ok(Self {
            config,
            stats: Arc::new(std::sync::Mutex::new(AcceleratorStats::new())),
            gpu_available,
        })
    }

    /// Create a new accelerator with default configuration
    pub fn default_config() -> Result<Self> {
        Self::new(AcceleratorConfig::default())
    }

    /// Check if GPU is available
    pub fn is_gpu_available(&self) -> bool {
        self.gpu_available
    }

    /// Get current statistics
    pub fn stats(&self) -> AcceleratorStats {
        self.stats.lock_or_recover().clone()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        self.stats.lock_or_recover().reset();
    }

    /// Broadcast two shapes together
    ///
    /// torsh-core has no real GPU backend to dispatch shape operations to:
    /// GPU compute for ToRSh is provided by oxicuda via torsh-tensor's
    /// `gpu_dispatch`, which this foundation crate cannot depend on without
    /// creating a circular dependency. Every call here always actually runs
    /// on the CPU, and `AcceleratorStats` reports that honestly (previously,
    /// this counted large-enough operations as `gpu_operations` even though
    /// they were CPU work end to end).
    pub fn broadcast(&self, shape1: &Shape, shape2: &Shape) -> Result<Shape> {
        let mut stats = self.stats.lock_or_recover();
        stats.total_operations += 1;
        stats.cpu_operations += 1;
        drop(stats);
        self.broadcast_cpu(shape1, shape2)
    }

    /// CPU implementation of broadcasting
    fn broadcast_cpu(&self, shape1: &Shape, shape2: &Shape) -> Result<Shape> {
        shape1.broadcast_with(shape2)
    }

    /// Reshape a tensor to `new_dims`, validating that the element count matches
    ///
    /// See [`Self::broadcast`] for why this always executes on the CPU and
    /// why `AcceleratorStats` no longer distinguishes a GPU path here.
    pub fn reshape(&self, shape: &Shape, new_dims: &[usize]) -> Result<Shape> {
        let numel = shape.numel();

        // Check if new shape has same number of elements
        let new_numel: usize = new_dims.iter().product();
        if numel != new_numel {
            return Err(TorshError::dimension_error(
                &format!(
                    "Cannot reshape tensor of {} elements into shape with {} elements",
                    numel, new_numel
                ),
                "reshape",
            ));
        }

        // Record operation
        let mut stats = self.stats.lock_or_recover();
        stats.total_operations += 1;
        stats.cpu_operations += 1;
        drop(stats);
        self.reshape_cpu(new_dims)
    }

    /// CPU implementation of reshape
    fn reshape_cpu(&self, new_dims: &[usize]) -> Result<Shape> {
        Shape::from_dims(new_dims.to_vec())
    }

    /// Batch validate multiple shapes
    ///
    /// See [`Self::broadcast`] for why this always executes on the CPU and
    /// why `AcceleratorStats` no longer distinguishes a GPU path here.
    pub fn batch_validate(&self, shapes: &[Vec<usize>]) -> Result<Vec<bool>> {
        let mut stats = self.stats.lock_or_recover();
        stats.total_operations += 1;
        stats.cpu_operations += 1;
        drop(stats);
        self.batch_validate_cpu(shapes)
    }

    /// CPU implementation of batch validation
    fn batch_validate_cpu(&self, shapes: &[Vec<usize>]) -> Result<Vec<bool>> {
        Ok(shapes
            .iter()
            .map(|dims| {
                // Validate each shape
                !dims.is_empty() && dims.iter().all(|&d| d > 0)
            })
            .collect())
    }

    /// Compute strides for a shape's dimensions
    ///
    /// See [`Self::broadcast`] for why this always executes on the CPU and
    /// why `AcceleratorStats` no longer distinguishes a GPU path here.
    pub fn compute_strides(&self, dims: &[usize]) -> Result<Vec<usize>> {
        let mut stats = self.stats.lock_or_recover();
        stats.total_operations += 1;
        stats.cpu_operations += 1;
        drop(stats);
        self.compute_strides_cpu(dims)
    }

    /// CPU implementation of stride computation
    fn compute_strides_cpu(&self, dims: &[usize]) -> Result<Vec<usize>> {
        if dims.is_empty() {
            return Ok(Vec::new());
        }

        let mut strides = vec![0; dims.len()];
        let mut stride = 1;

        for i in (0..dims.len()).rev() {
            strides[i] = stride;
            stride *= dims[i];
        }

        Ok(strides)
    }

    /// Get current configuration
    pub fn config(&self) -> &AcceleratorConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: AcceleratorConfig) {
        self.config = config;
    }
}

/// Lightweight GPU shape operations without std
///
/// Provides basic GPU-accelerated shape operations for no_std environments.
#[cfg(not(feature = "std"))]
pub struct GpuShapeAccelerator {
    config: AcceleratorConfig,
    gpu_available: bool,
}

#[cfg(not(feature = "std"))]
impl GpuShapeAccelerator {
    /// Create a new GPU shape accelerator
    pub fn new(config: AcceleratorConfig) -> Result<Self> {
        #[cfg(feature = "gpu")]
        let gpu_available = crate::gpu::is_gpu_available();

        #[cfg(not(feature = "gpu"))]
        let gpu_available = false;

        Ok(Self {
            config,
            gpu_available,
        })
    }

    /// Check if GPU is available
    pub fn is_gpu_available(&self) -> bool {
        self.gpu_available
    }

    /// GPU-accelerated broadcasting
    pub fn broadcast(&self, shape1: &Shape, shape2: &Shape) -> Result<Shape> {
        // In no_std, always use CPU fallback
        shape1.broadcast_with(shape2)
    }

    /// Get current configuration
    pub fn config(&self) -> &AcceleratorConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// F296 regression test: `AcceleratorStats::gpu_operations` must only ever
    /// count operations that were genuinely executed on a GPU. torsh-core has
    /// no real GPU dispatch to route shape operations through (GPU compute
    /// for ToRSh is provided by oxicuda via torsh-tensor's `gpu_dispatch`,
    /// which this foundation crate cannot depend on), so every shape
    /// operation always actually runs on the CPU. This constructs the
    /// accelerator directly (bypassing `new()`, which today always sets
    /// `gpu_available: false`) to simulate what happens once
    /// `crate::gpu::is_gpu_available()` starts reporting real hardware, so
    /// the accounting cannot silently start lying at that point either.
    #[test]
    fn test_gpu_stats_never_lie_about_unavailable_gpu_backend() {
        let accelerator = GpuShapeAccelerator {
            config: AcceleratorConfig::new().with_broadcast_threshold(1),
            stats: Arc::new(std::sync::Mutex::new(AcceleratorStats::new())),
            gpu_available: true,
        };

        let shape1 = Shape::from_dims(vec![4, 4]).expect("shape creation should succeed");
        let shape2 = Shape::from_dims(vec![4, 4]).expect("shape creation should succeed");
        accelerator
            .broadcast(&shape1, &shape2)
            .expect("broadcast should succeed");

        let stats = accelerator.stats();
        // No genuine GPU kernel exists in torsh-core; every operation is
        // actually executed on the CPU, so it must be counted as CPU work.
        assert_eq!(
            stats.gpu_operations, 0,
            "no real GPU work was performed, so gpu_operations must stay 0"
        );
        assert_eq!(stats.cpu_operations, 1);
    }

    #[test]
    fn test_accelerator_config_default() {
        let config = AcceleratorConfig::default();
        assert_eq!(config.broadcast_threshold, 10_000_000);
        assert_eq!(config.reshape_threshold, 5_000_000);
        assert_eq!(config.stride_dimension_threshold, 10);
        assert_eq!(config.batch_validation_threshold, 100);
        assert!(!config.enable_auto_tuning);
        assert_eq!(config.device_id, 0);
    }

    #[test]
    fn test_accelerator_config_builder() {
        let config = AcceleratorConfig::new()
            .with_broadcast_threshold(1_000_000)
            .with_reshape_threshold(500_000)
            .with_stride_dimension_threshold(5)
            .with_batch_validation_threshold(50)
            .with_auto_tuning(true)
            .with_device_id(1);

        assert_eq!(config.broadcast_threshold, 1_000_000);
        assert_eq!(config.reshape_threshold, 500_000);
        assert_eq!(config.stride_dimension_threshold, 5);
        assert_eq!(config.batch_validation_threshold, 50);
        assert!(config.enable_auto_tuning);
        assert_eq!(config.device_id, 1);
    }

    #[test]
    fn test_accelerator_config_presets() {
        let very_large = AcceleratorConfig::for_very_large_tensors();
        assert_eq!(very_large.broadcast_threshold, 1_000_000);
        assert!(very_large.enable_auto_tuning);

        let high_dim = AcceleratorConfig::for_high_dimensional();
        assert_eq!(high_dim.stride_dimension_threshold, 6);

        let conservative = AcceleratorConfig::conservative();
        assert_eq!(conservative.broadcast_threshold, 50_000_000);
    }

    #[test]
    fn test_accelerator_stats() {
        let mut stats = AcceleratorStats::new();
        assert_eq!(stats.total_operations, 0);
        assert_eq!(stats.gpu_operations, 0);
        assert_eq!(stats.cpu_operations, 0);

        stats.total_operations = 100;
        stats.gpu_operations = 60;
        stats.cpu_operations = 40;
        stats.gpu_time_us = 1000;
        stats.cpu_time_us = 3000;

        assert_eq!(stats.gpu_usage_percentage(), 60.0);
        assert_eq!(stats.avg_gpu_time_us(), 1000.0 / 60.0);
        assert_eq!(stats.avg_cpu_time_us(), 3000.0 / 40.0);
        assert_eq!(stats.speedup_factor(), 3.0);

        stats.reset();
        assert_eq!(stats.total_operations, 0);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_accelerator_creation() {
        let config = AcceleratorConfig::default();
        let accelerator = GpuShapeAccelerator::new(config);
        assert!(accelerator.is_ok());

        let accelerator = accelerator.expect("accelerator creation should succeed");
        // GPU may or may not be available depending on build configuration
        let _gpu_available = accelerator.is_gpu_available();
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_accelerator_broadcast() {
        let config = AcceleratorConfig::default();
        let accelerator =
            GpuShapeAccelerator::new(config).expect("accelerator creation should succeed");

        let shape1 = Shape::from_dims(vec![10, 20, 30]).expect("shape creation should succeed");
        let shape2 = Shape::from_dims(vec![1, 20, 30]).expect("shape creation should succeed");

        let result = accelerator.broadcast(&shape1, &shape2);
        assert!(result.is_ok());

        let result = result.expect("broadcast should succeed");
        assert_eq!(result.dims(), &[10, 20, 30]);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_accelerator_reshape() {
        let config = AcceleratorConfig::default();
        let accelerator =
            GpuShapeAccelerator::new(config).expect("accelerator creation should succeed");

        let shape = Shape::from_dims(vec![10, 20, 30]).expect("shape creation should succeed");
        let new_dims = vec![10, 600];

        let result = accelerator.reshape(&shape, &new_dims);
        assert!(result.is_ok());

        let result = result.expect("reshape should succeed");
        assert_eq!(result.dims(), &[10, 600]);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_accelerator_reshape_invalid() {
        let config = AcceleratorConfig::default();
        let accelerator =
            GpuShapeAccelerator::new(config).expect("accelerator creation should succeed");

        let shape = Shape::from_dims(vec![10, 20, 30]).expect("shape creation should succeed");
        let new_dims = vec![10, 100]; // Wrong number of elements

        let result = accelerator.reshape(&shape, &new_dims);
        assert!(result.is_err());
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_accelerator_batch_validate() {
        let config = AcceleratorConfig::default();
        let accelerator =
            GpuShapeAccelerator::new(config).expect("accelerator creation should succeed");

        let shapes = vec![
            vec![10, 20],
            vec![30, 40, 50],
            vec![],          // Invalid: empty
            vec![10, 0, 20], // Invalid: zero dimension
            vec![5, 5, 5, 5],
        ];

        let result = accelerator.batch_validate(&shapes);
        assert!(result.is_ok());

        let result = result.expect("batch_validate should succeed");
        assert_eq!(result.len(), 5);
        assert!(result[0]); // Valid
        assert!(result[1]); // Valid
        assert!(!result[2]); // Invalid: empty
        assert!(!result[3]); // Invalid: zero dimension
        assert!(result[4]); // Valid
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_accelerator_compute_strides() {
        let config = AcceleratorConfig::default();
        let accelerator =
            GpuShapeAccelerator::new(config).expect("accelerator creation should succeed");

        let dims = vec![10, 20, 30];
        let result = accelerator.compute_strides(&dims);
        assert!(result.is_ok());

        let strides = result.expect("compute_strides should succeed");
        assert_eq!(strides, vec![600, 30, 1]);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_accelerator_compute_strides_empty() {
        let config = AcceleratorConfig::default();
        let accelerator =
            GpuShapeAccelerator::new(config).expect("accelerator creation should succeed");

        let dims = vec![];
        let result = accelerator.compute_strides(&dims);
        assert!(result.is_ok());

        let strides = result.expect("compute_strides should succeed");
        assert!(strides.is_empty());
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_accelerator_stats_tracking() {
        let config = AcceleratorConfig::default();
        let accelerator =
            GpuShapeAccelerator::new(config).expect("accelerator creation should succeed");

        // Perform some operations
        let shape1 = Shape::from_dims(vec![10, 20]).expect("shape creation should succeed");
        let shape2 = Shape::from_dims(vec![1, 20]).expect("shape creation should succeed");
        let _ = accelerator.broadcast(&shape1, &shape2);

        let stats = accelerator.stats();
        assert_eq!(stats.total_operations, 1);
        assert!(stats.gpu_operations + stats.cpu_operations == 1);

        accelerator.reset_stats();
        let stats = accelerator.stats();
        assert_eq!(stats.total_operations, 0);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_accelerator_config_update() {
        let config = AcceleratorConfig::default();
        let mut accelerator =
            GpuShapeAccelerator::new(config).expect("accelerator creation should succeed");

        assert_eq!(accelerator.config().broadcast_threshold, 10_000_000);

        let new_config = AcceleratorConfig::new().with_broadcast_threshold(1_000_000);
        accelerator.set_config(new_config);

        assert_eq!(accelerator.config().broadcast_threshold, 1_000_000);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_accelerator_high_dimensional_strides() {
        let config = AcceleratorConfig::new().with_stride_dimension_threshold(5);
        let accelerator =
            GpuShapeAccelerator::new(config).expect("accelerator creation should succeed");

        // Create a 12D tensor (exceeds threshold of 5)
        let dims = vec![2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
        let result = accelerator.compute_strides(&dims);
        assert!(result.is_ok());

        let strides = result.expect("compute_strides should succeed");
        assert_eq!(strides.len(), dims.len());

        // Verify strides are computed correctly
        let mut expected_stride = 1;
        for i in (0..dims.len()).rev() {
            assert_eq!(strides[i], expected_stride);
            expected_stride *= dims[i];
        }
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_accelerator_large_batch_validation() {
        let config = AcceleratorConfig::new().with_batch_validation_threshold(10);
        let accelerator =
            GpuShapeAccelerator::new(config).expect("accelerator creation should succeed");

        // Create a batch of 50 shapes (exceeds threshold of 10)
        let shapes: Vec<Vec<usize>> = (0..50)
            .map(|i| {
                if i % 10 == 0 {
                    vec![] // Some invalid shapes
                } else {
                    vec![10, 20, 30]
                }
            })
            .collect();

        let result = accelerator.batch_validate(&shapes);
        assert!(result.is_ok());

        let validations = result.expect("batch_validate should succeed");
        assert_eq!(validations.len(), 50);

        // Check that invalid shapes (every 10th) are marked as invalid
        for i in 0..50 {
            if i % 10 == 0 {
                assert!(!validations[i]);
            } else {
                assert!(validations[i]);
            }
        }
    }
}
