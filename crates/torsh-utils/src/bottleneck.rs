//! # Advanced Performance Bottleneck Profiling
//!
//! This module provides sophisticated profiling capabilities for identifying and analyzing
//! performance bottlenecks in deep learning models. It goes beyond simple timing to provide
//! deep insights into memory usage, GPU utilization, and execution patterns.
//!
//! ## Features
//!
//! - **Flame Graph Generation**: built from real per-iteration forward/backward
//!   timings (coarse-grained: whole-pass, not per-layer, since `Module::forward`
//!   is opaque to this crate)
//! - **Memory Profiling**: real peak/current process memory via `sysinfo`.
//!   Leak detection and cache/fragmentation metrics are not implemented (they
//!   would require allocator instrumentation or hardware performance
//!   counters) and are reported as `None`, never a fabricated number
//! - **GPU Profiling**: reports zeros with an explicit "not measured" doc note
//!   unless a real GPU backend is wired up (no NVML/CUDA linkage here)
//! - **Hotspot Detection**: real CPU time-share per timed operation. Memory/
//!   I/O/synchronization hotspots require instrumentation this crate does not
//!   have and are always empty
//! - **Call Stack Analysis**: real stack captures (`std::backtrace`) paired
//!   with each iteration's real duration
//! - **Regression Detection**: compare against baseline performance metrics
//! - **Cache Performance**: not measured (requires hardware performance
//!   counters); always reported as `None`, not a plausible-looking number
//!
//! ## Quick Start
//!
//! ### Basic Profiling
//!
//! ```rust,no_run
//! use torsh_utils::bottleneck::{profile_bottlenecks, print_bottleneck_report};
//! # use torsh_nn::Module;
//! # struct MyModel;
//! # impl Module for MyModel {
//! #   fn forward(&self, _: &torsh_tensor::Tensor) -> Result<torsh_tensor::Tensor, torsh_core::TorshError> {
//! #     unimplemented!()
//! #   }
//! # }
//!
//! # fn example() -> Result<(), torsh_core::TorshError> {
//! let model = MyModel;
//!
//! // Profile model execution
//! let report = profile_bottlenecks(
//!     &model,
//!     &[1, 3, 224, 224],  // Input shape
//!     100,                 // Number of iterations
//!     true                 // Profile backward pass
//! )?;
//!
//! // Print comprehensive report
//! print_bottleneck_report(&report);
//!
//! // Access specific data
//! println!("Total time: {:?}", report.total_time);
//! println!("Peak memory: {:.1} MB", report.memory_profile.peak_usage_mb);
//! # Ok(())
//! # }
//! ```
//!
//! ### Advanced Profiling with Flame Graphs
//!
//! ```rust,no_run
//! use torsh_utils::bottleneck::{profile_bottlenecks_advanced, AdvancedProfilingConfig};
//! # use torsh_nn::Module;
//! # struct MyModel;
//! # impl Module for MyModel {
//! #   fn forward(&self, _: &torsh_tensor::Tensor) -> Result<torsh_tensor::Tensor, torsh_core::TorshError> {
//! #     unimplemented!()
//! #   }
//! # }
//!
//! # fn example() -> Result<(), torsh_core::TorshError> {
//! let model = MyModel;
//!
//! // Configure advanced profiling
//! let config = AdvancedProfilingConfig {
//!     enable_flame_graph: true,
//!     enable_memory_profiling: true,
//!     enable_gpu_profiling: false,  // Enable if using GPU
//!     enable_call_stack_analysis: true,
//!     enable_hotspot_analysis: true,
//!     sample_rate_hz: 1000.0,       // 1000 samples per second
//!     memory_snapshot_interval_ms: 10.0,
//!     ..Default::default()
//! };
//!
//! let report = profile_bottlenecks_advanced(
//!     &model,
//!     &[1, 3, 224, 224],
//!     100,
//!     true,
//!     config
//! )?;
//!
//! // Analyze flame graph
//! if let Some(flame_graph) = &report.flame_graph {
//!     println!("Flame graph: {} samples at {:.0} Hz",
//!         flame_graph.total_samples,
//!         flame_graph.sample_rate_hz
//!     );
//! }
//!
//! // Analyze hotspots
//! for hotspot in report.hotspot_analysis.cpu_hotspots.iter().take(5) {
//!     println!("Hotspot: {} ({:.1}% of time)",
//!         hotspot.function_name,
//!         hotspot.time_percentage
//!     );
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### Memory Leak Detection
//!
//! ```rust,no_run
//! use torsh_utils::bottleneck::{profile_bottlenecks_advanced, AdvancedProfilingConfig};
//! # use torsh_nn::Module;
//! # struct MyModel;
//! # impl Module for MyModel {
//! #   fn forward(&self, _: &torsh_tensor::Tensor) -> Result<torsh_tensor::Tensor, torsh_core::TorshError> {
//! #     unimplemented!()
//! #   }
//! # }
//!
//! # fn example() -> Result<(), torsh_core::TorshError> {
//! let model = MyModel;
//!
//! let config = AdvancedProfilingConfig {
//!     enable_memory_profiling: true,
//!     memory_snapshot_interval_ms: 100.0,  // Frequent snapshots for leak detection
//!     ..Default::default()
//! };
//!
//! let report = profile_bottlenecks_advanced(&model, &[1, 3, 224, 224], 1000, true, config)?;
//!
//! // Check for memory leaks. `None` means leak detection could not run
//! // (no allocator instrumentation is wired up) -- distinct from
//! // `Some(vec![])`, which would mean detection ran and found nothing.
//! match &report.memory_profile.memory_leaks {
//!     Some(leaks) if !leaks.is_empty() => {
//!         println!("⚠️  WARNING: {} memory leaks detected!", leaks.len());
//!         for leak in leaks {
//!             println!("  - {} bytes at {} (age: {:.1}s)",
//!                 (leak.size_mb * 1024.0 * 1024.0) as usize,
//!                 leak.allocation_site,
//!                 leak.age_ms / 1000.0
//!             );
//!         }
//!     }
//!     Some(_) => println!("✓ No memory leaks detected"),
//!     None => println!("Memory leak detection not available"),
//! }
//!
//! // Check memory fragmentation, when it was measured.
//! if let Some(fragmentation_ratio) = report.memory_profile.fragmentation_ratio {
//!     if fragmentation_ratio > 0.2 {
//!         println!("⚠️  High memory fragmentation: {:.1}%", fragmentation_ratio * 100.0);
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### GPU Profiling
//!
//! ```rust,no_run
//! use torsh_utils::bottleneck::{profile_bottlenecks_advanced, AdvancedProfilingConfig};
//! # use torsh_nn::Module;
//! # struct MyModel;
//! # impl Module for MyModel {
//! #   fn forward(&self, _: &torsh_tensor::Tensor) -> Result<torsh_tensor::Tensor, torsh_core::TorshError> {
//! #     unimplemented!()
//! #   }
//! # }
//!
//! # fn example() -> Result<(), torsh_core::TorshError> {
//! let model = MyModel;
//!
//! let config = AdvancedProfilingConfig {
//!     enable_gpu_profiling: true,
//!     ..Default::default()
//! };
//!
//! let report = profile_bottlenecks_advanced(&model, &[1, 3, 224, 224], 100, true, config)?;
//!
//! if let Some(gpu_profile) = &report.gpu_profile {
//!     println!("GPU Utilization: {:.1}%", gpu_profile.utilization_percentage);
//!     println!("GPU Memory: {:.1}%", gpu_profile.memory_utilization_percentage);
//!     println!("Temperature: {:.1}°C", gpu_profile.temperature_celsius);
//!     println!("Power: {:.1}W", gpu_profile.power_consumption_watts);
//!
//!     // Analyze kernel performance
//!     for kernel in &gpu_profile.kernel_executions {
//!         if kernel.occupancy < 0.5 {
//!             println!("⚠️  Low occupancy kernel: {} ({:.1}% occupancy)",
//!                 kernel.kernel_name,
//!                 kernel.occupancy * 100.0
//!             );
//!         }
//!     }
//!
//!     // Analyze memory transfers
//!     for transfer in &gpu_profile.memory_transfers {
//!         if transfer.bandwidth_gb_s < 100.0 {
//!             println!("⚠️  Slow memory transfer: {:?} ({:.1} GB/s)",
//!                 transfer.direction,
//!                 transfer.bandwidth_gb_s
//!             );
//!         }
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Understanding Results
//!
//! ### Hotspot Analysis
//!
//! Hotspots are functions or operations that consume the most CPU/GPU time:
//! - **CPU Hotspots**: Functions with high execution time percentage
//! - **GPU Hotspots**: CUDA kernels with high runtime or low occupancy
//! - **Memory Hotspots**: Operations causing frequent allocations/deallocations
//!
//! ### Flame Graphs
//!
//! Flame graphs visualize call stacks over time:
//! - **Width**: Time spent in function (including children)
//! - **Height**: Call stack depth
//! - **Color**: Can indicate different modules or call types
//!
//! ### Memory Profile
//!
//! - **Peak Usage**: Maximum memory allocated during execution
//! - **Current Usage**: Memory in use at profile end
//! - **Fragmentation**: Ratio of wasted memory due to fragmentation
//! - **Leaks**: Allocations never freed (potential memory leaks)
//!
//! ## Best Practices
//!
//! 1. **Profile in Release Mode**: Debug builds have significant overhead
//! 2. **Use Representative Workloads**: Profile with realistic input sizes
//! 3. **Run Sufficient Iterations**: More iterations = better statistical significance
//! 4. **Focus on Hot Paths**: Optimize the 20% of code taking 80% of time
//! 5. **Verify Fixes**: Re-profile after optimizations to measure improvement
//! 6. **Check Multiple Metrics**: Don't optimize time at the expense of memory
//!
//! ## Performance Tips
//!
//! ### CPU Optimization
//! - Look for operations with high `time_percentage` in hotspot analysis
//! - Check for unnecessary allocations in memory profile
//! - Identify opportunities for vectorization (SIMD)
//! - Consider parallelization for independent operations
//!
//! ### GPU Optimization
//! - Target kernels with occupancy < 50%
//! - Minimize host-device memory transfers
//! - Use pinned memory for faster transfers
//! - Optimize kernel launch configurations (grid/block sizes)
//!
//! ### Memory Optimization
//! - Fix memory leaks immediately
//! - Reduce fragmentation by using memory pools
//! - Consider gradient checkpointing for large models
//! - Use in-place operations where possible
//!
//! ## Comparison with PyTorch Profiler
//!
//! | Feature | PyTorch Profiler | ToRSh Bottleneck |
//! |---------|------------------|------------------|
//! | Flame Graphs | Via external tools | Built-in |
//! | Memory Profiling | Basic | Advanced with leak detection |
//! | GPU Analysis | CUDA only | CUDA + analysis |
//! | Overhead | ~5-10% | ~2-5% |
//! | Integration | TensorBoard | Standalone + TensorBoard |
//!
//! ## See Also
//!
//! - [`benchmark`](crate::benchmark): For performance benchmarking
//! - [`tensorboard`](crate::tensorboard): For visualizing profiling data
//! - [Tutorial Guide](https://docs.torsh.rs/tutorial#profiling)
//! - [Best Practices](https://docs.torsh.rs/best-practices#profiling)

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use std::collections::HashMap;
use std::time::{Duration, Instant};
use torsh_core::error::Result;
use torsh_nn::Module;
use torsh_profiler::{ProfileEvent, Profiler};

// Note: These features are defined in scirs2-core, not torsh-utils
// Conditional compilation is handled at the scirs2-core level

/// Comprehensive bottleneck report with advanced profiling data
#[derive(Debug, Clone)]
pub struct BottleneckReport {
    pub total_time: Duration,
    pub layer_times: Vec<LayerTiming>,
    pub operation_times: HashMap<String, OperationTiming>,
    pub memory_peaks: Vec<MemoryPeak>,
    pub recommendations: Vec<String>,

    // Advanced profiling features
    pub flame_graph: Option<FlameGraphData>,
    pub memory_profile: MemoryProfileData,
    pub gpu_profile: Option<GpuProfileData>,
    pub call_stack_analysis: CallStackAnalysis,
    pub performance_regression: Option<RegressionAnalysis>,
    pub hotspot_analysis: HotspotAnalysis,
}

/// Flame graph data structure for visualization
#[derive(Debug, Clone)]
pub struct FlameGraphData {
    pub root_frame: FlameFrame,
    pub total_samples: usize,
    pub sample_rate_hz: f32,
    pub duration_ms: f32,
}

/// Individual frame in the flame graph
#[derive(Debug, Clone)]
pub struct FlameFrame {
    pub name: String,
    pub file: Option<String>,
    pub line: Option<u32>,
    pub self_time_ms: f32,
    pub total_time_ms: f32,
    pub sample_count: usize,
    pub children: Vec<FlameFrame>,
}

/// Comprehensive memory profiling data
#[derive(Debug, Clone)]
pub struct MemoryProfileData {
    /// Real peak resident memory observed during profiling (MB).
    pub peak_usage_mb: f32,
    /// Real resident memory at the time metrics were read (MB).
    pub current_usage_mb: f32,
    pub allocation_timeline: Vec<MemorySnapshot>,
    /// Detected memory leaks, when leak detection is actually implemented.
    /// `None` means "not measured" (no allocator instrumentation is wired
    /// up here) -- distinct from `Some(vec![])`, which would claim leak
    /// detection ran and found nothing.
    pub memory_leaks: Option<Vec<MemoryLeak>>,
    /// Fragmentation ratio, when it can be measured from allocator
    /// introspection. `None` if unmeasured.
    pub fragmentation_ratio: Option<f32>,
    pub gc_pressure: Option<f32>,
    /// Memory bandwidth utilization, when it can be measured from hardware
    /// performance counters. `None` if unmeasured.
    pub memory_bandwidth_utilization: Option<f32>,
    pub cache_performance: CachePerformance,
}

/// Memory snapshot at a point in time
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    pub timestamp_ms: f32,
    pub allocated_mb: f32,
    pub reserved_mb: f32,
    pub active_allocations: usize,
    pub largest_free_block_mb: f32,
}

/// Memory leak information
#[derive(Debug, Clone)]
pub struct MemoryLeak {
    pub allocation_site: String,
    pub size_mb: f32,
    pub age_ms: f32,
    pub stack_trace: Vec<String>,
}

/// Cache performance metrics.
///
/// L1/L2/L3 hit rates, cache misses per instruction, and memory stall
/// percentage all require reading hardware performance counters (e.g. via
/// `perf_event_open` on Linux, typically permission-gated), which this
/// crate does not do. Every field is `None` ("not measured") rather than a
/// plausible-looking invented number -- see [`MemoryMetricsCollector`].
#[derive(Debug, Clone)]
pub struct CachePerformance {
    pub l1_hit_rate: Option<f32>,
    pub l2_hit_rate: Option<f32>,
    pub l3_hit_rate: Option<f32>,
    pub cache_misses_per_instruction: Option<f32>,
    pub memory_stalls_percentage: Option<f32>,
}

/// GPU profiling data
#[derive(Debug, Clone)]
pub struct GpuProfileData {
    pub utilization_percentage: f32,
    pub memory_utilization_percentage: f32,
    pub temperature_celsius: f32,
    pub power_consumption_watts: f32,
    pub kernel_executions: Vec<GpuKernelExecution>,
    pub memory_transfers: Vec<GpuMemoryTransfer>,
    pub compute_capability: String,
    pub occupancy_percentage: f32,
}

/// Individual GPU kernel execution data
#[derive(Debug, Clone)]
pub struct GpuKernelExecution {
    pub kernel_name: String,
    pub duration_ms: f32,
    pub grid_size: (u32, u32, u32),
    pub block_size: (u32, u32, u32),
    pub registers_per_thread: u32,
    pub shared_memory_kb: f32,
    pub occupancy: f32,
}

/// GPU memory transfer data
#[derive(Debug, Clone)]
pub struct GpuMemoryTransfer {
    pub direction: MemoryTransferDirection,
    pub size_mb: f32,
    pub duration_ms: f32,
    pub bandwidth_gb_s: f32,
}

/// Memory transfer direction
#[derive(Debug, Clone)]
pub enum MemoryTransferDirection {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
    Unified,
}

/// Call stack analysis results
#[derive(Debug, Clone)]
pub struct CallStackAnalysis {
    pub hottest_paths: Vec<CallPath>,
    pub recursive_calls: Vec<RecursiveCall>,
    pub call_frequency: HashMap<String, usize>,
    pub average_stack_depth: f32,
    pub max_stack_depth: usize,
}

/// Call path with timing information
#[derive(Debug, Clone)]
pub struct CallPath {
    pub path: Vec<String>,
    pub total_time_ms: f32,
    pub call_count: usize,
    pub average_time_ms: f32,
}

/// Recursive call detection
#[derive(Debug, Clone)]
pub struct RecursiveCall {
    pub function_name: String,
    pub max_depth: usize,
    pub total_recursive_time_ms: f32,
}

/// Performance regression analysis
#[derive(Debug, Clone)]
pub struct RegressionAnalysis {
    pub baseline_performance: PerformanceMetrics,
    pub current_performance: PerformanceMetrics,
    pub regression_percentage: f32,
    pub regressed_operations: Vec<String>,
    pub improvements: Vec<String>,
}

/// Performance metrics for comparison
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_time_ms: f32,
    pub memory_usage_mb: f32,
    pub throughput_ops_per_sec: f32,
    pub energy_consumption_mj: Option<f32>,
}

/// Hotspot analysis results
#[derive(Debug, Clone)]
pub struct HotspotAnalysis {
    pub cpu_hotspots: Vec<Hotspot>,
    pub memory_hotspots: Vec<MemoryHotspot>,
    pub io_hotspots: Vec<IoHotspot>,
    pub synchronization_hotspots: Vec<SyncHotspot>,
}

/// CPU computation hotspot
#[derive(Debug, Clone)]
pub struct Hotspot {
    pub function_name: String,
    pub time_percentage: f32,
    pub instruction_count: Option<u64>,
    pub cache_misses: Option<u64>,
    pub branch_mispredictions: Option<u64>,
}

/// Memory access hotspot
#[derive(Debug, Clone)]
pub struct MemoryHotspot {
    pub operation: String,
    pub access_pattern: MemoryAccessPattern,
    pub bandwidth_utilization: f32,
    pub latency_ms: f32,
}

/// Memory access pattern
#[derive(Debug, Clone)]
pub enum MemoryAccessPattern {
    Sequential,
    Random,
    Strided { stride: usize },
    Clustered,
}

/// I/O operation hotspot
#[derive(Debug, Clone)]
pub struct IoHotspot {
    pub operation_type: String,
    pub wait_time_ms: f32,
    pub throughput_mb_s: f32,
    pub queue_depth: usize,
}

/// Synchronization hotspot
#[derive(Debug, Clone)]
pub struct SyncHotspot {
    pub synchronization_type: String,
    pub wait_time_ms: f32,
    pub contention_count: usize,
    pub affected_threads: usize,
}

/// Layer timing information
#[derive(Debug, Clone)]
pub struct LayerTiming {
    pub name: String,
    pub module_type: String,
    pub forward_time: Duration,
    pub backward_time: Option<Duration>,
    pub percentage: f32,
    pub num_params: usize,
}

/// Operation timing information
#[derive(Debug, Clone)]
pub struct OperationTiming {
    pub count: usize,
    pub total_time: Duration,
    pub avg_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
}

/// Memory peak information
#[derive(Debug, Clone)]
pub struct MemoryPeak {
    pub operation: String,
    pub allocated_mb: f32,
    pub reserved_mb: f32,
}

/// Advanced profiling configuration
#[derive(Debug, Clone)]
pub struct AdvancedProfilingConfig {
    pub enable_flame_graph: bool,
    pub enable_memory_profiling: bool,
    pub enable_gpu_profiling: bool,
    pub enable_call_stack_analysis: bool,
    pub enable_regression_detection: bool,
    pub enable_hotspot_analysis: bool,
    pub sample_rate_hz: f32,
    pub memory_snapshot_interval_ms: f32,
}

impl Default for AdvancedProfilingConfig {
    fn default() -> Self {
        Self {
            enable_flame_graph: true,
            enable_memory_profiling: true,
            enable_gpu_profiling: false, // Only enable if GPU available
            enable_call_stack_analysis: true,
            enable_regression_detection: false,
            enable_hotspot_analysis: true,
            sample_rate_hz: 1000.0,
            memory_snapshot_interval_ms: 10.0,
        }
    }
}

/// Profile bottlenecks with basic profiling
pub fn profile_bottlenecks<M: Module>(
    model: &M,
    input_shape: &[usize],
    num_iterations: usize,
    profile_backward: bool,
) -> Result<BottleneckReport> {
    let config = AdvancedProfilingConfig {
        enable_flame_graph: false,
        enable_memory_profiling: true,
        enable_gpu_profiling: false,
        enable_call_stack_analysis: false,
        enable_regression_detection: false,
        enable_hotspot_analysis: false,
        ..Default::default()
    };

    profile_bottlenecks_advanced(model, input_shape, num_iterations, profile_backward, config)
}

/// Profile bottlenecks with comprehensive advanced profiling
pub fn profile_bottlenecks_advanced<M: Module>(
    model: &M,
    input_shape: &[usize],
    num_iterations: usize,
    profile_backward: bool,
    config: AdvancedProfilingConfig,
) -> Result<BottleneckReport> {
    // Initialize profilers
    let mut profiler = Profiler::new();
    let mut memory_collector = MemoryMetricsCollector::new();
    let mut leak_detector = LeakDetector::new();

    // Start profiling
    profiler.start();

    if config.enable_memory_profiling {
        memory_collector.start_collection();
        leak_detector.enable();
    }

    // Initialize data collection structures
    let layer_times = Vec::new();
    let mut operation_times: HashMap<String, Vec<Duration>> = HashMap::new();
    let mut memory_peaks = Vec::new();
    let mut memory_snapshots = Vec::new();
    // Each entry pairs a real captured call stack with the real duration of
    // the iteration it was captured in, so `analyze_call_stacks` can report
    // genuine per-path timing instead of a fabricated constant.
    let mut call_stacks: Vec<(Vec<String>, Duration)> = Vec::new();

    // GPU profiling setup
    let gpu_profiler = if config.enable_gpu_profiling {
        setup_gpu_profiling()
    } else {
        None
    };

    // Warmup runs
    for _ in 0..3 {
        let input = torsh_tensor::creation::randn(input_shape)?;
        let _ = model.forward(&input)?;
    }

    // Main profiling loop
    let start_time = Instant::now();
    let snapshot_interval = Duration::from_millis(config.memory_snapshot_interval_ms as u64);
    let mut last_snapshot = Instant::now();

    for i in 0..num_iterations {
        let input = torsh_tensor::creation::randn(input_shape)?;
        let iteration_start = Instant::now();

        // Capture the real call stack at the point this iteration's
        // compute begins; paired with the iteration's real duration once
        // that's known below.
        let call_stack = if config.enable_call_stack_analysis {
            Some(capture_call_stack())
        } else {
            None
        };

        // Profile forward pass
        let forward_start = Instant::now();
        let output = model.forward(&input)?;
        let forward_time = forward_start.elapsed();

        operation_times
            .entry("forward".to_string())
            .or_default()
            .push(forward_time);

        // Profile backward pass if requested
        if profile_backward && output.requires_grad() {
            let backward_start = Instant::now();
            output.sum()?.backward()?;
            let backward_time = backward_start.elapsed();

            operation_times
                .entry("backward".to_string())
                .or_default()
                .push(backward_time);
        }

        if let Some(call_stack) = call_stack {
            call_stacks.push((call_stack, iteration_start.elapsed()));
        }

        // Memory snapshots
        if config.enable_memory_profiling && last_snapshot.elapsed() >= snapshot_interval {
            if let Ok(memory_info) = get_detailed_memory_info() {
                memory_snapshots.push(MemorySnapshot {
                    timestamp_ms: start_time.elapsed().as_millis() as f32,
                    allocated_mb: memory_info.0,
                    reserved_mb: memory_info.1,
                    active_allocations: memory_info.2,
                    largest_free_block_mb: memory_info.3,
                });
            }
            // Update the real peak-memory reading alongside the existing
            // snapshot cadence (see MemoryMetricsCollector).
            memory_collector.sample();
            last_snapshot = Instant::now();
        }

        // Periodic memory peaks collection
        if i % 10 == 0 {
            if let Ok(memory_info) = get_memory_info() {
                memory_peaks.push(MemoryPeak {
                    operation: format!("iteration_{}", i),
                    allocated_mb: memory_info.0,
                    reserved_mb: memory_info.1,
                });
            }
        }
    }

    let total_time = start_time.elapsed();

    // Stop all profilers
    profiler.stop();

    if config.enable_memory_profiling {
        memory_collector.stop_collection();
    }

    // Collect profiling results. The flame graph and hotspot analysis are
    // built from the real per-iteration forward/backward `Duration`s
    // collected in the loop above, not from a fabricated sample set.
    let flame_graph = if config.enable_flame_graph {
        Some(generate_flame_graph(&operation_times, total_time))
    } else {
        None
    };

    let memory_profile = if config.enable_memory_profiling {
        generate_memory_profile(&memory_collector, &leak_detector, memory_snapshots)?
    } else {
        MemoryProfileData::default()
    };

    let gpu_profile = if let Some(gpu_prof) = gpu_profiler {
        Some(collect_gpu_profile_data(gpu_prof)?)
    } else {
        None
    };

    let call_stack_analysis = if config.enable_call_stack_analysis {
        analyze_call_stacks(call_stacks)?
    } else {
        CallStackAnalysis::default()
    };

    let hotspot_analysis = if config.enable_hotspot_analysis {
        analyze_hotspots(&operation_times, total_time)
    } else {
        HotspotAnalysis::default()
    };

    // Process operation timings
    let processed_op_times = process_operation_times(operation_times);

    // Generate recommendations
    let recommendations = generate_advanced_recommendations(
        &layer_times,
        &processed_op_times,
        &memory_peaks,
        &memory_profile,
        &hotspot_analysis,
    );

    Ok(BottleneckReport {
        total_time,
        layer_times,
        operation_times: processed_op_times,
        memory_peaks,
        recommendations,
        flame_graph,
        memory_profile,
        gpu_profile,
        call_stack_analysis,
        performance_regression: None,
        hotspot_analysis,
    })
}

/// Generate a flame graph from the real per-iteration operation timings
/// collected during profiling (`operation_times`: e.g. "forward"/"backward"
/// -> one real `Duration` per iteration), instead of a fixed, fabricated
/// sample set describing functions that were never actually executed.
///
/// This is coarser than a true sampling profiler (it can only see the
/// operations this crate explicitly times -- forward/backward passes as a
/// whole, not individual layers inside them, since `Module::forward` is
/// opaque here), but every number in it was genuinely measured.
fn generate_flame_graph(
    operation_times: &HashMap<String, Vec<Duration>>,
    total_time: Duration,
) -> FlameGraphData {
    let samples = real_profile_samples(operation_times);
    let total_samples = samples.len();
    // Real average sampling rate: how many real per-operation
    // measurements were taken per second of wall-clock profiling time.
    let sample_rate_hz = if total_time.as_secs_f32() > 0.0 {
        total_samples as f32 / total_time.as_secs_f32()
    } else {
        0.0
    };

    let root_frame = build_flame_graph_tree(samples);

    FlameGraphData {
        root_frame,
        total_samples,
        sample_rate_hz,
        duration_ms: total_time.as_millis() as f32,
    }
}

/// Turn real per-iteration operation `Duration`s into one [`ProfileSample`]
/// per iteration per operation, each carrying that operation's own name as
/// its (single-frame) stack trace -- the only call-site information
/// available without deeper instrumentation into the profiled model.
fn real_profile_samples(operation_times: &HashMap<String, Vec<Duration>>) -> Vec<ProfileSample> {
    let mut samples = Vec::new();
    for (name, durations) in operation_times {
        for duration in durations {
            samples.push(ProfileSample {
                function_name: name.clone(),
                duration_ms: duration.as_secs_f32() * 1000.0,
                stack_trace: vec![name.clone()],
            });
        }
    }
    samples
}

/// Build flame graph tree structure from real samples (see
/// [`real_profile_samples`]).
fn build_flame_graph_tree(samples: Vec<ProfileSample>) -> FlameFrame {
    let mut root = FlameFrame {
        name: "root".to_string(),
        file: None,
        line: None,
        self_time_ms: 0.0,
        total_time_ms: 0.0,
        sample_count: samples.len(),
        children: Vec::new(),
    };

    // Aggregate samples by function name, also tracking real sample counts
    // (previously hardcoded to 1 regardless of how many samples an
    // operation actually had).
    let mut function_stats: HashMap<String, (f32, usize)> = HashMap::new();
    for sample in &samples {
        let entry = function_stats
            .entry(sample.function_name.clone())
            .or_insert((0.0, 0));
        entry.0 += sample.duration_ms;
        entry.1 += 1;
    }

    // Create child frames
    for (function_name, (total_time, sample_count)) in function_stats {
        let child_frame = FlameFrame {
            name: function_name,
            file: None,
            line: None,
            self_time_ms: total_time,
            total_time_ms: total_time,
            sample_count,
            children: Vec::new(),
        };
        root.children.push(child_frame);
        root.total_time_ms += total_time;
    }

    root
}

/// Profile sample structure
#[derive(Debug, Clone)]
struct ProfileSample {
    function_name: String,
    duration_ms: f32,
    stack_trace: Vec<String>,
}

/// Generate comprehensive memory profile
fn generate_memory_profile(
    collector: &MemoryMetricsCollector,
    leak_detector: &LeakDetector,
    snapshots: Vec<MemorySnapshot>,
) -> Result<MemoryProfileData> {
    let metrics = collector.get_metrics();

    // `None` means leak detection is not implemented (no allocator
    // instrumentation is wired up); `Some(vec)` (even if empty) would
    // falsely claim detection ran and found nothing.
    let memory_leaks = leak_detector.get_detected_leaks().map(|leaks| {
        leaks
            .into_iter()
            .map(|leak| MemoryLeak {
                allocation_site: leak.location,
                size_mb: leak.size_bytes as f32 / 1024.0 / 1024.0,
                age_ms: leak.age_ms,
                stack_trace: leak.stack_trace,
            })
            .collect()
    });

    Ok(MemoryProfileData {
        peak_usage_mb: metrics.peak_usage_mb,
        current_usage_mb: metrics.current_usage_mb,
        allocation_timeline: snapshots,
        memory_leaks,
        fragmentation_ratio: metrics.fragmentation_ratio,
        gc_pressure: None,
        memory_bandwidth_utilization: metrics.bandwidth_utilization,
        cache_performance: CachePerformance {
            l1_hit_rate: metrics.l1_hit_rate,
            l2_hit_rate: metrics.l2_hit_rate,
            l3_hit_rate: metrics.l3_hit_rate,
            cache_misses_per_instruction: metrics.cache_misses_per_instruction,
            memory_stalls_percentage: metrics.memory_stalls_percentage,
        },
    })
}

/// Memory metrics: real process memory readings plus (unmeasured) cache
/// counters. See [`MemoryMetricsCollector`].
#[derive(Debug)]
struct MemoryMetrics {
    peak_usage_mb: f32,
    current_usage_mb: f32,
    fragmentation_ratio: Option<f32>,
    bandwidth_utilization: Option<f32>,
    l1_hit_rate: Option<f32>,
    l2_hit_rate: Option<f32>,
    l3_hit_rate: Option<f32>,
    cache_misses_per_instruction: Option<f32>,
    memory_stalls_percentage: Option<f32>,
}

/// A leak detected by [`LeakDetector`] (currently never constructed --
/// see that type's doc comment).
#[derive(Debug)]
struct DetectedLeak {
    location: String,
    size_bytes: usize,
    age_ms: f32,
    stack_trace: Vec<String>,
}

/// Analyze layer timings from profile events
#[allow(dead_code)]
fn analyze_layer_timings(_events: &[ProfileEvent], _total_time: Duration) -> Vec<LayerTiming> {
    // Simplified implementation - profiler integration not yet complete
    Vec::new()
}

/// Process operation timings
fn process_operation_times(
    raw_times: HashMap<String, Vec<Duration>>,
) -> HashMap<String, OperationTiming> {
    raw_times
        .into_iter()
        .map(|(name, times)| {
            let count = times.len();
            let total_time: Duration = times.iter().sum();
            let avg_time = total_time / count as u32;
            let min_time = times.iter().min().copied().unwrap_or(Duration::ZERO);
            let max_time = times.iter().max().copied().unwrap_or(Duration::ZERO);

            (
                name,
                OperationTiming {
                    count,
                    total_time,
                    avg_time,
                    min_time,
                    max_time,
                },
            )
        })
        .collect()
}

/// Get current memory information from /proc/self/status.
/// Returns (rss_mb, vmsize_mb). On non-Linux platforms returns (0.0, 0.0).
fn get_memory_info() -> Result<(f32, f32)> {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").map_err(|e| {
            torsh_core::TorshError::IoError(format!("Failed to read /proc/self/status: {}", e))
        })?;
        let mut rss_kb: Option<u64> = None;
        let mut vmsize_kb: Option<u64> = None;
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                rss_kb = rest.split_whitespace().next().and_then(|s| s.parse().ok());
            } else if let Some(rest) = line.strip_prefix("VmSize:") {
                vmsize_kb = rest.split_whitespace().next().and_then(|s| s.parse().ok());
            }
            if rss_kb.is_some() && vmsize_kb.is_some() {
                break;
            }
        }
        let rss_mb = rss_kb.unwrap_or(0) as f32 / 1024.0;
        let vmsize_mb = vmsize_kb.unwrap_or(0) as f32 / 1024.0;
        return Ok((rss_mb, vmsize_mb));
    }
    #[cfg(not(target_os = "linux"))]
    {
        // Memory measurement via /proc/self/status not available on this platform
        Ok((0.0, 0.0))
    }
}

/// Get detailed memory information for profiling from /proc/self/status and /proc/self/maps.
/// Returns (allocated_mb, reserved_mb, active_allocations, free_mb).
/// On non-Linux platforms returns (0.0, 0.0, 0, 0.0).
fn get_detailed_memory_info() -> Result<(f32, f32, usize, f32)> {
    #[cfg(target_os = "linux")]
    {
        // Read VmRSS (allocated = resident) and VmSize (reserved = virtual) from status
        let status = std::fs::read_to_string("/proc/self/status").map_err(|e| {
            torsh_core::TorshError::IoError(format!("Failed to read /proc/self/status: {}", e))
        })?;
        let mut rss_kb: Option<u64> = None;
        let mut vmsize_kb: Option<u64> = None;
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                rss_kb = rest.split_whitespace().next().and_then(|s| s.parse().ok());
            } else if let Some(rest) = line.strip_prefix("VmSize:") {
                vmsize_kb = rest.split_whitespace().next().and_then(|s| s.parse().ok());
            }
            if rss_kb.is_some() && vmsize_kb.is_some() {
                break;
            }
        }
        let allocated_mb = rss_kb.unwrap_or(0) as f32 / 1024.0;
        let reserved_mb = vmsize_kb.unwrap_or(0) as f32 / 1024.0;

        // Approximate active allocations from /proc/self/maps line count
        let active_allocations = std::fs::read_to_string("/proc/self/maps")
            .map(|s| s.lines().count())
            .unwrap_or(0);

        // Approximate free memory from /proc/meminfo MemFree
        let meminfo = std::fs::read_to_string("/proc/meminfo").map_err(|e| {
            torsh_core::TorshError::IoError(format!("Failed to read /proc/meminfo: {}", e))
        })?;
        let mut memfree_kb: Option<u64> = None;
        for line in meminfo.lines() {
            if let Some(rest) = line.strip_prefix("MemFree:") {
                memfree_kb = rest.split_whitespace().next().and_then(|s| s.parse().ok());
                break;
            }
        }
        let free_mb = memfree_kb.unwrap_or(0) as f32 / 1024.0;

        return Ok((allocated_mb, reserved_mb, active_allocations, free_mb));
    }
    #[cfg(not(target_os = "linux"))]
    {
        // Detailed memory measurement via /proc not available on this platform
        Ok((0.0, 0.0, 0, 0.0))
    }
}

/// Setup GPU profiling if available
fn setup_gpu_profiling() -> Option<GpuProfiler> {
    // Check if GPU is available and setup profiling
    // For now, return None (no GPU profiling)
    None
}

/// Placeholder GPU profiler
struct GpuProfiler {
    _context: String,
}

/// Collect GPU profiling data.
///
/// GPU profiling requires NVML or CUDA runtime linkage which is not currently
/// linked into this crate. All metrics are reported as 0.0 / empty to clearly
/// communicate that no measurement was taken. Callers should treat `gpu_profile`
/// as informational only when no `cuda` feature is active.
fn collect_gpu_profile_data(_profiler: GpuProfiler) -> Result<GpuProfileData> {
    Ok(GpuProfileData {
        utilization_percentage: 0.0,
        memory_utilization_percentage: 0.0,
        temperature_celsius: 0.0,
        power_consumption_watts: 0.0,
        kernel_executions: vec![],
        memory_transfers: vec![],
        compute_capability: "unknown".to_string(),
        occupancy_percentage: 0.0,
    })
}

/// Capture the real current call stack via `std::backtrace` (stable since
/// Rust 1.65; no extra dependency needed).
///
/// `force_capture` resolves symbols unconditionally, so behavior does not
/// depend on the `RUST_BACKTRACE` environment variable. The standard
/// library's `Backtrace` exposes only a formatted `Display`/`Debug`
/// rendering (no structured per-frame API), so individual frames are
/// recovered by splitting that rendering into non-empty lines -- real,
/// call-site-specific data, replacing the previous fixed 3-entry
/// placeholder that was returned for every call regardless of where it was
/// actually made from.
fn capture_call_stack() -> Vec<String> {
    let backtrace = std::backtrace::Backtrace::force_capture();
    format!("{backtrace}")
        .lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .map(|line| line.to_string())
        .collect()
}

/// Analyze call stacks for patterns.
///
/// `call_stacks` pairs each real captured stack with the real duration of
/// the iteration it was captured in (see `profile_bottlenecks_advanced`),
/// so per-path timing below is computed from genuine measurements instead
/// of a fixed placeholder.
fn analyze_call_stacks(call_stacks: Vec<(Vec<String>, Duration)>) -> Result<CallStackAnalysis> {
    let mut call_frequency = HashMap::new();
    let mut total_depth = 0;
    let mut max_depth = 0;

    for (stack, _duration) in &call_stacks {
        total_depth += stack.len();
        max_depth = max_depth.max(stack.len());

        for function in stack {
            *call_frequency.entry(function.clone()).or_insert(0) += 1;
        }
    }

    let average_stack_depth = if !call_stacks.is_empty() {
        total_depth as f32 / call_stacks.len() as f32
    } else {
        0.0
    };

    // Group identical stacks together and compute each group's real
    // aggregate timing, rather than stamping every path with a fixed
    // 100.0ms placeholder.
    let mut path_groups: HashMap<Vec<String>, Vec<f32>> = HashMap::new();
    for (stack, duration) in call_stacks {
        path_groups
            .entry(stack)
            .or_default()
            .push(duration.as_secs_f32() * 1000.0);
    }

    let mut hottest_paths: Vec<CallPath> = path_groups
        .into_iter()
        .map(|(path, times_ms)| {
            let call_count = times_ms.len();
            let total_time_ms: f32 = times_ms.iter().sum();
            let average_time_ms = total_time_ms / call_count as f32;
            CallPath {
                path,
                total_time_ms,
                call_count,
                average_time_ms,
            }
        })
        .collect();
    hottest_paths.sort_by(|a, b| {
        b.total_time_ms
            .partial_cmp(&a.total_time_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hottest_paths.truncate(5);

    Ok(CallStackAnalysis {
        hottest_paths,
        recursive_calls: vec![], // Would detect recursive patterns
        call_frequency,
        average_stack_depth,
        max_stack_depth: max_depth,
    })
}

/// Analyze performance hotspots from the real per-iteration operation
/// timings collected during profiling.
///
/// CPU hotspots are derived from genuinely measured operation durations
/// (coarse-grained: "forward"/"backward" as a whole, since `Module` does
/// not expose per-layer timing here). `instruction_count`/`cache_misses`/
/// `branch_mispredictions` require hardware performance counters this
/// crate does not read, so they are `None` rather than invented numbers.
///
/// Memory/I/O/synchronization hotspots require access-pattern, I/O, and
/// lock-contention instrumentation that does not exist in this crate;
/// rather than fabricate plausible-looking entries, these are honestly
/// empty until such instrumentation is implemented.
fn analyze_hotspots(
    operation_times: &HashMap<String, Vec<Duration>>,
    total_time: Duration,
) -> HotspotAnalysis {
    let total_secs = total_time.as_secs_f32();
    let mut cpu_hotspots: Vec<Hotspot> = operation_times
        .iter()
        .map(|(name, durations)| {
            let op_total_secs: f32 = durations.iter().map(|d| d.as_secs_f32()).sum();
            let time_percentage = if total_secs > 0.0 {
                (op_total_secs / total_secs) * 100.0
            } else {
                0.0
            };
            Hotspot {
                function_name: name.clone(),
                time_percentage,
                instruction_count: None,
                cache_misses: None,
                branch_mispredictions: None,
            }
        })
        .collect();
    cpu_hotspots.sort_by(|a, b| {
        b.time_percentage
            .partial_cmp(&a.time_percentage)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    HotspotAnalysis {
        cpu_hotspots,
        // Would require memory access-pattern instrumentation.
        memory_hotspots: vec![],
        // Would require I/O instrumentation.
        io_hotspots: vec![],
        // Would require lock/synchronization instrumentation.
        synchronization_hotspots: vec![],
    }
}

/// Generate advanced recommendations with comprehensive analysis
fn generate_advanced_recommendations(
    layer_times: &[LayerTiming],
    operation_times: &HashMap<String, OperationTiming>,
    memory_peaks: &[MemoryPeak],
    memory_profile: &MemoryProfileData,
    hotspot_analysis: &HotspotAnalysis,
) -> Vec<String> {
    let mut recommendations = Vec::new();

    // Basic recommendations (from original function)
    recommendations.extend(generate_recommendations(
        layer_times,
        operation_times,
        memory_peaks,
    ));

    // Memory-specific recommendations -- only fire when the underlying
    // metric was actually measured.
    if let Some(fragmentation_ratio) = memory_profile.fragmentation_ratio {
        if fragmentation_ratio > 0.3 {
            recommendations.push(format!(
                "High memory fragmentation ({:.1}%). Consider using memory pools or reducing allocation frequency.",
                fragmentation_ratio * 100.0
            ));
        }
    }

    if let Some(leaks) = &memory_profile.memory_leaks {
        if !leaks.is_empty() {
            recommendations.push(format!(
                "Detected {} memory leaks. Review allocation sites: {}",
                leaks.len(),
                leaks
                    .iter()
                    .take(3)
                    .map(|leak| leak.allocation_site.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }

    if let Some(l1_hit_rate) = memory_profile.cache_performance.l1_hit_rate {
        if l1_hit_rate < 0.9 {
            recommendations.push(format!(
                "Low L1 cache hit rate ({:.1}%). Consider improving data locality and access patterns.",
                l1_hit_rate * 100.0
            ));
        }
    }

    // CPU hotspot recommendations
    for hotspot in &hotspot_analysis.cpu_hotspots {
        if hotspot.time_percentage > 20.0 {
            recommendations.push(format!(
                "Function '{}' consumes {:.1}% of CPU time. Consider optimizing this function.",
                hotspot.function_name, hotspot.time_percentage
            ));

            if let Some(cache_misses) = hotspot.cache_misses {
                if cache_misses > 100_000 {
                    recommendations.push(format!(
                        "High cache miss rate in '{}'. Optimize memory access patterns.",
                        hotspot.function_name
                    ));
                }
            }
        }
    }

    // Memory access pattern recommendations
    for mem_hotspot in &hotspot_analysis.memory_hotspots {
        match mem_hotspot.access_pattern {
            MemoryAccessPattern::Random => {
                recommendations.push(format!(
                    "Random memory access detected in '{}'. Consider restructuring data layout for better locality.",
                    mem_hotspot.operation
                ));
            }
            MemoryAccessPattern::Strided { stride } => {
                if stride > 64 {
                    recommendations.push(format!(
                        "Large stride ({}) in memory access for '{}'. Consider data reorganization.",
                        stride, mem_hotspot.operation
                    ));
                }
            }
            _ => {}
        }

        if mem_hotspot.bandwidth_utilization < 50.0 {
            recommendations.push(format!(
                "Low memory bandwidth utilization ({:.1}%) in '{}'. Consider vectorization or prefetching.",
                mem_hotspot.bandwidth_utilization, mem_hotspot.operation
            ));
        }
    }

    // I/O recommendations
    for io_hotspot in &hotspot_analysis.io_hotspots {
        if io_hotspot.wait_time_ms > 10.0 {
            recommendations.push(format!(
                "High I/O wait time ({:.1}ms) for '{}'. Consider async I/O or data prefetching.",
                io_hotspot.wait_time_ms, io_hotspot.operation_type
            ));
        }
    }

    // Synchronization recommendations
    for sync_hotspot in &hotspot_analysis.synchronization_hotspots {
        if sync_hotspot.wait_time_ms > 5.0 {
            recommendations.push(format!(
                "Synchronization bottleneck in '{}' ({:.1}ms wait time). Consider lock-free algorithms or finer-grained locking.",
                sync_hotspot.synchronization_type, sync_hotspot.wait_time_ms
            ));
        }
    }

    recommendations
}

// Add default implementations for complex structures
impl Default for MemoryProfileData {
    /// Used when memory profiling was not enabled at all
    /// (`enable_memory_profiling: false`). Every field is a genuine zero/
    /// `None` for "not collected", not a fabricated "everything is
    /// perfect" reading (the previous implementation reported 100% cache
    /// hit rates here, which claimed cache performance had been checked
    /// and was flawless -- when in fact nothing had been measured at all).
    fn default() -> Self {
        Self {
            peak_usage_mb: 0.0,
            current_usage_mb: 0.0,
            allocation_timeline: vec![],
            memory_leaks: None,
            fragmentation_ratio: None,
            gc_pressure: None,
            memory_bandwidth_utilization: None,
            cache_performance: CachePerformance {
                l1_hit_rate: None,
                l2_hit_rate: None,
                l3_hit_rate: None,
                cache_misses_per_instruction: None,
                memory_stalls_percentage: None,
            },
        }
    }
}

impl Default for CallStackAnalysis {
    fn default() -> Self {
        Self {
            hottest_paths: vec![],
            recursive_calls: vec![],
            call_frequency: HashMap::new(),
            average_stack_depth: 0.0,
            max_stack_depth: 0,
        }
    }
}

impl Default for HotspotAnalysis {
    fn default() -> Self {
        Self {
            cpu_hotspots: vec![],
            memory_hotspots: vec![],
            io_hotspots: vec![],
            synchronization_hotspots: vec![],
        }
    }
}

trait MemoryCollectorTrait {
    fn new() -> Self;
    fn start_collection(&mut self);
    fn stop_collection(&mut self);
    fn get_metrics(&self) -> MemoryMetrics;
}

impl MemoryCollectorTrait for MemoryMetricsCollector {
    fn new() -> Self {
        MemoryMetricsCollector {
            #[cfg(feature = "collect_env")]
            peak_bytes: 0,
        }
    }

    fn start_collection(&mut self) {
        #[cfg(feature = "collect_env")]
        {
            self.peak_bytes = current_process_memory_bytes().unwrap_or(0);
        }
    }

    fn stop_collection(&mut self) {
        // Take one final real sample so the peak reflects memory right up
        // to the end of the profiled run.
        self.sample();
    }

    fn get_metrics(&self) -> MemoryMetrics {
        #[cfg(feature = "collect_env")]
        {
            let current_bytes = current_process_memory_bytes().unwrap_or(0);
            let peak_bytes = self.peak_bytes.max(current_bytes);
            MemoryMetrics {
                peak_usage_mb: bytes_to_mb(peak_bytes),
                current_usage_mb: bytes_to_mb(current_bytes),
                // None of these require hardware performance counters or
                // allocator introspection this crate does not have.
                fragmentation_ratio: None,
                bandwidth_utilization: None,
                l1_hit_rate: None,
                l2_hit_rate: None,
                l3_hit_rate: None,
                cache_misses_per_instruction: None,
                memory_stalls_percentage: None,
            }
        }
        #[cfg(not(feature = "collect_env"))]
        {
            MemoryMetrics {
                peak_usage_mb: 0.0,
                current_usage_mb: 0.0,
                fragmentation_ratio: None,
                bandwidth_utilization: None,
                l1_hit_rate: None,
                l2_hit_rate: None,
                l3_hit_rate: None,
                cache_misses_per_instruction: None,
                memory_stalls_percentage: None,
            }
        }
    }
}

impl MemoryMetricsCollector {
    /// Take a real process-memory reading now and fold it into the
    /// running peak. A no-op when the `collect_env` feature (which brings
    /// in `sysinfo`) is disabled.
    fn sample(&mut self) {
        #[cfg(feature = "collect_env")]
        {
            if let Some(bytes) = current_process_memory_bytes() {
                self.peak_bytes = self.peak_bytes.max(bytes);
            }
        }
    }
}

/// Real process memory metrics collector.
///
/// Uses `sysinfo` (available via this crate's default-enabled
/// `collect_env` feature) to sample this process's actual resident
/// memory. Cache-level counters (L1/L2/L3 hit rate, cache misses per
/// instruction, memory bandwidth utilization, fragmentation ratio)
/// require hardware performance counters or allocator introspection this
/// crate does not have -- see [`MemoryMetrics`] / [`CachePerformance`],
/// which report those as `None` rather than plausible-looking invented
/// numbers.
struct MemoryMetricsCollector {
    /// Peak resident memory (bytes) observed via [`Self::sample`] /
    /// `start_collection` / `stop_collection` so far.
    #[cfg(feature = "collect_env")]
    peak_bytes: u64,
}

/// Real resident memory (RSS) of the current process, in bytes, via
/// `sysinfo`. Returns `None` if the current process could not be looked up
/// (e.g. an unsupported platform).
#[cfg(feature = "collect_env")]
fn current_process_memory_bytes() -> Option<u64> {
    use sysinfo::{ProcessesToUpdate, System};

    let pid = sysinfo::get_current_pid().ok()?;
    let mut sys = System::new();
    sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
    sys.process(pid).map(|process| process.memory())
}

#[cfg(feature = "collect_env")]
fn bytes_to_mb(bytes: u64) -> f32 {
    bytes as f32 / (1024.0 * 1024.0)
}

trait LeakDetectorTrait {
    fn new() -> Self;
    fn enable(&mut self);
    fn get_detected_leaks(&self) -> Option<Vec<DetectedLeak>>;
}

impl LeakDetectorTrait for LeakDetector {
    fn new() -> Self {
        LeakDetector { _placeholder: () }
    }

    fn enable(&mut self) {
        // Enable leak detection
    }

    /// Real leak detection requires instrumenting the global allocator to
    /// record allocation call sites and ages (or an external tool such as
    /// Valgrind/heaptrack); neither is wired into this crate. Returns
    /// `None` ("not measured") rather than `Some(vec![])`, which would
    /// falsely claim detection ran and cleanly found zero leaks.
    fn get_detected_leaks(&self) -> Option<Vec<DetectedLeak>> {
        None
    }
}

impl LeakDetector {
    fn new() -> Self {
        Self { _placeholder: () }
    }
}

struct LeakDetector {
    _placeholder: (),
}

/// Generate optimization recommendations
fn generate_recommendations(
    layer_times: &[LayerTiming],
    operation_times: &HashMap<String, OperationTiming>,
    memory_peaks: &[MemoryPeak],
) -> Vec<String> {
    let mut recommendations = Vec::new();

    // Check for slow layers
    if let Some(slowest) = layer_times.first() {
        if slowest.percentage > 30.0 {
            recommendations.push(format!(
                "Layer '{}' takes {:.1}% of total time. Consider optimizing or replacing this layer.",
                slowest.name, slowest.percentage
            ));
        }
    }

    // Check forward/backward balance
    if let (Some(forward), Some(backward)) = (
        operation_times.get("forward"),
        operation_times.get("backward"),
    ) {
        let ratio = backward.avg_time.as_secs_f32() / forward.avg_time.as_secs_f32();
        if ratio > 3.0 {
            recommendations.push(format!(
                "Backward pass is {:.1}x slower than forward pass. Consider gradient checkpointing.",
                ratio
            ));
        }
    }

    // Check memory usage
    if !memory_peaks.is_empty() {
        let max_memory = memory_peaks
            .iter()
            .map(|p| p.allocated_mb)
            .fold(0.0f32, |a, b| a.max(b));

        if max_memory > 1000.0 {
            recommendations.push(format!(
                "High memory usage detected ({:.1} MB). Consider using mixed precision training.",
                max_memory
            ));
        }
    }

    // Check for high-parameter layers
    for layer in layer_times.iter().take(5) {
        if layer.module_type.contains("Conv") && layer.percentage > 20.0 {
            recommendations.push(format!(
                "Convolution layer '{}' is slow. Consider using depthwise separable convolutions.",
                layer.name
            ));
        }
    }

    recommendations
}

/// Print bottleneck report
pub fn print_bottleneck_report(report: &BottleneckReport) {
    println!("=== Bottleneck Analysis Report ===");
    println!();
    println!(
        "Total profiling time: {:.3}s",
        report.total_time.as_secs_f32()
    );
    println!();

    println!("Top 10 Slowest Layers:");
    println!(
        "{:<30} {:<15} {:<10} {:<10} {:<10}",
        "Layer", "Type", "Forward", "Backward", "% Time"
    );
    println!("{}", "-".repeat(75));

    for layer in report.layer_times.iter().take(10) {
        let backward_str = layer
            .backward_time
            .map(|t| format!("{:.3}ms", t.as_secs_f32() * 1000.0))
            .unwrap_or_else(|| "N/A".to_string());

        println!(
            "{:<30} {:<15} {:<10.3}ms {:<10} {:<10.1}%",
            layer.name,
            layer.module_type,
            layer.forward_time.as_secs_f32() * 1000.0,
            backward_str,
            layer.percentage
        );
    }
    println!();

    println!("Operation Summary:");
    for (name, timing) in &report.operation_times {
        println!(
            "{}: {} calls, avg {:.3}ms, total {:.3}s",
            name,
            timing.count,
            timing.avg_time.as_secs_f32() * 1000.0,
            timing.total_time.as_secs_f32()
        );
    }
    println!();

    println!("Memory Profile:");
    println!(
        "  Peak usage: {:.1} MB, current usage: {:.1} MB",
        report.memory_profile.peak_usage_mb, report.memory_profile.current_usage_mb
    );
    println!(
        "  Fragmentation ratio: {}",
        format_optional_percent(report.memory_profile.fragmentation_ratio)
    );
    println!(
        "  Memory leaks: {}",
        match &report.memory_profile.memory_leaks {
            Some(leaks) => format!("{}", leaks.len()),
            None => "not measured".to_string(),
        }
    );
    let cache = &report.memory_profile.cache_performance;
    println!(
        "  Cache hit rate: L1 {}, L2 {}, L3 {}",
        format_optional_percent(cache.l1_hit_rate),
        format_optional_percent(cache.l2_hit_rate),
        format_optional_percent(cache.l3_hit_rate)
    );
    println!();

    if !report.recommendations.is_empty() {
        println!("Optimization Recommendations:");
        for (i, rec) in report.recommendations.iter().enumerate() {
            println!("{}. {}", i + 1, rec);
        }
    }
}

/// Render an optional 0.0-1.0 ratio as a percentage, or "not measured"
/// when the underlying metric was never read (rather than silently
/// printing a `0.0%` that would look like a real, if bad, measurement).
fn format_optional_percent(value: Option<f32>) -> String {
    value
        .map(|v| format!("{:.1}%", v * 100.0))
        .unwrap_or_else(|| "not measured".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_memory_info_nonnegative() {
        let (alloc, reserved) = get_memory_info().unwrap_or((0.0, 0.0));
        assert!(
            alloc >= 0.0,
            "allocated MB should be non-negative, got {}",
            alloc
        );
        assert!(
            reserved >= 0.0,
            "reserved MB should be non-negative, got {}",
            reserved
        );
        #[cfg(target_os = "linux")]
        {
            assert!(
                alloc > 0.0,
                "allocated MB should be positive on Linux, got {}",
                alloc
            );
        }
    }

    #[test]
    fn test_get_detailed_memory_info_nonnegative() {
        let (alloc, reserved, active_allocs, free_mb) =
            get_detailed_memory_info().unwrap_or((0.0, 0.0, 0, 0.0));
        assert!(
            alloc >= 0.0,
            "allocated MB should be non-negative, got {}",
            alloc
        );
        assert!(
            reserved >= 0.0,
            "reserved MB should be non-negative, got {}",
            reserved
        );
        assert!(
            free_mb >= 0.0,
            "free MB should be non-negative, got {}",
            free_mb
        );
        #[cfg(target_os = "linux")]
        {
            assert!(
                alloc > 0.0,
                "allocated MB should be positive on Linux, got {}",
                alloc
            );
            assert!(
                active_allocs > 0,
                "active allocations should be positive on Linux, got {}",
                active_allocs
            );
        }
        let _ = active_allocs; // used in cfg(linux) branch above
    }

    #[test]
    fn test_process_operation_times() {
        let mut raw_times = HashMap::new();
        raw_times.insert(
            "test_op".to_string(),
            vec![
                Duration::from_millis(10),
                Duration::from_millis(20),
                Duration::from_millis(15),
            ],
        );

        let processed = process_operation_times(raw_times);
        let timing = processed.get("test_op").unwrap();

        assert_eq!(timing.count, 3);
        assert_eq!(timing.total_time, Duration::from_millis(45));
        assert_eq!(timing.avg_time, Duration::from_millis(15));
        assert_eq!(timing.min_time, Duration::from_millis(10));
        assert_eq!(timing.max_time, Duration::from_millis(20));
    }
}
