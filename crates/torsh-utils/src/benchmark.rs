//! # Model Performance Benchmarking
//!
//! This module provides comprehensive benchmarking utilities for measuring and analyzing
//! model performance, including inference speed, memory usage, and mobile-specific metrics.
//!
//! ## Features
//!
//! - **Multi-batch Benchmarking**: Test performance across different batch sizes
//! - **Memory Profiling**: Track memory allocation and usage
//! - **Backward Pass Profiling**: Measure training performance
//! - **Mobile Benchmarking**: Platform-specific performance validation
//! - **Statistical Analysis**: Mean, std dev, percentiles (p95, p99)
//! - **Throughput Metrics**: Samples per second calculation
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use torsh_utils::benchmark::{benchmark_model, BenchmarkConfig, print_benchmark_results};
//! # use torsh_nn::Module;
//!
//! # struct MyModel;
//! # impl Module for MyModel {
//! #    fn forward(&self, _input: &torsh_tensor::Tensor) -> Result<torsh_tensor::Tensor, torsh_core::TorshError> {
//! #       unimplemented!()
//! #    }
//! # }
//!
//! # fn example() -> Result<(), torsh_core::TorshError> {
//! let model = MyModel;
//!
//! // Configure benchmarking
//! let config = BenchmarkConfig {
//!     warmup_iterations: 10,
//!     benchmark_iterations: 100,
//!     batch_sizes: vec![1, 8, 16, 32],
//!     input_shapes: vec![vec![3, 224, 224]],
//!     profile_memory: true,
//!     profile_backward: true,
//!     device: torsh_core::DeviceType::Cpu,
//!     mobile_config: None,
//! };
//!
//! // Run benchmark
//! let results = benchmark_model(&model, config)?;
//!
//! // Print results
//! print_benchmark_results(&results);
//! # Ok(())
//! # }
//! ```
//!
//! ## Mobile Benchmarking
//!
//! Test model performance on mobile platforms with specific configurations:
//!
//! ```rust,ignore
//! use torsh_utils::benchmark::{BenchmarkConfig, MobileBenchmarkConfig, PlatformBenchmarkInfo, MobilePlatform};
//! # use torsh_nn::Module;
//!
//! # struct MyModel;
//! # impl Module for MyModel {
//! #    fn forward(&self, _input: &torsh_tensor::Tensor) -> Result<torsh_tensor::Tensor, torsh_core::TorshError> {
//! #       unimplemented!()
//! #    }
//! # }
//! # fn example() -> Result<(), torsh_core::TorshError> {
//! let mobile_config = MobileBenchmarkConfig {
//!     platform_info: PlatformBenchmarkInfo {
//!         platform: MobilePlatform::iOS {
//!             device: "iPhone 15 Pro".to_string(),
//!             ios_version: "17.0".to_string(),
//!         },
//!         chip: "A17 Pro".to_string(),
//!         cores: 6,
//!         gpu_cores: Some(6),
//!         ram_gb: 8.0,
//!     },
//!     monitor_thermal: true,
//!     measure_power: true,
//!     test_frequency_scaling: false,
//!     test_memory_pressure: true,
//!     stress_test_duration_minutes: Some(5),
//!     latency_thresholds: Default::default(),
//!     energy_targets: Some(Default::default()),
//! };
//!
//! let config = BenchmarkConfig {
//!     mobile_config: Some(mobile_config),
//!     ..Default::default()
//! };
//! # Ok(())
//! # }
//! ```
//!
//! ## Understanding Results
//!
//! Benchmark results include detailed statistics:
//!
//! - **Timing Statistics**: Mean, std dev, min, max, median, p95, p99
//! - **Throughput**: Samples processed per second
//! - **Memory**: Peak and average memory usage
//! - **Recommendations**: Automatic performance optimization suggestions
//!
//! ## Best Practices
//!
//! 1. **Warmup**: Always include warmup iterations to avoid cold start overhead
//! 2. **Representative Workload**: Use realistic input sizes and batch sizes
//! 3. **Multiple Runs**: Run benchmarks multiple times for statistical significance
//! 4. **Isolation**: Minimize background processes during benchmarking
//! 5. **Mobile Testing**: Test on actual target devices, not just simulators
//!
//! ## Performance Tips
//!
//! - Use larger batch sizes for higher throughput (up to memory limits)
//! - Consider batch size impact on latency vs throughput trade-off
//! - Monitor memory usage to avoid OOM errors
//! - Profile both forward and backward passes for training workloads

use std::collections::HashMap;
use std::time::{Duration, Instant};
use torsh_core::error::Result;
use torsh_nn::Module;

use crate::mobile_optimizer::{
    MobileBenchmarkResults, MobilePlatform, OptimizedModel, PlatformBenchmarkInfo, ThermalState,
};

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub warmup_iterations: usize,
    pub benchmark_iterations: usize,
    pub batch_sizes: Vec<usize>,
    pub input_shapes: Vec<Vec<usize>>,
    pub profile_memory: bool,
    pub profile_backward: bool,
    pub device: torsh_core::DeviceType,
    pub mobile_config: Option<MobileBenchmarkConfig>,
}

/// Mobile-specific benchmark configuration
#[derive(Debug, Clone)]
pub struct MobileBenchmarkConfig {
    /// Platform information for mobile benchmarking
    pub platform_info: PlatformBenchmarkInfo,
    /// Enable thermal monitoring
    pub monitor_thermal: bool,
    /// Enable power consumption measurement
    pub measure_power: bool,
    /// Test different CPU/GPU frequency settings
    pub test_frequency_scaling: bool,
    /// Test under memory pressure
    pub test_memory_pressure: bool,
    /// Stress test for sustained performance
    pub stress_test_duration_minutes: Option<u32>,
    /// Target latency thresholds for validation
    pub latency_thresholds: LatencyThresholds,
    /// Energy efficiency targets
    pub energy_targets: Option<EnergyTargets>,
}

/// Latency thresholds for mobile validation
#[derive(Debug, Clone)]
pub struct LatencyThresholds {
    /// Real-time inference threshold (ms)
    pub realtime_ms: f32,
    /// Interactive threshold (ms)
    pub interactive_ms: f32,
    /// Batch processing threshold (ms)
    pub batch_ms: f32,
}

impl Default for LatencyThresholds {
    fn default() -> Self {
        Self {
            realtime_ms: 16.67,    // 60 FPS
            interactive_ms: 100.0, // 100ms for interactive
            batch_ms: 1000.0,      // 1 second for batch
        }
    }
}

/// Energy efficiency targets
#[derive(Debug, Clone)]
pub struct EnergyTargets {
    /// Target inferences per joule
    pub inferences_per_joule: f32,
    /// Maximum power consumption (watts)
    pub max_power_watts: f32,
    /// Target battery life hours for continuous inference
    pub target_battery_hours: f32,
}

impl Default for EnergyTargets {
    fn default() -> Self {
        Self {
            inferences_per_joule: 1000.0,
            max_power_watts: 5.0,
            target_battery_hours: 8.0,
        }
    }
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            benchmark_iterations: 100,
            batch_sizes: vec![1, 8, 16, 32],
            input_shapes: vec![vec![3, 224, 224]],
            profile_memory: true,
            profile_backward: true,
            device: torsh_core::DeviceType::Cpu,
            mobile_config: None,
        }
    }
}

/// Benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub model_name: String,
    pub total_params: usize,
    pub results_by_batch: HashMap<usize, BatchResult>,
    pub summary: BenchmarkSummary,
    pub mobile_results: Option<MobileBenchmarkResults>,
    pub validation_results: Option<ValidationResults>,
}

/// Validation results for mobile deployment
#[derive(Debug, Clone)]
pub struct ValidationResults {
    /// Whether model meets real-time latency requirements
    pub meets_realtime_latency: bool,
    /// Whether model meets interactive latency requirements
    pub meets_interactive_latency: bool,
    /// Whether model meets energy efficiency targets
    pub meets_energy_targets: bool,
    /// Thermal throttling detected during testing
    pub thermal_throttling_detected: bool,
    /// Memory pressure impact on performance
    pub memory_pressure_impact: Option<f32>,
    /// Sustained performance degradation percentage
    pub sustained_performance_degradation: Option<f32>,
    /// Platform-specific validation results
    pub platform_validation: PlatformValidationResults,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}

/// Platform-specific validation results
#[derive(Debug, Clone)]
pub struct PlatformValidationResults {
    /// iOS App Store guidelines compliance
    pub ios_app_store_compliant: Option<bool>,
    /// Android performance class requirements
    pub android_performance_class: Option<String>,
    /// Device compatibility score (0-100)
    pub device_compatibility_score: f32,
    /// Estimated device support percentage
    pub device_support_percentage: f32,
}

/// Results for a specific batch size
#[derive(Debug, Clone)]
pub struct BatchResult {
    pub batch_size: usize,
    pub forward_time: TimingStats,
    pub backward_time: Option<TimingStats>,
    pub total_time: TimingStats,
    pub throughput: f32,
    pub memory_stats: Option<MemoryStats>,
}

/// Timing statistics
#[derive(Debug, Clone)]
pub struct TimingStats {
    pub mean: Duration,
    pub std: Duration,
    pub min: Duration,
    pub max: Duration,
    pub median: Duration,
    pub p95: Duration,
    pub p99: Duration,
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub peak_allocated_mb: f32,
    pub peak_reserved_mb: f32,
    pub avg_allocated_mb: f32,
}

/// Benchmark summary
#[derive(Debug, Clone)]
pub struct BenchmarkSummary {
    pub best_batch_size: usize,
    pub best_throughput: f32,
    pub optimal_memory_batch: usize,
    pub recommendations: Vec<String>,
}

/// Benchmark a model with optional mobile-specific testing
pub fn benchmark_model<M: Module>(model: &M, config: BenchmarkConfig) -> Result<BenchmarkResult> {
    let model_name = std::any::type_name::<M>()
        .split("::")
        .last()
        .unwrap_or("UnknownModel")
        .to_string();

    let total_params = count_parameters(model);
    let mut results_by_batch = HashMap::new();

    for batch_size in &config.batch_sizes {
        println!("Benchmarking batch size: {}", batch_size);

        let result = benchmark_batch_size(model, *batch_size, &config.input_shapes[0], &config)?;

        results_by_batch.insert(*batch_size, result);
    }

    let summary = generate_summary(&results_by_batch);

    // Perform mobile-specific benchmarking if configured
    let mobile_results = if let Some(mobile_config) = &config.mobile_config {
        Some(benchmark_mobile_model(model, mobile_config)?)
    } else {
        None
    };

    // Perform validation if mobile benchmarking was done. The three
    // additional mobile sub-tests run here (rather than inside
    // `benchmark_mobile_model`) so their real measured results can be
    // written directly onto the `ValidationResults` fields that report
    // them.
    let validation_results = if let Some(mobile_config) = &config.mobile_config {
        let mut validation =
            validate_mobile_performance(&results_by_batch, mobile_config, mobile_results.as_ref())?;

        // Same convention as `benchmark_batch_size`: prepend a batch size
        // of 1 to the caller's configured (batch-less) input shape.
        let mut mobile_input_shape = vec![1usize];
        mobile_input_shape.extend_from_slice(&config.input_shapes[0]);

        if mobile_config.test_memory_pressure {
            validation.memory_pressure_impact =
                run_memory_pressure_test(model, &mobile_input_shape, mobile_config)?;
        }

        if mobile_config.test_frequency_scaling {
            // Frequency-scaling control cannot be portably measured in
            // pure Rust (see `run_frequency_scaling_test`); report it as
            // skipped rather than failing the entire benchmark run over
            // one orthogonal, best-effort sub-test.
            if let Err(e) = run_frequency_scaling_test(model, mobile_config) {
                validation
                    .recommendations
                    .push(format!("Frequency scaling test skipped: {e}"));
            }
        }

        if let Some(duration) = mobile_config.stress_test_duration_minutes {
            validation.sustained_performance_degradation = run_sustained_performance_test(
                model,
                &mobile_input_shape,
                mobile_config,
                duration,
            )?;
        }

        Some(validation)
    } else {
        None
    };

    Ok(BenchmarkResult {
        model_name,
        total_params,
        results_by_batch,
        summary,
        mobile_results,
        validation_results,
    })
}

/// Benchmark model specifically for mobile deployment
pub fn benchmark_mobile_model<M: Module>(
    model: &M,
    mobile_config: &MobileBenchmarkConfig,
) -> Result<MobileBenchmarkResults> {
    use crate::mobile_optimizer::benchmark_mobile_model_advanced;

    println!("Running mobile-specific benchmark...");

    // Convert model to optimized representation for benchmarking
    // This is simplified - in practice would extract actual model structure
    let optimized_model = convert_to_optimized_model(model)?;

    // Input shapes for mobile benchmarking (typically smaller batches)
    let input_shapes = vec![vec![1, 3, 224, 224]]; // Mobile-typical input

    // Run comprehensive mobile benchmark
    let mobile_results = benchmark_mobile_model_advanced(
        &optimized_model,
        input_shapes,
        mobile_config.stress_test_duration_minutes.unwrap_or(5) as usize * 60, // Convert to iterations
        &mobile_config.platform_info,
    );

    // Memory-pressure, frequency-scaling, and sustained-performance testing
    // run in `benchmark_model` instead of here: that caller also builds
    // `ValidationResults`, so their real measured outputs can be written
    // directly onto `memory_pressure_impact` / `sustained_performance_degradation`
    // there rather than being computed here and then discarded.

    Ok(mobile_results)
}

/// Validate mobile performance against requirements
pub fn validate_mobile_performance(
    batch_results: &HashMap<usize, BatchResult>,
    mobile_config: &MobileBenchmarkConfig,
    mobile_results: Option<&MobileBenchmarkResults>,
) -> Result<ValidationResults> {
    let thresholds = &mobile_config.latency_thresholds;

    // Check latency requirements (use batch size 1 for mobile)
    let batch_1_result = batch_results.get(&1);
    let meets_realtime = batch_1_result
        .map(|r| r.total_time.mean.as_millis() as f32 <= thresholds.realtime_ms)
        .unwrap_or(false);

    let meets_interactive = batch_1_result
        .map(|r| r.total_time.mean.as_millis() as f32 <= thresholds.interactive_ms)
        .unwrap_or(false);

    // Check energy efficiency if targets are set
    let meets_energy = if let (Some(targets), Some(mobile_res)) =
        (&mobile_config.energy_targets, mobile_results)
    {
        mobile_res
            .detailed_metrics
            .energy_efficiency
            .map(|eff| eff >= targets.inferences_per_joule / 1000.0) // Convert to per mW
            .unwrap_or(false)
    } else {
        true // No targets set, consider as met
    };

    // Check thermal throttling
    let thermal_throttling = mobile_results
        .map(|r| {
            matches!(
                r.detailed_metrics.thermal_state,
                ThermalState::Hot | ThermalState::Critical
            )
        })
        .unwrap_or(false);

    // Platform-specific validation
    let platform_validation = validate_platform_requirements(&mobile_config.platform_info);

    // Generate recommendations
    let mut recommendations = Vec::new();

    if !meets_realtime {
        recommendations.push(
            "Model latency exceeds real-time requirements. Consider quantization or pruning."
                .to_string(),
        );
    }

    if !meets_interactive {
        recommendations.push(
            "Model latency exceeds interactive requirements. Optimize critical path operations."
                .to_string(),
        );
    }

    if !meets_energy {
        recommendations.push("Model energy efficiency below target. Consider lower precision or architectural changes.".to_string());
    }

    if thermal_throttling {
        recommendations.push(
            "Thermal throttling detected. Reduce computational intensity or add thermal breaks."
                .to_string(),
        );
    }

    Ok(ValidationResults {
        meets_realtime_latency: meets_realtime,
        meets_interactive_latency: meets_interactive,
        meets_energy_targets: meets_energy,
        thermal_throttling_detected: thermal_throttling,
        memory_pressure_impact: None, // Would be set by memory pressure test
        sustained_performance_degradation: None, // Would be set by sustained test
        platform_validation,
        recommendations,
    })
}

/// Convert module to optimized model representation for benchmarking.
///
/// This is an identity conversion: the returned [`ModelGraph`] is empty and
/// `weights` is empty because this crate has no generic way to extract a
/// layer graph from an arbitrary [`Module`] (the trait exposes no such
/// method). What genuinely *is* derived from `model` is its total
/// parameter byte size -- `numel * dtype size`, summed recursively over
/// every parameter including submodules via [`Module::all_parameters`] --
/// real data, replacing the fixed 10MB/8MB placeholder this used to report
/// for every model regardless of its actual size.
///
/// Because no optimization pass actually runs here, `optimized_size`
/// equals `original_size` (nothing was compressed) and
/// `compression_ratio`/`estimated_speedup` are the true values for an
/// identity transform (`1.0` each) rather than invented numbers.
fn convert_to_optimized_model<M: Module>(model: &M) -> Result<OptimizedModel> {
    use crate::mobile_optimizer::{ModelGraph, OptimizationMetadata};

    let total_bytes: usize = model
        .all_parameters()
        .values()
        .map(|param| {
            let tensor = param.tensor();
            let tensor = tensor.read();
            tensor.numel() * tensor.dtype().size_bytes()
        })
        .sum();

    Ok(OptimizedModel {
        graph: ModelGraph {
            nodes: vec![],
            edges: vec![],
            inputs: vec![],
            outputs: vec![],
        },
        weights: HashMap::new(),
        metadata: OptimizationMetadata {
            original_size: total_bytes,
            optimized_size: total_bytes,
            // original_size == optimized_size always here (no compression
            // actually happened), so 1.0 is the true ratio -- not computed
            // via a division that could divide zero by zero for a
            // parameter-less model.
            compression_ratio: 1.0,
            applied_passes: vec!["benchmark_conversion".to_string()],
            estimated_speedup: 1.0,
            backend_metadata: HashMap::new(),
        },
        backend_data: None,
    })
}

/// Run `samples` real forward passes of `model` and return each one's real
/// wall-clock duration in milliseconds.
fn time_forward_passes<M: Module>(
    model: &M,
    input_shape: &[usize],
    samples: usize,
) -> Result<Vec<f32>> {
    let mut latencies_ms = Vec::with_capacity(samples);
    for _ in 0..samples {
        let input = torsh_tensor::creation::randn(input_shape)?;
        let start = Instant::now();
        let _ = model.forward(&input)?;
        latencies_ms.push(start.elapsed().as_secs_f32() * 1000.0);
    }
    Ok(latencies_ms)
}

/// Arithmetic mean, or `0.0` for an empty slice.
fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

/// Run a real memory-pressure test: measure how much slower `model`'s
/// forward pass becomes while a large, genuinely-committed allocation
/// competes for memory, compared to an unpressured baseline.
///
/// Returns `Ok(None)` only if too few timing samples were usable to compute
/// a meaningful comparison (e.g. all baseline latencies were reported as
/// zero) -- never a fabricated placeholder percentage. `_config` is
/// currently unused (the pressure amount is a fixed, modest constant) but
/// kept for API stability and future tuning.
fn run_memory_pressure_test<M: Module>(
    model: &M,
    input_shape: &[usize],
    _config: &MobileBenchmarkConfig,
) -> Result<Option<f32>> {
    const SAMPLES: usize = 8;
    // 64 MiB: enough to create real memory pressure without being an
    // antisocial allocation on a machine that may be shared with other
    // concurrent work.
    const PRESSURE_BYTES: usize = 64 * 1024 * 1024;

    let baseline = time_forward_passes(model, input_shape, SAMPLES)?;

    // Force real page commitment: a freshly-zeroed `vec![0u8; N]` can stay
    // backed by a single shared zero page on some allocators/OSes until
    // written, which would measure noise while claiming to measure memory
    // pressure. Writing one byte per 4 KiB page (the smallest common page
    // size) guarantees every page is actually resident.
    let mut pressure = vec![0u8; PRESSURE_BYTES];
    for page in pressure.chunks_mut(4096) {
        page[0] = 1;
    }

    let pressured = time_forward_passes(model, input_shape, SAMPLES)?;
    drop(pressure);

    let baseline_mean_ms = mean(&baseline);
    let pressured_mean_ms = mean(&pressured);
    if baseline_mean_ms <= 0.0 {
        return Ok(None);
    }

    Ok(Some(
        (pressured_mean_ms - baseline_mean_ms) / baseline_mean_ms * 100.0,
    ))
}

/// CPU/GPU frequency-scaling control (e.g. Linux cpufreq governors) requires
/// privileged, platform-specific system APIs this crate does not have --
/// portably reading or writing them needs root and/or FFI, which is outside
/// this crate's pure-Rust, unprivileged default. Returns an honest error
/// rather than the fabricated `Ok(())` (a printed line and no actual test)
/// this used to unconditionally report; callers should treat this as
/// "frequency scaling testing is unavailable", not a real pass/fail result.
fn run_frequency_scaling_test<M: Module>(
    _model: &M,
    _config: &MobileBenchmarkConfig,
) -> Result<()> {
    Err(torsh_core::TorshError::NotImplemented(
        "CPU/GPU frequency-scaling control requires privileged, platform-specific system APIs \
         not available in pure Rust"
            .to_string(),
    ))
}

/// Run a real sustained-load test: repeatedly execute `model`'s forward
/// pass and compare the mean latency of the first half of the collected
/// samples against the second half, to detect real performance
/// degradation over time (e.g. thermal throttling) -- instead of the
/// fabricated `None` this used to unconditionally report after printing a
/// line and doing no actual work.
///
/// Bounded by both `duration_minutes` (the caller's requested wall-clock
/// budget) and a hard sample cap, whichever is reached first, so a
/// pathological configuration (a very fast model paired with a very long
/// requested duration) cannot run unbounded in an automated context. At
/// least a small minimum number of samples is always collected (even for
/// `duration_minutes == 0`) so a real, if brief, comparison can still be
/// made.
fn run_sustained_performance_test<M: Module>(
    model: &M,
    input_shape: &[usize],
    _config: &MobileBenchmarkConfig,
    duration_minutes: u32,
) -> Result<Option<f32>> {
    const MIN_SAMPLES: usize = 8;
    const MAX_SAMPLES: usize = 100_000;

    let budget = Duration::from_secs(u64::from(duration_minutes) * 60);
    let start = Instant::now();
    let mut latencies_ms = Vec::new();

    loop {
        let input = torsh_tensor::creation::randn(input_shape)?;
        let sample_start = Instant::now();
        let _ = model.forward(&input)?;
        latencies_ms.push(sample_start.elapsed().as_secs_f32() * 1000.0);

        let min_met = latencies_ms.len() >= MIN_SAMPLES;
        let time_up = start.elapsed() >= budget;
        let hit_cap = latencies_ms.len() >= MAX_SAMPLES;
        if (min_met && time_up) || hit_cap {
            break;
        }
    }

    if latencies_ms.len() < 2 {
        return Ok(None);
    }

    let half = latencies_ms.len() / 2;
    let first_half_mean = mean(&latencies_ms[..half]);
    let second_half_mean = mean(&latencies_ms[half..]);
    if first_half_mean <= 0.0 {
        return Ok(None);
    }

    Ok(Some(
        (second_half_mean - first_half_mean) / first_half_mean * 100.0,
    ))
}

/// Validate platform-specific requirements
fn validate_platform_requirements(
    platform_info: &PlatformBenchmarkInfo,
) -> PlatformValidationResults {
    match &platform_info.platform {
        MobilePlatform::iOS { .. } => PlatformValidationResults {
            ios_app_store_compliant: Some(true), // Would check actual guidelines
            android_performance_class: None,
            device_compatibility_score: 85.0,
            device_support_percentage: 95.0,
        },
        MobilePlatform::Android { .. } => PlatformValidationResults {
            ios_app_store_compliant: None,
            android_performance_class: Some("T".to_string()), // Tier classification
            device_compatibility_score: 80.0,
            device_support_percentage: 90.0,
        },
        MobilePlatform::Other(_) => PlatformValidationResults {
            ios_app_store_compliant: None,
            android_performance_class: None,
            device_compatibility_score: 70.0,
            device_support_percentage: 75.0,
        },
    }
}

/// Benchmark a specific batch size
fn benchmark_batch_size<M: Module>(
    model: &M,
    batch_size: usize,
    base_shape: &[usize],
    config: &BenchmarkConfig,
) -> Result<BatchResult> {
    let mut input_shape = vec![batch_size];
    input_shape.extend_from_slice(base_shape);

    let mut forward_times = Vec::new();
    let mut backward_times = Vec::new();
    let mut memory_samples = Vec::new();

    // Warmup
    for _ in 0..config.warmup_iterations {
        let input = torsh_tensor::creation::randn(&input_shape)?;
        let _ = model.forward(&input)?;
    }

    // Benchmark
    let benchmark_start = Instant::now();

    for _ in 0..config.benchmark_iterations {
        let input = torsh_tensor::creation::randn(&input_shape)?;

        // Forward pass
        let forward_start = Instant::now();
        let output = model.forward(&input)?;
        let forward_time = forward_start.elapsed();
        forward_times.push(forward_time);

        // Backward pass if requested
        if config.profile_backward && output.requires_grad() {
            let backward_start = Instant::now();
            output.sum()?.backward()?;
            let backward_time = backward_start.elapsed();
            backward_times.push(backward_time);
        }

        // Memory profiling
        if config.profile_memory {
            if let Ok((allocated, reserved)) = get_current_memory() {
                memory_samples.push((allocated, reserved));
            }
        }
    }

    let _total_time = benchmark_start.elapsed();

    // Calculate statistics
    let forward_stats = calculate_timing_stats(&forward_times);
    let backward_stats = if !backward_times.is_empty() {
        Some(calculate_timing_stats(&backward_times))
    } else {
        None
    };

    let total_times: Vec<Duration> = forward_times
        .iter()
        .zip(
            backward_times
                .iter()
                .chain(std::iter::repeat(&Duration::ZERO)),
        )
        .map(|(f, b)| *f + *b)
        .collect();

    let total_stats = calculate_timing_stats(&total_times);

    // Calculate throughput (samples per second)
    let avg_time_per_sample = total_stats.mean.as_secs_f32() / batch_size as f32;
    let throughput = 1.0 / avg_time_per_sample;

    // Memory statistics
    let memory_stats = if !memory_samples.is_empty() {
        Some(calculate_memory_stats(&memory_samples))
    } else {
        None
    };

    Ok(BatchResult {
        batch_size,
        forward_time: forward_stats,
        backward_time: backward_stats,
        total_time: total_stats,
        throughput,
        memory_stats,
    })
}

/// Count model parameters, recursively including submodules.
///
/// Uses [`Module::all_parameters`] (not the non-recursive `parameters()`)
/// so a model built from nested submodules is not undercounted.
fn count_parameters<M: Module>(model: &M) -> usize {
    model
        .all_parameters()
        .values()
        .map(|p| p.tensor().read().numel())
        .sum()
}

/// Calculate timing statistics
fn calculate_timing_stats(times: &[Duration]) -> TimingStats {
    let mut sorted_times = times.to_vec();
    sorted_times.sort();

    let n = sorted_times.len() as f32;
    let mean = sorted_times.iter().sum::<Duration>() / sorted_times.len() as u32;

    let variance = sorted_times
        .iter()
        .map(|t| {
            let diff = t.as_secs_f32() - mean.as_secs_f32();
            diff * diff
        })
        .sum::<f32>()
        / n;

    let std = Duration::from_secs_f32(variance.sqrt());

    TimingStats {
        mean,
        std,
        min: sorted_times[0],
        max: sorted_times[sorted_times.len() - 1],
        median: sorted_times[sorted_times.len() / 2],
        p95: sorted_times[(0.95 * n) as usize],
        p99: sorted_times[(0.99 * n) as usize],
    }
}

/// Calculate memory statistics
fn calculate_memory_stats(samples: &[(f32, f32)]) -> MemoryStats {
    let peak_allocated = samples.iter().map(|(a, _)| *a).fold(0.0f32, f32::max);
    let peak_reserved = samples.iter().map(|(_, r)| *r).fold(0.0f32, f32::max);
    let avg_allocated = samples.iter().map(|(a, _)| *a).sum::<f32>() / samples.len() as f32;

    MemoryStats {
        peak_allocated_mb: peak_allocated,
        peak_reserved_mb: peak_reserved,
        avg_allocated_mb: avg_allocated,
    }
}

/// Get current memory usage (RSS and peak) from /proc/self/status.
/// Returns (rss_mb, peak_mb). On non-Linux platforms returns (0.0, 0.0).
fn get_current_memory() -> Result<(f32, f32)> {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").map_err(|e| {
            torsh_core::TorshError::IoError(format!("Failed to read /proc/self/status: {}", e))
        })?;
        let mut rss_kb: Option<u64> = None;
        let mut peak_kb: Option<u64> = None;
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                rss_kb = rest.split_whitespace().next().and_then(|s| s.parse().ok());
            } else if let Some(rest) = line.strip_prefix("VmPeak:") {
                peak_kb = rest.split_whitespace().next().and_then(|s| s.parse().ok());
            }
            if rss_kb.is_some() && peak_kb.is_some() {
                break;
            }
        }
        let rss_mb = rss_kb.unwrap_or(0) as f32 / 1024.0;
        let peak_mb = peak_kb.unwrap_or(0) as f32 / 1024.0;
        return Ok((rss_mb, peak_mb));
    }
    #[cfg(not(target_os = "linux"))]
    {
        // Memory measurement via /proc/self/status not available on this platform
        Ok((0.0, 0.0))
    }
}

/// Generate benchmark summary
fn generate_summary(results: &HashMap<usize, BatchResult>) -> BenchmarkSummary {
    let mut best_throughput = 0.0;
    let mut best_batch_size = 0;
    let mut optimal_memory_batch = 0;
    let mut min_memory_per_sample = f32::INFINITY;

    for (batch_size, result) in results {
        if result.throughput > best_throughput {
            best_throughput = result.throughput;
            best_batch_size = *batch_size;
        }

        if let Some(mem) = &result.memory_stats {
            let memory_per_sample = mem.peak_allocated_mb / *batch_size as f32;
            if memory_per_sample < min_memory_per_sample {
                min_memory_per_sample = memory_per_sample;
                optimal_memory_batch = *batch_size;
            }
        }
    }

    let mut recommendations = Vec::new();

    // Throughput recommendation
    recommendations.push(format!(
        "Best throughput: {:.1} samples/sec at batch size {}",
        best_throughput, best_batch_size
    ));

    // Memory recommendation
    if optimal_memory_batch > 0 {
        recommendations.push(format!(
            "Most memory efficient: batch size {} ({:.1} MB/sample)",
            optimal_memory_batch, min_memory_per_sample
        ));
    }

    // Scaling recommendation
    let batch_sizes: Vec<usize> = results.keys().copied().collect();
    if batch_sizes.len() >= 2 {
        let min_batch = *batch_sizes.iter().min().expect("reduction should succeed");
        let max_batch = *batch_sizes.iter().max().expect("reduction should succeed");

        let min_result = &results[&min_batch];
        let max_result = &results[&max_batch];

        let scaling_efficiency =
            (max_result.throughput * max_batch as f32) / (min_result.throughput * min_batch as f32);

        if scaling_efficiency < 0.8 {
            recommendations.push(format!(
                "Poor scaling efficiency ({:.1}%). Consider optimizing data loading.",
                scaling_efficiency * 100.0
            ));
        }
    }

    BenchmarkSummary {
        best_batch_size,
        best_throughput,
        optimal_memory_batch,
        recommendations,
    }
}

/// Print benchmark results
pub fn print_benchmark_results(results: &BenchmarkResult) {
    println!("=== Benchmark Results for {} ===", results.model_name);
    println!("Total parameters: {}", results.total_params);
    println!();

    println!(
        "{:<10} {:<15} {:<15} {:<15} {:<15}",
        "Batch", "Forward (ms)", "Backward (ms)", "Total (ms)", "Throughput"
    );
    println!("{}", "-".repeat(75));

    for batch_size in results.results_by_batch.keys() {
        let result = &results.results_by_batch[batch_size];
        let backward_str = result
            .backward_time
            .as_ref()
            .map(|t| format!("{:.2}", t.mean.as_secs_f32() * 1000.0))
            .unwrap_or_else(|| "N/A".to_string());

        println!(
            "{:<10} {:<15.2} {:<15} {:<15.2} {:<15.1}",
            batch_size,
            result.forward_time.mean.as_secs_f32() * 1000.0,
            backward_str,
            result.total_time.mean.as_secs_f32() * 1000.0,
            result.throughput
        );
    }
    println!();

    println!("Summary:");
    for rec in &results.summary.recommendations {
        println!("  - {}", rec);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_current_memory_nonnegative() {
        let (rss, peak) = get_current_memory().unwrap_or((0.0, 0.0));
        assert!(rss >= 0.0, "RSS should be non-negative, got {}", rss);
        assert!(peak >= 0.0, "peak should be non-negative, got {}", peak);
        #[cfg(target_os = "linux")]
        {
            assert!(rss > 0.0, "RSS should be positive on Linux, got {}", rss);
        }
    }

    #[test]
    fn test_timing_stats() {
        let times = vec![
            Duration::from_millis(10),
            Duration::from_millis(12),
            Duration::from_millis(11),
            Duration::from_millis(13),
            Duration::from_millis(14),
        ];

        let stats = calculate_timing_stats(&times);
        assert_eq!(stats.min, Duration::from_millis(10));
        assert_eq!(stats.max, Duration::from_millis(14));
        assert_eq!(stats.median, Duration::from_millis(12));
    }

    // --- F173 regression tests ------------------------------------------
    //
    // `convert_to_optimized_model`, `run_memory_pressure_test`,
    // `run_frequency_scaling_test`, and `run_sustained_performance_test`
    // are private, so their real-measurement behavior is verified here
    // rather than from an external `tests/` integration test. See
    // `tests/hardening_profiler.rs` for the public-API-visible half of
    // F173 (the `ValidationResults` fields these feed into).

    fn f173_test_platform_info() -> PlatformBenchmarkInfo {
        use crate::mobile_optimizer::{CpuInfo, MemoryInfo};

        PlatformBenchmarkInfo {
            platform: MobilePlatform::iOS {
                chip: "A15".to_string(),
                neural_engine: true,
            },
            device_model: "test-device".to_string(),
            os_version: "1.0".to_string(),
            cpu_info: CpuInfo {
                cores_performance: 2,
                cores_efficiency: 4,
                max_frequency_ghz: 3.0,
                cache_l1_kb: 128,
                cache_l2_kb: 4096,
                cache_l3_kb: None,
            },
            memory_info: MemoryInfo {
                total_mb: 4096,
                bandwidth_gb_s: 30.0,
                memory_type: "LPDDR5".to_string(),
            },
            thermal_design_power: None,
        }
    }

    fn f173_test_mobile_config(
        test_memory_pressure: bool,
        test_frequency_scaling: bool,
    ) -> MobileBenchmarkConfig {
        MobileBenchmarkConfig {
            platform_info: f173_test_platform_info(),
            monitor_thermal: false,
            measure_power: false,
            test_frequency_scaling,
            test_memory_pressure,
            stress_test_duration_minutes: None,
            latency_thresholds: LatencyThresholds::default(),
            energy_targets: None,
        }
    }

    /// F173: `convert_to_optimized_model` must derive `original_size` from
    /// the real model passed in, not report the same fixed 10MB/8MB
    /// placeholder for every model regardless of its actual size.
    #[test]
    fn test_convert_to_optimized_model_uses_real_model_size() {
        use torsh_nn::layers::Linear;

        let small = Linear::new(4, 4, true);
        let large = Linear::new(256, 256, true);

        let small_result = convert_to_optimized_model(&small).expect("conversion should succeed");
        let large_result = convert_to_optimized_model(&large).expect("conversion should succeed");

        assert_ne!(
            small_result.metadata.original_size, 10_000_000,
            "original_size must not be the historical fixed placeholder"
        );
        assert_ne!(
            small_result.metadata.original_size, large_result.metadata.original_size,
            "differently-sized real models must report different real sizes"
        );

        // Linear(4, 4, bias=true): weight [4,4] + bias [4] = 20 f32 params.
        assert_eq!(small_result.metadata.original_size, 20 * 4);
        // Linear(256, 256, bias=true): weight [256,256] + bias [256] = 65_792 f32 params.
        assert_eq!(large_result.metadata.original_size, 65_792 * 4);

        // No real compression pass runs in this conversion, so the honest
        // values for an identity transform are 1.0, not an invented guess
        // like the historical 1.25 / 1.2.
        for result in [&small_result, &large_result] {
            assert_eq!(
                result.metadata.optimized_size,
                result.metadata.original_size
            );
            assert_eq!(result.metadata.compression_ratio, 1.0);
            assert_eq!(result.metadata.estimated_speedup, 1.0);
        }
    }

    /// F173: frequency-scaling control cannot be portably measured in pure
    /// Rust; this must return an honest error rather than the historical
    /// fabricated `Ok(())` (a printed line and no actual test).
    #[test]
    fn test_run_frequency_scaling_test_is_honest_about_being_unimplemented() {
        use torsh_nn::layers::Linear;

        let model = Linear::new(4, 4, true);
        let config = f173_test_mobile_config(false, true);

        let result = run_frequency_scaling_test(&model, &config);
        assert!(
            result.is_err(),
            "frequency scaling control must return an honest error, not a fabricated Ok(())"
        );
    }

    /// F173: memory pressure testing must return a real, finite measured
    /// value derived from actually timing the model, not the historical
    /// `Ok(())` that discarded any notion of measurement entirely.
    #[test]
    fn test_run_memory_pressure_test_returns_finite_measured_value() {
        use torsh_nn::layers::Linear;

        let model = Linear::new(8, 8, true);
        let config = f173_test_mobile_config(true, false);

        let result = run_memory_pressure_test(&model, &[1, 8], &config)
            .expect("memory pressure test should succeed on a real, working model");
        let value = result
            .expect("enough real timing samples were collected, so a comparison must be Some(_)");
        assert!(
            value.is_finite(),
            "measured degradation must be a real finite number, got {value}"
        );
    }

    /// F173: sustained-performance testing must return a real, finite
    /// measured value derived from actually timing the model repeatedly,
    /// not the historical unconditional `None`. `duration_minutes = 0`
    /// still takes a small minimum number of real samples, so this
    /// completes quickly while still being a genuine measurement.
    #[test]
    fn test_run_sustained_performance_test_returns_finite_measured_value() {
        use torsh_nn::layers::Linear;

        let model = Linear::new(8, 8, true);
        let config = f173_test_mobile_config(false, false);

        let result = run_sustained_performance_test(&model, &[1, 8], &config, 0)
            .expect("sustained performance test should succeed on a real, working model");
        let value = result.expect("enough real timing samples were collected for a comparison");
        assert!(
            value.is_finite(),
            "measured degradation must be a real finite number, got {value}"
        );
    }
}
