//! Regression tests for the production-hardening pass on torsh-utils's
//! bottleneck profiler (F076, F077, F174) and mobile benchmark validation
//! (F173) modules, exercised through torsh-utils's public API.
//!
//! `convert_to_optimized_model`, `run_memory_pressure_test`,
//! `run_frequency_scaling_test`, and `run_sustained_performance_test`
//! (also part of F173) are private and not reachable from here; their
//! real-measurement behavior is covered instead by the inline
//! `#[cfg(test)]` unit tests in `src/benchmark.rs`.

use std::collections::HashMap;
use torsh_nn::layers::Linear;
use torsh_nn::{Module, Parameter};
use torsh_tensor::Tensor;
use torsh_utils::benchmark::{
    benchmark_model, BenchmarkConfig, LatencyThresholds, MobileBenchmarkConfig,
};
use torsh_utils::bottleneck::{
    profile_bottlenecks, profile_bottlenecks_advanced, AdvancedProfilingConfig,
};
use torsh_utils::mobile_optimizer::{CpuInfo, MemoryInfo, MobilePlatform, PlatformBenchmarkInfo};

/// A tiny real model (two stacked `Linear` layers) for exercising the
/// profiler/benchmark public API with genuine forward-pass timings,
/// instead of the mock, non-`Module` structs used elsewhere in this
/// crate's other integration test files.
struct TinyMlp {
    fc1: Linear,
    fc2: Linear,
}

impl TinyMlp {
    fn new() -> Self {
        Self {
            fc1: Linear::new(8, 16, true),
            fc2: Linear::new(16, 4, true),
        }
    }
}

impl Module for TinyMlp {
    fn forward(&self, input: &Tensor) -> torsh_core::error::Result<Tensor> {
        let hidden = self.fc1.forward(input)?;
        self.fc2.forward(&hidden)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = self.fc1.parameters();
        for (name, param) in self.fc2.parameters() {
            params.insert(format!("fc2.{name}"), param);
        }
        params
    }
}

fn test_platform_info() -> PlatformBenchmarkInfo {
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

/// F076 / F174: `profile_bottlenecks`'s memory profile must report cache
/// counters and leak detection as honestly "not measured" (`None`) rather
/// than the historical fabricated constants (`l1_hit_rate: 0.95`,
/// `l2_hit_rate: 0.88`, `l3_hit_rate: 0.82`, `cache_misses_per_instruction:
/// 0.05`, `memory_stalls_percentage: 12.0`) or a false "no leaks found"
/// (`Some(vec![])`, which would claim leak detection ran when it did not).
#[test]
fn f076_f174_profile_bottlenecks_reports_unmeasured_metrics_as_none() {
    let model = TinyMlp::new();
    let report = profile_bottlenecks(&model, &[2, 8], 5, false)
        .expect("profiling a real, working model should succeed");

    let cache = &report.memory_profile.cache_performance;
    assert_eq!(cache.l1_hit_rate, None, "L1 hit rate was never measured");
    assert_eq!(cache.l2_hit_rate, None, "L2 hit rate was never measured");
    assert_eq!(cache.l3_hit_rate, None, "L3 hit rate was never measured");
    assert_eq!(
        cache.cache_misses_per_instruction, None,
        "cache misses per instruction were never measured"
    );
    assert_eq!(
        cache.memory_stalls_percentage, None,
        "memory stalls were never measured"
    );
    assert!(
        report.memory_profile.memory_leaks.is_none(),
        "leak detection is not implemented and must not claim 'no leaks found': got {:?}",
        report.memory_profile.memory_leaks
    );

    // Real memory readings, by contrast, must be actual measurements: a
    // running process always has positive resident memory. Gated to the
    // two platforms `sysinfo`'s process lookup (behind this crate's
    // `collect_env` feature) is expected to work on; other platforms fall
    // back to `0.0` (see `MemoryMetricsCollector::get_metrics`).
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        assert!(
            report.memory_profile.peak_usage_mb > 0.0,
            "sysinfo-backed peak_usage_mb should be a real positive reading on this platform"
        );
    }
}

/// F077: the flame graph and hotspot analysis must be built from the real
/// per-iteration operation timings actually recorded during profiling,
/// never the two hardcoded fake samples this used to return
/// unconditionally regardless of which model was profiled
/// ("conv2d_forward" at a fixed 10.0ms, "matrix_multiply" at a fixed
/// 8.0ms) -- this is the sharpest regression proof that those fake
/// samples are gone: real operation names must appear and the fake ones
/// must not, no matter what model is profiled.
#[test]
fn f077_flame_graph_and_hotspots_come_from_real_timings_not_fake_samples() {
    let model = TinyMlp::new();
    let config = AdvancedProfilingConfig {
        enable_flame_graph: true,
        enable_memory_profiling: false,
        enable_gpu_profiling: false,
        enable_call_stack_analysis: false,
        enable_regression_detection: false,
        enable_hotspot_analysis: true,
        sample_rate_hz: 1000.0,
        memory_snapshot_interval_ms: 10.0,
    };

    let report = profile_bottlenecks_advanced(&model, &[2, 8], 5, false, config)
        .expect("advanced profiling of a real, working model should succeed");

    let flame_graph = report
        .flame_graph
        .expect("flame_graph should be populated when enable_flame_graph is true");

    assert_eq!(
        flame_graph.total_samples, 5,
        "one real sample per iteration of the only timed operation (forward)"
    );

    let frame_names: Vec<&str> = flame_graph
        .root_frame
        .children
        .iter()
        .map(|f| f.name.as_str())
        .collect();
    assert!(
        !frame_names.contains(&"conv2d_forward") && !frame_names.contains(&"matrix_multiply"),
        "flame graph must not contain the historical hardcoded fake function names, got {frame_names:?}"
    );
    assert_eq!(
        frame_names,
        vec!["forward"],
        "flame graph should contain exactly the real operation timed here \
         (only 'forward', since profile_backward=false), got {frame_names:?}"
    );

    for hotspot in &report.hotspot_analysis.cpu_hotspots {
        assert_ne!(hotspot.function_name, "conv2d_forward");
        assert_ne!(hotspot.function_name, "matrix_multiply");
    }
}

/// F173: memory-pressure testing must produce a real, measured value in
/// the final validation report when requested, not the historical
/// unconditional `None` ("Would be set by memory pressure test") that a
/// caller could never distinguish from "not requested".
#[test]
fn f173_memory_pressure_test_reports_real_measured_value() {
    let model = TinyMlp::new();

    let mobile_config = MobileBenchmarkConfig {
        platform_info: test_platform_info(),
        monitor_thermal: false,
        measure_power: false,
        test_frequency_scaling: false,
        test_memory_pressure: true,
        stress_test_duration_minutes: None,
        latency_thresholds: LatencyThresholds::default(),
        energy_targets: None,
    };

    let config = BenchmarkConfig {
        warmup_iterations: 2,
        benchmark_iterations: 5,
        batch_sizes: vec![1],
        input_shapes: vec![vec![8]],
        profile_memory: false,
        profile_backward: false,
        device: torsh_core::DeviceType::Cpu,
        mobile_config: Some(mobile_config),
    };

    let result = benchmark_model(&model, config).expect("benchmarking a real model should succeed");
    let validation = result
        .validation_results
        .expect("validation_results should be populated when mobile_config is set");

    let value = validation.memory_pressure_impact.expect(
        "test_memory_pressure=true must produce a real measured value, not the historical None",
    );
    assert!(
        value.is_finite(),
        "measured memory pressure impact must be finite, got {value}"
    );
}

/// F173: sustained-performance testing must produce a real, measured value
/// in the final validation report when requested, not the historical
/// unconditional `None` ("would be set by sustained test").
#[test]
fn f173_sustained_performance_test_reports_real_measured_value() {
    let model = TinyMlp::new();

    let mobile_config = MobileBenchmarkConfig {
        platform_info: test_platform_info(),
        monitor_thermal: false,
        measure_power: false,
        test_frequency_scaling: false,
        test_memory_pressure: false,
        // NOTE: intentionally *not* `Some(0)` here -- `stress_test_duration_minutes:
        // Some(0)` drives `mobile_optimizer::functions::benchmark_mobile_model_advanced`'s
        // `num_runs` to exactly 0, which makes its (out-of-scope, unrelated)
        // `calculate_percentile` helper compute `sorted.len() - 1` on an
        // empty vec and panic with "attempt to subtract with overflow".
        // See the FOLLOW-UP note in the final report. `Some(1)` keeps
        // `num_runs` positive while `run_sustained_performance_test`'s own
        // MAX_SAMPLES cap (see src/benchmark.rs) still bounds this test to
        // a short, deterministic run instead of a real 60-second wait.
        stress_test_duration_minutes: Some(1),
        latency_thresholds: LatencyThresholds::default(),
        energy_targets: None,
    };

    let config = BenchmarkConfig {
        warmup_iterations: 2,
        benchmark_iterations: 5,
        batch_sizes: vec![1],
        input_shapes: vec![vec![8]],
        profile_memory: false,
        profile_backward: false,
        device: torsh_core::DeviceType::Cpu,
        mobile_config: Some(mobile_config),
    };

    let result = benchmark_model(&model, config).expect("benchmarking a real model should succeed");
    let validation = result
        .validation_results
        .expect("validation_results should be populated when mobile_config is set");

    let value = validation.sustained_performance_degradation.expect(
        "stress_test_duration_minutes=Some(_) must produce a real measured value, \
         not the historical None",
    );
    assert!(
        value.is_finite(),
        "measured sustained performance degradation must be finite, got {value}"
    );
}

/// F173: requesting frequency-scaling testing (which cannot be portably
/// measured in pure Rust) must not silently claim success or abort the
/// entire benchmark run -- it should be reported as skipped, with the
/// rest of a real report still intact.
#[test]
fn f173_frequency_scaling_request_is_reported_as_skipped_not_silently_ok() {
    let model = TinyMlp::new();

    let mobile_config = MobileBenchmarkConfig {
        platform_info: test_platform_info(),
        monitor_thermal: false,
        measure_power: false,
        test_frequency_scaling: true,
        test_memory_pressure: false,
        stress_test_duration_minutes: None,
        latency_thresholds: LatencyThresholds::default(),
        energy_targets: None,
    };

    let config = BenchmarkConfig {
        warmup_iterations: 2,
        benchmark_iterations: 5,
        batch_sizes: vec![1],
        input_shapes: vec![vec![8]],
        profile_memory: false,
        profile_backward: false,
        device: torsh_core::DeviceType::Cpu,
        mobile_config: Some(mobile_config),
    };

    let result = benchmark_model(&model, config)
        .expect("an unimplemented sub-test must not abort the whole benchmark run");
    let validation = result
        .validation_results
        .expect("validation_results should be populated when mobile_config is set");

    assert!(
        validation
            .recommendations
            .iter()
            .any(|r| r.contains("Frequency scaling test skipped")),
        "requesting frequency scaling testing should be disclosed as skipped, got {:?}",
        validation.recommendations
    );
}
