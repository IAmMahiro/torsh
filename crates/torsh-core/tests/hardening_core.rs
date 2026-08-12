//! Regression tests for the production-hardening campaign (wave1_F_core).
//!
//! Each test documents and guards against one specific finding from the
//! torsh-core hardening audit: a fabricated "detection" result, a panic on
//! malformed/edge-case input, or dead code hiding a real bug. Where a
//! finding required inspecting private state (lock poisoning, forcing an
//! internal field) the regression test lives inline in the corresponding
//! `src/*.rs` file's own `#[cfg(test)] mod tests` instead (see
//! `op_trace.rs::tests::test_trace_operation_result_survives_poisoned_lock`
//! for F091 and `gpu_shape_ops.rs::tests::test_gpu_stats_never_lie_about_unavailable_gpu_backend`
//! for F296); this file covers everything reachable through the public API.

use torsh_core::debug_validation::validate_shape_valid;
use torsh_core::device::{DeviceCapabilities, DeviceType};
use torsh_core::dtype::{calculate_qint32_params, calculate_qint8_params, calculate_quint8_params};
use torsh_core::runtime_config::{RuntimeConfig, ValidationLevel};
use torsh_core::shape::Shape;
use torsh_core::sparse::{CooIndices, CsrIndices};
use torsh_core::type_level_shapes::{Dim, DimList};

// ---------------------------------------------------------------------
// F093 / F094: device capability detection must not fabricate GPU specs
// ---------------------------------------------------------------------

/// F093/F094: with the `cuda` feature off (the default build), CUDA
/// capability detection must report unavailability rather than success.
#[test]
#[cfg(not(feature = "cuda"))]
fn f093_cuda_capabilities_honest_when_not_compiled() {
    let result = DeviceCapabilities::detect(DeviceType::Cuda(0));
    assert!(
        result.is_err(),
        "CUDA capabilities must not silently succeed when CUDA support isn't compiled in"
    );
}

/// F093/F094: with the `cuda` feature compiled in but no real CUDA driver
/// wiring available in torsh-core, capability detection must return an
/// honest error instead of the previous hardcoded fabricated spec (8 GiB
/// total memory, 900 GB/s bandwidth, 108 SMs, driver "12.0").
#[test]
#[cfg(feature = "cuda")]
fn f093_cuda_capabilities_honest_when_compiled_without_real_backend() {
    let result = DeviceCapabilities::detect(DeviceType::Cuda(0));
    assert!(
        result.is_err(),
        "torsh-core has no real CUDA driver access; capabilities must not report fabricated \
         success (e.g. the old hardcoded 8 GiB / 900 GB/s / 108-SM profile)"
    );
}

/// F206: `CudaDevice::new` must not fabricate a context handle / compute
/// capability as if a real CUDA context had been created.
#[test]
#[cfg(feature = "cuda")]
fn f206_cuda_device_new_no_fabricated_context() {
    let result = torsh_core::device::CudaDevice::new(0);
    assert!(
        result.is_err(),
        "torsh-core has no real CUDA driver access; CudaDevice::new must not fabricate a \
         context_handle/compute_capability (previously 0x12345678 / (8, 6)) as if a real \
         context were created"
    );
}

/// F206 / blocking consistency requirement: device discovery must never
/// leave a device registered whose construction "succeeded" but whose
/// capabilities subsequently fail -- that combination previously caused
/// downstream panics/errors in `DeviceManager` (`create_device_summary`,
/// `select_best_fit_device`, etc.). With CUDA capabilities and CUDA device
/// construction now both honestly failing, no CUDA device is ever
/// registered, so device selection must complete without panicking.
#[test]
#[cfg(feature = "cuda")]
fn f206_device_manager_discovery_stays_consistent_with_cuda_enabled() {
    use torsh_core::device::DeviceManager;

    let manager = DeviceManager::new();
    manager
        .discover_devices()
        .expect("discover_devices must not fail even though CUDA devices can't be constructed");

    let best = manager
        .get_best_device()
        .expect("get_best_device must not panic/error even with CUDA enabled");
    assert!(
        best.is_some(),
        "CPU should still be discovered even though CUDA construction honestly fails"
    );
}

/// F094: Metal capability detection is on by default for every macOS build
/// (no feature flag gates it). It must report real, queried values for
/// total memory and GPU core count, and must not fabricate memory
/// bandwidth, clock rate, driver version, or thermal telemetry that cannot
/// actually be queried without vendor-specific tooling this crate doesn't
/// have.
#[test]
#[cfg(target_os = "macos")]
fn f094_metal_capabilities_no_fabricated_bandwidth_clock_or_driver_version() {
    let cap = DeviceCapabilities::detect(DeviceType::Metal(0))
        .expect("Metal capability detection should succeed on macOS");

    assert_eq!(cap.device_type(), DeviceType::Metal(0));
    // Previously fabricated as `Some(400 * 1024^3)` (400 GB/s).
    assert_eq!(
        cap.memory_bandwidth(),
        None,
        "memory bandwidth cannot be honestly queried on macOS; must not be fabricated"
    );
    // Previously fabricated as `Some(1398)` MHz.
    assert_eq!(
        cap.clock_rate(),
        None,
        "GPU clock rate cannot be honestly queried on macOS; must not be fabricated"
    );
    // Previously fabricated as `Some("Metal 3.0")`.
    assert_eq!(
        cap.driver_version(),
        None,
        "Metal has no queryable driver version from user space; must not be fabricated"
    );
    // Previously fabricated via `ThermalInfo::mock_integrated()`.
    assert!(
        cap.thermal_info().is_none(),
        "thermal telemetry requires elevated privileges/IOKit access this crate doesn't have; \
         must not be fabricated"
    );

    // Real values: total memory should reflect this machine's actual RAM
    // (sanity bound well below any real Mac, real value queried via sysctl).
    assert!(cap.total_memory() >= 1024 * 1024 * 1024);
    // Real (or conservatively-defaulted, never 0) GPU core count.
    assert!(cap.compute_units() > 0);
}

/// F206: `MetalDevice::execute_compute_shader` used to silently return
/// `Ok(())` on macOS without compiling or dispatching anything, which would
/// make a caller believe their shader executed. It must now honestly report
/// that shader execution isn't implemented, matching the (already-correct)
/// non-macOS behavior.
#[test]
#[cfg(target_os = "macos")]
fn f206_metal_execute_compute_shader_reports_not_implemented() {
    let device =
        torsh_core::device::MetalDevice::new(0).expect("MetalDevice::new should succeed on macOS");
    let result = device.execute_compute_shader("kernel void noop() {}");
    assert!(
        result.is_err(),
        "execute_compute_shader used to silently return Ok(()) without running anything; it \
         must now honestly report that shader execution is not implemented"
    );
}

/// F206: the Metal device name must be the real, OS-queried chip name, not
/// the previous synthesized `"Apple GPU {index}"` placeholder presented as
/// if it were queried hardware information.
#[test]
#[cfg(target_os = "macos")]
fn f206_metal_device_name_is_not_the_old_synthesized_placeholder() {
    let device =
        torsh_core::device::MetalDevice::new(0).expect("MetalDevice::new should succeed on macOS");
    let name = device
        .metal_device_name()
        .expect("a macOS Metal device should have a name");
    assert_ne!(
        name, "Apple GPU 0",
        "device name must be a real queried chip name, not the old synthesized placeholder"
    );
    assert!(!name.is_empty());
}

/// F204: `DeviceCapabilities::query_gpu_memory`/`query_compute_capability`
/// used to be guarded by a permanently-false phantom `scirs2_gpu_available`
/// cfg that referenced a type (`crate::gpu::GpuDevice`) that no longer
/// exists (the dead branch has been removed as part of this pass). Behavior
/// is unchanged -- these were already always `None` -- confirmed here as a
/// regression guard against that cleanup accidentally changing behavior.
#[test]
fn f204_gpu_memory_and_compute_capability_queries_stay_honest_none() {
    assert!(DeviceCapabilities::query_gpu_memory(0).is_none());
    assert!(DeviceCapabilities::query_compute_capability(0).is_none());
}

/// F198: WebGPU support detection must never unconditionally claim WebGPU
/// is available. `detect_webgpu_support` itself is private; it is exercised
/// here through the public `BackendFeatureDetector::new()` entry point,
/// which is the only way it's reachable.
#[test]
#[cfg(feature = "wgpu")]
fn f198_webgpu_support_not_fabricated_true() {
    let detector = torsh_core::backend_detection::BackendFeatureDetector::new()
        .expect("BackendFeatureDetector::new should succeed");
    assert!(
        !detector.runtime_features.gpu_features.webgpu_available,
        "torsh-core has no real wgpu::Adapter probe wired up; webgpu_available must not be \
         hardcoded true (this used to return true unconditionally, even on headless CI with no \
         GPU at all)"
    );
}

// ---------------------------------------------------------------------
// F199: sparse index constructors must reject malformed input, not panic
// ---------------------------------------------------------------------

#[test]
fn f199_coo_indices_new_2d_rejects_mismatched_lengths() {
    let result = CooIndices::new_2d(vec![0, 1, 2], vec![0, 1]);
    assert!(
        result.is_err(),
        "mismatched row/col lengths must return Err, not panic via assert_eq!"
    );
}

#[test]
fn f199_coo_indices_new_2d_accepts_valid_input() {
    let coo = CooIndices::new_2d(vec![0, 1, 2], vec![1, 0, 2]).expect("valid input should succeed");
    assert_eq!(coo.nnz(), 3);
}

#[test]
fn f199_coo_indices_new_nd_rejects_too_few_dimensions() {
    let result = CooIndices::new_nd(vec![vec![0, 1]]);
    assert!(
        result.is_err(),
        "fewer than 2 dimensions must return Err, not panic"
    );
}

#[test]
fn f199_coo_indices_new_nd_rejects_mismatched_dimension_lengths() {
    let result = CooIndices::new_nd(vec![vec![0, 1, 2], vec![0, 1]]);
    assert!(
        result.is_err(),
        "mismatched per-dimension lengths must return Err, not panic via assert_eq!"
    );
}

#[test]
fn f199_csr_indices_new_rejects_inconsistent_row_ptrs() {
    // Last row pointer (5) does not equal col_indices.len() (2).
    let result = CsrIndices::new(vec![0, 1, 5], vec![0, 1]);
    assert!(
        result.is_err(),
        "row_ptrs inconsistent with col_indices must return Err, not panic via assert_eq!"
    );
}

#[test]
fn f199_csr_indices_new_rejects_non_monotonic_row_ptrs() {
    let result = CsrIndices::new(vec![0, 3, 2, 4], vec![0, 1, 2, 3]);
    assert!(
        result.is_err(),
        "non-monotonic row_ptrs must return Err, not panic via assert!"
    );
}

#[test]
fn f199_csr_indices_new_accepts_valid_input() {
    let csr = CsrIndices::new(vec![0, 2, 3, 5], vec![1, 2, 0, 1, 2]).expect("valid should succeed");
    assert_eq!(csr.nrows(), 3);
    assert_eq!(csr.nnz(), 5);
}

// ---------------------------------------------------------------------
// F200: DType cuDNN/CUDA conversions must return Result, not panic
// ---------------------------------------------------------------------

/// F200: `to_cuda_data_type` must return `Result` and report unsupported
/// dtypes as an error rather than panicking. `to_cudnn_data_type` has the
/// same fix applied but is additionally gated to x86_64 Linux/Windows, so
/// it cannot be exercised from this (or any single) test target; verified
/// instead by code inspection and by the fact that the crate compiles
/// cleanly with `--features cuda,cudnn` (see the FINAL REPORT).
#[test]
#[cfg(feature = "cuda")]
fn f200_to_cuda_data_type_returns_result_not_panic() {
    use torsh_core::dtype::DType;

    assert!(DType::F32.to_cuda_data_type().is_ok());
    assert!(DType::I8.to_cuda_data_type().is_ok());
    // Previously: `_ => panic!("Unsupported data type for CUDA: {:?}", self)`.
    assert!(
        DType::Bool.to_cuda_data_type().is_err(),
        "unsupported dtype must return Err, not panic"
    );
    assert!(DType::QInt8.to_cuda_data_type().is_err());
}

// ---------------------------------------------------------------------
// F201: quantization parameter calculation must reject non-finite/invalid
// calibration statistics, not panic via assert!
// ---------------------------------------------------------------------

#[test]
fn f201_calculate_qint8_params_rejects_nan() {
    assert!(
        calculate_qint8_params(f32::NAN, 10.0).is_err(),
        "NaN calibration statistics must return Err, not panic via assert!"
    );
    assert!(calculate_qint8_params(-10.0, f32::NAN).is_err());
    assert!(calculate_qint8_params(f32::INFINITY, f32::INFINITY).is_err());
}

#[test]
fn f201_calculate_qint8_params_rejects_max_less_than_min() {
    assert!(
        calculate_qint8_params(10.0, -10.0).is_err(),
        "max_value < min_value must return Err, not panic via assert!"
    );
}

#[test]
fn f201_calculate_qint8_params_accepts_valid_range() {
    let (scale, _zero_point) =
        calculate_qint8_params(-10.0, 10.0).expect("valid range should succeed");
    assert!(scale > 0.0);
}

#[test]
fn f201_calculate_quint8_and_qint32_also_reject_nan() {
    assert!(calculate_quint8_params(f32::NAN, 1.0).is_err());
    assert!(calculate_qint32_params(f32::NAN, 1.0).is_err());
    assert!(calculate_quint8_params(5.0, -5.0).is_err());
    assert!(calculate_qint32_params(5.0, -5.0).is_err());
}

// ---------------------------------------------------------------------
// F202: type-level DimList::get_dim must return Option, not panic on the
// empty-list base case
// ---------------------------------------------------------------------

#[test]
fn f202_dim_list_get_dim_out_of_bounds_returns_none() {
    type ShapeType = (Dim<3>, (Dim<4>, ()));
    assert_eq!(<ShapeType as DimList>::get_dim(0), Some(3));
    assert_eq!(<ShapeType as DimList>::get_dim(1), Some(4));
    // Previously: walks past the end into the `()` base case and panics
    // with "Index out of bounds".
    assert_eq!(<ShapeType as DimList>::get_dim(2), None);
    assert_eq!(<ShapeType as DimList>::get_dim(100), None);
}

#[test]
fn f202_empty_dim_list_get_dim_returns_none() {
    assert_eq!(<() as DimList>::get_dim(0), None);
}

// ---------------------------------------------------------------------
// F205: Shape::numel() saturates silently; try_numel() must report
// overflow instead
// ---------------------------------------------------------------------

#[test]
fn f205_shape_try_numel_detects_overflow() {
    let overflowing = Shape::new(vec![usize::MAX, 2]);
    assert!(
        overflowing.try_numel().is_err(),
        "overflowing element count must be reported as an error"
    );
    // numel() still saturates (now clearly documented as such), for contrast:
    assert_eq!(overflowing.numel(), usize::MAX);
}

#[test]
fn f205_shape_try_numel_matches_numel_when_no_overflow() {
    let shape = Shape::new(vec![2, 3, 4]);
    assert_eq!(shape.try_numel().expect("should not overflow"), 24);
    assert_eq!(
        shape.try_numel().expect("should not overflow"),
        shape.numel()
    );
}

/// F205: `validate_shape_valid`'s overflow check (`numel() == 0`) could
/// never fire once `numel()` was changed to saturate to `usize::MAX`
/// instead of wrapping to `0` on overflow -- it became dead code that only
/// ever caught the already-separately-checked "contains a literal zero
/// dimension" case. It must now use `try_numel()` to actually detect
/// overflow.
#[test]
fn f205_validate_shape_valid_detects_overflowing_numel() {
    let config = RuntimeConfig::global();
    let previous_level = config.validation_level();
    config.set_validation_level(ValidationLevel::Standard);

    let overflowing = Shape::new(vec![usize::MAX, 2]);
    let result = validate_shape_valid(&overflowing);

    config.set_validation_level(previous_level); // restore before asserting

    assert!(
        result.is_err(),
        "validate_shape_valid's overflow check was dead code under saturating numel(); it must \
         now detect the overflow via try_numel()"
    );
}
