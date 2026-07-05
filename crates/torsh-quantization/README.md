# torsh-quantization

Quantization toolkit for ToRSh, enabling efficient model deployment with reduced precision.

## Overview

This crate provides tensor-level quantization support for deep learning workloads:

- **Quantization Schemes**: INT8, INT4, binary, ternary, mixed-precision, per-channel and group-wise, exposed as `QuantConfig` presets
- **Observers/Calibration**: MinMax, moving-average, histogram and percentile observers for computing scale/zero-point
- **Auto-Configuration**: An ML-assisted `AutoConfigurator` that recommends a `QuantConfig` from tensor statistics and an optimization objective
- **Quality Analysis**: PSNR/SNR/MAE/cosine-similarity metrics, multi-config comparison, and auto-calibration helpers
- **Backends**: A `QuantBackend` enum (FBGEMM, QNNPACK, Native, XNNPACK) used to tag configs for downstream kernel selection

The stable, default-feature surface operates on individual `Tensor`s (`quantize_with_config` / `dequantize` / `calculate_quantization_metrics`). A model-level Post-Training/Quantization-Aware-Training pipeline (calibrating and converting a whole `Module`, ONNX/TensorRT export, etc.) also exists in the source tree, but it is gated behind the `experimental` feature and currently has no concrete implementor for the `Module` trait it depends on — see [TODO.md](./TODO.md) for the tracked status. Treat the per-tensor API below as the primary supported surface.

## Usage

### Per-Tensor Quantization

This is the primary, stable API surface (used directly in `examples/basic_quantization.rs`):

```rust
use torsh_quantization::{calculate_quantization_metrics, dequantize, quantize_with_config, QuantConfig};
use torsh_tensor::creation::tensor_1d;

let data = vec![-10.5, -5.2, -2.1, 0.0, 1.3, 3.7, 5.9, 8.4, 12.6, 15.8];
let tensor = tensor_1d(&data)?;

// Quantize using an INT8 preset configuration
let config = QuantConfig::int8();
let (quantized, scale, zero_point) = quantize_with_config(&tensor, &config)?;

// Dequantize back to floating point
let dequantized = dequantize(&quantized, scale, zero_point)?;

// Calculate quality metrics (bits_before, bits_after)
let metrics = calculate_quantization_metrics(&tensor, &dequantized, 32, 8)?;
println!("PSNR: {:.2} dB, compression: {:.2}x, MAE: {:.6}",
    metrics.psnr, metrics.compression_ratio, metrics.mae);
```

Model-level static/dynamic quantization (`prepare_static`/`convert`/`quantize_dynamic` over a whole model) is **not currently available**: the experimental `qat`/`post_training` modules define their own local placeholder `Module` trait (distinct from `torsh_nn::Module`) with no concrete implementor yet.

### Quantization-Aware Training (QAT)

A `QuantConfig::qat()` preset (fake-quant enabled, moving-average observer) is available today:

```rust
use torsh_quantization::QuantConfig;

let qat_config = QuantConfig::qat();
```

The full QAT training pipeline (`prepare_qat` wrapping an arbitrary model, training-loop integration, `convert` back to a quantized model) lives behind the `experimental` feature in `src/qat.rs`, but — like the post-training pipeline above — it operates on a local placeholder `Module` trait with no concrete implementor yet, so it is not usable end-to-end at the model level. Track status in [TODO.md](./TODO.md).

### Custom Quantization Configuration

Configuration is built with `QuantConfigBuilder` (there is no per-layer-name/per-module-type `QConfigDict`, and no `quantize_fx`):

```rust
use torsh_quantization::config::QuantConfigBuilder;
use torsh_quantization::{ObserverType, QScheme, QuantBackend};

let config = QuantConfigBuilder::new()
    .scheme(QScheme::PerChannelSymmetric)
    .observer(ObserverType::Histogram)
    .backend(QuantBackend::Fbgemm)
    .channel_axis(0)
    .build()?;
```

### Quantization Schemes

`QuantConfig` ships one preset constructor per scheme (see `QScheme` for the full enum: `PerTensorAffine`, `PerChannelAffine`, `PerTensorSymmetric`, `PerChannelSymmetric`, `Int4PerTensor`, `Int4PerChannel`, `MixedPrecision`, `Binary`, `Ternary`, `GroupWise`):

```rust
use torsh_quantization::QuantConfig;

let int8 = QuantConfig::int8();
let int4 = QuantConfig::int4();
let binary = QuantConfig::binary();
let ternary = QuantConfig::ternary();
let mixed = QuantConfig::mixed_precision();
let per_channel = QuantConfig::per_channel(0);           // ch_axis
let group_wise = QuantConfig::group_wise(0, 32);         // ch_axis, group_size

// Observers (a single `Observer` type, not per-scheme observer structs)
use torsh_quantization::observers::Observer;
let hist_observer = Observer::histogram_with_bins(1024);
let pct_observer = Observer::percentile_with_value(99.9);
```

### Model Analysis

There is no `compare_models`/`sensitivity_analysis` that takes a live model plus calibration/test data. The real, tensor-level equivalents are `compare_quantization_configs` (used in `examples/advanced_schemes.rs`) and a heuristic, name-pattern-based `quick_sensitivity_analysis`:

```rust
use torsh_quantization::{compare_quantization_configs, QuantConfig};

let configs = vec![
    QuantConfig::int8(),
    QuantConfig::int4(),
    QuantConfig::binary(),
    QuantConfig::ternary(),
];

// Returns Vec<(config, metrics, time_ms)>
let comparison = compare_quantization_configs(&tensor, &configs)?;
for (config, metrics, time_ms) in &comparison {
    println!("{:?}: PSNR {:.2} dB, {:.2}x compression, {:.2} ms",
        config.scheme, metrics.psnr, metrics.compression_ratio, time_ms);
}

// Heuristic sensitivity analysis by layer name (no live model/data is used)
use torsh_quantization::analysis::quick_sensitivity_analysis;
let layer_names = vec!["conv1".to_string(), "fc1".to_string()];
let sensitivity = quick_sensitivity_analysis(&layer_names)?;
```

### Export and Deployment

There are no free functions `optimize_for_mobile`/`export_quantized_onnx`/`export_tensorrt`. The `experimental`-gated `export` module instead provides `ExportFormat` (Onnx, TensorRT, Mobile, TFLite, CoreML), an `ExportConfig`, and a `ModelExporter`:

```rust
use torsh_quantization::export::{ExportConfig, ExportFormat, ModelExporter};

let config = ExportConfig {
    format: ExportFormat::Onnx,
    optimize_for_inference: true,
    ..Default::default()
};
let exporter = ModelExporter::new(config);
// exporter.export_model(&quantized_model, output_path)? — requires a `QuantizedModel`,
// which today must be constructed by hand since the automatic model-conversion
// pipeline is blocked on the same Module-trait gap noted above.
```

### Debugging and Visualization

`QuantizationDebugger` (`experimental` feature) works per-tensor via `debug_quantization`, not over a model + data loader:

```rust
use torsh_quantization::debugging::QuantizationDebugger;

let mut debugger = QuantizationDebugger::new();
debugger.debug_quantization("layer1", &input, &dequantized, &config, scale, zero_point)?;
println!("{}", debugger.generate_report());
```

There is no `get_observer_dict`/`plot_error_heatmap`; observer ranges are read directly from an `Observer` via `calculate_qparams(dtype)`.

### Advanced Features

There is no `LearnableFakeQuantize`, `StochasticQuantize`, or standalone `GroupWiseQuantConfig` (group-wise is a `QuantConfig::group_wise(...)` preset, shown above). A genuinely available advanced feature is constraint-based auto-configuration:

```rust
use torsh_quantization::auto_config::{AutoConfigurator, ConfigConstraints, ConfigObjective};

let constraints = ConfigConstraints::new()
    .with_min_bits(4)
    .with_target_compression(4.0);

let configurator = AutoConfigurator::new(ConfigObjective::BalancedQuality);
let config = configurator.recommend(&tensor, Some(constraints))?; // takes ConfigConstraints by value
```

### Quantization Backends

`QuantBackend` is a plain enum tag on the config (there is no platform-conditional compilation tied to it, and no `CustomBackend`):

```rust
use torsh_quantization::{QuantBackend, QuantConfig};

let fbgemm_config = QuantConfig::default().with_backend(QuantBackend::Fbgemm);
let qnnpack_config = QuantConfig::default().with_backend(QuantBackend::Qnnpack);
// QuantBackend also has Native and Xnnpack variants
```

## Scope

There are no layer-type-specific quantized kernels (no `QuantizedLinear`/`QuantizedConv2d`/`QuantizedLSTM`, etc.). Quantization operates generically on `Tensor`s regardless of which layer produced them — per-tensor, per-channel, or group-wise, at whatever scheme/bit-width the chosen `QuantConfig` specifies. An operation-fusion pattern matcher (Conv+BN, Conv+ReLU, Linear+ReLU, ...) exists behind the `experimental` feature for graph-level optimization passes, independent of the per-tensor API above.

## Best Practices

1. Use representative calibration data
2. Start with INT8 before trying lower bit widths
3. Use per-channel quantization for Conv/Linear layers
4. Keep sensitive layers in higher precision
5. Profile on target hardware

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../../LICENSE) for details.