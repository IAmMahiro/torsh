//! Production-hardening regression tests for `torsh-quantization`.
//!
//! Each test maps to a finding ID from the hardening campaign:
//!
//! * F139 - symmetric quantization must not reuse the affine scale formula
//! * F140 - `DType::U8` must use the full `[0, 255]` range
//! * F141 - per-channel / group-wise quantizers must expose invertible parameters
//! * F249 - AVX2 / AVX-512 kernels must compile and agree with the scalar path
//! * F250 - PTQ calibration must feed each observer its own layer activations

use torsh_core::device::DeviceType;
use torsh_core::DType;
use torsh_quantization::{
    algorithms::{
        calculate_symmetric_quantization_params, dequantize, dequantize_per_channel,
        dequantize_per_group, quantize_tensor_auto, quantize_with_config,
        quantize_with_config_full, QParams,
    },
    config::{QScheme, QuantConfig},
    specialized::{
        quantize_group_wise_full, quantize_int4_per_channel, quantize_int4_per_channel_full,
    },
};
use torsh_tensor::{creation::tensor_1d, Tensor};

fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// ---------------------------------------------------------------------------
// F139 - symmetric scale (always-compiled `algorithms` path)
// ---------------------------------------------------------------------------

#[test]
fn f139_live_symmetric_scale_is_max_abs_over_qmax() {
    // Asymmetric input range [-1.0, 3.0]: the affine formula would give
    // 4.0/255 = 0.01569 and clip 3.0 at 127 -> 1.99. The symmetric formula
    // must give 3.0/127 = 0.02362.
    let (scale, zero_point) =
        calculate_symmetric_quantization_params(-1.0, 3.0, DType::I8).expect("qparams");
    assert_eq!(zero_point, 0, "signed symmetric zero-point must be 0");
    assert!(
        (scale - 3.0 / 127.0).abs() < 1e-6,
        "expected max_abs/qmax, got {scale}"
    );

    let tensor = tensor_1d(&[-1.0, 0.0, 1.5, 3.0]).expect("tensor");
    let (quantized, scale, zero_point) =
        quantize_tensor_auto(&tensor, DType::I8, QScheme::PerTensorSymmetric).expect("quantize");
    let restored = dequantize(&quantized, scale, zero_point).expect("dequantize");
    let err = max_abs_error(&tensor.data().expect("d"), &restored.data().expect("d"));
    assert!(err <= scale, "round-trip error {err} exceeds scale {scale}");
}

// ---------------------------------------------------------------------------
// F140 - U8 must use the whole [0, 255] range
// ---------------------------------------------------------------------------

#[test]
fn f140_live_u8_uses_full_range() {
    let tensor = tensor_1d(&[0.0, 2.5, 5.0, 7.5, 10.0]).expect("tensor");
    let config = QuantConfig::uint8();

    let (quantized, scale, zero_point) = quantize_with_config(&tensor, &config).expect("quantize");
    let codes = quantized.data().expect("data");

    assert!(
        codes.iter().all(|&c| (0.0..=255.0).contains(&c)),
        "U8 codes out of range: {codes:?}"
    );
    assert!(
        codes.iter().cloned().fold(0.0f32, f32::max) > 200.0,
        "U8 quantization saturated early (max code {}), the top half of the \
         range is being discarded",
        codes.iter().cloned().fold(0.0f32, f32::max)
    );

    let restored = dequantize(&quantized, scale, zero_point).expect("dequantize");
    let err = max_abs_error(&tensor.data().expect("d"), &restored.data().expect("d"));
    assert!(err <= scale, "round-trip error {err} exceeds scale {scale}");
}

#[test]
fn f140_live_u8_negative_dominant_range() {
    // |min| > max: zero-point lands well above 127.
    let tensor = tensor_1d(&[-2.0, -1.0, 0.0, 0.5, 1.0]).expect("tensor");
    let config = QuantConfig::uint8();

    let (quantized, scale, zero_point) = quantize_with_config(&tensor, &config).expect("quantize");
    assert!(
        (0..=255).contains(&zero_point),
        "zero-point {zero_point} outside U8 range"
    );

    let restored = dequantize(&quantized, scale, zero_point).expect("dequantize");
    let err = max_abs_error(&tensor.data().expect("d"), &restored.data().expect("d"));
    assert!(err <= scale, "round-trip error {err} exceeds scale {scale}");
}

// ---------------------------------------------------------------------------
// F141 - per-channel / group-wise parameters must be recoverable
// ---------------------------------------------------------------------------

fn wide_range_matrix() -> Tensor {
    // Row 0 spans [0, 2]; row 1 spans [0, 100].
    Tensor::from_data(
        vec![0.0, 1.0, 2.0, 0.0, 50.0, 100.0],
        vec![2, 3],
        DeviceType::Cpu,
    )
    .expect("tensor")
}

#[test]
fn f141_per_channel_params_are_invertible() {
    let tensor = wide_range_matrix();
    let config = QuantConfig::per_channel(0);

    let quantized = quantize_with_config_full(&tensor, &config).expect("quantize");
    assert!(
        matches!(quantized.params, QParams::PerChannel { .. }),
        "per-channel scheme must yield per-channel parameters"
    );
    let restored = quantized.dequantize().expect("dequantize");

    // Channel 1 spans [0, 100]; with a correct parameter set the worst-case
    // error is one scale step of that channel (100/255 ~ 0.4).
    let err = max_abs_error(&tensor.data().expect("d"), &restored.data().expect("d"));
    assert!(
        err < 1.0,
        "per-channel parameters are not invertible: max error {err}"
    );

    // The legacy scalar API cannot carry these parameters; it reports only
    // channel 0's pair, which is documented and must stay documented.
    let (_, scale, zero_point) = quantize_with_config(&tensor, &config).expect("legacy");
    assert_eq!(scale, quantized.params.representative_scale());
    assert_eq!(zero_point, quantized.params.representative_zero_point());
}

fn grouped_matrix() -> Tensor {
    // 4 channels along axis 0, group size 2: group 0 spans [0, 2],
    // group 1 spans [0, 1000].
    Tensor::from_data(
        vec![0.0, 1.0, 0.0, 2.0, 0.0, 500.0, 0.0, 1000.0],
        vec![4, 2],
        DeviceType::Cpu,
    )
    .expect("tensor")
}

#[test]
fn f141_group_wise_params_are_invertible() {
    let tensor = grouped_matrix();
    let config = QuantConfig::group_wise(0, 2);

    let (quantized, scales, zero_points) =
        quantize_group_wise_full(&tensor, 0, 2, &config).expect("quantize");
    assert_eq!(scales.len(), 2, "one scale per group");
    assert_eq!(zero_points.len(), 2);

    let restored =
        dequantize_per_group(&quantized, 0, 2, &scales, &zero_points).expect("dequantize");
    let err = max_abs_error(&tensor.data().expect("d"), &restored.data().expect("d"));
    assert!(
        err < 10.0,
        "group-wise parameters are not invertible: max error {err}"
    );
}

#[test]
fn f141_group_wise_through_config_is_invertible() {
    let tensor = grouped_matrix();
    let config = QuantConfig::group_wise(0, 2);

    let quantized = quantize_with_config_full(&tensor, &config).expect("quantize");
    assert!(matches!(quantized.params, QParams::PerGroup { .. }));
    let restored = quantized.dequantize().expect("dequantize");
    let err = max_abs_error(&tensor.data().expect("d"), &restored.data().expect("d"));
    assert!(err < 10.0, "group-wise round-trip error {err}");
}

#[test]
fn f141_int4_per_channel_params_are_invertible() {
    let tensor = wide_range_matrix();
    let config = QuantConfig::int4();

    let (quantized, scales, zero_points) =
        quantize_int4_per_channel_full(&tensor, 0, &config).expect("quantize");
    assert_eq!(scales.len(), 2);

    let restored =
        dequantize_per_channel(&quantized, 0, &scales, &zero_points).expect("dequantize");
    let err = max_abs_error(&tensor.data().expect("d"), &restored.data().expect("d"));
    assert!(
        err < 15.0,
        "int4 per-channel parameters are not invertible: max error {err}"
    );

    // The legacy API averages the per-channel parameters and therefore cannot
    // invert the transform - pin that it is at least still self-consistent.
    let (_, avg_scale, _) = quantize_int4_per_channel(&tensor, 0, &config).expect("legacy");
    assert!((avg_scale - (scales[0] + scales[1]) / 2.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// Experimental-gated modules (`quantize`, `fake_quantize`, `post_training`)
// ---------------------------------------------------------------------------

#[cfg(feature = "experimental")]
mod experimental {
    use super::max_abs_error;
    use torsh_core::DType;
    use torsh_quantization::config::QScheme;
    use torsh_quantization::dequantize::dequantize_per_tensor_affine;
    use torsh_quantization::fake_quantize::fake_quantize_auto;
    use torsh_quantization::quantize::quantize_tensor_auto;
    use torsh_tensor::creation::tensor_1d;

    #[test]
    fn f139_symmetric_round_trip_covers_tensor_maximum() {
        let tensor = tensor_1d(&[-1.0, 0.0, 1.5, 3.0]).expect("tensor");
        let (quantized, scale, zero_point) =
            quantize_tensor_auto(&tensor, DType::I8, QScheme::PerTensorSymmetric)
                .expect("quantize");
        assert_eq!(zero_point, 0);

        let restored =
            dequantize_per_tensor_affine(&quantized, scale, zero_point).expect("dequantize");
        let err = max_abs_error(&tensor.data().expect("d"), &restored.data().expect("d"));
        assert!(err <= scale, "round-trip error {err} exceeds scale {scale}");
    }

    #[test]
    fn f139_fake_quantize_symmetric_covers_tensor_maximum() {
        let tensor = tensor_1d(&[-1.0, 0.0, 1.5, 3.0]).expect("tensor");
        let faked = fake_quantize_auto(&tensor, DType::I8, QScheme::PerTensorSymmetric)
            .expect("fake quantize");
        let err = max_abs_error(&tensor.data().expect("d"), &faked.data().expect("d"));
        assert!(err < 0.05, "fake-quantize symmetric error {err} too large");
    }

    /// The symmetric qparam helpers must reject non-finite input instead of
    /// emitting NaN codes (`f32::max` silently ignores NaN in a fold).
    #[test]
    fn f139_symmetric_rejects_non_finite_input() {
        use torsh_quantization::quantize::{
            calculate_per_channel_symmetric_qparams, calculate_symmetric_qparams,
        };

        let tensor = tensor_1d(&[1.0, f32::NAN, 3.0]).expect("tensor");
        assert!(calculate_symmetric_qparams(&tensor, -128, 127).is_err());
        assert!(calculate_per_channel_symmetric_qparams(&tensor, 0, DType::I8).is_err());
        assert!(
            quantize_tensor_auto(&tensor, DType::I8, QScheme::PerTensorSymmetric).is_err(),
            "symmetric quantization must not fabricate codes from NaN input"
        );
    }

    #[test]
    fn f140_u8_negative_dominant_range_is_supported() {
        let tensor = tensor_1d(&[-2.0, -1.0, 0.0, 0.5, 1.0]).expect("tensor");
        let (quantized, scale, zero_point) =
            quantize_tensor_auto(&tensor, DType::U8, QScheme::PerTensorAffine)
                .expect("U8 affine quantization must succeed");

        let codes = quantized.data().expect("data");
        assert!(codes.iter().all(|&c| (0.0..=255.0).contains(&c)));

        let restored =
            dequantize_per_tensor_affine(&quantized, scale, zero_point).expect("dequantize");
        let err = max_abs_error(&tensor.data().expect("d"), &restored.data().expect("d"));
        assert!(err <= scale, "round-trip error {err} exceeds scale {scale}");
    }

    #[test]
    fn f140_u8_uses_full_range() {
        let tensor = tensor_1d(&[0.0, 2.5, 5.0, 7.5, 10.0]).expect("tensor");
        let (quantized, _scale, _zp) =
            quantize_tensor_auto(&tensor, DType::U8, QScheme::PerTensorAffine).expect("quantize");
        let codes = quantized.data().expect("data");
        assert!(
            codes.iter().cloned().fold(0.0f32, f32::max) > 200.0,
            "U8 output saturated at the I8 bound"
        );
    }
}

#[cfg(feature = "experimental")]
mod ptq {
    use torsh_core::error::Result as TorshResult;
    use torsh_quantization::config::QuantConfig;
    use torsh_quantization::post_training::{
        calibrate_model, ActivationSink, CalibrationDataset, Module, PTQState,
    };
    use torsh_tensor::creation::tensor_1d;
    use torsh_tensor::Tensor;

    /// Two "layers": the second one scales activations by 100.
    struct TwoLayer {
        w1: Tensor,
        w2: Tensor,
    }

    impl Module for TwoLayer {
        fn forward(&self, input: &Tensor) -> TorshResult<Tensor> {
            let scaled: Vec<f32> = input.data()?.iter().map(|&x| x * 100.0).collect();
            Tensor::from_data(scaled, input.shape().dims().to_vec(), input.device())
        }

        fn forward_with_activations(
            &self,
            input: &Tensor,
            sink: &mut ActivationSink<'_>,
        ) -> TorshResult<Tensor> {
            // linear1 passes its input through; linear2 amplifies by 100.
            let first = input.clone();
            sink("linear1", &first)?;

            let second = self.forward(&first)?;
            sink("linear2", &second)?;

            Ok(second)
        }

        fn parameters(&self) -> Vec<&Tensor> {
            vec![&self.w1, &self.w2]
        }

        fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
            vec![&mut self.w1, &mut self.w2]
        }

        fn named_parameters(&self) -> Vec<(String, &Tensor)> {
            vec![
                ("linear1.weight".to_string(), &self.w1),
                ("linear2.weight".to_string(), &self.w2),
            ]
        }

        fn train(&mut self, _mode: bool) {}
    }

    #[test]
    fn f250_observers_receive_their_own_layer_activations() {
        let mut module = TwoLayer {
            w1: tensor_1d(&[1.0, 2.0]).expect("w1"),
            w2: tensor_1d(&[1.0, 2.0]).expect("w2"),
        };
        let mut state = PTQState::new(QuantConfig::int8());
        state.add_observer(
            "linear1".to_string(),
            torsh_quantization::observers::Observer::new(
                torsh_quantization::config::ObserverType::MinMax,
            ),
        );
        state.add_observer(
            "linear2".to_string(),
            torsh_quantization::observers::Observer::new(
                torsh_quantization::config::ObserverType::MinMax,
            ),
        );

        let dataset = CalibrationDataset::new(vec![tensor_1d(&[0.0, 1.0]).expect("s")]);
        calibrate_model(&mut module, &dataset, &mut state).expect("calibration");

        let p1 = state.get_layer_params("linear1").expect("linear1 params");
        let p2 = state.get_layer_params("linear2").expect("linear2 params");
        assert!(
            (p2.scales[0] / p1.scales[0]) > 50.0,
            "layer scales are identical ({} vs {}), observers were fed the \
             model input instead of their own activations",
            p1.scales[0],
            p2.scales[0]
        );
    }

    /// A model without the calibration hook must fail loudly instead of
    /// producing quantization parameters derived from the model input.
    struct NoHook {
        w: Tensor,
    }

    impl Module for NoHook {
        fn forward(&self, input: &Tensor) -> TorshResult<Tensor> {
            Ok(input.clone())
        }

        fn parameters(&self) -> Vec<&Tensor> {
            vec![&self.w]
        }

        fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
            vec![&mut self.w]
        }

        fn named_parameters(&self) -> Vec<(String, &Tensor)> {
            vec![("linear1.weight".to_string(), &self.w)]
        }

        fn train(&mut self, _mode: bool) {}
    }

    #[test]
    fn f250_calibration_without_hooks_reports_an_honest_error() {
        let mut module = NoHook {
            w: tensor_1d(&[1.0, 2.0]).expect("w"),
        };
        let mut state = PTQState::new(QuantConfig::int8());
        state.add_observer(
            "linear1".to_string(),
            torsh_quantization::observers::Observer::new(
                torsh_quantization::config::ObserverType::MinMax,
            ),
        );

        let dataset = CalibrationDataset::new(vec![tensor_1d(&[0.0, 1.0]).expect("s")]);
        let result = calibrate_model(&mut module, &dataset, &mut state);

        assert!(result.is_err(), "calibration must not fabricate qparams");
        assert!(
            state.get_layer_params("linear1").is_none(),
            "no quantization parameters may be published for an uncalibrated layer"
        );
    }
}
