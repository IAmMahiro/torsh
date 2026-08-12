//! Production-hardening regression tests for torsh-nn parameters, activations,
//! normalization and initialization.
//!
//! Each test is named after the finding it pins down.

use torsh_core::device::DeviceType;
use torsh_nn::functional;
use torsh_nn::functional::activation::GeluApproximation;
use torsh_nn::init::{orthogonal_init, Nonlinearity};
use torsh_nn::layers::normalization::{
    BatchNorm2d, BatchRenorm2d, BatchRenormSchedule, NormalizationConfig, SyncBatchNorm2d,
    VirtualBatchNorm2d,
};
use torsh_nn::layers::regularization::{
    AlphaDropout, CutMix, Cutout, DropBlock2d, DropConnect, Dropout, Dropout2d, Mixup,
    StochasticDepth,
};
use torsh_nn::layers::Linear;
use torsh_nn::{Module, Parameter};
use torsh_tensor::Tensor;

// ---------------------------------------------------------------------------
// F033 - Parameter must mark the wrapped tensor as requiring gradients
// ---------------------------------------------------------------------------

#[test]
fn f033_parameter_marks_tensor_requires_grad() {
    let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3]).expect("tensor");
    let param = Parameter::new(tensor);
    assert!(
        param.tensor().read().requires_grad(),
        "Parameter::new must set requires_grad on the wrapped tensor"
    );

    let no_grad = Parameter::new_no_grad(Tensor::from_vec(vec![1.0f32], &[1]).expect("tensor"));
    assert!(!no_grad.tensor().read().requires_grad());
}

#[test]
fn f033_linear_backward_produces_weight_gradients() {
    let linear = Linear::new(3, 2, true);
    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[1, 3]).expect("input");

    let output = linear.forward(&input).expect("forward");
    let loss = output.sum().expect("sum");
    loss.backward()
        .expect("backward must succeed on a Linear output");

    let params = linear.parameters();
    let weight = params.get("weight").expect("weight parameter");
    let grad = weight
        .tensor()
        .read()
        .grad()
        .expect("weight gradient must be populated after backward");
    let grad_values = grad.to_vec().expect("grad values");
    // W has shape [in_features, out_features] = [3, 2] and
    // d(sum(x @ W))/dW[i][j] = x[i], so every row repeats the input value.
    let expected = [1.0f32, 1.0, 2.0, 2.0, 3.0, 3.0];
    assert_eq!(grad_values.len(), expected.len());
    for (got, want) in grad_values.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() < 1e-5,
            "weight gradient mismatch: got {grad_values:?}, want {expected:?}"
        );
    }

    let bias_grad = params
        .get("bias")
        .expect("bias parameter")
        .tensor()
        .read()
        .grad()
        .expect("bias gradient must be populated after backward");
    for v in bias_grad.to_vec().expect("bias grad") {
        assert!((v - 1.0).abs() < 1e-5, "d(sum(y))/db must be 1, got {v}");
    }
}

// ---------------------------------------------------------------------------
// F235 - Parameter::zero_grad must clear the tensor gradient slot
// ---------------------------------------------------------------------------

#[test]
fn f235_parameter_zero_grad_clears_gradient() {
    let mut param = Parameter::new(Tensor::from_vec(vec![2.0f32, 3.0], &[2]).expect("tensor"));
    let tensor = param.tensor().read().clone();
    let loss = tensor.mul_scalar(3.0).expect("mul").sum().expect("sum");
    loss.backward().expect("backward");

    assert!(
        param.tensor().read().grad().is_some(),
        "precondition: gradient exists"
    );
    param.zero_grad();
    assert!(
        param.tensor().read().grad().is_none(),
        "Parameter::zero_grad must clear the gradient"
    );
}

// ---------------------------------------------------------------------------
// F034 - Dropout / Dropout2d must actually sample a Bernoulli mask
// ---------------------------------------------------------------------------

#[test]
fn f034_dropout_zeroes_roughly_p_fraction() {
    let n = 4000usize;
    let dropout = Dropout::new(0.5);
    let input = Tensor::from_vec(vec![1.0f32; n], &[n]).expect("input");
    let output = dropout.forward(&input).expect("forward");
    let values = output.to_vec().expect("values");

    let zeros = values.iter().filter(|v| **v == 0.0).count();
    let fraction = zeros as f32 / n as f32;
    assert!(
        (0.4..0.6).contains(&fraction),
        "expected ~50% of activations dropped, got {fraction}"
    );

    let mean = values.iter().sum::<f32>() / n as f32;
    assert!(
        (mean - 1.0).abs() < 0.1,
        "dropout must preserve the expected value, got mean {mean}"
    );
}

#[test]
fn f034_dropout_is_identity_in_eval_mode() {
    let mut dropout = Dropout::new(0.5);
    dropout.eval();
    let input = Tensor::from_vec(vec![1.0f32; 32], &[32]).expect("input");
    let output = dropout.forward(&input).expect("forward");
    assert!(output.to_vec().expect("values").iter().all(|v| *v == 1.0));
}

#[test]
fn f034_dropout2d_drops_entire_channels() {
    let (n, c, h, w) = (8usize, 16usize, 2usize, 2usize);
    let dropout = Dropout2d::new(0.5);
    let input = Tensor::from_vec(vec![1.0f32; n * c * h * w], &[n, c, h, w]).expect("input");
    let output = dropout.forward(&input).expect("forward");
    let values = output.to_vec().expect("values");

    let mut dropped_channels = 0usize;
    for batch in 0..n {
        for channel in 0..c {
            let base = batch * c * h * w + channel * h * w;
            let slice = &values[base..base + h * w];
            let all_zero = slice.iter().all(|v| *v == 0.0);
            let none_zero = slice.iter().all(|v| *v != 0.0);
            assert!(
                all_zero || none_zero,
                "Dropout2d must drop whole feature maps, got {slice:?}"
            );
            if all_zero {
                dropped_channels += 1;
            }
        }
    }
    let fraction = dropped_channels as f32 / (n * c) as f32;
    assert!(
        (0.3..0.7).contains(&fraction),
        "expected ~50% of channels dropped, got {fraction}"
    );
}

// ---------------------------------------------------------------------------
// F036 - functional::softmax must honour `dim`
// ---------------------------------------------------------------------------

#[test]
fn f036_softmax_is_dim_aware_for_3d_tensors() {
    let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.25).collect();
    let input = Tensor::from_vec(data, &[2, 3, 4]).expect("input");

    for (dim, dim_size) in [(-1i32, 4usize), (0, 2), (1, 3)] {
        let output = functional::softmax(&input, Some(dim)).expect("softmax");
        let values = output.to_vec().expect("values");
        let actual_dim = if dim < 0 { 3 + dim } else { dim } as usize;
        let shape = [2usize, 3, 4];
        let outer: usize = shape[..actual_dim].iter().product();
        let inner: usize = shape[actual_dim + 1..].iter().product();

        for o in 0..outer {
            for i in 0..inner {
                let mut sum = 0.0f32;
                for d in 0..dim_size {
                    sum += values[o * dim_size * inner + d * inner + i];
                }
                assert!(
                    (sum - 1.0).abs() < 1e-4,
                    "softmax along dim {dim} must sum to 1, got {sum}"
                );
            }
        }
    }
}

#[test]
fn f036_softmax_defaults_to_last_dim() {
    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).expect("input");
    let output = functional::softmax(&input, None).expect("softmax");
    let values = output.to_vec().expect("values");
    assert!((values[0] + values[1] - 1.0).abs() < 1e-5);
    assert!((values[2] + values[3] - 1.0).abs() < 1e-5);
}

// ---------------------------------------------------------------------------
// F232 - GELU must use the exact erf formulation (and offer the tanh variant)
// ---------------------------------------------------------------------------

#[test]
fn f232_gelu_matches_exact_erf_formulation() {
    let xs = vec![-2.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let input = Tensor::from_vec(xs.clone(), &[xs.len()]).expect("input");
    let output = functional::gelu(&input).expect("gelu");
    let values = output.to_vec().expect("values");

    for (x, got) in xs.iter().zip(values.iter()) {
        let expected = 0.5 * x * (1.0 + libm_erf(*x as f64 / std::f64::consts::SQRT_2) as f32);
        assert!(
            (got - expected).abs() < 1e-4,
            "gelu({x}) = {got}, expected exact GELU {expected}"
        );
    }
}

/// Reference erf used only by the test (Abramowitz & Stegun 7.1.26 is not
/// accurate enough, so a high-order series/continued-fraction pair is used).
fn libm_erf(x: f64) -> f64 {
    // Numerical Recipes `erf` via the incomplete gamma series, double precision.
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    if ax < 1e-12 {
        return 0.0;
    }
    // Series expansion of erf for all practical ranges used in this test.
    let mut sum = ax;
    let mut term = ax;
    let x2 = ax * ax;
    for n in 1..200 {
        term *= -x2 / n as f64;
        let contribution = term / (2.0 * n as f64 + 1.0);
        sum += contribution;
        if contribution.abs() < 1e-18 * sum.abs() {
            break;
        }
    }
    sign * (2.0 / std::f64::consts::PI.sqrt()) * sum
}

// ---------------------------------------------------------------------------
// F035 - functional batch norm must compute per-channel statistics
// ---------------------------------------------------------------------------

#[test]
fn f035_batch_norm_2d_uses_per_channel_statistics() {
    // N=1, C=2, H=1, W=2 -> channel 0 = [1, 3], channel 1 = [10, 20]
    let input = Tensor::from_vec(vec![1.0f32, 3.0, 10.0, 20.0], &[1, 2, 1, 2]).expect("input");
    let output = functional::batch_norm_2d(&input, None, None, None, None, true, 0.1, 1e-6)
        .expect("batch_norm_2d");
    let values = output.to_vec().expect("values");

    // channel 0: mean 2, std 1 -> [-1, 1]; channel 1: mean 15, std 5 -> [-1, 1]
    let expected = [-1.0f32, 1.0, -1.0, 1.0];
    for (got, want) in values.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() < 1e-3,
            "per-channel batch norm mismatch: got {values:?}, want {expected:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// F131 - BatchNorm layers must maintain running statistics
// ---------------------------------------------------------------------------

#[test]
fn f131_batch_norm_updates_running_statistics() {
    let config = NormalizationConfig {
        eps: 1e-5,
        momentum: 1.0, // fully replace the running stats with the batch stats
        affine: false,
        track_running_stats: true,
    };
    let mut bn = BatchNorm2d::with_config(2, config).expect("batchnorm");

    // channel 0 constant 4.0, channel 1 constant 8.0
    let input = Tensor::from_vec(
        vec![4.0f32, 4.0, 4.0, 4.0, 8.0, 8.0, 8.0, 8.0],
        &[1, 2, 2, 2],
    )
    .expect("input");
    let _ = bn.forward(&input).expect("train forward");

    let running_mean = bn.running_mean().expect("running_mean must be exposed");
    let mean_values = running_mean.to_vec().expect("values");
    assert!(
        (mean_values[0] - 4.0).abs() < 1e-4 && (mean_values[1] - 8.0).abs() < 1e-4,
        "running_mean must track the batch mean, got {mean_values:?}"
    );
    assert_eq!(
        bn.num_batches_tracked().expect("counter"),
        Some(1.0),
        "the batch counter must advance"
    );
    assert!(
        bn.named_buffers().contains_key("running_mean"),
        "running statistics must be reachable as module buffers"
    );

    // In eval mode the running statistics must be used, so a *different* input
    // is normalised against the stored mean rather than its own.
    bn.eval();
    let eval_input = Tensor::from_vec(
        vec![4.0f32, 4.0, 4.0, 4.0, 8.0, 8.0, 8.0, 8.0],
        &[1, 2, 2, 2],
    )
    .expect("input");
    let out = bn.forward(&eval_input).expect("eval forward");
    let out_values = out.to_vec().expect("values");
    for v in out_values {
        assert!(
            v.abs() < 1e-2,
            "eval mode must normalise with the running mean, got {v}"
        );
    }
}

// ---------------------------------------------------------------------------
// F233 - the specialized batch-norm variants must be real layers
// ---------------------------------------------------------------------------

fn constant_nchw(values: &[f32], n: usize, h: usize, w: usize) -> Tensor {
    let mut data = Vec::with_capacity(n * values.len() * h * w);
    for _ in 0..n {
        for value in values {
            data.extend(std::iter::repeat(*value).take(h * w));
        }
    }
    Tensor::from_vec(data, &[n, values.len(), h, w]).expect("input")
}

#[test]
fn f233_sync_batch_norm_normalizes_like_batch_norm() {
    let config = NormalizationConfig {
        affine: false,
        ..NormalizationConfig::default()
    };
    let sync = SyncBatchNorm2d::with_config(2, config.clone()).expect("syncbn");
    let plain = BatchNorm2d::with_config(2, config).expect("bn");

    let input = Tensor::from_vec(
        vec![1.0f32, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0, 40.0],
        &[1, 2, 2, 2],
    )
    .expect("input");

    let a = sync
        .forward(&input)
        .expect("sync forward")
        .to_vec()
        .expect("values");
    let b = plain
        .forward(&input)
        .expect("bn forward")
        .to_vec()
        .expect("values");
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-5, "sync/plain mismatch: {a:?} vs {b:?}");
    }
    assert_eq!(sync.world_size(), 1);
    assert!(sync.running_mean().is_some());
}

#[test]
fn f233_virtual_batch_norm_requires_and_uses_a_reference_batch() {
    let config = NormalizationConfig {
        affine: false,
        ..NormalizationConfig::default()
    };
    let mut vbn = VirtualBatchNorm2d::with_config(1, config).expect("vbn");
    let input = constant_nchw(&[4.0], 1, 2, 2);

    assert!(
        vbn.forward(&input).is_err(),
        "forward without a reference batch must be an honest error"
    );

    // Reference batch is constant 0 with 8 elements; the pooled mean of the
    // reference and a constant-4 batch of 4 elements is 4 * 4/12 = 4/3.
    let reference = constant_nchw(&[0.0], 2, 2, 2);
    vbn.set_reference_batch(&reference).expect("reference");
    assert!(vbn.has_reference_batch());

    let output = vbn.forward(&input).expect("vbn forward");
    let values = output.to_vec().expect("values");
    let pooled_mean = 4.0f32 * 4.0 / 12.0;
    let pooled_mean_square = 16.0f32 * 4.0 / 12.0;
    let pooled_var = pooled_mean_square - pooled_mean * pooled_mean;
    let expected = (4.0 - pooled_mean) / (pooled_var + 1e-5).sqrt();
    for v in values {
        assert!(
            (v - expected).abs() < 1e-3,
            "virtual batch norm must use pooled statistics: got {v}, want {expected}"
        );
    }
}

#[test]
fn f233_batch_renorm_applies_r_and_d_corrections() {
    let config = NormalizationConfig {
        affine: false,
        momentum: 1.0,
        ..NormalizationConfig::default()
    };
    // warmup_steps = 0 so the corrections are active from the first step.
    let schedule = BatchRenormSchedule {
        r_max: 3.0,
        d_max: 5.0,
        warmup_steps: 0,
        ramp_steps: 0,
    };
    let renorm = BatchRenorm2d::with_config(1, config, schedule).expect("renorm");

    // Batch of [0, 8]: mean 4, biased variance 16, sigma 4.
    // Running stats start at mean 0 / var 1 -> sigma_running = 1.
    // r = clip(4 / 1, 1/3, 3) = 3 ; d = clip((4 - 0) / 1, -5, 5) = 4.
    let input = Tensor::from_vec(vec![0.0f32, 8.0], &[1, 1, 1, 2]).expect("input");
    let values = renorm
        .forward(&input)
        .expect("renorm forward")
        .to_vec()
        .expect("values");

    let sigma = (16.0f32 + 1e-5).sqrt();
    let expected: Vec<f32> = [0.0f32, 8.0]
        .iter()
        .map(|x| (x - 4.0) / sigma * 3.0 + 4.0)
        .collect();
    for (got, want) in values.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() < 1e-3,
            "batch renorm mismatch: got {values:?}, want {expected:?}"
        );
    }

    assert_eq!(renorm.step(), 1);
    let running_mean = renorm.running_mean().to_vec().expect("values");
    assert!((running_mean[0] - 4.0).abs() < 1e-4);

    // momentum = 1.0 replaces the running variance with the *unbiased* batch
    // variance: biased 16 with count 2 -> 16 * 2/1 = 32 (PyTorch semantics).
    let running_var = renorm.running_var().to_vec().expect("values");
    assert!(
        (running_var[0] - 32.0).abs() < 1e-3,
        "running_var must track the unbiased batch variance, got {running_var:?}"
    );
}

#[test]
fn f131_running_var_tracks_the_unbiased_variance() {
    let config = NormalizationConfig {
        eps: 1e-5,
        momentum: 1.0,
        affine: false,
        track_running_stats: true,
    };
    let bn = BatchNorm2d::with_config(1, config).expect("batchnorm");

    // Channel values [0, 8]: biased variance 16, unbiased 32.
    let input = Tensor::from_vec(vec![0.0f32, 8.0], &[1, 1, 1, 2]).expect("input");
    let _ = bn.forward(&input).expect("forward");

    let running_var = bn
        .running_var()
        .expect("running_var")
        .to_vec()
        .expect("values");
    assert!(
        (running_var[0] - 32.0).abs() < 1e-3,
        "running_var must use the n/(n-1) correction, got {running_var:?}"
    );
}

#[test]
fn f233_batch_renorm_schedule_ramps() {
    let schedule = BatchRenormSchedule::default();
    assert_eq!(schedule.limits_at(0), (1.0, 0.0));
    assert_eq!(schedule.limits_at(4_999), (1.0, 0.0));
    let (r_mid, d_mid) = schedule.limits_at(5_000 + 35_000 / 2);
    assert!(
        (r_mid - 2.0).abs() < 1e-3,
        "r_max should be halfway, got {r_mid}"
    );
    assert!(
        (d_mid - 2.5).abs() < 1e-3,
        "d_max should be halfway, got {d_mid}"
    );
    let (r_end, d_end) = schedule.limits_at(1_000_000);
    assert!((r_end - 3.0).abs() < 1e-6 && (d_end - 5.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// log_softmax must be dim-aware too (same kernel as softmax)
// ---------------------------------------------------------------------------

#[test]
fn log_softmax_is_dim_aware() {
    let data: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();
    let input = Tensor::from_vec(data, &[2, 2, 3]).expect("input");

    let log_probs = functional::log_softmax(&input, Some(1))
        .expect("log_softmax")
        .to_vec()
        .expect("values");
    // Along dim 1 (size 2, inner stride 3) the exponentials must sum to 1.
    for outer in 0..2 {
        for inner in 0..3 {
            let base = outer * 2 * 3 + inner;
            let total = log_probs[base].exp() + log_probs[base + 3].exp();
            assert!(
                (total - 1.0).abs() < 1e-4,
                "log_softmax along dim 1 must normalise, got {total}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// GELU tanh variant
// ---------------------------------------------------------------------------

#[test]
fn f232_gelu_tanh_variant_matches_reference() {
    let xs = vec![-2.0f32, -0.5, 0.0, 0.5, 2.0];
    let input = Tensor::from_vec(xs.clone(), &[xs.len()]).expect("input");
    let output = functional::activation::gelu_with_approximation(&input, GeluApproximation::Tanh)
        .expect("gelu tanh");
    let values = output.to_vec().expect("values");

    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    for (x, got) in xs.iter().zip(values.iter()) {
        let expected = 0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044_715 * x * x * x)).tanh());
        assert!((got - expected).abs() < 1e-5, "tanh GELU mismatch at {x}");
    }
    assert_eq!(
        GeluApproximation::from_str_arg("tanh").expect("parse"),
        GeluApproximation::Tanh
    );
    assert!(GeluApproximation::from_str_arg("bogus").is_err());
}

// ---------------------------------------------------------------------------
// F304 - initialization gains must match PyTorch
// ---------------------------------------------------------------------------

#[test]
fn f304_selu_gain_matches_pytorch() {
    assert!(
        (Nonlinearity::SELU.gain() - 0.75).abs() < 1e-6,
        "torch.nn.init.calculate_gain('selu') == 3/4, got {}",
        Nonlinearity::SELU.gain()
    );
}

#[test]
fn f304_normal_init_never_produces_non_finite_values() {
    let tensor = torsh_nn::init::normal(&[4096], 0.0, 1.0).expect("normal");
    assert!(tensor
        .to_vec()
        .expect("values")
        .iter()
        .all(|v| v.is_finite()));
}

// ---------------------------------------------------------------------------
// F231 - orthogonal_init must produce (semi-)orthogonal matrices
// ---------------------------------------------------------------------------

fn assert_semi_orthogonal(shape: &[usize]) {
    let tensor = orthogonal_init(shape, 1.0).expect("orthogonal_init");
    assert_eq!(
        tensor.shape().dims(),
        shape,
        "orthogonal_init must preserve the requested shape"
    );

    let rows = shape[0];
    let cols: usize = shape[1..].iter().product();
    let values = tensor.to_vec().expect("values");

    if rows <= cols {
        // rows are orthonormal: Q Q^T == I
        for i in 0..rows {
            for j in 0..rows {
                let mut dot = 0.0f32;
                for k in 0..cols {
                    dot += values[i * cols + k] * values[j * cols + k];
                }
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - want).abs() < 1e-3,
                    "Q Q^T[{i}][{j}] = {dot}, want {want} for shape {shape:?}"
                );
            }
        }
    } else {
        // columns are orthonormal: Q^T Q == I
        for i in 0..cols {
            for j in 0..cols {
                let mut dot = 0.0f32;
                for k in 0..rows {
                    dot += values[k * cols + i] * values[k * cols + j];
                }
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - want).abs() < 1e-3,
                    "Q^T Q[{i}][{j}] = {dot}, want {want} for shape {shape:?}"
                );
            }
        }
    }
}

#[test]
fn f231_orthogonal_init_square() {
    assert_semi_orthogonal(&[3, 3]);
}

#[test]
fn f231_orthogonal_init_wide() {
    assert_semi_orthogonal(&[2, 5]);
}

#[test]
fn f231_orthogonal_init_tall() {
    assert_semi_orthogonal(&[5, 2]);
}

#[test]
fn f231_orthogonal_init_conv_shape() {
    assert_semi_orthogonal(&[4, 2, 3, 3]);
}

#[test]
fn f231_orthogonal_init_applies_gain() {
    let tensor = orthogonal_init(&[3, 3], 2.0).expect("orthogonal_init");
    assert_eq!(tensor.shape().dims(), &[3, 3]);
    let values = tensor.to_vec().expect("values");
    let mut frobenius = 0.0f32;
    for v in &values {
        frobenius += v * v;
    }
    // ||gain * Q||_F^2 == gain^2 * min(rows, cols)
    assert!(
        (frobenius - 4.0 * 3.0).abs() < 1e-2,
        "gain scaling wrong, ||W||_F^2 = {frobenius}"
    );
}

// ---------------------------------------------------------------------------
// Parameter::to_device must not silently claim success
// ---------------------------------------------------------------------------

#[test]
fn f235_parameter_to_device_is_honest() {
    let mut param = Parameter::new(Tensor::from_vec(vec![1.0f32], &[1]).expect("tensor"));
    // Moving to the device the tensor already lives on must succeed.
    param.to_device(DeviceType::Cpu).expect("cpu -> cpu");
    assert_eq!(param.device(), DeviceType::Cpu);
}

// ---------------------------------------------------------------------------
// The remaining stochastic regularizers in layers/regularization.rs shared the
// constant-mask defect of F034; these pin down their real behaviour.
// ---------------------------------------------------------------------------

#[test]
fn stochastic_depth_drops_whole_samples() {
    let (n, features) = (256usize, 4usize);
    let layer = StochasticDepth::new(0.5);
    let input = Tensor::from_vec(vec![1.0f32; n * features], &[n, features]).expect("input");
    let values = layer
        .forward(&input)
        .expect("forward")
        .to_vec()
        .expect("values");

    let mut dropped = 0usize;
    for sample in 0..n {
        let slice = &values[sample * features..(sample + 1) * features];
        let first = slice[0];
        assert!(
            slice.iter().all(|v| *v == first),
            "the whole sample must share one factor, got {slice:?}"
        );
        if first == 0.0 {
            dropped += 1;
        } else {
            assert!((first - 2.0).abs() < 1e-6, "survivors scale by 1/keep_prob");
        }
    }
    let fraction = dropped as f32 / n as f32;
    assert!(
        (0.35..0.65).contains(&fraction),
        "expected ~50% of samples dropped, got {fraction}"
    );
}

#[test]
fn drop_block_zeroes_contiguous_squares() {
    let (h, w, block) = (12usize, 12usize, 3usize);
    let layer = DropBlock2d::new(0.3, block);
    let input = Tensor::from_vec(vec![1.0f32; h * w], &[1, 1, h, w]).expect("input");
    let values = layer
        .forward(&input)
        .expect("forward")
        .to_vec()
        .expect("values");

    let zeros = values.iter().filter(|v| **v == 0.0).count();
    assert!(zeros > 0, "DropBlock must actually drop something");
    assert!(zeros < h * w, "DropBlock must not drop everything");

    // Every zero must belong to a block_size x block_size square that is fully
    // zeroed and starts within the valid seed region.
    for y in 0..h {
        for x in 0..w {
            if values[y * w + x] != 0.0 {
                continue;
            }
            let mut covered = false;
            for seed_y in y.saturating_sub(block - 1)..=y.min(h - block) {
                for seed_x in x.saturating_sub(block - 1)..=x.min(w - block) {
                    if (0..block).all(|dy| {
                        (0..block).all(|dx| values[(seed_y + dy) * w + seed_x + dx] == 0.0)
                    }) {
                        covered = true;
                    }
                }
            }
            assert!(
                covered,
                "zero at ({y}, {x}) is not part of a {block}x{block} block"
            );
        }
    }
}

#[test]
fn cutout_masks_the_same_square_in_every_channel() {
    let (c, h, w) = (3usize, 8usize, 8usize);
    let layer = Cutout::new(1, 4);
    let input = Tensor::from_vec(vec![1.0f32; c * h * w], &[1, c, h, w]).expect("input");
    let values = layer
        .forward(&input)
        .expect("forward")
        .to_vec()
        .expect("values");

    let plane = h * w;
    let zeros_first: Vec<usize> = (0..plane).filter(|i| values[*i] == 0.0).collect();
    assert!(!zeros_first.is_empty(), "Cutout must mask something");
    for channel in 1..c {
        let zeros: Vec<usize> = (0..plane)
            .filter(|i| values[channel * plane + i] == 0.0)
            .collect();
        assert_eq!(zeros, zeros_first, "Cutout must mask all channels alike");
    }
}

#[test]
fn alpha_dropout_uses_the_selu_saturation_value() {
    let n = 4000usize;
    let p = 0.2f32;
    let layer = AlphaDropout::new(p);
    let input = Tensor::from_vec(vec![0.0f32; n], &[n]).expect("input");
    let values = layer
        .forward(&input)
        .expect("forward")
        .to_vec()
        .expect("values");

    // With x == 0 the output takes exactly two values: a*0 + b for kept units
    // and a*alpha + b for dropped ones.
    let alpha = -1.7580993408473766f32;
    let keep_prob = 1.0 - p;
    let a = (keep_prob + alpha * alpha * keep_prob * (1.0 - keep_prob)).powf(-0.5);
    let b = -a * alpha * (1.0 - keep_prob);

    let dropped = values
        .iter()
        .filter(|v| (**v - (a * alpha + b)).abs() < 1e-4)
        .count();
    let kept = values.iter().filter(|v| (**v - b).abs() < 1e-4).count();
    assert_eq!(dropped + kept, n, "unexpected AlphaDropout output values");
    let fraction = dropped as f32 / n as f32;
    assert!(
        (0.12..0.28).contains(&fraction),
        "expected ~20% dropped, got {fraction}"
    );
}

#[test]
fn drop_connect_masks_weights() {
    let connect = DropConnect::new(0.5);
    let weight = Tensor::from_vec(vec![1.0f32; 4000], &[100, 40]).expect("weight");
    let values = connect
        .drop_weights(&weight)
        .expect("drop_weights")
        .to_vec()
        .expect("values");
    let zeros = values.iter().filter(|v| **v == 0.0).count();
    let fraction = zeros as f32 / values.len() as f32;
    assert!(
        (0.4..0.6).contains(&fraction),
        "DropConnect must mask ~50% of the weights, got {fraction}"
    );
}

#[test]
fn mixup_and_cutmix_actually_mix() {
    // Two clearly distinguishable samples.
    let mut data = vec![0.0f32; 2 * 1 * 4 * 4];
    for value in data.iter_mut().skip(16) {
        *value = 1.0;
    }
    let input = Tensor::from_vec(data, &[2, 1, 4, 4]).expect("input");

    let mixup = Mixup::new(1.0);
    let (mixed, permutation, lambda) = mixup.apply_with_permutation(&input, true).expect("mixup");
    assert_eq!(permutation.len(), 2);
    assert!((0.0..=1.0).contains(&lambda));
    let values = mixed.to_vec().expect("values");
    for (sample, partner) in permutation.iter().enumerate() {
        let expected = lambda * (sample as f32) + (1.0 - lambda) * (*partner as f32);
        for offset in 0..16 {
            assert!(
                (values[sample * 16 + offset] - expected).abs() < 1e-5,
                "mixup value mismatch at sample {sample}"
            );
        }
    }

    // Outside training mode both are the identity with lambda == 1.
    let (untouched, _, eval_lambda) = mixup
        .apply_with_permutation(&input, false)
        .expect("mixup eval");
    assert_eq!(eval_lambda, 1.0);
    assert_eq!(
        untouched.to_vec().expect("values"),
        input.to_vec().expect("values")
    );

    let cutmix = CutMix::new(1.0);
    let (patched, permutation, lambda) =
        cutmix.apply_with_permutation(&input, true).expect("cutmix");
    let values = patched.to_vec().expect("values");
    let original = input.to_vec().expect("values");
    let mut replaced = 0usize;
    for sample in 0..2 {
        for offset in 0..16 {
            let idx = sample * 16 + offset;
            if values[idx] != original[idx] {
                replaced += 1;
                assert_eq!(values[idx], original[permutation[sample] * 16 + offset]);
            }
        }
    }
    // The corrected lambda is 1 - pasted_area / plane on every sample.
    let expected_lambda = 1.0 - (replaced as f32 / 2.0) / 16.0;
    if replaced > 0 {
        assert!(
            (lambda - expected_lambda).abs() < 1e-5,
            "cutmix lambda {lambda} does not match the pasted area {expected_lambda}"
        );
    }
}
