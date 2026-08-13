//! Hardening regression tests for element-wise math ops, in-place activations
//! and convolutions.
//!
//! Each test pins down a behaviour touched by the wave-1 hardening pass:
//! zero-copy operand access for binary ops (F067), in-place mutation of
//! large / copy-on-write storage (F069), single-lock in-place activations
//! (F164), honest SIMD dispatch for f32 *and* f64 (F166/F263), uninitialised
//! pooled output buffers (F170), single-allocation SIMD activations (F270) and
//! hoisted storage access in the convolution loops (F070).

use approx::assert_relative_eq;
use torsh_core::device::DeviceType;
use torsh_tensor::Tensor;

fn t32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
    Tensor::from_data(data, shape, DeviceType::Cpu).expect("f32 tensor creation should succeed")
}

fn t64(data: Vec<f64>, shape: Vec<usize>) -> Tensor<f64> {
    Tensor::from_data(data, shape, DeviceType::Cpu).expect("f64 tensor creation should succeed")
}

/// 4096 f32 elements: above the SIMD fast-path threshold *and* above the
/// 10 KB `SimdOptimized` storage threshold.
fn ramp32(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32) * 0.25 - 8.0).collect()
}

/// 2048 f64 elements: above the 10 KB `SimdOptimized` threshold, and (before
/// the f64 SIMD dispatch landed) served only by the scalar path.
fn ramp64(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64) * 0.25 - 8.0).collect()
}

// ---------------------------------------------------------------- F067 ------

/// F067: a binary op must read both operands without deep-copying them, and it
/// must still produce exactly the scalar result.
#[test]
fn f067_binary_ops_match_scalar_reference_f32() {
    let n = 4096;
    let a = ramp32(n);
    let b: Vec<f32> = (0..n).map(|i| 1.0 + (i % 7) as f32).collect();
    let ta = t32(a.clone(), vec![n]);
    let tb = t32(b.clone(), vec![n]);

    let add = ta.add(&tb).expect("add should succeed");
    let sub = ta.sub(&tb).expect("sub should succeed");
    let mul = ta.mul(&tb).expect("mul should succeed");
    let div = ta.div(&tb).expect("div should succeed");

    let (add_v, sub_v) = (add.to_vec().expect("to_vec"), sub.to_vec().expect("to_vec"));
    let (mul_v, div_v) = (mul.to_vec().expect("to_vec"), div.to_vec().expect("to_vec"));
    for i in 0..n {
        assert_relative_eq!(add_v[i], a[i] + b[i], epsilon = 1e-6);
        assert_relative_eq!(sub_v[i], a[i] - b[i], epsilon = 1e-6);
        assert_relative_eq!(mul_v[i], a[i] * b[i], epsilon = 1e-5);
        assert_relative_eq!(div_v[i], a[i] / b[i], epsilon = 1e-5);
    }
}

/// F067 hazard: zero-copy operand access must not dead-lock when both operands
/// share one storage buffer (`t.add(&t)` and `t.add(&t.clone())`).
#[test]
fn f067_aliased_operands_do_not_deadlock() {
    let n = 4096;
    let a = ramp32(n);
    let ta = t32(a.clone(), vec![n]);

    let self_add = ta
        .add(&ta)
        .expect("add with aliased operand should succeed");
    let clone_add = ta
        .add(&ta.clone())
        .expect("add with cloned (storage-sharing) operand should succeed");

    let self_v = self_add.to_vec().expect("to_vec");
    let clone_v = clone_add.to_vec().expect("to_vec");
    for i in 0..n {
        assert_relative_eq!(self_v[i], a[i] + a[i], epsilon = 1e-6);
        assert_relative_eq!(clone_v[i], a[i] + a[i], epsilon = 1e-6);
    }
}

/// F067: strided views must keep working — a zero-copy fast path must not hand
/// the base buffer to a row-major loop.
#[test]
fn f067_binary_op_on_strided_view_reads_view_order() {
    let base = t32((0..12).map(|v| v as f32).collect(), vec![3, 4]);
    let view = base.transpose(0, 1).expect("transpose should succeed");
    let ones = t32(vec![1.0; 12], vec![4, 3]);

    let sum = view
        .add(&ones)
        .expect("add on a transposed view should succeed");
    assert_eq!(sum.shape().dims(), &[4, 3]);
    let expected: Vec<f32> = view
        .to_vec()
        .expect("to_vec")
        .into_iter()
        .map(|v| v + 1.0)
        .collect();
    assert_eq!(sum.to_vec().expect("to_vec"), expected);
}

// -------------------------------------------------------- F166 / F263 -------

/// F166/F263: f64 tensors above the dispatch threshold must go through a real
/// vectorised kernel and still match the scalar reference exactly.
#[test]
fn f166_f64_binary_ops_match_scalar_reference() {
    let n = 2048;
    let a = ramp64(n);
    let b: Vec<f64> = (0..n).map(|i| 1.0 + (i % 5) as f64).collect();
    let ta = t64(a.clone(), vec![n]);
    let tb = t64(b.clone(), vec![n]);

    let add = ta.add(&tb).expect("f64 add should succeed");
    let sub = ta.sub(&tb).expect("f64 sub should succeed");
    let mul = ta.mul(&tb).expect("f64 mul should succeed");
    let div = ta.div(&tb).expect("f64 div should succeed");

    let add_v = add.to_vec().expect("to_vec");
    let sub_v = sub.to_vec().expect("to_vec");
    let mul_v = mul.to_vec().expect("to_vec");
    let div_v = div.to_vec().expect("to_vec");
    for i in 0..n {
        assert_relative_eq!(add_v[i], a[i] + b[i], epsilon = 1e-12);
        assert_relative_eq!(sub_v[i], a[i] - b[i], epsilon = 1e-12);
        assert_relative_eq!(mul_v[i], a[i] * b[i], epsilon = 1e-12);
        assert_relative_eq!(div_v[i], a[i] / b[i], epsilon = 1e-12);
    }
}

/// F166: an integer element type has no vector kernel — it must still take the
/// generic path and produce the right answer.
///
/// `n` is deliberately larger than the largest chunk the element-wise chunk
/// config hands out (8192) and not a multiple of it, so the parallel path runs
/// several chunks with a ragged tail.
#[test]
fn f166_integer_elementwise_still_correct() {
    let n = 20_001;
    let a: Vec<i32> = (0..n).map(|i| i as i32 - 512).collect();
    let b: Vec<i32> = (0..n).map(|i| (i % 9) as i32 + 1).collect();
    let ta =
        Tensor::from_data(a.clone(), vec![n], DeviceType::Cpu).expect("i32 tensor creation failed");
    let tb =
        Tensor::from_data(b.clone(), vec![n], DeviceType::Cpu).expect("i32 tensor creation failed");

    let sum = ta
        .add(&tb)
        .expect("i32 add should succeed")
        .to_vec()
        .expect("to_vec");
    let prod = ta
        .mul(&tb)
        .expect("i32 mul should succeed")
        .to_vec()
        .expect("to_vec");
    for i in 0..n {
        assert_eq!(sum[i], a[i] + b[i]);
        assert_eq!(prod[i], a[i] * b[i]);
    }
}

// ---------------------------------------------------------------- F170 ------

/// F170: the pooled output buffer is handed to the kernel uninitialised, so
/// every element must be defined by the kernel itself. Running many ops in a
/// row recycles pool buffers and would expose any element the kernel misses.
#[test]
fn f170_pooled_output_is_fully_written() {
    let n = 1031; // deliberately not a multiple of any SIMD lane width
    let a = ramp32(n);
    let b: Vec<f32> = (0..n).map(|i| 2.0 + (i % 3) as f32).collect();
    let ta = t32(a.clone(), vec![n]);
    let tb = t32(b.clone(), vec![n]);

    for _ in 0..8 {
        let out = ta
            .add(&tb)
            .expect("add should succeed")
            .to_vec()
            .expect("to_vec");
        for i in 0..n {
            assert_relative_eq!(out[i], a[i] + b[i], epsilon = 1e-6);
        }
        let out = ta
            .div(&tb)
            .expect("div should succeed")
            .to_vec()
            .expect("to_vec");
        for i in 0..n {
            assert_relative_eq!(out[i], a[i] / b[i], epsilon = 1e-5);
        }
    }
}

// -------------------------------------------------------- F069 / F164 -------

/// F069: every in-place activation must work on tensors above the 10 KB
/// `SimdOptimized` storage threshold (f32 ≥ 2560, f64 ≥ 1280 elements).
#[test]
fn f069_inplace_activations_work_on_large_tensors() {
    let n = 4096;
    let src = ramp32(n);

    let mut relu = t32(src.clone(), vec![n]);
    relu.relu_().expect("relu_ must work on a 16 KB tensor");
    let mut sigmoid = t32(src.clone(), vec![n]);
    sigmoid
        .sigmoid_()
        .expect("sigmoid_ must work on a 16 KB tensor");
    let mut tanh = t32(src.clone(), vec![n]);
    tanh.tanh_().expect("tanh_ must work on a 16 KB tensor");
    let mut gelu = t32(src.clone(), vec![n]);
    gelu.gelu_().expect("gelu_ must work on a 16 KB tensor");
    let mut leaky = t32(src.clone(), vec![n]);
    leaky
        .leaky_relu_(0.01)
        .expect("leaky_relu_ must work on a 16 KB tensor");
    let mut clamp = t32(src.clone(), vec![n]);
    clamp
        .clamp_(-1.0, 1.0)
        .expect("clamp_ must work on a 16 KB tensor");

    let relu_v = relu.to_vec().expect("to_vec");
    let sigmoid_v = sigmoid.to_vec().expect("to_vec");
    let tanh_v = tanh.to_vec().expect("to_vec");
    let clamp_v = clamp.to_vec().expect("to_vec");
    let leaky_v = leaky.to_vec().expect("to_vec");
    for i in 0..n {
        let x = src[i];
        assert_relative_eq!(relu_v[i], if x > 0.0 { x } else { 0.0 }, epsilon = 1e-6);
        assert_relative_eq!(sigmoid_v[i], 1.0 / (1.0 + (-x).exp()), epsilon = 1e-6);
        assert_relative_eq!(tanh_v[i], x.tanh(), epsilon = 1e-6);
        assert_relative_eq!(clamp_v[i], x.clamp(-1.0, 1.0), epsilon = 1e-6);
        assert_relative_eq!(
            leaky_v[i],
            if x < 0.0 { 0.01 * x } else { x },
            epsilon = 1e-6
        );
    }
}

/// F069: the same for f64, whose 10 KB threshold is crossed at 1280 elements
/// and which has no f32 fast path to hide behind.
#[test]
fn f069_inplace_activations_work_on_large_f64_tensors() {
    let n = 2048;
    let src = ramp64(n);

    let mut sigmoid = t64(src.clone(), vec![n]);
    sigmoid
        .sigmoid_()
        .expect("f64 sigmoid_ must work on a 16 KB tensor");
    let mut tanh = t64(src.clone(), vec![n]);
    tanh.tanh_().expect("f64 tanh_ must work on a 16 KB tensor");
    let mut gelu = t64(src.clone(), vec![n]);
    gelu.gelu_().expect("f64 gelu_ must work on a 16 KB tensor");
    let mut relu = t64(src.clone(), vec![n]);
    relu.relu_().expect("f64 relu_ must work on a 16 KB tensor");
    let mut clamp = t64(src.clone(), vec![n]);
    clamp
        .clamp_(-1.0, 1.0)
        .expect("f64 clamp_ must work on a 16 KB tensor");

    let sigmoid_v = sigmoid.to_vec().expect("to_vec");
    let tanh_v = tanh.to_vec().expect("to_vec");
    let relu_v = relu.to_vec().expect("to_vec");
    let clamp_v = clamp.to_vec().expect("to_vec");
    for i in 0..n {
        let x = src[i];
        assert_relative_eq!(sigmoid_v[i], 1.0 / (1.0 + (-x).exp()), epsilon = 1e-12);
        assert_relative_eq!(tanh_v[i], x.tanh(), epsilon = 1e-12);
        assert_relative_eq!(relu_v[i], if x > 0.0 { x } else { 0.0 }, epsilon = 1e-12);
        assert_relative_eq!(clamp_v[i], x.clamp(-1.0, 1.0), epsilon = 1e-12);
    }
}

/// F069/F164: an in-place activation must not write through storage that is
/// shared with a clone — copy-on-write has to trigger first.
#[test]
fn f069_inplace_activations_respect_copy_on_write() {
    // Small enough to take the scalar tail of every activation.
    let n = 64;
    let src = ramp32(n);

    macro_rules! check_cow {
        ($name:expr, $apply:expr) => {{
            let mut owner = t32(src.clone(), vec![n]);
            let observer = owner.clone();
            let apply: fn(&mut Tensor<f32>) = $apply;
            apply(&mut owner);
            assert_eq!(
                observer.to_vec().expect("to_vec"),
                src,
                "{} mutated a tensor that only shares storage with the target",
                $name
            );
        }};
    }

    check_cow!("relu_", |t| {
        t.relu_().expect("relu_ should succeed");
    });
    check_cow!("sigmoid_", |t| {
        t.sigmoid_().expect("sigmoid_ should succeed");
    });
    check_cow!("tanh_", |t| {
        t.tanh_().expect("tanh_ should succeed");
    });
    check_cow!("gelu_", |t| {
        t.gelu_().expect("gelu_ should succeed");
    });
    check_cow!("leaky_relu_", |t| {
        t.leaky_relu_(0.01).expect("leaky_relu_ should succeed");
    });
    check_cow!("clamp_", |t| {
        t.clamp_(-1.0, 1.0).expect("clamp_ should succeed");
    });
}

/// F164/F069: an in-place activation on a strided view must touch exactly the
/// view's elements, in view order, and must not write through to the base.
///
/// The previous scalar tails iterated raw storage indices `0..storage.len()`,
/// ignoring strides and offsets entirely, so an activation on a transposed view
/// rewrote the *whole* base buffer. Routing through `apply_` makes every
/// activation behave like `relu_`'s SIMD fast path and like `add_`: the view is
/// materialised in view order and detached first.
#[test]
fn f164_inplace_activation_on_a_view_stays_inside_the_view() {
    let base = t32((0..16).map(|v| v as f32 - 8.0).collect(), vec![4, 4]);
    let untouched = base.to_vec().expect("to_vec");

    let mut transposed = base.transpose(0, 1).expect("transpose should succeed");
    let view_before = transposed.to_vec().expect("to_vec");
    transposed
        .tanh_()
        .expect("tanh_ on a transposed view should succeed");

    assert_eq!(
        base.to_vec().expect("to_vec"),
        untouched,
        "an in-place activation on a view must not rewrite the base buffer"
    );
    let got = transposed.to_vec().expect("to_vec");
    assert_eq!(got.len(), view_before.len());
    for (g, b) in got.iter().zip(view_before.iter()) {
        assert_relative_eq!(*g, b.tanh(), epsilon = 1e-6);
    }
}

/// F164: the small-tensor scalar tails must produce the same values as the
/// large-tensor SIMD paths (no behavioural fork between the two).
#[test]
fn f164_small_and_large_activation_paths_agree() {
    let small = 64;
    let src = ramp32(small);

    let mut a = t32(src.clone(), vec![small]);
    a.relu_().expect("relu_ should succeed");
    let mut b = t32(src.clone(), vec![small]);
    b.leaky_relu_(0.1).expect("leaky_relu_ should succeed");
    let mut c = t32(src.clone(), vec![small]);
    c.clamp_(-2.0, 2.0).expect("clamp_ should succeed");

    let a_v = a.to_vec().expect("to_vec");
    let b_v = b.to_vec().expect("to_vec");
    let c_v = c.to_vec().expect("to_vec");
    for i in 0..small {
        let x = src[i];
        assert_relative_eq!(a_v[i], if x > 0.0 { x } else { 0.0 }, epsilon = 1e-6);
        assert_relative_eq!(b_v[i], if x < 0.0 { 0.1 * x } else { x }, epsilon = 1e-6);
        assert_relative_eq!(c_v[i], x.clamp(-2.0, 2.0), epsilon = 1e-6);
    }
}

/// F164: NaN must pass through the in-place activations unchanged (PyTorch
/// semantics), on both the scalar tail and the vectorised path.
#[test]
fn f164_inplace_activations_pass_nan_through() {
    for &n in &[8_usize, 4096] {
        let mut data = ramp32(n);
        data[1] = f32::NAN;
        let mut relu = t32(data.clone(), vec![n]);
        relu.relu_().expect("relu_ should succeed");
        assert!(
            relu.to_vec().expect("to_vec")[1].is_nan(),
            "relu_ must pass NaN through (n = {n})"
        );

        let mut clamp = t32(data.clone(), vec![n]);
        clamp.clamp_(-1.0, 1.0).expect("clamp_ should succeed");
        assert!(
            clamp.to_vec().expect("to_vec")[1].is_nan(),
            "clamp_ must pass NaN through (n = {n})"
        );
    }
}

/// In-place activations stay forbidden on gradient-tracking tensors.
#[test]
fn inplace_activations_reject_requires_grad() {
    let mut t = t32(ramp32(4096), vec![4096]).requires_grad_(true);
    assert!(t.relu_().is_err(), "relu_ on requires_grad must error");
    assert!(
        t.sigmoid_().is_err(),
        "sigmoid_ on requires_grad must error"
    );
    assert!(t.tanh_().is_err(), "tanh_ on requires_grad must error");
    assert!(t.gelu_().is_err(), "gelu_ on requires_grad must error");
    assert!(
        t.clamp_(-1.0, 1.0).is_err(),
        "clamp_ on requires_grad must error"
    );
}

// ---------------------------------------------------------------- F270 ------

/// F270: the SIMD activations must return exactly the scalar result while
/// allocating a single output buffer.
#[test]
fn f270_simd_activations_match_scalar_reference() {
    let n = 4096;
    let src = ramp32(n);
    let t = t32(src.clone(), vec![n]);

    let relu = t
        .relu()
        .expect("relu should succeed")
        .to_vec()
        .expect("to_vec");
    let sigmoid = t
        .sigmoid()
        .expect("sigmoid should succeed")
        .to_vec()
        .expect("to_vec");
    for i in 0..n {
        let x = src[i];
        assert_relative_eq!(relu[i], if x > 0.0 { x } else { 0.0 }, epsilon = 1e-6);
        assert_relative_eq!(sigmoid[i], 1.0 / (1.0 + (-x).exp()), epsilon = 1e-5);
    }
    // The source tensor must be untouched by the out-of-place activations.
    assert_eq!(t.to_vec().expect("to_vec"), src);
}

// ---------------------------------------------------------------- F070 ------

/// F070: conv1d must keep producing the reference result after the storage
/// access is hoisted out of the accumulation loop.
#[test]
fn f070_conv1d_matches_reference() {
    let input = t32((1..=10).map(|v| v as f32).collect(), vec![1, 2, 5]);
    let weight = t32(
        vec![
            1.0, 0.0, -1.0, 0.5, 0.5, 0.5, -1.0, 2.0, -1.0, 1.0, 1.0, 1.0,
        ],
        vec![2, 2, 3],
    );
    let bias = t32(vec![0.5, -0.5], vec![2]);

    let out = input
        .conv1d(&weight, Some(&bias), 1, 1, 1, 1)
        .expect("conv1d should succeed");
    assert_eq!(out.shape().dims(), &[1, 2, 5]);

    let expected = reference_conv1d(
        &input.to_vec().expect("to_vec"),
        &weight.to_vec().expect("to_vec"),
        Some(&bias.to_vec().expect("to_vec")),
        (1, 2, 5),
        (2, 2, 3),
        1,
        1,
        1,
    );
    let got = out.to_vec().expect("to_vec");
    for (g, e) in got.iter().zip(expected.iter()) {
        assert_relative_eq!(g, e, epsilon = 1e-5);
    }
}

/// F070: grouped conv1d exercises the group offsets in the hoisted loop.
#[test]
fn f070_conv1d_grouped_matches_reference() {
    let input = t32((1..=8).map(|v| v as f32).collect(), vec![1, 2, 4]);
    let weight = t32(vec![1.0, -1.0, 2.0, 0.5], vec![2, 1, 2]);

    let out = input
        .conv1d(&weight, None, 1, 0, 1, 2)
        .expect("grouped conv1d should succeed");
    assert_eq!(out.shape().dims(), &[1, 2, 3]);

    let expected = reference_conv1d(
        &input.to_vec().expect("to_vec"),
        &weight.to_vec().expect("to_vec"),
        None,
        (1, 2, 4),
        (2, 1, 2),
        1,
        0,
        1,
    );
    let got = out.to_vec().expect("to_vec");
    for (g, e) in got.iter().zip(expected.iter()) {
        assert_relative_eq!(g, e, epsilon = 1e-5);
    }
}

/// F070: conv2d and depthwise_conv2d must stay bit-for-bit equivalent to the
/// straightforward reference loop after the storage hoist.
#[test]
fn f070_conv2d_and_depthwise_match_reference() {
    let input = t32((1..=32).map(|v| v as f32 * 0.5).collect(), vec![1, 2, 4, 4]);
    let weight = t32(
        (0..8).map(|v| (v as f32) * 0.25 - 1.0).collect(),
        vec![1, 2, 2, 2],
    );
    let out = input
        .conv2d(&weight, None, (1, 1), (0, 0), (1, 1), 1)
        .expect("conv2d should succeed");
    assert_eq!(out.shape().dims(), &[1, 1, 3, 3]);

    // Reference: direct triple loop over the same data.
    let inp = input.to_vec().expect("to_vec");
    let wgt = weight.to_vec().expect("to_vec");
    let mut expected = vec![0.0f32; 9];
    for (oh, row) in (0..3).zip(0..3) {
        let _ = row;
        for ow in 0..3 {
            let mut sum = 0.0f32;
            for ic in 0..2 {
                for kh in 0..2 {
                    for kw in 0..2 {
                        sum += inp[ic * 16 + (oh + kh) * 4 + ow + kw] * wgt[ic * 4 + kh * 2 + kw];
                    }
                }
            }
            expected[oh * 3 + ow] = sum;
        }
    }
    let got = out.to_vec().expect("to_vec");
    for (g, e) in got.iter().zip(expected.iter()) {
        assert_relative_eq!(g, e, epsilon = 1e-5);
    }

    // Depthwise: one kernel per input channel.
    let dw_weight = t32(
        vec![1.0, 0.0, 0.0, -1.0, 0.5, 0.5, 0.5, 0.5],
        vec![2, 1, 2, 2],
    );
    let dw = input
        .depthwise_conv2d(&dw_weight, None, (1, 1), (0, 0), (1, 1))
        .expect("depthwise_conv2d should succeed");
    assert_eq!(dw.shape().dims(), &[1, 2, 3, 3]);

    let dw_w = dw_weight.to_vec().expect("to_vec");
    let dw_got = dw.to_vec().expect("to_vec");
    for c in 0..2 {
        for oh in 0..3 {
            for ow in 0..3 {
                let mut sum = 0.0f32;
                for kh in 0..2 {
                    for kw in 0..2 {
                        sum += inp[c * 16 + (oh + kh) * 4 + ow + kw] * dw_w[c * 4 + kh * 2 + kw];
                    }
                }
                assert_relative_eq!(dw_got[c * 9 + oh * 3 + ow], sum, epsilon = 1e-5);
            }
        }
    }
}

/// Straightforward conv1d reference implementation (no optimisation at all).
#[allow(clippy::too_many_arguments)]
fn reference_conv1d(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    input_shape: (usize, usize, usize),
    weight_shape: (usize, usize, usize),
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Vec<f32> {
    let (batch, in_channels, length) = input_shape;
    let (out_channels, in_per_group, kernel) = weight_shape;
    let groups = in_channels / in_per_group;
    let effective = (kernel - 1) * dilation + 1;
    let out_len = (length + 2 * padding - effective) / stride + 1;
    let mut out = vec![0.0f32; batch * out_channels * out_len];

    for n in 0..batch {
        for oc in 0..out_channels {
            let g = oc / (out_channels / groups);
            for ol in 0..out_len {
                let mut sum = 0.0f32;
                for ic_rel in 0..in_per_group {
                    let ic = g * in_per_group + ic_rel;
                    for k in 0..kernel {
                        let il = (ol * stride + k * dilation) as i64 - padding as i64;
                        if il >= 0 && (il as usize) < length {
                            sum += input[n * in_channels * length + ic * length + il as usize]
                                * weight[oc * in_per_group * kernel + ic_rel * kernel + k];
                        }
                    }
                }
                if let Some(b) = bias {
                    sum += b[oc];
                }
                out[n * out_channels * out_len + oc * out_len + ol] = sum;
            }
        }
    }
    out
}
