//! Production-hardening regression tests for torsh-nn attention, recurrent and
//! convolution layers.
//!
//! Each test is named after the finding it pins down.

use torsh_core::device::DeviceType;
use torsh_nn::cuda_kernels::CudaNeuralOps;
use torsh_nn::functional;
use torsh_nn::gradcheck::{GradCheckResult, ParameterGradCheckResult};
use torsh_nn::hardware_opts::{HardwareContext, HardwareLinear};
use torsh_nn::layers::{Conv2d, MultiheadAttention, GRU, LSTM};
use torsh_nn::Module;
use torsh_tensor::Tensor;

/// Absolute comparison helper used across the file.
fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (index, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "{what}: element {index} = {g}, expected {w}"
        );
    }
}

/// Overwrite a module parameter in place (the `Parameter` shares the tensor
/// `Arc` with the module, so this is visible to the forward pass).
fn set_param(module: &dyn Module, name: &str, values: Vec<f32>, shape: &[usize]) {
    let params = module.named_parameters();
    let param = params
        .get(name)
        .unwrap_or_else(|| panic!("parameter {name} must exist"));
    let tensor = Tensor::from_vec(values, shape).expect("tensor");
    *param.tensor().write() = tensor.requires_grad_(true);
}

// ---------------------------------------------------------------------------
// F038 - cross attention must actually read key/value
// ---------------------------------------------------------------------------

#[test]
fn f038_cross_attention_depends_on_key_value() {
    let attention =
        MultiheadAttention::with_config(4, 2, 0.0, true, false, false, None, None, true);
    let query = Tensor::from_vec((1..=8).map(|v| v as f32 * 0.1).collect(), &[1, 2, 4])
        .expect("query tensor");
    let kv_a = Tensor::from_vec(vec![0.5f32; 8], &[1, 2, 4]).expect("kv a");
    let kv_b = Tensor::from_vec((1..=8).map(|v| v as f32).collect(), &[1, 2, 4]).expect("kv b");

    let out_a = attention
        .forward_cross_attention(&query, &kv_a, None)
        .expect("cross attention a");
    let out_b = attention
        .forward_cross_attention(&query, &kv_b, None)
        .expect("cross attention b");

    let a = out_a.to_vec().expect("to_vec");
    let b = out_b.to_vec().expect("to_vec");
    let differs = a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-6);
    assert!(
        differs,
        "cross attention must depend on the key/value tensor, got identical outputs {a:?}"
    );
}

#[test]
fn f038_self_attention_still_matches_fused_path() {
    // Self attention (q == k == v) must keep working and produce the documented
    // shape.
    let attention =
        MultiheadAttention::with_config(4, 2, 0.0, true, false, false, None, None, true);
    let x = Tensor::from_vec((1..=8).map(|v| v as f32 * 0.25).collect(), &[1, 2, 4])
        .expect("input tensor");
    let out = attention.forward(&x).expect("self attention");
    assert_eq!(out.shape().dims(), &[1, 2, 4]);
}

// ---------------------------------------------------------------------------
// F234 - batch_first=false must not mix independent batch entries
// ---------------------------------------------------------------------------

#[test]
fn f234_batch_first_false_keeps_batch_entries_independent() {
    // Default layout is [seq, batch, embed] (batch_first = false).
    let attention = MultiheadAttention::new(4, 2);
    let mut base = vec![0.0f32; 2 * 2 * 4];
    for (index, slot) in base.iter_mut().enumerate() {
        *slot = index as f32 * 0.1;
    }
    let input = Tensor::from_vec(base.clone(), &[2, 2, 4]).expect("input");

    // Perturb only batch entry 1 (index 1 along dim 1).
    let mut perturbed_data = base;
    for step in 0..2usize {
        for feature in 0..4usize {
            perturbed_data[step * 8 + 4 + feature] += 3.0;
        }
    }
    let perturbed = Tensor::from_vec(perturbed_data, &[2, 2, 4]).expect("perturbed");

    let out_a = attention.forward(&input).expect("forward a");
    let out_b = attention.forward(&perturbed).expect("forward b");
    assert_eq!(out_a.shape().dims(), &[2, 2, 4]);

    let a = out_a.to_vec().expect("to_vec");
    let b = out_b.to_vec().expect("to_vec");
    // Batch entry 0 occupies [step, 0, :] -> flat offsets step*8 + 0..4.
    for step in 0..2usize {
        for feature in 0..4usize {
            let offset = step * 8 + feature;
            assert!(
                (a[offset] - b[offset]).abs() < 1e-5,
                "batch entry 0 changed when entry 1 was perturbed at offset {offset}: \
                 {} vs {}",
                a[offset],
                b[offset]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// F230 - attention dropout must follow the module's training flag
// ---------------------------------------------------------------------------

#[test]
fn f230_attention_dropout_applies_in_training_only() {
    let mut attention =
        MultiheadAttention::with_config(8, 2, 0.5, true, false, false, None, None, true);
    let input =
        Tensor::from_vec((1..=32).map(|v| v as f32 * 0.1).collect(), &[1, 4, 8]).expect("input");

    attention.eval();
    let eval_a = attention
        .forward(&input)
        .expect("eval a")
        .to_vec()
        .expect("vec");
    let eval_b = attention
        .forward(&input)
        .expect("eval b")
        .to_vec()
        .expect("vec");
    assert_close(&eval_a, &eval_b, 1e-6, "eval mode must be deterministic");

    attention.train();
    let mut saw_difference = false;
    for _ in 0..8 {
        let train_a = attention
            .forward(&input)
            .expect("train a")
            .to_vec()
            .expect("vec");
        if train_a
            .iter()
            .zip(eval_a.iter())
            .any(|(x, y)| (x - y).abs() > 1e-6)
        {
            saw_difference = true;
            break;
        }
    }
    assert!(
        saw_difference,
        "dropout=0.5 in training mode must perturb the attention output"
    );
}

// ---------------------------------------------------------------------------
// F129 - scirs2 integration attention must not panic on parameter lookup
// ---------------------------------------------------------------------------

#[test]
fn f129_scirs2_attention_forward_returns_result() {
    use torsh_nn::scirs2_neural_integration::MultiHeadAttention;

    let attention = MultiHeadAttention::new(4, 2, 0.0, true, DeviceType::Cpu).expect("construct");
    let x = Tensor::from_vec((1..=8).map(|v| v as f32 * 0.1).collect(), &[1, 2, 4]).expect("x");
    let (output, weights) = attention.forward(&x, &x, &x, None).expect("forward");
    assert_eq!(output.shape().dims(), &[1, 2, 4]);
    assert_eq!(weights.shape().dims(), &[1, 2, 2]);

    // The projections must not be dead zeros: a zero-initialised projection makes
    // the whole layer output constant zero regardless of the input.
    let values = output.to_vec().expect("to_vec");
    assert!(
        values.iter().any(|v| v.abs() > 1e-8),
        "multi-head attention projections must be initialised, got all zeros"
    );
}

// ---------------------------------------------------------------------------
// F132 - LSTM/GRU must use every layer and per-direction weights
// ---------------------------------------------------------------------------

#[test]
fn f132_lstm_uses_every_layer() {
    let lstm = LSTM::new(3, 2, 3).expect("lstm");
    let input =
        Tensor::from_vec((1..=12).map(|v| v as f32 * 0.1).collect(), &[2, 2, 3]).expect("input");
    let before = lstm
        .forward(&input)
        .expect("forward")
        .to_vec()
        .expect("vec");

    // Perturb the *second* layer's input-hidden weights.
    set_param(&lstm, "weight_ih_l1", vec![0.7f32; 4 * 2 * 2], &[8, 2]);
    let after = lstm
        .forward(&input)
        .expect("forward")
        .to_vec()
        .expect("vec");

    let differs = before
        .iter()
        .zip(after.iter())
        .any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(
        differs,
        "a 3-layer LSTM must run layer 1; output was unchanged after perturbing weight_ih_l1"
    );
}

#[test]
fn f132_lstm_bidirectional_has_reverse_parameters() {
    let lstm = LSTM::with_config(3, 2, 1, true, false, 0.0, true).expect("bidirectional lstm");
    let params = lstm.named_parameters();
    for key in [
        "weight_ih_l0",
        "weight_hh_l0",
        "bias_ih_l0",
        "bias_hh_l0",
        "weight_ih_l0_reverse",
        "weight_hh_l0_reverse",
        "bias_ih_l0_reverse",
        "bias_hh_l0_reverse",
    ] {
        assert!(params.contains_key(key), "missing parameter {key}");
    }

    let input =
        Tensor::from_vec((1..=12).map(|v| v as f32 * 0.1).collect(), &[2, 2, 3]).expect("input");
    let output = lstm.forward(&input).expect("forward");
    assert_eq!(output.shape().dims(), &[2, 2, 4]);

    // Perturbing only the reverse weights must change the output.
    let before = output.to_vec().expect("vec");
    set_param(&lstm, "weight_ih_l0_reverse", vec![0.9f32; 8 * 3], &[8, 3]);
    let after = lstm
        .forward(&input)
        .expect("forward")
        .to_vec()
        .expect("vec");
    assert!(
        before
            .iter()
            .zip(after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6),
        "the reverse direction must use its own weights"
    );
}

#[test]
fn f132_lstm_forward_with_state_round_trips() {
    let lstm = LSTM::new(3, 2, 2).expect("lstm");
    let input =
        Tensor::from_vec((1..=12).map(|v| v as f32 * 0.1).collect(), &[2, 2, 3]).expect("input");
    let h0 = Tensor::from_vec(vec![0.0f32; 2 * 2 * 2], &[2, 2, 2]).expect("h0");
    let c0 = Tensor::from_vec(vec![0.0f32; 2 * 2 * 2], &[2, 2, 2]).expect("c0");

    let (output, (hn, cn)) = lstm
        .forward_with_state(&input, Some((&h0, &c0)))
        .expect("forward_with_state");
    assert_eq!(output.shape().dims(), &[2, 2, 2]);
    assert_eq!(hn.shape().dims(), &[2, 2, 2]);
    assert_eq!(cn.shape().dims(), &[2, 2, 2]);

    // Zero initial state must equal the default forward pass.
    let plain = lstm
        .forward(&input)
        .expect("forward")
        .to_vec()
        .expect("vec");
    assert_close(
        &output.to_vec().expect("vec"),
        &plain,
        1e-6,
        "zero initial state must match the default forward",
    );

    // A non-zero initial hidden state must change the output.
    let h1 = Tensor::from_vec(vec![0.5f32; 2 * 2 * 2], &[2, 2, 2]).expect("h1");
    let (shifted, _) = lstm
        .forward_with_state(&input, Some((&h1, &c0)))
        .expect("forward_with_state");
    assert!(
        shifted
            .to_vec()
            .expect("vec")
            .iter()
            .zip(plain.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6),
        "a non-zero initial hidden state must influence the output"
    );
}

#[test]
fn f132_gru_uses_every_layer_and_reverse_weights() {
    let gru = GRU::new(3, 2, 2).expect("gru");
    let input =
        Tensor::from_vec((1..=12).map(|v| v as f32 * 0.1).collect(), &[2, 2, 3]).expect("input");
    let before = gru.forward(&input).expect("forward").to_vec().expect("vec");
    set_param(&gru, "weight_ih_l1", vec![0.7f32; 3 * 2 * 2], &[6, 2]);
    let after = gru.forward(&input).expect("forward").to_vec().expect("vec");
    assert!(
        before
            .iter()
            .zip(after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6),
        "a 2-layer GRU must run layer 1"
    );

    let bidirectional = GRU::with_config(3, 2, 1, true, false, 0.0, true).expect("bi gru");
    let params = bidirectional.named_parameters();
    assert!(params.contains_key("weight_ih_l0_reverse"));
    let output = bidirectional.forward(&input).expect("forward");
    assert_eq!(output.shape().dims(), &[2, 2, 4]);
}

#[test]
fn f132_gru_forward_with_state_round_trips() {
    let gru = GRU::new(3, 2, 1).expect("gru");
    let input =
        Tensor::from_vec((1..=12).map(|v| v as f32 * 0.1).collect(), &[2, 2, 3]).expect("input");
    let (output, hn) = gru.forward_with_state(&input, None).expect("state forward");
    assert_eq!(output.shape().dims(), &[2, 2, 2]);
    assert_eq!(hn.shape().dims(), &[1, 2, 2]);
}

// ---------------------------------------------------------------------------
// F133 - RNN sequence stacking must preserve values (autograd gap documented)
// ---------------------------------------------------------------------------

#[test]
fn f133_lstm_stacking_preserves_timestep_values() {
    // The stacked [seq, batch, hidden] output must contain exactly the per-step
    // hidden states, in order. This pins the data layout that a future
    // graph-preserving `Tensor::stack` has to reproduce.
    let lstm = LSTM::new(2, 2, 1).expect("lstm");
    set_param(&lstm, "weight_ih_l0", vec![0.1f32; 8 * 2], &[8, 2]);
    set_param(&lstm, "weight_hh_l0", vec![0.0f32; 8 * 2], &[8, 2]);
    set_param(&lstm, "bias_ih_l0", vec![0.0f32; 8], &[8]);
    set_param(&lstm, "bias_hh_l0", vec![0.0f32; 8], &[8]);

    let input = Tensor::from_vec(vec![1.0f32, 1.0, 2.0, 2.0], &[2, 1, 2]).expect("input");
    let output = lstm.forward(&input).expect("forward");
    assert_eq!(output.shape().dims(), &[2, 1, 2]);
    let values = output.to_vec().expect("vec");

    // Hand-computed reference for the first step (h0 = c0 = 0, all gate weights
    // 0.1, no bias): gate pre-activation = 0.1 * (1 + 1) = 0.2 for every gate.
    let gate = 1.0f32 / (1.0 + (-0.2f32).exp());
    let cell = gate * 0.2f32.tanh();
    let hidden = gate * cell.tanh();
    assert_close(
        &values[0..2],
        &[hidden, hidden],
        1e-5,
        "first timestep hidden state",
    );
}

#[test]
fn f133_rnn_gradient_gap_is_upstream_of_stacking() {
    // Characterisation test for the remaining RNN autograd gap.
    //
    // The recurrent graph is severed *inside* the cell, where `narrow` splits
    // the gate pre-activations: torsh-tensor rebuilds a slice through
    // `from_data`, producing a fresh leaf. Replacing the output stacking with a
    // graph-preserving `Tensor::stack` therefore changes nothing observable
    // until a differentiable narrow/slice exists.
    //
    // When torsh-tensor gains that primitive this assertion must be flipped to
    // require a real gradient.
    let lstm = LSTM::new(2, 2, 1).expect("lstm");
    let input = Tensor::from_vec(vec![1.0f32, 1.0, 2.0, 2.0], &[2, 1, 2])
        .expect("input")
        .requires_grad_(true);

    let output = lstm.forward(&input).expect("forward");
    let backward = output.sum().expect("sum").backward();

    let params = lstm.named_parameters();
    let weight = params.get("weight_ih_l0").expect("weight");
    let has_grad = weight.tensor().read().grad().is_some();
    assert!(
        backward.is_err() || !has_grad,
        "RNN weight gradients now flow: flip this characterisation test and drop the \
         to_vec/from_vec stacking in favour of a graph-preserving Tensor::stack"
    );
}

// ---------------------------------------------------------------------------
// F037 - transposed convolutions must compute real values
// ---------------------------------------------------------------------------

#[test]
fn f037_conv_transpose1d_matches_reference() {
    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[1, 1, 3]).expect("input");
    let weight = Tensor::from_vec(vec![1.0f32, 10.0], &[1, 1, 2]).expect("weight");

    let out = functional::conv_transpose1d(&input, &weight, None, 1, 0, 0, 1, 1).expect("ct1d");
    assert_eq!(out.shape().dims(), &[1, 1, 4]);
    assert_close(
        &out.to_vec().expect("vec"),
        &[1.0, 12.0, 23.0, 30.0],
        1e-5,
        "conv_transpose1d stride 1",
    );

    let strided = functional::conv_transpose1d(&input, &weight, None, 2, 0, 0, 1, 1).expect("ct1d");
    assert_eq!(strided.shape().dims(), &[1, 1, 6]);
    assert_close(
        &strided.to_vec().expect("vec"),
        &[1.0, 10.0, 2.0, 20.0, 3.0, 30.0],
        1e-5,
        "conv_transpose1d stride 2",
    );
}

#[test]
fn f037_conv_transpose2d_matches_reference() {
    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).expect("input");
    let weight =
        Tensor::from_vec(vec![1.0f32, 10.0, 100.0, 1000.0], &[1, 1, 2, 2]).expect("weight");

    let out =
        functional::conv_transpose2d(&input, &weight, None, (1, 1), (0, 0), (0, 0), 1, (1, 1))
            .expect("ct2d");
    assert_eq!(out.shape().dims(), &[1, 1, 3, 3]);
    assert_close(
        &out.to_vec().expect("vec"),
        &[
            1.0, 12.0, 20.0, 103.0, 1234.0, 2040.0, 300.0, 3400.0, 4000.0,
        ],
        1e-3,
        "conv_transpose2d",
    );

    // Padding crops the border; output_padding grows the bottom/right edge.
    let cropped =
        functional::conv_transpose2d(&input, &weight, None, (1, 1), (1, 1), (0, 0), 1, (1, 1))
            .expect("ct2d padded");
    assert_eq!(cropped.shape().dims(), &[1, 1, 1, 1]);
    assert_close(
        &cropped.to_vec().expect("vec"),
        &[1234.0],
        1e-3,
        "conv_transpose2d with padding",
    );
}

#[test]
fn f037_conv_transpose3d_matches_reference() {
    let input = Tensor::from_vec(vec![1.0f32, 2.0], &[1, 1, 1, 1, 2]).expect("input");
    let weight = Tensor::from_vec(vec![1.0f32, 10.0], &[1, 1, 1, 1, 2]).expect("weight");
    let out = functional::conv_transpose3d(
        &input,
        &weight,
        None,
        (1, 1, 1),
        (0, 0, 0),
        (0, 0, 0),
        1,
        (1, 1, 1),
    )
    .expect("ct3d");
    assert_eq!(out.shape().dims(), &[1, 1, 1, 1, 3]);
    assert_close(
        &out.to_vec().expect("vec"),
        &[1.0, 12.0, 20.0],
        1e-5,
        "conv_transpose3d",
    );
}

#[test]
fn f037_conv_transpose2d_groups_are_separate() {
    // Two groups, one channel each: group g must only see weight slice g.
    let input = Tensor::from_vec(vec![1.0f32, 2.0], &[1, 2, 1, 1]).expect("input");
    let weight = Tensor::from_vec(vec![3.0f32, 5.0], &[2, 1, 1, 1]).expect("weight");
    let out =
        functional::conv_transpose2d(&input, &weight, None, (1, 1), (0, 0), (0, 0), 2, (1, 1))
            .expect("ct2d groups");
    assert_eq!(out.shape().dims(), &[1, 2, 1, 1]);
    assert_close(
        &out.to_vec().expect("vec"),
        &[3.0, 10.0],
        1e-5,
        "grouped conv_transpose2d",
    );
}

// ---------------------------------------------------------------------------
// F032 - fused conv+bn+relu must actually apply batch norm
// ---------------------------------------------------------------------------

#[test]
fn f032_fused_conv_bn_relu_applies_batch_norm() {
    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).expect("input");
    let weight = Tensor::from_vec(vec![1.0f32], &[1, 1, 1, 1]).expect("weight");
    let bn_weight = Tensor::from_vec(vec![2.0f32], &[1]).expect("bn weight");
    let bn_bias = Tensor::from_vec(vec![-1.0f32], &[1]).expect("bn bias");
    let bn_mean = Tensor::from_vec(vec![2.0f32], &[1]).expect("bn mean");
    let bn_var = Tensor::from_vec(vec![1.0f32], &[1]).expect("bn var");

    let out = CudaNeuralOps::fused_conv_bn_relu(
        &input,
        &weight,
        None,
        &bn_weight,
        &bn_bias,
        &bn_mean,
        &bn_var,
        0.0,
        (1, 1),
        (0, 0),
    )
    .expect("fused op");

    // conv is identity here; bn: 2 * (x - 2) / 1 - 1; relu clamps negatives.
    let expected = [0.0f32, 0.0, 1.0, 3.0];
    assert_close(
        &out.to_vec().expect("vec"),
        &expected,
        1e-5,
        "fused conv+bn+relu",
    );
}

// ---------------------------------------------------------------------------
// F039 - Conv2d must honour groups and let gradients reach the weight
// ---------------------------------------------------------------------------

#[test]
fn f039_conv2d_honours_groups() {
    let conv = Conv2d::new(2, 2, (1, 1), (1, 1), (0, 0), (1, 1), false, 2);
    set_param(&conv, "weight", vec![2.0f32, 3.0], &[2, 1, 1, 1]);
    let input = Tensor::from_vec(vec![5.0f32, 7.0], &[1, 2, 1, 1]).expect("input");
    let out = conv.forward(&input).expect("forward");
    assert_eq!(out.shape().dims(), &[1, 2, 1, 1]);
    assert_close(
        &out.to_vec().expect("vec"),
        &[10.0, 21.0],
        1e-5,
        "grouped conv2d",
    );
}

#[test]
fn f039_conv2d_matches_reference() {
    let conv = Conv2d::new(1, 1, (2, 2), (1, 1), (0, 0), (1, 1), false, 1);
    set_param(&conv, "weight", vec![1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    let input = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[1, 1, 3, 3],
    )
    .expect("input");
    let out = conv.forward(&input).expect("forward");
    assert_eq!(out.shape().dims(), &[1, 1, 2, 2]);
    // Hand-computed 2x2 correlation over the 3x3 input.
    assert_close(
        &out.to_vec().expect("vec"),
        &[37.0, 47.0, 67.0, 77.0],
        1e-5,
        "conv2d reference",
    );
}

#[test]
fn f039_conv2d_weight_receives_gradient() {
    let conv = Conv2d::new(1, 1, (2, 2), (1, 1), (0, 0), (1, 1), false, 1);
    set_param(&conv, "weight", vec![1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).expect("input");

    let out = conv.forward(&input).expect("forward");
    out.sum().expect("sum").backward().expect("backward");

    let params = conv.named_parameters();
    let weight = params.get("weight").expect("weight parameter");
    let grad = weight
        .tensor()
        .read()
        .grad()
        .expect("Conv2d weight must receive a gradient");
    assert_close(
        &grad.to_vec().expect("vec"),
        &[1.0, 2.0, 3.0, 4.0],
        1e-5,
        "d(sum(conv2d))/dW is the input patch",
    );
}

#[test]
fn f039_conv2d_rejects_input_channel_mismatch() {
    let conv = Conv2d::new(2, 1, (1, 1), (1, 1), (0, 0), (1, 1), false, 1);
    let input = Tensor::from_vec(vec![1.0f32], &[1, 1, 1, 1]).expect("input");
    assert!(
        conv.forward(&input).is_err(),
        "a channel-count mismatch must be rejected instead of silently dropping taps"
    );
}

// ---------------------------------------------------------------------------
// F128 - HardwareLinear must not advertise fake specialisation
// ---------------------------------------------------------------------------

#[test]
fn f128_hardware_linear_matches_functional_linear() {
    let ctx = HardwareContext::auto_detect();
    let layer = HardwareLinear::new(3, 2, true, &ctx).expect("layer");
    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[1, 3]).expect("input");

    let params = layer.named_parameters();
    let weight = params
        .get("weight")
        .expect("weight")
        .tensor()
        .read()
        .clone();
    let bias = params.get("bias").expect("bias").tensor().read().clone();
    let reference = functional::linear(&input, &weight, Some(&bias)).expect("reference");

    let got = layer.forward(&input).expect("forward");
    assert_close(
        &got.to_vec().expect("vec"),
        &reference.to_vec().expect("vec"),
        1e-6,
        "hardware dispatch must match the portable path",
    );
}

// ---------------------------------------------------------------------------
// F130 - gradcheck must report NaN, not panic on it
// ---------------------------------------------------------------------------

#[test]
fn f130_worst_parameter_reports_nan_instead_of_panicking() {
    let result = GradCheckResult {
        passed: false,
        parameter_results: vec![
            ParameterGradCheckResult {
                name: "clean".to_string(),
                passed: true,
                max_abs_diff: 1e-9,
                max_rel_diff: 1e-9,
                elements_checked: 4,
                error: None,
            },
            ParameterGradCheckResult {
                name: "nan".to_string(),
                passed: false,
                max_abs_diff: f64::NAN,
                max_rel_diff: f64::NAN,
                elements_checked: 4,
                error: None,
            },
        ],
        summary: String::new(),
    };

    let worst = result
        .worst_parameter()
        .expect("worst_parameter must return a result");
    assert_eq!(
        worst.name, "nan",
        "a NaN difference is the most severe gradcheck outcome"
    );
}

#[test]
fn f039_conv2d_input_gradient_matches_finite_differences() {
    // Gradcheck for the input: the im2col gather records a col2im backward, so
    // dL/dx must match a central finite difference of the same loss.
    let conv = Conv2d::new(1, 1, (2, 2), (1, 1), (0, 0), (1, 1), false, 1);
    set_param(&conv, "weight", vec![1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    let values: Vec<f32> = (1..=9).map(|v| v as f32 * 0.5).collect();

    let loss_at = |data: &[f32]| -> f32 {
        let input = Tensor::from_vec(data.to_vec(), &[1, 1, 3, 3]).expect("input");
        let out = conv.forward(&input).expect("forward");
        out.sum().expect("sum").to_vec().expect("vec")[0]
    };

    let input = Tensor::from_vec(values.clone(), &[1, 1, 3, 3])
        .expect("input")
        .requires_grad_(true);
    let out = conv.forward(&input).expect("forward");
    out.sum().expect("sum").backward().expect("backward");
    let analytic = input
        .grad()
        .expect("Conv2d must produce an input gradient")
        .to_vec()
        .expect("vec");

    let eps = 1e-2f32;
    for index in 0..values.len() {
        let mut plus = values.clone();
        plus[index] += eps;
        let mut minus = values.clone();
        minus[index] -= eps;
        let numeric = (loss_at(&plus) - loss_at(&minus)) / (2.0 * eps);
        assert!(
            (analytic[index] - numeric).abs() < 1e-2,
            "input grad[{index}] = {}, finite difference {numeric}",
            analytic[index]
        );
    }
}

#[test]
fn f039_stacked_conv2d_propagates_gradients_to_every_layer() {
    // Two convolutions in a row: the first layer's weight only receives a
    // gradient if the second layer's backward reaches through its input.
    let first = Conv2d::new(1, 1, (2, 2), (1, 1), (0, 0), (1, 1), false, 1);
    let second = Conv2d::new(1, 1, (2, 2), (1, 1), (0, 0), (1, 1), false, 1);
    set_param(
        &first,
        "weight",
        vec![1.0f32, 0.5, -0.5, 2.0],
        &[1, 1, 2, 2],
    );
    set_param(
        &second,
        "weight",
        vec![0.25f32, -1.0, 1.5, 0.75],
        &[1, 1, 2, 2],
    );

    let input =
        Tensor::from_vec((1..=9).map(|v| v as f32 * 0.3).collect(), &[1, 1, 3, 3]).expect("input");
    let hidden = first.forward(&input).expect("first conv");
    assert_eq!(hidden.shape().dims(), &[1, 1, 2, 2]);
    let output = second.forward(&hidden).expect("second conv");
    output.sum().expect("sum").backward().expect("backward");

    for (name, conv) in [("first", &first), ("second", &second)] {
        let params = conv.named_parameters();
        let weight = params.get("weight").expect("weight");
        let grad = weight.tensor().read().grad();
        assert!(
            grad.is_some(),
            "the {name} convolution must receive a weight gradient"
        );
    }
}

#[test]
fn f039_conv2d_gradcheck_with_stride_padding_and_groups() {
    // Exercise the im2col/col2im pair away from the trivial configuration:
    // asymmetric stride, padding and two channel groups.
    let conv = Conv2d::new(2, 4, (2, 2), (2, 1), (1, 1), (1, 1), false, 2);
    let weight_values: Vec<f32> = (1..=16).map(|v| v as f32 * 0.25 - 1.0).collect();
    set_param(&conv, "weight", weight_values, &[4, 1, 2, 2]);

    let values: Vec<f32> = (1..=32).map(|v| (v % 7) as f32 * 0.3 - 0.5).collect();
    let loss_at = |data: &[f32]| -> f32 {
        let input = Tensor::from_vec(data.to_vec(), &[1, 2, 4, 4]).expect("input");
        let out = conv.forward(&input).expect("forward");
        out.sum().expect("sum").to_vec().expect("vec")[0]
    };

    let input = Tensor::from_vec(values.clone(), &[1, 2, 4, 4])
        .expect("input")
        .requires_grad_(true);
    let out = conv.forward(&input).expect("forward");
    assert_eq!(out.shape().dims(), &[1, 4, 3, 5]);
    out.sum().expect("sum").backward().expect("backward");

    let analytic = input.grad().expect("input gradient").to_vec().expect("vec");
    let eps = 1e-2f32;
    for index in 0..values.len() {
        let mut plus = values.clone();
        plus[index] += eps;
        let mut minus = values.clone();
        minus[index] -= eps;
        let numeric = (loss_at(&plus) - loss_at(&minus)) / (2.0 * eps);
        assert!(
            (analytic[index] - numeric).abs() < 1e-2,
            "input grad[{index}] = {}, finite difference {numeric}",
            analytic[index]
        );
    }

    let params = conv.named_parameters();
    let weight = params.get("weight").expect("weight");
    let weight_grad = weight
        .tensor()
        .read()
        .grad()
        .expect("weight gradient")
        .to_vec()
        .expect("vec");
    assert_eq!(weight_grad.len(), 16);
    assert!(
        weight_grad.iter().any(|g| g.abs() > 1e-6),
        "grouped conv weights must receive non-zero gradients"
    );
}

// ---------------------------------------------------------------------------
// TransformerDecoderLayer (crate TODO audit): self-attn + cross-attn + FFN
// ---------------------------------------------------------------------------

#[test]
fn transformer_decoder_layer_attends_to_memory() {
    use torsh_nn::scirs2_neural_integration::TransformerDecoderLayer;

    let layer = TransformerDecoderLayer::new(4, 2, 8, 0.0, DeviceType::Cpu).expect("decoder layer");
    let tgt = Tensor::from_vec((1..=8).map(|v| v as f32 * 0.1).collect(), &[1, 2, 4]).expect("tgt");
    let memory_a = Tensor::from_vec(vec![0.5f32; 12], &[1, 3, 4]).expect("memory a");
    let memory_b =
        Tensor::from_vec((1..=12).map(|v| v as f32 * 0.2).collect(), &[1, 3, 4]).expect("memory b");

    let out_a = layer
        .forward(&tgt, &memory_a, None, None)
        .expect("decode a");
    let out_b = layer
        .forward(&tgt, &memory_b, None, None)
        .expect("decode b");
    assert_eq!(out_a.shape().dims(), &[1, 2, 4]);

    let a = out_a.to_vec().expect("vec");
    let b = out_b.to_vec().expect("vec");
    assert!(
        a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-6),
        "the decoder output must depend on the encoder memory"
    );

    // Every sublayer must expose its parameters for the optimizer.
    let params = layer.named_parameters();
    for key in [
        "self_attn.q_proj",
        "cross_attn.q_proj",
        "linear1",
        "linear2",
        "norm1.weight",
        "norm3.bias",
    ] {
        assert!(params.contains_key(key), "missing parameter {key}");
    }
}
