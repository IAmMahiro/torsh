//! Regression tests for the torsh-functional production-hardening findings.
//!
//! Every test in this file pins down behaviour that was previously fabricated,
//! unconditionally failing, or silently wrong.

use torsh_core::device::DeviceType;
use torsh_core::Result as TorshResult;
use torsh_functional::attention::{scaled_dot_product_attention, MultiHeadAttentionWeights};
use torsh_functional::autograd::{apply_custom_function, CustomAutogradFunction};
use torsh_functional::loss::common::ReductionType;
use torsh_functional::random_ops::bernoulli;
use torsh_functional::{
    binary_cross_entropy_with_logits, conv_transpose1d, cross_entropy, dropout, focal_loss,
    gradient_penalty, grid_sample, multi_head_attention, multi_margin_loss, nll_loss,
    r1_gradient_penalty, resize, spectral_gradient_penalty, InterpolationMode,
};
use torsh_tensor::Tensor;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor {
    Tensor::from_data(data, shape.to_vec(), DeviceType::Cpu)
        .expect("tensor creation should succeed")
}

fn leaf(data: Vec<f32>, shape: &[usize]) -> Tensor {
    tensor(data, shape).requires_grad_(true)
}

fn assert_close(actual: f32, expected: f32, tolerance: f32, what: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{what}: got {actual}, expected {expected}"
    );
}

// ---------------------------------------------------------------- F018

#[test]
fn f018_bernoulli_rejects_out_of_range_probabilities() {
    let probs = tensor(vec![0.1, 1.5, 0.5], &[3]);
    let error = bernoulli(&probs, Some(7)).expect_err("out-of-range probability must be an error");
    let message = format!("{error}");
    assert!(
        message.contains("between 0 and 1"),
        "error should name the violated range, got: {message}"
    );
}

#[test]
fn f018_bernoulli_rejects_nan_probabilities() {
    let probs = tensor(vec![0.25, f32::NAN], &[2]);
    assert!(
        bernoulli(&probs, Some(7)).is_err(),
        "NaN probability must be reported through the Result channel"
    );
}

#[test]
fn f018_bernoulli_accepts_valid_probabilities() -> TorshResult<()> {
    let probs = tensor(vec![0.0, 1.0, 0.5, 0.25], &[2, 2]);
    let samples = bernoulli(&probs, Some(42))?;
    assert_eq!(samples.shape().dims(), &[2, 2]);
    for value in samples.data()? {
        assert!(value == 0.0 || value == 1.0);
    }
    Ok(())
}

// ---------------------------------------------------------------- F019

/// A critic `D(x) = x @ w` has `∇_x D(x) = wᵀ` for every sample, so the WGAN-GP
/// penalty is exactly `lambda * (||w|| - 1)^2`. Fabricated gradients cannot
/// reproduce that number.
#[test]
fn f019_gradient_penalty_uses_the_real_critic_gradient() -> TorshResult<()> {
    let real = tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let fake = tensor(vec![-1.0, 0.5, 2.0, -3.0], &[2, 2]);
    let weight = tensor(vec![3.0, 4.0], &[2, 1]); // ||w|| = 5

    let critic = |x: &Tensor| x.matmul(&weight);
    let penalty = gradient_penalty(&real, &fake, critic, 10.0, "mean")?;

    // 10 * (5 - 1)^2 = 160, independent of the (random) interpolation coefficient.
    assert_close(penalty.item()?, 160.0, 1e-3, "WGAN-GP penalty");
    Ok(())
}

#[test]
fn f019_gradient_penalty_reacts_to_the_critic() -> TorshResult<()> {
    let real = tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let fake = tensor(vec![0.0, 0.0, 0.0, 0.0], &[2, 2]);

    let unit_weight = tensor(vec![1.0, 0.0], &[2, 1]); // ||w|| = 1 -> penalty 0
    let big_weight = tensor(vec![6.0, 8.0], &[2, 1]); // ||w|| = 10 -> penalty 81

    let first = gradient_penalty(
        &real,
        &fake,
        |x: &Tensor| x.matmul(&unit_weight),
        1.0,
        "mean",
    )?;
    let second = gradient_penalty(
        &real,
        &fake,
        |x: &Tensor| x.matmul(&big_weight),
        1.0,
        "mean",
    )?;

    assert_close(first.item()?, 0.0, 1e-4, "unit-norm critic penalty");
    assert_close(second.item()?, 81.0, 1e-3, "large critic penalty");
    Ok(())
}

#[test]
fn f019_gradient_penalty_does_not_pollute_caller_gradients() -> TorshResult<()> {
    let real = leaf(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let fake = leaf(vec![0.5, 0.5, 0.5, 0.5], &[2, 2]);
    let weight = tensor(vec![1.0, 1.0], &[2, 1]);

    let _ = gradient_penalty(&real, &fake, |x: &Tensor| x.matmul(&weight), 1.0, "mean")?;

    assert!(
        real.grad().is_none() && fake.grad().is_none(),
        "the internal backward pass must not accumulate into the caller's tensors"
    );
    Ok(())
}

#[test]
fn f019_r1_penalty_matches_the_analytic_value() -> TorshResult<()> {
    let real = tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let weight = tensor(vec![3.0, 4.0], &[2, 1]); // ||w||^2 = 25

    let penalty = r1_gradient_penalty(&real, |x: &Tensor| x.matmul(&weight), 2.0, "mean")?;

    // 0.5 * lambda * ||grad||^2 = 0.5 * 2 * 25 = 25
    assert_close(penalty.item()?, 25.0, 1e-4, "R1 penalty");
    Ok(())
}

#[test]
fn f019_gradient_penalty_reports_a_detached_critic() {
    let real = tensor(vec![1.0, 2.0], &[1, 2]);
    let fake = tensor(vec![0.0, 0.0], &[1, 2]);

    // A critic that rebuilds its output from raw data is disconnected from the graph.
    let detached_critic = |x: &Tensor| {
        let data = x.data()?;
        Ok(tensor(vec![data.iter().sum::<f32>()], &[1, 1]))
    };

    assert!(
        gradient_penalty(&real, &fake, detached_critic, 1.0, "mean").is_err(),
        "a critic detached from the graph must be reported, not silently penalised"
    );
}

#[test]
fn f019_spectral_penalty_uses_the_real_gradient() -> TorshResult<()> {
    // Every sample gets the same gradient wᵀ = (3, 4), so the [2, 2] gradient
    // matrix has singular value ||w|| * sqrt(2) = 5 * sqrt(2).
    let input = tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let weight = tensor(vec![3.0, 4.0], &[2, 1]);

    let penalty =
        spectral_gradient_penalty(|x: &Tensor| x.matmul(&weight), &input, 1.0, "mean", 50)?;

    let expected = (5.0 * 2.0f32.sqrt() - 1.0).powi(2);
    assert_close(penalty.item()?, expected, 1e-3, "spectral penalty");
    Ok(())
}

// ---------------------------------------------------------------- F111

fn mha_weight(embed_dim: usize, scale: f32) -> Tensor {
    let mut data = vec![0.0f32; embed_dim * embed_dim];
    for i in 0..embed_dim {
        data[i * embed_dim + i] = scale;
    }
    tensor(data, &[embed_dim, embed_dim])
}

#[test]
fn f111_multi_head_attention_is_deterministic() -> TorshResult<()> {
    let embed_dim = 4;
    let input = tensor(
        (0..16).map(|i| i as f32 * 0.1).collect(),
        &[2, 2, embed_dim],
    );
    let (wq, wk) = (mha_weight(embed_dim, 1.0), mha_weight(embed_dim, 1.0));
    let (wv, wo) = (mha_weight(embed_dim, 1.0), mha_weight(embed_dim, 1.0));
    let weights = MultiHeadAttentionWeights::new(&wq, &wk, &wv, &wo);

    let (first, _) = multi_head_attention(
        &input, &input, &input, &weights, embed_dim, 2, 0.0, true, None,
    )?;
    let (second, _) = multi_head_attention(
        &input, &input, &input, &weights, embed_dim, 2, 0.0, true, None,
    )?;

    assert_eq!(
        first.data()?,
        second.data()?,
        "identical inputs and weights must produce identical outputs"
    );
    Ok(())
}

#[test]
fn f111_multi_head_attention_uses_the_supplied_weights() -> TorshResult<()> {
    let embed_dim = 4;
    let input = tensor(
        (0..16).map(|i| i as f32 * 0.1).collect(),
        &[2, 2, embed_dim],
    );
    let identity = mha_weight(embed_dim, 1.0);
    let doubled = mha_weight(embed_dim, 2.0);

    let base = MultiHeadAttentionWeights::new(&identity, &identity, &identity, &identity);
    let scaled_value = MultiHeadAttentionWeights::new(&identity, &identity, &doubled, &identity);

    let (plain, _) =
        multi_head_attention(&input, &input, &input, &base, embed_dim, 2, 0.0, true, None)?;
    let (scaled, _) = multi_head_attention(
        &input,
        &input,
        &input,
        &scaled_value,
        embed_dim,
        2,
        0.0,
        true,
        None,
    )?;

    let plain_data = plain.data()?;
    let scaled_data = scaled.data()?;
    for (index, (&p, &s)) in plain_data.iter().zip(scaled_data.iter()).enumerate() {
        assert_close(
            s,
            2.0 * p,
            1e-4,
            &format!("doubled value projection at {index}"),
        );
    }
    Ok(())
}

#[test]
fn f111_multi_head_attention_validates_weight_shapes() {
    let embed_dim = 4;
    let input = tensor(vec![0.0; 16], &[2, 2, embed_dim]);
    let good = mha_weight(embed_dim, 1.0);
    let bad = tensor(vec![0.0; 8], &[2, 4]);
    let weights = MultiHeadAttentionWeights::new(&bad, &good, &good, &good);

    assert!(
        multi_head_attention(&input, &input, &input, &weights, embed_dim, 2, 0.0, true, None)
            .is_err(),
        "a mis-shaped projection must be rejected"
    );
}

// ---------------------------------------------------------------- F112 / F113 / F300

#[test]
fn f112_nll_loss_is_differentiable() -> TorshResult<()> {
    let log_probs = leaf(vec![-0.5, -1.5, -2.0, -0.2, -3.0, -1.0], &[2, 3]);
    let target = tensor(vec![1.0, 2.0], &[2]);

    let loss = nll_loss(&log_probs, &target, None, "mean", None)?;
    assert_close(loss.item()?, (1.5 + 1.0) / 2.0, 1e-6, "mean NLL value");

    loss.backward()?;
    let grad = log_probs
        .grad()
        .expect("nll_loss must stay connected to its input")
        .data()?;
    // d/d log_probs = -one_hot / batch_size
    assert_eq!(grad, vec![0.0, -0.5, 0.0, 0.0, 0.0, -0.5]);
    Ok(())
}

#[test]
fn f112_nll_loss_supports_weight_and_ignore_index() -> TorshResult<()> {
    let log_probs = tensor(vec![-0.5, -1.5, -2.0, -0.2, -3.0, -1.0], &[2, 3]);
    let target = tensor(vec![1.0, 2.0], &[2]);
    let weight = tensor(vec![1.0, 2.0, 3.0], &[3]);

    // weighted mean = (2 * 1.5 + 3 * 1.0) / (2 + 3)
    let weighted = nll_loss(&log_probs, &target, Some(&weight), "mean", None)?;
    assert_close(weighted.item()?, 6.0 / 5.0, 1e-6, "weighted NLL");

    // ignoring class 2 leaves only the first sample
    let ignored = nll_loss(&log_probs, &target, None, "mean", Some(2))?;
    assert_close(ignored.item()?, 1.5, 1e-6, "NLL with ignore_index");
    Ok(())
}

#[test]
fn f112_nll_loss_sum_reduction_is_differentiable() -> TorshResult<()> {
    // `cross_entropy` cannot yet be differentiated end-to-end because
    // torsh-tensor's `log_softmax` registers no backward node; what this crate
    // owns — the gather — is differentiable and is what is exercised here.
    let log_probs = leaf(vec![-1.0, -0.5, -2.0, -1.2], &[2, 2]);
    let target = tensor(vec![0.0, 1.0], &[2]);
    let loss = nll_loss(&log_probs, &target, None, "sum", None)?;
    assert_close(loss.item()?, 1.0 + 1.2, 1e-6, "sum NLL value");
    loss.backward()?;
    assert!(log_probs.grad().is_some());
    Ok(())
}

#[test]
fn f112_multi_margin_loss_matches_the_reference_and_has_a_gradient() -> TorshResult<()> {
    let input = leaf(vec![0.2, 0.5, 0.1, 1.0, -0.5, 0.3], &[2, 3]);
    let target = tensor(vec![0.0, 1.0], &[2]);
    let margin = 1.0f32;

    // NOTE: this crate normalises by (C - 1); PyTorch's MultiMarginLoss divides by
    // C. The expectations below follow the current implementation, not PyTorch.
    let loss = multi_margin_loss(&input, &target, 1, margin, None, ReductionType::None)?;
    let values = loss.data()?;

    // sample 0: target score 0.2 -> max(0, 1 - 0.2 + 0.5) + max(0, 1 - 0.2 + 0.1)
    let expected_0 = ((1.0 - 0.2 + 0.5) + (1.0 - 0.2 + 0.1)) / 2.0;
    // sample 1: target score -0.5 -> max(0, 1 + 0.5 + 1.0) + max(0, 1 + 0.5 + 0.3)
    let expected_1 = ((1.0 + 0.5 + 1.0) + (1.0 + 0.5 + 0.3)) / 2.0;
    assert_close(values[0], expected_0, 1e-6, "multi_margin sample 0");
    assert_close(values[1], expected_1, 1e-6, "multi_margin sample 1");

    let scalar = multi_margin_loss(&input, &target, 1, margin, None, ReductionType::Sum)?;
    scalar.backward()?;
    let grad = input
        .grad()
        .expect("multi_margin_loss must stay connected to its input")
        .data()?;
    // Every non-target class violates the margin here, so d/dx_j = 1/(C-1) and
    // d/dx_target = -(C-1)/(C-1) = -1.
    assert_close(grad[0], -1.0, 1e-6, "gradient at target of sample 0");
    assert_close(grad[1], 0.5, 1e-6, "gradient at class 1 of sample 0");
    Ok(())
}

#[test]
fn f112_focal_loss_matches_the_reference() -> TorshResult<()> {
    let input = tensor(vec![1.0, 2.0, 0.5, 3.0, 1.5, 0.8], &[2, 3]);
    let target = tensor(vec![1.0, 2.0], &[2]);
    let (alpha, gamma) = (0.25f32, 2.0f32);

    let loss = focal_loss(&input, &target, alpha, gamma, ReductionType::None)?;
    let values = loss.data()?;

    let reference = |logits: [f32; 3], class: usize| -> f32 {
        let max = logits.iter().cloned().fold(f32::MIN, f32::max);
        let exp_sum: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
        let log_p = logits[class] - max - exp_sum.ln();
        let p = log_p.exp();
        -alpha * (1.0 - p).powf(gamma) * log_p
    };

    assert_close(
        values[0],
        reference([1.0, 2.0, 0.5], 1),
        1e-5,
        "focal sample 0",
    );
    assert_close(
        values[1],
        reference([3.0, 1.5, 0.8], 2),
        1e-5,
        "focal sample 1",
    );
    Ok(())
}

#[test]
fn f113_cross_entropy_with_label_smoothing_succeeds() -> TorshResult<()> {
    let input = tensor(vec![2.0, 1.0, 0.1, 0.5, 2.5, 0.2], &[2, 3]);
    let target = tensor(vec![0.0, 1.0], &[2]);

    let plain = cross_entropy(&input, &target, None, "mean", None, 0.0)?.item()?;
    let smoothed = cross_entropy(&input, &target, None, "mean", None, 0.1)?.item()?;

    // Smoothing spreads probability mass onto the other classes, so the loss grows,
    // but stays below the fully-uniform target loss.
    let uniform = cross_entropy(&input, &target, None, "mean", None, 0.999_9)?.item()?;
    assert!(
        plain < smoothed && smoothed < uniform,
        "expected {plain} < {smoothed} < {uniform}"
    );
    Ok(())
}

#[test]
fn f113_label_smoothing_none_reduction_keeps_the_batch_axis() -> TorshResult<()> {
    let input = tensor(vec![2.0, 1.0, 0.1, 0.5, 2.5, 0.2], &[2, 3]);
    let target = tensor(vec![0.0, 1.0], &[2]);

    let loss = cross_entropy(&input, &target, None, "none", None, 0.2)?;
    assert_eq!(loss.shape().dims(), &[2]);
    Ok(())
}

#[test]
fn f300_bce_with_logits_pos_weight_matches_pytorch_on_soft_targets() -> TorshResult<()> {
    let x = 0.7f32;
    let t = 0.5f32;
    let pw = 3.0f32;

    let input = tensor(vec![x], &[1]);
    let target = tensor(vec![t], &[1]);
    let pos_weight = tensor(vec![pw], &[1]);

    let loss = binary_cross_entropy_with_logits(
        &input,
        &target,
        None,
        ReductionType::Mean,
        Some(&pos_weight),
    )?;

    // PyTorch: -(pw * t * log(sigmoid(x)) + (1 - t) * log(1 - sigmoid(x)))
    let sigmoid = 1.0 / (1.0 + (-x).exp());
    let expected = -(pw * t * sigmoid.ln() + (1.0 - t) * (1.0 - sigmoid).ln());
    assert_close(loss.item()?, expected, 1e-5, "pos_weight with soft target");
    Ok(())
}

#[test]
fn f300_bce_with_logits_unchanged_for_hard_targets() -> TorshResult<()> {
    let input = tensor(vec![0.7, -1.2], &[2]);
    let target = tensor(vec![1.0, 0.0], &[2]);
    let pos_weight = tensor(vec![3.0, 3.0], &[2]);

    let loss = binary_cross_entropy_with_logits(
        &input,
        &target,
        None,
        ReductionType::None,
        Some(&pos_weight),
    )?;
    let values = loss.data()?;

    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    assert_close(values[0], -3.0 * sigmoid(0.7).ln(), 1e-5, "hard positive");
    assert_close(
        values[1],
        -(1.0 - sigmoid(-1.2)).ln(),
        1e-5,
        "hard negative",
    );
    Ok(())
}

// ---------------------------------------------------------------- F216

#[test]
fn f216_scaled_dot_product_attention_supports_cross_attention() -> TorshResult<()> {
    let (batch, heads, q_len, kv_len, head_dim) = (2, 2, 4, 7, 3);
    let query = tensor(
        (0..batch * heads * q_len * head_dim)
            .map(|i| (i % 7) as f32 * 0.1)
            .collect(),
        &[batch, heads, q_len, head_dim],
    );
    let key = tensor(
        (0..batch * heads * kv_len * head_dim)
            .map(|i| (i % 5) as f32 * 0.2)
            .collect(),
        &[batch, heads, kv_len, head_dim],
    );
    let value = key.clone();

    let (output, weights) = scaled_dot_product_attention(&query, &key, &value, None, 0.0, false)?;

    assert_eq!(output.shape().dims(), &[batch, heads, q_len, head_dim]);
    assert_eq!(weights.shape().dims(), &[batch, heads, q_len, kv_len]);

    // Every attention row is a probability distribution over kv_len keys.
    let weight_data = weights.data()?;
    for row in weight_data.chunks(kv_len) {
        assert_close(row.iter().sum::<f32>(), 1.0, 1e-4, "attention row sum");
    }
    Ok(())
}

/// Shapes alone cannot tell a correct cross-attention from one that reinterprets
/// a strided buffer, so the values are checked against a hand-rolled reference.
#[test]
fn f216_cross_attention_values_match_the_reference() -> TorshResult<()> {
    let (q_len, kv_len, head_dim) = (2usize, 3usize, 2usize);
    let q_data = vec![1.0, 0.0, 0.0, 1.0];
    let k_data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let v_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let query = tensor(q_data.clone(), &[1, 1, q_len, head_dim]);
    let key = tensor(k_data.clone(), &[1, 1, kv_len, head_dim]);
    let value = tensor(v_data.clone(), &[1, 1, kv_len, head_dim]);

    let (output, weights) = scaled_dot_product_attention(&query, &key, &value, None, 0.0, false)?;

    let scale = 1.0 / (head_dim as f32).sqrt();
    let output_data = output.data()?;
    let weight_data = weights.data()?;
    for i in 0..q_len {
        // scores = q_i . k_j / sqrt(d)
        let scores: Vec<f32> = (0..kv_len)
            .map(|j| {
                (0..head_dim)
                    .map(|d| q_data[i * head_dim + d] * k_data[j * head_dim + d])
                    .sum::<f32>()
                    * scale
            })
            .collect();
        let max = scores.iter().cloned().fold(f32::MIN, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let expected_weights: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        for j in 0..kv_len {
            assert_close(
                weight_data[i * kv_len + j],
                expected_weights[j],
                1e-5,
                &format!("attention weight ({i}, {j})"),
            );
        }

        for d in 0..head_dim {
            let expected: f32 = (0..kv_len)
                .map(|j| expected_weights[j] * v_data[j * head_dim + d])
                .sum();
            assert_close(
                output_data[i * head_dim + d],
                expected,
                1e-5,
                &format!("attention output ({i}, {d})"),
            );
        }
    }
    Ok(())
}

#[test]
fn f216_causal_masking_works_for_rectangular_scores() -> TorshResult<()> {
    let (batch, heads, q_len, kv_len, head_dim) = (1, 1, 3, 5, 2);
    let query = tensor(
        vec![0.1; batch * heads * q_len * head_dim],
        &[batch, heads, q_len, head_dim],
    );
    let key = tensor(
        vec![0.2; batch * heads * kv_len * head_dim],
        &[batch, heads, kv_len, head_dim],
    );
    let value = key.clone();

    let (_, weights) = scaled_dot_product_attention(&query, &key, &value, None, 0.0, true)?;
    let data = weights.data()?;

    // Query i must not attend to key j > i.
    for i in 0..q_len {
        for j in (i + 1)..kv_len {
            assert_close(data[i * kv_len + j], 0.0, 1e-5, "masked attention weight");
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- F217

struct DetachedFunction;

impl CustomAutogradFunction for DetachedFunction {
    fn forward(&self, inputs: &[Tensor]) -> TorshResult<Vec<Tensor>> {
        // Rebuilding from raw data detaches the result from the graph.
        let data: Vec<f32> = inputs[0].data()?.iter().map(|x| x * x).collect();
        Ok(vec![tensor(data, inputs[0].shape().dims())])
    }

    fn backward(
        &self,
        grad_outputs: &[Tensor],
        inputs: &[Tensor],
    ) -> TorshResult<Vec<Option<Tensor>>> {
        Ok(vec![Some(grad_outputs[0].mul_op(&inputs[0])?)])
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "detached"
    }
}

#[test]
fn f217_custom_function_reports_an_unreachable_backward() {
    let input = leaf(vec![1.0, 2.0], &[2]);
    let error = apply_custom_function(DetachedFunction, &[input])
        .expect_err("a detached custom forward must not silently drop the backward");
    let message = format!("{error}");
    assert!(
        message.contains("custom backward"),
        "error should explain the missing backward, got: {message}"
    );
}

#[test]
fn f217_custom_function_still_works_without_gradients() -> TorshResult<()> {
    let input = tensor(vec![3.0, 4.0], &[2]);
    let outputs = apply_custom_function(DetachedFunction, &[input])?;
    assert_eq!(outputs[0].data()?, vec![9.0, 16.0]);
    Ok(())
}

// ---------------------------------------------------------------- F218

#[test]
fn f218_conv_transpose1d_handles_groups_and_bias() -> TorshResult<()> {
    // 2 input channels, 2 groups: each output channel sees only its own group.
    let input = tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3]);
    let weight = tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 1, 2]);
    let bias = tensor(vec![10.0, 100.0], &[2]);

    let output = conv_transpose1d(&input, &weight, Some(&bias), 1, 0, 0, 2, 1)?;
    assert_eq!(output.shape().dims(), &[1, 2, 4]);

    let expected = vec![
        1.0 + 10.0,
        4.0 + 10.0,
        7.0 + 10.0,
        6.0 + 10.0,
        12.0 + 100.0,
        31.0 + 100.0,
        38.0 + 100.0,
        24.0 + 100.0,
    ];
    assert_eq!(output.data()?, expected);
    Ok(())
}

#[test]
fn f218_conv_transpose1d_bias_is_added_per_channel() -> TorshResult<()> {
    // C_out == L_out would let an unreshaped bias broadcast along the wrong axis.
    let input = tensor(vec![1.0, 1.0], &[1, 1, 2]);
    let weight = tensor(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], &[1, 3, 2]);
    let bias = tensor(vec![0.0, 1.0, 2.0], &[3]);

    let output = conv_transpose1d(&input, &weight, Some(&bias), 1, 0, 0, 1, 1)?;
    assert_eq!(output.shape().dims(), &[1, 3, 3]);

    let data = output.data()?;
    for channel in 0..3 {
        for position in 0..3 {
            let base = if position == 1 { 2.0 } else { 1.0 };
            assert_close(
                data[channel * 3 + position],
                base + channel as f32,
                1e-6,
                "per-channel bias",
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- F215

#[test]
fn f215_dropout_does_not_mutate_its_input() -> TorshResult<()> {
    let input = tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let before = input.data()?;

    let _ = dropout(&input, 0.5, true, true)?;

    assert_eq!(
        input.data()?,
        before,
        "inplace=true must not silently mutate a tensor borrowed immutably"
    );
    Ok(())
}

// ---------------------------------------------------------------- InterpolationMode

#[test]
fn interpolation_mode_is_shared_between_resize_and_grid_sample() -> TorshResult<()> {
    let image = tensor((0..16).map(|i| i as f32).collect(), &[1, 1, 4, 4]);
    let mode = InterpolationMode::Nearest;

    let resized = resize(&image, (2, 2), mode, false)?;
    assert_eq!(resized.shape().dims(), &[1, 1, 2, 2]);

    let grid = tensor(
        vec![-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        &[1, 2, 2, 2],
    );
    // The very same enum value is accepted by grid_sample.
    let sampled = grid_sample(&image, &grid, mode, "zeros", true)?;
    assert_eq!(sampled.shape().dims(), &[1, 1, 2, 2]);
    Ok(())
}

#[test]
fn interpolation_bicubic_is_not_silently_bilinear() -> TorshResult<()> {
    let image = tensor(
        (0..36).map(|i| ((i * 7) % 13) as f32).collect(),
        &[1, 1, 6, 6],
    );

    let bilinear = resize(&image, (4, 4), InterpolationMode::Bilinear, false)?.data()?;
    let bicubic = resize(&image, (4, 4), InterpolationMode::Bicubic, false)?.data()?;

    assert_ne!(
        bilinear, bicubic,
        "bicubic resizing must use a cubic kernel, not fall back to bilinear"
    );
    Ok(())
}

#[test]
fn interpolation_unsupported_modes_report_an_error() {
    let image = tensor(vec![0.0; 16], &[1, 1, 4, 4]);
    assert!(
        resize(&image, (2, 2), InterpolationMode::Lanczos, false).is_err(),
        "resize must reject a mode it cannot honour instead of degrading silently"
    );
}
