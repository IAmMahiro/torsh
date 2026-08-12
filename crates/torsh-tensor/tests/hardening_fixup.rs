//! Wave-2 fixup regression tests.
//!
//! Each test here pins a leftover from the first hardening wave:
//!
//! * batched / vector `matmul` must record autograd (the backward rule in
//!   `core_ops::autograd` is rank-generic, the forward gate was not),
//! * `mul_scalar` / `div_scalar` must join the graph instead of producing a
//!   detached leaf that silently swallows the gradient,
//! * the `f32` fast paths of `mul` / `div` must stay numerically identical
//!   while dropping their operand deep-copies,
//! * `stack` must keep its exact interleaving after the materialise-once
//!   rewrite,
//! * `log_softmax` must stay finite for large-magnitude logits.

use torsh_core::device::DeviceType;
use torsh_tensor::Tensor;

/// Build an `f32` CPU tensor.
fn tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
    Tensor::from_data(data, shape, DeviceType::Cpu).expect("tensor creation should succeed")
}

/// Central-difference gradient of `loss` with respect to each element of `values`.
fn numerical_gradient<F>(values: &[f32], loss: F) -> Vec<f32>
where
    F: Fn(&[f32]) -> f32,
{
    const STEP: f32 = 1e-2;
    (0..values.len())
        .map(|index| {
            let mut plus = values.to_vec();
            plus[index] += STEP;
            let mut minus = values.to_vec();
            minus[index] -= STEP;
            (loss(&plus) - loss(&minus)) / (2.0 * STEP)
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Item 1 — batched matmul autograd
// ─────────────────────────────────────────────────────────────────────────────

/// A 3-D @ 3-D product must record the graph and produce gradients that match a
/// finite-difference estimate on both operands.
#[test]
fn batched_matmul_gradients_match_finite_differences() {
    let lhs_values: Vec<f32> = (1..=12).map(|v| v as f32 * 0.25).collect(); // [2, 2, 3]
    let rhs_values: Vec<f32> = (1..=12).map(|v| v as f32 * 0.5 - 2.0).collect(); // [2, 3, 2]

    let lhs = tensor(lhs_values.clone(), vec![2, 2, 3]).requires_grad_(true);
    let rhs = tensor(rhs_values.clone(), vec![2, 3, 2]).requires_grad_(true);

    // Weighted sum so the loss is not symmetric in the output elements.
    let weights: Vec<f32> = (1..=8).map(|v| v as f32 * 0.125).collect();
    let weight_tensor = tensor(weights.clone(), vec![2, 2, 2]);

    let product = lhs
        .matmul(&rhs)
        .expect("batched matmul must record autograd");
    assert_eq!(product.shape().dims(), &[2, 2, 2]);
    let loss = product
        .mul(&weight_tensor)
        .expect("weighting")
        .sum()
        .expect("sum");
    loss.backward().expect("backward");

    let forward = |lhs_data: &[f32], rhs_data: &[f32]| -> f32 {
        let l = tensor(lhs_data.to_vec(), vec![2, 2, 3]);
        let r = tensor(rhs_data.to_vec(), vec![2, 3, 2]);
        let p = l.matmul(&r).expect("matmul");
        p.mul(&weight_tensor)
            .expect("weighting")
            .sum()
            .expect("sum")
            .item()
            .expect("item")
    };

    let lhs_grad = lhs.grad().expect("lhs gradient").to_vec().expect("to_vec");
    let expected_lhs = numerical_gradient(&lhs_values, |v| forward(v, &rhs_values));
    for (index, (&got, &want)) in lhs_grad.iter().zip(expected_lhs.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "lhs grad[{index}] = {got}, finite difference {want}"
        );
    }

    let rhs_grad = rhs.grad().expect("rhs gradient").to_vec().expect("to_vec");
    let expected_rhs = numerical_gradient(&rhs_values, |v| forward(&lhs_values, v));
    for (index, (&got, &want)) in rhs_grad.iter().zip(expected_rhs.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "rhs grad[{index}] = {got}, finite difference {want}"
        );
    }
}

/// A batch axis that is broadcast on one side accumulates over the whole batch.
#[test]
fn batched_matmul_broadcasts_and_accumulates_the_shared_operand() {
    let lhs = tensor((1..=12).map(|v| v as f32).collect(), vec![2, 2, 3]).requires_grad_(true);
    let rhs = tensor((1..=6).map(|v| v as f32).collect(), vec![3, 2]).requires_grad_(true);

    let product = lhs.matmul(&rhs).expect("broadcast batched matmul");
    assert_eq!(product.shape().dims(), &[2, 2, 2]);
    product.sum().expect("sum").backward().expect("backward");

    // rhs is reused by both batches, so its gradient sums the columns of lhs.
    let rhs_grad = rhs.grad().expect("rhs gradient").to_vec().expect("to_vec");
    let lhs_data: Vec<f32> = (1..=12).map(|v| v as f32).collect();
    for p in 0..3usize {
        let expected: f32 = (0..4).map(|row| lhs_data[row * 3 + p]).sum();
        for j in 0..2usize {
            let got = rhs_grad[p * 2 + j];
            assert!(
                (got - expected).abs() < 1e-4,
                "rhs grad[{p},{j}] = {got}, expected {expected}"
            );
        }
    }
}

/// 1-D @ 1-D (dot product) is also rank-generic in the backward rule.
#[test]
fn vector_dot_product_records_gradients() {
    let lhs = tensor(vec![1.0, 2.0, 3.0], vec![3]).requires_grad_(true);
    let rhs = tensor(vec![4.0, 5.0, 6.0], vec![3]).requires_grad_(true);

    let product = lhs.matmul(&rhs).expect("dot product must record autograd");
    product.sum().expect("sum").backward().expect("backward");

    assert_eq!(
        lhs.grad().expect("lhs gradient").to_vec().expect("to_vec"),
        vec![4.0, 5.0, 6.0]
    );
    assert_eq!(
        rhs.grad().expect("rhs gradient").to_vec().expect("to_vec"),
        vec![1.0, 2.0, 3.0]
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Item 2 — scalar multiply / divide autograd
// ─────────────────────────────────────────────────────────────────────────────

/// `d/dx (x * s) = s`, so the gradient must be the scalar, not a dropped edge.
#[test]
fn mul_scalar_propagates_gradients() {
    let x = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).requires_grad_(true);
    let scaled = x.mul_scalar(3.0).expect("mul_scalar");
    scaled.sum().expect("sum").backward().expect("backward");

    let grad = x.grad().expect("mul_scalar must record a graph edge");
    assert_eq!(grad.to_vec().expect("to_vec"), vec![3.0; 4]);
}

/// `d/dx (x / s) = 1 / s`.
#[test]
fn div_scalar_propagates_gradients() {
    let x = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).requires_grad_(true);
    let scaled = x.div_scalar(4.0).expect("div_scalar");
    scaled.sum().expect("sum").backward().expect("backward");

    let grad = x.grad().expect("div_scalar must record a graph edge");
    assert_eq!(grad.to_vec().expect("to_vec"), vec![0.25; 4]);
}

/// Chained scalar ops still compose with the rest of the graph.
#[test]
fn scalar_ops_compose_with_multiplication() {
    let x = tensor(vec![2.0, 3.0], vec![2]).requires_grad_(true);
    // loss = sum((x * 2) * x) = 2 * sum(x^2) -> d/dx = 4x
    let scaled = x.mul_scalar(2.0).expect("mul_scalar");
    let loss = scaled.mul(&x).expect("mul").sum().expect("sum");
    loss.backward().expect("backward");

    let grad = x.grad().expect("gradient").to_vec().expect("to_vec");
    assert!((grad[0] - 8.0).abs() < 1e-4, "grad[0] = {}", grad[0]);
    assert!((grad[1] - 12.0).abs() < 1e-4, "grad[1] = {}", grad[1]);
}

/// A tensor that does not track gradients must stay a detached leaf.
#[test]
fn scalar_ops_without_requires_grad_stay_detached() {
    let x = tensor(vec![1.0, 2.0], vec![2]);
    let scaled = x.mul_scalar(5.0).expect("mul_scalar");
    assert!(!scaled.requires_grad());
    assert_eq!(scaled.to_vec().expect("to_vec"), vec![5.0, 10.0]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Item 3 — mul / div fast paths
// ─────────────────────────────────────────────────────────────────────────────

/// The `f32` vector path (≥ 1024 elements) must agree with the scalar reference
/// and keep recording autograd, including when both operands are the same
/// tensor (shared storage lock).
#[test]
fn large_f32_mul_and_div_match_the_scalar_reference() {
    const N: usize = 4096;
    let a_values: Vec<f32> = (0..N).map(|i| (i % 17) as f32 - 8.0).collect();
    let b_values: Vec<f32> = (0..N).map(|i| (i % 13) as f32 + 1.0).collect();

    let a = tensor(a_values.clone(), vec![N]);
    let b = tensor(b_values.clone(), vec![N]);

    let product = a.mul(&b).expect("mul").to_vec().expect("to_vec");
    let quotient = a.div(&b).expect("div").to_vec().expect("to_vec");
    for i in 0..N {
        assert!(
            (product[i] - a_values[i] * b_values[i]).abs() < 1e-5,
            "mul[{i}]"
        );
        assert!(
            (quotient[i] - a_values[i] / b_values[i]).abs() < 1e-5,
            "div[{i}]"
        );
    }

    // Self-multiplication reads one storage twice: must not deadlock or copy wrong.
    let squared = a.mul(&a).expect("a * a").to_vec().expect("to_vec");
    for i in 0..N {
        assert!(
            (squared[i] - a_values[i] * a_values[i]).abs() < 1e-4,
            "sq[{i}]"
        );
    }
}

/// The same vector path exists for `f64`; it must produce exact results.
#[test]
fn large_f64_mul_and_div_use_the_vector_path() {
    const N: usize = 2048;
    let a_values: Vec<f64> = (0..N).map(|i| (i % 7) as f64 + 0.5).collect();
    let b_values: Vec<f64> = (0..N).map(|i| (i % 5) as f64 + 1.0).collect();

    let a = Tensor::from_data(a_values.clone(), vec![N], DeviceType::Cpu).expect("tensor");
    let b = Tensor::from_data(b_values.clone(), vec![N], DeviceType::Cpu).expect("tensor");

    let product = a.mul(&b).expect("mul").to_vec().expect("to_vec");
    let quotient = a.div(&b).expect("div").to_vec().expect("to_vec");
    for i in 0..N {
        assert!(
            (product[i] - a_values[i] * b_values[i]).abs() < 1e-12,
            "mul[{i}]"
        );
        assert!(
            (quotient[i] - a_values[i] / b_values[i]).abs() < 1e-12,
            "div[{i}]"
        );
    }
}

/// Autograd is still recorded on the vector path.
#[test]
fn large_f32_mul_records_autograd() {
    const N: usize = 2048;
    let a = tensor(vec![2.0f32; N], vec![N]).requires_grad_(true);
    let b = tensor(vec![3.0f32; N], vec![N]);

    let product = a.mul(&b).expect("mul");
    assert!(product.requires_grad());
    product.sum().expect("sum").backward().expect("backward");
    let grad = a.grad().expect("gradient").to_vec().expect("to_vec");
    assert!(grad.iter().all(|&g| (g - 3.0).abs() < 1e-6));
}

// ─────────────────────────────────────────────────────────────────────────────
// Item 4 — stack interleaving
// ─────────────────────────────────────────────────────────────────────────────

/// `stack` must interleave identically for every insertion axis after the
/// materialise-once rewrite.
#[test]
fn stack_interleaves_on_every_axis() {
    let a = tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = tensor(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![2, 3]);

    let front = Tensor::stack(&[a.clone(), b.clone()], 0).expect("stack dim 0");
    assert_eq!(front.shape().dims(), &[2, 2, 3]);
    assert_eq!(
        front.to_vec().expect("to_vec"),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    );

    let middle = Tensor::stack(&[a.clone(), b.clone()], 1).expect("stack dim 1");
    assert_eq!(middle.shape().dims(), &[2, 2, 3]);
    assert_eq!(
        middle.to_vec().expect("to_vec"),
        vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 4.0, 5.0, 6.0, 10.0, 11.0, 12.0]
    );

    let back = Tensor::stack(&[a, b], -1).expect("stack dim -1");
    assert_eq!(back.shape().dims(), &[2, 3, 2]);
    assert_eq!(
        back.to_vec().expect("to_vec"),
        vec![1.0, 7.0, 2.0, 8.0, 3.0, 9.0, 4.0, 10.0, 5.0, 11.0, 6.0, 12.0]
    );
}

/// Stacking strided views (which need materialisation) must still be correct.
#[test]
fn stack_handles_transposed_views() {
    let a = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])
        .transpose(0, 1)
        .expect("transpose");
    let b = tensor(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    let stacked = Tensor::stack(&[a, b], 0).expect("stack");
    assert_eq!(stacked.shape().dims(), &[2, 2, 2]);
    assert_eq!(
        stacked.to_vec().expect("to_vec"),
        vec![1.0, 3.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Item 5 — copy-on-write isolation for lock-free SIMD storage
// ─────────────────────────────────────────────────────────────────────────────

/// A clone must never observe an in-place write made through its sibling, even
/// for tensors large enough to land in lock-free `SimdOptimized` storage.
#[test]
fn in_place_writes_do_not_leak_into_clones_of_large_tensors() {
    // 4096 f32 = 16 KiB, above the SimdOptimized threshold (10 KiB).
    const N: usize = 4096;
    let original = tensor(vec![1.0f32; N], vec![N]);
    let sibling = original.clone();

    let mut writer = original.clone();
    writer.add_scalar_(5.0).expect("in-place add");

    assert!(
        writer.to_vec().expect("to_vec").iter().all(|&v| v == 6.0),
        "the writer must see its own update"
    );
    assert!(
        original.to_vec().expect("to_vec").iter().all(|&v| v == 1.0),
        "the source tensor must be unchanged"
    );
    assert!(
        sibling.to_vec().expect("to_vec").iter().all(|&v| v == 1.0),
        "a sibling clone must be unchanged"
    );
}

/// Repeated in-place writes on an unshared large tensor stay correct.
#[test]
fn repeated_in_place_writes_on_large_tensors_accumulate() {
    const N: usize = 4096;
    let mut tensor_large = tensor(vec![0.0f32; N], vec![N]);
    for _ in 0..3 {
        tensor_large.add_scalar_(2.0).expect("in-place add");
    }
    assert!(tensor_large
        .to_vec()
        .expect("to_vec")
        .iter()
        .all(|&v| (v - 6.0).abs() < 1e-6));
}

// ─────────────────────────────────────────────────────────────────────────────
// Item 9 — log_softmax numerical stability
// ─────────────────────────────────────────────────────────────────────────────

/// `log(softmax(x))` underflows to `-inf` as soon as one logit is ~90 below the
/// row maximum. `log_softmax` must use the log-sum-exp identity instead.
#[test]
fn log_softmax_stays_finite_for_large_logits() {
    let logits = tensor(vec![0.0, -200.0, 150.0, -400.0], vec![1, 4]);
    let log_probs = logits.log_softmax(-1).expect("log_softmax");
    let values = log_probs.to_vec().expect("to_vec");

    for (index, &value) in values.iter().enumerate() {
        assert!(
            value.is_finite(),
            "log_softmax[{index}] = {value} is not finite"
        );
    }

    // log_softmax(x) = x - max - log(sum(exp(x - max))); here the max dominates
    // the sum so the shift is ~0.
    let expected = [-150.0f32, -350.0, 0.0, -550.0];
    for (index, (&got, &want)) in values.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-2,
            "log_softmax[{index}] = {got}, expected ~{want}"
        );
    }
}

/// On well-scaled inputs it must still agree with `log(softmax(x))`.
#[test]
fn log_softmax_matches_log_of_softmax_for_small_logits() {
    let logits = tensor(vec![1.0, 2.0, 3.0, 0.5, -1.0, 0.0], vec![2, 3]);
    let log_probs = logits
        .log_softmax(-1)
        .expect("log_softmax")
        .to_vec()
        .expect("to_vec");
    let reference = logits
        .softmax(-1)
        .expect("softmax")
        .log()
        .expect("log")
        .to_vec()
        .expect("to_vec");

    for (index, (&got, &want)) in log_probs.iter().zip(reference.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-5,
            "log_softmax[{index}] = {got}, log(softmax) = {want}"
        );
    }

    // Each row must exponentiate back to a probability distribution.
    for row in 0..2usize {
        let total: f32 = (0..3).map(|c| log_probs[row * 3 + c].exp()).sum();
        assert!((total - 1.0).abs() < 1e-5, "row {row} sums to {total}");
    }
}
