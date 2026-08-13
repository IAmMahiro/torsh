//! Hardening regression tests for `Tensor::sum_dim`'s autograd recording.
//!
//! Before this fix, `sum_dim` always returned a fresh `Operation::Leaf` on both
//! its single-axis and multi-axis code paths (`dim_ops.rs`), silently severing
//! the autograd graph regardless of `requires_grad`. `Operation::Mean` papered
//! over part of this by recording its own node, but its backward rule reused
//! `Sum`'s right-aligned `expand_grad_to_shape` directly, which for a
//! non-keepdim reduction re-attaches the gradient to the wrong axis whenever
//! the shapes happen to align (a square input is the discriminating case, and
//! every non-square case passes by accident).
//!
//! Every test here pins the real backward rule -- `Operation::SumDim` -- with
//! an analytic or finite-difference check, and several exercise the ops that
//! compose on top of `sum_dim` (`mean`, `var`, `std`, `mean_stats`, softmax, an
//! NLL-style reduction) so the fix is verified where it actually matters.

use torsh_core::device::DeviceType;
use torsh_tensor::stats::StatMode;
use torsh_tensor::Tensor;

fn t32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
    Tensor::from_data(data, shape, DeviceType::Cpu).expect("f32 tensor creation should succeed")
}

/// Distinct, non-uniform ramp so a mis-split / mis-scatter backward cannot pass
/// by symmetry.
fn ramp(shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 + 1.0).collect();
    t32(data, shape.to_vec())
}

/// Central finite-difference gradient of a scalar loss w.r.t. every element of
/// `base` (which carries `shape`).
fn fd_grad<F>(base: &[f32], shape: &[usize], loss: F) -> Vec<f32>
where
    F: Fn(&Tensor<f32>) -> f32,
{
    let eps = 1e-3_f32;
    let mut g = vec![0.0_f32; base.len()];
    for i in 0..base.len() {
        let mut plus = base.to_vec();
        let mut minus = base.to_vec();
        plus[i] += eps;
        minus[i] -= eps;
        let lp = loss(&t32(plus, shape.to_vec()));
        let lm = loss(&t32(minus, shape.to_vec()));
        g[i] = (lp - lm) / (2.0 * eps);
    }
    g
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, ctx: &str) {
    assert_eq!(a.len(), b.len(), "{ctx}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() <= tol,
            "{ctx}: index {i}: {x} vs {y} (tol {tol})"
        );
    }
}

// ---------------------------------------------------------------------------
// 1. Single-axis, non-keepdim.
// ---------------------------------------------------------------------------

#[test]
fn sum_dim_backward_nonkeepdim_matches_fd() {
    let shape = [2usize, 3];
    let base = ramp(&shape).to_vec().unwrap();
    let weight = t32(vec![1.3, -2.1], vec![2]);
    let weight_for_loss = weight.clone();

    let loss = move |xx: &Tensor<f32>| -> f32 {
        xx.sum_dim(&[1], false)
            .expect("sum_dim")
            .mul(&weight_for_loss)
            .expect("mul")
            .sum()
            .expect("sum")
            .item()
            .expect("item")
    };

    let x = t32(base.clone(), shape.to_vec()).requires_grad_(true);
    let y = x.sum_dim(&[1], false).expect("sum_dim");
    assert!(
        y.requires_grad(),
        "sum_dim must record autograd (pre-fix this is false and backward() errors)"
    );
    y.mul(&weight)
        .expect("mul")
        .sum()
        .expect("sum")
        .backward()
        .expect("sum_dim nonkeepdim backward");

    let analytic = x
        .grad()
        .expect("x.grad() must exist (sum_dim severed the graph)")
        .to_vec()
        .unwrap();
    let numeric = fd_grad(&base, &shape, loss);
    assert_close(&analytic, &numeric, 2e-2, "sum_dim nonkeepdim grad vs FD");
}

// ---------------------------------------------------------------------------
// 2. Single-axis, keepdim.
// ---------------------------------------------------------------------------

#[test]
fn sum_dim_backward_keepdim_matches_fd() {
    let shape = [2usize, 3];
    let base = ramp(&shape).to_vec().unwrap();
    let weight = t32(vec![1.3, -2.1], vec![2, 1]);
    let weight_for_loss = weight.clone();

    let loss = move |xx: &Tensor<f32>| -> f32 {
        xx.sum_dim(&[1], true)
            .expect("sum_dim")
            .mul(&weight_for_loss)
            .expect("mul")
            .sum()
            .expect("sum")
            .item()
            .expect("item")
    };

    let x = t32(base.clone(), shape.to_vec()).requires_grad_(true);
    let y = x.sum_dim(&[1], true).expect("sum_dim");
    assert!(
        y.requires_grad(),
        "sum_dim(keepdim=true) must record autograd"
    );
    y.mul(&weight)
        .expect("mul")
        .sum()
        .expect("sum")
        .backward()
        .expect("sum_dim keepdim backward");

    let analytic = x
        .grad()
        .expect("x.grad() must exist (sum_dim severed the graph)")
        .to_vec()
        .unwrap();
    let numeric = fd_grad(&base, &shape, loss);
    assert_close(&analytic, &numeric, 2e-2, "sum_dim keepdim grad vs FD");
}

// ---------------------------------------------------------------------------
// 3. Multi-axis path (dim_ops.rs:279-322), both keepdim values.
// ---------------------------------------------------------------------------

#[test]
fn sum_dim_backward_multi_axis() {
    let shape = [2usize, 3, 4];
    let base = ramp(&shape).to_vec().unwrap();

    // keepdim = false: reduces dims 0 and 2, leaving shape [3].
    {
        let weight = t32(vec![0.6, -1.4, 2.2], vec![3]);
        let weight_for_loss = weight.clone();
        let loss = move |xx: &Tensor<f32>| -> f32 {
            xx.sum_dim(&[0, 2], false)
                .expect("sum_dim")
                .mul(&weight_for_loss)
                .expect("mul")
                .sum()
                .expect("sum")
                .item()
                .expect("item")
        };

        let x = t32(base.clone(), shape.to_vec()).requires_grad_(true);
        let y = x.sum_dim(&[0, 2], false).expect("sum_dim");
        assert!(y.requires_grad(), "multi-axis sum_dim must record autograd");
        y.mul(&weight)
            .expect("mul")
            .sum()
            .expect("sum")
            .backward()
            .expect("multi-axis backward (keepdim=false)");
        let analytic = x.grad().expect("x.grad() must exist").to_vec().unwrap();
        let numeric = fd_grad(&base, &shape, loss);
        assert_close(
            &analytic,
            &numeric,
            2e-2,
            "multi-axis sum_dim grad vs FD (keepdim=false)",
        );
    }

    // keepdim = true: same reduction, shape [1,3,1].
    {
        let weight = t32(vec![0.6, -1.4, 2.2], vec![1, 3, 1]);
        let weight_for_loss = weight.clone();
        let loss = move |xx: &Tensor<f32>| -> f32 {
            xx.sum_dim(&[0, 2], true)
                .expect("sum_dim")
                .mul(&weight_for_loss)
                .expect("mul")
                .sum()
                .expect("sum")
                .item()
                .expect("item")
        };

        let x = t32(base.clone(), shape.to_vec()).requires_grad_(true);
        let y = x.sum_dim(&[0, 2], true).expect("sum_dim");
        assert!(
            y.requires_grad(),
            "multi-axis sum_dim(keepdim=true) must record autograd"
        );
        y.mul(&weight)
            .expect("mul")
            .sum()
            .expect("sum")
            .backward()
            .expect("multi-axis backward (keepdim=true)");
        let analytic = x.grad().expect("x.grad() must exist").to_vec().unwrap();
        let numeric = fd_grad(&base, &shape, loss);
        assert_close(
            &analytic,
            &numeric,
            2e-2,
            "multi-axis sum_dim grad vs FD (keepdim=true)",
        );
    }
}

// ---------------------------------------------------------------------------
// 4. THE DISCRIMINATING TEST: a square input.
// ---------------------------------------------------------------------------

/// A naive right-aligned `expand_grad_to_shape` call (i.e. an implementation
/// that skips the reshape-to-keepdim step) replicates the gradient along the
/// wrong axis whenever the input happens to be square, producing a
/// column-constant gradient instead of the correct row-constant one -- and
/// every non-square test above passes regardless (verified: probe INV-D,
/// `expand([3] -> [3,3])` replicates along dim0).
#[test]
fn sum_dim_backward_square_shape_uses_the_right_axis() {
    let shape = [3usize, 3];
    let x = ramp(&shape).requires_grad_(true);
    let w = t32(vec![1.0, 10.0, 100.0], vec![3]);
    let y = x.sum_dim(&[1], false).expect("sum_dim");
    assert!(
        y.requires_grad(),
        "square-shape sum_dim must record autograd"
    );
    y.mul(&w)
        .expect("mul")
        .sum()
        .expect("sum")
        .backward()
        .expect("square-shape backward");

    let grad = x.grad().expect("x.grad() must exist").to_vec().unwrap();
    let expected: Vec<f32> = vec![
        1.0, 1.0, 1.0, //
        10.0, 10.0, 10.0, //
        100.0, 100.0, 100.0,
    ];
    assert_close(
        &grad,
        &expected,
        1e-5,
        "square-shape grad must be row-constant, not column-constant (transposed)",
    );
}

// ---------------------------------------------------------------------------
// 5. Negative dim indices normalise the same as positive ones.
// ---------------------------------------------------------------------------

#[test]
fn sum_dim_backward_negative_dim_equals_positive() {
    let shape = [2usize, 3, 4];
    for &keepdim in &[false, true] {
        let x_neg = ramp(&shape).requires_grad_(true);
        let y_neg = x_neg.sum_dim(&[-1], keepdim).expect("sum_dim(-1)");
        let out_shape = y_neg.shape().dims().to_vec();
        let seed = ramp(&out_shape);

        let x_pos = ramp(&shape).requires_grad_(true);
        let y_pos = x_pos.sum_dim(&[2], keepdim).expect("sum_dim(2)");

        y_neg
            .backward_with_grad(Some(&seed))
            .expect("sum_dim(-1) backward");
        y_pos
            .backward_with_grad(Some(&seed))
            .expect("sum_dim(2) backward");

        let g_neg = x_neg.grad().expect("x_neg grad").to_vec().unwrap();
        let g_pos = x_pos.grad().expect("x_pos grad").to_vec().unwrap();
        assert_close(
            &g_neg,
            &g_pos,
            1e-6,
            &format!("sum_dim(-1) vs sum_dim(2), keepdim={keepdim}"),
        );
    }
}

// ---------------------------------------------------------------------------
// 6. Duplicate dim indices dedup the same on the backward path as forward.
// ---------------------------------------------------------------------------

#[test]
fn sum_dim_backward_duplicate_dims_equal_single() {
    let shape = [2usize, 3, 4];
    for &keepdim in &[false, true] {
        let x_dup = ramp(&shape).requires_grad_(true);
        let y_dup = x_dup.sum_dim(&[0, 0], keepdim).expect("sum_dim([0,0])");
        let out_shape = y_dup.shape().dims().to_vec();
        let seed = ramp(&out_shape);

        let x_single = ramp(&shape).requires_grad_(true);
        let y_single = x_single.sum_dim(&[0], keepdim).expect("sum_dim([0])");

        y_dup
            .backward_with_grad(Some(&seed))
            .expect("sum_dim([0,0]) backward");
        y_single
            .backward_with_grad(Some(&seed))
            .expect("sum_dim([0]) backward");

        let g_dup = x_dup.grad().expect("x_dup grad").to_vec().unwrap();
        let g_single = x_single.grad().expect("x_single grad").to_vec().unwrap();
        assert_close(
            &g_dup,
            &g_single,
            1e-6,
            &format!("sum_dim([0,0]) vs sum_dim([0]), keepdim={keepdim}"),
        );
    }
}

// ---------------------------------------------------------------------------
// 7. An empty `dims` slice must still delegate to `sum()`'s existing record.
// ---------------------------------------------------------------------------

#[test]
fn sum_dim_empty_dims_still_flows_through_whole_tensor_sum() {
    let shape = [2usize, 3];
    let x_empty = ramp(&shape).requires_grad_(true);
    let y_empty = x_empty.sum_dim(&[], false).expect("sum_dim(&[])");
    y_empty.backward().expect("sum_dim(&[]) backward");

    let x_sum = ramp(&shape).requires_grad_(true);
    let y_sum = x_sum.sum().expect("sum");
    y_sum.backward().expect("sum backward");

    let g_empty = x_empty.grad().expect("x_empty grad").to_vec().unwrap();
    let g_sum = x_sum.grad().expect("x_sum grad").to_vec().unwrap();
    assert_close(&g_empty, &g_sum, 1e-6, "sum_dim(&[]) vs sum() grad");
}

// ---------------------------------------------------------------------------
// 8. keepdim=true/false write the identical buffer -- the invariant the
//    backward rule's reshape-then-expand strategy rests on.
// ---------------------------------------------------------------------------

#[test]
fn sum_dim_keepdim_and_nonkeepdim_share_one_buffer() {
    let cases: Vec<(Vec<usize>, Vec<i32>, Vec<usize>)> = vec![
        (vec![2, 3], vec![1], vec![2, 1]),
        (vec![2, 3, 4], vec![0, 2], vec![1, 3, 1]),
        (vec![2, 3, 4, 5], vec![1, 3], vec![2, 1, 4, 1]),
    ];
    for (shape, dims, expected_keepdim_shape) in cases {
        let x = ramp(&shape);
        let non_keep = x
            .sum_dim(&dims, false)
            .expect("sum_dim false")
            .to_vec()
            .unwrap();
        let keep = x.sum_dim(&dims, true).expect("sum_dim true");
        assert_eq!(
            keep.shape().dims(),
            expected_keepdim_shape.as_slice(),
            "keepdim shape mismatch for {shape:?} dims={dims:?}"
        );
        assert_eq!(
            non_keep,
            keep.to_vec().unwrap(),
            "sum_dim keepdim/non-keepdim buffers diverge for {shape:?} dims={dims:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// 9. `mean()`'s composed sum_dim -> div_scalar graph, all four (dims,keepdim)
//    combinations -- the regression pin for deleting the `Operation::Mean`
//    producer.
// ---------------------------------------------------------------------------

#[test]
fn mean_dim_backward_matches_fd_all_four_combinations() {
    // (a) [3,3], dims=[1], keepdim=false -- the headline regression: this used
    // to come back transposed (column-constant) with no error at all.
    {
        let shape = [3usize, 3];
        let base = ramp(&shape).to_vec().unwrap();
        let weight = t32(vec![1.0, 10.0, 100.0], vec![3]);
        let weight_for_loss = weight.clone();
        let loss = move |xx: &Tensor<f32>| -> f32 {
            xx.mean(Some(&[1]), false)
                .expect("mean")
                .mul(&weight_for_loss)
                .expect("mul")
                .sum()
                .expect("sum")
                .item()
                .expect("item")
        };

        let x = t32(base.clone(), shape.to_vec()).requires_grad_(true);
        let y = x.mean(Some(&[1]), false).expect("mean");
        y.mul(&weight)
            .expect("mul")
            .sum()
            .expect("sum")
            .backward()
            .expect("mean([3,3],dims=[1]) backward");
        let analytic = x.grad().expect("x.grad() must exist").to_vec().unwrap();

        let expected_exact: Vec<f32> = vec![
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
            10.0 / 3.0,
            10.0 / 3.0,
            10.0 / 3.0,
            100.0 / 3.0,
            100.0 / 3.0,
            100.0 / 3.0,
        ];
        assert_close(
            &analytic,
            &expected_exact,
            1e-3,
            "mean([3,3],dims=[1]) grad must be row-constant, not transposed",
        );
        let numeric = fd_grad(&base, &shape, loss);
        assert_close(&analytic, &numeric, 2e-2, "mean([3,3],dims=[1]) grad vs FD");
    }

    // (b) [2,3], dims=[1], keepdim=false -- used to hard-error ("mean backward
    // cannot map a [2] gradient onto a [2, 3] input").
    {
        let shape = [2usize, 3];
        let base = ramp(&shape).to_vec().unwrap();
        let weight = t32(vec![0.7, -1.9], vec![2]);
        let weight_for_loss = weight.clone();
        let loss = move |xx: &Tensor<f32>| -> f32 {
            xx.mean(Some(&[1]), false)
                .expect("mean")
                .mul(&weight_for_loss)
                .expect("mul")
                .sum()
                .expect("sum")
                .item()
                .expect("item")
        };

        let x = t32(base.clone(), shape.to_vec()).requires_grad_(true);
        let y = x.mean(Some(&[1]), false).expect("mean");
        y.mul(&weight)
            .expect("mul")
            .sum()
            .expect("sum")
            .backward()
            .expect("mean([2,3],dims=[1]) backward (pre-fix: hard error)");
        let analytic = x.grad().expect("x.grad() must exist").to_vec().unwrap();
        let numeric = fd_grad(&base, &shape, loss);
        assert_close(&analytic, &numeric, 2e-2, "mean([2,3],dims=[1]) grad vs FD");
    }

    // (c) [2,3,4], dims=[0,2], keepdim=true.
    {
        let shape = [2usize, 3, 4];
        let base = ramp(&shape).to_vec().unwrap();
        let weight = t32(vec![0.4, -0.9, 1.6], vec![1, 3, 1]);
        let weight_for_loss = weight.clone();
        let loss = move |xx: &Tensor<f32>| -> f32 {
            xx.mean(Some(&[0, 2]), true)
                .expect("mean")
                .mul(&weight_for_loss)
                .expect("mul")
                .sum()
                .expect("sum")
                .item()
                .expect("item")
        };

        let x = t32(base.clone(), shape.to_vec()).requires_grad_(true);
        let y = x.mean(Some(&[0, 2]), true).expect("mean");
        y.mul(&weight)
            .expect("mul")
            .sum()
            .expect("sum")
            .backward()
            .expect("mean([2,3,4],dims=[0,2],keepdim=true) backward");
        let analytic = x.grad().expect("x.grad() must exist").to_vec().unwrap();
        let numeric = fd_grad(&base, &shape, loss);
        assert_close(
            &analytic,
            &numeric,
            2e-2,
            "mean multi-axis keepdim grad vs FD",
        );
    }

    // (d) None reduction, both keepdim values -- keepdim=true pins the
    // Sum -> View{Reshape} -> DivScalar chain, the one path not covered above.
    for &keepdim in &[false, true] {
        let shape = [2usize, 3];
        let x = ramp(&shape).requires_grad_(true);
        let y = x.mean(None, keepdim).expect("mean(None)");
        y.backward().expect("mean(None) backward");
        let grad = x.grad().expect("x.grad() must exist").to_vec().unwrap();
        let expected = vec![1.0f32 / 6.0; 6];
        assert_close(
            &grad,
            &expected,
            1e-4,
            &format!("mean(None, keepdim={keepdim}) grad"),
        );
    }
}

// ---------------------------------------------------------------------------
// 10. var / std / mean_stats become differentiable for free once sum_dim
//     records and the Operation::Mean overwrite is removed.
// ---------------------------------------------------------------------------

#[test]
fn var_and_std_dim_are_differentiable() {
    let shape = [2usize, 3];
    let base = ramp(&shape).to_vec().unwrap();

    // var, Population mode, keepdim=false -> output shape [2].
    {
        let weight = t32(vec![0.8, -1.6], vec![2]);
        let weight_for_loss = weight.clone();
        let loss = move |xx: &Tensor<f32>| -> f32 {
            xx.var(Some(&[1]), false, StatMode::Population)
                .expect("var")
                .mul(&weight_for_loss)
                .expect("mul")
                .sum()
                .expect("sum")
                .item()
                .expect("item")
        };

        let x = t32(base.clone(), shape.to_vec()).requires_grad_(true);
        let y = x.var(Some(&[1]), false, StatMode::Population).expect("var");
        assert!(
            y.requires_grad(),
            "var(Some(dims)) must record autograd (pre-fix this is false)"
        );
        y.mul(&weight)
            .expect("mul")
            .sum()
            .expect("sum")
            .backward()
            .expect("var backward");
        let analytic = x
            .grad()
            .expect("var input grad must exist")
            .to_vec()
            .unwrap();
        let numeric = fd_grad(&base, &shape, loss);
        assert_close(
            &analytic,
            &numeric,
            2e-2,
            "var(Population,keepdim=false) grad vs FD",
        );
    }

    // var, Sample mode, keepdim=true -> output shape [2,1].
    {
        let weight = t32(vec![0.8, -1.6], vec![2, 1]);
        let weight_for_loss = weight.clone();
        let loss = move |xx: &Tensor<f32>| -> f32 {
            xx.var(Some(&[1]), true, StatMode::Sample)
                .expect("var")
                .mul(&weight_for_loss)
                .expect("mul")
                .sum()
                .expect("sum")
                .item()
                .expect("item")
        };

        let x = t32(base.clone(), shape.to_vec()).requires_grad_(true);
        let y = x.var(Some(&[1]), true, StatMode::Sample).expect("var");
        assert!(
            y.requires_grad(),
            "var(Some(dims),keepdim=true) must record autograd"
        );
        y.mul(&weight)
            .expect("mul")
            .sum()
            .expect("sum")
            .backward()
            .expect("var backward");
        let analytic = x
            .grad()
            .expect("var input grad must exist")
            .to_vec()
            .unwrap();
        let numeric = fd_grad(&base, &shape, loss);
        assert_close(
            &analytic,
            &numeric,
            2e-2,
            "var(Sample,keepdim=true) grad vs FD",
        );
    }

    // std, Population mode, keepdim=false.
    {
        let weight = t32(vec![0.8, -1.6], vec![2]);
        let weight_for_loss = weight.clone();
        let loss = move |xx: &Tensor<f32>| -> f32 {
            xx.std(Some(&[1]), false, StatMode::Population)
                .expect("std")
                .mul(&weight_for_loss)
                .expect("mul")
                .sum()
                .expect("sum")
                .item()
                .expect("item")
        };

        let x = t32(base.clone(), shape.to_vec()).requires_grad_(true);
        let y = x.std(Some(&[1]), false, StatMode::Population).expect("std");
        assert!(y.requires_grad(), "std(Some(dims)) must record autograd");
        y.mul(&weight)
            .expect("mul")
            .sum()
            .expect("sum")
            .backward()
            .expect("std backward");
        let analytic = x
            .grad()
            .expect("std input grad must exist")
            .to_vec()
            .unwrap();
        let numeric = fd_grad(&base, &shape, loss);
        assert_close(
            &analytic,
            &numeric,
            2e-2,
            "std(Population,keepdim=false) grad vs FD",
        );
    }

    // mean_stats, keepdim=false.
    {
        let weight = t32(vec![0.8, -1.6], vec![2]);
        let weight_for_loss = weight.clone();
        let loss = move |xx: &Tensor<f32>| -> f32 {
            xx.mean_stats(Some(&[1]), false)
                .expect("mean_stats")
                .mul(&weight_for_loss)
                .expect("mul")
                .sum()
                .expect("sum")
                .item()
                .expect("item")
        };

        let x = t32(base.clone(), shape.to_vec()).requires_grad_(true);
        let y = x.mean_stats(Some(&[1]), false).expect("mean_stats");
        assert!(
            y.requires_grad(),
            "mean_stats(Some(dims)) must record autograd"
        );
        y.mul(&weight)
            .expect("mul")
            .sum()
            .expect("sum")
            .backward()
            .expect("mean_stats backward");
        let analytic = x
            .grad()
            .expect("mean_stats input grad must exist")
            .to_vec()
            .unwrap();
        let numeric = fd_grad(&base, &shape, loss);
        assert_close(&analytic, &numeric, 2e-2, "mean_stats grad vs FD");
    }
}

// ---------------------------------------------------------------------------
// 11. The headline correctness pin: a hand-built softmax whose denominator is
//     `sum_dim(exp, keepdim=true)`, exactly mirroring
//     `torsh_nn::layers::activation::softmax::Softmax::forward`.
// ---------------------------------------------------------------------------

#[test]
fn softmax_composition_gradient_matches_fd() {
    let shape = [2usize, 3];
    let base = ramp(&shape).to_vec().unwrap();
    let weight = t32(vec![0.4, -1.1, 2.3, -0.6, 1.7, -2.2], shape.to_vec());
    let weight_for_loss = weight.clone();

    let loss = move |xx: &Tensor<f32>| -> f32 {
        let m = xx.max_dim(1, true).expect("max_dim");
        let e = xx.sub(&m).expect("sub").exp().expect("exp");
        let s = e.sum_dim(&[1], true).expect("sum_dim");
        let y = e.div(&s).expect("div");
        y.mul(&weight_for_loss)
            .expect("mul")
            .sum()
            .expect("sum")
            .item()
            .expect("item")
    };

    let x = t32(base.clone(), shape.to_vec()).requires_grad_(true);
    let m = x.max_dim(1, true).expect("max_dim");
    let e = x.sub(&m).expect("sub").exp().expect("exp");
    let s = e.sum_dim(&[1], true).expect("sum_dim");
    let y = e.div(&s).expect("div");
    y.mul(&weight)
        .expect("mul")
        .sum()
        .expect("sum")
        .backward()
        .expect("softmax composition backward");

    let analytic = x
        .grad()
        .expect("x.grad() must exist (softmax denominator was detached)")
        .to_vec()
        .unwrap();
    let numeric = fd_grad(&base, &shape, loss);
    assert_close(&analytic, &numeric, 2e-2, "softmax composition grad vs FD");
}

// ---------------------------------------------------------------------------
// 12. An NLL-style reduction (mirrors the negative-log-likelihood computation
//     inside `crates/torsh-nn/src/functional/loss.rs:56`'s `cross_entropy`).
// ---------------------------------------------------------------------------

/// `mul_scalar(-1.0)` stands in for `Tensor::neg()`: `neg()` (and the `Neg`
/// operator overload) go through `map()`, which propagates `requires_grad` but
/// leaves `operation: Operation::Leaf` -- a *second*, independent autograd gap
/// (a `map()`-derived tensor with `requires_grad=true` silently swallows its
/// gradient into its own, otherwise-unreachable `grad` slot instead of
/// forwarding it upstream). That gap is outside this fix's ownership
/// (`math_ops_trig.rs` / `math_ops.rs`, not `dim_ops.rs`/`autograd.rs`), so
/// `mul_scalar` -- which *does* record (`Operation::MulScalar`) -- isolates
/// this test to the `sum_dim`/`mean` fix under test. See the task report for
/// the empirical confirmation and the `cross_entropy` follow-up this implies.
#[test]
fn nll_style_reduction_backward_flows() {
    let batch = 2usize;
    let classes = 3usize;
    let onehot = t32(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], vec![batch, classes]);
    let logp = ramp(&[batch, classes]).requires_grad_(true);

    let loss = logp
        .mul_op(&onehot)
        .expect("mul_op")
        .mul_scalar(-1.0)
        .expect("mul_scalar(-1.0)")
        .sum_dim(&[-1], false)
        .expect("sum_dim")
        .mean(None, false)
        .expect("mean");
    loss.backward()
        .expect("nll-style backward (pre-fix: severed sum_dim -> Err)");

    let grad = logp
        .grad()
        .expect("logp.grad() must exist after nll-style backward")
        .to_vec()
        .unwrap();
    let expected: Vec<f32> = onehot
        .to_vec()
        .unwrap()
        .iter()
        .map(|&v| -v / batch as f32)
        .collect();
    assert_close(
        &grad,
        &expected,
        1e-5,
        "nll-style grad vs -onehot/batch_size",
    );
}

// ---------------------------------------------------------------------------
// 13. Gradient mode gate: an input that does not require grad must still
//     produce a plain leaf.
// ---------------------------------------------------------------------------

/// `torsh-autograd`'s `no_grad()` guard is not a dev-dependency of
/// `torsh-tensor` (it is commented out in `Cargo.toml`), so the grad-mode half
/// of `should_record_grad` cannot be exercised from this crate's tests; only
/// the `requires_grad=false` input case is reachable here. The mode-gate
/// itself is still exercised indirectly by every other test in this file,
/// each of which requires `y.requires_grad()` to flip to `true`.
#[test]
fn sum_dim_records_nothing_without_grad_mode() {
    let x = ramp(&[2, 3]);
    assert!(!x.requires_grad(), "test setup: x must not require grad");
    let y = x.sum_dim(&[1], false).expect("sum_dim");
    assert!(
        !y.requires_grad(),
        "sum_dim must not record when the input does not require grad"
    );
}
