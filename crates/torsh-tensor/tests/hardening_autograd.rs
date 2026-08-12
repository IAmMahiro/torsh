//! Production-hardening regression tests for the autograd engine and the
//! stride-aware element accessors of `torsh-tensor`.
//!
//! Findings covered: F059/F072 (mul/div never joined the graph), F151/F160
//! (broadcast gradients kept the broadcast shape), F150/F274 (`backward_with_grad`
//! was a silent no-op), F071 (recursive backward without a visited set), F159
//! (flat accessors ignored strides and the storage offset).

use std::time::Instant;

use torsh_core::device::DeviceType;
use torsh_core::error::Result;
use torsh_tensor::Tensor;

/// Build an `f32` CPU tensor.
fn tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
    Tensor::from_data(data, shape, DeviceType::Cpu).expect("tensor creation should succeed")
}

/// Central-difference gradient of `loss` with respect to each element of `values`.
///
/// `backward()` is only implemented for element types convertible to `f32`, so
/// the checks run in single precision: a comparatively large step keeps the
/// truncation error well below the rounding error.
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

/// Assert two gradient vectors agree within a relative tolerance.
fn assert_close(analytic: &[f32], numeric: &[f32], what: &str) {
    assert_eq!(analytic.len(), numeric.len(), "{what}: length mismatch");
    for (index, (got, expected)) in analytic.iter().zip(numeric.iter()).enumerate() {
        let tolerance = 2e-2 * expected.abs().max(1.0);
        assert!(
            (got - expected).abs() <= tolerance,
            "{what}: gradient[{index}] = {got}, finite differences gave {expected}"
        );
    }
}

// ---------------------------------------------------------------------------
// F059 / F072 - mul and div must participate in the autograd graph
// ---------------------------------------------------------------------------

#[test]
fn f059_mul_records_autograd_and_matches_finite_differences() -> Result<()> {
    let a_data = vec![1.0f32, -2.0, 3.0, 0.5, 2.5, -1.5];
    let b_data = vec![0.5f32, 1.5, -2.0, 4.0, -0.25, 3.0];
    let a = tensor(a_data.clone(), vec![2, 3]).requires_grad_(true);
    let b = tensor(b_data.clone(), vec![2, 3]).requires_grad_(true);

    let product = a.mul(&b)?;
    assert!(
        product.requires_grad(),
        "mul must propagate requires_grad to its result"
    );

    let loss = product.sum()?;
    loss.backward()?;

    let a_grad = a
        .grad()
        .expect("mul must produce a gradient for lhs")
        .to_vec()?;
    let b_grad = b
        .grad()
        .expect("mul must produce a gradient for rhs")
        .to_vec()?;

    let numeric_a = numerical_gradient(&a_data, |values| {
        values.iter().zip(b_data.iter()).map(|(x, y)| x * y).sum()
    });
    let numeric_b = numerical_gradient(&b_data, |values| {
        values.iter().zip(a_data.iter()).map(|(y, x)| x * y).sum()
    });
    assert_close(&a_grad, &numeric_a, "mul d/dlhs");
    assert_close(&b_grad, &numeric_b, "mul d/drhs");
    Ok(())
}

#[test]
fn f072_div_records_autograd_and_matches_finite_differences() -> Result<()> {
    let a_data = vec![1.0f32, -2.0, 3.0, 0.5];
    // Denominators stay away from zero so the finite differences stay stable.
    let b_data = vec![2.0f32, 1.5, -2.5, 4.0];
    let a = tensor(a_data.clone(), vec![2, 2]).requires_grad_(true);
    let b = tensor(b_data.clone(), vec![2, 2]).requires_grad_(true);

    let quotient = a.div(&b)?;
    assert!(
        quotient.requires_grad(),
        "div must propagate requires_grad to its result"
    );

    quotient.sum()?.backward()?;

    let a_grad = a
        .grad()
        .expect("div must produce a gradient for lhs")
        .to_vec()?;
    let b_grad = b
        .grad()
        .expect("div must produce a gradient for rhs")
        .to_vec()?;

    let numeric_a = numerical_gradient(&a_data, |values| {
        values.iter().zip(b_data.iter()).map(|(x, y)| x / y).sum()
    });
    let numeric_b = numerical_gradient(&b_data, |values| {
        values.iter().zip(a_data.iter()).map(|(y, x)| x / y).sum()
    });
    assert_close(&a_grad, &numeric_a, "div d/dlhs");
    assert_close(&b_grad, &numeric_b, "div d/drhs");
    Ok(())
}

#[test]
fn f059_mul_gradients_are_not_differentiable_tensors() -> Result<()> {
    // The backward pass must not extend the graph it is walking: a stored
    // gradient is a plain leaf, otherwise it would pin the whole graph alive.
    let a = tensor(vec![2.0, 3.0], vec![2]).requires_grad_(true);
    let b = tensor(vec![4.0, 5.0], vec![2]).requires_grad_(true);
    a.mul(&b)?.sum()?.backward()?;
    let grad = a.grad().expect("gradient");
    assert!(
        !grad.requires_grad(),
        "stored gradients must not require gradients themselves"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// F151 / F160 - broadcast gradients must be reduced to the operand shape
// ---------------------------------------------------------------------------

#[test]
fn f160_bias_add_gradient_keeps_the_bias_shape() -> Result<()> {
    let x = tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).requires_grad_(true);
    let bias = tensor(vec![0.5, 0.5, 0.5], vec![3]).requires_grad_(true);

    x.add(&bias)?.sum()?.backward()?;

    let bias_grad = bias.grad().expect("bias gradient");
    assert_eq!(
        bias_grad.shape().dims(),
        &[3],
        "the bias gradient must have the bias shape, not the activation shape"
    );
    assert_eq!(bias_grad.to_vec()?, vec![2.0, 2.0, 2.0]);

    let x_grad = x.grad().expect("input gradient");
    assert_eq!(x_grad.shape().dims(), &[2, 3]);
    assert_eq!(x_grad.to_vec()?, vec![1.0; 6]);
    Ok(())
}

#[test]
fn f151_add_broadcast_gradients_are_reduced_on_both_sides() -> Result<()> {
    let a = tensor(vec![1.0, 2.0, 3.0], vec![3, 1]).requires_grad_(true);
    let b = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).requires_grad_(true);

    let sum = a.add(&b)?;
    assert_eq!(sum.shape().dims(), &[3, 4]);
    sum.sum()?.backward()?;

    let a_grad = a.grad().expect("lhs gradient");
    assert_eq!(a_grad.shape().dims(), &[3, 1]);
    assert_eq!(a_grad.to_vec()?, vec![4.0, 4.0, 4.0]);

    let b_grad = b.grad().expect("rhs gradient");
    assert_eq!(b_grad.shape().dims(), &[1, 4]);
    assert_eq!(b_grad.to_vec()?, vec![3.0, 3.0, 3.0, 3.0]);
    Ok(())
}

#[test]
fn f151_sub_broadcast_gradient_is_reduced_and_negated() -> Result<()> {
    let x = tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).requires_grad_(true);
    let bias = tensor(vec![1.0, 1.0, 1.0], vec![3]).requires_grad_(true);

    x.sub(&bias)?.sum()?.backward()?;

    let bias_grad = bias.grad().expect("bias gradient");
    assert_eq!(bias_grad.shape().dims(), &[3]);
    assert_eq!(bias_grad.to_vec()?, vec![-2.0, -2.0, -2.0]);
    Ok(())
}

#[test]
fn f151_mul_broadcast_gradient_matches_finite_differences() -> Result<()> {
    let x_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let scale_data = vec![0.5f32, -1.5, 2.0];
    let x = tensor(x_data.clone(), vec![2, 3]).requires_grad_(true);
    let scale = tensor(scale_data.clone(), vec![3]).requires_grad_(true);

    x.mul(&scale)?.sum()?.backward()?;

    let scale_grad = scale.grad().expect("scale gradient");
    assert_eq!(scale_grad.shape().dims(), &[3]);

    let numeric = numerical_gradient(&scale_data, |values| {
        x_data
            .iter()
            .enumerate()
            .map(|(index, x)| x * values[index % 3])
            .sum()
    });
    assert_close(&scale_grad.to_vec()?, &numeric, "broadcast mul d/drhs");
    Ok(())
}

// ---------------------------------------------------------------------------
// F150 / F274 - backward_with_grad must actually seed the backward pass
// ---------------------------------------------------------------------------

#[test]
fn f150_backward_with_grad_propagates_the_seed() -> Result<()> {
    let x = tensor(vec![1.0, 2.0, 3.0], vec![3]).requires_grad_(true);
    let y = x.mul(&x)?; // non-scalar output: backward() rejects it
    assert!(
        y.backward().is_err(),
        "backward() only accepts scalar outputs"
    );

    let seed = tensor(vec![1.0, 0.5, 2.0], vec![3]);
    y.backward_with_grad(Some(&seed))?;

    // d(x*x)/dx = 2x, weighted by the seed.
    let grad = x.grad().expect("gradient").to_vec()?;
    let expected = vec![2.0 * 1.0 * 1.0, 2.0 * 2.0 * 0.5, 2.0 * 3.0 * 2.0];
    assert_close(&grad, &expected, "vector-Jacobian product");
    Ok(())
}

#[test]
fn f274_backward_with_grad_validates_its_arguments() -> Result<()> {
    let x = tensor(vec![1.0, 2.0, 3.0], vec![3]).requires_grad_(true);
    let y = x.mul(&x)?;

    let wrong_shape = tensor(vec![1.0, 1.0], vec![2]);
    assert!(
        y.backward_with_grad(Some(&wrong_shape)).is_err(),
        "a seed with the wrong shape must be rejected"
    );
    assert!(
        y.backward_with_grad(None).is_err(),
        "a missing seed on a non-scalar output must be rejected"
    );
    assert!(
        x.grad().is_none(),
        "a rejected backward call must not write gradients"
    );

    let detached = tensor(vec![1.0], vec![1]);
    assert!(
        detached.backward_with_grad(None).is_err(),
        "a tensor that does not require grad must be rejected"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// F071 - backward must be a topological walk, not an exponential tree walk
// ---------------------------------------------------------------------------

#[test]
fn f071_diamond_graph_accumulates_each_path_exactly_once() -> Result<()> {
    let x = tensor(vec![3.0], vec![1]).requires_grad_(true);
    let left = x.mul(&tensor(vec![2.0], vec![1]))?;
    let right = x.mul(&tensor(vec![5.0], vec![1]))?;
    left.add(&right)?.sum()?.backward()?;

    // dL/dx = 2 + 5, contributed once per path.
    assert_eq!(x.grad().expect("gradient").to_vec()?, vec![7.0]);
    Ok(())
}

#[test]
fn f071_deep_residual_graph_backward_stays_linear() -> Result<()> {
    // `h = h + h` doubles the number of distinct paths to `x` at every step, so
    // a recursive walk without a visited set performs ~2^DEPTH sub-traversals
    // (measured: ~9 s at DEPTH 26 before the fix). The topological engine walks
    // DEPTH + 1 nodes and finishes in well under a millisecond.
    const DEPTH: u32 = 26;

    let x = tensor(vec![1.0], vec![1]).requires_grad_(true);
    let mut head = x.add(&tensor(vec![0.0], vec![1]))?;
    for _ in 1..DEPTH {
        head = head.add(&head)?;
    }

    let started = Instant::now();
    head.sum()?.backward()?;
    let elapsed = started.elapsed();

    let expected = 2f32.powi(DEPTH as i32 - 1);
    assert_eq!(x.grad().expect("gradient").to_vec()?, vec![expected]);
    assert!(
        elapsed.as_secs() < 5,
        "backward over a depth-{DEPTH} residual graph took {elapsed:?}; \
         the traversal is not linear in the graph size"
    );
    Ok(())
}

#[test]
fn f071_shared_weight_accumulates_from_every_use() -> Result<()> {
    // A weight used by two different consumers must receive the sum of both
    // contributions, exactly once each.
    let w = tensor(vec![2.0, 3.0], vec![2]).requires_grad_(true);
    let a = tensor(vec![1.0, 1.0], vec![2]).requires_grad_(true);

    let first = a.mul(&w)?;
    let second = a.add(&w)?;
    first.add(&second)?.sum()?.backward()?;

    // d/dw (a*w + a + w) = a + 1
    assert_eq!(w.grad().expect("gradient").to_vec()?, vec![2.0, 2.0]);
    // d/da (a*w + a + w) = w + 1
    assert_eq!(a.grad().expect("gradient").to_vec()?, vec![3.0, 4.0]);
    Ok(())
}

// ---------------------------------------------------------------------------
// F159 - flat accessors must honour strides and the storage offset
// ---------------------------------------------------------------------------

#[test]
fn f159_get_flat_reads_view_order_elements() -> Result<()> {
    let base = tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let transposed = base.transpose_view(0, 1)?;
    assert_eq!(transposed.shape().dims(), &[3, 2]);

    // View order is [1, 4, 2, 5, 3, 6]; the base buffer order is [1..6].
    let expected = transposed.to_vec()?;
    let by_flat: Vec<f32> = (0..transposed.numel())
        .map(|index| transposed.get_flat(index).expect("get_flat should succeed"))
        .collect();
    assert_eq!(by_flat, expected);
    assert!(transposed.get_flat(transposed.numel()).is_err());
    Ok(())
}

#[test]
fn f159_get_flat_honours_the_storage_offset() -> Result<()> {
    let base = tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let rows = base.slice_tensor(0, 1, 3)?;
    assert_eq!(rows.shape().dims(), &[2, 2]);
    assert_eq!(rows.get_flat(0)?, 3.0);
    assert_eq!(rows.get_slice(0, 4)?, vec![3.0, 4.0, 5.0, 6.0]);
    Ok(())
}

#[test]
fn f159_get_and_set_slice_round_trip_through_a_strided_view() -> Result<()> {
    let base = tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let transposed = base.transpose_view(0, 1)?;

    assert_eq!(transposed.get_slice(0, 4)?, vec![1.0, 4.0, 2.0, 5.0]);

    transposed.set_slice(2, &[20.0, 50.0])?;
    assert_eq!(
        transposed.get_slice(0, 6)?,
        vec![1.0, 4.0, 20.0, 50.0, 3.0, 6.0]
    );
    // The write landed in the shared base storage at the transposed positions.
    assert_eq!(base.to_vec()?, vec![1.0, 20.0, 3.0, 4.0, 50.0, 6.0]);

    assert!(transposed.set_slice(5, &[1.0, 2.0]).is_err());
    Ok(())
}

#[test]
fn f159_with_data_slice_rejects_non_contiguous_views() -> Result<()> {
    let base = tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    // A base tensor still gets zero-copy access to exactly `numel` elements.
    let length = base.with_data_slice(|data| Ok(data.len()))?;
    assert_eq!(length, 6);

    let transposed = base.transpose_view(0, 1)?;
    assert!(
        transposed.with_data_slice(|data| Ok(data.len())).is_err(),
        "handing a strided view the base buffer in base order is silently wrong"
    );
    assert!(transposed
        .with_data_slice_mut(|data| Ok(data.len()))
        .is_err());

    // The documented escape hatch keeps working.
    let contiguous = transposed.contiguous()?;
    let values = contiguous.with_data_slice(|data| Ok(data.to_vec()))?;
    assert_eq!(values, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    Ok(())
}

// ---------------------------------------------------------------------------
// F063 - matmul backward for 2-D operands stays correct end to end
// ---------------------------------------------------------------------------

#[test]
fn f063_matmul_2d_backward_matches_finite_differences() -> Result<()> {
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![0.5f32, -1.0, 2.0, 1.5, -0.5, 3.0];
    let a = tensor(a_data.clone(), vec![2, 3]).requires_grad_(true);
    let b = tensor(b_data.clone(), vec![3, 2]).requires_grad_(true);

    a.matmul(&b)?.sum()?.backward()?;

    let reference = |lhs: &[f32], rhs: &[f32]| -> f32 {
        let mut total = 0.0;
        for i in 0..2 {
            for j in 0..2 {
                for p in 0..3 {
                    total += lhs[i * 3 + p] * rhs[p * 2 + j];
                }
            }
        }
        total
    };
    let numeric_a = numerical_gradient(&a_data, |values| reference(values, &b_data));
    let numeric_b = numerical_gradient(&b_data, |values| reference(&a_data, values));

    assert_close(
        &a.grad().expect("lhs gradient").to_vec()?,
        &numeric_a,
        "matmul d/dlhs",
    );
    assert_close(
        &b.grad().expect("rhs gradient").to_vec()?,
        &numeric_b,
        "matmul d/drhs",
    );
    Ok(())
}
