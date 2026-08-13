//! Wave-4 hardening: four element-wise primitives that silently DETACHED the
//! autograd graph (`abs`, `maximum`, `minimum`, `clamp`) plus the GELU forward
//! that was DISCONTINUOUS across its own SIMD dispatch threshold.
//!
//! Each of these is a leaf of the nn-side reconnection work:
//!
//! * `abs`     — `nn::functional::l1_loss` is `(input - target).abs()`.
//! * `maximum` — `nn::functional::relu` is `input.maximum(&zeros)`.
//! * `minimum` — the mirror rule, used by clamping/loss code.
//! * `clamp`   — `binary_cross_entropy`'s probability clamp.
//! * `gelu`    — the f32 SIMD kernel computed a *different function* than the
//!   scalar/parallel paths, so the forward jumped at `numel == 1000` and the
//!   recorded backward differentiated neither one consistently.

use torsh_core::device::DeviceType;
use torsh_tensor::Tensor;

/// Serializes every test in this file against the `with_grad_mode(false, ..)`
/// windows opened by the `*_do_not_record_under_no_grad` tests below.
///
/// Grad mode is a process-global `AtomicBool` (`torsh-core/src/grad_mode.rs`)
/// and plain `cargo test` runs a whole binary's tests in ONE process across a
/// thread pool (unlike `cargo nextest`'s process-per-test), so a no_grad scope
/// entered on one thread transiently suppresses recording for every other
/// test's tensor ops. Confining the manipulation to the no_grad tests is NOT
/// sufficient on its own — every test that records or asserts recording has to
/// hold this lock for its full body. Mirrors the identical guards in
/// `hardening_autograd_unary.rs` and `hardening_autograd_complete.rs`.
static GRAD_MODE_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn t32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
    Tensor::from_data(data, shape, DeviceType::Cpu).expect("f32 tensor creation should succeed")
}

/// Central-difference step mandated by the campaign's gradcheck protocol.
const FD_STEP: f32 = 1e-2;

/// Campaign tolerance: `2e-2 * max(|numeric|, 1)`.
fn fd_tolerance(numeric: f32) -> f32 {
    2e-2 * numeric.abs().max(1.0)
}

/// Central finite-difference gradient of a scalar loss w.r.t. every element of
/// `base` (which carries `shape`).
fn fd_grad<F>(base: &[f32], shape: &[usize], loss: F) -> Vec<f32>
where
    F: Fn(&Tensor<f32>) -> f32,
{
    let mut g = vec![0.0_f32; base.len()];
    for (i, slot) in g.iter_mut().enumerate() {
        let mut plus = base.to_vec();
        let mut minus = base.to_vec();
        plus[i] += FD_STEP;
        minus[i] -= FD_STEP;
        let lp = loss(&t32(plus, shape.to_vec()));
        let lm = loss(&t32(minus, shape.to_vec()));
        *slot = (lp - lm) / (2.0 * FD_STEP);
    }
    g
}

/// Per-element central difference that reads back element `i` of the *forward*
/// tensor instead of a scalar reduction.
///
/// Summing 2048 values of order 1 in f32 and then differencing two such sums
/// leaves ~6e-3 of cancellation noise on a ~2e-2 signal, which would swamp the
/// very discrepancy these tests exist to detect. Reading `out[i]` directly has
/// no summation at all, so the estimate is limited only by the f32 resolution
/// of `gelu` itself.
fn fd_elementwise<F>(base: &[f32], shape: &[usize], indices: &[usize], forward: F) -> Vec<f32>
where
    F: Fn(&Tensor<f32>) -> Vec<f32>,
{
    indices
        .iter()
        .map(|&i| {
            let mut plus = base.to_vec();
            let mut minus = base.to_vec();
            plus[i] += FD_STEP;
            minus[i] -= FD_STEP;
            let fp = forward(&t32(plus, shape.to_vec()));
            let fm = forward(&t32(minus, shape.to_vec()));
            (fp[i] - fm[i]) / (2.0 * FD_STEP)
        })
        .collect()
}

fn assert_fd_close(analytic: &[f32], numeric: &[f32], ctx: &str) {
    assert_eq!(analytic.len(), numeric.len(), "{ctx}: length mismatch");
    for (i, (a, n)) in analytic.iter().zip(numeric.iter()).enumerate() {
        let tol = fd_tolerance(*n);
        assert!(
            (a - n).abs() <= tol,
            "{ctx}: index {i}: analytic {a} vs numeric {n} (tol {tol})"
        );
    }
}

fn assert_exact(actual: &[f32], want: &[f32], tol: f32, ctx: &str) {
    assert_eq!(actual.len(), want.len(), "{ctx}: length mismatch");
    for (i, (a, w)) in actual.iter().zip(want.iter()).enumerate() {
        assert!(
            (a - w).abs() <= tol,
            "{ctx}: index {i}: {a} vs expected {w} (tol {tol})"
        );
    }
}

/// The exact-tanh GELU the crate documents:
/// `0.5*x*(1 + tanh(k*(x + c*x^3)))`, `k = sqrt(2/pi)`, `c = 0.044715`.
fn gelu_reference(x: f32) -> f32 {
    let k = (2.0f32 / std::f32::consts::PI).sqrt();
    let c = 0.044_715f32;
    0.5 * x * (1.0 + (k * (x + c * x * x * x)).tanh())
}

/// Analytic derivative of [`gelu_reference`].
fn gelu_reference_grad(x: f32) -> f32 {
    let k = (2.0f32 / std::f32::consts::PI).sqrt();
    let c = 0.044_715f32;
    let t = (k * (x + c * x * x * x)).tanh();
    0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * k * (1.0 + 3.0 * c * x * x)
}

/// A spread of inputs that covers the saturating tails, the kink region and
/// the linear region — including `x = -2.40`, the worst measured disagreement
/// between the SIMD kernel and the exact-tanh formula (515% relative).
fn gelu_probe_values() -> Vec<f32> {
    vec![
        -3.0, -2.4, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.4, 3.0,
    ]
}

// ---------------------------------------------------------------------------
// (a) GELU is one function on every dispatch path.
// ---------------------------------------------------------------------------

/// The forward must not jump when the tensor crosses the `numel > 1000` f32
/// SIMD dispatch threshold. `n = 999` takes the parallel exact-tanh path and
/// `n = 2048` took scirs2's clamped-Pade SIMD kernel; the same input value at
/// the same position must produce the same output on both.
#[test]
fn gelu_forward_is_continuous_across_the_dispatch_threshold() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    let probes = gelu_probe_values();
    let small_n = 999usize;
    let big_n = 2048usize;

    let small_v: Vec<f32> = (0..small_n).map(|i| probes[i % probes.len()]).collect();
    // The first `small_n` entries of the big tensor are bit-identical to the
    // whole small tensor; the tail only exists to push `numel` over 1000.
    let big_v: Vec<f32> = (0..big_n)
        .map(|i| probes[i % probes.len()])
        .collect::<Vec<_>>();

    let small_out = t32(small_v.clone(), vec![small_n])
        .gelu()
        .expect("gelu (parallel path)")
        .to_vec()
        .expect("small gelu data");
    let big_out = t32(big_v.clone(), vec![big_n])
        .gelu()
        .expect("gelu (simd-sized path)")
        .to_vec()
        .expect("big gelu data");

    for i in 0..small_n {
        assert!(
            (small_out[i] - big_out[i]).abs() <= 1e-6,
            "gelu forward is discontinuous across the dispatch threshold at \
             x={}: n=999 gives {} but n=2048 gives {} (delta {})",
            small_v[i],
            small_out[i],
            big_out[i],
            (small_out[i] - big_out[i]).abs()
        );
    }
}

/// Every dispatch band — scalar (`numel <= 100`), parallel (`101..=1000`) and
/// the former SIMD band (`numel > 1000`) — must compute the exact-tanh closed
/// form the recorded derivative differentiates.
#[test]
fn gelu_forward_matches_the_exact_tanh_closed_form_at_every_size() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    let probes = gelu_probe_values();
    for &n in &[8usize, 256usize, 2048usize] {
        let xv: Vec<f32> = (0..n).map(|i| probes[i % probes.len()]).collect();
        let out = t32(xv.clone(), vec![n])
            .gelu()
            .unwrap_or_else(|e| panic!("gelu at n={n}: {e:?}"))
            .to_vec()
            .expect("gelu data");
        for (i, &x) in xv.iter().enumerate() {
            let want = gelu_reference(x);
            assert!(
                (out[i] - want).abs() <= 1e-6,
                "gelu forward at n={n}, x={x}: {} vs exact-tanh {want} (delta {})",
                out[i],
                (out[i] - want).abs()
            );
        }
    }
}

/// Finite-difference gradcheck at `n = 2048`, previously impossible: the SIMD
/// forward was a different function from the recorded backward, so FD compared
/// `d/dx(Pade GELU)` against `d/dx(tanh GELU)`.
#[test]
fn gelu_backward_matches_finite_difference_above_the_simd_threshold() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    let probes = gelu_probe_values();
    let n = 2048usize;
    let xv: Vec<f32> = (0..n).map(|i| probes[i % probes.len()]).collect();

    let x = t32(xv.clone(), vec![n]).requires_grad_(true);
    x.gelu()
        .expect("gelu")
        .sum()
        .expect("sum")
        .backward()
        .expect("gelu backward must be recorded at n=2048");
    let analytic_all = x
        .grad()
        .expect("gelu grad missing at n=2048")
        .to_vec()
        .expect("grad data");

    // GELU is C-infinity — it has no kink anywhere — so no FD step can
    // straddle a discontinuity and the usual "keep 2*step away from the
    // boundary" rule has nothing to apply to here. One index per distinct
    // probe value is enough coverage, and it keeps the cost at
    // 2 * 13 full-tensor forwards rather than 2 * 2048.
    let indices: Vec<usize> = (0..probes.len()).collect();
    let numeric = fd_elementwise(&xv, &[n], &indices, |t| {
        t.gelu().expect("gelu forward").to_vec().expect("gelu data")
    });
    let analytic: Vec<f32> = indices.iter().map(|&i| analytic_all[i]).collect();
    assert_fd_close(&analytic, &numeric, "gelu grad vs FD at n=2048");
}

/// The recorded rule is the analytic derivative of the exact-tanh closed form
/// on every size band, not only the small ones.
#[test]
fn gelu_backward_matches_closed_form_at_every_size() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    let probes = gelu_probe_values();
    for &n in &[8usize, 256usize, 2048usize] {
        let xv: Vec<f32> = (0..n).map(|i| probes[i % probes.len()]).collect();
        let x = t32(xv.clone(), vec![n]).requires_grad_(true);
        x.gelu()
            .expect("gelu")
            .sum()
            .expect("sum")
            .backward()
            .unwrap_or_else(|e| panic!("gelu backward at n={n}: {e:?}"));
        let g = x
            .grad()
            .unwrap_or_else(|| panic!("gelu grad missing at n={n}"))
            .to_vec()
            .expect("grad data");
        let want: Vec<f32> = xv.iter().map(|&v| gelu_reference_grad(v)).collect();
        assert_exact(&g, &want, 1e-5, &format!("gelu closed-form grad (n={n})"));
    }
}

// ---------------------------------------------------------------------------
// (b) abs records `sign(x)`.
// ---------------------------------------------------------------------------

/// `abs` must record; the derivative is `sign(x)`. Every sample sits at least
/// `2 * FD_STEP` away from the kink at 0, so central differences are valid.
#[test]
fn abs_backward_matches_finite_difference() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    let xv = vec![-2.5f32, -1.25, -0.3, 0.3, 1.25, 2.5];
    let x = t32(xv.clone(), vec![6]).requires_grad_(true);
    x.abs()
        .expect("abs")
        .sum()
        .expect("sum")
        .backward()
        .expect("abs backward must be recorded");
    let analytic = x
        .grad()
        .expect("abs grad missing — Tensor::abs still detaches")
        .to_vec()
        .expect("grad data");
    let numeric = fd_grad(&xv, &[6], |t| {
        t.abs()
            .expect("abs forward")
            .sum()
            .expect("sum")
            .item()
            .expect("item")
    });
    assert_fd_close(&analytic, &numeric, "abs grad vs FD");
}

/// PyTorch's convention: `d|x|/dx == 0` exactly at `x == 0`. Analytic, not FD —
/// FD across the kink would return 0 for the wrong reason.
#[test]
fn abs_gradient_at_zero_is_zero() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    let x = t32(vec![-1.5f32, 0.0, 2.0], vec![3]).requires_grad_(true);
    x.abs()
        .expect("abs")
        .sum()
        .expect("sum")
        .backward()
        .expect("abs backward must be recorded");
    let g = x
        .grad()
        .expect("abs grad missing at the kink test")
        .to_vec()
        .expect("grad data");
    assert_exact(&g, &[-1.0, 0.0, 1.0], 0.0, "abs sign() gradient");
}

/// The `l1_loss` shape: `(input - target).abs().mean()`. This is the chain
/// `nn::functional::l1_loss` walks, and it was severed at `abs`.
#[test]
fn abs_backward_flows_through_an_l1_style_loss() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    let iv = vec![1.5f32, -0.75, 2.25, -1.0];
    let tv = vec![0.5f32, 0.75, 3.25, 1.0];
    let input = t32(iv.clone(), vec![4]).requires_grad_(true);
    let target = t32(tv.clone(), vec![4]);
    input
        .sub(&target)
        .expect("sub")
        .abs()
        .expect("abs")
        .mean(None, false)
        .expect("mean")
        .backward()
        .expect("l1-style backward must reach the input");
    let analytic = input
        .grad()
        .expect("l1-style loss grad missing")
        .to_vec()
        .expect("grad data");
    let numeric = fd_grad(&iv, &[4], |t| {
        t.sub(&t32(tv.clone(), vec![4]))
            .expect("sub")
            .abs()
            .expect("abs")
            .mean(None, false)
            .expect("mean")
            .item()
            .expect("item")
    });
    assert_fd_close(&analytic, &numeric, "l1-style loss grad vs FD");
}

// ---------------------------------------------------------------------------
// (c) maximum / minimum route the gradient to the winning operand.
// ---------------------------------------------------------------------------

/// No ties: every element's gradient goes wholly to whichever operand won.
#[test]
fn maximum_routes_grad_to_the_winner() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    let a = t32(vec![3.0f32, -1.0, 0.5, -4.0], vec![4]).requires_grad_(true);
    let b = t32(vec![1.0f32, 2.0, -0.5, -2.0], vec![4]).requires_grad_(true);
    a.maximum(&b)
        .expect("maximum")
        .sum()
        .expect("sum")
        .backward()
        .expect("maximum backward must be recorded");
    let ga = a
        .grad()
        .expect("maximum lhs grad missing — Tensor::maximum still detaches")
        .to_vec()
        .expect("grad data");
    let gb = b
        .grad()
        .expect("maximum rhs grad missing")
        .to_vec()
        .expect("grad data");
    assert_exact(&ga, &[1.0, 0.0, 1.0, 0.0], 0.0, "maximum lhs grad");
    assert_exact(&gb, &[0.0, 1.0, 0.0, 1.0], 0.0, "maximum rhs grad");
}

/// Mirror rule for `minimum`.
#[test]
fn minimum_routes_grad_to_the_winner() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    let a = t32(vec![3.0f32, -1.0, 0.5, -4.0], vec![4]).requires_grad_(true);
    let b = t32(vec![1.0f32, 2.0, -0.5, -2.0], vec![4]).requires_grad_(true);
    a.minimum(&b)
        .expect("minimum")
        .sum()
        .expect("sum")
        .backward()
        .expect("minimum backward must be recorded");
    let ga = a
        .grad()
        .expect("minimum lhs grad missing — Tensor::minimum still detaches")
        .to_vec()
        .expect("grad data");
    let gb = b
        .grad()
        .expect("minimum rhs grad missing")
        .to_vec()
        .expect("grad data");
    assert_exact(&ga, &[0.0, 1.0, 0.0, 1.0], 0.0, "minimum lhs grad");
    assert_exact(&gb, &[1.0, 0.0, 1.0, 0.0], 0.0, "minimum rhs grad");
}

/// PyTorch splits the gradient evenly on an exact tie. Analytic by
/// construction — FD cannot see a measure-zero tie.
#[test]
fn maximum_and_minimum_split_grad_evenly_on_exact_ties() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    for is_max in [true, false] {
        let a = t32(vec![2.0f32, 2.0, -1.0], vec![3]).requires_grad_(true);
        let b = t32(vec![2.0f32, 5.0, -1.0], vec![3]).requires_grad_(true);
        let out = if is_max {
            a.maximum(&b).expect("maximum")
        } else {
            a.minimum(&b).expect("minimum")
        };
        out.sum()
            .expect("sum")
            .backward()
            .expect("tie backward must be recorded");
        let ga = a.grad().expect("tie lhs grad").to_vec().expect("grad data");
        let gb = b.grad().expect("tie rhs grad").to_vec().expect("grad data");
        // Index 0 and 2 are exact ties (0.5 each); index 1 has a strict winner.
        let (want_a, want_b) = if is_max {
            ([0.5, 0.0, 0.5], [0.5, 1.0, 0.5])
        } else {
            ([0.5, 1.0, 0.5], [0.5, 0.0, 0.5])
        };
        assert_exact(&ga, &want_a, 0.0, "tie lhs grad");
        assert_exact(&gb, &want_b, 0.0, "tie rhs grad");
    }
}

/// FD gradcheck with every pair separated by more than `2 * FD_STEP`, so no
/// perturbation can flip a winner mid-difference.
#[test]
fn maximum_backward_matches_finite_difference() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    let av = vec![3.0f32, -1.0, 0.5, -4.0, 2.0, 0.1];
    let bv = vec![1.0f32, 2.0, -0.5, -2.0, -3.0, -0.1];
    let a = t32(av.clone(), vec![6]).requires_grad_(true);
    let b = t32(bv.clone(), vec![6]);
    a.maximum(&b)
        .expect("maximum")
        .sum()
        .expect("sum")
        .backward()
        .expect("maximum backward must be recorded");
    let analytic = a
        .grad()
        .expect("maximum grad missing")
        .to_vec()
        .expect("grad data");
    let numeric = fd_grad(&av, &[6], |t| {
        t.maximum(&t32(bv.clone(), vec![6]))
            .expect("maximum forward")
            .sum()
            .expect("sum")
            .item()
            .expect("item")
    });
    assert_fd_close(&analytic, &numeric, "maximum grad vs FD");
}

/// The forward broadcasts (`elementwise_operation` falls through to
/// `broadcast_binary_op`), so each operand's gradient must be folded back to
/// its OWN shape: `[2, 3]` against `[3]`.
#[test]
fn maximum_broadcast_folds_grad_back_to_each_operand_shape() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    let a = t32(vec![3.0f32, -1.0, 0.5, -4.0, 2.0, 0.25], vec![2, 3]).requires_grad_(true);
    let b = t32(vec![1.0f32, 2.0, -0.5], vec![3]).requires_grad_(true);
    let out = a.maximum(&b).expect("broadcast maximum");
    assert_eq!(
        out.shape().dims(),
        &[2, 3],
        "broadcast maximum output shape"
    );
    out.sum()
        .expect("sum")
        .backward()
        .expect("broadcast maximum backward must be recorded");
    let ga = a
        .grad()
        .expect("broadcast maximum lhs grad missing")
        .to_vec()
        .expect("grad data");
    let gb = b
        .grad()
        .expect("broadcast maximum rhs grad missing")
        .to_vec()
        .expect("grad data");
    // a = [[3, -1, 0.5], [-4, 2, 0.25]] against broadcast b = [[1, 2, -0.5],
    //                                                          [1, 2, -0.5]]
    // winners row 0: a, b, a
    // winners row 1: b, TIE (2 == 2, so 0.5 each), a
    assert_exact(&ga, &[1.0, 0.0, 1.0, 0.0, 0.5, 1.0], 0.0, "broadcast lhs");
    assert_eq!(
        b.grad().expect("rhs grad").shape().dims(),
        &[3],
        "rhs grad must be folded back to [3]"
    );
    // b's gradient is the column sum of [[0, 1, 0], [1, 0.5, 0]] — the fold
    // that `reduce_grad_to_shape` performs, and the tie's 0.5 has to survive it.
    assert_exact(&gb, &[1.0, 1.5, 0.0], 0.0, "broadcast rhs");
}

/// `nn::functional::relu` is literally `input.maximum(&zeros)`. This is the
/// exact chain that was severed.
#[test]
fn relu_via_maximum_with_zeros_is_differentiable() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    let xv = vec![-2.0f32, -0.5, 0.5, 2.0];
    let x = t32(xv.clone(), vec![4]).requires_grad_(true);
    let zeros = t32(vec![0.0f32; 4], vec![4]);
    x.maximum(&zeros)
        .expect("relu via maximum")
        .sum()
        .expect("sum")
        .backward()
        .expect("relu-via-maximum backward must be recorded");
    let g = x
        .grad()
        .expect("relu-via-maximum grad missing")
        .to_vec()
        .expect("grad data");
    assert_exact(&g, &[0.0, 0.0, 1.0, 1.0], 0.0, "relu via maximum grad");
}

// ---------------------------------------------------------------------------
// (d) clamp passes the gradient only where the input was not clamped.
// ---------------------------------------------------------------------------

/// PyTorch's `clamp_backward` is `grad * (x >= min) * (x <= max)` — inclusive
/// at both bounds. The two boundary samples (`-1.0` and `1.0`) therefore keep
/// their gradient; the two clamped samples lose it.
#[test]
fn clamp_backward_is_inclusive_at_both_bounds() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    let x = t32(vec![-3.0f32, -1.0, 0.0, 1.0, 3.0], vec![5]).requires_grad_(true);
    x.clamp(-1.0, 1.0)
        .expect("clamp")
        .sum()
        .expect("sum")
        .backward()
        .expect("clamp backward must be recorded");
    let g = x
        .grad()
        .expect("clamp grad missing — Tensor::clamp still detaches")
        .to_vec()
        .expect("grad data");
    assert_exact(
        &g,
        &[0.0, 1.0, 1.0, 1.0, 0.0],
        0.0,
        "clamp inclusive-bound gradient",
    );
}

/// FD gradcheck with every sample at least `2 * FD_STEP` away from either
/// bound, so no perturbation crosses the kink.
#[test]
fn clamp_backward_matches_finite_difference() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    let xv = vec![-2.5f32, -1.5, -0.5, 0.25, 0.9, 1.5, 2.5];
    let x = t32(xv.clone(), vec![7]).requires_grad_(true);
    x.clamp(-1.0, 1.0)
        .expect("clamp")
        .sum()
        .expect("sum")
        .backward()
        .expect("clamp backward must be recorded");
    let analytic = x
        .grad()
        .expect("clamp grad missing")
        .to_vec()
        .expect("grad data");
    let numeric = fd_grad(&xv, &[7], |t| {
        t.clamp(-1.0, 1.0)
            .expect("clamp forward")
            .sum()
            .expect("sum")
            .item()
            .expect("item")
    });
    assert_fd_close(&analytic, &numeric, "clamp grad vs FD");
}

/// `binary_cross_entropy` clamps its probabilities before the log; the chain
/// through the clamp has to survive.
#[test]
fn clamp_backward_flows_through_a_bce_style_chain() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    let xv = vec![0.2f32, 0.45, 0.7, 0.95];
    let x = t32(xv.clone(), vec![4]).requires_grad_(true);
    x.clamp(1e-3, 1.0 - 1e-3)
        .expect("clamp")
        .ln()
        .expect("ln")
        .mean(None, false)
        .expect("mean")
        .backward()
        .expect("bce-style backward must reach the input");
    let analytic = x
        .grad()
        .expect("bce-style clamp grad missing")
        .to_vec()
        .expect("grad data");
    let numeric = fd_grad(&xv, &[4], |t| {
        t.clamp(1e-3, 1.0 - 1e-3)
            .expect("clamp")
            .ln()
            .expect("ln")
            .mean(None, false)
            .expect("mean")
            .item()
            .expect("item")
    });
    assert_fd_close(&analytic, &numeric, "bce-style clamp grad vs FD");
}

/// `Operation::ClampBounds` stores `Option` bounds so a one-sided clamp is
/// representable. `clamp_min` / `clamp_max` are what make the `None` arms
/// reachable — without them half the backward rule would be untestable.
#[test]
fn one_sided_clamps_pass_grad_on_the_unbounded_side() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    let xv = vec![-3.0f32, -1.0, 0.0, 1.0, 3.0];

    let lower = t32(xv.clone(), vec![5]).requires_grad_(true);
    let lower_out = lower.clamp_min(-1.0).expect("clamp_min");
    assert_exact(
        &lower_out.to_vec().expect("clamp_min data"),
        &[-1.0, -1.0, 0.0, 1.0, 3.0],
        0.0,
        "clamp_min forward",
    );
    lower_out
        .sum()
        .expect("sum")
        .backward()
        .expect("clamp_min backward must be recorded");
    // Unbounded above: everything at or above -1.0 keeps its gradient.
    assert_exact(
        &lower
            .grad()
            .expect("clamp_min grad")
            .to_vec()
            .expect("grad data"),
        &[0.0, 1.0, 1.0, 1.0, 1.0],
        0.0,
        "clamp_min gradient",
    );

    let upper = t32(xv.clone(), vec![5]).requires_grad_(true);
    let upper_out = upper.clamp_max(1.0).expect("clamp_max");
    assert_exact(
        &upper_out.to_vec().expect("clamp_max data"),
        &[-3.0, -1.0, 0.0, 1.0, 1.0],
        0.0,
        "clamp_max forward",
    );
    upper_out
        .sum()
        .expect("sum")
        .backward()
        .expect("clamp_max backward must be recorded");
    // Unbounded below: everything at or below 1.0 keeps its gradient.
    assert_exact(
        &upper
            .grad()
            .expect("clamp_max grad")
            .to_vec()
            .expect("grad data"),
        &[1.0, 1.0, 1.0, 1.0, 0.0],
        0.0,
        "clamp_max gradient",
    );
}

/// A two-sided `clamp` must stay bit-identical to the composition of the two
/// one-sided ones on ordinary inputs — the refactor into `clamp_bounds` may not
/// have changed the forward.
#[test]
fn clamp_forward_is_unchanged_by_the_optional_bounds_refactor() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    let xv: Vec<f32> = (0..64).map(|i| (i as f32) * 0.125 - 4.0).collect();
    let x = t32(xv.clone(), vec![64]);
    let both = x.clamp(-1.5, 2.25).expect("clamp").to_vec().expect("data");
    let staged = x
        .clamp_min(-1.5)
        .expect("clamp_min")
        .clamp_max(2.25)
        .expect("clamp_max")
        .to_vec()
        .expect("data");
    let want: Vec<f32> = xv.iter().map(|&v| v.clamp(-1.5, 2.25)).collect();
    assert_exact(&both, &want, 0.0, "two-sided clamp forward");
    assert_exact(&staged, &want, 0.0, "staged one-sided clamp forward");
}

// ---------------------------------------------------------------------------
// no_grad guards: none of the four primitives may record inside `no_grad`.
// ---------------------------------------------------------------------------

/// `with_grad_mode(false, ..)` must suppress recording for all four new rules.
/// Holds [`GRAD_MODE_GUARD`] like every other test in this file — the flag is
/// process-global.
#[test]
fn new_primitives_do_not_record_under_no_grad() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    let x = t32(vec![-1.5f32, 0.5, 2.0], vec![3]).requires_grad_(true);
    let y = t32(vec![0.5f32, -0.5, 3.0], vec![3]).requires_grad_(true);
    torsh_core::grad_mode::with_grad_mode(false, || {
        assert!(
            !x.abs().expect("abs under no_grad").requires_grad(),
            "abs must not record under no_grad"
        );
        assert!(
            !x.maximum(&y)
                .expect("maximum under no_grad")
                .requires_grad(),
            "maximum must not record under no_grad"
        );
        assert!(
            !x.minimum(&y)
                .expect("minimum under no_grad")
                .requires_grad(),
            "minimum must not record under no_grad"
        );
        assert!(
            !x.clamp(-1.0, 1.0)
                .expect("clamp under no_grad")
                .requires_grad(),
            "clamp must not record under no_grad"
        );
        assert!(
            !x.clamp_min(-1.0)
                .expect("clamp_min under no_grad")
                .requires_grad(),
            "clamp_min must not record under no_grad"
        );
        assert!(
            !x.clamp_max(1.0)
                .expect("clamp_max under no_grad")
                .requires_grad(),
            "clamp_max must not record under no_grad"
        );
        assert!(
            !x.gelu().expect("gelu under no_grad").requires_grad(),
            "gelu must not record under no_grad"
        );
    });
    // Outside the no_grad scope recording resumes.
    assert!(
        x.abs().expect("abs outside no_grad").requires_grad(),
        "abs must record again outside no_grad"
    );
}
