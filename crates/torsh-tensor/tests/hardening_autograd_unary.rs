//! Task U hardening: record autograd for the remaining map()-sink unary ops.
//!
//! impl_c3.md's "REMAINING map()-SINK OPS" follow-up flagged `square`,
//! `rsqrt`, `reciprocal`, `log10`, `log2`, `tan`, `asin`, `acos`, `atan`,
//! `sinh`, `cosh` (math_ops_trig.rs) and `relu_scirs2`/`sigmoid_scirs2`/
//! `tanh_scirs2` (advanced_ops.rs): after `Tensor::map` became forward-only,
//! every one of these ops silently returned an honestly-detached leaf
//! instead of recording a real backward rule. Each test here fails on the
//! pre-Task-U tree with "Called backward on tensor that doesn't require
//! grad", and pins the real derivative with a finite-difference check (or,
//! at a domain edge where FD is meaningless, an analytic inf/NaN check).

use torsh_core::device::DeviceType;
use torsh_tensor::Tensor;

/// Serializes every test in this file against
/// `unary_ops_do_not_record_under_no_grad`'s `with_grad_mode(false, ..)`
/// window. Grad mode is a process-global `AtomicBool`
/// (`torsh-core/src/grad_mode.rs`), so under `cargo test` -- which runs a
/// whole binary's tests in one process across a thread pool, unlike
/// `cargo nextest`'s process-per-test -- a no_grad scope on one thread
/// transiently suppresses recording for every other test's tensor ops too.
/// Measured: with no serialization, `tanh_scirs2_backward_matches_tanh`
/// (whose two independent `tanh_scirs2()` calls straddle the race window)
/// failed 2 of 3 default-parallel `cargo test` runs. Every test that records
/// or checks recording takes this lock for its full body, which costs
/// nothing measurable (every op here runs on an 8-element tensor).
static GRAD_MODE_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn t32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
    Tensor::from_data(data, shape, DeviceType::Cpu).expect("f32 tensor creation should succeed")
}

/// Central finite-difference gradient of a scalar loss w.r.t. every element of
/// `base` (which carries `shape`). Mirrors hardening_autograd_complete.rs.
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

/// Forward+backward `op` on `xv`, check `requires_grad` propagation, and pin
/// the recorded gradient against central finite differences.
fn check_unary_fd<F>(name: &str, xv: &[f32], tol: f32, op: F)
where
    F: Fn(&Tensor<f32>) -> Tensor<f32>,
{
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    let n = xv.len();
    let x = t32(xv.to_vec(), vec![n]).requires_grad_(true);
    let y = op(&x);
    assert!(
        y.requires_grad(),
        "{name} output must propagate requires_grad from its input"
    );
    y.sum()
        .expect("sum")
        .backward()
        .unwrap_or_else(|e| panic!("{name} backward must be recorded: {e}"));
    let analytic = x
        .grad()
        .unwrap_or_else(|| panic!("{name} grad missing"))
        .to_vec()
        .unwrap();
    let numeric = fd_grad(xv, &[n], |t| op(t).sum().unwrap().item().unwrap());
    assert_close(&analytic, &numeric, tol, &format!("{name} grad vs FD"));
}

// ---------------------------------------------------------------------------
// FD gradchecks, one per op, at points chosen to sit safely inside the
// domain: close enough to representative values, far enough from a domain
// edge or asymptote that central-difference truncation error (~h^2/6 *
// f'''(x), h = 1e-3) cannot approach the comparison tolerance.
// ---------------------------------------------------------------------------

#[test]
fn square_backward_matches_finite_difference() {
    // The discriminating case for the `Operation::Power{exponent: 2.0}`
    // route: negative x. d/dx x^2 = 2x is exact for a negative base with an
    // integer-valued exponent (`powf` on a negative base only returns NaN
    // for a *non*-integer exponent) -- an all-positive ramp would pass even
    // if that were subtly wrong. Central differences of a quadratic have no
    // *truncation* error, but computing `(x+h)^2 - (x-h)^2` in f32 still
    // cancels most of the leading digits (e.g. 8.994001 - 9.006001 at
    // x=-3), so this still needs the suite's normal FD tolerance, not a
    // tighter one; |x| <= 2.0 keeps that cancellation error (which grows
    // with x^2) comfortably under the tolerance.
    let xv = vec![-2.0f32, -1.5, -0.25, 0.0, 0.25, 1.5, 2.0, -1.0];
    check_unary_fd("square", &xv, 1e-3, |t| t.square().unwrap());
}

#[test]
fn rsqrt_backward_matches_finite_difference() {
    let xv = vec![0.5f32, 1.0, 2.0, 4.0, 0.75, 3.0, 1.25, 2.5];
    check_unary_fd("rsqrt", &xv, 1e-3, |t| t.rsqrt().unwrap());
}

#[test]
fn reciprocal_backward_matches_finite_difference() {
    let xv = vec![-4.0f32, -2.0, -0.5, 0.5, 2.0, 4.0, -1.0, 1.0];
    check_unary_fd("reciprocal", &xv, 1e-3, |t| t.reciprocal().unwrap());
}

#[test]
fn log10_backward_matches_finite_difference() {
    let xv = vec![0.5f32, 1.0, 2.0, 4.0, 0.75, 3.0, 1.25, 2.5];
    check_unary_fd("log10", &xv, 1e-3, |t| t.log10().unwrap());
}

#[test]
fn log2_backward_matches_finite_difference() {
    let xv = vec![0.5f32, 1.0, 2.0, 4.0, 0.75, 3.0, 1.25, 2.5];
    check_unary_fd("log2", &xv, 1e-3, |t| t.log2().unwrap());
}

#[test]
fn tan_backward_matches_finite_difference() {
    // |x| <= 1.0: clear of the +-pi/2 (~1.5708) asymptote where sec^2(x)
    // diverges. At |x| == 1.5 the FD truncation error alone is ~4e-2 --
    // comfortably over any tolerance this suite uses -- so this range is a
    // correctness requirement, not just a style choice.
    let xv = vec![-1.0f32, -0.5, -0.1, 0.1, 0.5, 1.0, -0.75, 0.75];
    check_unary_fd("tan", &xv, 1e-3, |t| t.tan().unwrap());
}

#[test]
fn asin_backward_matches_finite_difference() {
    // |x| <= 0.7: clear of the domain edge +-1, where 1/sqrt(1-x^2) diverges.
    let xv = vec![-0.7f32, -0.4, -0.1, 0.1, 0.4, 0.7, -0.55, 0.55];
    check_unary_fd("asin", &xv, 1e-3, |t| t.asin().unwrap());
}

#[test]
fn acos_backward_matches_finite_difference() {
    let xv = vec![-0.7f32, -0.4, -0.1, 0.1, 0.4, 0.7, -0.55, 0.55];
    check_unary_fd("acos", &xv, 1e-3, |t| t.acos().unwrap());
}

#[test]
fn atan_backward_matches_finite_difference() {
    // atan's derivative 1/(1+x^2) has no singularity anywhere, so this needs
    // no domain restriction.
    let xv = vec![-5.0f32, -2.0, -0.5, 0.5, 2.0, 5.0, -1.0, 1.0];
    check_unary_fd("atan", &xv, 1e-3, |t| t.atan().unwrap());
}

#[test]
fn sinh_backward_matches_finite_difference() {
    // sinh/cosh grow exponentially, so (unlike atan) an unbounded range would
    // eventually blow the FD truncation error through the tolerance; |x| <=
    // 2 keeps comfortable headroom (analytic estimate: error ~6e-7).
    let xv = vec![-2.0f32, -1.0, -0.25, 0.25, 1.0, 2.0, -1.5, 1.5];
    check_unary_fd("sinh", &xv, 1e-3, |t| t.sinh().unwrap());
}

#[test]
fn cosh_backward_matches_finite_difference() {
    let xv = vec![-2.0f32, -1.0, -0.25, 0.25, 1.0, 2.0, -1.5, 1.5];
    check_unary_fd("cosh", &xv, 1e-3, |t| t.cosh().unwrap());
}

// ---------------------------------------------------------------------------
// Domain-edge behaviour: "produce inf/NaN exactly as the MATH dictates -- do
// NOT clamp." Finite differences are meaningless at a singularity, so these
// are analytic-only checks on the recorded gradient.
// ---------------------------------------------------------------------------

#[test]
fn domain_edge_gradients_are_honest_not_clamped() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());

    // rsqrt at x == 0: d/dx x^(-1/2) = -0.5 / (x * sqrt(x)) = -0.5 / +0.0 = -inf.
    let x = t32(vec![0.0f32], vec![1]).requires_grad_(true);
    x.rsqrt().unwrap().sum().unwrap().backward().unwrap();
    let g = x.grad().expect("rsqrt grad at 0").to_vec().unwrap();
    assert!(
        g[0].is_infinite() && g[0].is_sign_negative(),
        "rsqrt'(0) should be -inf, got {}",
        g[0]
    );

    // rsqrt outside its domain (negative x): 1/sqrt(negative) is NaN in the
    // reals, and NaN must propagate honestly, never silently become 0.
    let x = t32(vec![-1.0f32], vec![1]).requires_grad_(true);
    x.rsqrt().unwrap().sum().unwrap().backward().unwrap();
    let g = x.grad().expect("rsqrt grad at -1").to_vec().unwrap();
    assert!(g[0].is_nan(), "rsqrt'(-1) should be NaN, got {}", g[0]);

    // reciprocal at x == 0: d/dx 1/x = -1/x^2 = -1/0 = -inf.
    let x = t32(vec![0.0f32], vec![1]).requires_grad_(true);
    x.reciprocal().unwrap().sum().unwrap().backward().unwrap();
    let g = x.grad().expect("reciprocal grad at 0").to_vec().unwrap();
    assert!(
        g[0].is_infinite() && g[0].is_sign_negative(),
        "reciprocal'(0) should be -inf, got {}",
        g[0]
    );

    // log10 at x == 0: d/dx log10(x) = 1/(x*ln10) = 1/+0.0 = +inf.
    let x = t32(vec![0.0f32], vec![1]).requires_grad_(true);
    x.log10().unwrap().sum().unwrap().backward().unwrap();
    let g = x.grad().expect("log10 grad at 0").to_vec().unwrap();
    assert!(
        g[0].is_infinite() && g[0].is_sign_positive(),
        "log10'(0) should be +inf, got {}",
        g[0]
    );

    // log2 at x == 0: same shape as log10.
    let x = t32(vec![0.0f32], vec![1]).requires_grad_(true);
    x.log2().unwrap().sum().unwrap().backward().unwrap();
    let g = x.grad().expect("log2 grad at 0").to_vec().unwrap();
    assert!(
        g[0].is_infinite() && g[0].is_sign_positive(),
        "log2'(0) should be +inf, got {}",
        g[0]
    );

    // asin at the domain edges x == 1 and x == -1: 1/sqrt(1-x^2) = 1/0 =
    // +inf at BOTH edges (the formula is symmetric and never negative).
    for &edge in &[1.0f32, -1.0] {
        let x = t32(vec![edge], vec![1]).requires_grad_(true);
        x.asin().unwrap().sum().unwrap().backward().unwrap();
        let g = x.grad().expect("asin grad at edge").to_vec().unwrap();
        assert!(
            g[0].is_infinite() && g[0].is_sign_positive(),
            "asin'({edge}) should be +inf, got {}",
            g[0]
        );
    }

    // asin strictly outside its domain: no real derivative exists, so NaN is
    // the honest answer, not a value to clamp away.
    let x = t32(vec![1.5f32], vec![1]).requires_grad_(true);
    x.asin().unwrap().sum().unwrap().backward().unwrap();
    let g = x
        .grad()
        .expect("asin grad outside domain")
        .to_vec()
        .unwrap();
    assert!(
        g[0].is_nan(),
        "asin'(1.5) should be NaN (outside [-1,1]), got {}",
        g[0]
    );

    // acos at the domain edges: -1/sqrt(1-x^2) = -inf at BOTH edges (the
    // formula is symmetric and never positive).
    for &edge in &[1.0f32, -1.0] {
        let x = t32(vec![edge], vec![1]).requires_grad_(true);
        x.acos().unwrap().sum().unwrap().backward().unwrap();
        let g = x.grad().expect("acos grad at edge").to_vec().unwrap();
        assert!(
            g[0].is_infinite() && g[0].is_sign_negative(),
            "acos'({edge}) should be -inf, got {}",
            g[0]
        );
    }
}

// ---------------------------------------------------------------------------
// relu_scirs2 / sigmoid_scirs2 / tanh_scirs2: same UnaryKind as their
// established siblings (relu / sigmoid / tanh), since their forward math is
// the exact same closed form (not an approximation like the GELU SIMD
// path), so their recorded gradient must match their sibling's bit-for-bit
// (well within float tolerance) on the same input.
// ---------------------------------------------------------------------------

#[test]
fn relu_scirs2_backward_matches_relu() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    let xv = vec![-2.0f32, -0.5, 0.5, 2.0, -1.5, 1.5, -0.25, 0.25];
    let x1 = t32(xv.clone(), vec![8]).requires_grad_(true);
    assert!(x1.relu_scirs2().unwrap().requires_grad());
    x1.relu_scirs2()
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .expect("relu_scirs2 backward must be recorded");
    let g1 = x1
        .grad()
        .expect("relu_scirs2 grad missing")
        .to_vec()
        .unwrap();

    let x2 = t32(xv, vec![8]).requires_grad_(true);
    x2.relu().unwrap().sum().unwrap().backward().unwrap();
    let g2 = x2.grad().expect("relu grad missing").to_vec().unwrap();

    assert_close(&g1, &g2, 1e-6, "relu_scirs2 vs relu grad");
}

#[test]
fn sigmoid_scirs2_backward_matches_sigmoid() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    let xv = vec![-2.0f32, -0.5, 0.0, 0.5, 2.0, -1.5, 1.5, 0.25];
    let x1 = t32(xv.clone(), vec![8]).requires_grad_(true);
    assert!(x1.sigmoid_scirs2().unwrap().requires_grad());
    x1.sigmoid_scirs2()
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .expect("sigmoid_scirs2 backward must be recorded");
    let g1 = x1
        .grad()
        .expect("sigmoid_scirs2 grad missing")
        .to_vec()
        .unwrap();

    let x2 = t32(xv, vec![8]).requires_grad_(true);
    x2.sigmoid().unwrap().sum().unwrap().backward().unwrap();
    let g2 = x2.grad().expect("sigmoid grad missing").to_vec().unwrap();

    assert_close(&g1, &g2, 1e-6, "sigmoid_scirs2 vs sigmoid grad");
}

#[test]
fn tanh_scirs2_backward_matches_tanh() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    let xv = vec![-2.0f32, -0.5, 0.0, 0.5, 2.0, -1.5, 1.5, 0.25];
    let x1 = t32(xv.clone(), vec![8]).requires_grad_(true);
    assert!(x1.tanh_scirs2().unwrap().requires_grad());
    x1.tanh_scirs2()
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .expect("tanh_scirs2 backward must be recorded");
    let g1 = x1
        .grad()
        .expect("tanh_scirs2 grad missing")
        .to_vec()
        .unwrap();

    let x2 = t32(xv, vec![8]).requires_grad_(true);
    x2.tanh().unwrap().sum().unwrap().backward().unwrap();
    let g2 = x2.grad().expect("tanh grad missing").to_vec().unwrap();

    assert_close(&g1, &g2, 1e-6, "tanh_scirs2 vs tanh grad");
}

// ---------------------------------------------------------------------------
// Regression guard for the should_record_grad gating, matching the
// established pattern in hardening_autograd_complete.rs
// (gelu_leaky_relu_do_not_record_under_no_grad). Calls the tensor methods
// directly rather than through `async_ops.rs`'s scheduled `*_scirs2`
// wrappers, which run on a worker thread and would race this scope's
// grad-mode flag regardless of the lock below (the wrapper's own thread is
// not a participant in `GRAD_MODE_GUARD`).
// ---------------------------------------------------------------------------

#[test]
fn unary_ops_do_not_record_under_no_grad() {
    use torsh_core::grad_mode::with_grad_mode;

    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());

    // In-domain for every op tested here, including asin/acos (|x| <= 1)
    // and rsqrt/reciprocal/log10/log2 (x > 0).
    let xv = vec![0.3f32, 0.4, 0.5, 0.6, 0.7, 0.8, 0.35, 0.55];
    let x = t32(xv, vec![8]).requires_grad_(true);

    let flags = with_grad_mode(false, || {
        vec![
            ("square", x.square().unwrap().requires_grad()),
            ("rsqrt", x.rsqrt().unwrap().requires_grad()),
            ("reciprocal", x.reciprocal().unwrap().requires_grad()),
            ("log10", x.log10().unwrap().requires_grad()),
            ("log2", x.log2().unwrap().requires_grad()),
            ("tan", x.tan().unwrap().requires_grad()),
            ("asin", x.asin().unwrap().requires_grad()),
            ("acos", x.acos().unwrap().requires_grad()),
            ("atan", x.atan().unwrap().requires_grad()),
            ("sinh", x.sinh().unwrap().requires_grad()),
            ("cosh", x.cosh().unwrap().requires_grad()),
            ("relu_scirs2", x.relu_scirs2().unwrap().requires_grad()),
            (
                "sigmoid_scirs2",
                x.sigmoid_scirs2().unwrap().requires_grad(),
            ),
            ("tanh_scirs2", x.tanh_scirs2().unwrap().requires_grad()),
        ]
    });
    for (name, requires_grad) in flags {
        assert!(!requires_grad, "{name} must not record under no_grad");
    }
}
