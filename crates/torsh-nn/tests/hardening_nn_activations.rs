//! Wave-4 hardening for `torsh_nn::functional::activation`.
//!
//! Every function in that module was audited and classified before this file
//! was written. The measured pre-fix state (probe output reproduced verbatim in
//! the wave report) was:
//!
//! | function                | pre-fix state                                   |
//! |-------------------------|-------------------------------------------------|
//! | `relu`, `relu_inplace`  | records (via the now-recording `Tensor::maximum`)|
//! | `leaky_relu`            | records (via `maximum`/`minimum`)                |
//! | `softmax`, `log_softmax`| records (fixed in wave 3)                        |
//! | `dropout`               | records (fixed in wave 3)                        |
//! | `sigmoid`               | **detached** — `Tensor::from_data` leaf          |
//! | `tanh`                  | **detached** — `Tensor::from_data` leaf          |
//! | `gelu` (both variants)  | **detached** — `Tensor::from_data` leaf          |
//! | `elu`                   | **detached** — `gt`/`where_tensor` do not record |
//! | `selu`                  | **detached** — inherits `elu`                    |
//! | `swish`                 | **silently wrong** — 21.9% gradient error        |
//! | `mish`                  | **silently wrong** — 26.3%, sign-flipped at -1.3 |
//!
//! The tests below pin three things at once, because fixing only the first
//! would have been easy and useless:
//!
//! 1. **connectivity** — the result requires grad and `backward()` populates the
//!    input's gradient;
//! 2. **the gradient itself** — central finite differences at step 1e-2 with
//!    tolerance `2e-2 * max(|numeric|, 1)`, which is what catches `swish` and
//!    `mish` being attached but wrong;
//! 3. **the forward values** — frozen against the values the *pre-fix* kernels
//!    produced, so a delegation cannot silently change what the function
//!    computes. Where a replacement is bit-for-bit identical the pin says so.

use torsh_core::error::Result;
use torsh_nn::functional;
use torsh_nn::functional::activation::{gelu_with_approximation, relu_inplace, GeluApproximation};
use torsh_tensor::Tensor;

/// Serializes every test in this file against
/// `activations_do_not_record_under_no_grad`'s `with_grad_mode(false, ..)`
/// window. Grad mode is a process-global `AtomicBool`
/// (`torsh-core/src/grad_mode.rs`); `cargo test` runs a whole binary's tests in
/// one process across a thread pool, so a no_grad scope on one thread otherwise
/// suppresses recording for every other test's tensor ops. `cargo nextest`
/// hides this entirely (process per test), which is exactly how the same defect
/// reached wave 3's gatekeeper.
static GRAD_MODE_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Finite-difference step mandated for this campaign's gradient checks.
const FD_STEP: f32 = 1e-2;

/// Campaign tolerance: relative for large gradients, absolute for small ones.
fn fd_tolerance(numeric: f32) -> f32 {
    2e-2 * numeric.abs().max(1.0)
}

/// The probe batch every forward pin uses. Deliberately straddles zero, stays
/// away from `0.0` for the FD checks of the kinked activations, and includes a
/// point (`-1.3`) where `mish`'s pre-fix gradient had the wrong *sign*.
const PROBE: [f32; 10] = [-2.4, -1.3, -0.6, -0.05, 0.0, 0.05, 0.35, 1.1, 2.2, 3.7];

/// `PROBE` minus the exact zero, for gradient checks: `relu`, `leaky_relu`,
/// `elu` and `selu` are kinked at the origin, where a finite difference
/// straddles both branches and measures neither one-sided derivative.
const PROBE_SMOOTH: [f32; 9] = [-2.4, -1.3, -0.6, -0.05, 0.05, 0.35, 1.1, 2.2, 3.7];

fn probe_tensor(data: &[f32]) -> Tensor {
    Tensor::from_vec(data.to_vec(), &[data.len()]).expect("probe tensor")
}

/// Evaluate `f` and collapse the (element-wise) result to a scalar objective.
///
/// Summing is the right objective for an element-wise activation: it makes
/// `d(loss)/dx_i` exactly `f'(x_i)`, so the finite difference measures the
/// derivative of the function under test and nothing else.
fn scalar_value<F>(data: &[f32], f: &F) -> f32
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    f(&probe_tensor(data))
        .expect("finite-difference forward pass")
        .to_vec()
        .expect("finite-difference output")
        .iter()
        .sum()
}

/// Central finite differences of `f` with respect to every element of `data`.
fn numeric_gradient<F>(data: &[f32], f: F) -> Vec<f32>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    (0..data.len())
        .map(|i| {
            let mut plus = data.to_vec();
            let mut minus = data.to_vec();
            plus[i] += FD_STEP;
            minus[i] -= FD_STEP;
            (scalar_value(&plus, &f) - scalar_value(&minus, &f)) / (2.0 * FD_STEP)
        })
        .collect()
}

fn assert_matches_finite_differences(label: &str, analytic: &[f32], numeric: &[f32]) {
    assert_eq!(
        analytic.len(),
        numeric.len(),
        "{label}: gradient length {} does not match the {} inputs perturbed",
        analytic.len(),
        numeric.len()
    );
    for (i, (&got, &want)) in analytic.iter().zip(numeric.iter()).enumerate() {
        let tol = fd_tolerance(want);
        assert!(
            (got - want).abs() <= tol,
            "{label}: gradient[{i}] = {got}, finite differences gave {want} (tolerance {tol})\n\
             analytic = {analytic:?}\nnumeric  = {numeric:?}"
        );
    }
}

/// Run the full connectivity + finite-difference gradient check for `f` at
/// `data`, and return the analytic gradient so callers can pin extra facts.
fn gradcheck<F>(label: &str, data: &[f32], f: F) -> Vec<f32>
where
    F: Fn(&Tensor) -> Result<Tensor> + Copy,
{
    let input = probe_tensor(data).requires_grad_(true);
    let output = f(&input).expect("forward pass");
    assert!(
        output.requires_grad(),
        "{label}: output is detached from its input — every consumer silently \
         loses its gradient"
    );
    output
        .sum()
        .expect("sum reduction")
        .backward()
        .unwrap_or_else(|e| panic!("{label}: backward failed: {e:?}"));
    let analytic = input
        .grad()
        .unwrap_or_else(|| panic!("{label}: backward did not populate the input gradient"))
        .to_vec()
        .expect("gradient values");
    let numeric = numeric_gradient(data, f);
    assert_matches_finite_differences(label, &analytic, &numeric);
    analytic
}

fn assert_bit_identical(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{label}: forward[{i}] = {g} is not bit-identical to the pre-fix value {w}\n\
             got  = {got:?}\nwant = {want:?}"
        );
    }
}

fn assert_close(label: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "{label}: forward[{i}] = {g} drifted from the pre-fix value {w} by more than {tol}\n\
             got  = {got:?}\nwant = {want:?}"
        );
    }
}

fn forward(f: impl Fn(&Tensor) -> Result<Tensor>) -> Vec<f32> {
    f(&probe_tensor(&PROBE))
        .expect("forward pass")
        .to_vec()
        .expect("forward values")
}

// ---------------------------------------------------------------------------
// Pre-fix forward values, captured from the tree before this wave's edits.
// These are the contract: a delegation is only allowed if it reproduces them.
// ---------------------------------------------------------------------------

const GOLDEN_RELU: [f32; 10] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.35, 1.1, 2.2, 3.7];
const GOLDEN_LEAKY_RELU: [f32; 10] = [
    -0.024,
    -0.012999999,
    -0.006,
    -0.0005,
    0.0,
    0.05,
    0.35,
    1.1,
    2.2,
    3.7,
];
const GOLDEN_GELU_EXACT: [f32; 10] = [
    -0.01967404,
    -0.12584072,
    -0.16455182,
    -0.024003055,
    0.0,
    0.025996948,
    0.22289073,
    0.9507673,
    2.1694126,
    3.6996012,
];
const GOLDEN_GELU_TANH: [f32; 10] = [
    -0.019276857,
    -0.12607098,
    -0.1645848,
    -0.024003062,
    0.0,
    0.025996938,
    0.22288619,
    0.9505811,
    2.1696784,
    3.6997283,
];
const GOLDEN_SIGMOID: [f32; 10] = [
    0.08317269, 0.21416503, 0.35434368, 0.4875026, 0.5, 0.5124974, 0.5866176, 0.7502601,
    0.90024954, 0.975873,
];
const GOLDEN_TANH: [f32; 10] = [
    -0.9836749,
    -0.8617231,
    -0.5370496,
    -0.04995837,
    0.0,
    0.049958397,
    0.33637553,
    0.800499,
    0.9757431,
    0.9987782,
];
const GOLDEN_SWISH: [f32; 10] = [
    -0.19961445,
    -0.27841452,
    -0.21260622,
    -0.024375131,
    0.0,
    0.025624871,
    0.20531616,
    0.82528615,
    1.980549,
    3.6107302,
];
const GOLDEN_MISH: [f32; 10] = [
    -0.20788442,
    -0.3073824,
    -0.24693607,
    -0.029198283,
    0.0,
    0.03079771,
    0.24783836,
    0.9708416,
    2.1566505,
    3.695695,
];
const GOLDEN_ELU: [f32; 10] = [
    -0.9092821,
    -0.7274682,
    -0.4511884,
    -0.048770607,
    0.0,
    0.05,
    0.35,
    1.1,
    2.2,
    3.7,
];
const GOLDEN_SELU: [f32; 10] = [
    -1.5986083,
    -1.2789613,
    -0.793234,
    -0.08574357,
    0.0,
    0.052535053,
    0.36774534,
    1.1557711,
    2.3115423,
    3.8875937,
];

// ---------------------------------------------------------------------------
// 1. Gradient correctness — the headline defects
// ---------------------------------------------------------------------------

/// `swish(x) = x * sigmoid(x)`. Pre-fix the inner `sigmoid` was a detached
/// `from_data` leaf, so only the outer multiply recorded and backward returned
/// `grad * sigma(x)` instead of `sigma(x) + x * sigma'(x)` — attached, no error,
/// 21.9% wrong at `x = 1.1`.
#[test]
fn w4_swish_gradient_matches_finite_differences() {
    let _serial = GRAD_MODE_GUARD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    gradcheck("swish", &PROBE_SMOOTH, functional::swish);
}

/// `mish(x) = x * tanh(softplus(x))`. Pre-fix the `softplus` half recorded but
/// the `tanh` was a detached leaf, dropping the entire
/// `x * d/dx[tanh(softplus(x))]` term. At `x = -1.3` that flipped the sign of
/// the gradient (+0.213 analytic vs -0.024 measured).
#[test]
fn w4_mish_gradient_matches_finite_differences() {
    let _serial = GRAD_MODE_GUARD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    gradcheck("mish", &PROBE_SMOOTH, functional::mish);
}

/// `mish`'s pre-fix gradient at `x = -1.3` was positive where the true
/// derivative is negative. A sign check is coarser than the FD check above but
/// says out loud what the defect was.
#[test]
fn w4_mish_gradient_sign_is_correct_on_the_negative_branch() {
    let _serial = GRAD_MODE_GUARD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let analytic = gradcheck("mish@-1.3", &[-1.3], functional::mish);
    assert!(
        analytic[0] < 0.0,
        "mish'(-1.3) must be negative, got {}",
        analytic[0]
    );
}

#[test]
fn w4_sigmoid_gradient_matches_finite_differences() {
    let _serial = GRAD_MODE_GUARD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    gradcheck("sigmoid", &PROBE, functional::sigmoid);
}

#[test]
fn w4_tanh_gradient_matches_finite_differences() {
    let _serial = GRAD_MODE_GUARD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    gradcheck("tanh", &PROBE, functional::tanh);
}

#[test]
fn w4_gelu_exact_gradient_matches_finite_differences() {
    let _serial = GRAD_MODE_GUARD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    gradcheck("gelu(none)", &PROBE, functional::gelu);
}

#[test]
fn w4_gelu_tanh_gradient_matches_finite_differences() {
    let _serial = GRAD_MODE_GUARD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    gradcheck("gelu(tanh)", &PROBE, |x| {
        gelu_with_approximation(x, GeluApproximation::Tanh)
    });
}

#[test]
fn w4_relu_gradient_matches_finite_differences() {
    let _serial = GRAD_MODE_GUARD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    gradcheck("relu", &PROBE_SMOOTH, functional::relu);
}

#[test]
fn w4_leaky_relu_gradient_matches_finite_differences() {
    let _serial = GRAD_MODE_GUARD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for slope in [0.01f32, 0.2, 0.5] {
        let analytic = gradcheck(&format!("leaky_relu({slope})"), &PROBE_SMOOTH, |x| {
            functional::leaky_relu(x, slope)
        });
        // The negative branch's derivative *is* the slope, exactly.
        assert!(
            (analytic[0] - slope).abs() <= 1e-6,
            "leaky_relu({slope}): d/dx at -2.4 = {}, expected {slope}",
            analytic[0]
        );
    }
}

#[test]
fn w4_elu_gradient_matches_finite_differences() {
    let _serial = GRAD_MODE_GUARD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for alpha in [0.5f32, 1.0, 1.673_263_2] {
        gradcheck(&format!("elu({alpha})"), &PROBE_SMOOTH, |x| {
            functional::elu(x, alpha)
        });
    }
}

#[test]
fn w4_selu_gradient_matches_finite_differences() {
    let _serial = GRAD_MODE_GUARD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    gradcheck("selu", &PROBE_SMOOTH, functional::selu);
}

// ---------------------------------------------------------------------------
// 2. Connectivity through a non-leaf input
// ---------------------------------------------------------------------------

/// The gradient must reach a *producer* of the activation's input, not just a
/// leaf that happens to be the input itself.
///
/// This is the discriminating test for `gelu`'s exact (erf) branch, which has
/// no recording tensor primitive and therefore attaches its analytic derivative
/// through a `input - frozen_copy_of_input` residual. If the frozen copy were
/// built with `detach()` — which clones the `operation` field — the residual's
/// negative leg would flow back into the input's own subgraph and cancel the
/// positive one. A leaf-input gradcheck cannot see that; this one can.
#[test]
fn w4_activations_differentiate_through_a_non_leaf_input() {
    let _serial = GRAD_MODE_GUARD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let base = [-1.2f32, -0.3, 0.4, 1.7];
    let cases: [(&str, fn(&Tensor) -> Result<Tensor>); 8] = [
        ("relu", functional::relu),
        ("sigmoid", functional::sigmoid),
        ("tanh", functional::tanh),
        ("gelu(none)", functional::gelu),
        ("gelu(tanh)", |x| {
            gelu_with_approximation(x, GeluApproximation::Tanh)
        }),
        ("swish", functional::swish),
        ("mish", functional::mish),
        ("selu", functional::selu),
    ];

    for (label, f) in cases {
        // h = 2 * w, y = f(h): d(loss)/dw = 2 * f'(h).
        let w = probe_tensor(&base).requires_grad_(true);
        let hidden = w.mul_scalar(2.0).expect("hidden");
        let output = f(&hidden).expect("forward");
        assert!(
            output.requires_grad(),
            "{label}: output detached from a non-leaf input"
        );
        output
            .sum()
            .expect("sum")
            .backward()
            .unwrap_or_else(|e| panic!("{label}: backward through a non-leaf failed: {e:?}"));
        let analytic = w
            .grad()
            .unwrap_or_else(|| panic!("{label}: no gradient reached the producer"))
            .to_vec()
            .expect("gradient values");
        let numeric = numeric_gradient(&base, |x| {
            let hidden = x.mul_scalar(2.0)?;
            f(&hidden)
        });
        assert_matches_finite_differences(&format!("{label} through 2*w"), &analytic, &numeric);
        assert!(
            analytic.iter().any(|g| g.abs() > 1e-4),
            "{label}: gradient through a non-leaf collapsed to zero: {analytic:?}"
        );
    }
}

/// `relu` is what `compile_time.rs`'s MLP builder uses (`:80`, `:446`); a
/// detached `relu` made that whole production path untrainable.
#[test]
fn w4_relu_keeps_a_two_layer_chain_trainable() {
    let _serial = GRAD_MODE_GUARD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let base = [0.7f32, -0.4, 1.1, 0.25];
    let weights = probe_tensor(&[1.5, -2.0, 0.5, 3.0]);
    let w = probe_tensor(&base).requires_grad_(true);

    let chain = |x: &Tensor| -> Result<Tensor> {
        let scaled = x.mul_op(&weights)?;
        let activated = functional::relu(&scaled)?;
        functional::sigmoid(&activated)
    };

    let output = chain(&w).expect("chain forward");
    assert!(output.requires_grad(), "relu -> sigmoid chain is detached");
    output
        .sum()
        .expect("sum")
        .backward()
        .expect("chain backward");
    let analytic = w
        .grad()
        .expect("chain gradient")
        .to_vec()
        .expect("gradient values");
    let numeric = numeric_gradient(&base, chain);
    assert_matches_finite_differences("relu -> sigmoid chain", &analytic, &numeric);
}

// ---------------------------------------------------------------------------
// 3. requires_grad / no_grad contract
// ---------------------------------------------------------------------------

#[test]
fn w4_every_activation_propagates_requires_grad() {
    let _serial = GRAD_MODE_GUARD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let input = probe_tensor(&PROBE).requires_grad_(true);
    let cases: [(&str, fn(&Tensor) -> Result<Tensor>); 12] = [
        ("relu", functional::relu),
        ("leaky_relu", |x| functional::leaky_relu(x, 0.01)),
        ("sigmoid", functional::sigmoid),
        ("tanh", functional::tanh),
        ("gelu(none)", functional::gelu),
        ("gelu(tanh)", |x| {
            gelu_with_approximation(x, GeluApproximation::Tanh)
        }),
        ("swish", functional::swish),
        ("mish", functional::mish),
        ("elu", |x| functional::elu(x, 1.0)),
        ("selu", functional::selu),
        ("softmax", |x| functional::softmax(x, None)),
        ("log_softmax", |x| functional::log_softmax(x, None)),
    ];
    for (label, f) in cases {
        assert!(
            f(&input).expect("forward").requires_grad(),
            "{label}: must propagate requires_grad"
        );
    }

    // A non-requiring input must stay non-requiring.
    let plain = probe_tensor(&PROBE);
    for (label, f) in cases {
        assert!(
            !f(&plain).expect("forward").requires_grad(),
            "{label}: must not manufacture requires_grad out of a plain input"
        );
    }
}

/// In-place ReLU must keep the tensor on the graph, not silently replace it
/// with a detached leaf.
#[test]
fn w4_relu_inplace_records_and_matches_relu() {
    let _serial = GRAD_MODE_GUARD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let input = probe_tensor(&PROBE_SMOOTH).requires_grad_(true);
    let mut buffer = input.clone();
    relu_inplace(&mut buffer).expect("relu_inplace");
    assert!(
        buffer.requires_grad(),
        "relu_inplace must keep the tensor attached to the graph"
    );
    assert_bit_identical(
        "relu_inplace vs relu",
        &buffer.to_vec().expect("values"),
        &functional::relu(&input)
            .expect("relu")
            .to_vec()
            .expect("values"),
    );

    buffer
        .sum()
        .expect("sum")
        .backward()
        .expect("relu_inplace backward");
    let analytic = input
        .grad()
        .expect("relu_inplace gradient")
        .to_vec()
        .expect("values");
    let numeric = numeric_gradient(&PROBE_SMOOTH, |x| {
        let mut buffer = x.clone();
        relu_inplace(&mut buffer)?;
        Ok(buffer)
    });
    assert_matches_finite_differences("relu_inplace", &analytic, &numeric);
}

/// No activation may record inside a `no_grad` scope.
#[test]
fn w4_activations_do_not_record_under_no_grad() {
    let _serial = GRAD_MODE_GUARD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let input = probe_tensor(&PROBE).requires_grad_(true);
    torsh_core::with_grad_mode(false, || {
        let cases: [(&str, fn(&Tensor) -> Result<Tensor>); 10] = [
            ("relu", functional::relu),
            ("leaky_relu", |x| functional::leaky_relu(x, 0.01)),
            ("sigmoid", functional::sigmoid),
            ("tanh", functional::tanh),
            ("gelu(none)", functional::gelu),
            ("gelu(tanh)", |x| {
                gelu_with_approximation(x, GeluApproximation::Tanh)
            }),
            ("swish", functional::swish),
            ("mish", functional::mish),
            ("elu", |x| functional::elu(x, 1.0)),
            ("selu", functional::selu),
        ];
        for (label, f) in cases {
            let output = f(&input).expect("forward");
            assert!(
                !output.requires_grad(),
                "{label}: recorded a graph inside a no_grad scope"
            );
        }
    });

    // ... and recording resumes afterwards.
    assert!(
        functional::gelu(&input).expect("forward").requires_grad(),
        "gelu stopped recording after the no_grad scope closed"
    );
}

// ---------------------------------------------------------------------------
// 4. Forward-value regression pins
// ---------------------------------------------------------------------------

/// Delegations that are bit-for-bit identical to the kernels they replaced.
#[test]
fn w4_forward_values_are_bit_identical_to_the_pre_fix_kernels() {
    assert_bit_identical("relu", &forward(functional::relu), &GOLDEN_RELU);
    assert_bit_identical(
        "leaky_relu(0.01)",
        &forward(|x| functional::leaky_relu(x, 0.01)),
        &GOLDEN_LEAKY_RELU,
    );
    assert_bit_identical("gelu(none)", &forward(functional::gelu), &GOLDEN_GELU_EXACT);
    assert_bit_identical(
        "gelu(tanh)",
        &forward(|x| gelu_with_approximation(x, GeluApproximation::Tanh)),
        &GOLDEN_GELU_TANH,
    );
    assert_bit_identical(
        "elu(1.0)",
        &forward(|x| functional::elu(x, 1.0)),
        &GOLDEN_ELU,
    );
    assert_bit_identical("selu", &forward(functional::selu), &GOLDEN_SELU);
}

/// Turning gradient tracking on must not move a single forward bit.
///
/// This matters most for `gelu(none)`, whose grad-enabled path adds a
/// `(input - frozen) * derivative` residual to the forward values; the residual
/// is `+0.0` everywhere, and this is the assertion that says so. `elu`'s masked
/// composition and the delegated kernels are covered for the same reason.
#[test]
fn w4_enabling_grad_does_not_perturb_forward_values() {
    let _serial = GRAD_MODE_GUARD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let plain = probe_tensor(&PROBE);
    let tracked = probe_tensor(&PROBE).requires_grad_(true);
    let cases: [(&str, fn(&Tensor) -> Result<Tensor>); 10] = [
        ("relu", functional::relu),
        ("leaky_relu", |x| functional::leaky_relu(x, 0.01)),
        ("sigmoid", functional::sigmoid),
        ("tanh", functional::tanh),
        ("gelu(none)", functional::gelu),
        ("gelu(tanh)", |x| {
            gelu_with_approximation(x, GeluApproximation::Tanh)
        }),
        ("swish", functional::swish),
        ("mish", functional::mish),
        ("elu", |x| functional::elu(x, 1.0)),
        ("selu", functional::selu),
    ];
    for (label, f) in cases {
        assert_bit_identical(
            &format!("{label} (grad on vs off)"),
            &f(&tracked).expect("tracked forward").to_vec().expect("v"),
            &f(&plain).expect("plain forward").to_vec().expect("v"),
        );
    }
}

/// Delegations that move by at most a few ULP because the replacement evaluates
/// the same mathematical function with a different (better-conditioned)
/// expression. The measured worst-case drift over a 1631-point sweep of
/// `[-30, 30]` plus the denormal/overflow edges was:
/// sigmoid 1.2e-7, tanh 1.2e-7, swish 2.4e-7, mish 9.5e-7.
#[test]
fn w4_forward_values_match_the_pre_fix_kernels_to_within_ulp_drift() {
    assert_close(
        "sigmoid",
        &forward(functional::sigmoid),
        &GOLDEN_SIGMOID,
        2e-6,
    );
    assert_close("tanh", &forward(functional::tanh), &GOLDEN_TANH, 2e-6);
    assert_close("swish", &forward(functional::swish), &GOLDEN_SWISH, 2e-6);
    assert_close("mish", &forward(functional::mish), &GOLDEN_MISH, 2e-6);
}

/// `gelu()` must stay the *exact* erf formulation (PyTorch's
/// `approximate="none"`), not silently become the tanh approximation: the two
/// differ by up to 4e-4 on this batch, which is 3000x the ULP drift above.
#[test]
fn w4_gelu_default_is_the_exact_erf_formulation() {
    let exact = forward(functional::gelu);
    let approx = forward(|x| gelu_with_approximation(x, GeluApproximation::Tanh));
    let worst = exact
        .iter()
        .zip(approx.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst > 1e-4,
        "gelu(none) and gelu(tanh) agree to {worst} — the default variant has \
         been switched to the approximation"
    );
    assert_bit_identical("gelu(none)", &exact, &GOLDEN_GELU_EXACT);

    // Reference: 0.5 * x * (1 + erf(x / sqrt(2))) computed in f64.
    for (&x, &got) in PROBE.iter().zip(exact.iter()) {
        let want = 0.5 * f64::from(x) * (1.0 + erf_f64(f64::from(x) / std::f64::consts::SQRT_2));
        assert!(
            (f64::from(got) - want).abs() <= 1e-5,
            "gelu(none) at x={x}: {got} vs exact {want}"
        );
    }
}

/// Abramowitz & Stegun 7.1.26 in f64 — an independent erf reference, accurate
/// to ~1.5e-7, which is ample for the 1e-5 comparison above.
fn erf_f64(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    sign * y
}

// ---------------------------------------------------------------------------
// 5. Dispatch-band consistency and numerical edges
// ---------------------------------------------------------------------------

/// `Tensor`'s activations switch kernels at numel > 100 (parallel) and
/// numel > 1000 (SIMD). Delegating means `functional::*` inherits those bands,
/// so the same input must produce the same output on both sides of them.
#[test]
fn w4_forward_is_consistent_across_the_dispatch_bands() {
    let cases: [(&str, fn(&Tensor) -> Result<Tensor>); 6] = [
        ("relu", functional::relu),
        ("sigmoid", functional::sigmoid),
        ("tanh", functional::tanh),
        ("gelu(none)", functional::gelu),
        ("gelu(tanh)", |x| {
            gelu_with_approximation(x, GeluApproximation::Tanh)
        }),
        ("swish", functional::swish),
    ];

    for (label, f) in cases {
        let small = probe_tensor(&PROBE);
        let small_out = f(&small).expect("small forward").to_vec().expect("values");

        for width in [128usize, 2048] {
            let mut data = vec![0.0f32; width];
            for (slot, &x) in data.iter_mut().zip(PROBE.iter()) {
                *slot = x;
            }
            let big = Tensor::from_vec(data, &[width]).expect("wide tensor");
            let big_out = f(&big).expect("wide forward").to_vec().expect("values");
            assert_close(
                &format!("{label} @ numel={width}"),
                &big_out[..PROBE.len()],
                &small_out,
                1e-6,
            );
        }
    }
}

/// A non-contiguous input must produce exactly the values its contiguous copy
/// does.
///
/// This is the discriminating test for `gelu(none)`'s surrogate: it is the only
/// activation here that mixes a `to_vec()`-derived tensor (logical order,
/// contiguous) back into a tensor operation against the *original* handle
/// (`input.sub(&frozen)`). If the subtraction ignored strides, the residual
/// would not be `+0.0` and the forward values would move — a failure mode the
/// previous `to_vec()` -> compute -> `from_data` kernel could not have. The
/// other activations are covered here too, cheaply, since they inherit the
/// tensor-level kernels' stride handling.
#[test]
fn w4_forward_is_stride_agnostic() {
    let _serial = GRAD_MODE_GUARD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let flat: Vec<f32> = vec![
        -2.4, -1.3, -0.6, -0.05, 0.05, 0.35, 1.1, 2.2, 3.7, -0.9, 0.75, 1.6,
    ];
    let base = Tensor::from_vec(flat, &[3, 4])
        .expect("base tensor")
        .requires_grad_(true);
    let view = base.transpose(-1, -2).expect("transposed view");
    let materialised = Tensor::from_vec(view.to_vec().expect("view values"), view.shape().dims())
        .expect("contiguous copy");

    let cases: [(&str, fn(&Tensor) -> Result<Tensor>); 10] = [
        ("relu", functional::relu),
        ("leaky_relu", |x| functional::leaky_relu(x, 0.01)),
        ("sigmoid", functional::sigmoid),
        ("tanh", functional::tanh),
        ("gelu(none)", functional::gelu),
        ("gelu(tanh)", |x| {
            gelu_with_approximation(x, GeluApproximation::Tanh)
        }),
        ("swish", functional::swish),
        ("mish", functional::mish),
        ("elu", |x| functional::elu(x, 1.0)),
        ("selu", functional::selu),
    ];
    for (label, f) in cases {
        let strided = f(&view).expect("strided forward");
        assert_eq!(
            strided.shape().dims(),
            materialised.shape().dims(),
            "{label}: shape changed on a strided input"
        );
        assert_bit_identical(
            &format!("{label} on a transposed view"),
            &strided.to_vec().expect("strided values"),
            &f(&materialised)
                .expect("contiguous forward")
                .to_vec()
                .expect("contiguous values"),
        );
    }
}

/// `mish`'s pre-fix softplus was `ln(exp(x) + 1)`, which overflows to `+inf`
/// for `x > 88.7`. The forward survived (`tanh(inf) == 1`, so `mish(x) == x`)
/// but the *gradient* does not once `tanh` records: `d/du ln(u)` is
/// `1/inf == 0` and `d/dx exp(x)` is `inf`, and `0 * inf` is `NaN`. The
/// assertion at the end of this test measures that on the naive expression
/// directly, so the claim is not merely derived. The replacement evaluates
/// `softplus` as `max(x, 0) + ln(1 + exp(-|x|))`, which never overflows.
#[test]
fn w4_mish_gradient_is_finite_in_the_softplus_overflow_band() {
    let _serial = GRAD_MODE_GUARD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let data = [30.0f32, 88.0, 90.0, 120.0, -90.0];
    let input = probe_tensor(&data).requires_grad_(true);
    let output = functional::mish(&input).expect("mish forward");
    let values = output.to_vec().expect("forward values");
    for (&x, &y) in data.iter().zip(values.iter()) {
        assert!(y.is_finite(), "mish({x}) = {y} is not finite");
    }
    // For x well above the softplus knee mish(x) == x to f32 precision.
    assert!(
        (values[3] - 120.0).abs() <= 1e-3,
        "mish(120) = {} should be ~120",
        values[3]
    );

    output
        .sum()
        .expect("sum")
        .backward()
        .expect("mish backward");
    let analytic = input
        .grad()
        .expect("mish gradient")
        .to_vec()
        .expect("gradient values");
    for (&x, &g) in data.iter().zip(analytic.iter()) {
        assert!(
            g.is_finite(),
            "mish'({x}) = {g} is not finite — the softplus overflowed"
        );
    }
    // mish'(x) -> 1 as x -> +inf.
    assert!(
        (analytic[3] - 1.0).abs() <= 1e-3,
        "mish'(120) = {} should be ~1",
        analytic[3]
    );

    // And the counterfactual: the naive `ln(exp(x) + 1)` softplus, rebuilt here
    // out of the same recording primitives `mish` used to use, really does
    // produce a NaN gradient above the overflow knee. Without the stable
    // rewrite, delegating `tanh` would have swapped a silently-wrong gradient
    // for a NaN one.
    let naive_input = probe_tensor(&data).requires_grad_(true);
    let naive = naive_input
        .exp()
        .expect("exp")
        .add_scalar(1.0)
        .expect("+1")
        .ln()
        .expect("ln")
        .tanh()
        .expect("tanh");
    let naive_mish = naive_input.mul_op(&naive).expect("naive mish");
    naive_mish
        .sum()
        .expect("sum")
        .backward()
        .expect("naive backward");
    let naive_grad = naive_input
        .grad()
        .expect("naive gradient")
        .to_vec()
        .expect("gradient values");
    assert!(
        naive_grad[3].is_nan(),
        "the naive ln(exp(x)+1) softplus was expected to give a NaN gradient at \
         x = 120, got {} — if this is finite the stable rewrite's justification \
         no longer holds and this test should be revisited",
        naive_grad[3]
    );
}

/// `elu`'s replacement must not manufacture `NaN` where the old `where_tensor`
/// select produced a finite value: `alpha * (exp(x) - 1)` overflows for large
/// positive `x`, and the negative branch's value at `-inf` is `-alpha`.
#[test]
fn w4_elu_survives_the_exponential_overflow_and_infinite_edges() {
    let data = [100.0f32, 200.0, f32::INFINITY, f32::NEG_INFINITY];
    let values = functional::elu(&probe_tensor(&data), 1.0)
        .expect("elu forward")
        .to_vec()
        .expect("values");
    assert_eq!(values[0], 100.0, "elu(100) must pass through");
    assert_eq!(values[1], 200.0, "elu(200) must pass through");
    assert_eq!(values[2], f32::INFINITY, "elu(inf) must stay +inf");
    assert_eq!(values[3], -1.0, "elu(-inf) must saturate at -alpha");
}

/// Both `swish` and `mish` must stay finite where their pre-fix forwards were.
#[test]
fn w4_swish_gradient_is_finite_at_the_saturation_edges() {
    let _serial = GRAD_MODE_GUARD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let data = [-100.0f32, -40.0, 40.0, 100.0];
    let input = probe_tensor(&data).requires_grad_(true);
    let output = functional::swish(&input).expect("swish forward");
    for (&x, &y) in data.iter().zip(output.to_vec().expect("values").iter()) {
        assert!(y.is_finite(), "swish({x}) = {y} is not finite");
    }
    output
        .sum()
        .expect("sum")
        .backward()
        .expect("swish backward");
    let analytic = input
        .grad()
        .expect("swish gradient")
        .to_vec()
        .expect("gradient values");
    for (&x, &g) in data.iter().zip(analytic.iter()) {
        assert!(g.is_finite(), "swish'({x}) = {g} is not finite");
    }
    assert!(
        (analytic[3] - 1.0).abs() <= 1e-3,
        "swish'(100) = {} should be ~1",
        analytic[3]
    );
}
