//! Production-hardening regression tests for the **Module-based** normalization
//! layers of `torsh-nn` (`layers::normalization::**`).
//!
//! # The finding
//!
//! Every one of these layers used to compute its mean/variance with a
//! hand-rolled `to_vec()` loop and hand the numbers back to
//! `Tensor::from_data`, which yields a *detached leaf*. The statistics were
//! therefore constants as far as `backward()` was concerned: the graph only ever
//! saw `(x - c1) / c2`, so `d loss / d input` came out as the gradient of a plain
//! affine rescaling instead of the real normalization Jacobian. PyTorch
//! back-propagates through the batch statistics; the functional API in
//! `functional/norm.rs` already did; only the Modules did not.
//!
//! The error is not small. For `LayerNorm` the true Jacobian subtracts the
//! per-sample mean *and* the projection onto the normalized direction; the
//! statistics-as-constant gradient keeps neither, so with a non-uniform loss
//! seed the two disagree in both magnitude and sign.
//!
//! # What is pinned here
//!
//! * **Forward values** — literal arrays captured from the *pre-rewrite* tree, so
//!   the rewrite cannot change what the layers compute (tolerance `1e-6`; the
//!   only intended numerical change is `E[x²] − E[x]²` → `E[(x − µ)²]`, which is
//!   the same quantity computed stably).
//! * **Input gradients** — central finite differences, step `1e-2`, tolerance
//!   `2e-2 · max(|numeric|, 1)`.
//! * **Affine (gamma/beta) gradients** — likewise; these flowed through the outer
//!   `mul`/`add` even before the rewrite, so they act as a control.
//! * **Running statistics** — value pins plus the invariant that the buffers are
//!   *never* graph nodes, and that eval mode consumes them as constants.
//!
//! # Harness rules (inherited from `hardening_nn_recurrent.rs`)
//!
//! * every analytic gradient is read *before* the first finite-difference
//!   forward, because an FD forward rewrites parameter tensors and resets the
//!   `grad` slots;
//! * every parameter write-back re-applies `requires_grad_(true)`;
//! * the loss seed is non-uniform (`sum(output * pattern)`) — a uniform seed
//!   cancels exactly the mean-subtraction term that the bug dropped;
//! * every finite-difference probe rebuilds the layer from scratch, so
//!   statefulness (`BatchNorm`'s running statistics, `BatchRenorm`'s step
//!   counter) cannot leak between probes;
//! * every test shape has `W != C`, because `utils::apply_normalization` falls
//!   back to a *trailing-axis* broadcast and would silently align the channel
//!   vector with the last axis whenever the two extents coincide.

use std::collections::HashMap;

use torsh_nn::layers::normalization::{
    BatchNorm1d, BatchNorm2d, BatchNorm3d, BatchRenorm2d, BatchRenormSchedule, GroupNorm,
    InstanceNorm1d, InstanceNorm2d, InstanceNorm3d, LayerNorm, NormalizationConfig, RMSNorm,
    SwitchableNorm2d, SyncBatchNorm2d, VirtualBatchNorm2d,
};
use torsh_nn::{Module, Parameter};
use torsh_tensor::Tensor;

/// Serializes every test in this file.
///
/// `torsh_core`'s grad mode is a process-global `AtomicBool`, and plain
/// `cargo test` runs a whole binary's tests in one process across a thread pool.
/// Tests here assert *recording* behaviour (`requires_grad` on the running-stat
/// buffers), which another thread's no-grad window would silently invalidate.
static GRAD_MODE_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serialize() -> std::sync::MutexGuard<'static, ()> {
    GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner())
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

/// Central finite-difference step.
const FD_STEP: f32 = 1e-2;

/// Tolerance for the forward-value regression pins.
const PIN_TOL: f32 = 1e-6;

/// Tolerance for a single finite-difference comparison: relative for large
/// gradients, absolute for small ones.
fn fd_tolerance(numeric: f32) -> f32 {
    2e-2 * numeric.abs().max(1.0)
}

/// Deterministic input values, centered on zero with an O(1) spread.
fn seeded(len: usize, salt: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let k = (((i * 37 + salt * 17 + 11) % 29) as f32) / 29.0;
            (k - 0.5) * 2.0
        })
        .collect()
}

/// Deterministic gamma values, spread around 1.
fn seeded_weight(len: usize, salt: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let k = (((i * 23 + salt * 13 + 5) % 19) as f32) / 19.0;
            0.7 + 0.6 * k
        })
        .collect()
}

/// Deterministic beta values, spread around 0.
fn seeded_bias(len: usize, salt: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let k = (((i * 31 + salt * 7 + 3) % 17) as f32) / 17.0;
            (k - 0.5) * 0.8
        })
        .collect()
}

/// Non-uniform loss seed. A uniform seed annihilates the mean-subtraction term
/// of the true normalization Jacobian, which is exactly the term the bug
/// dropped — it would make the broken gradient look correct.
fn pattern(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let k = (((i * 41 + 7) % 23) as f32) / 23.0;
            0.3 + 1.4 * k
        })
        .collect()
}

fn plain_input(dims: &[usize], salt: usize) -> Tensor {
    let len: usize = dims.iter().product();
    Tensor::from_vec(seeded(len, salt), dims).expect("input tensor")
}

/// `sum(output * pattern)` as a differentiable tensor.
fn loss_tensor(output: &Tensor) -> Tensor {
    let dims = output.shape().dims().to_vec();
    let len: usize = dims.iter().product();
    let seed = Tensor::from_vec(pattern(len), &dims).expect("loss seed");
    output
        .mul_op(&seed)
        .expect("loss product")
        .sum()
        .expect("loss sum")
}

/// The same loss evaluated numerically.
fn loss_value(output: &Tensor) -> f32 {
    let len: usize = output.shape().dims().iter().product();
    output
        .to_vec()
        .expect("output values")
        .iter()
        .zip(pattern(len).iter())
        .map(|(value, weight)| value * weight)
        .sum()
}

/// Overwrite a module parameter in place, re-applying `requires_grad`.
fn set_param(module: &dyn Module, name: &str, values: &[f32], shape: &[usize]) {
    let params = module.named_parameters();
    let param = params
        .get(name)
        .unwrap_or_else(|| panic!("parameter {name} must exist"));
    let tensor = Tensor::from_vec(values.to_vec(), shape).expect("parameter tensor");
    *param.tensor().write() = tensor.requires_grad_(true);
}

/// Install deterministic gamma/beta on whichever of them the layer registered.
fn set_affine(module: &dyn Module, shape: &[usize], salt: usize) {
    let params: HashMap<String, Parameter> = module.named_parameters();
    let len: usize = shape.iter().product();
    if params.contains_key("weight") {
        set_param(module, "weight", &seeded_weight(len, salt), shape);
    }
    if params.contains_key("bias") {
        set_param(module, "bias", &seeded_bias(len, salt), shape);
    }
}

/// Read a parameter's analytic gradient.
fn param_grad(module: &dyn Module, name: &str) -> Vec<f32> {
    let params = module.named_parameters();
    let param = params
        .get(name)
        .unwrap_or_else(|| panic!("parameter {name} must exist"));
    let handle = param.tensor();
    let guard = handle.read();
    guard
        .grad()
        .unwrap_or_else(|| {
            panic!("{name} has no analytic gradient — the affine path is off the graph")
        })
        .to_vec()
        .expect("gradient values")
}

fn assert_matches_fd(label: &str, analytic: &[f32], numeric: &[f32]) {
    assert_eq!(
        analytic.len(),
        numeric.len(),
        "{label}: analytic gradient has {} entries, {} were perturbed",
        analytic.len(),
        numeric.len()
    );
    for (i, (&got, &want)) in analytic.iter().zip(numeric.iter()).enumerate() {
        let tol = fd_tolerance(want);
        assert!(
            (got - want).abs() <= tol,
            "{label}: gradient[{i}] = {got}, central finite differences gave {want} \
             (tolerance {tol})\nanalytic = {analytic:?}\nnumeric  = {numeric:?}"
        );
    }
}

fn assert_pinned(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: produced {} values, the pre-rewrite tree produced {}",
        got.len(),
        want.len()
    );
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= PIN_TOL,
            "{label}: value[{i}] = {g}, the pre-rewrite implementation produced {w} \
             (tolerance {PIN_TOL})"
        );
    }
}

/// Finite-difference check of `d loss / d input`.
///
/// `make` must return a *freshly built* layer on every call: `BatchNorm` folds
/// each forward into its running statistics and `BatchRenorm` advances a step
/// counter, so a shared instance would make the two FD legs incomparable.
fn check_input_gradient<F>(label: &str, make: F, dims: &[usize], salt: usize)
where
    F: Fn() -> Box<dyn Module>,
{
    let data = seeded(dims.iter().product(), salt);
    let input = Tensor::from_vec(data.clone(), dims)
        .expect("input tensor")
        .requires_grad_(true);

    let module = make();
    let output = module
        .forward(&input)
        .unwrap_or_else(|e| panic!("{label}: forward failed: {e:?}"));
    loss_tensor(&output)
        .backward()
        .unwrap_or_else(|e| panic!("{label}: backward failed: {e:?}"));
    let analytic = input
        .grad()
        .unwrap_or_else(|| panic!("{label}: the input never reached the autograd graph"))
        .to_vec()
        .expect("input gradient values");

    let numeric: Vec<f32> = (0..data.len())
        .map(|i| {
            let mut plus = data.clone();
            let mut minus = data.clone();
            plus[i] += FD_STEP;
            minus[i] -= FD_STEP;
            let tp = Tensor::from_vec(plus, dims).expect("fd input");
            let tm = Tensor::from_vec(minus, dims).expect("fd input");
            let lp = loss_value(&make().forward(&tp).expect("fd forward"));
            let lm = loss_value(&make().forward(&tm).expect("fd forward"));
            (lp - lm) / (2.0 * FD_STEP)
        })
        .collect();

    assert_matches_fd(label, &analytic, &numeric);
}

/// Finite-difference check of `d loss / d <name>` for an affine parameter.
fn check_param_gradient<F>(
    label: &str,
    make: F,
    input: &Tensor,
    name: &str,
    base: &[f32],
    shape: &[usize],
) where
    F: Fn() -> Box<dyn Module>,
{
    let module = make();
    let output = module
        .forward(input)
        .unwrap_or_else(|e| panic!("{label}: forward failed: {e:?}"));
    loss_tensor(&output)
        .backward()
        .unwrap_or_else(|e| panic!("{label}: backward failed: {e:?}"));
    // Snapshot before any finite-difference forward: those rebuild the layer and
    // would drop this gradient slot.
    let analytic = param_grad(&*module, name);

    let numeric: Vec<f32> = (0..base.len())
        .map(|i| {
            let mut plus = base.to_vec();
            let mut minus = base.to_vec();
            plus[i] += FD_STEP;
            minus[i] -= FD_STEP;
            let mp = make();
            set_param(&*mp, name, &plus, shape);
            let mm = make();
            set_param(&*mm, name, &minus, shape);
            let lp = loss_value(&mp.forward(input).expect("fd forward"));
            let lm = loss_value(&mm.forward(input).expect("fd forward"));
            (lp - lm) / (2.0 * FD_STEP)
        })
        .collect();

    assert_matches_fd(label, &analytic, &numeric);
}

// ---------------------------------------------------------------------------
// Layer factories
// ---------------------------------------------------------------------------

const LN_SHAPE: [usize; 3] = [2, 3, 4];
const GN_SHAPE: [usize; 4] = [2, 4, 3, 2];
const BN1_SHAPE: [usize; 2] = [2, 3];
const BN2_SHAPE: [usize; 4] = [2, 3, 2, 4];
const BN3_SHAPE: [usize; 5] = [2, 3, 2, 2, 2];

fn layer_norm(normalized: Vec<usize>, salt: usize) -> Box<dyn Module> {
    let module = LayerNorm::new(normalized.clone()).expect("LayerNorm");
    set_affine(&module, &normalized, salt);
    Box::new(module)
}

fn layer_norm_non_affine(normalized: Vec<usize>) -> Box<dyn Module> {
    Box::new(
        LayerNorm::with_config(normalized, NormalizationConfig::non_affine()).expect("LayerNorm"),
    )
}

fn group_norm(salt: usize) -> Box<dyn Module> {
    let module = GroupNorm::new(2, 4).expect("GroupNorm");
    set_affine(&module, &[4], salt);
    Box::new(module)
}

fn batch_norm_1d(salt: usize) -> Box<dyn Module> {
    let module = BatchNorm1d::new(3).expect("BatchNorm1d");
    set_affine(&module, &[3], salt);
    Box::new(module)
}

fn batch_norm_2d(salt: usize) -> Box<dyn Module> {
    let module = BatchNorm2d::new(3).expect("BatchNorm2d");
    set_affine(&module, &[3], salt);
    Box::new(module)
}

fn batch_norm_3d(salt: usize) -> Box<dyn Module> {
    let module = BatchNorm3d::new(3).expect("BatchNorm3d");
    set_affine(&module, &[3], salt);
    Box::new(module)
}

fn instance_norm_1d(salt: usize) -> Box<dyn Module> {
    let module = InstanceNorm1d::new(3).expect("InstanceNorm1d");
    set_affine(&module, &[3], salt);
    Box::new(module)
}

fn instance_norm_2d(salt: usize) -> Box<dyn Module> {
    let module = InstanceNorm2d::new(3).expect("InstanceNorm2d");
    set_affine(&module, &[3], salt);
    Box::new(module)
}

fn instance_norm_3d(salt: usize) -> Box<dyn Module> {
    let module = InstanceNorm3d::new(3).expect("InstanceNorm3d");
    set_affine(&module, &[3], salt);
    Box::new(module)
}

fn rms_norm(normalized: Vec<usize>, salt: usize) -> Box<dyn Module> {
    let module = RMSNorm::new(normalized.clone()).expect("RMSNorm");
    set_affine(&module, &normalized, salt);
    Box::new(module)
}

fn renorm_schedule() -> BatchRenormSchedule {
    BatchRenormSchedule {
        r_max: 3.0,
        d_max: 5.0,
        warmup_steps: 0,
        ramp_steps: 0,
    }
}

fn batch_renorm_2d(salt: usize) -> Box<dyn Module> {
    let module = BatchRenorm2d::with_config(3, NormalizationConfig::default(), renorm_schedule())
        .expect("BatchRenorm2d");
    set_affine(&module, &[3], salt);
    Box::new(module)
}

/// The warmup regime: `r_max = 1` and `d_max = 0` pin the correction terms at
/// their identity values, so the layer *is* batch normalization and the
/// stop-gradient on `r`/`d` is a no-op.
fn batch_renorm_2d_warmup(salt: usize) -> Box<dyn Module> {
    let schedule = BatchRenormSchedule {
        r_max: 1.0,
        d_max: 0.0,
        warmup_steps: 1,
        ramp_steps: 0,
    };
    let module = BatchRenorm2d::with_config(3, NormalizationConfig::default(), schedule)
        .expect("BatchRenorm2d");
    set_affine(&module, &[3], salt);
    Box::new(module)
}

fn virtual_batch_norm_2d(salt: usize) -> Box<dyn Module> {
    let mut module = VirtualBatchNorm2d::new(3).expect("VirtualBatchNorm2d");
    set_affine(&module, &[3], salt);
    let reference = plain_input(&[1, 3, 2, 4], 12);
    module
        .set_reference_batch(&reference)
        .expect("reference batch");
    Box::new(module)
}

fn switchable_norm_2d(salt: usize) -> Box<dyn Module> {
    let module = SwitchableNorm2d::new(3).expect("SwitchableNorm2d");
    set_affine(&module, &[3], salt);
    set_param(&module, "switch_weight", &seeded(9, 14), &[3, 3]);
    Box::new(module)
}

// ---------------------------------------------------------------------------
// Forward-value pins, captured from the pre-rewrite tree
// ---------------------------------------------------------------------------

#[rustfmt::skip]
const LAYER_NORM_1D: [f32; 24] = [
    1.62023878, -1.19090688, -0.58685315, 0.27044746, -1.63115454, -0.42600062, 0.20013705,
    1.05223250, -0.68436968, 0.25305948, 1.08661783, -1.63252807, -0.68436986, 0.25305939,
    1.08661795, -1.63252819, -0.68436980, 0.25305936, 1.08661783, -1.63252807, 0.48606592,
    1.02275586, -1.47888529, -0.69601190,
];

#[rustfmt::skip]
const LAYER_NORM_2D: [f32; 24] = [
    1.27661657, -0.67762816, 0.19884902, 0.83347297, -1.47562146, -0.79893613, 0.90207696,
    1.22825015, -0.81796014, -0.18333632, 0.67561519, -2.32398820, -1.11342740, 0.65729880,
    1.08512425, -1.12259901, -0.44942430, 0.47432008, 2.44863439, -0.48208693, 0.14410605,
    1.02086866, -1.95813918, -1.03439486,
];

#[rustfmt::skip]
const LAYER_NORM_NON_AFFINE: [f32; 24] = [
    1.22171617, -1.40967250, -0.40723881, 0.59519517, -1.34162295, -0.44720763, 0.44720763,
    1.34162307, -0.59519506, 0.40723881, 1.40967250, -1.22171617, -0.59519517, 0.40723872,
    1.40967262, -1.22171628, -0.59519511, 0.40723869, 1.40967250, -1.22171617, 0.32755500,
    1.37573099, -1.37573099, -0.32755500,
];

#[rustfmt::skip]
const GROUP_NORM: [f32; 48] = [
    -1.13404417, -0.29722637, 0.53959143, 1.37640905, -0.82023746, 0.01658022, 0.84343863,
    -1.66515803, -0.70950210, 0.24615362, 1.20180953, -1.30678701, -0.42418256, 0.71567965,
    -2.27645874, -1.13659644, 0.00326580, 1.14312816, -1.35565090, -0.08972079, 1.17620921,
    2.44213939, -0.88092721, 0.38500294, 0.64861709, -1.58891070, -0.73651916, 0.11587246,
    0.96826386, -1.26926374, -0.60722518, 0.36621603, 1.33965731, -1.21562600, -0.24218483,
    0.73125648, -1.71844375, -0.64394987, 0.43054381, 1.50503778, -1.31550848, -0.24101472,
    1.62351954, -1.50897658, -0.31564465, 0.87768722, 2.07101917, -1.06147718,
];

#[rustfmt::skip]
const BATCH_NORM_1D_TRAIN: [f32; 6] = [
    0.95870578, -0.70866269, 0.92894185, -0.44105873, 0.94395679, -0.97600096,
];

#[rustfmt::skip]
const BATCH_NORM_1D_RUNNING_MEAN: [f32; 3] = [0.02758620, -0.01724138, -0.06206897];

#[rustfmt::skip]
const BATCH_NORM_1D_RUNNING_VAR: [f32; 3] = [0.90594530, 1.03697979, 0.90594530];

#[rustfmt::skip]
const BATCH_NORM_2D_TRAIN: [f32; 48] = [
    -1.02636218, 0.02685478, 1.08007157, -1.68462253, -0.63140577, 0.42181107, 1.47502792,
    -1.28966630, -0.19408593, 1.00200152, -2.13772774, -0.94164038, 0.25444689, 1.45053434,
    -1.68919504, -0.49310768, 1.08842373, -0.88852310, -0.13540044, 0.61772221, 1.37084496,
    -0.60610211, 0.14702053, 0.90014315, 1.34337604, -1.42131853, -0.36810166, 0.68511522,
    -2.07957888, -1.02636218, 0.02685478, 1.08007157, -1.83870590, -0.64261854, 0.55346882,
    1.74955618, -1.39017308, -0.19408593, 1.00200152, -2.13772774, 0.05288023, 0.80600286,
    1.55912554, -0.41782144, 0.33530122, 1.08842373, -0.88852310, -0.13540044,
];

#[rustfmt::skip]
const BATCH_NORM_2D_RUNNING_MEAN_STEP2: [f32; 3] = [0.00297414, -0.00090517, -0.01728448];

#[rustfmt::skip]
const BATCH_NORM_2D_RUNNING_VAR_STEP2: [f32; 3] = [0.87582731, 0.87510318, 0.87753993];

#[rustfmt::skip]
const BATCH_NORM_2D_EVAL: [f32; 48] = [
    -0.66539687, -0.01070169, 0.64399338, -1.07458127, -0.41988614, 0.23480898, 0.88950408,
    -0.82907057, -0.30615294, 0.42331156, -1.49153268, -0.76206815, -0.03260377, 0.69686079,
    -1.21798348, -0.48851898, 0.68515754, -0.49470052, -0.04523078, 0.40423900, 0.85370886,
    -0.32614937, 0.12332039, 0.57279015, 0.80766726, -0.91090751, -0.25621241, 0.39848271,
    -1.32009196, -0.66539687, -0.01070169, 0.64399338, -1.30916655, -0.57970202, 0.14976248,
    0.87922680, -1.03561735, -0.30615294, 0.42331156, -1.49153268, 0.06713668, 0.51660639,
    0.96607625, -0.21378192, 0.23568782, 0.68515754, -0.49470052, -0.04523078,
];

#[rustfmt::skip]
const BATCH_NORM_3D_TRAIN: [f32; 48] = [
    1.44674337, -0.89135277, -0.00064944, 0.89005375, -1.44804239, -0.55733907, 0.33336428,
    1.22406769, -1.24189627, -0.24857807, 0.74474019, 1.73805857, -0.86940199, 0.12391624,
    1.11723459, -1.49022591, -1.02563119, 0.08748442, 1.20059979, -1.72132814, -0.60821283,
    0.50490272, 1.61801815, -1.30390990, 0.22202630, 1.11272967, -1.22536659, -0.33466318,
    0.55604005, 1.44674337, -0.89135277, -0.00064944, 0.99306965, -1.61439073, -0.62107241,
    0.37224582, 1.36556423, -1.24189627, -0.24857807, 0.74474019, 1.47887886, -1.44304943,
    -0.32993400, 0.78318149, -2.13874650, -1.02563119, 0.08748442, 1.20059979,
];

#[rustfmt::skip]
const BATCH_NORM_3D_RUNNING_MEAN: [f32; 3] = [-0.00301724, -0.01163793, 0.00474138];

#[rustfmt::skip]
const BATCH_NORM_3D_RUNNING_VAR: [f32; 3] = [0.93471855, 0.93609786, 0.93609786];

#[rustfmt::skip]
const INSTANCE_NORM_1D: [f32; 6] = [
    -0.35294119, 0.30588236, 0.16470590, -0.35294119, 0.30588236, 0.16470590,
];

#[rustfmt::skip]
const INSTANCE_NORM_2D: [f32; 48] = [
    -0.90257829, 0.22260432, 1.34778666, -1.60581732, -0.48063475, 0.64454770, 1.76973009,
    -1.18387377, -0.00163442, 1.30293810, -2.12156439, -0.81699198, 0.48758024, 1.79215264,
    -1.63234973, -0.32777739, 0.50154614, -1.55372655, -0.77076554, 0.01219553, 0.79515672,
    -1.26011622, -0.47715512, 0.30580589, 1.52850950, -1.21614885, -0.17056470, 0.87501931,
    -1.86963880, -0.82405472, 0.22152944, 1.26711333, -1.59809256, -0.43686789, 0.72435683,
    1.88558161, -1.16263330, -0.00140874, 1.15981627, -1.88839889, -0.56367874, 0.22198644,
    1.00765169, -1.05471945, -0.26905429, 0.51661086, -1.54576015, -0.76009500,
];

#[rustfmt::skip]
const INSTANCE_NORM_3D_NON_AFFINE: [f32; 48] = [
    1.35892797, -1.04422891, -0.12874053, 0.78674763, -1.61640894, -0.70092070, 0.21456756,
    1.13005602, -1.13005602, -0.21456760, 0.70092076, 1.61640918, -0.78674787, 0.12874052,
    1.04422891, -1.35892808, -0.76967418, 0.21550880, 1.20069158, -1.38541341, -0.40023050,
    0.58495235, 1.57013512, -1.01596975, 0.12856255, 1.15706372, -1.54275143, -0.51425046,
    0.51425046, 1.54275155, -1.15706360, -0.12856261, 1.01596963, -1.57013512, -0.58495229,
    0.40023047, 1.38541329, -1.20069158, -0.21550876, 0.76967394, 1.35892808, -1.04422903,
    -0.12874058, 0.78674775, -1.61640918, -0.70092082, 0.21456766, 1.13005590,
];

#[rustfmt::skip]
const RMS_NORM: [f32; 24] = [
    1.16841567, -0.75656027, 0.05759999, 1.10096693, -1.06488156, -0.44680959, 0.39687029,
    1.46615827, -0.78065276, -0.14366277, 0.71094650, -1.78317523, -0.60715765, 0.16330446,
    1.18113911, -1.60759878, -0.33066750, 0.49551249, 1.57189226, -1.20180511, -0.04653051,
    0.81348121, -1.67220950, -0.76598805,
];

#[rustfmt::skip]
const BATCH_RENORM_2D_TRAIN: [f32; 48] = [
    -0.28048247, 0.36706430, 1.01461089, -0.68519914, -0.03765242, 0.60989428, 1.25744092,
    -0.44236913, 0.04766721, 0.43387222, -0.57991582, -0.19371083, 0.19249406, 0.57869905,
    -0.43508893, -0.04888399, 0.25276843, -0.94395864, -0.48806259, -0.03216655, 0.42372954,
    -0.77299762, -0.31710160, 0.13879445, 1.17649770, -0.52331245, 0.12423421, 0.77178097,
    -0.92802918, -0.28048247, 0.36706430, 1.01461089, -0.48336455, -0.09715959, 0.28904536,
    0.67525023, -0.33853769, 0.04766721, 0.43387222, -0.57991582, -0.37408856, 0.08180743,
    0.53770351, -0.65902358, -0.20312756, 0.25276843, -0.94395864, -0.48806259,
];

#[rustfmt::skip]
const BATCH_RENORM_2D_RUNNING_MEAN: [f32; 3] = [0.00474138, -0.00387931, -0.01250000];

#[rustfmt::skip]
const BATCH_RENORM_2D_RUNNING_VAR: [f32; 3] = [0.93609786, 0.93471855, 0.93333924];

#[rustfmt::skip]
const VIRTUAL_BATCH_NORM_2D: [f32; 48] = [
    -1.01507080, -0.06948620, 0.87609816, -1.60606098, -0.66047651, 0.28510794, 1.23069239,
    -1.25146675, 0.52187359, 1.60304165, -1.23502433, -0.15385625, 0.92731154, 2.00847983,
    -0.82958639, 0.25158173, 1.30643904, -1.84622324, -0.64520895, 0.55580527, 1.75681949,
    -1.39584303, -0.19482866, 1.00618553, 1.11249435, -1.36966491, -0.42408046, 0.52150404,
    -1.96065521, -1.01507080, -0.06948620, 0.87609816, -0.96473229, 0.11643575, 1.19760370,
    2.27877164, -0.55929422, 0.52187359, 1.60304165, -1.23502433, -0.34495535, 0.85605872,
    2.05707312, -1.09558940, 0.10542493, 1.30643904, -1.84622324, -0.64520895,
];

#[rustfmt::skip]
const SWITCHABLE_NORM_2D: [f32; 48] = [
    -0.54822350, 0.22208162, 0.99238646, -1.02966404, -0.25935903, 0.51094598, 1.28125083,
    -0.74079973, -0.00031035, 0.91256821, -1.48373795, -0.57085931, 0.34201908, 1.25489759,
    -1.14140844, -0.22852990, 0.74832487, -1.96666598, -0.93238378, 0.10189858, 1.13618100,
    -1.57881021, -0.54452789, 0.48975441, 1.14864743, -0.79912651, -0.05711737, 0.68489170,
    -1.26288223, -0.52087307, 0.22113612, 0.96314508, -1.19121253, -0.32397211, 0.54326826,
    1.41050851, -0.86599737, 0.00124294, 0.86848342, -1.40802264, -0.65050733, 0.36112210,
    1.37275183, -1.28277576, -0.27114627, 0.74048316, -1.91504431, -0.90341473,
];

// ---------------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------------

#[test]
fn layer_norm_forward_matches_the_pre_rewrite_values() {
    let _serial = serialize();
    let x = plain_input(&LN_SHAPE, 1);

    let affine_1d = layer_norm(vec![4], 1);
    assert_pinned(
        "LayerNorm([4])",
        &affine_1d.forward(&x).expect("forward").to_vec().expect("v"),
        &LAYER_NORM_1D,
    );

    let affine_2d = layer_norm(vec![3, 4], 2);
    assert_pinned(
        "LayerNorm([3, 4])",
        &affine_2d.forward(&x).expect("forward").to_vec().expect("v"),
        &LAYER_NORM_2D,
    );

    let non_affine = layer_norm_non_affine(vec![4]);
    assert_pinned(
        "LayerNorm([4], non-affine)",
        &non_affine
            .forward(&x)
            .expect("forward")
            .to_vec()
            .expect("v"),
        &LAYER_NORM_NON_AFFINE,
    );
}

#[test]
fn layer_norm_input_gradient_matches_finite_differences() {
    let _serial = serialize();
    check_input_gradient(
        "LayerNorm([4]) d/dx",
        || layer_norm(vec![4], 1),
        &LN_SHAPE,
        1,
    );
    check_input_gradient(
        "LayerNorm([3, 4]) d/dx",
        || layer_norm(vec![3, 4], 2),
        &LN_SHAPE,
        1,
    );
    check_input_gradient(
        "LayerNorm([4], non-affine) d/dx",
        || layer_norm_non_affine(vec![4]),
        &LN_SHAPE,
        1,
    );
}

#[test]
fn layer_norm_affine_gradients_match_finite_differences() {
    let _serial = serialize();
    let x = plain_input(&LN_SHAPE, 1);
    check_param_gradient(
        "LayerNorm gamma",
        || layer_norm(vec![4], 1),
        &x,
        "weight",
        &seeded_weight(4, 1),
        &[4],
    );
    check_param_gradient(
        "LayerNorm beta",
        || layer_norm(vec![4], 1),
        &x,
        "bias",
        &seeded_bias(4, 1),
        &[4],
    );
}

// ---------------------------------------------------------------------------
// GroupNorm
// ---------------------------------------------------------------------------

#[test]
fn group_norm_forward_matches_the_pre_rewrite_values() {
    let _serial = serialize();
    let x = plain_input(&GN_SHAPE, 3);
    assert_pinned(
        "GroupNorm(2, 4)",
        &group_norm(3)
            .forward(&x)
            .expect("forward")
            .to_vec()
            .expect("v"),
        &GROUP_NORM,
    );
}

#[test]
fn group_norm_input_gradient_matches_finite_differences() {
    let _serial = serialize();
    check_input_gradient("GroupNorm d/dx", || group_norm(3), &GN_SHAPE, 3);
}

#[test]
fn group_norm_affine_gradients_match_finite_differences() {
    let _serial = serialize();
    let x = plain_input(&GN_SHAPE, 3);
    check_param_gradient(
        "GroupNorm gamma",
        || group_norm(3),
        &x,
        "weight",
        &seeded_weight(4, 3),
        &[4],
    );
    check_param_gradient(
        "GroupNorm beta",
        || group_norm(3),
        &x,
        "bias",
        &seeded_bias(4, 3),
        &[4],
    );
}

// ---------------------------------------------------------------------------
// BatchNorm{1,2,3}d
// ---------------------------------------------------------------------------

#[test]
fn batch_norm_forward_matches_the_pre_rewrite_values() {
    let _serial = serialize();
    assert_pinned(
        "BatchNorm1d",
        &batch_norm_1d(4)
            .forward(&plain_input(&BN1_SHAPE, 4))
            .expect("forward")
            .to_vec()
            .expect("v"),
        &BATCH_NORM_1D_TRAIN,
    );
    assert_pinned(
        "BatchNorm2d",
        &batch_norm_2d(5)
            .forward(&plain_input(&BN2_SHAPE, 5))
            .expect("forward")
            .to_vec()
            .expect("v"),
        &BATCH_NORM_2D_TRAIN,
    );
    assert_pinned(
        "BatchNorm3d",
        &batch_norm_3d(6)
            .forward(&plain_input(&BN3_SHAPE, 6))
            .expect("forward")
            .to_vec()
            .expect("v"),
        &BATCH_NORM_3D_TRAIN,
    );
}

#[test]
fn batch_norm_input_gradient_matches_finite_differences() {
    let _serial = serialize();
    check_input_gradient("BatchNorm1d d/dx", || batch_norm_1d(4), &BN1_SHAPE, 4);
    check_input_gradient("BatchNorm2d d/dx", || batch_norm_2d(5), &BN2_SHAPE, 5);
    check_input_gradient("BatchNorm3d d/dx", || batch_norm_3d(6), &BN3_SHAPE, 6);
}

#[test]
fn batch_norm_affine_gradients_match_finite_differences() {
    let _serial = serialize();
    let x = plain_input(&BN2_SHAPE, 5);
    check_param_gradient(
        "BatchNorm2d gamma",
        || batch_norm_2d(5),
        &x,
        "weight",
        &seeded_weight(3, 5),
        &[3],
    );
    check_param_gradient(
        "BatchNorm2d beta",
        || batch_norm_2d(5),
        &x,
        "bias",
        &seeded_bias(3, 5),
        &[3],
    );
}

#[test]
fn batch_norm_running_statistics_are_unchanged_by_the_rewrite() {
    let _serial = serialize();

    let bn1 = batch_norm_1d(4);
    let _ = bn1.forward(&plain_input(&BN1_SHAPE, 4)).expect("forward");
    // NOTE: `&bn1` would *unsize* through `impl Module for Box<dyn Module>`
    // (core/mod.rs:903), which forwards only a subset of the trait and lets
    // `named_buffers` fall back to the empty default. `&*bn1` reborrows the
    // inner trait object and dispatches to the real layer.
    let bn1 = RunningStatsSnapshot::of(&*bn1);
    assert_pinned(
        "BatchNorm1d running_mean",
        &bn1.running_mean,
        &BATCH_NORM_1D_RUNNING_MEAN,
    );
    assert_pinned(
        "BatchNorm1d running_var",
        &bn1.running_var,
        &BATCH_NORM_1D_RUNNING_VAR,
    );

    let bn3 = batch_norm_3d(6);
    let _ = bn3.forward(&plain_input(&BN3_SHAPE, 6)).expect("forward");
    let bn3 = RunningStatsSnapshot::of(&*bn3);
    assert_pinned(
        "BatchNorm3d running_mean",
        &bn3.running_mean,
        &BATCH_NORM_3D_RUNNING_MEAN,
    );
    assert_pinned(
        "BatchNorm3d running_var",
        &bn3.running_var,
        &BATCH_NORM_3D_RUNNING_VAR,
    );
}

/// Running statistics read straight out of a layer's published buffers, so the
/// helper works for any layer that registers them.
struct RunningStatsSnapshot {
    running_mean: Vec<f32>,
    running_var: Vec<f32>,
    running_mean_requires_grad: bool,
    running_var_requires_grad: bool,
}

impl RunningStatsSnapshot {
    fn of(module: &dyn Module) -> Self {
        let buffers = module.named_buffers();
        let mean = buffers
            .get("running_mean")
            .expect("running_mean buffer")
            .read()
            .clone();
        let var = buffers
            .get("running_var")
            .expect("running_var buffer")
            .read()
            .clone();
        Self {
            running_mean: mean.to_vec().expect("running_mean values"),
            running_var: var.to_vec().expect("running_var values"),
            running_mean_requires_grad: mean.requires_grad(),
            running_var_requires_grad: var.requires_grad(),
        }
    }
}

#[test]
fn batch_norm_two_step_running_statistics_and_eval_forward_are_unchanged() {
    let _serial = serialize();

    let mut bn = BatchNorm2d::new(3).expect("BatchNorm2d");
    set_affine(&bn, &[3], 5);
    let x = plain_input(&BN2_SHAPE, 5);
    let x_second = plain_input(&BN2_SHAPE, 9);

    let _ = bn.forward(&x).expect("train step 1");
    let _ = bn.forward(&x_second).expect("train step 2");

    assert_pinned(
        "BatchNorm2d running_mean after two steps",
        &bn.running_mean().expect("rm").to_vec().expect("v"),
        &BATCH_NORM_2D_RUNNING_MEAN_STEP2,
    );
    assert_pinned(
        "BatchNorm2d running_var after two steps",
        &bn.running_var().expect("rv").to_vec().expect("v"),
        &BATCH_NORM_2D_RUNNING_VAR_STEP2,
    );
    assert_eq!(
        bn.num_batches_tracked().expect("counter"),
        Some(2.0),
        "momentum semantics must be unchanged"
    );

    bn.eval();
    assert_pinned(
        "BatchNorm2d eval forward",
        &bn.forward(&x).expect("eval forward").to_vec().expect("v"),
        &BATCH_NORM_2D_EVAL,
    );
}

#[test]
fn batch_norm_running_statistics_never_become_graph_nodes() {
    let _serial = serialize();

    let bn = BatchNorm2d::new(3).expect("BatchNorm2d");
    set_affine(&bn, &[3], 5);
    // Feed a gradient-tracking input: with the batch statistics on the graph, a
    // naive momentum update would splice the running buffers into it.
    for salt in [5usize, 9, 13] {
        let input = plain_input(&BN2_SHAPE, salt).requires_grad_(true);
        let output = bn.forward(&input).expect("train forward");
        loss_tensor(&output).backward().expect("backward");
    }

    let handle = RunningStatsSnapshot::of(&bn);
    assert!(
        !handle.running_mean_requires_grad,
        "running_mean is a buffer, never a graph node"
    );
    assert!(
        !handle.running_var_requires_grad,
        "running_var is a buffer, never a graph node"
    );
}

#[test]
fn batch_norm_eval_gradient_treats_running_statistics_as_constants() {
    let _serial = serialize();

    let mut bn = BatchNorm2d::new(3).expect("BatchNorm2d");
    set_affine(&bn, &[3], 5);
    // Two training steps so the running statistics are no longer 0 / 1.
    let _ = bn.forward(&plain_input(&BN2_SHAPE, 5)).expect("train 1");
    let _ = bn.forward(&plain_input(&BN2_SHAPE, 9)).expect("train 2");

    let running_var = bn.running_var().expect("rv").to_vec().expect("v");
    let gamma = seeded_weight(3, 5);
    bn.eval();

    let input = plain_input(&BN2_SHAPE, 5).requires_grad_(true);
    let output = bn.forward(&input).expect("eval forward");
    loss_tensor(&output).backward().expect("backward");
    let analytic = input.grad().expect("input grad").to_vec().expect("v");

    // Closed form: with the statistics constant, d loss / d x[n, c, h, w] is
    // exactly seed[n, c, h, w] * gamma[c] / sqrt(running_var[c] + eps).
    let dims = BN2_SHAPE;
    let spatial = dims[2] * dims[3];
    let seed = pattern(analytic.len());
    let expected: Vec<f32> = (0..analytic.len())
        .map(|i| {
            let channel = (i / spatial) % dims[1];
            seed[i] * gamma[channel] / (running_var[channel] + 1e-5).sqrt()
        })
        .collect();
    for (i, (&got, &want)) in analytic.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= 1e-4,
            "eval-mode gradient[{i}] = {got}, closed form gives {want}; \
             the running statistics must not be differentiated through"
        );
    }
}

#[test]
fn sync_batch_norm_tracks_batch_norm_exactly() {
    let _serial = serialize();

    let sync = SyncBatchNorm2d::new(3).expect("SyncBatchNorm2d");
    set_affine(&sync, &[3], 5);
    let input = plain_input(&BN2_SHAPE, 5).requires_grad_(true);
    let output = sync.forward(&input).expect("forward");
    assert_pinned(
        "SyncBatchNorm2d forward",
        &output.to_vec().expect("v"),
        &BATCH_NORM_2D_TRAIN,
    );
    loss_tensor(&output).backward().expect("backward");
    let sync_grad = input.grad().expect("input grad").to_vec().expect("v");

    let plain = batch_norm_2d(5);
    let plain_input_tensor = plain_input(&BN2_SHAPE, 5).requires_grad_(true);
    let plain_output = plain.forward(&plain_input_tensor).expect("forward");
    loss_tensor(&plain_output).backward().expect("backward");
    let plain_grad = plain_input_tensor
        .grad()
        .expect("input grad")
        .to_vec()
        .expect("v");

    assert_pinned("SyncBatchNorm2d d/dx", &sync_grad, &plain_grad);
}

// ---------------------------------------------------------------------------
// InstanceNorm{1,2,3}d
// ---------------------------------------------------------------------------

#[test]
fn instance_norm_forward_matches_the_pre_rewrite_values() {
    let _serial = serialize();
    assert_pinned(
        "InstanceNorm1d",
        &instance_norm_1d(7)
            .forward(&plain_input(&BN1_SHAPE, 4))
            .expect("forward")
            .to_vec()
            .expect("v"),
        &INSTANCE_NORM_1D,
    );
    assert_pinned(
        "InstanceNorm2d",
        &instance_norm_2d(8)
            .forward(&plain_input(&BN2_SHAPE, 5))
            .expect("forward")
            .to_vec()
            .expect("v"),
        &INSTANCE_NORM_2D,
    );
    let non_affine =
        InstanceNorm3d::with_config(3, NormalizationConfig::non_affine()).expect("InstanceNorm3d");
    assert_pinned(
        "InstanceNorm3d (non-affine)",
        &non_affine
            .forward(&plain_input(&BN3_SHAPE, 6))
            .expect("forward")
            .to_vec()
            .expect("v"),
        &INSTANCE_NORM_3D_NON_AFFINE,
    );
}

#[test]
fn instance_norm_input_gradient_matches_finite_differences() {
    let _serial = serialize();
    check_input_gradient("InstanceNorm2d d/dx", || instance_norm_2d(8), &BN2_SHAPE, 5);
    check_input_gradient("InstanceNorm3d d/dx", || instance_norm_3d(9), &BN3_SHAPE, 6);
}

#[test]
fn instance_norm_affine_gradients_match_finite_differences() {
    let _serial = serialize();
    let x = plain_input(&BN2_SHAPE, 5);
    check_param_gradient(
        "InstanceNorm2d gamma",
        || instance_norm_2d(8),
        &x,
        "weight",
        &seeded_weight(3, 8),
        &[3],
    );
    check_param_gradient(
        "InstanceNorm2d beta",
        || instance_norm_2d(8),
        &x,
        "bias",
        &seeded_bias(3, 8),
        &[3],
    );
}

#[test]
fn instance_norm_1d_gradient_is_exactly_zero() {
    let _serial = serialize();

    // `InstanceNorm1d` normalizes an (N, C) input over an empty spatial extent:
    // the per-instance mean *is* the element itself and the variance is 0, so the
    // output is the bias, constant in the input. The honest gradient is therefore
    // exactly zero. Before the rewrite the mean was a detached constant, which
    // produced `gamma / sqrt(eps)` — roughly 316 * gamma — instead.
    let input = plain_input(&BN1_SHAPE, 4).requires_grad_(true);
    let module = instance_norm_1d(7);
    let output = module.forward(&input).expect("forward");
    loss_tensor(&output).backward().expect("backward");
    let analytic = input.grad().expect("input grad").to_vec().expect("v");

    for (i, &g) in analytic.iter().enumerate() {
        assert!(
            g.abs() <= 1e-4,
            "InstanceNorm1d gradient[{i}] = {g}, but the output does not depend on the input"
        );
    }

    check_input_gradient("InstanceNorm1d d/dx", || instance_norm_1d(7), &BN1_SHAPE, 4);
}

/// Independent instance-normalization oracle written straight from the
/// definition, used where the pre-rewrite implementation was wrong and there is
/// therefore no old value worth pinning.
fn instance_norm_reference(
    data: &[f32],
    dims: &[usize],
    weight: &[f32],
    bias: &[f32],
    eps: f32,
) -> Vec<f32> {
    let channels = dims[1];
    let spatial: usize = dims[2..].iter().product();
    let mut out = vec![0.0f32; data.len()];
    for instance in 0..(dims[0] * channels) {
        let channel = instance % channels;
        let base = instance * spatial;
        let slice = &data[base..base + spatial];
        let mean = slice.iter().sum::<f32>() / spatial as f32;
        let var = slice.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / spatial as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for (offset, value) in slice.iter().enumerate() {
            out[base + offset] = (value - mean) * inv_std * weight[channel] + bias[channel];
        }
    }
    out
}

#[test]
fn instance_norm_2d_affine_uses_the_channel_axis_not_the_last_axis() {
    let _serial = serialize();

    // W == C == 3. `utils::apply_normalization`'s trailing-axis broadcast used to
    // align the per-channel gamma/beta with the *width* axis here, scaling the
    // wrong activations entirely.
    let dims = [2usize, 3, 2, 3];
    let data = seeded(dims.iter().product(), 15);
    let input = Tensor::from_vec(data.clone(), &dims).expect("input");
    let module = instance_norm_2d(8);
    let got = module
        .forward(&input)
        .expect("forward")
        .to_vec()
        .expect("v");

    let want =
        instance_norm_reference(&data, &dims, &seeded_weight(3, 8), &seeded_bias(3, 8), 1e-5);
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= 1e-5,
            "InstanceNorm2d[{i}] = {g}, definition gives {w}: gamma/beta must be \
             broadcast along the channel axis even when W == C"
        );
    }
}

#[test]
fn instance_norm_3d_affine_forward_works_on_five_dimensional_inputs() {
    let _serial = serialize();

    // Before the rewrite this errored out with
    // `BroadcastError { shape1: [2, 3, 2, 2, 2], shape2: [3] }`: the manual
    // broadcast fallback only knew how to reshape a per-channel vector for 2-D
    // and 4-D inputs, so the affine 3-D instance norm was unusable.
    let dims = BN3_SHAPE;
    let data = seeded(dims.iter().product(), 6);
    let input = Tensor::from_vec(data.clone(), &dims).expect("input");
    let module = instance_norm_3d(9);
    let got = module
        .forward(&input)
        .expect("InstanceNorm3d must support affine parameters")
        .to_vec()
        .expect("v");

    let want =
        instance_norm_reference(&data, &dims, &seeded_weight(3, 9), &seeded_bias(3, 9), 1e-5);
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= 1e-5,
            "InstanceNorm3d[{i}] = {g}, definition gives {w}"
        );
    }
}

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

#[test]
fn rms_norm_forward_matches_the_pre_rewrite_values() {
    let _serial = serialize();
    assert_pinned(
        "RMSNorm([4])",
        &rms_norm(vec![4], 10)
            .forward(&plain_input(&LN_SHAPE, 1))
            .expect("forward")
            .to_vec()
            .expect("v"),
        &RMS_NORM,
    );
}

#[test]
fn rms_norm_input_gradient_matches_finite_differences() {
    let _serial = serialize();
    check_input_gradient("RMSNorm d/dx", || rms_norm(vec![4], 10), &LN_SHAPE, 1);
}

#[test]
fn rms_norm_affine_gradient_matches_finite_differences() {
    let _serial = serialize();
    let x = plain_input(&LN_SHAPE, 1);
    check_param_gradient(
        "RMSNorm gamma",
        || rms_norm(vec![4], 10),
        &x,
        "weight",
        &seeded_weight(4, 10),
        &[4],
    );
}

// ---------------------------------------------------------------------------
// BatchRenorm2d
// ---------------------------------------------------------------------------

#[test]
fn batch_renorm_eval_gradient_treats_running_statistics_as_constants() {
    let _serial = serialize();

    // `BatchRenorm2d` has its own evaluation branch, separate from
    // `batch_norm_forward`'s: it normalizes with the running statistics and must
    // consume them detached, exactly like `BatchNorm2d`.
    let mut renorm =
        BatchRenorm2d::with_config(3, NormalizationConfig::default(), renorm_schedule())
            .expect("BatchRenorm2d");
    set_affine(&renorm, &[3], 11);
    // The training inputs must *require gradients*: only then does a running
    // update that folds in the live batch statistics splice the buffers into the
    // graph, which is precisely the regression this test exists to catch.
    for salt in [5usize, 9] {
        let training_input = plain_input(&BN2_SHAPE, salt).requires_grad_(true);
        let output = renorm.forward(&training_input).expect("train forward");
        loss_tensor(&output).backward().expect("backward");
    }

    let running_var = renorm.running_var().to_vec().expect("v");
    assert!(
        !renorm.running_mean().requires_grad(),
        "running_mean is a buffer, never a graph node"
    );
    assert!(
        !renorm.running_var().requires_grad(),
        "running_var is a buffer, never a graph node"
    );
    let buffers = RunningStatsSnapshot::of(&renorm);
    assert!(
        !buffers.running_mean_requires_grad && !buffers.running_var_requires_grad,
        "the published buffers must match the snapshots"
    );

    let gamma = seeded_weight(3, 11);
    renorm.eval();

    let input = plain_input(&BN2_SHAPE, 5).requires_grad_(true);
    let output = renorm.forward(&input).expect("eval forward");
    loss_tensor(&output).backward().expect("backward");
    let analytic = input.grad().expect("input grad").to_vec().expect("v");

    let dims = BN2_SHAPE;
    let spatial = dims[2] * dims[3];
    let seed = pattern(analytic.len());
    for (i, &got) in analytic.iter().enumerate() {
        let channel = (i / spatial) % dims[1];
        let want = seed[i] * gamma[channel] / (running_var[channel] + 1e-5).sqrt();
        assert!(
            (got - want).abs() <= 1e-4,
            "BatchRenorm2d eval gradient[{i}] = {got}, closed form gives {want}; \
             the running statistics must not be differentiated through"
        );
    }
}

#[test]
fn batch_renorm_forward_and_running_statistics_are_unchanged() {
    let _serial = serialize();
    let renorm = BatchRenorm2d::with_config(3, NormalizationConfig::default(), renorm_schedule())
        .expect("BatchRenorm2d");
    set_affine(&renorm, &[3], 11);
    let x = plain_input(&BN2_SHAPE, 5);

    assert_pinned(
        "BatchRenorm2d forward",
        &renorm.forward(&x).expect("forward").to_vec().expect("v"),
        &BATCH_RENORM_2D_TRAIN,
    );
    assert_pinned(
        "BatchRenorm2d running_mean",
        &renorm.running_mean().to_vec().expect("v"),
        &BATCH_RENORM_2D_RUNNING_MEAN,
    );
    assert_pinned(
        "BatchRenorm2d running_var",
        &renorm.running_var().to_vec().expect("v"),
        &BATCH_RENORM_2D_RUNNING_VAR,
    );
    assert_eq!(renorm.step(), 1, "the step counter must still advance");
}

#[test]
fn batch_renorm_input_gradient_matches_finite_differences_during_warmup() {
    let _serial = serialize();

    // Batch renormalization deliberately *stops* the gradient at `r` and `d`
    // (Ioffe 2017, Algorithm 1: backward propagates through `mu_B` and
    // `sigma_B`, never through the correction terms). Finite differences see
    // `r` and `d` move with the input, so they are only a valid oracle where
    // both terms sit exactly on their clipping bounds and are therefore locally
    // constant. The warmup regime is precisely that case: `r_max = 1` and
    // `d_max = 0` pin `r ≡ 1` and `d ≡ 0`, and the layer reduces to plain batch
    // normalization — which is what makes this the regime that actually
    // exercises the statistics path.
    check_input_gradient(
        "BatchRenorm2d (warmup) d/dx",
        || batch_renorm_2d_warmup(11),
        &BN2_SHAPE,
        5,
    );
}

#[test]
fn batch_renorm_warmup_is_indistinguishable_from_batch_norm() {
    let _serial = serialize();

    let x = plain_input(&BN2_SHAPE, 5);
    let warmup = batch_renorm_2d_warmup(5)
        .forward(&x)
        .expect("forward")
        .to_vec()
        .expect("v");
    assert_pinned(
        "BatchRenorm2d warmup forward",
        &warmup,
        &BATCH_NORM_2D_TRAIN,
    );
}

#[test]
fn batch_renorm_gradient_is_the_batch_norm_gradient_scaled_by_r() {
    let _serial = serialize();

    // With `r` and `d` held constant, `y = gamma * (x_hat * r + d) + beta` and
    // `y_bn = gamma * x_hat + beta` differ by a per-channel constant factor, so
    // `d y / d x = r_c * d y_bn / d x` exactly. That identity *is* Ioffe's
    // backward rule, and unlike finite differences it stays valid in the
    // unclipped regime.
    let dims = BN2_SHAPE;
    let data = seeded(dims.iter().product(), 5);

    let renorm_input = Tensor::from_vec(data.clone(), &dims)
        .expect("input")
        .requires_grad_(true);
    let renorm = batch_renorm_2d(11);
    loss_tensor(&renorm.forward(&renorm_input).expect("forward"))
        .backward()
        .expect("backward");
    let renorm_grad = renorm_input.grad().expect("grad").to_vec().expect("v");

    let bn_input = Tensor::from_vec(data.clone(), &dims)
        .expect("input")
        .requires_grad_(true);
    let bn = BatchNorm2d::with_config(
        3,
        NormalizationConfig {
            track_running_stats: false,
            ..NormalizationConfig::default()
        },
    )
    .expect("BatchNorm2d");
    set_affine(&bn, &[3], 11);
    loss_tensor(&bn.forward(&bn_input).expect("forward"))
        .backward()
        .expect("backward");
    let bn_grad = bn_input.grad().expect("grad").to_vec().expect("v");

    // r_c = clip(sigma_batch_c / sigma_running_c, 1/r_max, r_max) with the
    // running statistics still at their initial (0, 1).
    let channels = dims[1];
    let spatial = dims[2] * dims[3];
    let per_channel = dims[0] * spatial;
    let eps = 1e-5f32;
    let r: Vec<f32> = (0..channels)
        .map(|c| {
            let values: Vec<f32> = (0..data.len())
                .filter(|i| (i / spatial) % channels == c)
                .map(|i| data[i])
                .collect();
            let mean = values.iter().sum::<f32>() / per_channel as f32;
            let var =
                values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / per_channel as f32;
            ((var + eps).sqrt() / (1.0f32 + eps).sqrt()).clamp(1.0 / 3.0, 3.0)
        })
        .collect();

    for (i, (&got, &plain)) in renorm_grad.iter().zip(bn_grad.iter()).enumerate() {
        let want = plain * r[(i / spatial) % channels];
        assert!(
            (got - want).abs() <= 1e-4,
            "BatchRenorm2d gradient[{i}] = {got}, but r * (BatchNorm2d gradient) = {want}"
        );
    }
}

#[test]
fn batch_renorm_affine_gradients_match_finite_differences() {
    let _serial = serialize();
    let x = plain_input(&BN2_SHAPE, 5);
    check_param_gradient(
        "BatchRenorm2d gamma",
        || batch_renorm_2d(11),
        &x,
        "weight",
        &seeded_weight(3, 11),
        &[3],
    );
    check_param_gradient(
        "BatchRenorm2d beta",
        || batch_renorm_2d(11),
        &x,
        "bias",
        &seeded_bias(3, 11),
        &[3],
    );
}

// ---------------------------------------------------------------------------
// VirtualBatchNorm2d
// ---------------------------------------------------------------------------

#[test]
fn virtual_batch_norm_forward_is_unchanged() {
    let _serial = serialize();
    assert_pinned(
        "VirtualBatchNorm2d",
        &virtual_batch_norm_2d(12)
            .forward(&plain_input(&BN2_SHAPE, 5))
            .expect("forward")
            .to_vec()
            .expect("v"),
        &VIRTUAL_BATCH_NORM_2D,
    );
}

#[test]
fn virtual_batch_norm_input_gradient_matches_finite_differences() {
    let _serial = serialize();
    check_input_gradient(
        "VirtualBatchNorm2d d/dx",
        || virtual_batch_norm_2d(12),
        &BN2_SHAPE,
        5,
    );
}

// ---------------------------------------------------------------------------
// SwitchableNorm2d
// ---------------------------------------------------------------------------

#[test]
fn switchable_norm_forward_is_unchanged() {
    let _serial = serialize();
    assert_pinned(
        "SwitchableNorm2d",
        &switchable_norm_2d(13)
            .forward(&plain_input(&BN2_SHAPE, 5))
            .expect("forward")
            .to_vec()
            .expect("v"),
        &SWITCHABLE_NORM_2D,
    );
}

#[test]
fn switchable_norm_input_gradient_matches_finite_differences() {
    let _serial = serialize();
    check_input_gradient(
        "SwitchableNorm2d d/dx",
        || switchable_norm_2d(13),
        &BN2_SHAPE,
        5,
    );
}

#[test]
fn switchable_norm_switch_weight_gradient_matches_finite_differences() {
    let _serial = serialize();
    let x = plain_input(&BN2_SHAPE, 5);
    check_param_gradient(
        "SwitchableNorm2d switch_weight",
        || switchable_norm_2d(13),
        &x,
        "switch_weight",
        &seeded(9, 14),
        &[3, 3],
    );
    check_param_gradient(
        "SwitchableNorm2d gamma",
        || switchable_norm_2d(13),
        &x,
        "weight",
        &seeded_weight(3, 13),
        &[3],
    );
}
