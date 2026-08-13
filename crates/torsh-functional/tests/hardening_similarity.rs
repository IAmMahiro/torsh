//! Task W6-F hardening: reconnect torsh-functional's loss functions to autograd.
//!
//! The W5 sweep of `torsh-nn` flagged `crates/torsh-functional/src/loss/similarity.rs`
//! as "probably severed": its branch selection ran through
//! [`torsh_tensor::Tensor::where_tensor`], which rebuilds its result from raw
//! data and records nothing, so every loss built on it returned an honestly
//! detached leaf.
//!
//! A `requires_grad` probe over *every* public loss function in
//! `torsh-functional::loss` measured the real picture rather than assuming it:
//!
//! | loss | before W6-F | mechanism |
//! |---|---|---|
//! | `cosine_embedding_loss` | severed | `where_tensor` |
//! | `hinge_embedding_loss` | severed | `where_tensor` |
//! | `contrastive_loss` | severed | `where_tensor` |
//! | `smooth_l1_loss` (regression.rs) | severed | `where_tensor` |
//! | `gradient_penalty_loss` (specialized.rs) | severed | `Tensor::norm` |
//! | `temporal_consistency_loss` (specialized.rs) | severed | `slice(..).to_tensor()` |
//! | `seq2seq_loss_with_attention` (specialized.rs) | severed | scalar `get()` loop |
//! | `margin_ranking_loss` | ATTACHED | `clamp` records since wave 4 |
//! | `triplet_margin_loss` | ATTACHED but NaN at zero distance | `sqrt'(0) = inf` |
//! | `triplet_margin_with_distance_loss` | ATTACHED | caller-supplied distance |
//!
//! Everything else in `loss/` (regression, classification, information, plus
//! `wasserstein_loss`) already carried gradients and is pinned here only where
//! this task changed a shared helper.
//!
//! Every forward value asserted below was captured from the **predecessor**
//! implementation before any edit, so the rewrite cannot quietly change a
//! formula.

use torsh_core::device::DeviceType;
use torsh_core::Result as TorshResult;
use torsh_functional::loss::{
    contrastive_loss, cosine_embedding_loss, gradient_penalty_loss, hinge_embedding_loss,
    margin_ranking_loss, seq2seq_loss_with_attention, smooth_l1_loss, temporal_consistency_loss,
    triplet_margin_loss, triplet_margin_with_distance_loss, ReductionType,
};
use torsh_tensor::Tensor;

/// Serializes every test in this file.
///
/// Grad mode is a process-global `AtomicBool` (`torsh-core/src/grad_mode.rs`),
/// so under plain `cargo test` -- which runs a whole binary's tests in one
/// process across a thread pool, unlike `cargo nextest`'s process-per-test --
/// a `no_grad` window opened anywhere in the process transiently suppresses
/// recording for every other test's tensor ops. Every test in this file either
/// records or asserts recording, so every test takes this lock for its full
/// body. The cost is nil: the largest tensor here has 100 elements.
///
/// Pattern copied from `crates/torsh-tensor/tests/hardening_autograd_unary.rs:19-28`.
static GRAD_MODE_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Takes [`GRAD_MODE_GUARD`], recovering from a poisoned lock so that one
/// failing test does not cascade into a wall of unrelated failures.
fn serialize() -> std::sync::MutexGuard<'static, ()> {
    GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner())
}

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor {
    Tensor::from_data(data, shape.to_vec(), DeviceType::Cpu)
        .expect("tensor creation should succeed")
}

fn leaf(data: Vec<f32>, shape: &[usize]) -> Tensor {
    tensor(data, shape).requires_grad_(true)
}

fn values(t: &Tensor) -> Vec<f32> {
    t.data().expect("tensor data should be readable")
}

fn assert_values(actual: &Tensor, expected: &[f32], what: &str) {
    let got = values(actual);
    assert_eq!(
        got.len(),
        expected.len(),
        "{what}: element count {} != expected {}",
        got.len(),
        expected.len()
    );
    for (index, (a, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= 1e-5,
            "{what}: element {index} is {a}, pinned value is {e} (whole tensor: {got:?})"
        );
    }
}

/// Central finite-difference gradient of a scalar loss w.r.t. every element of
/// `base`, which carries `shape`.
fn fd_grad<F>(base: &[f32], shape: &[usize], loss: F) -> Vec<f32>
where
    F: Fn(&Tensor) -> f32,
{
    let eps = 1e-3_f32;
    let mut out = Vec::with_capacity(base.len());
    for index in 0..base.len() {
        let mut up = base.to_vec();
        up[index] += eps;
        let mut down = base.to_vec();
        down[index] -= eps;
        let high = loss(&tensor(up, shape));
        let low = loss(&tensor(down, shape));
        out.push((high - low) / (2.0 * eps));
    }
    out
}

/// Compares an analytic gradient against a finite-difference estimate with a
/// mixed absolute/relative tolerance -- `eps = 1e-3` in `f32` leaves roughly
/// three good digits, so a purely absolute bound would be either useless on
/// large gradients or vacuous on small ones.
fn assert_grad_close(analytic: &[f32], finite_difference: &[f32], what: &str) {
    assert_eq!(
        analytic.len(),
        finite_difference.len(),
        "{what}: gradient length mismatch"
    );
    for (index, (a, f)) in analytic.iter().zip(finite_difference.iter()).enumerate() {
        assert!(
            a.is_finite(),
            "{what}: analytic gradient element {index} is {a} (whole gradient: {analytic:?})"
        );
        let tolerance = 2e-2 * (1.0 + f.abs());
        assert!(
            (a - f).abs() <= tolerance,
            "{what}: element {index} analytic {a} vs finite-difference {f} \
             (tolerance {tolerance}); analytic={analytic:?} fd={finite_difference:?}"
        );
    }
}

/// Reduces `loss` to a single number the way `backward()` needs it.
fn scalarize(loss: &Tensor) -> TorshResult<Tensor> {
    if loss.shape().dims().is_empty() {
        Ok(loss.clone())
    } else {
        loss.sum()
    }
}

fn grad_of(input: &Tensor, what: &str) -> Vec<f32> {
    values(
        &input
            .grad()
            .unwrap_or_else(|| panic!("{what}: the input received no gradient at all")),
    )
}

// =============================================================================
// Shared fixtures -- every one of these is the exact input the pinned forward
// values were measured on.
// =============================================================================

fn cosine_fixture() -> (Tensor, Tensor, Tensor) {
    (
        leaf(vec![1.0, 2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]),
        leaf(vec![1.1, 1.9, 2.5, -0.5, 1.0, 0.25], &[2, 3]),
        tensor(vec![1.0, -1.0], &[2]),
    )
}

/// A configuration in which both branches are *strictly* inside their arm, so
/// a finite-difference check never straddles the hinge kink.
fn cosine_fd_fixture() -> (Vec<f32>, Vec<f32>, Tensor) {
    (
        vec![1.0, 2.0, 0.5, 0.7, -0.4, 1.3],
        vec![0.8, 1.7, 0.9, 0.9, -0.2, 1.1],
        tensor(vec![1.0, -1.0], &[2]),
    )
}

fn contrastive_fixture() -> (Tensor, Tensor, Tensor) {
    (
        leaf(vec![1.0, 2.0, 0.0, 1.0], &[2, 2]),
        leaf(vec![1.2, 1.8, 0.5, 1.5], &[2, 2]),
        tensor(vec![0.0, 1.0], &[2]),
    )
}

fn triplet_fixture() -> (Tensor, Tensor, Tensor) {
    (
        leaf(vec![1.0, 2.0, 0.0, 1.0], &[2, 2]),
        leaf(vec![1.2, 1.8, 0.5, 1.5], &[2, 2]),
        leaf(vec![-1.0, 0.5, 2.0, -1.0], &[2, 2]),
    )
}

// =============================================================================
// cosine_embedding_loss
// =============================================================================

#[test]
fn sm_cosine_embedding_loss_forward_values_are_pinned() -> TorshResult<()> {
    let _serial = serialize();
    let (input1, input2, target) = cosine_fixture();

    let none = cosine_embedding_loss(&input1, &input2, &target, 0.5, ReductionType::None)?;
    assert_eq!(
        none.shape().dims(),
        &[2],
        "'none' keeps one loss per sample"
    );
    assert_values(&none, &[0.003943801, 0.0], "cosine none");

    let mean = cosine_embedding_loss(&input1, &input2, &target, 0.5, ReductionType::Mean)?;
    assert_eq!(mean.shape().dims(), &[] as &[usize]);
    assert_values(&mean, &[0.0019719005], "cosine mean");

    let sum = cosine_embedding_loss(&input1, &input2, &target, 0.5, ReductionType::Sum)?;
    assert_eq!(sum.shape().dims(), &[] as &[usize]);
    assert_values(&sum, &[0.003943801], "cosine sum");
    Ok(())
}

/// `where_tensor` used to be the only thing checking `target` against the
/// cosine tensor, so the mask rewrite must carry that check itself -- verbatim,
/// down to the `ShapeMismatch` payload.
#[test]
fn sm_cosine_embedding_loss_rejects_a_target_of_the_wrong_shape() {
    let _serial = serialize();
    let input1 = leaf(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let input2 = leaf(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let target = tensor(vec![1.0, 1.0, 1.0], &[3]);

    let error = cosine_embedding_loss(&input1, &input2, &target, 0.0, ReductionType::None)
        .expect_err("one label per sample is the documented contract");
    let message = format!("{error}");
    assert!(
        message.contains("expected [2]") && message.contains("got [3]"),
        "the pre-existing diagnostic named both shapes; got: {message}"
    );
}

#[test]
fn sm_cosine_embedding_loss_stays_on_the_autograd_graph() -> TorshResult<()> {
    let _serial = serialize();
    let (input1, input2, target) = cosine_fixture();

    let loss = cosine_embedding_loss(&input1, &input2, &target, 0.5, ReductionType::Mean)?;
    assert!(
        loss.requires_grad(),
        "cosine_embedding_loss must stay attached to its embeddings"
    );
    loss.backward()?;
    let g1 = grad_of(&input1, "cosine input1");
    let g2 = grad_of(&input2, "cosine input2");
    assert!(
        g1.iter().any(|v| *v != 0.0) && g1.iter().all(|v| v.is_finite()),
        "input1 gradient must be finite and non-trivial, got {g1:?}"
    );
    assert!(
        g2.iter().any(|v| *v != 0.0) && g2.iter().all(|v| v.is_finite()),
        "input2 gradient must be finite and non-trivial, got {g2:?}"
    );
    Ok(())
}

#[test]
fn sm_cosine_embedding_loss_gradient_matches_finite_differences() -> TorshResult<()> {
    let _serial = serialize();
    let (base1, base2, target) = cosine_fd_fixture();
    let shape = [2usize, 3];

    // The pinned forward values of this FD-friendly configuration.
    let pin_input1 = leaf(base1.clone(), &shape);
    let pin_input2 = tensor(base2.clone(), &shape);
    let pinned =
        cosine_embedding_loss(&pin_input1, &pin_input2, &target, 0.0, ReductionType::None)?;
    assert_values(&pinned, &[0.025844216, 0.9747029], "cosine fd fixture");

    let input1 = leaf(base1.clone(), &shape);
    let input2 = tensor(base2.clone(), &shape);
    let loss = cosine_embedding_loss(&input1, &input2, &target, 0.0, ReductionType::Sum)?;
    loss.backward()?;
    let analytic = grad_of(&input1, "cosine fd");

    let fixed_input2 = tensor(base2, &shape);
    let numeric = fd_grad(&base1, &shape, |candidate| {
        cosine_embedding_loss(candidate, &fixed_input2, &target, 0.0, ReductionType::Sum)
            .and_then(|l| l.item())
            .expect("forward should succeed")
    });
    assert_grad_close(&analytic, &numeric, "cosine_embedding_loss d/dinput1");
    Ok(())
}

/// The whole point of the constant-mask composition: sample 0 (`target = +1`)
/// must be pulled *together* and sample 1 (`target = -1`) pushed *apart*, so the
/// two rows' gradients cannot both come from the same branch.
#[test]
fn sm_cosine_embedding_loss_routes_each_row_to_its_own_branch() -> TorshResult<()> {
    let _serial = serialize();
    // Two identical rows; only the label differs.
    let input1 = leaf(vec![1.0, 0.0, 1.0, 0.0], &[2, 2]);
    let input2 = leaf(vec![0.6, 0.8, 0.6, 0.8], &[2, 2]);
    let target = tensor(vec![1.0, -1.0], &[2]);

    let loss = cosine_embedding_loss(&input1, &input2, &target, 0.0, ReductionType::Sum)?;
    loss.backward()?;
    let gradient = grad_of(&input1, "cosine branch routing");

    // d(1 - cos)/dx = -d(cos)/dx and d(cos - margin)/dx = +d(cos)/dx, so with
    // identical rows the two halves must be exact negatives of each other.
    assert!(
        (gradient[0] + gradient[2]).abs() <= 1e-5 && (gradient[1] + gradient[3]).abs() <= 1e-5,
        "the +1 and -1 rows must receive opposite gradients, got {gradient:?}"
    );
    assert!(
        gradient[0] != 0.0 || gradient[1] != 0.0,
        "neither row may receive a zero gradient here, got {gradient:?}"
    );
    Ok(())
}

// =============================================================================
// hinge_embedding_loss
// =============================================================================

#[test]
fn sm_hinge_embedding_loss_forward_values_are_pinned() -> TorshResult<()> {
    let _serial = serialize();
    let input = leaf(vec![0.3, 1.4, -0.7, 2.0], &[4]);
    let target = tensor(vec![1.0, -1.0, 1.0, -1.0], &[4]);

    let none = hinge_embedding_loss(&input, &target, 1.0, ReductionType::None)?;
    assert_eq!(none.shape().dims(), &[4]);
    // The `target == 1` branch is the *unclamped* input, so a negative entry is
    // correct and a "fix" that clamps it would be a regression.
    assert_values(&none, &[0.3, 0.0, -0.7, 0.0], "hinge none");

    let mean = hinge_embedding_loss(&input, &target, 1.0, ReductionType::Mean)?;
    assert_values(&mean, &[-0.099999994], "hinge mean (legitimately negative)");

    let sum = hinge_embedding_loss(&input, &target, 1.0, ReductionType::Sum)?;
    assert_values(&sum, &[-0.39999998], "hinge sum");
    Ok(())
}

#[test]
fn sm_hinge_embedding_loss_keeps_its_elementwise_shape() -> TorshResult<()> {
    let _serial = serialize();
    let input = leaf(vec![0.3, 1.4, -0.7, 2.0], &[2, 2]);
    let target = tensor(vec![1.0, -1.0, 1.0, -1.0], &[2, 2]);

    let loss = hinge_embedding_loss(&input, &target, 1.0, ReductionType::None)?;
    assert_eq!(
        loss.shape().dims(),
        &[2, 2],
        "hinge_embedding_loss is element-wise: 'none' keeps the input's shape"
    );
    assert_values(&loss, &[0.3, 0.0, -0.7, 0.0], "hinge 2-D");
    Ok(())
}

#[test]
fn sm_hinge_embedding_loss_stays_on_the_autograd_graph() -> TorshResult<()> {
    let _serial = serialize();
    let input = leaf(vec![0.3, 1.4, -0.7, 2.0], &[4]);
    let target = tensor(vec![1.0, -1.0, 1.0, -1.0], &[4]);

    let loss = hinge_embedding_loss(&input, &target, 2.5, ReductionType::Sum)?;
    assert!(
        loss.requires_grad(),
        "hinge_embedding_loss must stay attached to its input"
    );
    loss.backward()?;
    let gradient = grad_of(&input, "hinge");
    // margin 2.5 keeps both `target == -1` samples strictly inside the hinge,
    // so every element carries a real (unit-magnitude) subgradient.
    assert_grad_close(
        &gradient,
        &[1.0, -1.0, 1.0, -1.0],
        "hinge analytic subgradient",
    );
    Ok(())
}

#[test]
fn sm_hinge_embedding_loss_gradient_matches_finite_differences() -> TorshResult<()> {
    let _serial = serialize();
    let base = vec![0.3, 1.4, -0.7, 2.0];
    let target = tensor(vec![1.0, -1.0, 1.0, -1.0], &[4]);

    let pinned_input = leaf(base.clone(), &[4]);
    let pinned = hinge_embedding_loss(&pinned_input, &target, 2.5, ReductionType::None)?;
    assert_values(&pinned, &[0.3, 1.1, -0.7, 0.5], "hinge margin-2.5 fixture");

    let input = leaf(base.clone(), &[4]);
    let loss = hinge_embedding_loss(&input, &target, 2.5, ReductionType::Mean)?;
    loss.backward()?;
    let analytic = grad_of(&input, "hinge fd");

    let numeric = fd_grad(&base, &[4], |candidate| {
        hinge_embedding_loss(candidate, &target, 2.5, ReductionType::Mean)
            .and_then(|l| l.item())
            .expect("forward should succeed")
    });
    assert_grad_close(&analytic, &numeric, "hinge_embedding_loss d/dinput");
    Ok(())
}

/// A clamped-out sample must contribute *zero*, not a leaked subgradient from
/// the other branch -- the failure mode a naive `mask * a + mask * b` would have.
#[test]
fn sm_hinge_embedding_loss_zeroes_a_clamped_negative_sample() -> TorshResult<()> {
    let _serial = serialize();
    let input = leaf(vec![0.3, 5.0], &[2]);
    let target = tensor(vec![1.0, -1.0], &[2]);

    let loss = hinge_embedding_loss(&input, &target, 1.0, ReductionType::Sum)?;
    loss.backward()?;
    let gradient = grad_of(&input, "hinge clamped");
    assert!(
        (gradient[0] - 1.0).abs() <= 1e-5,
        "the positive branch passes its gradient through, got {gradient:?}"
    );
    assert!(
        gradient[1] == 0.0,
        "a sample clamped to zero must receive no gradient, got {gradient:?}"
    );
    Ok(())
}

// =============================================================================
// contrastive_loss
// =============================================================================

#[test]
fn sm_contrastive_loss_forward_values_are_pinned() -> TorshResult<()> {
    let _serial = serialize();
    let (input1, input2, target) = contrastive_fixture();

    let none = contrastive_loss(&input1, &input2, &target, 1.0, ReductionType::None)?;
    assert_eq!(none.shape().dims(), &[2]);
    assert_values(&none, &[0.04000002, 0.042893223], "contrastive none");

    let mean = contrastive_loss(&input1, &input2, &target, 1.0, ReductionType::Mean)?;
    assert_values(&mean, &[0.041446622], "contrastive mean");

    let sum = contrastive_loss(&input1, &input2, &target, 1.0, ReductionType::Sum)?;
    assert_values(&sum, &[0.082893245], "contrastive sum");
    Ok(())
}

#[test]
fn sm_contrastive_loss_rejects_a_target_of_the_wrong_shape() {
    let _serial = serialize();
    let input1 = leaf(vec![1.0, 2.0, 1.0, 2.0], &[2, 2]);
    let input2 = leaf(vec![1.0, 2.0, 1.0, 2.0], &[2, 2]);
    let target = tensor(vec![0.0, 1.0, 0.0], &[3]);

    let error = contrastive_loss(&input1, &input2, &target, 1.0, ReductionType::None)
        .expect_err("one label per pair is the documented contract");
    let message = format!("{error}");
    assert!(
        message.contains("expected [2]") && message.contains("got [3]"),
        "the pre-existing diagnostic named both shapes; got: {message}"
    );
}

#[test]
fn sm_contrastive_loss_stays_on_the_autograd_graph() -> TorshResult<()> {
    let _serial = serialize();
    let (input1, input2, target) = contrastive_fixture();

    let loss = contrastive_loss(&input1, &input2, &target, 1.0, ReductionType::Mean)?;
    assert!(
        loss.requires_grad(),
        "contrastive_loss must stay attached to its embeddings"
    );
    loss.backward()?;
    let g1 = grad_of(&input1, "contrastive input1");
    let g2 = grad_of(&input2, "contrastive input2");
    assert!(
        g1.iter().any(|v| *v != 0.0) && g1.iter().all(|v| v.is_finite()),
        "input1 gradient must be finite and non-trivial, got {g1:?}"
    );
    assert!(
        g2.iter().any(|v| *v != 0.0) && g2.iter().all(|v| v.is_finite()),
        "input2 gradient must be finite and non-trivial, got {g2:?}"
    );
    Ok(())
}

#[test]
fn sm_contrastive_loss_gradient_matches_finite_differences() -> TorshResult<()> {
    let _serial = serialize();
    let base1 = vec![1.0, 2.0, 0.0, 1.0];
    let base2 = vec![1.2, 1.8, 0.5, 1.5];
    let target = tensor(vec![0.0, 1.0], &[2]);

    let input1 = leaf(base1.clone(), &[2, 2]);
    let input2 = tensor(base2.clone(), &[2, 2]);
    let loss = contrastive_loss(&input1, &input2, &target, 1.0, ReductionType::Sum)?;
    loss.backward()?;
    let analytic = grad_of(&input1, "contrastive fd");

    let fixed = tensor(base2, &[2, 2]);
    let numeric = fd_grad(&base1, &[2, 2], |candidate| {
        contrastive_loss(candidate, &fixed, &target, 1.0, ReductionType::Sum)
            .and_then(|l| l.item())
            .expect("forward should succeed")
    });
    assert_grad_close(&analytic, &numeric, "contrastive_loss d/dinput1");
    Ok(())
}

/// Two identical embeddings are an *ordinary* input for a metric-learning loss,
/// and they put the Euclidean distance's `sqrt` exactly at zero, where its
/// derivative is infinite. Once the branches are multiplied by a constant mask,
/// the masked-*out* dissimilar branch still evaluates that `sqrt`, so `0 * inf`
/// would poison the whole batch with `NaN`.
#[test]
fn sm_contrastive_loss_survives_identical_embeddings() -> TorshResult<()> {
    let _serial = serialize();
    let input1 = leaf(vec![1.0, 2.0, 1.0, 2.0], &[2, 2]);
    let input2 = leaf(vec![1.0, 2.0, 1.0, 2.0], &[2, 2]);
    // Row 0 is a similar pair, row 1 a dissimilar one: both branches see d = 0.
    let target = tensor(vec![0.0, 1.0], &[2]);

    let loss = contrastive_loss(&input1, &input2, &target, 1.0, ReductionType::None)?;
    assert_values(&loss, &[0.0, 0.5], "contrastive at zero distance");

    scalarize(&loss)?.backward()?;
    let g1 = grad_of(&input1, "contrastive degenerate input1");
    let g2 = grad_of(&input2, "contrastive degenerate input2");
    assert!(
        g1.iter().all(|v| v.is_finite()),
        "identical embeddings must not back-propagate NaN/inf, got {g1:?}"
    );
    assert!(
        g2.iter().all(|v| v.is_finite()),
        "identical embeddings must not back-propagate NaN/inf, got {g2:?}"
    );
    Ok(())
}

// =============================================================================
// margin_ranking_loss -- already attached before W6-F; guard against regression
// =============================================================================

#[test]
fn sm_margin_ranking_loss_forward_values_are_pinned() -> TorshResult<()> {
    let _serial = serialize();
    let input1 = leaf(vec![2.0, 3.0, 0.5], &[3]);
    let input2 = leaf(vec![1.0, 1.5, 2.0], &[3]);
    let target = tensor(vec![1.0, -1.0, 1.0], &[3]);

    let none = margin_ranking_loss(&input1, &input2, &target, 0.5, ReductionType::None)?;
    assert_eq!(none.shape().dims(), &[3]);
    assert_values(&none, &[0.0, 2.0, 2.0], "margin ranking none");
    assert_values(
        &margin_ranking_loss(&input1, &input2, &target, 0.5, ReductionType::Mean)?,
        &[1.3333334],
        "margin ranking mean",
    );
    assert_values(
        &margin_ranking_loss(&input1, &input2, &target, 0.5, ReductionType::Sum)?,
        &[4.0],
        "margin ranking sum",
    );
    assert_values(
        &margin_ranking_loss(&input1, &input2, &target, 2.0, ReductionType::None)?,
        &[1.0, 3.5, 3.5],
        "margin ranking margin=2",
    );
    Ok(())
}

#[test]
fn sm_margin_ranking_loss_gradient_matches_finite_differences() -> TorshResult<()> {
    let _serial = serialize();
    let base1 = vec![2.0, 3.0, 0.5];
    let base2 = vec![1.0, 1.5, 2.0];
    let target = tensor(vec![1.0, -1.0, 1.0], &[3]);

    let input1 = leaf(base1.clone(), &[3]);
    let input2 = tensor(base2.clone(), &[3]);
    // margin 2.0 puts every sample strictly inside the hinge.
    let loss = margin_ranking_loss(&input1, &input2, &target, 2.0, ReductionType::Sum)?;
    assert!(
        loss.requires_grad(),
        "margin_ranking_loss must stay attached"
    );
    loss.backward()?;
    let analytic = grad_of(&input1, "margin ranking");

    let fixed = tensor(base2, &[3]);
    let numeric = fd_grad(&base1, &[3], |candidate| {
        margin_ranking_loss(candidate, &fixed, &target, 2.0, ReductionType::Sum)
            .and_then(|l| l.item())
            .expect("forward should succeed")
    });
    assert_grad_close(&analytic, &numeric, "margin_ranking_loss d/dinput1");
    Ok(())
}

// =============================================================================
// triplet_margin_loss
// =============================================================================

#[test]
fn sm_triplet_margin_loss_forward_values_are_pinned() -> TorshResult<()> {
    let _serial = serialize();
    let (anchor, positive, negative) = triplet_fixture();

    let none = triplet_margin_loss(
        &anchor,
        &positive,
        &negative,
        3.0,
        2.0,
        1e-6,
        false,
        ReductionType::None,
    )?;
    assert_eq!(none.shape().dims(), &[2]);
    assert_values(&none, &[0.7828429, 0.87867975], "triplet p=2 none");

    assert_values(
        &triplet_margin_loss(
            &anchor,
            &positive,
            &negative,
            3.0,
            2.0,
            1e-6,
            false,
            ReductionType::Mean,
        )?,
        &[0.8307613],
        "triplet mean",
    );
    assert_values(
        &triplet_margin_loss(
            &anchor,
            &positive,
            &negative,
            3.0,
            2.0,
            1e-6,
            false,
            ReductionType::Sum,
        )?,
        &[1.6615226],
        "triplet sum",
    );
    // Manhattan (p = 1) and a general L_p both keep their predecessor values.
    assert_values(
        &triplet_margin_loss(
            &anchor,
            &positive,
            &negative,
            3.0,
            1.0,
            1e-6,
            false,
            ReductionType::None,
        )?,
        &[0.0, 0.0],
        "triplet p=1",
    );
    assert_values(
        &triplet_margin_loss(
            &anchor,
            &positive,
            &negative,
            3.0,
            1.5,
            1e-6,
            false,
            ReductionType::None,
        )?,
        &[0.5253551, 0.6188984],
        "triplet p=1.5",
    );
    // `swap` picks min(d(a,n), d(p,n)); on this fixture d(a,n) already wins.
    assert_values(
        &triplet_margin_loss(
            &anchor,
            &positive,
            &negative,
            3.0,
            2.0,
            1e-6,
            true,
            ReductionType::None,
        )?,
        &[0.7828429, 0.87867975],
        "triplet swap",
    );
    Ok(())
}

#[test]
fn sm_triplet_margin_loss_gradient_matches_finite_differences() -> TorshResult<()> {
    let _serial = serialize();
    let base = vec![1.0, 2.0, 0.0, 1.0];
    let positive = tensor(vec![1.2, 1.8, 0.5, 1.5], &[2, 2]);
    let negative = tensor(vec![-1.0, 0.5, 2.0, -1.0], &[2, 2]);

    let anchor = leaf(base.clone(), &[2, 2]);
    let loss = triplet_margin_loss(
        &anchor,
        &positive,
        &negative,
        3.0,
        2.0,
        1e-6,
        false,
        ReductionType::Sum,
    )?;
    assert!(
        loss.requires_grad(),
        "triplet_margin_loss must stay attached"
    );
    loss.backward()?;
    let analytic = grad_of(&anchor, "triplet");

    let numeric = fd_grad(&base, &[2, 2], |candidate| {
        triplet_margin_loss(
            candidate,
            &positive,
            &negative,
            3.0,
            2.0,
            1e-6,
            false,
            ReductionType::Sum,
        )
        .and_then(|l| l.item())
        .expect("forward should succeed")
    });
    assert_grad_close(&analytic, &numeric, "triplet_margin_loss d/danchor");
    Ok(())
}

/// An anchor that coincides with its positive is the *target state* of triplet
/// training, and it lands the Euclidean distance's `sqrt` on zero, where the
/// derivative is infinite. Before W6-F this produced `a.grad = [NaN, NaN]`.
#[test]
fn sm_triplet_margin_loss_survives_a_zero_distance() -> TorshResult<()> {
    let _serial = serialize();
    for p in [1.0_f32, 1.5, 2.0] {
        let anchor = leaf(vec![1.0, 2.0], &[1, 2]);
        let positive = leaf(vec![1.0, 2.0], &[1, 2]);
        let negative = leaf(vec![5.0, 6.0], &[1, 2]);

        let loss = triplet_margin_loss(
            &anchor,
            &positive,
            &negative,
            10.0,
            p,
            1e-6,
            false,
            ReductionType::None,
        )?;
        scalarize(&loss)?.backward()?;

        let anchor_grad = grad_of(&anchor, "triplet degenerate anchor");
        let positive_grad = grad_of(&positive, "triplet degenerate positive");
        assert!(
            anchor_grad.iter().all(|v| v.is_finite()),
            "p={p}: a coincident anchor/positive must not back-propagate NaN, got {anchor_grad:?}"
        );
        assert!(
            positive_grad.iter().all(|v| v.is_finite()),
            "p={p}: a coincident anchor/positive must not back-propagate NaN, got {positive_grad:?}"
        );
    }
    Ok(())
}

#[test]
fn sm_triplet_margin_loss_zero_distance_forward_is_pinned() -> TorshResult<()> {
    let _serial = serialize();
    let anchor = leaf(vec![1.0, 2.0], &[1, 2]);
    let positive = leaf(vec![1.0, 2.0], &[1, 2]);
    let negative = leaf(vec![5.0, 6.0], &[1, 2]);

    assert_values(
        &triplet_margin_loss(
            &anchor,
            &positive,
            &negative,
            10.0,
            2.0,
            1e-6,
            false,
            ReductionType::None,
        )?,
        &[4.343146],
        "triplet zero-distance p=2",
    );
    assert_values(
        &triplet_margin_loss(
            &anchor,
            &positive,
            &negative,
            10.0,
            1.0,
            1e-6,
            false,
            ReductionType::None,
        )?,
        &[2.0],
        "triplet zero-distance p=1",
    );
    Ok(())
}

#[test]
fn sm_triplet_margin_with_distance_loss_stays_on_the_autograd_graph() -> TorshResult<()> {
    let _serial = serialize();
    let (anchor, positive, negative) = triplet_fixture();
    let squared = |x: &Tensor, y: &Tensor| x.sub(y)?.pow_scalar(2.0)?.sum_dim(&[1], false);

    // margin 8.0 puts both triplets strictly inside the hinge.
    let loss = triplet_margin_with_distance_loss(
        &anchor,
        &positive,
        &negative,
        squared,
        8.0,
        false,
        ReductionType::None,
    )?;
    // The only W6-F edit to this function is `clamp(0.0, f32::MAX)` ->
    // `clamp_min(0.0)`, which cannot move a finite forward value, so this pin is
    // derived analytically: with a squared-distance closure the two triplets
    // score max(0, 0.08 - 6.25 + 8) and max(0, 0.5 - 8.0 + 8).
    assert_values(&loss, &[1.83, 0.5], "triplet with distance forward");

    let loss = triplet_margin_with_distance_loss(
        &anchor,
        &positive,
        &negative,
        squared,
        8.0,
        false,
        ReductionType::Sum,
    )?;
    assert!(
        loss.requires_grad(),
        "triplet_margin_with_distance_loss must stay attached"
    );
    loss.backward()?;
    let gradient = grad_of(&anchor, "triplet with distance");
    assert!(
        gradient.iter().any(|v| *v != 0.0) && gradient.iter().all(|v| v.is_finite()),
        "anchor gradient must be finite and non-trivial, got {gradient:?}"
    );
    Ok(())
}

// =============================================================================
// smooth_l1_loss (regression.rs -- same `where_tensor` pattern, probe-confirmed)
// =============================================================================

#[test]
fn sm_smooth_l1_loss_forward_values_are_pinned() -> TorshResult<()> {
    let _serial = serialize();
    let input = leaf(vec![1.0, 2.0, 3.0], &[3]);
    let target = tensor(vec![1.1, 2.1, 4.0], &[3]);

    let none = smooth_l1_loss(&input, &target, ReductionType::None, 1.0)?;
    assert_eq!(none.shape().dims(), &[3]);
    assert_values(&none, &[0.005000002, 0.0049999906, 0.5], "smooth_l1 none");
    assert_values(
        &smooth_l1_loss(&input, &target, ReductionType::Mean, 1.0)?,
        &[0.17],
        "smooth_l1 mean",
    );
    assert_values(
        &smooth_l1_loss(&input, &target, ReductionType::Sum, 1.0)?,
        &[0.51],
        "smooth_l1 sum",
    );

    // A second beta, and a target that is unambiguously in the L1 arm.
    let far = tensor(vec![1.1, 2.1, 4.5], &[3]);
    assert_values(
        &smooth_l1_loss(&input, &far, ReductionType::None, 1.0)?,
        &[0.005000002, 0.0049999906, 1.0],
        "smooth_l1 far beta=1",
    );
    assert_values(
        &smooth_l1_loss(&input, &far, ReductionType::None, 0.5)?,
        &[0.010000004, 0.009999981, 1.25],
        "smooth_l1 far beta=0.5",
    );
    Ok(())
}

#[test]
fn sm_smooth_l1_loss_stays_on_the_autograd_graph() -> TorshResult<()> {
    let _serial = serialize();
    let input = leaf(vec![1.0, 2.0, 3.0], &[3]);
    let target = tensor(vec![1.1, 2.1, 4.5], &[3]);

    let loss = smooth_l1_loss(&input, &target, ReductionType::Mean, 1.0)?;
    assert!(
        loss.requires_grad(),
        "smooth_l1_loss must stay attached to its input"
    );
    loss.backward()?;
    let gradient = grad_of(&input, "smooth_l1");
    assert!(
        gradient.iter().any(|v| *v != 0.0) && gradient.iter().all(|v| v.is_finite()),
        "gradient must be finite and non-trivial, got {gradient:?}"
    );
    Ok(())
}

#[test]
fn sm_smooth_l1_loss_gradient_matches_finite_differences() -> TorshResult<()> {
    let _serial = serialize();
    let base = vec![1.0, 2.0, 3.0];
    let target = tensor(vec![1.1, 2.1, 4.5], &[3]);

    let input = leaf(base.clone(), &[3]);
    let loss = smooth_l1_loss(&input, &target, ReductionType::Sum, 1.0)?;
    loss.backward()?;
    let analytic = grad_of(&input, "smooth_l1 fd");

    let numeric = fd_grad(&base, &[3], |candidate| {
        smooth_l1_loss(candidate, &target, ReductionType::Sum, 1.0)
            .and_then(|l| l.item())
            .expect("forward should succeed")
    });
    assert_grad_close(&analytic, &numeric, "smooth_l1_loss d/dinput");

    // Both arms must be exercised: element 0 sits in the quadratic arm
    // (d/dx = diff / beta) and element 2 in the linear arm (d/dx = sign(diff)).
    assert!(
        (analytic[0] - (-0.100000024_f32)).abs() <= 1e-4,
        "quadratic arm gradient, got {analytic:?}"
    );
    assert!(
        (analytic[2] + 1.0).abs() <= 1e-4,
        "linear arm gradient, got {analytic:?}"
    );
    Ok(())
}

// =============================================================================
// specialized.rs -- severed by other mechanisms, found by the same sweep
// =============================================================================

#[test]
fn sm_gradient_penalty_loss_forward_values_are_pinned() -> TorshResult<()> {
    let _serial = serialize();
    let gradients = leaf(vec![1.5, 0.8, 1.2], &[3]);
    for reduction in [ReductionType::None, ReductionType::Mean, ReductionType::Sum] {
        let loss = gradient_penalty_loss(&gradients, 10.0, reduction)?;
        assert_eq!(
            loss.shape().dims(),
            &[] as &[usize],
            "the penalty is already a scalar, so every reduction is a no-op"
        );
        assert_values(&loss, &[11.682695], "gradient penalty");
    }
    let matrix = leaf(vec![1.5, 0.8, 1.2, -0.4], &[2, 2]);
    assert_values(
        &gradient_penalty_loss(&matrix, 10.0, ReductionType::Mean)?,
        &[12.520761],
        "gradient penalty 2-D",
    );
    Ok(())
}

#[test]
fn sm_gradient_penalty_loss_stays_on_the_autograd_graph() -> TorshResult<()> {
    let _serial = serialize();
    let base = vec![1.5, 0.8, 1.2];
    let gradients = leaf(base.clone(), &[3]);

    let loss = gradient_penalty_loss(&gradients, 10.0, ReductionType::Mean)?;
    assert!(
        loss.requires_grad(),
        "gradient_penalty_loss must stay attached to the gradients it penalises"
    );
    loss.backward()?;
    let analytic = grad_of(&gradients, "gradient penalty");

    let numeric = fd_grad(&base, &[3], |candidate| {
        gradient_penalty_loss(candidate, 10.0, ReductionType::Mean)
            .and_then(|l| l.item())
            .expect("forward should succeed")
    });
    assert_grad_close(&analytic, &numeric, "gradient_penalty_loss d/dgradients");
    Ok(())
}

#[test]
fn sm_temporal_consistency_loss_forward_values_are_pinned() -> TorshResult<()> {
    let _serial = serialize();
    let predictions = leaf(
        vec![1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 1.2, 2.2, 3.2],
        &[1, 3, 3],
    );
    for reduction in [ReductionType::None, ReductionType::Mean, ReductionType::Sum] {
        let loss = temporal_consistency_loss(&predictions, 1.0, reduction)?;
        assert_eq!(loss.shape().dims(), &[] as &[usize]);
        assert_values(&loss, &[0.010000004], "temporal consistency");
    }

    let varied = leaf(
        vec![1.0, 2.0, 3.0, 1.5, 2.4, 3.6, 0.5, 2.9, 2.2, 1.1, 1.4, 3.0],
        &[2, 3, 2],
    );
    assert_values(
        &temporal_consistency_loss(&varied, 0.5, ReductionType::Mean)?,
        &[1.2125],
        "temporal consistency varied",
    );
    Ok(())
}

#[test]
fn sm_temporal_consistency_loss_stays_on_the_autograd_graph() -> TorshResult<()> {
    let _serial = serialize();
    let base = vec![1.0, 2.0, 3.0, 1.5, 2.4, 3.6, 0.5, 2.9, 2.2, 1.1, 1.4, 3.0];
    let predictions = leaf(base.clone(), &[2, 3, 2]);

    let loss = temporal_consistency_loss(&predictions, 0.5, ReductionType::Mean)?;
    assert!(
        loss.requires_grad(),
        "temporal_consistency_loss must stay attached to its predictions"
    );
    loss.backward()?;
    let analytic = grad_of(&predictions, "temporal consistency");

    let numeric = fd_grad(&base, &[2, 3, 2], |candidate| {
        temporal_consistency_loss(candidate, 0.5, ReductionType::Mean)
            .and_then(|l| l.item())
            .expect("forward should succeed")
    });
    assert_grad_close(
        &analytic,
        &numeric,
        "temporal_consistency_loss d/dpredictions",
    );
    Ok(())
}

#[test]
fn sm_seq2seq_loss_forward_values_are_pinned() -> TorshResult<()> {
    let _serial = serialize();
    let predictions = leaf(vec![0.1_f32; 2 * 5 * 10], &[2, 5, 10]);
    let targets = tensor(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0, 3.0, 4.0],
        &[2, 5],
    );

    let none = seq2seq_loss_with_attention(&predictions, &targets, None, 0.0, ReductionType::None)?;
    assert_eq!(
        none.shape().dims(),
        &[1],
        "the predecessor returned a [1] tensor under 'none'"
    );
    assert_values(&none, &[2.3025854], "seq2seq none");
    assert_eq!(
        seq2seq_loss_with_attention(&predictions, &targets, None, 0.0, ReductionType::Mean)?
            .shape()
            .dims(),
        &[] as &[usize]
    );
    assert_values(
        &seq2seq_loss_with_attention(&predictions, &targets, None, 0.0, ReductionType::Mean)?,
        &[2.3025854],
        "seq2seq mean",
    );
    assert_values(
        &seq2seq_loss_with_attention(&predictions, &targets, None, 0.0, ReductionType::Sum)?,
        &[2.3025854],
        "seq2seq sum",
    );

    let attention = leaf(vec![0.25_f32; 2 * 5 * 4], &[2, 5, 4]);
    assert_values(
        &seq2seq_loss_with_attention(
            &predictions,
            &targets,
            Some(&attention),
            0.5,
            ReductionType::Mean,
        )?,
        &[2.3338354],
        "seq2seq with attention regularisation",
    );
    Ok(())
}

#[test]
fn sm_seq2seq_loss_stays_on_the_autograd_graph() -> TorshResult<()> {
    let _serial = serialize();
    let base = vec![
        0.5, -0.2, 1.3, 0.1, -1.0, 0.7, 0.4, 0.9, -0.3, 1.1, 0.2, -0.6,
    ];
    let targets = tensor(vec![0.0, 2.0, 1.0, 0.0], &[2, 2]);

    let pinned = leaf(base.clone(), &[2, 2, 3]);
    assert_values(
        &seq2seq_loss_with_attention(&pinned, &targets, None, 0.0, ReductionType::Mean)?,
        &[0.74311393],
        "seq2seq fd fixture",
    );

    let predictions = leaf(base.clone(), &[2, 2, 3]);
    let loss = seq2seq_loss_with_attention(&predictions, &targets, None, 0.0, ReductionType::Mean)?;
    assert!(
        loss.requires_grad(),
        "seq2seq_loss_with_attention must stay attached to its logits"
    );
    loss.backward()?;
    let analytic = grad_of(&predictions, "seq2seq");

    let numeric = fd_grad(&base, &[2, 2, 3], |candidate| {
        seq2seq_loss_with_attention(candidate, &targets, None, 0.0, ReductionType::Mean)
            .and_then(|l| l.item())
            .expect("forward should succeed")
    });
    assert_grad_close(
        &analytic,
        &numeric,
        "seq2seq_loss_with_attention d/dpredictions",
    );
    Ok(())
}

#[test]
fn sm_seq2seq_attention_regularisation_reaches_the_attention_weights() -> TorshResult<()> {
    let _serial = serialize();
    let predictions = leaf(
        vec![
            0.5, -0.2, 1.3, 0.1, -1.0, 0.7, 0.4, 0.9, -0.3, 1.1, 0.2, -0.6,
        ],
        &[2, 2, 3],
    );
    let targets = tensor(vec![0.0, 2.0, 1.0, 0.0], &[2, 2]);
    let attention_base = vec![0.6, 0.4, 0.3, 0.7, 0.2, 0.8, 0.5, 0.5];
    let attention = leaf(attention_base.clone(), &[2, 2, 2]);

    let loss = seq2seq_loss_with_attention(
        &predictions,
        &targets,
        Some(&attention),
        0.25,
        ReductionType::Mean,
    )?;
    assert_values(&loss, &[0.81436396], "seq2seq attention fixture");
    loss.backward()?;
    let gradient = grad_of(&attention, "seq2seq attention");

    // d/dA [reg * mean(A^2)] = reg * 2A / n, with reg = 0.25 and n = 8.
    let expected: Vec<f32> = attention_base
        .iter()
        .map(|a| 0.25 * 2.0 * a / 8.0)
        .collect();
    assert_grad_close(&gradient, &expected, "attention regularisation gradient");
    Ok(())
}

// =============================================================================
// Cross-cutting: every reduction of every rewritten loss keeps the graph
// =============================================================================

#[test]
fn sm_every_reduction_of_every_rewritten_loss_keeps_the_graph() -> TorshResult<()> {
    let _serial = serialize();
    for reduction in [ReductionType::None, ReductionType::Mean, ReductionType::Sum] {
        let (c1, c2, ct) = cosine_fixture();
        assert!(
            cosine_embedding_loss(&c1, &c2, &ct, 0.5, reduction)?.requires_grad(),
            "cosine_embedding_loss lost the graph under {reduction:?}"
        );

        let hinge_input = leaf(vec![0.3, 1.4, -0.7, 2.0], &[4]);
        let hinge_target = tensor(vec![1.0, -1.0, 1.0, -1.0], &[4]);
        assert!(
            hinge_embedding_loss(&hinge_input, &hinge_target, 1.0, reduction)?.requires_grad(),
            "hinge_embedding_loss lost the graph under {reduction:?}"
        );

        let (k1, k2, kt) = contrastive_fixture();
        assert!(
            contrastive_loss(&k1, &k2, &kt, 1.0, reduction)?.requires_grad(),
            "contrastive_loss lost the graph under {reduction:?}"
        );

        let (anchor, positive, negative) = triplet_fixture();
        assert!(
            triplet_margin_loss(&anchor, &positive, &negative, 3.0, 2.0, 1e-6, false, reduction)?
                .requires_grad(),
            "triplet_margin_loss lost the graph under {reduction:?}"
        );

        let m1 = leaf(vec![2.0, 3.0, 0.5], &[3]);
        let m2 = leaf(vec![1.0, 1.5, 2.0], &[3]);
        let mt = tensor(vec![1.0, -1.0, 1.0], &[3]);
        assert!(
            margin_ranking_loss(&m1, &m2, &mt, 2.0, reduction)?.requires_grad(),
            "margin_ranking_loss lost the graph under {reduction:?}"
        );

        let s_input = leaf(vec![1.0, 2.0, 3.0], &[3]);
        let s_target = tensor(vec![1.1, 2.1, 4.5], &[3]);
        assert!(
            smooth_l1_loss(&s_input, &s_target, reduction, 1.0)?.requires_grad(),
            "smooth_l1_loss lost the graph under {reduction:?}"
        );

        let penalty_input = leaf(vec![1.5, 0.8, 1.2], &[3]);
        assert!(
            gradient_penalty_loss(&penalty_input, 10.0, reduction)?.requires_grad(),
            "gradient_penalty_loss lost the graph under {reduction:?}"
        );

        let sequence = leaf(
            vec![1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 1.2, 2.2, 3.2],
            &[1, 3, 3],
        );
        assert!(
            temporal_consistency_loss(&sequence, 1.0, reduction)?.requires_grad(),
            "temporal_consistency_loss lost the graph under {reduction:?}"
        );

        let logits = leaf(
            vec![
                0.5, -0.2, 1.3, 0.1, -1.0, 0.7, 0.4, 0.9, -0.3, 1.1, 0.2, -0.6,
            ],
            &[2, 2, 3],
        );
        let sequence_targets = tensor(vec![0.0, 2.0, 1.0, 0.0], &[2, 2]);
        assert!(
            seq2seq_loss_with_attention(&logits, &sequence_targets, None, 0.0, reduction)?
                .requires_grad(),
            "seq2seq_loss_with_attention lost the graph under {reduction:?}"
        );
    }
    Ok(())
}
