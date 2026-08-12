//! Production-hardening regression tests for `torsh-optim`.
//!
//! Every test here pins down a behaviour that was previously broken; each one is
//! named after the finding it guards.

use parking_lot::RwLock;
use std::sync::Arc;
use torsh_core::device::DeviceType;
use torsh_optim::Optimizer;
use torsh_tensor::Tensor;

/// Build a parameter handle holding `data` with gradient tracking enabled.
fn param(data: Vec<f32>) -> Arc<RwLock<Tensor>> {
    let len = data.len();
    let tensor = Tensor::from_data(data, vec![len], DeviceType::Cpu)
        .expect("parameter creation must succeed")
        .requires_grad_(true);
    Arc::new(RwLock::new(tensor))
}

fn grad(data: Vec<f32>) -> Tensor {
    let len = data.len();
    Tensor::from_data(data, vec![len], DeviceType::Cpu).expect("gradient creation must succeed")
}

fn values(p: &Arc<RwLock<Tensor>>) -> Vec<f32> {
    p.read().to_vec().expect("to_vec must succeed")
}

// ---------------------------------------------------------------------------
// F040 — optimizers must update parameters in place, never rebind `*param`
// ---------------------------------------------------------------------------

#[test]
fn f040_sgd_two_steps_move_parameter_twice() {
    let p = param(vec![1.0, 2.0, 3.0]);
    p.read().set_grad(Some(grad(vec![1.0, 1.0, 1.0])));

    let mut opt = torsh_optim::sgd::SGD::new(vec![Arc::clone(&p)], 0.1, None, None, None, false);
    opt.step().expect("first step");
    opt.step().expect("second step");

    // Two steps with the same gradient must move the parameter by 2 * lr * g.
    let got = values(&p);
    let expected = [0.8f32, 1.8, 2.8];
    for (g, e) in got.iter().zip(expected.iter()) {
        assert!((g - e).abs() < 1e-5, "got {got:?}, expected {expected:?}");
    }
}

#[test]
fn f040_step_preserves_gradient_and_requires_grad() {
    let p = param(vec![1.0, 2.0]);
    p.read().set_grad(Some(grad(vec![0.5, 0.5])));

    let mut opt = torsh_optim::sgd::SGD::new(vec![Arc::clone(&p)], 0.1, None, None, None, false);
    opt.step().expect("step");

    let guard = p.read();
    assert!(
        guard.requires_grad(),
        "optimizer step must not clear requires_grad"
    );
    assert!(
        guard.has_grad(),
        "optimizer step must not discard the parameter's gradient"
    );
}

#[test]
fn f040_inplace_update_survives_shared_storage() {
    let p = param(vec![1.0, 2.0, 3.0]);
    // Keep both a graph node and a direct storage-sharing snapshot alive across
    // the update: a real training loop keeps the loss around, and optimizer state
    // (Lookahead slow weights, continual-learning anchors) keeps clones around.
    // In-place updates are copy-on-write, so the snapshot must keep the old
    // values while the parameter takes the new ones.
    let live_node = p.read().mul_scalar(2.0).expect("forward");
    let snapshot = p.read().clone();
    p.read().set_grad(Some(grad(vec![1.0, 1.0, 1.0])));

    let mut opt = torsh_optim::sgd::SGD::new(vec![Arc::clone(&p)], 0.1, None, None, None, false);
    opt.step().expect("step");
    opt.step().expect("step");

    let got = values(&p);
    let expected = [0.8f32, 1.8, 2.8];
    for (g, e) in got.iter().zip(expected.iter()) {
        assert!((g - e).abs() < 1e-5, "got {got:?}, expected {expected:?}");
    }
    assert!(p.read().has_grad(), "gradient must survive the update");
    assert_eq!(
        snapshot.to_vec().expect("to_vec"),
        vec![1.0, 2.0, 3.0],
        "a snapshot taken before the step must not observe the update"
    );
    drop(live_node);
}

// ---------------------------------------------------------------------------
// F135 — AMSGrad must bias-correct the second moment before the max
// ---------------------------------------------------------------------------

#[test]
fn f135_amsgrad_first_step_matches_plain_adam() {
    let make = |amsgrad: bool| {
        let p = param(vec![0.0, 0.0]);
        p.read().set_grad(Some(grad(vec![1.0, 1.0])));
        let mut opt = torsh_optim::adam::Adam::new(
            vec![Arc::clone(&p)],
            Some(0.1),
            None,
            None,
            None,
            amsgrad,
        );
        opt.step().expect("step");
        values(&p)
    };

    // At t = 1 the running max equals the (corrected) second moment, so both
    // variants must take the same step. Without the correction the AMSGrad step
    // is inflated by 1/sqrt(1 - beta2) ~= 31.6x.
    let plain = make(false);
    let amsgrad = make(true);
    for (a, b) in amsgrad.iter().zip(plain.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "amsgrad {amsgrad:?} must match plain adam {plain:?} at t=1"
        );
    }
}

// ---------------------------------------------------------------------------
// F136 — the step counter must not be a parameter-shaped tensor
// ---------------------------------------------------------------------------

#[test]
fn f136_step_counter_is_a_scalar() {
    let p = param(vec![0.0; 64]);
    p.read().set_grad(Some(grad(vec![1.0; 64])));

    for (name, mut opt) in [
        (
            "Adam",
            Box::new(torsh_optim::adam::Adam::new(
                vec![Arc::clone(&p)],
                Some(0.1),
                None,
                None,
                None,
                false,
            )) as Box<dyn Optimizer>,
        ),
        (
            "AdamW",
            Box::new(torsh_optim::adam::AdamW::new(
                vec![Arc::clone(&p)],
                Some(0.1),
                None,
                None,
                None,
                false,
            )) as Box<dyn Optimizer>,
        ),
    ] {
        opt.step().expect("step");
        opt.step().expect("step");
        let dict = opt.state_dict().expect("state_dict");
        let entry = dict
            .state
            .values()
            .next()
            .expect("per-parameter state must exist");
        let step = entry.get("step").expect("step state must exist");
        assert_eq!(
            step.numel(),
            1,
            "{name}: step counter must be a scalar, got {} elements",
            step.numel()
        );
        assert_eq!(
            step.to_vec().expect("to_vec")[0],
            2.0,
            "{name}: step counter must track the number of steps"
        );
    }
}

#[test]
fn f040_adam_two_steps_move_parameter_monotonically() {
    let p = param(vec![0.0, 0.0]);
    p.read().set_grad(Some(grad(vec![1.0, 1.0])));

    let mut opt =
        torsh_optim::adam::Adam::new(vec![Arc::clone(&p)], Some(0.1), None, None, None, false);
    opt.step().expect("first step");
    let after_first = values(&p);
    opt.step().expect("second step");
    let after_second = values(&p);

    assert!(
        after_first[0] < -1e-3,
        "first Adam step must move the parameter, got {after_first:?}"
    );
    assert!(
        after_second[0] < after_first[0] - 1e-3,
        "second Adam step must move the parameter further: {after_first:?} -> {after_second:?}"
    );
}

// ---------------------------------------------------------------------------
// F238 — schedulers must preserve per-parameter-group learning rates
// ---------------------------------------------------------------------------

#[test]
fn f238_scheduler_preserves_per_group_learning_rates() {
    use std::collections::HashMap;
    use torsh_optim::lr_scheduler::{ExponentialLR, LRScheduler};

    let backbone = param(vec![1.0]);
    let head = param(vec![1.0]);

    let mut sgd = torsh_optim::sgd::SGD::new(vec![backbone], 0.01, None, None, None, false);
    let mut options = HashMap::new();
    options.insert("lr".to_string(), 0.1);
    sgd.add_param_group(vec![head], options);
    assert_eq!(sgd.get_lr(), vec![0.01, 0.1]);

    let mut scheduler = ExponentialLR::new(sgd, 0.5);
    scheduler.step().expect("scheduler step");

    let lrs = scheduler.optimizer().get_lr();
    assert!(
        (lrs[0] - 0.005).abs() < 1e-8 && (lrs[1] - 0.05).abs() < 1e-8,
        "each group must be decayed from its own base lr, got {lrs:?}"
    );

    scheduler.reset();
    let lrs = scheduler.optimizer().get_lr();
    assert!(
        (lrs[0] - 0.01).abs() < 1e-8 && (lrs[1] - 0.1).abs() < 1e-8,
        "reset must restore each group's base lr, got {lrs:?}"
    );
}

// ---------------------------------------------------------------------------
// F236 / F305 — invalid configuration must be reportable without unwinding
// ---------------------------------------------------------------------------

#[test]
fn f236_sgd_try_new_rejects_invalid_configuration() {
    let p = param(vec![1.0]);

    // Nesterov without momentum.
    assert!(
        torsh_optim::sgd::SGD::try_new(vec![Arc::clone(&p)], 0.1, None, None, None, true).is_err()
    );
    // Nesterov with dampening.
    assert!(torsh_optim::sgd::SGD::try_new(
        vec![Arc::clone(&p)],
        0.1,
        Some(0.9),
        Some(0.5),
        None,
        true
    )
    .is_err());
    // Learning rate validation (not checked at all by the panicking constructor).
    assert!(
        torsh_optim::sgd::SGD::try_new(vec![Arc::clone(&p)], -1.0, None, None, None, false)
            .is_err()
    );
    assert!(torsh_optim::sgd::SGD::try_new(
        vec![Arc::clone(&p)],
        f32::NAN,
        None,
        None,
        None,
        false
    )
    .is_err());
    // A valid configuration still succeeds.
    assert!(torsh_optim::sgd::SGD::try_new(
        vec![Arc::clone(&p)],
        0.1,
        Some(0.9),
        None,
        Some(1e-4),
        true
    )
    .is_ok());
}

#[test]
fn f236_gradient_accumulator_try_new_rejects_zero_steps() {
    use torsh_optim::grad_accumulation::{GradientAccumulationSupport, GradientAccumulator};

    let p = param(vec![1.0]);
    let sgd = torsh_optim::sgd::SGD::new(vec![Arc::clone(&p)], 0.1, None, None, None, false);
    assert!(GradientAccumulator::try_new(sgd, 0).is_err());

    let sgd = torsh_optim::sgd::SGD::new(vec![Arc::clone(&p)], 0.1, None, None, None, false);
    assert!(GradientAccumulator::try_new(sgd, 4).is_ok());

    let sgd = torsh_optim::sgd::SGD::new(vec![p], 0.1, None, None, None, false);
    let mut accumulating = torsh_optim::grad_accumulation::AccumulatingOptimizer::new(sgd);
    assert!(accumulating.set_accumulation_steps(0).is_err());
    assert!(accumulating.set_accumulation_steps(3).is_ok());
    assert_eq!(accumulating.get_accumulation_steps(), 3);
}

// ---------------------------------------------------------------------------
// F237 — unsupported gradient rank must degrade, not abort
// ---------------------------------------------------------------------------

#[test]
fn f237_block_sparse_rejects_unsupported_rank() {
    use torsh_optim::sparse_updates::{
        BlockSparseGradient, SparseUpdateManager, SparseUpdateResult,
    };

    let gradient = vec![0.0f32; 2 * 3 * 4 * 5];
    assert!(
        BlockSparseGradient::from_dense(&gradient, vec![2, 3, 4, 5], 2, 0.1).is_err(),
        "rank-4 conv weights have no block-sparse layout here"
    );
    assert!(BlockSparseGradient::from_dense(&gradient, vec![120], 2, 0.1).is_ok());

    // The optimizer path must fall back to a dense update rather than abort.
    let mut manager = SparseUpdateManager::new(Default::default());
    let result = manager.process_gradient_block_sparse(
        "conv.weight".to_string(),
        gradient.clone(),
        vec![2, 3, 4, 5],
    );
    assert!(matches!(result, SparseUpdateResult::Dense(_)));
}

// ---------------------------------------------------------------------------
// F183 — trust-region reduction ratios must be measured, not hardcoded
// ---------------------------------------------------------------------------

#[test]
fn f183_trust_region_requires_an_objective() {
    use torsh_optim::trust_region::TrustRegionMethod;

    let p = param(vec![1.0, 2.0]);
    p.read().set_grad(Some(grad(vec![1.0, 1.0])));

    let mut opt = TrustRegionMethod::new(vec![Arc::clone(&p)], None, None, None);
    assert!(
        opt.step().is_err(),
        "without an objective the actual decrease cannot be measured"
    );

    // With an objective the step runs and the reduction ratio is a real
    // measurement: for f(x) = 0.5*||x||^2 a descent step must reduce f.
    opt.set_objective(|x: &Tensor| Ok(0.5 * x.dot(x)?.item()?));
    let before = 0.5 * values(&p).iter().map(|v| v * v).sum::<f32>();
    opt.step().expect("step with an objective");
    let after = 0.5 * values(&p).iter().map(|v| v * v).sum::<f32>();
    assert!(
        after <= before,
        "an accepted trust-region step must not increase the objective: {before} -> {after}"
    );
}

#[test]
fn f183_newton_cg_reduction_ratio_uses_the_objective() {
    use torsh_optim::newton_cg::NewtonCG;

    let p = param(vec![1.0, 2.0]);
    p.read().set_grad(Some(grad(vec![1.0, 2.0])));

    let mut opt = NewtonCG::new(vec![Arc::clone(&p)], Some(0.1), None, None, None);
    assert!(opt.step().is_err(), "trust region needs an objective");

    opt.set_objective(|x: &Tensor| Ok(0.5 * x.dot(x)?.item()?));
    opt.step().expect("step with an objective");
}

// ---------------------------------------------------------------------------
// ROADMAP v0.2.0 — advanced optimizers must own and update real parameters
// ---------------------------------------------------------------------------

#[test]
fn roadmap_advanced_optimizers_state_and_param_groups_round_trip() {
    use std::collections::HashMap;
    use torsh_optim::advanced::{AdvancedAdam, Lookahead, LAMB};

    // AdvancedAdam: add_param_group is honoured and state_dict carries the
    // parameter groups and the moment buffers.
    let a = param(vec![1.0]);
    let b = param(vec![1.0]);
    a.read().set_grad(Some(grad(vec![1.0])));
    b.read().set_grad(Some(grad(vec![1.0])));

    let mut adam = AdvancedAdam::with_params(0.1, vec![Arc::clone(&a)]);
    let mut options = HashMap::new();
    options.insert("lr".to_string(), 0.2);
    adam.add_param_group(vec![Arc::clone(&b)], options);
    adam.step().expect("step");

    let dict = adam.state_dict().expect("state_dict");
    assert_eq!(
        dict.param_groups.len(),
        2,
        "param groups must be serialized"
    );
    assert_eq!(
        dict.state.len(),
        2,
        "per-parameter state must be serialized"
    );
    assert!(values(&a)[0] < 1.0 && values(&b)[0] < 1.0);

    // LAMB likewise.
    let c = param(vec![1.0, 1.0]);
    c.read().set_grad(Some(grad(vec![1.0, 1.0])));
    let mut lamb = LAMB::with_params(0.01, vec![Arc::clone(&c)]);
    lamb.step().expect("step");
    assert!(values(&c)[0] < 1.0, "LAMB must update its parameters");
    assert_eq!(
        lamb.state_dict().expect("state_dict").state.len(),
        1,
        "LAMB per-parameter state must be serialized"
    );

    // Lookahead: slow weights actually move on the k-th step.
    let d = param(vec![0.0]);
    d.read().set_grad(Some(grad(vec![1.0])));
    let mut lookahead =
        Lookahead::new(AdvancedAdam::with_params(0.1, vec![Arc::clone(&d)]), 0.5, 2);
    lookahead.step().expect("step 1");
    assert!(
        lookahead
            .slow_weights
            .values()
            .all(|w| w.to_vec().expect("to_vec")[0] == 0.0),
        "slow weights must not move before the k-th step"
    );
    lookahead.step().expect("step 2");
    assert!(
        lookahead
            .slow_weights
            .values()
            .all(|w| w.to_vec().expect("to_vec")[0] < 0.0),
        "slow weights must be updated on the k-th step"
    );
}

#[test]
fn roadmap_saga_state_dict_round_trips_the_gradient_table() {
    use torsh_optim::online_learning::SAGA;

    let p = param(vec![1.0, 2.0]);
    p.read().set_grad(Some(grad(vec![0.5, 0.5])));

    let mut saga = SAGA::new(vec![Arc::clone(&p)], 0.1, 3);
    saga.initialize().expect("initialize");
    saga.saga_step(0, &[grad(vec![0.5, 0.5])])
        .expect("saga step on data point 0");

    let dict = saga.state_dict().expect("state_dict");
    assert!(
        dict.state.contains_key("gradient_sum"),
        "the running gradient sum must be serialized"
    );
    assert!(
        dict.state.keys().any(|k| k.starts_with("data_")),
        "the gradient table must be serialized, got keys {:?}",
        dict.state.keys().collect::<Vec<_>>()
    );

    let mut restored = SAGA::new(vec![Arc::clone(&p)], 0.0, 3);
    restored.load_state_dict(dict).expect("load_state_dict");
    let restored_dict = restored.state_dict().expect("state_dict");
    assert!(restored_dict.state.contains_key("gradient_sum"));
    assert!(restored_dict.state.keys().any(|k| k.starts_with("data_")));
}

/// The momentum buffer is now mutated in place inside the optimizer state; this
/// pins the analytic trajectory so the fusion cannot silently change semantics.
#[test]
fn f136_sgd_momentum_buffer_accumulates_analytically() {
    let p = param(vec![0.0]);
    p.read().set_grad(Some(grad(vec![1.0])));

    let mut opt =
        torsh_optim::sgd::SGD::new(vec![Arc::clone(&p)], 0.1, Some(0.9), None, None, false);

    // buf: 1.0, 1.9, 2.71 -> param: -0.1, -0.29, -0.561
    for expected in [-0.1f32, -0.29, -0.561] {
        opt.step().expect("step");
        let got = values(&p)[0];
        assert!(
            (got - expected).abs() < 1e-5,
            "expected {expected}, got {got}"
        );
    }
}
