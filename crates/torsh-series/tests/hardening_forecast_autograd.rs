//! Regression tests for [`LSTMForecaster`]'s autograd-based training.
//!
//! Finding F047 (second half): `LSTMForecaster::fit` used to hand-roll the LSTM
//! forward *and* backward recurrence over raw `f32` because
//! `torsh_nn::layers::recurrent::LSTM::forward` severed the autograd graph. Now
//! that the module keeps the graph alive, `fit` trains through
//! `Tensor::backward`, and these tests pin the properties the hand-written path
//! could never provide:
//!
//! * every trainable parameter stays a grad-tracking leaf across an update
//!   (the hand-written write-back dropped `requires_grad`),
//! * every LSTM parameter is genuinely reachable from the loss, and
//! * the loss `fit` reports is the loss of the module forward pass it trains —
//!   including for multi-layer models, which the single-layer hand-written
//!   recurrence could not model at all.

use torsh_series::forecast::LSTMForecaster;
use torsh_series::TimeSeries;
use torsh_tensor::Tensor;

fn sine_series(len: usize) -> TimeSeries {
    let data: Vec<f32> = (0..len).map(|i| (i as f32 * 0.2).sin()).collect();
    TimeSeries::new(Tensor::from_vec(data, &[len]).expect("tensor creation should succeed"))
}

/// Mean squared error of `LSTMForecaster::forward` over the training windows.
fn forward_mse(model: &LSTMForecaster, series: &TimeSeries) -> f32 {
    let (inputs, targets) = model
        .create_sequences(series)
        .expect("sequence creation should succeed");
    let predictions = model
        .forward(&inputs)
        .expect("forward should succeed")
        .to_vec()
        .expect("to_vec should succeed");
    let expected = targets.to_vec().expect("to_vec should succeed");
    predictions
        .iter()
        .zip(expected.iter())
        .map(|(p, t)| (p - t) * (p - t))
        .sum::<f32>()
        / predictions.len() as f32
}

/// A value in `[0, 1)` that is a pure function of `name` and `index` — no RNG,
/// no thread-local state, no process-global generation counter. FNV-1a folds
/// `name`'s bytes into a 64-bit seed; SplitMix64 (the same finalizer
/// `torsh_tensor::creation` uses internally to derive per-thread seeds from a
/// `manual_seed` value) avalanches `(seed, index)` into a well-distributed
/// value with no visible correlation between neighboring indices.
fn deterministic_unit_interval(name: &str, index: u64) -> f32 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a offset basis
    for byte in name.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3); // FNV-1a prime
    }
    hash ^= index.wrapping_add(0x9E37_79B9_7F4A_7C15);
    hash = (hash ^ (hash >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    hash = (hash ^ (hash >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    hash ^= hash >> 31;
    // The top 24 bits, normalized against `2^24`, give a value in `[0, 1)`
    // with no rounding bias and plenty of entropy for an `f32` payload.
    ((hash >> 40) as f32) / (1u64 << 24) as f32
}

/// Deterministically overwrites every trainable *weight* parameter of
/// `model` with a pure function of `(parameter name, flat index)`, in place.
/// Bias parameters (`lstm.bias_*`, `readout.bias`) are left untouched: they
/// are already `zeros(..)` at construction (see `LSTM::with_config` /
/// `Linear::new`), so they're already deterministic, and overwriting them
/// with nonzero values would make the reinitialized model diverge from what
/// `LSTMForecaster::new` actually produces for no determinism gained. Every
/// bias tensor in this model happens to be 1-D and every weight tensor 2-D,
/// so "1-D" is used as the skip test rather than name-matching "bias".
///
/// # Why `LSTMForecaster::new` can't be seeded from this test
///
/// `LSTMForecaster::new` builds its `LSTM`/`Linear` sublayers, whose weights
/// are drawn by `torsh_nn::init::xavier_uniform`. That function samples via
/// `scirs2_core::random::quick::random_f32`, which draws from
/// `scirs2_core`'s own OS-entropy-seeded `rand::rngs::ThreadRng`  — a
/// generator entirely separate from `torsh_tensor::creation`'s thread-local
/// RNG, the one `torsh_tensor::creation::manual_seed` controls. Calling
/// `manual_seed` before `LSTMForecaster::new` therefore has **no effect** on
/// its initial weights (confirmed empirically: two `LSTMForecaster::new`
/// calls sandwiching the same `manual_seed` value produced different
/// weights). `torsh_nn` and `scirs2_core` are both out of scope for this
/// file, so there is no seed to pin from here — reinitializing the
/// already-constructed parameters in place is the only lever available.
///
/// # Method
///
/// Each weight is refilled with a Xavier/Glorot-uniform-*shaped* value
/// (`bound = sqrt(6 / (fan_in + fan_out))`, the same bound
/// `torsh_nn::init::xavier_uniform` itself uses) so training still behaves
/// like a freshly Xavier-initialized model; only the *source* of the values
/// (a pure hash instead of an RNG draw) changes. Values are written through
/// the parameter's existing `Arc<RwLock<Tensor>>` handle via
/// [`torsh_tensor::Tensor::set_data`], which preserves the tensor's identity
/// (and `requires_grad`) rather than replacing it, so this must run before
/// the model has taken part in any forward/backward pass.
fn deterministic_reinit(model: &LSTMForecaster) {
    for (name, parameter) in model.trainable_parameters() {
        let handle = parameter.tensor();
        let mut guard = handle.write();
        let dims = guard.shape().dims().to_vec();
        if dims.len() != 2 {
            // 1-D: a bias, already zeros(..) and therefore already
            // deterministic. 0-D would be a degenerate empty parameter that
            // doesn't occur in this model; skip it too rather than divide by
            // a zero fan sum.
            continue;
        }
        let numel: usize = dims.iter().product();
        let bound = (6.0 / (dims[0] + dims[1]) as f32).sqrt();
        let values: Vec<f32> = (0..numel)
            .map(|i| bound * (2.0 * deterministic_unit_interval(&name, i as u64) - 1.0))
            .collect();
        guard
            .set_data(&values)
            .expect("deterministic reinit set_data should succeed");
    }
}

#[test]
fn f047_fit_keeps_every_parameter_on_the_autograd_graph() {
    let series = sine_series(120);
    let mut model = LSTMForecaster::new(1, 6, 1)
        .expect("forecaster creation should succeed")
        .with_sequence_length(8);

    model.fit(&series, 3, 0.05).expect("fit should succeed");

    for (name, parameter) in model.trainable_parameters() {
        assert!(
            parameter.requires_grad(),
            "parameter `{name}` lost its Parameter-level grad flag during fit"
        );
        let handle = parameter.tensor();
        let guard = handle.read();
        assert!(
            guard.requires_grad(),
            "parameter `{name}` was written back without requires_grad, so the \
             next fit() cannot compute a gradient for it"
        );
    }
}

#[test]
fn f047_every_lstm_parameter_receives_a_gradient_after_training() {
    let series = sine_series(120);
    let mut model = LSTMForecaster::new(1, 6, 1)
        .expect("forecaster creation should succeed")
        .with_sequence_length(8);

    model.fit(&series, 2, 0.05).expect("fit should succeed");

    // One more forward/backward through the *module* pass: it must reach every
    // trainable parameter of the just-trained model.
    let (inputs, targets) = model
        .create_sequences(&series)
        .expect("sequence creation should succeed");
    let predictions = model.forward(&inputs).expect("forward should succeed");
    assert!(
        predictions.requires_grad(),
        "the forecaster's forward output must stay on the autograd graph"
    );
    let residual = predictions.sub(&targets).expect("sub should succeed");
    let loss = residual
        .mul(&residual)
        .expect("mul should succeed")
        .sum()
        .expect("sum should succeed");
    loss.backward().expect("backward should succeed");

    for (name, parameter) in model.trainable_parameters() {
        let handle = parameter.tensor();
        let guard = handle.read();
        let grad = guard
            .grad()
            .unwrap_or_else(|| panic!("parameter `{name}` received no gradient"));
        let values = grad.to_vec().expect("to_vec should succeed");
        assert!(
            values.iter().all(|v| v.is_finite()),
            "parameter `{name}` received a non-finite gradient: {values:?}"
        );
    }

    // At least one recurrent weight must carry a non-zero gradient, otherwise
    // "Some(grad)" would be satisfied by an all-zero placeholder.
    let recurrent = model
        .trainable_parameters()
        .into_iter()
        .find(|(name, _)| name == "lstm.weight_hh_l0")
        .map(|(_, parameter)| parameter)
        .expect("weight_hh_l0 must exist");
    let handle = recurrent.tensor();
    let guard = handle.read();
    let grad = guard
        .grad()
        .expect("weight_hh_l0 must receive a gradient")
        .to_vec()
        .expect("to_vec should succeed");
    assert!(
        grad.iter().any(|v| v.abs() > 0.0),
        "the recurrent weight gradient is identically zero: {grad:?}"
    );
}

#[test]
fn f047_fit_reports_the_loss_of_the_module_forward_pass() {
    let series = sine_series(120);
    let mut model = LSTMForecaster::new(1, 6, 1)
        .expect("forecaster creation should succeed")
        .with_sequence_length(8);

    let expected = forward_mse(&model, &series);
    let reported = model.fit(&series, 1, 0.05).expect("fit should succeed")[0];

    assert!(
        (reported - expected).abs() <= 1e-3 * expected.abs().max(1e-3),
        "fit reported {reported} but LSTMForecaster::forward scores {expected}"
    );
}

#[test]
fn f047_fit_trains_a_multi_layer_forecaster() {
    let series = sine_series(140);
    let mut model = LSTMForecaster::new(1, 6, 2)
        .expect("forecaster creation should succeed")
        .with_sequence_length(8);
    // F047/flake: `LSTMForecaster::new`'s Xavier init draws from an
    // OS-entropy-seeded RNG this file cannot pin (see
    // `deterministic_reinit`), so an unseeded run occasionally lands on an
    // initialization from which 20 epochs cannot reach the 5% loss-reduction
    // bound below. Reinitializing in place — before `expected` is captured,
    // so `fit`'s first reported loss still matches the forward pass it
    // trains — makes the run, and this assertion, reproducible.
    deterministic_reinit(&model);

    // The reported loss must be the loss of the *two-layer* forward pass; a
    // single-layer surrogate recurrence scores a completely different model.
    let expected = forward_mse(&model, &series);
    let history = model.fit(&series, 20, 0.05).expect("fit should succeed");
    assert!(
        (history[0] - expected).abs() <= 1e-3 * expected.abs().max(1e-3),
        "fit reported {} for a 2-layer model but its forward pass scores {expected}",
        history[0]
    );

    assert!(
        history.iter().all(|loss| loss.is_finite()),
        "training losses must be finite: {history:?}"
    );
    assert!(
        history[history.len() - 1] < history[0] * 0.95,
        "training must measurably reduce the loss: {:?} -> {:?}",
        history[0],
        history[history.len() - 1]
    );

    // Both layers must be trained, not just layer 0.
    let (inputs, targets) = model
        .create_sequences(&series)
        .expect("sequence creation should succeed");
    let predictions = model.forward(&inputs).expect("forward should succeed");
    let residual = predictions.sub(&targets).expect("sub should succeed");
    let loss = residual
        .mul(&residual)
        .expect("mul should succeed")
        .sum()
        .expect("sum should succeed");
    loss.backward().expect("backward should succeed");

    for name in [
        "lstm.weight_ih_l0",
        "lstm.weight_hh_l0",
        "lstm.weight_ih_l1",
        "lstm.weight_hh_l1",
    ] {
        let parameter = model
            .trainable_parameters()
            .into_iter()
            .find(|(key, _)| key == name)
            .map(|(_, parameter)| parameter)
            .unwrap_or_else(|| panic!("parameter `{name}` must exist"));
        let handle = parameter.tensor();
        let guard = handle.read();
        let grad = guard
            .grad()
            .unwrap_or_else(|| panic!("parameter `{name}` received no gradient"))
            .to_vec()
            .expect("to_vec should succeed");
        assert!(
            grad.iter().any(|v| v.abs() > 0.0),
            "parameter `{name}` received an all-zero gradient, so layer {name} is not trained"
        );
    }
}

#[test]
fn f047_fit_reduces_training_loss_on_a_single_layer_model() {
    let series = sine_series(160);
    let mut model = LSTMForecaster::new(1, 8, 1)
        .expect("forecaster creation should succeed")
        .with_sequence_length(8);
    // F047/flake: see `deterministic_reinit` — this test carries the same
    // hard 5%-loss-reduction threshold as
    // `f047_fit_trains_a_multi_layer_forecaster` and the same unseedable
    // Xavier init, so it gets the same fix.
    deterministic_reinit(&model);

    let history = model.fit(&series, 20, 0.05).expect("fit should succeed");
    assert_eq!(history.len(), 20, "fit should report one loss per epoch");
    assert!(
        history.iter().all(|loss| loss.is_finite()),
        "training losses must be finite: {history:?}"
    );
    assert!(
        history[history.len() - 1] < history[0] * 0.95,
        "training must measurably reduce the loss: {:?} -> {:?}",
        history[0],
        history[history.len() - 1]
    );
}
