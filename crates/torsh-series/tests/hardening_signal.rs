//! Production-hardening regression tests for `torsh-series`.
//!
//! Test names carry the campaign finding IDs.

use torsh_series::forecast::LSTMForecaster;
use torsh_series::state_space::KalmanFilter;
use torsh_series::TimeSeries;
use torsh_tensor::creation::eye;
use torsh_tensor::Tensor;

// ---------------------------------------------------------------------------
// F142: multivariate Kalman gain divided by a single scalar.
// ---------------------------------------------------------------------------

fn diag2(a: f32, b: f32) -> Tensor {
    Tensor::from_vec(vec![a, 0.0, 0.0, b], &[2, 2]).expect("tensor creation should succeed")
}

#[test]
fn f142_multivariate_kalman_gain_inverts_innovation_covariance() {
    // H = I, P = I, R = diag(0.1, 10) => S = diag(1.1, 11), K = P H^T S^-1 = diag(1/1.1, 1/11).
    let kf = KalmanFilter::with_matrices(
        2,
        2,
        eye(2).expect("eye should succeed"),
        eye(2).expect("eye should succeed"),
        eye(2).expect("eye should succeed"),
        diag2(0.1, 10.0),
    );

    let gain = kf.kalman_gain().expect("kalman gain should succeed");
    let k = gain.to_vec().expect("to_vec should succeed");
    assert_eq!(k.len(), 4);

    assert!(
        (k[0] - 1.0 / 1.1).abs() < 1e-3,
        "K[0][0] should be 1/1.1, got {}",
        k[0]
    );
    assert!(k[1].abs() < 1e-3, "K[0][1] should be 0, got {}", k[1]);
    assert!(k[2].abs() < 1e-3, "K[1][0] should be 0, got {}", k[2]);
    assert!(
        (k[3] - 1.0 / 11.0).abs() < 1e-3,
        "K[1][1] should be 1/11 (the bug divides by S[0][0] and gives ~0.909), got {}",
        k[3]
    );
}

#[test]
fn f142_multivariate_update_uses_the_full_gain() {
    let mut kf = KalmanFilter::with_matrices(
        2,
        2,
        eye(2).expect("eye should succeed"),
        eye(2).expect("eye should succeed"),
        eye(2).expect("eye should succeed"),
        diag2(0.1, 10.0),
    );

    let obs = Tensor::from_vec(vec![1.0f32, 1.0], &[2, 1]).expect("tensor creation");
    kf.update(&obs).expect("update should succeed");

    let state = kf.state().to_vec().expect("to_vec should succeed");
    assert!(
        (state[0] - 1.0 / 1.1).abs() < 1e-3,
        "x[0] should be ~0.909, got {}",
        state[0]
    );
    assert!(
        (state[1] - 1.0 / 11.0).abs() < 1e-3,
        "x[1] should be ~0.0909 (the bug gives ~0.909), got {}",
        state[1]
    );

    // The updated covariance must stay symmetric positive definite.
    let cov = kf.covariance().to_vec().expect("to_vec should succeed");
    assert!(
        (cov[1] - cov[2]).abs() < 1e-5,
        "covariance must stay symmetric, got {:?}",
        cov
    );
    assert!(
        cov[0] > 0.0 && cov[3] > 0.0,
        "covariance diagonal must be positive"
    );
}

// ---------------------------------------------------------------------------
// F047: LSTMForecaster::fit() had an empty body.
// ---------------------------------------------------------------------------

fn sine_series(len: usize) -> TimeSeries {
    let data: Vec<f32> = (0..len).map(|i| (i as f32 * 0.2).sin()).collect();
    TimeSeries::new(Tensor::from_vec(data, &[len]).expect("tensor creation should succeed"))
}

/// Documents *why* [`LSTMForecaster::fit`] implements backpropagation through
/// time by hand instead of relying on `Tensor::backward`:
/// `torsh_nn::layers::recurrent::LSTM::forward` rebuilds its stacked output
/// with `Tensor::from_vec`, which severs the autograd graph, so no gradient can
/// ever reach the recurrent weights.
#[test]
fn f047_torsh_nn_lstm_forward_severs_the_autograd_graph() {
    use torsh_nn::layers::recurrent::LSTM;
    use torsh_nn::Module;

    let lstm = LSTM::with_config(1, 3, 1, true, true, 0.0, false).expect("lstm creation");

    for param in lstm.parameters().values() {
        let handle = param.tensor();
        let mut guard = handle.write();
        let enabled = guard.clone().requires_grad_(true);
        *guard = enabled;
    }

    let x = Tensor::from_vec(vec![0.1f32, 0.2, 0.3, 0.4], &[1, 4, 1]).expect("tensor creation");
    let out = lstm.forward(&x).expect("forward should succeed");

    // Deliberately non-blocking: a torsh-nn fix must not fail a torsh-series
    // test, it should just tell us the hand-written BPTT can be retired.
    if out.requires_grad() {
        eprintln!(
            "NOTE: torsh-nn's LSTM now keeps the autograd graph alive; \
             LSTMForecaster::fit can switch from the hand-written BPTT in \
             forecast/lstm_bptt.rs to Tensor::backward"
        );
    }
}

#[test]
fn f047_fit_reduces_training_loss() {
    let series = sine_series(160);
    let mut model = LSTMForecaster::new(1, 8, 1)
        .expect("forecaster creation should succeed")
        .with_sequence_length(8);

    let before = model
        .forecast(&series, 4)
        .expect("forecast should succeed")
        .values
        .to_vec()
        .expect("to_vec should succeed");

    let history = model
        .fit(&series, 20, 0.05)
        .expect("fit should train the model");

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

    let after = model
        .forecast(&series, 4)
        .expect("forecast should succeed")
        .values
        .to_vec()
        .expect("to_vec should succeed");
    let delta: f32 = before
        .iter()
        .zip(after.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(delta > 1e-6, "training must change the model's forecasts");

    // The hand-written BPTT forward pass must agree with the module forward
    // pass it trains: measure the MSE of `LSTMForecaster::forward` now, then
    // run one more epoch, whose first reported loss is measured on exactly
    // these weights (before that epoch's update).
    let (inputs, targets) = model
        .create_sequences(&series)
        .expect("sequence creation should succeed");
    let predictions = model
        .forward(&inputs)
        .expect("forward should succeed")
        .to_vec()
        .expect("to_vec should succeed");
    let expected_targets = targets.to_vec().expect("to_vec should succeed");
    let mse: f32 = predictions
        .iter()
        .zip(expected_targets.iter())
        .map(|(p, t)| (p - t) * (p - t))
        .sum::<f32>()
        / predictions.len() as f32;
    let reported = model.fit(&series, 1, 0.05).expect("fit should succeed")[0];
    assert!(
        (reported - mse).abs() <= 1e-3 * mse.abs().max(1e-3),
        "hand-written BPTT forward ({reported}) disagrees with LSTM::forward ({mse})"
    );
}
