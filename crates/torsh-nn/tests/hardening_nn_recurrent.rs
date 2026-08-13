//! Production-hardening regression tests for the recurrent layers of torsh-nn.
//!
//! The findings pinned down here are all "the recurrent graph is cut somewhere
//! between the cell and the loss": `stack_time_major` and `concat_features`
//! rebuilt their result through `Tensor::from_vec`, `functional::dropout`
//! rebuilt it through `Tensor::from_data`, and `RNN::forward` never read its
//! weights at all.
//!
//! Every gradcheck below obeys the harness rules that the investigation
//! established the hard way:
//!
//! * every analytic gradient is snapshotted immediately after `backward()` and
//!   before the first finite-difference forward — each finite-difference
//!   forward rewrites the parameter tensors, which resets every `grad` slot;
//! * every parameter write-back re-applies `requires_grad_(true)`, because
//!   `Tensor::from_vec` yields a detached leaf and iteration 2+ would otherwise
//!   produce no analytic gradient at all;
//! * the loss seed is non-uniform (`sum(output * pattern)`): a uniform seed
//!   cancels sign errors and gate permutations;
//! * central differences with a step of `1e-2` and a relative tolerance of
//!   `2e-2`, the convention of `torsh-tensor`'s `hardening_autograd.rs`
//!   (`backward()` is f32-only, so a large step keeps the truncation error
//!   below the rounding error).

use std::collections::HashMap;

use torsh_core::error::Result;
use torsh_nn::functional;
use torsh_nn::layers::{CustomRNNCell, GRUCell, LSTMCell, RNNCell, GRU, LSTM, RNN};
use torsh_nn::{Module, Parameter};
use torsh_tensor::creation::zeros;
use torsh_tensor::Tensor;

/// Finite-difference step, matching `hardening_autograd.rs`.
const STEP: f32 = 1e-2;

/// Parameter values keyed by name, together with the shape they are written
/// back at.
type ParamValues = HashMap<String, (Vec<f32>, Vec<usize>)>;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

/// Deterministic, non-constant parameter/input values.
fn seeded(len: usize, offset: f32) -> Vec<f32> {
    (0..len)
        .map(|index| ((((index * 37 + 11) % 23) as f32 / 23.0) - 0.5) * 0.9 + offset)
        .collect()
}

/// Non-uniform loss seed (rule R3).
fn pattern(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| 0.3 + 0.7 * ((index % 5) as f32) - 0.11 * (index as f32))
        .collect()
}

/// `sum(output * pattern)` as a differentiable tensor.
fn loss_tensor(output: &Tensor) -> Result<Tensor> {
    let dims = output.shape().dims().to_vec();
    let len: usize = dims.iter().product();
    let seed = Tensor::from_vec(pattern(len), &dims)?;
    output.mul_op(&seed)?.sum()
}

/// The same loss evaluated numerically, for the finite-difference probes.
fn loss_value(output: &Tensor) -> Result<f32> {
    let len: usize = output.shape().dims().iter().product();
    let seed = pattern(len);
    Ok(output
        .to_vec()?
        .iter()
        .zip(seed.iter())
        .map(|(value, weight)| value * weight)
        .sum())
}

/// Overwrite a module parameter in place, re-applying `requires_grad` (rule R2).
fn set_param(params: &HashMap<String, Parameter>, name: &str, values: &[f32], shape: &[usize]) {
    let param = params
        .get(name)
        .unwrap_or_else(|| panic!("parameter {name} must exist"));
    let tensor = Tensor::from_vec(values.to_vec(), shape).expect("parameter tensor");
    *param.tensor().write() = tensor.requires_grad_(true);
}

/// Write every parameter of `names` back from `values`.
fn write_params(params: &HashMap<String, Parameter>, names: &[&str], values: &ParamValues) {
    for name in names {
        let (data, shape) = &values[*name];
        set_param(params, name, data, shape);
    }
}

fn param_shape(params: &HashMap<String, Parameter>, name: &str) -> Vec<usize> {
    let param = params
        .get(name)
        .unwrap_or_else(|| panic!("parameter {name} must exist"));
    let handle = param.tensor();
    let guard = handle.read();
    guard.shape().dims().to_vec()
}

fn param_grad(params: &HashMap<String, Parameter>, name: &str) -> Option<Vec<f32>> {
    let param = params
        .get(name)
        .unwrap_or_else(|| panic!("parameter {name} must exist"));
    let handle = param.tensor();
    let guard = handle.read();
    guard
        .grad()
        .map(|grad| grad.to_vec().expect("gradient to_vec"))
}

/// Snapshot every analytic gradient before any finite-difference run (rule R1).
fn snapshot_grads(
    params: &HashMap<String, Parameter>,
    names: &[&str],
) -> HashMap<String, Option<Vec<f32>>> {
    names
        .iter()
        .map(|name| ((*name).to_string(), param_grad(params, name)))
        .collect()
}

/// Deterministic starting values for every parameter in `names`.
fn initial_values(params: &HashMap<String, Parameter>, names: &[&str]) -> ParamValues {
    names
        .iter()
        .enumerate()
        .map(|(index, name)| {
            let shape = param_shape(params, name);
            let len: usize = shape.iter().product();
            (
                (*name).to_string(),
                (seeded(len, 0.05 * index as f32 + 0.03), shape),
            )
        })
        .collect()
}

/// Central-difference gradient of the loss with respect to one parameter (R4).
fn fd_grad<F>(init: &ParamValues, name: &str, forward: &F) -> Vec<f32>
where
    F: Fn(&ParamValues) -> Result<Tensor>,
{
    let (base, shape) = init[name].clone();
    (0..base.len())
        .map(|index| {
            let mut plus = init.clone();
            let mut shifted = base.clone();
            shifted[index] += STEP;
            plus.insert(name.to_string(), (shifted, shape.clone()));
            let up = loss_value(&forward(&plus).expect("finite-difference forward"))
                .expect("finite-difference loss");

            let mut minus = init.clone();
            let mut shifted = base.clone();
            shifted[index] -= STEP;
            minus.insert(name.to_string(), (shifted, shape.clone()));
            let down = loss_value(&forward(&minus).expect("finite-difference forward"))
                .expect("finite-difference loss");

            (up - down) / (2.0 * STEP)
        })
        .collect()
}

/// Assert two gradient vectors agree within the shared relative tolerance.
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

/// Run a full gradcheck and return the analytic gradients for extra assertions.
fn gradcheck<F>(
    params: &HashMap<String, Parameter>,
    init: &ParamValues,
    names: &[&str],
    forward: F,
) -> HashMap<String, Vec<f32>>
where
    F: Fn(&ParamValues) -> Result<Tensor>,
{
    let output = forward(init).expect("analytic forward");
    assert!(
        output.requires_grad(),
        "the forward output must stay on the autograd graph"
    );

    let loss = loss_tensor(&output).expect("loss");
    loss.backward().expect("backward");

    // Rule R1: snapshot before the first finite-difference forward.
    let analytic = snapshot_grads(params, names);

    let mut checked = HashMap::new();
    for name in names {
        let got = analytic
            .get(*name)
            .and_then(|grad| grad.clone())
            .unwrap_or_else(|| panic!("{name} received no gradient: the graph is severed"));
        let numeric = fd_grad(init, name, &forward);
        assert_close(&got, &numeric, name);
        checked.insert((*name).to_string(), got);
    }
    checked
}

fn max_abs(values: &[f32]) -> f32 {
    values
        .iter()
        .fold(0.0f32, |acc, value| acc.max(value.abs()))
}

const CELL_NAMES: [&str; 4] = ["weight_ih", "weight_hh", "bias_ih", "bias_hh"];
const LAYER0_NAMES: [&str; 4] = ["weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"];

// ---------------------------------------------------------------------------
// B1 - LSTMCell, single step: the regression floor for narrow/transpose/Gather
// ---------------------------------------------------------------------------

#[test]
fn lstm_cell_single_step_gradcheck() {
    let batch = 2;
    let input_size = 3;
    let hidden_size = 2;
    let cell = LSTMCell::new(input_size, hidden_size).expect("lstm cell");
    let params = cell.parameters();
    let init = initial_values(&params, &CELL_NAMES);
    let inputs = seeded(batch * input_size, 0.1);

    let forward = |values: &ParamValues| -> Result<Tensor> {
        write_params(&params, &CELL_NAMES, values);
        let input = Tensor::from_vec(inputs.clone(), &[batch, input_size])?;
        let hidden = zeros(&[batch, hidden_size])?;
        let cell_state = zeros(&[batch, hidden_size])?;
        let (new_hidden, _) = cell.forward_cell(&input, &hidden, &cell_state)?;
        Ok(new_hidden)
    };

    gradcheck(&params, &init, &CELL_NAMES, forward);
}

// ---------------------------------------------------------------------------
// B2 - LSTMCell, three-step unroll: the hidden path must carry gradient
// ---------------------------------------------------------------------------

#[test]
fn lstm_cell_three_step_unroll_gradcheck() {
    let batch = 2;
    let input_size = 3;
    let hidden_size = 2;
    let steps = 3;
    let cell = LSTMCell::new(input_size, hidden_size).expect("lstm cell");
    let params = cell.parameters();
    let init = initial_values(&params, &CELL_NAMES);
    let sequence: Vec<Vec<f32>> = (0..steps)
        .map(|step| seeded(batch * input_size, 0.2 * step as f32 + 0.1))
        .collect();

    let forward = |values: &ParamValues| -> Result<Tensor> {
        write_params(&params, &CELL_NAMES, values);
        let mut hidden = zeros(&[batch, hidden_size])?;
        let mut cell_state = zeros(&[batch, hidden_size])?;
        for step in sequence.iter() {
            let input = Tensor::from_vec(step.clone(), &[batch, input_size])?;
            let (new_hidden, new_cell) = cell.forward_cell(&input, &hidden, &cell_state)?;
            hidden = new_hidden;
            cell_state = new_cell;
        }
        Ok(hidden)
    };

    let analytic = gradcheck(&params, &init, &CELL_NAMES, forward);
    assert!(
        max_abs(&analytic["weight_hh"]) > 1e-4,
        "a multi-step unroll must push gradient through weight_hh"
    );
}

// ---------------------------------------------------------------------------
// B3 - GRUCell: the `(1 - z)` branch must contribute
// ---------------------------------------------------------------------------

#[test]
fn gru_cell_single_step_gradcheck() {
    let batch = 2;
    let input_size = 3;
    let hidden_size = 2;
    let cell = GRUCell::new(input_size, hidden_size).expect("gru cell");
    let params = cell.parameters();
    let init = initial_values(&params, &CELL_NAMES);
    let inputs = seeded(batch * input_size, 0.1);
    // A non-zero h0 is required: with h0 = 0 the `(1 - z) * n` branch barely
    // moves the loss and a broken `add_scalar` is hard to see.
    let hidden_values = seeded(batch * hidden_size, -0.2);

    let forward = |values: &ParamValues| -> Result<Tensor> {
        write_params(&params, &CELL_NAMES, values);
        let input = Tensor::from_vec(inputs.clone(), &[batch, input_size])?;
        let hidden = Tensor::from_vec(hidden_values.clone(), &[batch, hidden_size])?;
        cell.forward_cell(&input, &hidden)
    };

    gradcheck(&params, &init, &CELL_NAMES, forward);
}

// ---------------------------------------------------------------------------
// B4 - the LSTM sequence module must keep output/h_n/c_n on the graph
// ---------------------------------------------------------------------------

#[test]
fn lstm_module_sequence_keeps_the_graph() {
    let seq_len = 3;
    let batch = 2;
    let input_size = 3;
    let hidden_size = 2;
    let lstm = LSTM::new(input_size, hidden_size, 1).expect("lstm");
    let params = lstm.parameters();
    let init = initial_values(&params, &LAYER0_NAMES);
    let inputs = seeded(seq_len * batch * input_size, 0.1);

    write_params(&params, &LAYER0_NAMES, &init);
    let input = Tensor::from_vec(inputs.clone(), &[seq_len, batch, input_size]).expect("input");
    let (output, (h_n, c_n)) = lstm.forward_with_state(&input, None).expect("forward");
    assert!(
        output.requires_grad(),
        "LSTM output must stay on the autograd graph"
    );
    assert!(
        h_n.requires_grad(),
        "LSTM h_n must stay on the autograd graph"
    );
    assert!(
        c_n.requires_grad(),
        "LSTM c_n must stay on the autograd graph"
    );

    let forward = |values: &ParamValues| -> Result<Tensor> {
        write_params(&params, &LAYER0_NAMES, values);
        let input = Tensor::from_vec(inputs.clone(), &[seq_len, batch, input_size])?;
        let (output, _) = lstm.forward_with_state(&input, None)?;
        Ok(output)
    };

    gradcheck(&params, &init, &LAYER0_NAMES, forward);
}

// ---------------------------------------------------------------------------
// B5 - gradient must also flow when the loss is built from h_n alone
// ---------------------------------------------------------------------------

#[test]
fn lstm_module_gradient_flows_through_h_n() {
    let seq_len = 3;
    let batch = 2;
    let input_size = 3;
    let hidden_size = 2;
    let lstm = LSTM::new(input_size, hidden_size, 1).expect("lstm");
    let params = lstm.parameters();
    let init = initial_values(&params, &LAYER0_NAMES);
    let inputs = seeded(seq_len * batch * input_size, 0.1);

    let forward = |values: &ParamValues| -> Result<Tensor> {
        write_params(&params, &LAYER0_NAMES, values);
        let input = Tensor::from_vec(inputs.clone(), &[seq_len, batch, input_size])?;
        let (_, (h_n, _)) = lstm.forward_with_state(&input, None)?;
        Ok(h_n)
    };

    gradcheck(&params, &init, &LAYER0_NAMES, forward);
}

// ---------------------------------------------------------------------------
// B6 - two layers: the layer-to-layer narrow/squeeze must carry gradient
// ---------------------------------------------------------------------------

#[test]
fn lstm_two_layer_gradcheck() {
    let seq_len = 3;
    let batch = 2;
    let input_size = 3;
    let hidden_size = 2;
    let lstm = LSTM::new(input_size, hidden_size, 2).expect("lstm");
    let params = lstm.parameters();
    let names = [
        "weight_ih_l0",
        "weight_hh_l0",
        "bias_ih_l0",
        "bias_hh_l0",
        "weight_ih_l1",
        "weight_hh_l1",
        "bias_ih_l1",
        "bias_hh_l1",
    ];
    let init = initial_values(&params, &names);
    let inputs = seeded(seq_len * batch * input_size, 0.1);

    let forward = |values: &ParamValues| -> Result<Tensor> {
        write_params(&params, &names, values);
        let input = Tensor::from_vec(inputs.clone(), &[seq_len, batch, input_size])?;
        let (output, _) = lstm.forward_with_state(&input, None)?;
        Ok(output)
    };

    let analytic = gradcheck(&params, &init, &names, forward);
    assert!(
        max_abs(&analytic["weight_ih_l0"]) > 1e-4,
        "layer 0 must receive gradient through layer 1"
    );
}

// ---------------------------------------------------------------------------
// B7 - inter-layer dropout must not sever the graph
// ---------------------------------------------------------------------------

#[test]
fn lstm_two_layer_dropout_keeps_the_graph() {
    let seq_len = 3;
    let batch = 2;
    let input_size = 3;
    let hidden_size = 2;
    let lstm =
        LSTM::with_config(input_size, hidden_size, 2, true, false, 0.5, false).expect("lstm");
    assert!(lstm.training(), "the module must default to training mode");

    let params = lstm.parameters();
    let input = Tensor::from_vec(
        seeded(seq_len * batch * input_size, 0.1),
        &[seq_len, batch, input_size],
    )
    .expect("input");

    let output = lstm.forward(&input).expect("forward");
    assert!(
        output.requires_grad(),
        "inter-layer dropout must keep the LSTM output on the autograd graph"
    );

    loss_tensor(&output)
        .expect("loss")
        .backward()
        .expect("backward");

    // Dropout is random per forward, so only the presence of a gradient is a
    // stable claim here; a magnitude assertion would be flaky.
    for name in LAYER0_NAMES.iter() {
        assert!(
            param_grad(&params, name).is_some(),
            "{name} must receive a gradient through the dropout layer"
        );
    }
}

// ---------------------------------------------------------------------------
// B8 - bidirectional LSTM: `concat_features` must stay differentiable
// ---------------------------------------------------------------------------

#[test]
fn bidirectional_lstm_gradcheck() {
    let seq_len = 3;
    let batch = 2;
    let input_size = 3;
    let hidden_size = 2;
    let lstm = LSTM::with_config(input_size, hidden_size, 1, true, false, 0.0, true).expect("lstm");
    let params = lstm.parameters();
    let names = [
        "weight_ih_l0",
        "weight_hh_l0",
        "bias_ih_l0",
        "bias_hh_l0",
        "weight_ih_l0_reverse",
        "weight_hh_l0_reverse",
        "bias_ih_l0_reverse",
        "bias_hh_l0_reverse",
    ];
    let init = initial_values(&params, &names);
    let inputs = seeded(seq_len * batch * input_size, 0.1);

    let forward = |values: &ParamValues| -> Result<Tensor> {
        write_params(&params, &names, values);
        let input = Tensor::from_vec(inputs.clone(), &[seq_len, batch, input_size])?;
        let (output, _) = lstm.forward_with_state(&input, None)?;
        Ok(output)
    };

    let analytic = gradcheck(&params, &init, &names, forward);
    assert!(
        max_abs(&analytic["weight_ih_l0_reverse"]) > 1e-4,
        "the reverse direction must receive gradient"
    );
}

// ---------------------------------------------------------------------------
// B9 - GRU sequence module, unidirectional and bidirectional
// ---------------------------------------------------------------------------

#[test]
fn gru_module_sequence_keeps_the_graph() {
    let seq_len = 3;
    let batch = 2;
    let input_size = 3;
    let hidden_size = 2;
    let gru = GRU::new(input_size, hidden_size, 1).expect("gru");
    let params = gru.parameters();
    let init = initial_values(&params, &LAYER0_NAMES);
    let inputs = seeded(seq_len * batch * input_size, 0.1);

    write_params(&params, &LAYER0_NAMES, &init);
    let input = Tensor::from_vec(inputs.clone(), &[seq_len, batch, input_size]).expect("input");
    let (output, h_n) = gru.forward_with_state(&input, None).expect("forward");
    assert!(
        output.requires_grad(),
        "GRU output must stay on the autograd graph"
    );
    assert!(
        h_n.requires_grad(),
        "GRU h_n must stay on the autograd graph"
    );

    let forward = |values: &ParamValues| -> Result<Tensor> {
        write_params(&params, &LAYER0_NAMES, values);
        let input = Tensor::from_vec(inputs.clone(), &[seq_len, batch, input_size])?;
        let (output, _) = gru.forward_with_state(&input, None)?;
        Ok(output)
    };

    gradcheck(&params, &init, &LAYER0_NAMES, forward);
}

#[test]
fn bidirectional_gru_gradcheck() {
    let seq_len = 3;
    let batch = 2;
    let input_size = 3;
    let hidden_size = 2;
    let gru = GRU::with_config(input_size, hidden_size, 1, true, false, 0.0, true).expect("gru");
    let params = gru.parameters();
    let names = [
        "weight_ih_l0",
        "weight_hh_l0",
        "bias_ih_l0",
        "bias_hh_l0",
        "weight_ih_l0_reverse",
        "weight_hh_l0_reverse",
        "bias_ih_l0_reverse",
        "bias_hh_l0_reverse",
    ];
    let init = initial_values(&params, &names);
    let inputs = seeded(seq_len * batch * input_size, 0.1);

    let forward = |values: &ParamValues| -> Result<Tensor> {
        write_params(&params, &names, values);
        let input = Tensor::from_vec(inputs.clone(), &[seq_len, batch, input_size])?;
        let (output, _) = gru.forward_with_state(&input, None)?;
        Ok(output)
    };

    let analytic = gradcheck(&params, &init, &names, forward);
    assert!(
        max_abs(&analytic["weight_ih_l0_reverse"]) > 1e-4,
        "the reverse direction must receive gradient"
    );
}

// ---------------------------------------------------------------------------
// B10 - the highway cell's carry gate `1 - T`
// ---------------------------------------------------------------------------

#[test]
fn custom_rnn_cell_highway_gradcheck() {
    let batch = 2;
    let input_size = 3;
    let hidden_size = 2;
    let cell = CustomRNNCell::highway(input_size, hidden_size).expect("highway cell");
    let params = cell.parameters();
    let names = [
        "weight_ih",
        "weight_hh",
        "bias_ih",
        "bias_hh",
        "weight_ih_t",
        "weight_hh_t",
        "bias_ih_t",
        "bias_hh_t",
    ];
    let init = initial_values(&params, &names);
    let inputs = seeded(batch * input_size, 0.1);
    let hidden_values = seeded(batch * hidden_size, -0.2);

    let forward = |values: &ParamValues| -> Result<Tensor> {
        write_params(&params, &names, values);
        let input = Tensor::from_vec(inputs.clone(), &[batch, input_size])?;
        let hidden = Tensor::from_vec(hidden_values.clone(), &[batch, hidden_size])?;
        RNNCell::forward(&cell, &input, &hidden)
    };

    gradcheck(&params, &init, &names, forward);
}

// ---------------------------------------------------------------------------
// B11 - RNN::forward must actually read its weights
// ---------------------------------------------------------------------------

#[test]
fn rnn_forward_uses_its_weights() {
    let seq_len = 3;
    let batch = 2;
    let input_size = 3;
    let hidden_size = 2;
    let rnn = RNN::new(input_size, hidden_size, 1).expect("rnn");
    let params = rnn.parameters();
    let init = initial_values(&params, &LAYER0_NAMES);
    let inputs = seeded(seq_len * batch * input_size, 0.1);

    write_params(&params, &LAYER0_NAMES, &init);
    let input = Tensor::from_vec(inputs.clone(), &[seq_len, batch, input_size]).expect("input");
    let output = rnn.forward(&input).expect("forward");
    assert_eq!(output.shape().dims(), &[seq_len, batch, hidden_size]);
    let values = output.to_vec().expect("to_vec");
    assert!(
        values.iter().any(|value| value.abs() > 1e-6),
        "RNN::forward must read its weights, not return zeros"
    );

    let forward = |values: &ParamValues| -> Result<Tensor> {
        write_params(&params, &LAYER0_NAMES, values);
        let input = Tensor::from_vec(inputs.clone(), &[seq_len, batch, input_size])?;
        rnn.forward(&input)
    };

    gradcheck(&params, &init, &LAYER0_NAMES, forward);
}

#[test]
fn rnn_bidirectional_gradcheck() {
    let seq_len = 3;
    let batch = 2;
    let input_size = 3;
    let hidden_size = 2;
    let rnn = RNN::with_config(input_size, hidden_size, 1, true, false, 0.0, true).expect("rnn");
    let params = rnn.parameters();
    let names = [
        "weight_ih_l0",
        "weight_hh_l0",
        "bias_ih_l0",
        "bias_hh_l0",
        "weight_ih_l0_reverse",
        "weight_hh_l0_reverse",
        "bias_ih_l0_reverse",
        "bias_hh_l0_reverse",
    ];
    let init = initial_values(&params, &names);
    let inputs = seeded(seq_len * batch * input_size, 0.1);

    write_params(&params, &names, &init);
    let input = Tensor::from_vec(inputs.clone(), &[seq_len, batch, input_size]).expect("input");
    let output = rnn.forward(&input).expect("forward");
    assert_eq!(
        output.shape().dims(),
        &[seq_len, batch, hidden_size * 2],
        "a bidirectional RNN emits hidden_size * 2 features"
    );

    let forward = |values: &ParamValues| -> Result<Tensor> {
        write_params(&params, &names, values);
        let input = Tensor::from_vec(inputs.clone(), &[seq_len, batch, input_size])?;
        rnn.forward(&input)
    };

    let analytic = gradcheck(&params, &init, &names, forward);
    assert!(
        max_abs(&analytic["weight_ih_l0_reverse"]) > 1e-4,
        "the reverse direction must receive gradient"
    );
}

#[test]
fn rnn_forward_with_state_reads_h0() {
    let seq_len = 3;
    let batch = 2;
    let input_size = 3;
    let hidden_size = 2;
    let rnn = RNN::new(input_size, hidden_size, 1).expect("rnn");
    let params = rnn.parameters();
    let init = initial_values(&params, &LAYER0_NAMES);
    write_params(&params, &LAYER0_NAMES, &init);

    let input = Tensor::from_vec(
        seeded(seq_len * batch * input_size, 0.1),
        &[seq_len, batch, input_size],
    )
    .expect("input");
    let h0 = Tensor::from_vec(seeded(batch * hidden_size, -0.4), &[1, batch, hidden_size])
        .expect("h0")
        .requires_grad_(true);

    let (zero_state, _) = rnn.forward_with_state(&input, None).expect("forward");
    let (with_state, h_n) = rnn
        .forward_with_state(&input, Some(&h0))
        .expect("forward with state");

    assert_eq!(h_n.shape().dims(), &[1, batch, hidden_size]);
    let zero_state = zero_state.to_vec().expect("to_vec");
    let with_state_values = with_state.to_vec().expect("to_vec");
    assert!(
        zero_state
            .iter()
            .zip(with_state_values.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6),
        "a non-zero h0 must change the output"
    );

    // The initial state is part of the graph too.
    loss_tensor(&with_state)
        .expect("loss")
        .backward()
        .expect("backward");
    assert!(
        h0.grad().is_some(),
        "gradient must reach the initial hidden state"
    );

    // A mis-shaped h0 is rejected rather than silently broadcast.
    let wrong = Tensor::from_vec(vec![0.0f32; batch * hidden_size], &[batch, hidden_size])
        .expect("wrong h0");
    assert!(
        rnn.forward_with_state(&input, Some(&wrong)).is_err(),
        "h0 must be validated against [num_layers * num_directions, batch, hidden]"
    );
}

#[test]
fn rnn_batch_first_transposes_both_ends() {
    let seq_len = 3;
    let batch = 2;
    let input_size = 3;
    let hidden_size = 2;
    let time_major = RNN::with_config(input_size, hidden_size, 1, true, false, 0.0, false)
        .expect("time-major rnn");
    let batch_first = RNN::with_config(input_size, hidden_size, 1, true, true, 0.0, false)
        .expect("batch-first rnn");

    let init = initial_values(&time_major.parameters(), &LAYER0_NAMES);
    write_params(&time_major.parameters(), &LAYER0_NAMES, &init);
    write_params(&batch_first.parameters(), &LAYER0_NAMES, &init);

    let inputs = seeded(seq_len * batch * input_size, 0.1);
    let time_major_input =
        Tensor::from_vec(inputs.clone(), &[seq_len, batch, input_size]).expect("input");
    let batch_first_input = time_major_input.transpose(0, 1).expect("transpose");

    let expected = time_major
        .forward(&time_major_input)
        .expect("time-major forward");
    let got = batch_first
        .forward(&batch_first_input)
        .expect("batch-first forward");
    assert_eq!(got.shape().dims(), &[batch, seq_len, hidden_size]);

    let expected = expected
        .transpose(0, 1)
        .expect("transpose")
        .to_vec()
        .expect("to_vec");
    let got = got.to_vec().expect("to_vec");
    assert_close(&got, &expected, "batch_first output");
}

// ---------------------------------------------------------------------------
// B12 - functional::dropout must be differentiable
// ---------------------------------------------------------------------------

#[test]
fn dropout_is_differentiable() {
    let probability = 0.5f32;
    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .expect("input")
        .requires_grad_(true);

    let output = functional::dropout(&input, probability, true).expect("dropout");
    assert!(
        output.requires_grad(),
        "dropout must keep its input on the autograd graph"
    );

    // A uniform seed makes the expected gradient exactly the Bernoulli mask.
    output.sum().expect("sum").backward().expect("backward");

    let forward = output.to_vec().expect("to_vec");
    let grad = input
        .grad()
        .expect("dropout must produce a gradient")
        .to_vec()
        .expect("to_vec");
    let scale = 1.0 / (1.0 - probability);
    for (index, value) in forward.iter().enumerate() {
        let expected = if *value == 0.0 { 0.0 } else { scale };
        assert!(
            (grad[index] - expected).abs() <= 1e-6,
            "dropout gradient[{index}] = {}, expected {expected}",
            grad[index]
        );
    }
}

#[test]
fn dropout_drop_everything_stays_differentiable() {
    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[4])
        .expect("input")
        .requires_grad_(true);

    let output = functional::dropout(&input, 1.0, true).expect("dropout");
    assert!(
        output.requires_grad(),
        "dropout with p = 1 must stay on the autograd graph"
    );
    assert!(
        output.to_vec().expect("to_vec").iter().all(|v| *v == 0.0),
        "dropout with p = 1 must zero every element"
    );

    output.sum().expect("sum").backward().expect("backward");
    let grad = input
        .grad()
        .expect("dropout must produce a gradient")
        .to_vec()
        .expect("to_vec");
    assert!(
        grad.iter().all(|value| *value == 0.0),
        "dropout with p = 1 has a zero gradient everywhere"
    );
}

#[test]
fn dropout_rejects_an_out_of_range_probability() {
    let input = Tensor::from_vec(vec![1.0f32, 2.0], &[2]).expect("input");
    assert!(
        functional::dropout(&input, 1.5, true).is_err(),
        "a probability outside [0, 1] must be rejected"
    );
}
