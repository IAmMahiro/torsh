//! Backpropagation-through-time training for the LSTM forecaster.
//!
//! `torsh-nn`'s `LSTM::forward` materialises its per-time-step outputs through
//! `Tensor::from_vec`, which severs the autograd graph: gradients computed with
//! `Tensor::backward` never reach the recurrent weights. Rather than pretend to
//! train (or train the read-out only), this module implements the LSTM forward
//! and backward recurrences directly on the layer's parameter tensors, so
//! training updates exactly the weights that `LSTM::forward` later uses.
//!
//! The recurrence mirrors `torsh_nn::layers::recurrent::LSTM::lstm_cell`:
//!
//! ```text
//! z      = W_ih x_t + b_ih + W_hh h_{t-1} + b_hh      (4H values, gate order i,f,g,o)
//! i,f,o  = sigmoid(z_i), sigmoid(z_f), sigmoid(z_o)
//! g      = tanh(z_g)
//! c_t    = f * c_{t-1} + i * g
//! h_t    = o * tanh(c_t)
//! y      = h_T W_out + b_out
//! ```

use torsh_core::error::{Result, TorshError};
use torsh_nn::{
    layers::{linear::Linear, recurrent::LSTM},
    Module, Parameter,
};
use torsh_tensor::Tensor;

/// Maximum global gradient norm; larger gradients are rescaled.
const GRAD_CLIP_NORM: f32 = 5.0;

/// Flat view of a parameter tensor plus the handle needed to write it back.
struct FlatParameter {
    parameter: Parameter,
    values: Vec<f32>,
    shape: Vec<usize>,
}

impl FlatParameter {
    fn load(params: &std::collections::HashMap<String, Parameter>, name: &str) -> Result<Self> {
        let parameter = params
            .get(name)
            .ok_or_else(|| {
                TorshError::InvalidArgument(format!("LSTM parameter `{name}` is missing"))
            })?
            .clone();
        let (values, shape) = {
            let handle = parameter.tensor();
            let guard = handle.read();
            (guard.to_vec()?, guard.shape().dims().to_vec())
        };
        Ok(Self {
            parameter,
            values,
            shape,
        })
    }

    fn store(&self) -> Result<()> {
        let handle = self.parameter.tensor();
        let mut guard = handle.write();
        *guard = Tensor::from_vec(self.values.clone(), &self.shape)?;
        Ok(())
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Train a single-layer LSTM plus linear read-out with full-batch gradient
/// descent on the mean squared error.
///
/// * `inputs` - `[batch, seq_len, input_size]`
/// * `targets` - `[batch, out_features]`
///
/// Returns the mean squared error observed at the start of each epoch.
pub(crate) fn train_lstm_readout(
    lstm: &LSTM,
    output_layer: &Linear,
    inputs: &Tensor,
    targets: &Tensor,
    epochs: usize,
    learning_rate: f32,
) -> Result<Vec<f32>> {
    let input_dims = inputs.shape().dims().to_vec();
    if input_dims.len() != 3 {
        return Err(TorshError::InvalidArgument(format!(
            "LSTM training expects [batch, seq_len, input_size] inputs, got {input_dims:?}"
        )));
    }
    let (batch, seq_len, input_size) = (input_dims[0], input_dims[1], input_dims[2]);
    let target_dims = targets.shape().dims().to_vec();
    if target_dims.len() != 2 || target_dims[0] != batch {
        return Err(TorshError::InvalidArgument(format!(
            "LSTM training expects [batch, out_features] targets, got {target_dims:?}"
        )));
    }
    let out_features = target_dims[1];
    if batch == 0 || seq_len == 0 {
        return Err(TorshError::InvalidArgument(
            "LSTM training requires at least one window of at least one step".to_string(),
        ));
    }

    let lstm_params = lstm.parameters();
    let mut weight_ih = FlatParameter::load(&lstm_params, "weight_ih_l0")?;
    let mut weight_hh = FlatParameter::load(&lstm_params, "weight_hh_l0")?;
    let mut bias_ih = FlatParameter::load(&lstm_params, "bias_ih_l0")?;
    let mut bias_hh = FlatParameter::load(&lstm_params, "bias_hh_l0")?;

    let readout_params = output_layer.parameters();
    let mut weight_out = FlatParameter::load(&readout_params, "weight")?;
    let mut bias_out = FlatParameter::load(&readout_params, "bias")?;

    let hidden = weight_hh.shape[1];
    if weight_ih.shape != vec![4 * hidden, input_size] {
        return Err(TorshError::InvalidArgument(format!(
            "LSTM input weight shape {:?} does not match [4*hidden, input_size] = [{}, {}]",
            weight_ih.shape,
            4 * hidden,
            input_size
        )));
    }
    if weight_out.shape != vec![hidden, out_features] {
        return Err(TorshError::InvalidArgument(format!(
            "Read-out weight shape {:?} does not match [hidden, out_features] = [{hidden}, {out_features}]",
            weight_out.shape
        )));
    }

    let x = inputs.to_vec()?;
    let y = targets.to_vec()?;
    let step = learning_rate / batch as f32;
    let mut history = Vec::with_capacity(epochs);

    for _ in 0..epochs {
        // Gradient accumulators.
        let mut g_w_ih = vec![0.0f32; weight_ih.values.len()];
        let mut g_w_hh = vec![0.0f32; weight_hh.values.len()];
        let mut g_b_ih = vec![0.0f32; bias_ih.values.len()];
        let mut g_b_hh = vec![0.0f32; bias_hh.values.len()];
        let mut g_w_out = vec![0.0f32; weight_out.values.len()];
        let mut g_b_out = vec![0.0f32; bias_out.values.len()];
        let mut loss_sum = 0.0f64;

        for b in 0..batch {
            // ---- forward -------------------------------------------------
            let mut gate_i = vec![0.0f32; seq_len * hidden];
            let mut gate_f = vec![0.0f32; seq_len * hidden];
            let mut gate_g = vec![0.0f32; seq_len * hidden];
            let mut gate_o = vec![0.0f32; seq_len * hidden];
            let mut cell = vec![0.0f32; seq_len * hidden];
            let mut tanh_cell = vec![0.0f32; seq_len * hidden];
            let mut hid = vec![0.0f32; seq_len * hidden];

            for t in 0..seq_len {
                let x_off = (b * seq_len + t) * input_size;
                for unit in 0..hidden {
                    let mut z = [0.0f32; 4];
                    for (gate, z_value) in z.iter_mut().enumerate() {
                        let row = gate * hidden + unit;
                        let mut acc = bias_ih.values[row] + bias_hh.values[row];
                        for k in 0..input_size {
                            acc += weight_ih.values[row * input_size + k] * x[x_off + k];
                        }
                        if t > 0 {
                            for k in 0..hidden {
                                acc +=
                                    weight_hh.values[row * hidden + k] * hid[(t - 1) * hidden + k];
                            }
                        }
                        *z_value = acc;
                    }

                    let idx = t * hidden + unit;
                    gate_i[idx] = sigmoid(z[0]);
                    gate_f[idx] = sigmoid(z[1]);
                    gate_g[idx] = z[2].tanh();
                    gate_o[idx] = sigmoid(z[3]);

                    let previous_cell = if t > 0 {
                        cell[(t - 1) * hidden + unit]
                    } else {
                        0.0
                    };
                    let new_cell = gate_f[idx] * previous_cell + gate_i[idx] * gate_g[idx];
                    cell[idx] = new_cell;
                    tanh_cell[idx] = new_cell.tanh();
                    hid[idx] = gate_o[idx] * tanh_cell[idx];
                }
            }

            // ---- read-out ------------------------------------------------
            let last = (seq_len - 1) * hidden;
            let mut d_hidden = vec![0.0f32; hidden];
            for feature in 0..out_features {
                let mut prediction = bias_out.values[feature];
                for unit in 0..hidden {
                    prediction +=
                        hid[last + unit] * weight_out.values[unit * out_features + feature];
                }
                let residual = prediction - y[b * out_features + feature];
                loss_sum += (residual * residual) as f64;

                // d(loss)/d(prediction) for the summed squared error.
                let d_pred = 2.0 * residual;
                g_b_out[feature] += d_pred;
                for unit in 0..hidden {
                    g_w_out[unit * out_features + feature] += d_pred * hid[last + unit];
                    d_hidden[unit] += d_pred * weight_out.values[unit * out_features + feature];
                }
            }

            // ---- backward through time -----------------------------------
            let mut d_cell = vec![0.0f32; hidden];
            for t in (0..seq_len).rev() {
                let mut d_hidden_prev = vec![0.0f32; hidden];
                for unit in 0..hidden {
                    let idx = t * hidden + unit;
                    let d_h = d_hidden[unit];
                    let tanh_c = tanh_cell[idx];

                    let d_o = d_h * tanh_c;
                    d_cell[unit] += d_h * gate_o[idx] * (1.0 - tanh_c * tanh_c);

                    let previous_cell = if t > 0 {
                        cell[(t - 1) * hidden + unit]
                    } else {
                        0.0
                    };
                    let d_i = d_cell[unit] * gate_g[idx];
                    let d_g = d_cell[unit] * gate_i[idx];
                    let d_f = d_cell[unit] * previous_cell;

                    // Pre-activation gradients (gate order: i, f, g, o).
                    let dz = [
                        d_i * gate_i[idx] * (1.0 - gate_i[idx]),
                        d_f * gate_f[idx] * (1.0 - gate_f[idx]),
                        d_g * (1.0 - gate_g[idx] * gate_g[idx]),
                        d_o * gate_o[idx] * (1.0 - gate_o[idx]),
                    ];

                    // Propagate the cell gradient to the previous step.
                    d_cell[unit] *= gate_f[idx];

                    let x_off = (b * seq_len + t) * input_size;
                    for (gate, &dz_value) in dz.iter().enumerate() {
                        let row = gate * hidden + unit;
                        g_b_ih[row] += dz_value;
                        g_b_hh[row] += dz_value;
                        for k in 0..input_size {
                            g_w_ih[row * input_size + k] += dz_value * x[x_off + k];
                        }
                        if t > 0 {
                            for k in 0..hidden {
                                let h_prev = hid[(t - 1) * hidden + k];
                                g_w_hh[row * hidden + k] += dz_value * h_prev;
                                d_hidden_prev[k] += dz_value * weight_hh.values[row * hidden + k];
                            }
                        }
                    }
                }
                d_hidden = d_hidden_prev;
            }
        }

        let denominator = (batch * out_features) as f64;
        let mse = (loss_sum / denominator) as f32;
        if !mse.is_finite() {
            return Err(TorshError::ComputeError(format!(
                "LSTM training diverged: loss is {mse}"
            )));
        }
        history.push(mse);

        // ---- gradient clipping ------------------------------------------
        let mut norm_sq = 0.0f64;
        for slice in [&g_w_ih, &g_w_hh, &g_b_ih, &g_b_hh, &g_w_out, &g_b_out] {
            for value in slice.iter() {
                norm_sq += (*value as f64) * (*value as f64);
            }
        }
        let norm = (norm_sq.sqrt() as f32) * step;
        let scale = if norm > GRAD_CLIP_NORM && norm.is_finite() {
            GRAD_CLIP_NORM / norm
        } else {
            1.0
        };
        let effective_step = step * scale;

        // ---- parameter update -------------------------------------------
        let apply = |values: &mut Vec<f32>, grads: &[f32]| {
            for (value, grad) in values.iter_mut().zip(grads.iter()) {
                *value -= effective_step * grad;
            }
        };
        apply(&mut weight_ih.values, &g_w_ih);
        apply(&mut weight_hh.values, &g_w_hh);
        apply(&mut bias_ih.values, &g_b_ih);
        apply(&mut bias_hh.values, &g_b_hh);
        apply(&mut weight_out.values, &g_w_out);
        apply(&mut bias_out.values, &g_b_out);
    }

    // Write the trained weights back into the layers.
    weight_ih.store()?;
    weight_hh.store()?;
    bias_ih.store()?;
    bias_hh.store()?;
    weight_out.store()?;
    bias_out.store()?;

    Ok(history)
}
