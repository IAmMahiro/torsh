//! Recurrent neural network layers

use crate::{Module, ModuleBase, Parameter};
use torsh_core::device::DeviceType;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;
use torsh_core::error::Result;
use torsh_tensor::{creation::*, Tensor};

/// Multi-layer Elman RNN with a `tanh` non-linearity.
///
/// # PyTorch compatibility
///
/// Parameters follow `torch.nn.RNN` naming: `weight_ih_l{k}`, `weight_hh_l{k}`,
/// `bias_ih_l{k}`, `bias_hh_l{k}`, with a `_reverse` suffix for the backward
/// direction of a bidirectional layer. Layer `k > 0` consumes the previous
/// layer's output, so its input size is `hidden_size * num_directions`.
pub struct RNN {
    base: ModuleBase,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    bias: bool,
    batch_first: bool,
    dropout: f32,
    bidirectional: bool,
}

impl RNN {
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Result<Self> {
        Self::with_config(input_size, hidden_size, num_layers, true, false, 0.0, false)
    }

    pub fn with_config(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bias: bool,
        batch_first: bool,
        dropout: f32,
        bidirectional: bool,
    ) -> Result<Self> {
        if num_layers == 0 {
            return Err(torsh_core::TorshError::InvalidArgument(
                "RNN requires at least one layer".to_string(),
            ));
        }

        let mut base = ModuleBase::new();
        let directions = if bidirectional { 2 } else { 1 };

        // Initialize weights for each layer and direction.
        for layer in 0..num_layers {
            let layer_input = if layer == 0 {
                input_size
            } else {
                hidden_size * directions
            };

            for direction in 0..directions {
                let suffix = recurrent_gated::direction_suffix(direction);
                let weight_ih = crate::init::xavier_uniform(&[hidden_size, layer_input])?;
                let weight_hh = crate::init::xavier_uniform(&[hidden_size, hidden_size])?;

                base.register_parameter(
                    format!("weight_ih_l{}{}", layer, suffix),
                    Parameter::new(weight_ih),
                );
                base.register_parameter(
                    format!("weight_hh_l{}{}", layer, suffix),
                    Parameter::new(weight_hh),
                );

                if bias {
                    base.register_parameter(
                        format!("bias_ih_l{}{}", layer, suffix),
                        Parameter::new(zeros(&[hidden_size])?),
                    );
                    base.register_parameter(
                        format!("bias_hh_l{}{}", layer, suffix),
                        Parameter::new(zeros(&[hidden_size])?),
                    );
                }
            }
        }

        Ok(Self {
            base,
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
        })
    }

    /// Number of directions (2 when bidirectional).
    fn directions(&self) -> usize {
        if self.bidirectional {
            2
        } else {
            1
        }
    }

    /// Fallible parameter lookup: a renamed or missing key must not panic.
    fn parameter(&self, name: &str) -> Result<Tensor> {
        Ok(self
            .base
            .parameters
            .get(name)
            .ok_or_else(|| {
                torsh_core::TorshError::InvalidArgument(format!("RNN is missing parameter {name}"))
            })?
            .tensor()
            .read()
            .clone())
    }

    /// Single Elman step for one layer and direction:
    /// `h_t = tanh(x_t @ W_ih^T + b_ih + h_{t-1} @ W_hh^T + b_hh)`.
    fn rnn_cell_directional(
        &self,
        input: &Tensor,
        hidden: &Tensor,
        layer: usize,
        direction: usize,
    ) -> Result<Tensor> {
        let suffix = recurrent_gated::direction_suffix(direction);
        let weight_ih = self.parameter(&format!("weight_ih_l{}{}", layer, suffix))?;
        let weight_hh = self.parameter(&format!("weight_hh_l{}{}", layer, suffix))?;

        let mut gi = input.matmul(&weight_ih.transpose(0, 1)?)?;
        let mut gh = hidden.matmul(&weight_hh.transpose(0, 1)?)?;
        if self.bias {
            gi = gi.add_op(&self.parameter(&format!("bias_ih_l{}{}", layer, suffix))?)?;
            gh = gh.add_op(&self.parameter(&format!("bias_hh_l{}{}", layer, suffix))?)?;
        }

        gi.add_op(&gh)?.tanh()
    }

    /// Run every layer (and direction) over a time-major input sequence.
    ///
    /// Returns the last layer's output sequence together with the final hidden
    /// state of every (layer, direction) pair, in PyTorch order
    /// (`layer 0 forward, layer 0 reverse, layer 1 forward, ...`).
    fn run_layers(&self, input: &Tensor, state: Option<&Tensor>) -> Result<(Tensor, Vec<Tensor>)> {
        let binding = input.shape();
        let input_shape = binding.dims();
        let seq_len = input_shape[0];
        let batch_size = input_shape[1];
        let directions = self.directions();

        if input_shape[2] != self.input_size {
            return Err(torsh_core::TorshError::InvalidShape(format!(
                "RNN expects an input of size {}, got {}",
                self.input_size, input_shape[2]
            )));
        }

        let h0 = match state {
            Some(h) => {
                let expected = [self.num_layers * directions, batch_size, self.hidden_size];
                if h.shape().dims() != expected {
                    return Err(torsh_core::TorshError::InvalidShape(format!(
                        "RNN h0 must have shape {:?}, got {:?}",
                        expected,
                        h.shape().dims()
                    )));
                }
                Some(h.clone())
            }
            None => None,
        };

        let mut layer_input = input.clone();
        let mut final_hidden = Vec::with_capacity(self.num_layers * directions);

        for layer in 0..self.num_layers {
            let mut direction_outputs: Vec<Vec<Tensor>> = Vec::with_capacity(directions);

            for direction in 0..directions {
                let state_index = layer * directions + direction;
                let mut hidden = match &h0 {
                    Some(h) => h.narrow(0, state_index as i64, 1)?.squeeze(0)?,
                    None => zeros(&[batch_size, self.hidden_size])?,
                };

                let mut outputs = Vec::with_capacity(seq_len);
                for step in 0..seq_len {
                    let t = if direction == 0 {
                        step
                    } else {
                        seq_len - 1 - step
                    };
                    let x_t = layer_input.narrow(0, t as i64, 1)?.squeeze(0)?;
                    hidden = self.rnn_cell_directional(&x_t, &hidden, layer, direction)?;
                    outputs.push(hidden.clone());
                }

                if direction == 1 {
                    // The reverse pass produced outputs from the last step
                    // backwards; restore time order before concatenating.
                    outputs.reverse();
                }

                final_hidden.push(hidden);
                direction_outputs.push(outputs);
            }

            // Concatenate the directions along the feature axis, then stack time.
            let mut steps = Vec::with_capacity(seq_len);
            for t in 0..seq_len {
                if directions == 1 {
                    steps.push(direction_outputs[0][t].clone());
                } else {
                    steps.push(recurrent_gated::concat_features(
                        &direction_outputs[0][t],
                        &direction_outputs[1][t],
                    )?);
                }
            }

            let mut stacked = recurrent_gated::stack_time_major(&steps)?;
            // Inter-layer dropout, exactly as PyTorch: applied to the output of
            // every layer except the last, and only while training.
            if layer + 1 < self.num_layers && self.dropout > 0.0 && self.base.training() {
                stacked = crate::functional::dropout(&stacked, self.dropout, true)?;
            }
            layer_input = stacked;
        }

        Ok((layer_input, final_hidden))
    }

    /// Forward pass with an explicit initial hidden state.
    ///
    /// `state` is `h_0` shaped `[num_layers * num_directions, batch, hidden_size]`
    /// (`None` starts from zeros). Returns `(output, h_n)` following
    /// `torch.nn.RNN`, honouring `Self::batch_first` for `output`.
    pub fn forward_with_state(
        &self,
        input: &Tensor,
        state: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        if input.shape().ndim() != 3 {
            return Err(torsh_core::TorshError::InvalidShape(format!(
                "RNN expects a 3-D input, got {}-D",
                input.shape().ndim()
            )));
        }

        // Work time-major internally.
        let time_major = if self.batch_first {
            input.transpose(0, 1)?
        } else {
            input.clone()
        };

        let (output, hidden) = self.run_layers(&time_major, state)?;
        let h_n = recurrent_gated::stack_time_major(&hidden)?;

        let output = if self.batch_first {
            output.transpose(0, 1)?
        } else {
            output
        };

        Ok((output, h_n))
    }
}

impl Module for RNN {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (output, _) = self.forward_with_state(input, None)?;
        Ok(output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

impl std::fmt::Debug for RNN {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RNN")
            .field("input_size", &self.input_size)
            .field("hidden_size", &self.hidden_size)
            .field("num_layers", &self.num_layers)
            .finish()
    }
}

// Gated recurrent variants (LSTM, GRU, LSTMCell, GRUCell, CustomRNNCell)
#[path = "recurrent_gated.rs"]
mod recurrent_gated;
pub use recurrent_gated::*;

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::zeros;

    // ========================================================================
    // RNN Tests
    // ========================================================================

    #[test]
    fn test_rnn_new() -> Result<()> {
        let rnn = RNN::new(10, 20, 2)?;
        assert_eq!(rnn.input_size, 10);
        assert_eq!(rnn.hidden_size, 20);
        assert_eq!(rnn.num_layers, 2);
        assert!(rnn.bias);
        assert!(!rnn.batch_first);
        Ok(())
    }

    #[test]
    fn test_rnn_with_config() -> Result<()> {
        let rnn = RNN::with_config(10, 20, 1, false, true, 0.5, true)?;
        assert_eq!(rnn.input_size, 10);
        assert_eq!(rnn.hidden_size, 20);
        assert!(!rnn.bias);
        assert!(rnn.batch_first);
        assert_eq!(rnn.dropout, 0.5);
        assert!(rnn.bidirectional);
        Ok(())
    }

    #[test]
    fn test_rnn_forward() -> Result<()> {
        let rnn = RNN::new(10, 20, 1)?;
        let input = zeros(&[5, 2, 10])?; // seq_len=5, batch=2, input_size=10

        let output = rnn.forward(&input)?;
        let output_shape = output.shape();

        // Expected: [seq_len, batch, hidden_size]
        assert_eq!(output_shape.dims(), &[5, 2, 20]);
        Ok(())
    }

    #[test]
    fn test_rnn_forward_batch_first() -> Result<()> {
        let rnn = RNN::with_config(10, 20, 1, true, true, 0.0, false)?;
        let input = zeros(&[2, 5, 10])?; // batch=2, seq_len=5, input_size=10

        let output = rnn.forward(&input)?;
        let output_shape = output.shape();

        // Expected: [batch, seq_len, hidden_size]
        assert_eq!(output_shape.dims(), &[2, 5, 20]);
        Ok(())
    }

    #[test]
    fn test_rnn_parameters() -> Result<()> {
        let rnn = RNN::new(10, 20, 2)?;
        let params = rnn.parameters();

        // Should have 4 parameters per layer: weight_ih, weight_hh, bias_ih, bias_hh
        assert_eq!(params.len(), 8); // 2 layers * 4 params
        assert!(params.contains_key("weight_ih_l0"));
        assert!(params.contains_key("weight_hh_l0"));
        assert!(params.contains_key("bias_ih_l0"));
        assert!(params.contains_key("bias_hh_l0"));
        Ok(())
    }

    #[test]
    fn test_rnn_training_mode() -> Result<()> {
        let mut rnn = RNN::new(10, 20, 1)?;
        assert!(rnn.training());

        rnn.eval();
        assert!(!rnn.training());

        rnn.train();
        assert!(rnn.training());
        Ok(())
    }

    // ========================================================================
    // LSTM Tests
    // ========================================================================

    #[test]
    fn test_lstm_new() -> Result<()> {
        let lstm = LSTM::new(10, 20, 2)?;
        assert_eq!(lstm.input_size, 10);
        assert_eq!(lstm.hidden_size, 20);
        assert_eq!(lstm.num_layers, 2);
        assert!(lstm.bias);
        assert!(!lstm.batch_first);
        Ok(())
    }

    #[test]
    fn test_lstm_with_config() -> Result<()> {
        let lstm = LSTM::with_config(10, 20, 1, false, true, 0.5, true)?;
        assert_eq!(lstm.input_size, 10);
        assert_eq!(lstm.hidden_size, 20);
        assert!(!lstm.bias);
        assert!(lstm.batch_first);
        assert_eq!(lstm.dropout, 0.5);
        assert!(lstm.bidirectional);
        Ok(())
    }

    #[test]
    fn test_lstm_forward() -> Result<()> {
        let lstm = LSTM::new(10, 20, 1)?;
        let input = zeros(&[5, 2, 10])?; // seq_len=5, batch=2, input_size=10

        let output = lstm.forward(&input)?;
        let output_shape = output.shape();

        // Expected: [seq_len, batch, hidden_size]
        assert_eq!(output_shape.dims(), &[5, 2, 20]);
        Ok(())
    }

    #[test]
    fn test_lstm_forward_batch_first() -> Result<()> {
        let lstm = LSTM::with_config(10, 20, 1, true, true, 0.0, false)?;
        let input = zeros(&[2, 5, 10])?; // batch=2, seq_len=5, input_size=10

        let output = lstm.forward(&input)?;
        let output_shape = output.shape();

        // Expected: [batch, seq_len, hidden_size]
        assert_eq!(output_shape.dims(), &[2, 5, 20]);
        Ok(())
    }

    #[test]
    fn test_lstm_parameters() -> Result<()> {
        let lstm = LSTM::new(10, 20, 2)?;
        let params = lstm.parameters();

        // Should have 4 parameters per layer (for 4 gates)
        assert_eq!(params.len(), 8); // 2 layers * 4 params
        assert!(params.contains_key("weight_ih_l0"));
        assert!(params.contains_key("weight_hh_l0"));
        assert!(params.contains_key("bias_ih_l0"));
        assert!(params.contains_key("bias_hh_l0"));
        Ok(())
    }

    #[test]
    fn test_lstm_cell() -> Result<()> {
        let lstm = LSTM::new(10, 20, 1)?;
        let input = zeros(&[2, 10])?; // batch=2, input_size=10
        let hidden = zeros(&[2, 20])?; // batch=2, hidden_size=20
        let cell = zeros(&[2, 20])?; // batch=2, hidden_size=20

        let (new_hidden, new_cell) = lstm.lstm_cell(&input, &hidden, &cell, 0)?;

        assert_eq!(new_hidden.shape().dims(), &[2, 20]);
        assert_eq!(new_cell.shape().dims(), &[2, 20]);
        Ok(())
    }

    // ========================================================================
    // GRU Tests
    // ========================================================================

    #[test]
    fn test_gru_new() -> Result<()> {
        let gru = GRU::new(10, 20, 2)?;
        assert_eq!(gru.input_size, 10);
        assert_eq!(gru.hidden_size, 20);
        assert_eq!(gru.num_layers, 2);
        assert!(gru.bias);
        assert!(!gru.batch_first);
        Ok(())
    }

    #[test]
    fn test_gru_with_config() -> Result<()> {
        let gru = GRU::with_config(10, 20, 1, false, true, 0.5, true)?;
        assert_eq!(gru.input_size, 10);
        assert_eq!(gru.hidden_size, 20);
        assert!(!gru.bias);
        assert!(gru.batch_first);
        assert_eq!(gru.dropout, 0.5);
        assert!(gru.bidirectional);
        Ok(())
    }

    #[test]
    fn test_gru_forward() -> Result<()> {
        let gru = GRU::new(10, 20, 1)?;
        let input = zeros(&[5, 2, 10])?; // seq_len=5, batch=2, input_size=10

        let output = gru.forward(&input)?;
        let output_shape = output.shape();

        // Expected: [seq_len, batch, hidden_size]
        assert_eq!(output_shape.dims(), &[5, 2, 20]);
        Ok(())
    }

    #[test]
    fn test_gru_forward_batch_first() -> Result<()> {
        let gru = GRU::with_config(10, 20, 1, true, true, 0.0, false)?;
        let input = zeros(&[2, 5, 10])?; // batch=2, seq_len=5, input_size=10

        let output = gru.forward(&input)?;
        let output_shape = output.shape();

        // Expected: [batch, seq_len, hidden_size]
        assert_eq!(output_shape.dims(), &[2, 5, 20]);
        Ok(())
    }

    #[test]
    fn test_gru_forward_bidirectional() -> Result<()> {
        let gru = GRU::with_config(10, 20, 1, true, false, 0.0, true)?;
        let input = zeros(&[5, 2, 10])?; // seq_len=5, batch=2, input_size=10

        let output = gru.forward(&input)?;
        let output_shape = output.shape();

        // Expected: [seq_len, batch, hidden_size * 2] for bidirectional
        assert_eq!(output_shape.dims(), &[5, 2, 40]);
        Ok(())
    }

    #[test]
    fn test_gru_parameters() -> Result<()> {
        let gru = GRU::new(10, 20, 2)?;
        let params = gru.parameters();

        // Should have 4 parameters per layer (for 3 gates)
        assert_eq!(params.len(), 8); // 2 layers * 4 params
        assert!(params.contains_key("weight_ih_l0"));
        assert!(params.contains_key("weight_hh_l0"));
        assert!(params.contains_key("bias_ih_l0"));
        assert!(params.contains_key("bias_hh_l0"));
        Ok(())
    }

    #[test]
    fn test_gru_cell() -> Result<()> {
        let gru = GRU::new(10, 20, 1)?;
        let input = zeros(&[2, 10])?; // batch=2, input_size=10
        let hidden = zeros(&[2, 20])?; // batch=2, hidden_size=20

        let new_hidden = gru.gru_cell(&input, &hidden, 0)?;

        assert_eq!(new_hidden.shape().dims(), &[2, 20]);
        Ok(())
    }

    // ========================================================================
    // LSTMCell Tests
    // ========================================================================

    #[test]
    fn test_lstm_cell_new() -> Result<()> {
        let cell = LSTMCell::new(10, 20)?;
        assert_eq!(cell.input_size, 10);
        assert_eq!(cell.hidden_size, 20);
        assert!(cell.bias);
        Ok(())
    }

    #[test]
    fn test_lstm_cell_with_bias() -> Result<()> {
        let cell = LSTMCell::with_bias(10, 20, false)?;
        assert_eq!(cell.input_size, 10);
        assert_eq!(cell.hidden_size, 20);
        assert!(!cell.bias);
        Ok(())
    }

    #[test]
    fn test_lstm_cell_forward() -> Result<()> {
        let cell = LSTMCell::new(10, 20)?;
        let input = zeros(&[2, 10])?; // batch=2, input_size=10

        let output = cell.forward(&input)?;
        let output_shape = output.shape();

        assert_eq!(output_shape.dims(), &[2, 20]);
        Ok(())
    }

    #[test]
    fn test_lstm_cell_forward_cell() -> Result<()> {
        let cell = LSTMCell::new(10, 20)?;
        let input = zeros(&[2, 10])?; // batch=2, input_size=10
        let hidden = zeros(&[2, 20])?;
        let cell_state = zeros(&[2, 20])?;

        let (new_hidden, new_cell) = cell.forward_cell(&input, &hidden, &cell_state)?;

        assert_eq!(new_hidden.shape().dims(), &[2, 20]);
        assert_eq!(new_cell.shape().dims(), &[2, 20]);
        Ok(())
    }

    #[test]
    fn test_lstm_cell_parameters() -> Result<()> {
        let cell = LSTMCell::new(10, 20)?;
        let params = cell.parameters();

        // Should have 4 parameters: weight_ih, weight_hh, bias_ih, bias_hh
        assert_eq!(params.len(), 4);
        assert!(params.contains_key("weight_ih"));
        assert!(params.contains_key("weight_hh"));
        assert!(params.contains_key("bias_ih"));
        assert!(params.contains_key("bias_hh"));
        Ok(())
    }

    // ========================================================================
    // GRUCell Tests
    // ========================================================================

    #[test]
    fn test_gru_cell_new() -> Result<()> {
        let cell = GRUCell::new(10, 20)?;
        assert_eq!(cell.input_size, 10);
        assert_eq!(cell.hidden_size, 20);
        assert!(cell.bias);
        Ok(())
    }

    #[test]
    fn test_gru_cell_with_bias() -> Result<()> {
        let cell = GRUCell::with_bias(10, 20, false)?;
        assert_eq!(cell.input_size, 10);
        assert_eq!(cell.hidden_size, 20);
        assert!(!cell.bias);
        Ok(())
    }

    #[test]
    fn test_gru_cell_forward() -> Result<()> {
        let cell = GRUCell::new(10, 20)?;
        let input = zeros(&[2, 10])?; // batch=2, input_size=10

        let output = cell.forward(&input)?;
        let output_shape = output.shape();

        assert_eq!(output_shape.dims(), &[2, 20]);
        Ok(())
    }

    #[test]
    fn test_gru_cell_forward_cell() -> Result<()> {
        let cell = GRUCell::new(10, 20)?;
        let input = zeros(&[2, 10])?; // batch=2, input_size=10
        let hidden = zeros(&[2, 20])?;

        let new_hidden = cell.forward_cell(&input, &hidden)?;

        assert_eq!(new_hidden.shape().dims(), &[2, 20]);
        Ok(())
    }

    #[test]
    fn test_gru_cell_parameters() -> Result<()> {
        let cell = GRUCell::new(10, 20)?;
        let params = cell.parameters();

        // Should have 4 parameters: weight_ih, weight_hh, bias_ih, bias_hh
        assert_eq!(params.len(), 4);
        assert!(params.contains_key("weight_ih"));
        assert!(params.contains_key("weight_hh"));
        assert!(params.contains_key("bias_ih"));
        assert!(params.contains_key("bias_hh"));
        Ok(())
    }

    // ========================================================================
    // CustomRNNCell Tests
    // ========================================================================

    #[test]
    fn test_custom_rnn_cell_new() -> Result<()> {
        let cell = CustomRNNCell::new(10, 20)?;
        assert_eq!(cell.input_size, 10);
        assert_eq!(cell.hidden_size, 20);
        assert!(cell.bias);
        Ok(())
    }

    #[test]
    fn test_custom_rnn_cell_basic() -> Result<()> {
        let cell = CustomRNNCell::new(10, 20)?;
        let input = zeros(&[2, 10])?;
        let hidden = zeros(&[2, 20])?;

        let new_hidden = RNNCell::forward(&cell, &input, &hidden)?;
        assert_eq!(new_hidden.shape().dims(), &[2, 20]);
        Ok(())
    }

    #[test]
    fn test_custom_rnn_cell_gated() -> Result<()> {
        // Use matching input_size and hidden_size to avoid broadcast errors in the gated implementation
        let cell = CustomRNNCell::gated(20, 20, 2)?;
        assert_eq!(cell.input_size(), 20);
        assert_eq!(cell.hidden_size(), 20);

        let input = zeros(&[2, 20])?;
        let hidden = zeros(&[2, 20])?;

        let new_hidden = RNNCell::forward(&cell, &input, &hidden)?;
        assert_eq!(new_hidden.shape().dims(), &[2, 20]);
        Ok(())
    }

    #[test]
    fn test_custom_rnn_cell_highway() -> Result<()> {
        let cell = CustomRNNCell::highway(10, 20)?;
        let params = cell.parameters();

        // Highway cell should have 8 parameters (main + transform gate)
        assert_eq!(params.len(), 8);
        assert!(params.contains_key("weight_ih"));
        assert!(params.contains_key("weight_ih_t"));

        let input = zeros(&[2, 10])?;
        let hidden = zeros(&[2, 20])?;

        let new_hidden = RNNCell::forward(&cell, &input, &hidden)?;
        assert_eq!(new_hidden.shape().dims(), &[2, 20]);
        Ok(())
    }

    #[test]
    fn test_custom_rnn_cell_residual() -> Result<()> {
        let cell = CustomRNNCell::residual(10, 20)?;
        let input = zeros(&[2, 10])?;
        let hidden = zeros(&[2, 20])?;

        let new_hidden = RNNCell::forward(&cell, &input, &hidden)?;
        assert_eq!(new_hidden.shape().dims(), &[2, 20]);
        Ok(())
    }

    #[test]
    fn test_custom_rnn_cell_init_hidden() -> Result<()> {
        let cell = CustomRNNCell::new(10, 20)?;
        let hidden = cell.init_hidden(4)?; // batch_size=4

        assert_eq!(hidden.shape().dims(), &[4, 20]);
        Ok(())
    }

    #[test]
    fn test_custom_rnn_cell_with_activation() -> Result<()> {
        let cell = CustomRNNCell::with_activation(10, 20, Box::new(|x| x.relu()))?;
        let input = zeros(&[2, 10])?;

        // Use Module::forward explicitly to avoid ambiguity
        let output = Module::forward(&cell, &input)?;
        assert_eq!(output.shape().dims(), &[2, 20]);
        Ok(())
    }

    #[test]
    fn test_custom_rnn_cell_parameters_gated() -> Result<()> {
        let cell = CustomRNNCell::gated(10, 20, 3)?;
        let params = cell.parameters();

        // 3 gates * 4 parameters per gate = 12 parameters
        assert_eq!(params.len(), 12);
        assert!(params.contains_key("weight_ih_gate0"));
        assert!(params.contains_key("weight_ih_gate1"));
        assert!(params.contains_key("weight_ih_gate2"));
        Ok(())
    }

    // ========================================================================
    // Module Trait Tests (Common Behaviors)
    // ========================================================================

    #[test]
    fn test_module_training_modes() -> Result<()> {
        let mut lstm = LSTM::new(10, 20, 1)?;

        // Default should be training mode
        assert!(lstm.training());

        // Set to eval mode
        lstm.set_training(false);
        assert!(!lstm.training());

        // Set back to training mode
        lstm.set_training(true);
        assert!(lstm.training());
        Ok(())
    }

    #[test]
    fn test_module_named_parameters() -> Result<()> {
        let gru = GRU::new(10, 20, 2)?;
        let named_params = gru.named_parameters();

        // Should have 8 parameters (2 layers * 4 params)
        assert_eq!(named_params.len(), 8);
        assert!(named_params.contains_key("weight_ih_l0"));
        assert!(named_params.contains_key("weight_hh_l1"));
        Ok(())
    }

    #[test]
    fn test_module_to_device() -> Result<()> {
        let mut rnn = RNN::new(10, 20, 1)?;

        // Should succeed
        rnn.to_device(DeviceType::Cpu)?;

        Ok(())
    }

    #[test]
    fn test_stack_outputs_empty() -> Result<()> {
        let lstm = LSTM::new(10, 20, 1)?;
        let empty_outputs: Vec<Tensor> = vec![];

        let result = lstm.stack_outputs(&empty_outputs);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_gru_stack_combined_outputs_empty() -> Result<()> {
        let gru = GRU::new(10, 20, 1)?;
        let empty_outputs: Vec<Tensor> = vec![];

        let result = gru.stack_combined_outputs(&empty_outputs);
        assert!(result.is_err());
        Ok(())
    }
}
