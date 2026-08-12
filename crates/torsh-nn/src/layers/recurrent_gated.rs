//! Gated recurrent layers: LSTM, GRU, LSTMCell, GRUCell, CustomRNNCell
//!
//! This module is included from `recurrent.rs` via `#[path]` and re-exports
//! everything so callers see a flat namespace.

use super::*;

// ============================================================================
// Shared helpers
// ============================================================================

/// PyTorch parameter-name suffix for a direction (`""` forward, `"_reverse"`).
fn direction_suffix(direction: usize) -> &'static str {
    if direction == 0 {
        ""
    } else {
        "_reverse"
    }
}

/// Stack `[batch, features]` step tensors into `[steps, batch, features]`.
///
/// # Autograd
///
/// The step tensors are copied element-wise, so the result is a fresh leaf: the
/// stacked sequence is *not* connected to the cells that produced it. A
/// graph-preserving version needs a differentiable `Tensor::stack` in
/// torsh-tensor (`cat`/`stack` currently rebuild through `from_data`), which is
/// tracked as follow-up work; the recurrent graph is severed earlier anyway,
/// where `narrow` splits the gate pre-activations.
fn stack_time_major(steps: &[Tensor]) -> Result<Tensor> {
    if steps.is_empty() {
        return Err(torsh_core::TorshError::InvalidArgument(
            "No outputs to stack".to_string(),
        ));
    }

    let binding = steps[0].shape();
    let first = binding.dims();
    if first.len() != 2 {
        return Err(torsh_core::TorshError::InvalidShape(format!(
            "expected 2-D [batch, features] step tensors, got {}-D",
            first.len()
        )));
    }
    let batch_size = first[0];
    let features = first[1];

    let mut stacked_data = Vec::with_capacity(steps.len() * batch_size * features);
    for step in steps {
        if step.shape().dims() != [batch_size, features] {
            return Err(torsh_core::TorshError::ShapeMismatch {
                expected: vec![batch_size, features],
                got: step.shape().dims().to_vec(),
            });
        }
        stacked_data.extend(step.to_vec()?);
    }

    Tensor::from_vec(stacked_data, &[steps.len(), batch_size, features])
}

/// Concatenate two `[batch, features]` tensors along the feature axis.
fn concat_features(left: &Tensor, right: &Tensor) -> Result<Tensor> {
    let left_binding = left.shape();
    let left_shape = left_binding.dims();
    let right_binding = right.shape();
    let right_shape = right_binding.dims();

    if left_shape.len() != 2 || right_shape.len() != 2 || left_shape[0] != right_shape[0] {
        return Err(torsh_core::TorshError::ShapeMismatch {
            expected: left_shape.to_vec(),
            got: right_shape.to_vec(),
        });
    }

    let batch_size = left_shape[0];
    let left_features = left_shape[1];
    let right_features = right_shape[1];
    let left_data = left.to_vec()?;
    let right_data = right.to_vec()?;

    let mut combined = Vec::with_capacity(batch_size * (left_features + right_features));
    for batch in 0..batch_size {
        combined.extend_from_slice(&left_data[batch * left_features..(batch + 1) * left_features]);
        combined
            .extend_from_slice(&right_data[batch * right_features..(batch + 1) * right_features]);
    }

    Tensor::from_vec(combined, &[batch_size, left_features + right_features])
}

// ============================================================================
// LSTM
// ============================================================================

/// Multi-layer LSTM.
///
/// # PyTorch compatibility
///
/// Parameters follow `torch.nn.LSTM` naming: `weight_ih_l{k}`, `weight_hh_l{k}`,
/// `bias_ih_l{k}`, `bias_hh_l{k}`, with a `_reverse` suffix for the backward
/// direction of a bidirectional layer. Layer `k > 0` consumes the previous
/// layer's output, so its input size is `hidden_size * num_directions`.
/// Gate order inside the packed weights is input, forget, cell, output.
pub struct LSTM {
    pub(super) base: ModuleBase,
    pub(super) input_size: usize,
    pub(super) hidden_size: usize,
    pub(super) num_layers: usize,
    pub(super) bias: bool,
    pub(super) batch_first: bool,
    pub(super) dropout: f32,
    pub(super) bidirectional: bool,
}

impl LSTM {
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
                "LSTM requires at least one layer".to_string(),
            ));
        }

        let mut base = ModuleBase::new();
        let directions = if bidirectional { 2 } else { 1 };

        // Initialize weights for each layer and direction (4 gates: input, forget, cell, output)
        for layer in 0..num_layers {
            let layer_input = if layer == 0 {
                input_size
            } else {
                hidden_size * directions
            };

            for direction in 0..directions {
                let suffix = direction_suffix(direction);
                let weight_ih = crate::init::xavier_uniform(&[4 * hidden_size, layer_input])?;
                let weight_hh = crate::init::xavier_uniform(&[4 * hidden_size, hidden_size])?;

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
                        Parameter::new(zeros(&[4 * hidden_size])?),
                    );
                    base.register_parameter(
                        format!("bias_hh_l{}{}", layer, suffix),
                        Parameter::new(zeros(&[4 * hidden_size])?),
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
                torsh_core::TorshError::InvalidArgument(format!("LSTM is missing parameter {name}"))
            })?
            .tensor()
            .read()
            .clone())
    }

    /// Single LSTM cell computation for the forward direction of `layer`.
    pub(super) fn lstm_cell(
        &self,
        input: &Tensor,
        hidden: &Tensor,
        cell: &Tensor,
        layer: usize,
    ) -> Result<(Tensor, Tensor)> {
        self.lstm_cell_directional(input, hidden, cell, layer, 0)
    }

    /// Single LSTM cell computation for one layer and direction.
    fn lstm_cell_directional(
        &self,
        input: &Tensor,
        hidden: &Tensor,
        cell: &Tensor,
        layer: usize,
        direction: usize,
    ) -> Result<(Tensor, Tensor)> {
        let suffix = direction_suffix(direction);
        let weight_ih = self.parameter(&format!("weight_ih_l{}{}", layer, suffix))?;
        let weight_hh = self.parameter(&format!("weight_hh_l{}{}", layer, suffix))?;

        // Compute input and hidden transformations
        let mut gi = input.matmul(&weight_ih.transpose(0, 1)?)?;
        let mut gh = hidden.matmul(&weight_hh.transpose(0, 1)?)?;
        if self.bias {
            gi = gi.add_op(&self.parameter(&format!("bias_ih_l{}{}", layer, suffix))?)?;
            gh = gh.add_op(&self.parameter(&format!("bias_hh_l{}{}", layer, suffix))?)?;
        }
        let gates = gi.add_op(&gh)?;

        // Split into 4 gates
        let chunk_size = self.hidden_size;
        let input_gate = gates.narrow(1, 0, chunk_size)?.sigmoid()?;
        let forget_gate = gates.narrow(1, chunk_size as i64, chunk_size)?.sigmoid()?;
        let cell_gate = gates
            .narrow(1, (2 * chunk_size) as i64, chunk_size)?
            .tanh()?;
        let output_gate = gates
            .narrow(1, (3 * chunk_size) as i64, chunk_size)?
            .sigmoid()?;

        // Compute new cell and hidden states
        let new_cell = forget_gate
            .mul_op(cell)?
            .add_op(&input_gate.mul_op(&cell_gate)?)?;
        let new_hidden = output_gate.mul_op(&new_cell.tanh()?)?;

        Ok((new_hidden, new_cell))
    }

    /// Stack outputs from time steps
    pub(super) fn stack_outputs(&self, outputs: &[Tensor]) -> Result<Tensor> {
        stack_time_major(outputs)
    }

    /// Stack outputs for bidirectional case (hidden_size * 2)
    fn stack_combined_outputs(&self, outputs: &[Tensor]) -> Result<Tensor> {
        stack_time_major(outputs)
    }

    /// Run every layer (and direction) over a time-major input sequence.
    ///
    /// Returns the last layer's output sequence together with the final hidden
    /// and cell state of every (layer, direction) pair, in PyTorch order
    /// (`layer 0 forward, layer 0 reverse, layer 1 forward, ...`).
    fn run_layers(
        &self,
        input: &Tensor,
        state: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Vec<Tensor>, Vec<Tensor>)> {
        let binding = input.shape();
        let input_shape = binding.dims();
        let seq_len = input_shape[0];
        let batch_size = input_shape[1];
        let directions = self.directions();

        if input_shape[2] != self.input_size {
            return Err(torsh_core::TorshError::InvalidShape(format!(
                "LSTM expects an input of size {}, got {}",
                self.input_size, input_shape[2]
            )));
        }

        let (h0, c0) = match state {
            Some((h, c)) => {
                let expected = [self.num_layers * directions, batch_size, self.hidden_size];
                for (name, tensor) in [("h0", h), ("c0", c)] {
                    if tensor.shape().dims() != expected {
                        return Err(torsh_core::TorshError::InvalidShape(format!(
                            "LSTM {name} must have shape {:?}, got {:?}",
                            expected,
                            tensor.shape().dims()
                        )));
                    }
                }
                (Some(h.clone()), Some(c.clone()))
            }
            None => (None, None),
        };

        let mut layer_input = input.clone();
        let mut final_hidden = Vec::with_capacity(self.num_layers * directions);
        let mut final_cell = Vec::with_capacity(self.num_layers * directions);

        for layer in 0..self.num_layers {
            let mut direction_outputs: Vec<Vec<Tensor>> = Vec::with_capacity(directions);

            for direction in 0..directions {
                let state_index = layer * directions + direction;
                let mut hidden = match &h0 {
                    Some(h) => h.narrow(0, state_index as i64, 1)?.squeeze(0)?,
                    None => zeros(&[batch_size, self.hidden_size])?,
                };
                let mut cell = match &c0 {
                    Some(c) => c.narrow(0, state_index as i64, 1)?.squeeze(0)?,
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
                    let (new_hidden, new_cell) = if direction == 0 {
                        self.lstm_cell(&x_t, &hidden, &cell, layer)?
                    } else {
                        self.lstm_cell_directional(&x_t, &hidden, &cell, layer, direction)?
                    };
                    hidden = new_hidden;
                    cell = new_cell;
                    outputs.push(hidden.clone());
                }

                if direction == 1 {
                    // The reverse pass produced outputs from the last step
                    // backwards; restore time order before concatenating.
                    outputs.reverse();
                }

                final_hidden.push(hidden);
                final_cell.push(cell);
                direction_outputs.push(outputs);
            }

            // Concatenate the directions along the feature axis, then stack time.
            let mut steps = Vec::with_capacity(seq_len);
            for t in 0..seq_len {
                if directions == 1 {
                    steps.push(direction_outputs[0][t].clone());
                } else {
                    steps.push(concat_features(
                        &direction_outputs[0][t],
                        &direction_outputs[1][t],
                    )?);
                }
            }

            let mut stacked = if directions == 1 {
                self.stack_outputs(&steps)?
            } else {
                self.stack_combined_outputs(&steps)?
            };
            // Inter-layer dropout, exactly as PyTorch: applied to the output of
            // every layer except the last, and only while training.
            if layer + 1 < self.num_layers && self.dropout > 0.0 && self.base.training() {
                stacked = crate::functional::dropout(&stacked, self.dropout, true)?;
            }
            layer_input = stacked;
        }

        Ok((layer_input, final_hidden, final_cell))
    }

    /// Forward pass with explicit initial state.
    ///
    /// `state` is `(h_0, c_0)`, each shaped
    /// `[num_layers * num_directions, batch, hidden_size]`; `None` starts from
    /// zeros. Returns `(output, (h_n, c_n))` with the same layout conventions as
    /// `torch.nn.LSTM`, honouring [`Self::batch_first`] for `output`.
    pub fn forward_with_state(
        &self,
        input: &Tensor,
        state: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, (Tensor, Tensor))> {
        if input.shape().ndim() != 3 {
            return Err(torsh_core::TorshError::InvalidShape(format!(
                "LSTM expects a 3-D input, got {}-D",
                input.shape().ndim()
            )));
        }

        // Work time-major internally.
        let time_major = if self.batch_first {
            input.transpose(0, 1)?
        } else {
            input.clone()
        };

        let (output, hidden, cell) = self.run_layers(&time_major, state)?;
        let h_n = stack_time_major(&hidden)?;
        let c_n = stack_time_major(&cell)?;

        let output = if self.batch_first {
            output.transpose(0, 1)?
        } else {
            output
        };

        Ok((output, (h_n, c_n)))
    }
}

impl Module for LSTM {
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

impl std::fmt::Debug for LSTM {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LSTM")
            .field("input_size", &self.input_size)
            .field("hidden_size", &self.hidden_size)
            .field("num_layers", &self.num_layers)
            .finish()
    }
}

// ============================================================================
// GRU
// ============================================================================

/// Multi-layer GRU.
///
/// # PyTorch compatibility
///
/// Parameters follow `torch.nn.GRU` naming: `weight_ih_l{k}`, `weight_hh_l{k}`,
/// `bias_ih_l{k}`, `bias_hh_l{k}`, with a `_reverse` suffix for the backward
/// direction of a bidirectional layer. Gate order inside the packed weights is
/// reset, update, new.
pub struct GRU {
    pub(super) base: ModuleBase,
    pub(super) input_size: usize,
    pub(super) hidden_size: usize,
    pub(super) num_layers: usize,
    pub(super) bias: bool,
    pub(super) batch_first: bool,
    pub(super) dropout: f32,
    pub(super) bidirectional: bool,
}

impl GRU {
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
                "GRU requires at least one layer".to_string(),
            ));
        }

        let mut base = ModuleBase::new();
        let directions = if bidirectional { 2 } else { 1 };

        // Initialize weights for each layer and direction (3 gates: reset, update, new)
        for layer in 0..num_layers {
            let layer_input = if layer == 0 {
                input_size
            } else {
                hidden_size * directions
            };

            for direction in 0..directions {
                let suffix = direction_suffix(direction);
                let weight_ih = crate::init::xavier_uniform(&[3 * hidden_size, layer_input])?;
                let weight_hh = crate::init::xavier_uniform(&[3 * hidden_size, hidden_size])?;

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
                        Parameter::new(zeros(&[3 * hidden_size])?),
                    );
                    base.register_parameter(
                        format!("bias_hh_l{}{}", layer, suffix),
                        Parameter::new(zeros(&[3 * hidden_size])?),
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
                torsh_core::TorshError::InvalidArgument(format!("GRU is missing parameter {name}"))
            })?
            .tensor()
            .read()
            .clone())
    }

    /// Single GRU cell computation for the forward direction of `layer`.
    pub(super) fn gru_cell(&self, input: &Tensor, hidden: &Tensor, layer: usize) -> Result<Tensor> {
        self.gru_cell_directional(input, hidden, layer, 0)
    }

    /// Single GRU cell computation for one layer and direction.
    fn gru_cell_directional(
        &self,
        input: &Tensor,
        hidden: &Tensor,
        layer: usize,
        direction: usize,
    ) -> Result<Tensor> {
        let suffix = direction_suffix(direction);
        let weight_ih = self.parameter(&format!("weight_ih_l{}{}", layer, suffix))?;
        let weight_hh = self.parameter(&format!("weight_hh_l{}{}", layer, suffix))?;

        // Compute input and hidden transformations
        let mut gi = input.matmul(&weight_ih.transpose(0, 1)?)?;
        let mut gh = hidden.matmul(&weight_hh.transpose(0, 1)?)?;
        if self.bias {
            gi = gi.add_op(&self.parameter(&format!("bias_ih_l{}{}", layer, suffix))?)?;
            gh = gh.add_op(&self.parameter(&format!("bias_hh_l{}{}", layer, suffix))?)?;
        }

        // Split into 3 gates
        let chunk_size = self.hidden_size;

        // Reset gate and update gate use both input and hidden
        let i_reset = gi.narrow(1, 0, chunk_size)?;
        let h_reset = gh.narrow(1, 0, chunk_size)?;
        let reset_gate = (i_reset.add_op(&h_reset)?).sigmoid()?;

        let i_update = gi.narrow(1, chunk_size as i64, chunk_size)?;
        let h_update = gh.narrow(1, chunk_size as i64, chunk_size)?;
        let update_gate = (i_update.add_op(&h_update)?).sigmoid()?;

        // New gate uses input and reset-modulated hidden
        let i_new = gi.narrow(1, (2 * chunk_size) as i64, chunk_size)?;
        let h_new = gh.narrow(1, (2 * chunk_size) as i64, chunk_size)?;
        let reset_hidden = reset_gate.mul_op(&h_new)?;
        let new_gate = (i_new.add_op(&reset_hidden)?).tanh()?;

        // Compute new hidden state
        let one_minus_update = update_gate.mul_scalar(-1.0)?.add_scalar(1.0)?;
        let new_hidden = update_gate
            .mul_op(hidden)?
            .add_op(&one_minus_update.mul_op(&new_gate)?)?;

        Ok(new_hidden)
    }

    /// Stack outputs from time steps
    pub(super) fn stack_outputs(&self, outputs: &[Tensor]) -> Result<Tensor> {
        stack_time_major(outputs)
    }

    /// Stack outputs for bidirectional case (hidden_size * 2)
    pub(super) fn stack_combined_outputs(&self, outputs: &[Tensor]) -> Result<Tensor> {
        stack_time_major(outputs)
    }

    /// Run every layer (and direction) over a time-major input sequence.
    fn run_layers(&self, input: &Tensor, state: Option<&Tensor>) -> Result<(Tensor, Vec<Tensor>)> {
        let binding = input.shape();
        let input_shape = binding.dims();
        let seq_len = input_shape[0];
        let batch_size = input_shape[1];
        let directions = self.directions();

        if input_shape[2] != self.input_size {
            return Err(torsh_core::TorshError::InvalidShape(format!(
                "GRU expects an input of size {}, got {}",
                self.input_size, input_shape[2]
            )));
        }

        let h0 = match state {
            Some(h) => {
                let expected = [self.num_layers * directions, batch_size, self.hidden_size];
                if h.shape().dims() != expected {
                    return Err(torsh_core::TorshError::InvalidShape(format!(
                        "GRU h0 must have shape {:?}, got {:?}",
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
                    hidden = if direction == 0 {
                        self.gru_cell(&x_t, &hidden, layer)?
                    } else {
                        self.gru_cell_directional(&x_t, &hidden, layer, direction)?
                    };
                    outputs.push(hidden.clone());
                }

                if direction == 1 {
                    outputs.reverse();
                }

                final_hidden.push(hidden);
                direction_outputs.push(outputs);
            }

            let mut steps = Vec::with_capacity(seq_len);
            for t in 0..seq_len {
                if directions == 1 {
                    steps.push(direction_outputs[0][t].clone());
                } else {
                    steps.push(concat_features(
                        &direction_outputs[0][t],
                        &direction_outputs[1][t],
                    )?);
                }
            }

            let mut stacked = if directions == 1 {
                self.stack_outputs(&steps)?
            } else {
                self.stack_combined_outputs(&steps)?
            };
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
    /// `torch.nn.GRU`, honouring [`Self::batch_first`] for `output`.
    pub fn forward_with_state(
        &self,
        input: &Tensor,
        state: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        if input.shape().ndim() != 3 {
            return Err(torsh_core::TorshError::InvalidShape(format!(
                "GRU expects a 3-D input, got {}-D",
                input.shape().ndim()
            )));
        }

        let time_major = if self.batch_first {
            input.transpose(0, 1)?
        } else {
            input.clone()
        };

        let (output, hidden) = self.run_layers(&time_major, state)?;
        let h_n = stack_time_major(&hidden)?;

        let output = if self.batch_first {
            output.transpose(0, 1)?
        } else {
            output
        };

        Ok((output, h_n))
    }
}

impl Module for GRU {
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

impl std::fmt::Debug for GRU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GRU")
            .field("input_size", &self.input_size)
            .field("hidden_size", &self.hidden_size)
            .field("num_layers", &self.num_layers)
            .finish()
    }
}

// ============================================================================
// LSTMCell
// ============================================================================

/// LSTM Cell - processes a single time step
pub struct LSTMCell {
    pub(super) base: ModuleBase,
    pub(super) input_size: usize,
    pub(super) hidden_size: usize,
    pub(super) bias: bool,
}

impl LSTMCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Result<Self> {
        let mut base = ModuleBase::new();

        let weight_ih = crate::init::xavier_uniform(&[4 * hidden_size, input_size])?;
        let weight_hh = crate::init::xavier_uniform(&[4 * hidden_size, hidden_size])?;
        let bias_ih = zeros(&[4 * hidden_size])?;
        let bias_hh = zeros(&[4 * hidden_size])?;

        base.register_parameter("weight_ih".to_string(), Parameter::new(weight_ih));
        base.register_parameter("weight_hh".to_string(), Parameter::new(weight_hh));
        base.register_parameter("bias_ih".to_string(), Parameter::new(bias_ih));
        base.register_parameter("bias_hh".to_string(), Parameter::new(bias_hh));

        Ok(Self {
            base,
            input_size,
            hidden_size,
            bias: true,
        })
    }

    pub fn with_bias(input_size: usize, hidden_size: usize, bias: bool) -> Result<Self> {
        let mut cell = Self::new(input_size, hidden_size)?;
        cell.bias = bias;
        Ok(cell)
    }

    /// Forward pass returning (h_new, c_new)
    pub fn forward_cell(
        &self,
        input: &Tensor,
        hidden: &Tensor,
        cell: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let weight_ih = self.base.parameters["weight_ih"].tensor().read().clone();
        let weight_hh = self.base.parameters["weight_hh"].tensor().read().clone();

        let mut gi = input.matmul(&weight_ih.transpose(0, 1)?)?;
        let mut gh = hidden.matmul(&weight_hh.transpose(0, 1)?)?;

        if self.bias {
            let bias_ih = self.base.parameters["bias_ih"].tensor().read().clone();
            let bias_hh = self.base.parameters["bias_hh"].tensor().read().clone();
            gi = gi.add_op(&bias_ih)?;
            gh = gh.add_op(&bias_hh)?;
        }

        let gates = gi.add_op(&gh)?;

        let chunk_size = self.hidden_size;
        let input_gate = gates.narrow(1, 0, chunk_size)?.sigmoid()?;
        let forget_gate = gates.narrow(1, chunk_size as i64, chunk_size)?.sigmoid()?;
        let cell_gate = gates
            .narrow(1, (2 * chunk_size) as i64, chunk_size)?
            .tanh()?;
        let output_gate = gates
            .narrow(1, (3 * chunk_size) as i64, chunk_size)?
            .sigmoid()?;

        let new_cell = forget_gate
            .mul_op(cell)?
            .add_op(&input_gate.mul_op(&cell_gate)?)?;
        let new_hidden = output_gate.mul_op(&new_cell.tanh()?)?;

        Ok((new_hidden, new_cell))
    }
}

impl Module for LSTMCell {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let batch_size = input.shape().dims()[0];
        let hidden = zeros(&[batch_size, self.hidden_size])?;
        let cell = zeros(&[batch_size, self.hidden_size])?;
        let (new_hidden, _) = self.forward_cell(input, &hidden, &cell)?;
        Ok(new_hidden)
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

impl std::fmt::Debug for LSTMCell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LSTMCell")
            .field("input_size", &self.input_size)
            .field("hidden_size", &self.hidden_size)
            .field("bias", &self.bias)
            .finish()
    }
}

// ============================================================================
// GRUCell
// ============================================================================

/// GRU Cell - processes a single time step
pub struct GRUCell {
    pub(super) base: ModuleBase,
    pub(super) input_size: usize,
    pub(super) hidden_size: usize,
    pub(super) bias: bool,
}

impl GRUCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Result<Self> {
        let mut base = ModuleBase::new();

        let weight_ih = crate::init::xavier_uniform(&[3 * hidden_size, input_size])?;
        let weight_hh = crate::init::xavier_uniform(&[3 * hidden_size, hidden_size])?;
        let bias_ih = zeros(&[3 * hidden_size])?;
        let bias_hh = zeros(&[3 * hidden_size])?;

        base.register_parameter("weight_ih".to_string(), Parameter::new(weight_ih));
        base.register_parameter("weight_hh".to_string(), Parameter::new(weight_hh));
        base.register_parameter("bias_ih".to_string(), Parameter::new(bias_ih));
        base.register_parameter("bias_hh".to_string(), Parameter::new(bias_hh));

        Ok(Self {
            base,
            input_size,
            hidden_size,
            bias: true,
        })
    }

    pub fn with_bias(input_size: usize, hidden_size: usize, bias: bool) -> Result<Self> {
        let mut cell = Self::new(input_size, hidden_size)?;
        cell.bias = bias;
        Ok(cell)
    }

    /// Forward pass returning new hidden state
    pub fn forward_cell(&self, input: &Tensor, hidden: &Tensor) -> Result<Tensor> {
        let weight_ih = self.base.parameters["weight_ih"].tensor().read().clone();
        let weight_hh = self.base.parameters["weight_hh"].tensor().read().clone();

        let mut gi = input.matmul(&weight_ih.transpose(0, 1)?)?;
        let mut gh = hidden.matmul(&weight_hh.transpose(0, 1)?)?;

        if self.bias {
            let bias_ih = self.base.parameters["bias_ih"].tensor().read().clone();
            let bias_hh = self.base.parameters["bias_hh"].tensor().read().clone();
            gi = gi.add_op(&bias_ih)?;
            gh = gh.add_op(&bias_hh)?;
        }

        let chunk_size = self.hidden_size;

        let i_reset = gi.narrow(1, 0, chunk_size)?;
        let h_reset = gh.narrow(1, 0, chunk_size)?;
        let reset_gate = (i_reset.add_op(&h_reset)?).sigmoid()?;

        let i_update = gi.narrow(1, chunk_size as i64, chunk_size)?;
        let h_update = gh.narrow(1, chunk_size as i64, chunk_size)?;
        let update_gate = (i_update.add_op(&h_update)?).sigmoid()?;

        let i_new = gi.narrow(1, (2 * chunk_size) as i64, chunk_size)?;
        let h_new = gh.narrow(1, (2 * chunk_size) as i64, chunk_size)?;
        let reset_hidden = reset_gate.mul_op(&h_new)?;
        let new_gate = (i_new.add_op(&reset_hidden)?).tanh()?;

        let one_minus_update = update_gate.mul_scalar(-1.0)?.add_scalar(1.0)?;
        let new_hidden = update_gate
            .mul_op(hidden)?
            .add_op(&one_minus_update.mul_op(&new_gate)?)?;

        Ok(new_hidden)
    }
}

impl Module for GRUCell {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let batch_size = input.shape().dims()[0];
        let hidden = zeros(&[batch_size, self.hidden_size])?;
        self.forward_cell(input, &hidden)
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

impl std::fmt::Debug for GRUCell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GRUCell")
            .field("input_size", &self.input_size)
            .field("hidden_size", &self.hidden_size)
            .field("bias", &self.bias)
            .finish()
    }
}

// ============================================================================
// RNNCell trait + CustomRNNCell
// ============================================================================

/// Trait for custom RNN cell implementations
pub trait RNNCell {
    /// Apply the cell computation for a single time step
    fn forward(&self, input: &Tensor, hidden: &Tensor) -> Result<Tensor>;

    /// Get the hidden size of the cell
    fn hidden_size(&self) -> usize;

    /// Get the input size of the cell
    fn input_size(&self) -> usize;

    /// Initialize hidden state with proper shape
    fn init_hidden(&self, batch_size: usize) -> Result<Tensor> {
        Ok(zeros(&[batch_size, self.hidden_size()])?)
    }
}

/// Custom RNN cell activation function type
pub type ActivationFn = Box<dyn Fn(&Tensor) -> Result<Tensor> + Send + Sync>;

/// Custom RNN cell with configurable activation and computation
pub struct CustomRNNCell {
    pub(super) base: ModuleBase,
    pub(super) input_size: usize,
    pub(super) hidden_size: usize,
    pub(super) bias: bool,
    pub(super) activation: ActivationFn,
    pub(super) cell_type: CustomCellType,
}

/// Types of custom RNN cells
#[derive(Clone)]
pub enum CustomCellType {
    /// Basic RNN cell: h_t = activation(W_ih @ x_t + W_hh @ h_{t-1} + b)
    Basic,
    /// Gated cell with configurable gates: similar to GRU but with custom gating
    Gated {
        num_gates: usize,
        gate_activation: String, // "sigmoid", "tanh", "relu", etc.
    },
    /// Highway cell: h_t = T * activation(W_ih @ x_t + W_hh @ h_{t-1}) + (1-T) * h_{t-1}
    Highway,
    /// Residual cell: h_t = activation(W_ih @ x_t + W_hh @ h_{t-1}) + h_{t-1}
    Residual,
}

impl CustomRNNCell {
    /// Create a new custom RNN cell with basic activation
    pub fn new(input_size: usize, hidden_size: usize) -> Result<Self> {
        Self::with_activation(input_size, hidden_size, Box::new(|x| x.tanh()))
    }

    /// Create a custom RNN cell with specified activation function
    pub fn with_activation(
        input_size: usize,
        hidden_size: usize,
        activation: ActivationFn,
    ) -> Result<Self> {
        let mut base = ModuleBase::new();

        // Initialize weights
        let weight_ih = crate::init::xavier_uniform(&[hidden_size, input_size])?;
        let weight_hh = crate::init::xavier_uniform(&[hidden_size, hidden_size])?;
        let bias_ih = zeros(&[hidden_size])?;
        let bias_hh = zeros(&[hidden_size])?;

        base.register_parameter("weight_ih".to_string(), Parameter::new(weight_ih));
        base.register_parameter("weight_hh".to_string(), Parameter::new(weight_hh));
        base.register_parameter("bias_ih".to_string(), Parameter::new(bias_ih));
        base.register_parameter("bias_hh".to_string(), Parameter::new(bias_hh));

        Ok(Self {
            base,
            input_size,
            hidden_size,
            bias: true,
            activation,
            cell_type: CustomCellType::Basic,
        })
    }

    /// Create a gated custom RNN cell
    pub fn gated(input_size: usize, hidden_size: usize, num_gates: usize) -> Result<Self> {
        let mut base = ModuleBase::new();

        // Initialize weights for gates
        for gate in 0..num_gates {
            let weight_ih = crate::init::xavier_uniform(&[hidden_size, input_size])?;
            let weight_hh = crate::init::xavier_uniform(&[hidden_size, hidden_size])?;
            let bias_ih = zeros(&[hidden_size])?;
            let bias_hh = zeros(&[hidden_size])?;

            base.register_parameter(format!("weight_ih_gate{}", gate), Parameter::new(weight_ih));
            base.register_parameter(format!("weight_hh_gate{}", gate), Parameter::new(weight_hh));
            base.register_parameter(format!("bias_ih_gate{}", gate), Parameter::new(bias_ih));
            base.register_parameter(format!("bias_hh_gate{}", gate), Parameter::new(bias_hh));
        }

        Ok(Self {
            base,
            input_size,
            hidden_size,
            bias: true,
            activation: Box::new(|x| x.tanh()),
            cell_type: CustomCellType::Gated {
                num_gates,
                gate_activation: "sigmoid".to_string(),
            },
        })
    }

    /// Create a highway RNN cell
    pub fn highway(input_size: usize, hidden_size: usize) -> Result<Self> {
        let mut base = ModuleBase::new();

        // Main transformation weights
        let weight_ih = crate::init::xavier_uniform(&[hidden_size, input_size])?;
        let weight_hh = crate::init::xavier_uniform(&[hidden_size, hidden_size])?;
        let bias_ih = zeros(&[hidden_size])?;
        let bias_hh = zeros(&[hidden_size])?;

        // Transform gate weights (T in the highway equation)
        let weight_ih_t = crate::init::xavier_uniform(&[hidden_size, input_size])?;
        let weight_hh_t = crate::init::xavier_uniform(&[hidden_size, hidden_size])?;
        let bias_ih_t = ones(&[hidden_size])?.mul_scalar(-1.0)?; // Initialize to favor carrying
        let bias_hh_t = zeros(&[hidden_size])?;

        base.register_parameter("weight_ih".to_string(), Parameter::new(weight_ih));
        base.register_parameter("weight_hh".to_string(), Parameter::new(weight_hh));
        base.register_parameter("bias_ih".to_string(), Parameter::new(bias_ih));
        base.register_parameter("bias_hh".to_string(), Parameter::new(bias_hh));

        base.register_parameter("weight_ih_t".to_string(), Parameter::new(weight_ih_t));
        base.register_parameter("weight_hh_t".to_string(), Parameter::new(weight_hh_t));
        base.register_parameter("bias_ih_t".to_string(), Parameter::new(bias_ih_t));
        base.register_parameter("bias_hh_t".to_string(), Parameter::new(bias_hh_t));

        Ok(Self {
            base,
            input_size,
            hidden_size,
            bias: true,
            activation: Box::new(|x| x.tanh()),
            cell_type: CustomCellType::Highway,
        })
    }

    /// Create a residual RNN cell
    pub fn residual(input_size: usize, hidden_size: usize) -> Result<Self> {
        let mut cell = Self::new(input_size, hidden_size)?;
        cell.cell_type = CustomCellType::Residual;
        Ok(cell)
    }

    /// Set the activation function
    pub fn with_activation_fn(mut self, activation: ActivationFn) -> Self {
        self.activation = activation;
        self
    }
}

impl RNNCell for CustomRNNCell {
    fn forward(&self, input: &Tensor, hidden: &Tensor) -> Result<Tensor> {
        match &self.cell_type {
            CustomCellType::Basic => {
                let weight_ih = self.base.parameters["weight_ih"].tensor().read().clone();
                let weight_hh = self.base.parameters["weight_hh"].tensor().read().clone();
                let bias_ih = self.base.parameters["bias_ih"].tensor().read().clone();
                let bias_hh = self.base.parameters["bias_hh"].tensor().read().clone();

                let gi = input.matmul(&weight_ih.transpose(0, 1)?)?;
                let gh = hidden.matmul(&weight_hh.transpose(0, 1)?)?;

                let gi = gi.add_op(&bias_ih)?;
                let gh = gh.add_op(&bias_hh)?;

                let new_h = gi.add_op(&gh)?;
                (self.activation)(&new_h)
            }

            CustomCellType::Gated {
                num_gates,
                gate_activation,
            } => {
                let mut gates = Vec::new();

                // Compute all gates
                for gate in 0..*num_gates {
                    let weight_ih = self.base.parameters[&format!("weight_ih_gate{}", gate)]
                        .tensor()
                        .read()
                        .clone();
                    let weight_hh = self.base.parameters[&format!("weight_hh_gate{}", gate)]
                        .tensor()
                        .read()
                        .clone();
                    let bias_ih = self.base.parameters[&format!("bias_ih_gate{}", gate)]
                        .tensor()
                        .read()
                        .clone();
                    let bias_hh = self.base.parameters[&format!("bias_hh_gate{}", gate)]
                        .tensor()
                        .read()
                        .clone();

                    let gi = input.matmul(&weight_ih.transpose(0, 1)?)?;
                    let gh = hidden.matmul(&weight_hh.transpose(0, 1)?)?;
                    let gi = gi.add_op(&bias_ih)?;
                    let gh = gh.add_op(&bias_hh)?;
                    let gate_val = gi.add_op(&gh)?;

                    let activated_gate = match gate_activation.as_str() {
                        "sigmoid" => gate_val.sigmoid()?,
                        "tanh" => gate_val.tanh()?,
                        "relu" => gate_val.relu()?,
                        _ => gate_val.sigmoid()?, // default to sigmoid
                    };
                    gates.push(activated_gate);
                }

                // Simple gated computation: use first gate as forget, second as input
                if gates.len() >= 2 {
                    let forget_gate = &gates[0];
                    let input_gate = &gates[1];

                    let candidate = (self.activation)(input)?;
                    let new_h = hidden
                        .mul_op(forget_gate)?
                        .add_op(&candidate.mul_op(input_gate)?)?;
                    Ok(new_h)
                } else {
                    // Single gate - just use it as input gate
                    let gate = &gates[0];
                    let candidate = (self.activation)(input)?;
                    Ok(hidden.add_op(&candidate.mul_op(gate)?)?)
                }
            }

            CustomCellType::Highway => {
                let weight_ih = self.base.parameters["weight_ih"].tensor().read().clone();
                let weight_hh = self.base.parameters["weight_hh"].tensor().read().clone();
                let bias_ih = self.base.parameters["bias_ih"].tensor().read().clone();
                let bias_hh = self.base.parameters["bias_hh"].tensor().read().clone();

                let weight_ih_t = self.base.parameters["weight_ih_t"].tensor().read().clone();
                let weight_hh_t = self.base.parameters["weight_hh_t"].tensor().read().clone();
                let bias_ih_t = self.base.parameters["bias_ih_t"].tensor().read().clone();
                let bias_hh_t = self.base.parameters["bias_hh_t"].tensor().read().clone();

                // Main transformation: h_candidate = activation(W_ih @ x + W_hh @ h + b)
                let gi = input.matmul(&weight_ih.transpose(0, 1)?)?;
                let gh = hidden.matmul(&weight_hh.transpose(0, 1)?)?;
                let gi = gi.add_op(&bias_ih)?;
                let gh = gh.add_op(&bias_hh)?;
                let h_candidate = (self.activation)(&gi.add_op(&gh)?)?;

                // Transform gate: T = sigmoid(W_ih_t @ x + W_hh_t @ h + b_t)
                let gi_t = input.matmul(&weight_ih_t.transpose(0, 1)?)?;
                let gh_t = hidden.matmul(&weight_hh_t.transpose(0, 1)?)?;
                let gi_t = gi_t.add_op(&bias_ih_t)?;
                let gh_t = gh_t.add_op(&bias_hh_t)?;
                let transform_gate = gi_t.add_op(&gh_t)?.sigmoid()?;

                // Highway connection: h_new = T * h_candidate + (1 - T) * h_prev
                let carry_gate = transform_gate.neg()?.add_scalar(1.0)?;
                let new_h = h_candidate
                    .mul_op(&transform_gate)?
                    .add_op(&hidden.mul_op(&carry_gate)?)?;
                Ok(new_h)
            }

            CustomCellType::Residual => {
                let weight_ih = self.base.parameters["weight_ih"].tensor().read().clone();
                let weight_hh = self.base.parameters["weight_hh"].tensor().read().clone();
                let bias_ih = self.base.parameters["bias_ih"].tensor().read().clone();
                let bias_hh = self.base.parameters["bias_hh"].tensor().read().clone();

                let gi = input.matmul(&weight_ih.transpose(0, 1)?)?;
                let gh = hidden.matmul(&weight_hh.transpose(0, 1)?)?;
                let gi = gi.add_op(&bias_ih)?;
                let gh = gh.add_op(&bias_hh)?;

                let h_candidate = (self.activation)(&gi.add_op(&gh)?)?;
                // Residual connection: h_new = h_candidate + h_prev
                Ok(h_candidate.add_op(hidden)?)
            }
        }
    }

    fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn input_size(&self) -> usize {
        self.input_size
    }
}

impl Module for CustomRNNCell {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // For single step forward, we need to provide a hidden state
        // This is a simplified version - in practice, users should call the RNNCell::forward directly
        let batch_size = input.shape().dims()[0];
        let hidden = self.init_hidden(batch_size)?;
        RNNCell::forward(self, input, &hidden)
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

impl std::fmt::Debug for CustomRNNCell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomRNNCell")
            .field("input_size", &self.input_size)
            .field("hidden_size", &self.hidden_size)
            .field("bias", &self.bias)
            .field("cell_type", &format!("{:?}", self.cell_type))
            .finish()
    }
}

impl std::fmt::Debug for CustomCellType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CustomCellType::Basic => write!(f, "Basic"),
            CustomCellType::Gated {
                num_gates,
                gate_activation,
            } => {
                write!(
                    f,
                    "Gated(gates: {}, activation: {})",
                    num_gates, gate_activation
                )
            }
            CustomCellType::Highway => write!(f, "Highway"),
            CustomCellType::Residual => write!(f, "Residual"),
        }
    }
}
