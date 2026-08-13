//! Deep learning models for time series forecasting

use crate::TimeSeries;
use torsh_core::error::Result;
use torsh_nn::{
    layers::{
        conv::Conv1d,
        linear::Linear,
        recurrent::{GRU, LSTM},
        regularization::Dropout,
    },
    Module, Parameter,
};
use torsh_tensor::Tensor;

/// Maximum L2 norm of a single [`LSTMForecaster::fit`] update.
///
/// The bound is on the *step* (`learning_rate * gradient`), not on the raw
/// gradient, so the same limit holds however the loss is scaled.
pub const LSTM_GRAD_CLIP_NORM: f32 = 5.0;

/// LSTM-based time series forecaster
pub struct LSTMForecaster {
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    sequence_length: usize,
    dropout_rate: f32,
    lstm: LSTM,
    dropout: Option<Dropout>,
    output_layer: Linear,
}

impl LSTMForecaster {
    /// Create a new LSTM forecaster
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Result<Self> {
        let lstm = torsh_nn::layers::recurrent::LSTM::with_config(
            input_size,
            hidden_size,
            num_layers,
            true,  // bias
            true,  // batch_first
            0.0,   // dropout
            false, // bidirectional
        )?;
        let output_layer = Linear::new(hidden_size, 1, true);

        Ok(Self {
            input_size,
            hidden_size,
            num_layers,
            sequence_length: 10,
            dropout_rate: 0.0,
            lstm,
            dropout: None,
            output_layer,
        })
    }

    /// Set sequence length for training
    pub fn with_sequence_length(mut self, seq_len: usize) -> Self {
        self.sequence_length = seq_len;
        self
    }

    /// Set dropout rate
    pub fn with_dropout(mut self, dropout_rate: f32) -> Self {
        self.dropout_rate = dropout_rate;
        if dropout_rate > 0.0 {
            self.dropout = Some(Dropout::new(dropout_rate));
        } else {
            self.dropout = None;
        }
        self
    }

    /// Get model parameters
    pub fn params(&self) -> (usize, usize, usize, usize, f32) {
        (
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.sequence_length,
            self.dropout_rate,
        )
    }

    /// Every trainable parameter of the forecaster, in a deterministic order.
    ///
    /// Names are namespaced (`lstm.weight_ih_l0`, `readout.weight`, …) because
    /// the recurrent layer and the linear read-out both use unqualified
    /// parameter names. The order is lexicographic, so an update that folds all
    /// parameters together (gradient clipping, norm reporting) is reproducible.
    pub fn trainable_parameters(&self) -> Vec<(String, Parameter)> {
        let mut named: Vec<(String, Parameter)> = self
            .lstm
            .parameters()
            .into_iter()
            .map(|(name, parameter)| (format!("lstm.{name}"), parameter))
            .chain(
                self.output_layer
                    .parameters()
                    .into_iter()
                    .map(|(name, parameter)| (format!("readout.{name}"), parameter)),
            )
            .collect();
        named.sort_by(|left, right| left.0.cmp(&right.0));
        named
    }

    /// Forward pass
    ///
    /// Routes the input through the real LSTM layer, extracts the hidden state
    /// at the final time step, optionally applies dropout, and projects it
    /// through the linear output layer to produce one prediction per sequence.
    ///
    /// The result stays connected to the autograd graph, so
    /// `prediction.backward()` reaches every parameter reported by
    /// [`trainable_parameters`](Self::trainable_parameters).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_core(x, self.dropout.as_ref())
    }

    /// Shared forward body.
    ///
    /// `dropout` is `None` during training, which is what makes the loss
    /// [`fit`](Self::fit) reports comparable across epochs (a resampled mask
    /// would add noise to the reported convergence curve).
    fn forward_core(&self, x: &Tensor, dropout: Option<&Dropout>) -> Result<Tensor> {
        // Input shape: [batch_size, seq_len, input_size] (LSTM is batch_first)
        let dims_binding = x.shape();
        let dims = dims_binding.dims();
        if dims.len() != 3 {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "LSTMForecaster::forward expects a 3D [batch, seq_len, input_size] tensor, got {dims:?}"
            )));
        }
        let seq_len = dims[1];
        if seq_len == 0 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "LSTMForecaster::forward requires seq_len >= 1".to_string(),
            ));
        }

        // Pass through LSTM → [batch_size, seq_len, hidden_size] (batch_first).
        let lstm_out = self.lstm.forward(x)?;

        // Apply dropout if specified.
        let lstm_out = match dropout {
            Some(dropout) => dropout.forward(&lstm_out)?,
            None => lstm_out,
        };

        // Take the hidden state at the last time step:
        // [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size].
        let last_hidden = lstm_out.narrow(1, (seq_len - 1) as i64, 1)?.squeeze(1)?;

        // Project through the linear output layer → [batch_size, 1].
        let prediction = self.output_layer.forward(&last_hidden)?;
        Ok(prediction)
    }

    /// Train the model on `series` with gradient descent on the mean squared error.
    ///
    /// The series is windowed into `(input, target)` pairs of length
    /// [`sequence_length`](Self::with_sequence_length); every epoch runs a full
    /// forward pass through the LSTM recurrence and the linear read-out, calls
    /// [`Tensor::backward`] to backpropagate the error through time, and updates
    /// every parameter reported by [`trainable_parameters`](Self::trainable_parameters) in place, so
    /// subsequent [`forward`](Self::forward) and [`forecast`](Self::forecast)
    /// calls use the trained weights.
    ///
    /// Training runs without dropout so that the reported loss curve measures
    /// the model rather than the mask. Every layer is trained, including the
    /// deeper layers of a multi-layer forecaster, and the update is rescaled
    /// whenever its L2 norm would exceed [`LSTM_GRAD_CLIP_NORM`].
    ///
    /// # Arguments
    ///
    /// * `series` - training series (must be longer than `sequence_length`)
    /// * `epochs` - number of full passes over the training windows
    /// * `learning_rate` - gradient-descent step size
    ///
    /// # Returns
    ///
    /// The mean squared error measured at the start of each epoch, so callers
    /// can verify convergence.
    ///
    /// # Errors
    ///
    /// Returns an error if the series is too short for a single window, if the
    /// hyper-parameters are invalid, if a parameter turns out to be unreachable
    /// from the loss, or if training diverges to a non-finite loss.
    pub fn fit(
        &mut self,
        series: &TimeSeries,
        epochs: usize,
        learning_rate: f32,
    ) -> Result<Vec<f32>> {
        if epochs == 0 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "LSTMForecaster::fit requires at least one epoch".to_string(),
            ));
        }
        if !(learning_rate > 0.0) || !learning_rate.is_finite() {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "LSTMForecaster::fit requires a positive, finite learning rate, got {learning_rate}"
            )));
        }

        let (inputs, targets) = self.create_sequences(series)?;
        let sample_count = targets.numel();
        if sample_count == 0 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "LSTMForecaster::fit requires at least one training window".to_string(),
            ));
        }

        let trainable = self.trainable_parameters();
        // Every parameter must be a grad-tracking leaf *before* the first
        // forward pass, and must carry no gradient left over from an earlier
        // call: `backward` accumulates into the slot rather than replacing it.
        for (_, parameter) in &trainable {
            let handle = parameter.tensor();
            let mut guard = handle.write();
            guard.zero_grad();
            if !guard.requires_grad() {
                let tracked = guard.clone().requires_grad_(true);
                *guard = tracked;
            }
        }

        let mut history = Vec::with_capacity(epochs);
        for _ in 0..epochs {
            let predictions = self.forward_core(&inputs, None)?;
            let residual = predictions.sub(&targets)?;
            let loss = residual
                .mul(&residual)?
                .sum()?
                .div_scalar(sample_count as f32)?;

            let loss_value = loss.get_item_flat(0)?;
            if !loss_value.is_finite() {
                return Err(torsh_core::error::TorshError::ComputeError(format!(
                    "LSTM training diverged: loss is {loss_value}"
                )));
            }
            history.push(loss_value);

            loss.backward()?;

            // Snapshot every gradient *before* any write-back: an update
            // replaces the parameter tensor, and with it the gradient slot the
            // remaining reads would otherwise observe.
            let gradients = Self::snapshot_gradients(&trainable)?;
            let step = learning_rate * Self::clip_scale(&gradients, learning_rate)?;
            Self::apply_update(&trainable, &gradients, step)?;
        }

        Ok(history)
    }

    /// Read the gradient of every trainable parameter into a flat buffer.
    fn snapshot_gradients(trainable: &[(String, Parameter)]) -> Result<Vec<Vec<f32>>> {
        let mut gradients = Vec::with_capacity(trainable.len());
        for (name, parameter) in trainable {
            let handle = parameter.tensor();
            let guard = handle.read();
            let gradient = guard.grad().ok_or_else(|| {
                torsh_core::error::TorshError::ComputeError(format!(
                    "LSTM training: parameter `{name}` is not reachable from the loss, \
                     so backpropagation produced no gradient for it"
                ))
            })?;
            let values = gradient.to_vec()?;
            if values.len() != guard.numel() {
                return Err(torsh_core::error::TorshError::ShapeMismatch {
                    expected: guard.shape().dims().to_vec(),
                    got: gradient.shape().dims().to_vec(),
                });
            }
            gradients.push(values);
        }
        Ok(gradients)
    }

    /// Rescaling factor that keeps the L2 norm of one update at or below
    /// [`LSTM_GRAD_CLIP_NORM`].
    fn clip_scale(gradients: &[Vec<f32>], learning_rate: f32) -> Result<f32> {
        let mut norm_squared = 0.0f64;
        for gradient in gradients {
            for value in gradient {
                norm_squared += f64::from(*value) * f64::from(*value);
            }
        }
        let norm = (norm_squared.sqrt() as f32) * learning_rate;
        if !norm.is_finite() {
            return Err(torsh_core::error::TorshError::ComputeError(format!(
                "LSTM training diverged: the gradient norm is {norm}"
            )));
        }
        if norm > LSTM_GRAD_CLIP_NORM {
            Ok(LSTM_GRAD_CLIP_NORM / norm)
        } else {
            Ok(1.0)
        }
    }

    /// Apply one gradient-descent step to every trainable parameter.
    ///
    /// `Tensor::from_vec` yields a detached tensor, so `requires_grad` is
    /// re-applied on every write-back; without it the next epoch's forward pass
    /// would silently stop recording and no gradient would reach the weights.
    fn apply_update(
        trainable: &[(String, Parameter)],
        gradients: &[Vec<f32>],
        step: f32,
    ) -> Result<()> {
        for ((_, parameter), gradient) in trainable.iter().zip(gradients.iter()) {
            let handle = parameter.tensor();
            let mut guard = handle.write();
            let shape = guard.shape().dims().to_vec();
            let updated: Vec<f32> = guard
                .to_vec()?
                .iter()
                .zip(gradient.iter())
                .map(|(value, grad)| value - step * grad)
                .collect();
            *guard = Tensor::from_vec(updated, &shape)?.requires_grad_(true);
        }
        Ok(())
    }

    /// Forecast future values
    pub fn forecast(&self, series: &TimeSeries, steps: usize) -> Result<TimeSeries> {
        // Use the last sequence_length values from the series for forecasting
        let series_len = series.len();
        if series_len < self.sequence_length {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Series length {} is less than required sequence length {}",
                series_len, self.sequence_length
            )));
        }

        let mut forecasts = Vec::new();
        let mut current_window = Vec::new();

        // Initialize window with last sequence_length values
        let start_idx = series_len - self.sequence_length;
        for i in start_idx..series_len {
            let val = series.values.get_item_flat(i)?;
            current_window.push(val);
        }

        // Generate forecasts step by step
        for _step in 0..steps {
            // Create input tensor from current window
            // Shape: [batch_size=1, seq_len, input_size=1]
            let input_data = current_window.clone();
            let input_tensor = Tensor::from_vec(input_data, &[1, self.sequence_length, 1])?;

            // Make prediction
            let pred_tensor = self.forward(&input_tensor)?;
            let pred_value = pred_tensor.get_item_flat(0)?;

            forecasts.push(pred_value);

            // Update window for next prediction (sliding window)
            current_window.remove(0); // Remove first element
            current_window.push(pred_value); // Add prediction
        }

        // Create forecast time series
        let forecast_tensor = Tensor::from_vec(forecasts, &[steps, 1])?;
        Ok(TimeSeries::new(forecast_tensor))
    }

    /// Create sequences for training from time series data
    pub fn create_sequences(&self, series: &TimeSeries) -> Result<(Tensor, Tensor)> {
        let series_len = series.len();
        if series_len <= self.sequence_length {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "Series too short for sequence creation".to_string(),
            ));
        }

        let num_sequences = series_len - self.sequence_length;
        let mut sequences = Vec::new();
        let mut targets = Vec::new();

        for i in 0..num_sequences {
            // Input sequence
            for j in 0..self.sequence_length {
                let val = series.values.get_item_flat(i + j)?;
                sequences.push(val);
            }

            // Target (next value after sequence)
            let target = series.values.get_item_flat(i + self.sequence_length)?;
            targets.push(target);
        }

        // Create tensors
        let x = Tensor::from_vec(sequences, &[num_sequences, self.sequence_length, 1])?;
        let y = Tensor::from_vec(targets, &[num_sequences, 1])?;

        Ok((x, y))
    }

    /// Get hidden state size
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
}

/// GRU-based time series forecaster
///
/// Wraps a real [`GRU`] layer plus a linear read-out, mirroring the LSTM
/// forecaster: the hidden state at the final time step is projected to a
/// single forecast value.
pub struct GRUForecaster {
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    sequence_length: usize,
    gru: GRU,
    output_layer: Linear,
}

impl GRUForecaster {
    /// Create a new GRU forecaster
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Result<Self> {
        let gru = GRU::with_config(
            input_size,
            hidden_size,
            num_layers,
            true,  // bias
            true,  // batch_first
            0.0,   // dropout
            false, // bidirectional
        )?;
        let output_layer = Linear::new(hidden_size, 1, true);

        Ok(Self {
            input_size,
            hidden_size,
            num_layers,
            sequence_length: 10,
            gru,
            output_layer,
        })
    }

    /// Set sequence length
    pub fn with_sequence_length(mut self, seq_len: usize) -> Self {
        self.sequence_length = seq_len;
        self
    }

    /// Get model parameters
    pub fn params(&self) -> (usize, usize, usize, usize) {
        (
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.sequence_length,
        )
    }

    /// Forward pass through the real GRU layer.
    ///
    /// Input shape: `[batch_size, seq_len, input_size]`. Output: `[batch_size, 1]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims_binding = x.shape();
        let dims = dims_binding.dims();
        if dims.len() != 3 {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "GRUForecaster::forward expects a 3D [batch, seq_len, input_size] tensor, got {dims:?}"
            )));
        }
        let seq_len = dims[1];
        if seq_len == 0 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "GRUForecaster::forward requires seq_len >= 1".to_string(),
            ));
        }

        // GRU (batch_first) -> [batch, seq_len, hidden_size].
        let gru_out = self.gru.forward(x)?;

        // Last time step -> [batch, hidden_size].
        let last_hidden = gru_out.narrow(1, (seq_len - 1) as i64, 1)?.squeeze(1)?;

        // Project to a single forecast value -> [batch, 1].
        self.output_layer.forward(&last_hidden)
    }

    /// Forecast future values via autoregressive rollout through the GRU.
    pub fn forecast(&self, series: &TimeSeries, steps: usize) -> Result<TimeSeries> {
        let series_len = series.len();
        if series_len < self.sequence_length {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Series length {} is less than required sequence length {}",
                series_len, self.sequence_length
            )));
        }

        let mut forecasts = Vec::new();
        let mut current_window = Vec::new();

        let start_idx = series_len - self.sequence_length;
        for i in start_idx..series_len {
            current_window.push(series.values.get_item_flat(i)?);
        }

        for _step in 0..steps {
            let input_data = current_window.clone();
            let input_tensor = Tensor::from_vec(input_data, &[1, self.sequence_length, 1])?;
            let pred_tensor = self.forward(&input_tensor)?;
            let pred_value = pred_tensor.get_item_flat(0)?;
            forecasts.push(pred_value);
            current_window.remove(0);
            current_window.push(pred_value);
        }

        let forecast_tensor = Tensor::from_vec(forecasts, &[steps, 1])?;
        Ok(TimeSeries::new(forecast_tensor))
    }

    /// Get hidden state size
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
}

/// Transformer-based time series forecaster.
///
/// Routes a univariate sequence through an input projection, sinusoidal
/// positional encoding, `num_layers` genuine self-attention + feed-forward
/// encoder blocks, and an output projection.
///
/// The attention is computed with 2-D scaled dot-product attention over a
/// single sequence (`[seq_len, d_model]`). The active `Tensor::matmul` in this
/// workspace only supports 2-D operands, so the standard batched/4-D
/// multi-head path is not available; attention here therefore operates over the
/// full model dimension (one effective head). `nhead` is retained for API
/// compatibility and validated to divide `d_model`.
pub struct TransformerForecaster {
    d_model: usize,
    nhead: usize,
    num_layers: usize,
    seq_len: usize,
    dropout: f32,
    /// Projects the univariate input (1 feature) up to `d_model`.
    input_projection: Linear,
    /// One encoder block per layer (Q/K/V/out projections + feed-forward).
    layers: Vec<TransformerBlock>,
    /// Projects the final hidden state down to a single forecast value.
    output_projection: Linear,
}

/// A single self-attention + feed-forward encoder block (2-D, single batch).
struct TransformerBlock {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    ff1: Linear,
    ff2: Linear,
}

impl TransformerBlock {
    fn new(d_model: usize) -> Self {
        Self {
            q_proj: Linear::new(d_model, d_model, true),
            k_proj: Linear::new(d_model, d_model, true),
            v_proj: Linear::new(d_model, d_model, true),
            out_proj: Linear::new(d_model, d_model, true),
            ff1: Linear::new(d_model, d_model * 4, true),
            ff2: Linear::new(d_model * 4, d_model, true),
        }
    }

    /// Apply the block to a 2-D `[seq_len, d_model]` hidden-state matrix.
    ///
    /// Implements genuine scaled dot-product self-attention with residual
    /// connections:
    ///   attn = softmax(Q Kᵀ / sqrt(d_model)) V
    ///   h    = x + out_proj(attn)
    ///   out  = h + ff2(relu(ff1(h)))
    fn forward(&self, x: &Tensor, d_model: usize) -> Result<Tensor> {
        let q = self.q_proj.forward(x)?; // [seq, d_model]
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // scores = Q Kᵀ / sqrt(d_model)   -> [seq, seq]
        let scale = 1.0_f32 / (d_model as f32).sqrt();
        let scores = q.matmul(&k.transpose(0, 1)?)?.mul_scalar(scale)?;
        let attn = scores.softmax(1)?; // row-wise softmax over keys
        let context = attn.matmul(&v)?; // [seq, d_model]

        // Residual after output projection.
        let attended = self.out_proj.forward(&context)?;
        let h = x.add(&attended)?;

        // Position-wise feed-forward with residual.
        let ff = self.ff2.forward(&self.ff1.forward(&h)?.relu()?)?;
        h.add(&ff)
    }
}

impl TransformerForecaster {
    /// Create a new transformer forecaster.
    ///
    /// `nhead` must divide `d_model`; otherwise an error is returned.
    pub fn new(d_model: usize, nhead: usize, num_layers: usize) -> Result<Self> {
        if nhead == 0 || d_model % nhead != 0 {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "d_model ({d_model}) must be a positive multiple of nhead ({nhead})"
            )));
        }
        if num_layers == 0 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "TransformerForecaster requires num_layers >= 1".to_string(),
            ));
        }

        let dropout = 0.1;
        let input_projection = Linear::new(1, d_model, true);
        let layers = (0..num_layers)
            .map(|_| TransformerBlock::new(d_model))
            .collect();
        let output_projection = Linear::new(d_model, 1, true);

        Ok(Self {
            d_model,
            nhead,
            num_layers,
            seq_len: 100,
            dropout,
            input_projection,
            layers,
            output_projection,
        })
    }

    /// Set sequence length
    pub fn with_sequence_length(mut self, seq_len: usize) -> Self {
        self.seq_len = seq_len;
        self
    }

    /// Set dropout rate (configuration only; rebuild the model to apply it to layers)
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Get model configuration
    pub fn config(&self) -> (usize, usize, usize, usize, f32) {
        (
            self.d_model,
            self.nhead,
            self.num_layers,
            self.seq_len,
            self.dropout,
        )
    }

    /// Forward pass through the real self-attention encoder blocks.
    ///
    /// Input shape: `[batch_size, seq_len, 1]` with `batch_size == 1`
    /// (autoregressive forecasting uses a single sequence). Output: `[1, 1]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims_binding = x.shape();
        let dims = dims_binding.dims();
        if dims.len() != 3 {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "TransformerForecaster::forward expects a 3D [batch, seq_len, 1] tensor, got {dims:?}"
            )));
        }
        let batch_size = dims[0];
        let seq_len = dims[1];
        if seq_len == 0 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "TransformerForecaster::forward requires seq_len >= 1".to_string(),
            ));
        }
        if batch_size != 1 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "TransformerForecaster::forward currently supports batch_size == 1".to_string(),
            ));
        }

        // 1. Project [1, seq, 1] -> [seq, d_model] (2-D for the attention math).
        let flat = x.reshape(&[seq_len as i32, 1])?;
        let projected = self.input_projection.forward(&flat)?; // [seq, d_model]

        // 2. Add sinusoidal positional encoding [seq, d_model].
        let pos = self.positional_encoding(seq_len)?;
        let mut hidden = projected.add(&pos)?;

        // 3. Run each genuine encoder block.
        for layer in &self.layers {
            hidden = layer.forward(&hidden, self.d_model)?;
        }

        // 4. Take the last time step -> [1, d_model].
        let last = hidden.narrow(0, (seq_len - 1) as i64, 1)?;

        // 5. Project to a single forecast value -> [1, 1].
        self.output_projection.forward(&last)
    }

    /// Forecast future values via autoregressive rollout through the encoder.
    pub fn forecast(&self, series: &TimeSeries, steps: usize) -> Result<TimeSeries> {
        // Use last seq_len values for forecasting
        let series_len = series.len();
        if series_len < self.seq_len {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Series length {} is less than required sequence length {}",
                series_len, self.seq_len
            )));
        }

        let mut forecasts = Vec::new();
        let mut current_window = Vec::new();

        // Initialize window
        let start_idx = series_len - self.seq_len;
        for i in start_idx..series_len {
            let val = series.values.get_item_flat(i)?;
            current_window.push(val);
        }

        // Generate forecasts
        for _step in 0..steps {
            // Create input tensor [batch_size=1, seq_len, 1]
            let input_data = current_window.clone();
            let input_tensor = Tensor::from_vec(input_data, &[1, self.seq_len, 1])?;

            // Make prediction
            let pred_tensor = self.forward(&input_tensor)?;
            let pred_value = pred_tensor.get_item_flat(0)?;

            forecasts.push(pred_value);

            // Update sliding window
            current_window.remove(0);
            current_window.push(pred_value);
        }

        let forecast_tensor = Tensor::from_vec(forecasts, &[steps, 1])?;
        Ok(TimeSeries::new(forecast_tensor))
    }

    /// Create sinusoidal positional encodings for transformer input.
    pub fn positional_encoding(&self, seq_len: usize) -> Result<Tensor> {
        let mut pos_enc = Vec::with_capacity(seq_len * self.d_model);

        for pos in 0..seq_len {
            for i in 0..self.d_model {
                let val = if i % 2 == 0 {
                    ((pos as f32) / (10000.0_f32.powf((2.0 * (i as f32)) / (self.d_model as f32))))
                        .sin()
                } else {
                    ((pos as f32)
                        / (10000.0_f32.powf((2.0 * ((i - 1) as f32)) / (self.d_model as f32))))
                    .cos()
                };
                pos_enc.push(val);
            }
        }

        Tensor::from_vec(pos_enc, &[seq_len, self.d_model])
    }
}

/// Convolutional Neural Network for time series.
///
/// Stacks real [`Conv1d`] layers (with `same`-style padding so the temporal
/// length is preserved), applies ReLU between them, performs global average
/// pooling over time, and projects the result to a single forecast value.
pub struct CNNForecaster {
    channels: Vec<usize>,
    kernel_sizes: Vec<usize>,
    seq_len: usize,
    dropout: f32,
    /// One Conv1d per (channel transition, kernel) pair.
    conv_layers: Vec<Conv1d>,
    /// Final linear projection from the last channel count to a scalar.
    output_layer: Linear,
}

impl CNNForecaster {
    /// Create a new CNN forecaster.
    ///
    /// `channels[0]` is the input channel count (1 for a univariate series);
    /// each subsequent entry is the output channel count of a Conv1d layer.
    /// `kernel_sizes` must have exactly `channels.len() - 1` entries.
    pub fn new(channels: Vec<usize>, kernel_sizes: Vec<usize>) -> Result<Self> {
        if channels.len() < 2 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "CNNForecaster requires at least an input and one output channel count".to_string(),
            ));
        }
        if kernel_sizes.len() != channels.len() - 1 {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "kernel_sizes length ({}) must equal channels.len()-1 ({})",
                kernel_sizes.len(),
                channels.len() - 1
            )));
        }

        let mut conv_layers = Vec::with_capacity(kernel_sizes.len());
        for (i, &kernel_size) in kernel_sizes.iter().enumerate() {
            if kernel_size == 0 {
                return Err(torsh_core::error::TorshError::InvalidArgument(
                    "Conv kernel size must be >= 1".to_string(),
                ));
            }
            // `same` padding for odd kernels keeps the sequence length constant.
            let padding = (kernel_size - 1) / 2;
            let conv = Conv1d::new(
                channels[i],
                channels[i + 1],
                kernel_size,
                1,       // stride
                padding, // padding
                1,       // dilation
                true,    // bias
                1,       // groups
            );
            conv_layers.push(conv);
        }

        let last_channels = *channels.last().ok_or_else(|| {
            torsh_core::error::TorshError::InvalidArgument("empty channels".to_string())
        })?;
        let output_layer = Linear::new(last_channels, 1, true);

        Ok(Self {
            channels,
            kernel_sizes,
            seq_len: 50,
            dropout: 0.2,
            conv_layers,
            output_layer,
        })
    }

    /// Set sequence length
    pub fn with_sequence_length(mut self, seq_len: usize) -> Self {
        self.seq_len = seq_len;
        self
    }

    /// Set dropout rate
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Get architecture configuration
    pub fn config(&self) -> (&[usize], &[usize], usize, f32) {
        (
            &self.channels,
            &self.kernel_sizes,
            self.seq_len,
            self.dropout,
        )
    }

    /// Forward pass through the real Conv1d stack.
    ///
    /// Input shape: `[batch_size, seq_len, in_channels]`. Output: `[batch_size, 1]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims_binding = x.shape();
        let dims = dims_binding.dims();
        if dims.len() != 3 {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "CNNForecaster::forward expects a 3D [batch, seq_len, channels] tensor, got {dims:?}"
            )));
        }
        let batch_size = dims[0];
        let seq_len = dims[1];
        let in_channels = dims[2];
        if seq_len == 0 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "CNNForecaster::forward requires seq_len >= 1".to_string(),
            ));
        }

        // Conv1d expects [batch, channels, length]; series tensors are
        // [batch, length, channels] -> transpose the last two dims.
        let mut hidden = x.transpose(1, 2)?; // [batch, in_channels, seq_len]
        let _ = in_channels;

        // Apply each conv + ReLU.
        for conv in &self.conv_layers {
            hidden = conv.forward(&hidden)?;
            hidden = hidden.relu()?;
        }

        // Global average pooling over the temporal axis (dim 2) ->
        // [batch, last_channels].
        let pooled = hidden.mean(Some(&[2]), false)?;

        // Ensure 2D [batch, channels] before the linear projection.
        let last_channels = *self.channels.last().ok_or_else(|| {
            torsh_core::error::TorshError::InvalidArgument("empty channels".to_string())
        })?;
        let pooled = pooled.reshape(&[batch_size as i32, last_channels as i32])?;

        // Project to a single forecast value -> [batch, 1].
        self.output_layer.forward(&pooled)
    }

    /// Forecast future values via autoregressive rollout through the conv stack.
    pub fn forecast(&self, series: &TimeSeries, steps: usize) -> Result<TimeSeries> {
        let series_len = series.len();
        if series_len < self.seq_len {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Series length {} is less than required sequence length {}",
                series_len, self.seq_len
            )));
        }

        let mut forecasts = Vec::new();
        let mut current_window = Vec::new();

        // Initialize with last seq_len values
        let start_idx = series_len - self.seq_len;
        for i in start_idx..series_len {
            let val = series.values.get_item_flat(i)?;
            current_window.push(val);
        }

        // Generate forecasts using sliding window
        for _step in 0..steps {
            // Create input tensor [batch_size=1, seq_len, channels=1]
            let input_data = current_window.clone();
            let input_tensor = Tensor::from_vec(input_data, &[1, self.seq_len, 1])?;

            // Make prediction
            let pred_tensor = self.forward(&input_tensor)?;
            let pred_value = pred_tensor.get_item_flat(0)?;

            forecasts.push(pred_value);

            // Update window
            current_window.remove(0);
            current_window.push(pred_value);
        }

        let forecast_tensor = Tensor::from_vec(forecasts, &[steps, 1])?;
        Ok(TimeSeries::new(forecast_tensor))
    }

    /// Compute receptive field size
    pub fn receptive_field(&self) -> usize {
        // Calculate the receptive field of the CNN
        // This is the effective input window size that influences one output
        let mut field = 1;
        for &kernel_size in &self.kernel_sizes {
            field += kernel_size - 1;
        }
        field
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::{creation::zeros, Tensor};

    fn create_test_series() -> TimeSeries {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor = Tensor::from_vec(data, &[8]).expect("Tensor should succeed");
        TimeSeries::new(tensor)
    }

    #[test]
    fn test_lstm_forecaster_creation() {
        let lstm = LSTMForecaster::new(1, 64, 2).expect("LSTMForecaster should succeed");
        let (input_size, hidden_size, num_layers, seq_len, dropout) = lstm.params();
        assert_eq!(input_size, 1);
        assert_eq!(hidden_size, 64);
        assert_eq!(num_layers, 2);
        assert_eq!(seq_len, 10);
        assert_eq!(dropout, 0.0);
    }

    #[test]
    fn test_lstm_forecaster_config() {
        let lstm = LSTMForecaster::new(1, 64, 2)
            .expect("operation should succeed")
            .with_sequence_length(20)
            .with_dropout(0.2);
        let (_, _, _, seq_len, dropout) = lstm.params();
        assert_eq!(seq_len, 20);
        assert_eq!(dropout, 0.2);
    }

    #[test]
    fn test_lstm_forecaster_forward() {
        let lstm = LSTMForecaster::new(1, 64, 2).expect("LSTMForecaster should succeed");
        let input = zeros(&[1, 8, 1]).expect("zeros should succeed"); // batch_size=1, seq_len=8, features=1
        let output = lstm.forward(&input).expect("forward pass should succeed");

        // Output should have shape [batch_size, 1] (single prediction)
        assert_eq!(output.shape().dims(), [1, 1]);
    }

    #[test]
    fn test_lstm_forecaster_forward_nonzero() {
        // A real LSTM forward over a non-trivial input must NOT return zeros.
        // (A genuine forward of an all-zero input with zero biases is zero, so
        // we feed a ramp to exercise the gates and projection.)
        let lstm = LSTMForecaster::new(1, 16, 1).expect("LSTMForecaster should succeed");
        let data: Vec<f32> = (1..=8).map(|v| v as f32).collect();
        let input = Tensor::from_vec(data, &[1, 8, 1]).expect("input tensor");
        let output = lstm.forward(&input).expect("forward pass should succeed");

        assert_eq!(output.shape().dims(), [1, 1]);
        let val = output.get_item_flat(0).expect("output value");
        assert!(
            val.abs() > 1e-9,
            "LSTM forecaster output should be non-zero, got {val}"
        );
    }

    #[test]
    fn test_lstm_forecaster_forecast() {
        let series = create_test_series();
        let lstm = LSTMForecaster::new(1, 64, 2)
            .expect("operation should succeed")
            .with_sequence_length(5); // Use shorter sequence
        let forecast = lstm
            .forecast(&series, 4)
            .expect("forecast computation should succeed");

        assert_eq!(forecast.len(), 4);

        // Forecasts must be genuine (non-zero) numbers from the wired network.
        let vals = forecast.values.to_vec().expect("forecast values");
        assert!(
            vals.iter().any(|&v| v.abs() > 1e-9),
            "LSTM forecast should contain non-zero values, got {vals:?}"
        );
    }

    #[test]
    fn test_gru_forecaster_creation() {
        let gru = GRUForecaster::new(1, 32, 1).expect("gru should build");
        let (input_size, hidden_size, num_layers, seq_len) = gru.params();
        assert_eq!(input_size, 1);
        assert_eq!(hidden_size, 32);
        assert_eq!(num_layers, 1);
        assert_eq!(seq_len, 10);
    }

    #[test]
    fn test_gru_forecaster_config() {
        let gru = GRUForecaster::new(1, 32, 1)
            .expect("gru should build")
            .with_sequence_length(15);
        let (_, _, _, seq_len) = gru.params();
        assert_eq!(seq_len, 15);
    }

    #[test]
    fn test_gru_forecaster_forecast() {
        let series = create_test_series();
        let gru = GRUForecaster::new(1, 32, 1)
            .expect("gru should build")
            .with_sequence_length(5);
        let forecast = gru
            .forecast(&series, 3)
            .expect("forecast computation should succeed");

        assert_eq!(forecast.len(), 3);

        // The wired GRU must produce finite, non-zero forecasts.
        let vals = forecast.values.to_vec().expect("forecast values");
        assert!(
            vals.iter().any(|&v| v.abs() > 1e-9),
            "GRU forecast should contain non-zero values, got {vals:?}"
        );
    }

    #[test]
    fn test_transformer_forecaster_creation() {
        let transformer = TransformerForecaster::new(64, 8, 6).expect("transformer should build");
        let (d_model, nhead, num_layers, seq_len, dropout) = transformer.config();
        assert_eq!(d_model, 64);
        assert_eq!(nhead, 8);
        assert_eq!(num_layers, 6);
        assert_eq!(seq_len, 100);
        assert_eq!(dropout, 0.1);
    }

    #[test]
    fn test_transformer_forecaster_invalid_heads() {
        // nhead must divide d_model.
        assert!(TransformerForecaster::new(10, 3, 2).is_err());
    }

    #[test]
    fn test_transformer_forecaster_config() {
        let transformer = TransformerForecaster::new(64, 8, 6)
            .expect("transformer should build")
            .with_sequence_length(200)
            .with_dropout(0.2);
        let (_, _, _, seq_len, dropout) = transformer.config();
        assert_eq!(seq_len, 200);
        assert_eq!(dropout, 0.2);
    }

    #[test]
    fn test_transformer_forecast() {
        let series = create_test_series();
        let transformer = TransformerForecaster::new(8, 2, 1)
            .expect("transformer should build")
            .with_sequence_length(5); // Use shorter sequence
        let forecast = transformer
            .forecast(&series, 3)
            .expect("forecast computation should succeed");

        assert_eq!(forecast.len(), 3);

        // The wired encoder must produce finite, non-zero forecasts.
        let vals = forecast.values.to_vec().expect("forecast values");
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "transformer forecast values must be finite, got {vals:?}"
        );
        assert!(
            vals.iter().any(|&v| v.abs() > 1e-9),
            "transformer forecast should contain non-zero values, got {vals:?}"
        );
    }

    #[test]
    fn test_cnn_forecaster_creation() {
        let channels = vec![1, 32, 64];
        let kernel_sizes = vec![3, 3];
        let cnn =
            CNNForecaster::new(channels.clone(), kernel_sizes.clone()).expect("cnn should build");
        let (ch, ks, seq_len, dropout) = cnn.config();
        assert_eq!(ch, &channels);
        assert_eq!(ks, &kernel_sizes);
        assert_eq!(seq_len, 50);
        assert_eq!(dropout, 0.2);
    }

    #[test]
    fn test_cnn_forecaster_invalid_kernels() {
        // kernel_sizes length must equal channels.len() - 1.
        assert!(CNNForecaster::new(vec![1, 16, 32], vec![3]).is_err());
    }

    #[test]
    fn test_cnn_forecaster_config() {
        let channels = vec![1, 16];
        let kernel_sizes = vec![5];
        let cnn = CNNForecaster::new(channels, kernel_sizes)
            .expect("cnn should build")
            .with_sequence_length(30)
            .with_dropout(0.3);
        let (_, _, seq_len, dropout) = cnn.config();
        assert_eq!(seq_len, 30);
        assert_eq!(dropout, 0.3);
    }

    #[test]
    fn test_cnn_forecaster_forecast() {
        let series = create_test_series();
        let channels = vec![1, 16];
        let kernel_sizes = vec![3];
        let cnn = CNNForecaster::new(channels, kernel_sizes)
            .expect("cnn should build")
            .with_sequence_length(5); // Use shorter sequence
        let forecast = cnn
            .forecast(&series, 2)
            .expect("forecast computation should succeed");

        assert_eq!(forecast.len(), 2);

        // The wired conv stack must produce finite forecasts.
        let vals = forecast.values.to_vec().expect("forecast values");
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "cnn forecast values must be finite, got {vals:?}"
        );
    }
}
