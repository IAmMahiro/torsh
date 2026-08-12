//! Gradient checkpointing: trade recomputation for activation memory.
//!
//! A checkpointed segment runs its forward pass with gradient recording
//! **disabled**, so none of its intermediate activations are retained; only the
//! segment's inputs and its output survive. When gradients are needed, the
//! segment's forward is re-executed with recording enabled and the backward pass
//! runs through that freshly built graph, which is then dropped again. Peak
//! memory therefore scales with the largest single segment rather than with the
//! whole model, at the cost of one extra forward pass.
//!
//! # Deterministic replay
//!
//! Recomputation is only sound if the second forward pass reproduces the first
//! one exactly. For a stochastic segment (dropout, stochastic depth, sampled
//! noise) that requires the same random draws both times — otherwise the
//! gradients belong to a *different* function than the one that produced the
//! output, and nothing reports the mismatch.
//!
//! This API makes that structural instead of hopeful: the checkpointed closure
//! receives a [`CheckpointRng`], and both the original forward and every
//! recompute hand it an RNG freshly seeded from the same value. Stochastic ops
//! inside the segment must draw from that RNG rather than from an ambient
//! thread-local generator, which the closure signature makes hard to get wrong.
//! Use [`checkpoint_with_seed`] to pin the seed explicitly (for reproducible
//! runs); [`checkpoint`] draws one once, at construction, and reuses it.
//!
//! # Shape of the API
//!
//! Backward is an explicit method rather than a graph node. `torsh-tensor`'s
//! `Operation` enum is closed and carries no callable payload, so a checkpoint
//! node cannot currently be spliced into the tape from this crate; splicing it
//! in would mean adding a closure-carrying variant to the tensor crate's graph.
//! [`Checkpoint::backward`] and [`CheckpointSequence::backward`] therefore take
//! the upstream gradient and return the input gradients, which composes with a
//! hand-written training step and keeps the memory saving intact.
//!
//! # Example
//!
//! ```rust
//! use torsh_autograd::checkpoint::checkpoint_with_seed;
//! use torsh_core::device::DeviceType;
//! use torsh_tensor::Tensor;
//!
//! # fn main() -> torsh_core::error::Result<()> {
//! let x = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu)?;
//!
//! // Segment: y = x * x, checkpointed.
//! let segment = checkpoint_with_seed(|inputs: &[Tensor<f32>], _rng| inputs[0].mul(&inputs[0]), &[x], 0)?;
//! assert_eq!(segment.output().to_vec()?, vec![1.0, 4.0, 9.0]);
//!
//! // dy/dx = 2x, recomputed on demand.
//! let seed = Tensor::from_data(vec![1.0f32; 3], vec![3], DeviceType::Cpu)?;
//! let grads = segment.backward(&seed)?;
//! assert_eq!(grads[0].to_vec()?, vec![2.0, 4.0, 6.0]);
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use scirs2_core::random::{random, rngs::StdRng, seeded_rng, CoreRandom};
use torsh_core::dtype::FloatElement;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

/// Random number generator handed to a checkpointed segment.
///
/// Seeded identically on the original forward pass and on every recompute, so
/// stochastic segments replay bit-for-bit.
pub type CheckpointRng = CoreRandom<StdRng>;

/// A checkpointed segment: a closure over several inputs producing one output.
type SegmentFn<T> =
    Arc<dyn Fn(&[Tensor<T>], &mut CheckpointRng) -> Result<Tensor<T>> + Send + Sync>;

/// A checkpointed step of a sequence: one input, one output.
type StepFn<T> = Arc<dyn Fn(&Tensor<T>, &mut CheckpointRng) -> Result<Tensor<T>> + Send + Sync>;

/// Element types a checkpoint can back-propagate through.
///
/// This is exactly what `Tensor::backward_with_grad` requires, gathered into one
/// name so the bound does not have to be repeated on every method.
pub trait CheckpointElement:
    FloatElement
    + Copy
    + Default
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::fmt::Debug
{
}

impl<T> CheckpointElement for T where
    T: FloatElement
        + Copy
        + Default
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::fmt::Debug
{
}

/// A forward pass whose intermediate activations were discarded.
///
/// Holds only the segment inputs, the segment output and enough information to
/// re-run the forward exactly: the closure itself and the RNG seed it was run
/// with.
pub struct Checkpoint<T: FloatElement> {
    /// Detached copies of the segment inputs (fresh leaves, own gradient slots).
    inputs: Vec<Tensor<T>>,
    /// The segment output, detached from any graph.
    output: Tensor<T>,
    /// The segment forward pass, replayable.
    function: SegmentFn<T>,
    /// Seed used for every execution of `function`.
    seed: u64,
}

impl<T: FloatElement> Checkpoint<T> {
    /// The segment's output.
    ///
    /// Detached: it carries no graph, which is the point of checkpointing.
    pub fn output(&self) -> &Tensor<T> {
        &self.output
    }

    /// The segment's inputs, as stored for recomputation.
    pub fn inputs(&self) -> &[Tensor<T>] {
        &self.inputs
    }

    /// The RNG seed replayed on every recompute.
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

/// Build a leaf tensor that shares nothing with `source`.
///
/// `Tensor::clone` (and therefore `detach`) is shallow and keeps the original's
/// gradient slot, which is what the autograd engine uses as node identity. A
/// recompute must not alias the caller's tensors, so the data is copied into a
/// genuinely fresh leaf.
fn fresh_leaf<T: FloatElement + Copy>(
    source: &Tensor<T>,
    requires_grad: bool,
) -> Result<Tensor<T>> {
    let leaf = Tensor::from_data(
        source.to_vec()?,
        source.shape().dims().to_vec(),
        source.device(),
    )?;
    Ok(leaf.requires_grad_(requires_grad))
}

/// Run `function` with gradient recording disabled, discarding its activations.
///
/// The seed is drawn once here and stored, so the recompute in
/// [`Checkpoint::backward`] replays the same random draws.
///
/// # Errors
///
/// Propagates any error from `function`, and fails if an input tensor cannot be
/// copied into a fresh leaf.
pub fn checkpoint<T, F>(function: F, inputs: &[Tensor<T>]) -> Result<Checkpoint<T>>
where
    T: FloatElement + Copy,
    F: Fn(&[Tensor<T>], &mut CheckpointRng) -> Result<Tensor<T>> + Send + Sync + 'static,
{
    let seed: u64 = random();
    checkpoint_with_seed(function, inputs, seed)
}

/// [`checkpoint`] with an explicit RNG seed, for reproducible runs.
///
/// # Errors
///
/// Propagates any error from `function`, and fails if an input tensor cannot be
/// copied into a fresh leaf.
pub fn checkpoint_with_seed<T, F>(
    function: F,
    inputs: &[Tensor<T>],
    seed: u64,
) -> Result<Checkpoint<T>>
where
    T: FloatElement + Copy,
    F: Fn(&[Tensor<T>], &mut CheckpointRng) -> Result<Tensor<T>> + Send + Sync + 'static,
{
    if inputs.is_empty() {
        return Err(TorshError::AutogradError(
            "checkpoint requires at least one input tensor".to_string(),
        ));
    }

    let stored: Vec<Tensor<T>> = inputs
        .iter()
        .map(|input| fresh_leaf(input, false))
        .collect::<Result<_>>()?;

    let function: SegmentFn<T> = Arc::new(function);

    // The forward runs with recording off: nothing inside the segment joins a
    // graph, so the intermediates are freed as soon as the closure returns.
    let output =
        torsh_core::grad_mode::with_grad_mode(false, || function(&stored, &mut seeded_rng(seed)))?;

    Ok(Checkpoint {
        inputs: stored,
        output: output.detach(),
        function,
        seed,
    })
}

impl<T: CheckpointElement> Checkpoint<T>
where
    f32: From<T>,
{
    /// Recompute the segment and back-propagate `grad_output` through it.
    ///
    /// Returns one gradient per input, in input order. An input the segment
    /// never used gets a zero gradient (its true derivative), not a missing
    /// entry.
    ///
    /// # Errors
    ///
    /// - `grad_output` does not match the segment output's shape
    /// - the recompute produces a different shape than the original forward
    ///   (which means the closure is not deterministic and the checkpoint is
    ///   invalid)
    /// - any error raised by the segment closure or by the backward pass
    pub fn backward(&self, grad_output: &Tensor<T>) -> Result<Vec<Tensor<T>>> {
        if grad_output.shape().dims() != self.output.shape().dims() {
            return Err(TorshError::ShapeMismatch {
                expected: self.output.shape().dims().to_vec(),
                got: grad_output.shape().dims().to_vec(),
            });
        }

        // Fresh differentiable leaves, so the recomputed graph is rooted here
        // and cannot leak gradients into the caller's tensors.
        let leaves: Vec<Tensor<T>> = self
            .inputs
            .iter()
            .map(|input| fresh_leaf(input, true))
            .collect::<Result<_>>()?;

        let function = Arc::clone(&self.function);
        let seed = self.seed;
        let recomputed = torsh_core::grad_mode::with_grad_mode(true, || {
            function(&leaves, &mut seeded_rng(seed))
        })?;

        if recomputed.shape().dims() != self.output.shape().dims() {
            return Err(TorshError::AutogradError(format!(
                "checkpoint recompute produced shape {:?} but the original forward produced {:?}; \
                 the checkpointed function is not deterministic, so its recomputed gradients \
                 would not belong to the recorded output",
                recomputed.shape().dims(),
                self.output.shape().dims()
            )));
        }

        if !recomputed.requires_grad() {
            return Err(TorshError::AutogradError(
                "checkpoint recompute produced a tensor with no graph: the checkpointed function \
                 detaches its inputs or discards them, so no gradient can flow to them"
                    .to_string(),
            ));
        }

        recomputed.backward_with_grad(Some(grad_output))?;

        leaves
            .iter()
            .map(|leaf| match leaf.grad() {
                Some(grad) => Ok(grad),
                // Untouched input: the derivative genuinely is zero.
                None => Tensor::zeros(&leaf.shape().dims().to_vec(), leaf.device()),
            })
            .collect()
    }
}

/// A chain of checkpointed segments over a single tensor.
///
/// Produced by [`checkpoint_sequential`]. Only the boundary tensors between
/// segments are retained; everything inside a segment is recomputed on demand.
pub struct CheckpointSequence<T: FloatElement> {
    segments: Vec<Checkpoint<T>>,
}

impl<T: FloatElement> CheckpointSequence<T> {
    /// Output of the final segment.
    pub fn output(&self) -> Result<&Tensor<T>> {
        self.segments
            .last()
            .map(|segment| segment.output())
            .ok_or_else(|| {
                TorshError::AutogradError("checkpoint sequence has no segments".to_string())
            })
    }

    /// Number of checkpointed segments.
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// Whether the sequence has no segments.
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// The individual segments, in forward order.
    pub fn segments(&self) -> &[Checkpoint<T>] {
        &self.segments
    }
}

impl<T: CheckpointElement> CheckpointSequence<T>
where
    f32: From<T>,
{
    /// Back-propagate `grad_output` through every segment, last to first.
    ///
    /// Returns the gradient with respect to the sequence's original input. Each
    /// segment is recomputed exactly once, and its graph is dropped before the
    /// next segment is recomputed, so peak memory stays at one segment.
    ///
    /// # Errors
    ///
    /// Propagates the errors of [`Checkpoint::backward`], and fails if the
    /// sequence is empty.
    pub fn backward(&self, grad_output: &Tensor<T>) -> Result<Tensor<T>> {
        let mut gradient = grad_output.clone();
        for segment in self.segments.iter().rev() {
            let grads = segment.backward(&gradient)?;
            gradient = grads.into_iter().next().ok_or_else(|| {
                TorshError::AutogradError(
                    "checkpoint segment returned no input gradient".to_string(),
                )
            })?;
        }
        Ok(gradient)
    }
}

/// Split `functions` into `segments` consecutive groups and checkpoint each one.
///
/// This is the classic `torch.utils.checkpoint.checkpoint_sequential` layout: a
/// long chain of layers is cut into roughly equal segments, and only the tensors
/// on the segment boundaries are kept. With `n` layers in `s` segments, memory
/// drops from `O(n)` stored activations to `O(s + n/s)`.
///
/// # Errors
///
/// Fails if `functions` is empty or `segments` is zero, and propagates any error
/// raised while running a segment forward.
pub fn checkpoint_sequential<T, F>(
    functions: Vec<F>,
    segments: usize,
    input: &Tensor<T>,
) -> Result<CheckpointSequence<T>>
where
    T: FloatElement + Copy,
    F: Fn(&Tensor<T>, &mut CheckpointRng) -> Result<Tensor<T>> + Send + Sync + 'static,
{
    if functions.is_empty() {
        return Err(TorshError::AutogradError(
            "checkpoint_sequential requires at least one function".to_string(),
        ));
    }
    if segments == 0 {
        return Err(TorshError::AutogradError(
            "checkpoint_sequential requires at least one segment".to_string(),
        ));
    }

    let steps: Vec<StepFn<T>> = functions
        .into_iter()
        .map(|f| Arc::new(f) as StepFn<T>)
        .collect();

    let segment_count = segments.min(steps.len());
    // Distribute the remainder over the leading segments so the group sizes
    // differ by at most one.
    let base = steps.len() / segment_count;
    let remainder = steps.len() % segment_count;

    let mut built = Vec::with_capacity(segment_count);
    let mut current = input.clone();
    let mut offset = 0;

    for index in 0..segment_count {
        let size = base + usize::from(index < remainder);
        let group: Vec<StepFn<T>> = steps[offset..offset + size].to_vec();
        offset += size;

        let segment = checkpoint(
            move |inputs: &[Tensor<T>], rng: &mut CheckpointRng| {
                let mut value = inputs
                    .first()
                    .ok_or_else(|| {
                        TorshError::AutogradError(
                            "checkpoint segment received no input".to_string(),
                        )
                    })?
                    .clone();
                for step in &group {
                    value = step(&value, rng)?;
                }
                Ok(value)
            },
            &[current],
        )?;

        current = segment.output().clone();
        built.push(segment);
    }

    Ok(CheckpointSequence { segments: built })
}
