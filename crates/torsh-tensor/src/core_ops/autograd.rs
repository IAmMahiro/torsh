//! Reverse-mode automatic differentiation engine for [`Tensor`].
//!
//! # Engine shape
//!
//! Backward is a two-phase, iterative walk instead of a recursive one:
//!
//! 1. **Discovery** — starting at the root, every reachable node that requires a
//!    gradient is collected and the number of *consumer edges* pointing at it is
//!    counted.
//! 2. **Propagation** — nodes are processed in reverse-topological order. A node
//!    only emits gradients to its parents once every consumer has delivered its
//!    contribution, so a value used by several consumers (a residual/skip
//!    connection, or a weight used twice) accumulates exactly once per edge and
//!    its ancestors are visited exactly once in total.
//!
//! A recursive walk without this bookkeeping re-traverses the whole ancestor
//! subtree once per path, which is exponential in the number of stacked skip
//! connections.
//!
//! # Node identity
//!
//! Graph edges store `Arc<Tensor<T>>` *clones* of their operands, and
//! `Tensor::clone` is shallow: clones share the `grad` slot
//! (`Arc<RwLock<Option<Tensor<T>>>>`). The address of that slot is therefore the
//! only identity that survives cloning, and it is what the engine keys on. Two
//! independently created tensors always have distinct slots, and a tensor used
//! twice always resolves to one node.
//!
//! Because the whole engine rests on that invariant, it is checked rather than
//! assumed: a node that lists itself as its own parent, or a graph that cannot be
//! fully drained, returns [`TorshError::AutogradError`] instead of quietly
//! producing zero gradients.
//!
//! # Broadcasting
//!
//! Element-wise operations record their *un-broadcast* operands, so every
//! gradient is folded back to the operand's own shape with
//! [`reduce_grad_to_shape`] before it is handed upstream. Without that, a
//! `[out_features]` bias added to a `[batch, out_features]` activation would
//! receive a `[batch, out_features]` gradient.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::Arc;

use torsh_core::{
    dtype::{FloatElement, TensorElement},
    error::{Result, TorshError},
};

use super::types::{Im2ColConfig, Operation, Tensor, UnaryKind, ViewKind};

/// Identity of a node in the autograd graph (the address of its gradient slot).
type NodeKey = *const ();

/// Advance a multi-dimensional odometer over `dims` by one element.
fn increment_coords(coords: &mut [usize], dims: &[usize]) {
    for axis in (0..coords.len()).rev() {
        coords[axis] += 1;
        if coords[axis] < dims[axis] {
            return;
        }
        coords[axis] = 0;
    }
}

/// Sum `grad` back down to `target`, undoing NumPy/PyTorch broadcasting.
///
/// Axes that were prepended to the operand, and axes where the operand has
/// extent 1 while the gradient has extent `n`, are summed over. `[3, 4]` reduced
/// to `[3, 1]` keeps the trailing axis (`keepdim` semantics); reduced to `[4]` it
/// sums over the leading axis.
pub(crate) fn reduce_grad_to_shape<T>(grad: &Tensor<T>, target: &[usize]) -> Result<Tensor<T>>
where
    T: TensorElement + Copy + std::ops::Add<Output = T>,
{
    let grad_shape = grad.shape();
    let grad_dims = grad_shape.dims();
    if grad_dims == target {
        return Ok(grad.detach());
    }
    if target.len() > grad_dims.len() {
        return Err(TorshError::ShapeMismatch {
            expected: target.to_vec(),
            got: grad_dims.to_vec(),
        });
    }

    // Leading gradient axes have no counterpart in `target` and are summed away.
    let offset = grad_dims.len() - target.len();

    // Stride of each target axis inside the flat output buffer. A broadcast axis
    // (target extent 1) gets stride 0, which is exactly what folds every copy of
    // the broadcast value into the same accumulator slot.
    let mut target_strides = vec![0usize; target.len()];
    let mut stride = 1usize;
    for axis in (0..target.len()).rev() {
        let extent = target[axis];
        let grad_extent = grad_dims[offset + axis];
        if extent != grad_extent && extent != 1 {
            return Err(TorshError::ShapeMismatch {
                expected: target.to_vec(),
                got: grad_dims.to_vec(),
            });
        }
        target_strides[axis] = if extent == 1 { 0 } else { stride };
        stride *= extent;
    }

    let target_numel: usize = target.iter().product();
    let mut reduced = vec![<T as TensorElement>::zero(); target_numel];
    let data = grad.to_vec()?;
    let mut coords = vec![0usize; grad_dims.len()];
    for &value in data.iter() {
        let mut index = 0usize;
        for (axis, target_stride) in target_strides.iter().enumerate() {
            index += coords[offset + axis] * target_stride;
        }
        if let Some(slot) = reduced.get_mut(index) {
            *slot = *slot + value;
        }
        increment_coords(&mut coords, grad_dims);
    }

    Tensor::from_data(reduced, target.to_vec(), grad.device)
}

/// Broadcast `grad` up to `target`, the inverse of [`reduce_grad_to_shape`].
///
/// Used by reductions: the gradient of `sum`/`mean` is the (scaled) output
/// gradient replicated across every element that was reduced.
pub(crate) fn expand_grad_to_shape<T>(grad: &Tensor<T>, target: &[usize]) -> Result<Tensor<T>>
where
    T: TensorElement + Copy,
{
    let grad_shape = grad.shape();
    let grad_dims = grad_shape.dims();
    if grad_dims == target {
        return Ok(grad.detach());
    }
    if grad_dims.len() > target.len() {
        return Err(TorshError::ShapeMismatch {
            expected: target.to_vec(),
            got: grad_dims.to_vec(),
        });
    }

    let offset = target.len() - grad_dims.len();
    let mut grad_strides = vec![0usize; grad_dims.len()];
    let mut stride = 1usize;
    for axis in (0..grad_dims.len()).rev() {
        let extent = grad_dims[axis];
        let target_extent = target[offset + axis];
        if extent != target_extent && extent != 1 {
            return Err(TorshError::ShapeMismatch {
                expected: target.to_vec(),
                got: grad_dims.to_vec(),
            });
        }
        grad_strides[axis] = if extent == 1 { 0 } else { stride };
        stride *= extent;
    }

    let data = grad.to_vec()?;
    let target_numel: usize = target.iter().product();
    let mut expanded = Vec::with_capacity(target_numel);
    let mut coords = vec![0usize; target.len()];
    for _ in 0..target_numel {
        let mut index = 0usize;
        for (axis, grad_stride) in grad_strides.iter().enumerate() {
            index += coords[offset + axis] * grad_stride;
        }
        let value = *data.get(index).ok_or_else(|| TorshError::ShapeMismatch {
            expected: target.to_vec(),
            got: grad_dims.to_vec(),
        })?;
        expanded.push(value);
        increment_coords(&mut coords, target);
    }

    Tensor::from_data(expanded, target.to_vec(), grad.device)
}

/// Broadcast two batch-dimension lists following NumPy/PyTorch rules.
fn broadcast_batch_dims(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>> {
    let rank = lhs.len().max(rhs.len());
    let mut out = vec![1usize; rank];
    for (axis, slot) in out.iter_mut().enumerate() {
        let left = if axis + lhs.len() >= rank {
            lhs[axis + lhs.len() - rank]
        } else {
            1
        };
        let right = if axis + rhs.len() >= rank {
            rhs[axis + rhs.len() - rank]
        } else {
            1
        };
        *slot = if left == right {
            left
        } else if left == 1 {
            right
        } else if right == 1 {
            left
        } else {
            return Err(TorshError::ShapeMismatch {
                expected: lhs.to_vec(),
                got: rhs.to_vec(),
            });
        };
    }
    Ok(out)
}

/// Flat offset of the matrix selected by `coords` inside a (possibly broadcast)
/// batch shape. Batch axes of extent 1 are read from index 0, which is what makes
/// the gradient of a broadcast operand accumulate over the batch.
fn batch_offset(coords: &[usize], shape: &[usize], matrix_stride: usize) -> usize {
    let rank = shape.len();
    let mut offset = 0usize;
    let mut stride = matrix_stride;
    for axis in (0..rank).rev() {
        let coord_index = coords.len() + axis - rank;
        let coord = if shape[axis] == 1 {
            0
        } else {
            coords[coord_index]
        };
        offset += coord * stride;
        stride *= shape[axis];
    }
    offset
}

/// Diagnostic name of an operation.
fn operation_name<T: TensorElement>(operation: &Operation<T>) -> String {
    match operation {
        Operation::Leaf => "leaf".to_string(),
        Operation::Power { .. } => "pow".to_string(),
        Operation::Add { .. } => "add".to_string(),
        Operation::Sub { .. } => "sub".to_string(),
        Operation::Mul { .. } => "mul".to_string(),
        Operation::Div { .. } => "div".to_string(),
        Operation::MulScalar { .. } => "mul_scalar".to_string(),
        Operation::DivScalar { .. } => "div_scalar".to_string(),
        Operation::Mean { .. } => "mean".to_string(),
        Operation::Sum { .. } => "sum".to_string(),
        Operation::MatMul { .. } => "matmul".to_string(),
        Operation::Custom(name, _) => format!("custom({name})"),
        Operation::View { kind, .. } => format!("view({kind:?})"),
        Operation::Im2Col { .. } => "im2col".to_string(),
        Operation::Concat { dim, .. } => format!("cat(dim={dim})"),
        Operation::Stack { dim, .. } => format!("stack(dim={dim})"),
        Operation::Gather { .. } => "gather".to_string(),
        Operation::LogSoftmax { dim, .. } => format!("log_softmax(dim={dim})"),
        Operation::Unary { kind, .. } => format!("unary({kind:?})"),
    }
}

/// Operands an operation reads, in the order their gradients are produced.
fn graph_parents<T: TensorElement>(operation: &Operation<T>) -> Vec<Arc<Tensor<T>>> {
    match operation {
        Operation::Leaf => Vec::new(),
        Operation::Power { input, .. }
        | Operation::Mean { input, .. }
        | Operation::Sum { input }
        | Operation::MulScalar { input, .. }
        | Operation::DivScalar { input, .. }
        | Operation::View { input, .. }
        | Operation::Im2Col { input, .. }
        | Operation::Gather { input, .. }
        | Operation::LogSoftmax { input, .. }
        | Operation::Unary { input, .. } => vec![Arc::clone(input)],
        Operation::Add { lhs, rhs }
        | Operation::Sub { lhs, rhs }
        | Operation::Mul { lhs, rhs }
        | Operation::Div { lhs, rhs }
        | Operation::MatMul { lhs, rhs } => vec![Arc::clone(lhs), Arc::clone(rhs)],
        // Multi-input ops list every operand once per occurrence, so a tensor
        // concatenated with itself is counted twice and its gradient accumulates.
        Operation::Concat { inputs, .. } | Operation::Stack { inputs, .. } => {
            inputs.iter().map(Arc::clone).collect()
        }
        // Dead weak references simply contribute no parents; the backward rule
        // below reports the missing derivative rather than silently succeeding.
        Operation::Custom(_, inputs) => inputs.iter().filter_map(|input| input.upgrade()).collect(),
    }
}

impl<T: TensorElement + Copy> Tensor<T> {
    /// Stable identity of this tensor as an autograd node.
    ///
    /// Clones share their gradient slot, so the slot address identifies the node
    /// across every clone stored in the graph.
    pub(crate) fn node_key(&self) -> NodeKey {
        Arc::as_ptr(&self.grad) as *const ()
    }
}

impl<T> Tensor<T>
where
    T: FloatElement
        + Copy
        + Default
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    /// Run the backward pass from this tensor with `grad_output` as the seed.
    ///
    /// Gradients are accumulated into the `grad` slot of every reachable leaf.
    pub(crate) fn backward_impl(&self, grad_output: &Self) -> Result<()> {
        let root = Arc::new(self.clone());
        let root_key = root.node_key();

        // ---- Phase 1: discover reachable nodes and count consumer edges ----
        let mut nodes: HashMap<NodeKey, Arc<Self>> = HashMap::new();
        let mut pending: HashMap<NodeKey, usize> = HashMap::new();
        nodes.insert(root_key, Arc::clone(&root));
        pending.insert(root_key, 0);

        let mut stack = vec![Arc::clone(&root)];
        while let Some(node) = stack.pop() {
            let node_key = node.node_key();
            for parent in graph_parents(&node.operation) {
                if !parent.requires_grad {
                    continue;
                }
                let parent_key = parent.node_key();
                if parent_key == node_key {
                    return Err(TorshError::AutogradError(format!(
                        "autograd graph is malformed: operation '{}' lists its own output as an input",
                        operation_name(&node.operation)
                    )));
                }
                *pending.entry(parent_key).or_insert(0) += 1;
                if let Entry::Vacant(slot) = nodes.entry(parent_key) {
                    slot.insert(Arc::clone(&parent));
                    stack.push(parent);
                }
            }
        }

        // ---- Phase 2: drain in reverse-topological order ----
        let mut grads: HashMap<NodeKey, Self> = HashMap::new();
        grads.insert(root_key, grad_output.detach());
        let mut ready = vec![Arc::clone(&root)];
        let mut processed = 0usize;

        while let Some(node) = ready.pop() {
            let node_key = node.node_key();
            let grad = grads.remove(&node_key).ok_or_else(|| {
                TorshError::AutogradError(
                    "autograd engine scheduled a node before its gradient was available"
                        .to_string(),
                )
            })?;
            processed += 1;

            if matches!(node.operation, Operation::Leaf) {
                node.accumulate_leaf_grad(&grad)?;
                continue;
            }

            for (parent, contribution) in node.parent_gradients(&grad)? {
                let parent_key = parent.node_key();
                match grads.entry(parent_key) {
                    Entry::Occupied(mut slot) => {
                        let accumulated = slot.get().detach().add(&contribution)?;
                        slot.insert(accumulated);
                    }
                    Entry::Vacant(slot) => {
                        slot.insert(contribution);
                    }
                }
                if let Some(remaining) = pending.get_mut(&parent_key) {
                    *remaining -= 1;
                    if *remaining == 0 {
                        if let Some(node) = nodes.get(&parent_key) {
                            ready.push(Arc::clone(node));
                        }
                    }
                }
            }
        }

        if processed != nodes.len() {
            return Err(TorshError::AutogradError(format!(
                "autograd graph is not a DAG: {} of {} nodes could not be resolved",
                nodes.len() - processed,
                nodes.len()
            )));
        }
        Ok(())
    }

    /// Accumulate `grad` into this leaf's gradient slot.
    fn accumulate_leaf_grad(&self, grad: &Self) -> Result<()> {
        let shape = self.shape();
        let reduced = reduce_grad_to_shape(grad, shape.dims())?;
        let mut slot = self.grad.write().map_err(|_| {
            TorshError::AutogradError("gradient slot lock was poisoned".to_string())
        })?;
        let accumulated = match slot.as_ref() {
            Some(existing) => existing.detach().add(&reduced)?,
            None => reduced,
        };
        *slot = Some(accumulated);
        Ok(())
    }

    /// Gradient contribution this node sends to each of its operands.
    ///
    /// Every intermediate value is computed on detached operands so the backward
    /// pass never extends the graph it is walking.
    fn parent_gradients(&self, grad_output: &Self) -> Result<Vec<(Arc<Self>, Self)>> {
        let mut contributions: Vec<(Arc<Self>, Self)> = Vec::new();
        match &self.operation {
            Operation::Leaf => {}
            Operation::Power { input, exponent } => {
                if input.requires_grad {
                    // d/dx x^n = n * x^(n-1)
                    let exponent_value = Self::scalar_from_f64(f64::from(*exponent))?;
                    let exponent_minus_one = Self::scalar_from_f64(f64::from(*exponent) - 1.0)?;
                    let input_data = input.to_vec()?;
                    let local: Vec<T> = input_data
                        .iter()
                        .map(|&value| exponent_value * value.powf(exponent_minus_one))
                        .collect();
                    let input_shape = input.shape();
                    let local = Self::from_data(local, input_shape.dims().to_vec(), input.device)?;
                    contributions.push((Arc::clone(input), local.mul(&grad_output.detach())?));
                }
            }
            Operation::Add { lhs, rhs } => {
                // d/dlhs (lhs + rhs) = grad, d/drhs = grad (both un-broadcast).
                if lhs.requires_grad {
                    let shape = lhs.shape();
                    contributions.push((
                        Arc::clone(lhs),
                        reduce_grad_to_shape(grad_output, shape.dims())?,
                    ));
                }
                if rhs.requires_grad {
                    let shape = rhs.shape();
                    contributions.push((
                        Arc::clone(rhs),
                        reduce_grad_to_shape(grad_output, shape.dims())?,
                    ));
                }
            }
            Operation::Sub { lhs, rhs } => {
                if lhs.requires_grad {
                    let shape = lhs.shape();
                    contributions.push((
                        Arc::clone(lhs),
                        reduce_grad_to_shape(grad_output, shape.dims())?,
                    ));
                }
                if rhs.requires_grad {
                    // d/drhs (lhs - rhs) = -grad
                    let shape = rhs.shape();
                    let reduced = reduce_grad_to_shape(grad_output, shape.dims())?;
                    let negated = reduced.mul_scalar(Self::scalar_from_f64(-1.0)?)?;
                    contributions.push((Arc::clone(rhs), negated));
                }
            }
            Operation::Mul { lhs, rhs } => {
                // d/dlhs (lhs * rhs) = rhs, d/drhs = lhs
                let grad = grad_output.detach();
                if lhs.requires_grad {
                    let product = grad.mul(&rhs.detach())?;
                    let shape = lhs.shape();
                    contributions.push((
                        Arc::clone(lhs),
                        reduce_grad_to_shape(&product, shape.dims())?,
                    ));
                }
                if rhs.requires_grad {
                    let product = grad.mul(&lhs.detach())?;
                    let shape = rhs.shape();
                    contributions.push((
                        Arc::clone(rhs),
                        reduce_grad_to_shape(&product, shape.dims())?,
                    ));
                }
            }
            Operation::Div { lhs, rhs } => {
                // d/dlhs (lhs / rhs) = 1 / rhs, d/drhs = -lhs / rhs^2
                let grad = grad_output.detach();
                if lhs.requires_grad {
                    let quotient = grad.div(&rhs.detach())?;
                    let shape = lhs.shape();
                    contributions.push((
                        Arc::clone(lhs),
                        reduce_grad_to_shape(&quotient, shape.dims())?,
                    ));
                }
                if rhs.requires_grad {
                    let denominator = rhs.detach().mul(&rhs.detach())?;
                    let numerator = grad.mul(&lhs.detach())?;
                    let quotient = numerator.div(&denominator)?;
                    let negated = quotient.mul_scalar(Self::scalar_from_f64(-1.0)?)?;
                    let shape = rhs.shape();
                    contributions.push((
                        Arc::clone(rhs),
                        reduce_grad_to_shape(&negated, shape.dims())?,
                    ));
                }
            }
            Operation::MulScalar { input, scalar } => {
                // d/dinput (input * s) = s, so the gradient is scaled by the same
                // constant. The gradient is detached first, which is what keeps
                // this multiply from recording a new node while backward runs.
                if input.requires_grad {
                    let scaled = grad_output.detach().mul_scalar(*scalar)?;
                    let shape = input.shape();
                    contributions.push((
                        Arc::clone(input),
                        reduce_grad_to_shape(&scaled, shape.dims())?,
                    ));
                }
            }
            Operation::DivScalar { input, scalar } => {
                // d/dinput (input / s) = 1 / s.
                if input.requires_grad {
                    let scaled = grad_output.detach().div_scalar(*scalar)?;
                    let shape = input.shape();
                    contributions.push((
                        Arc::clone(input),
                        reduce_grad_to_shape(&scaled, shape.dims())?,
                    ));
                }
            }
            Operation::Mean { input, count } => {
                if input.requires_grad {
                    // d/dinput mean(input) = grad / count, replicated over the
                    // elements that were averaged.
                    if *count == 0.0 {
                        return Err(TorshError::AutogradError(
                            "mean backward with a zero element count".to_string(),
                        ));
                    }
                    let shape = input.shape();
                    let expanded =
                        expand_grad_to_shape(grad_output, shape.dims()).map_err(|_| {
                            TorshError::AutogradError(format!(
                                "mean backward cannot map a {:?} gradient onto a {:?} input: the \
                             reduced axes are not recorded, so only whole-tensor means and \
                             keepdim reductions can be differentiated",
                                grad_output.shape().dims(),
                                shape.dims()
                            ))
                        })?;
                    let scale = Self::scalar_from_f64(1.0 / *count)?;
                    contributions.push((Arc::clone(input), expanded.mul_scalar(scale)?));
                }
            }
            Operation::Sum { input } => {
                if input.requires_grad {
                    // d/dinput sum(input) = grad replicated over every element.
                    let shape = input.shape();
                    let expanded =
                        expand_grad_to_shape(grad_output, shape.dims()).map_err(|_| {
                            TorshError::AutogradError(format!(
                                "sum backward cannot map a {:?} gradient onto a {:?} input",
                                grad_output.shape().dims(),
                                shape.dims()
                            ))
                        })?;
                    contributions.push((Arc::clone(input), expanded));
                }
            }
            Operation::MatMul { lhs, rhs } => {
                contributions.extend(Self::matmul_gradients(lhs, rhs, grad_output)?);
            }
            Operation::View { input, kind } => {
                if input.requires_grad {
                    let input_shape = input.shape();
                    let dims = input_shape.dims();
                    let grad = grad_output.detach();
                    let contribution = match kind {
                        // Element order is unchanged: reshape the gradient back.
                        ViewKind::Reshape => {
                            Self::from_data(grad.to_vec()?, dims.to_vec(), input.device)?
                        }
                        // Undo the axis permutation, then repack in input order.
                        ViewKind::Permute(perm) => {
                            let mut inverse = vec![0usize; perm.len()];
                            for (output_axis, &input_axis) in perm.iter().enumerate() {
                                let slot = inverse.get_mut(input_axis).ok_or_else(|| {
                                    TorshError::AutogradError(format!(
                                        "permute backward received an out-of-range axis {input_axis} \
                                         for a {}-D tensor",
                                        perm.len()
                                    ))
                                })?;
                                *slot = output_axis;
                            }
                            let axes: Vec<i32> = inverse.iter().map(|&axis| axis as i32).collect();
                            let permuted = grad.permute(&axes)?;
                            Self::from_data(permuted.to_vec()?, dims.to_vec(), input.device)?
                        }
                        // Broadcast axes accumulate every copy of the value.
                        ViewKind::Expand => reduce_grad_to_shape(&grad, dims)?,
                    };
                    contributions.push((Arc::clone(input), contribution));
                }
            }
            Operation::Im2Col { input, config } => {
                if input.requires_grad {
                    let contribution = Self::col2im_gradient(&grad_output.detach(), config)?;
                    contributions.push((Arc::clone(input), contribution));
                }
            }
            Operation::Concat { inputs, dim } => {
                // Each input owns a contiguous slab of the output along `dim`;
                // its gradient is that slab of the upstream gradient. The offset
                // advances for every input so positions stay correct even when
                // some inputs do not require a gradient.
                let mut offset = 0usize;
                for input in inputs {
                    let len = input.shape().dims().get(*dim).copied().ok_or_else(|| {
                        TorshError::AutogradError(format!(
                            "cat backward: dim {dim} is out of range for a {:?} input",
                            input.shape().dims()
                        ))
                    })?;
                    if input.requires_grad {
                        let slab = Self::slab_along_dim(grad_output, *dim, offset, len, true)?;
                        contributions.push((Arc::clone(input), slab));
                    }
                    offset += len;
                }
            }
            Operation::Stack { inputs, dim } => {
                // stack inserts a fresh axis; input `i` receives the slice of the
                // gradient at index `i` along that axis, with the axis removed.
                for (index, input) in inputs.iter().enumerate() {
                    if input.requires_grad {
                        let slab = Self::slab_along_dim(grad_output, *dim, index, 1, false)?;
                        contributions.push((Arc::clone(input), slab));
                    }
                }
            }
            Operation::Gather { input, index_map } => {
                if input.requires_grad {
                    let grad = grad_output.to_vec()?;
                    if grad.len() != index_map.len() {
                        return Err(TorshError::AutogradError(format!(
                            "gather backward: gradient has {} elements but the index map has {}",
                            grad.len(),
                            index_map.len()
                        )));
                    }
                    let input_numel = input.numel();
                    let zero = <T as TensorElement>::zero();
                    let mut input_grad = vec![zero; input_numel];
                    for (out_pos, &in_pos) in index_map.iter().enumerate() {
                        let slot = input_grad.get_mut(in_pos).ok_or_else(|| {
                            TorshError::AutogradError(format!(
                                "gather backward: index {in_pos} out of range for a {input_numel}-element input"
                            ))
                        })?;
                        *slot = *slot + grad[out_pos];
                    }
                    let input_shape = input.shape();
                    contributions.push((
                        Arc::clone(input),
                        Self::from_data(input_grad, input_shape.dims().to_vec(), input.device)?,
                    ));
                }
            }
            Operation::LogSoftmax { input, dim } => {
                if input.requires_grad {
                    let contribution = Self::log_softmax_gradient(input, *dim, grad_output)?;
                    contributions.push((Arc::clone(input), contribution));
                }
            }
            Operation::Unary { input, kind } => {
                if input.requires_grad {
                    let local = Self::unary_local_gradient(input, *kind)?;
                    contributions.push((Arc::clone(input), local.mul(&grad_output.detach())?));
                }
            }
            Operation::Custom(op_name, _) => {
                return Err(match op_name.as_str() {
                    "conv1d" | "conv2d" | "conv3d" | "depthwise_conv2d" | "separable_conv2d"
                    | "conv_transpose2d" => TorshError::AutogradError(format!(
                        "backward through '{op_name}' is not implemented: the forward pass records \
                         Operation::Custom with weak (already dropped) input references and none of \
                         the stride/padding/dilation/groups configuration, so no gradient can be \
                         reconstructed. Build the layer from differentiable primitives (im2col + \
                         matmul) or record a dedicated convolution operation instead."
                    )),
                    _ => TorshError::AutogradError(format!(
                        "backward through custom operation '{op_name}' is not implemented: no \
                         derivative is registered for it"
                    )),
                });
            }
        }
        Ok(contributions)
    }

    /// Backward rule of [`Tensor::im2col_2d`]: col2im.
    ///
    /// Each patch element was a copy of one input element, so its gradient is
    /// scatter-added back onto that position. Overlapping windows (stride
    /// smaller than the kernel span) therefore accumulate, which is exactly the
    /// sum over all output positions a given input pixel contributed to.
    fn col2im_gradient(grad_output: &Self, config: &Im2ColConfig) -> Result<Self> {
        let [batch, channels, height, width] = config.input_shape;
        let (kernel_h, kernel_w) = config.kernel;
        let (stride_h, stride_w) = config.stride;
        let (pad_h, pad_w) = config.padding;
        let (dilation_h, dilation_w) = config.dilation;
        let (out_h, out_w) = config.output;
        let groups = config.groups;

        let per_group = channels / groups;
        let patch_len = per_group * kernel_h * kernel_w;
        let rows = batch * out_h * out_w;

        let expected = [groups, rows, patch_len];
        if grad_output.shape().dims() != expected {
            return Err(TorshError::AutogradError(format!(
                "im2col backward expected a {expected:?} gradient, got {:?}",
                grad_output.shape().dims()
            )));
        }

        let grad_data = grad_output.to_vec()?;
        let mut input_grad =
            vec![<T as num_traits::Zero>::zero(); batch * channels * height * width];

        for batch_index in 0..batch {
            for group in 0..groups {
                let group_base = group * rows * patch_len;
                for out_y in 0..out_h {
                    for out_x in 0..out_w {
                        let row = group_base
                            + (batch_index * out_h * out_w + out_y * out_w + out_x) * patch_len;
                        for channel in 0..per_group {
                            let global_channel = group * per_group + channel;
                            let channel_base =
                                (batch_index * channels + global_channel) * height * width;
                            for ky in 0..kernel_h {
                                let in_y = out_y * stride_h + ky * dilation_h;
                                if in_y < pad_h {
                                    continue;
                                }
                                let in_y = in_y - pad_h;
                                if in_y >= height {
                                    continue;
                                }
                                for kx in 0..kernel_w {
                                    let in_x = out_x * stride_w + kx * dilation_w;
                                    if in_x < pad_w {
                                        continue;
                                    }
                                    let in_x = in_x - pad_w;
                                    if in_x >= width {
                                        continue;
                                    }
                                    let column = (channel * kernel_h + ky) * kernel_w + kx;
                                    let target = channel_base + in_y * width + in_x;
                                    input_grad[target] =
                                        input_grad[target] + grad_data[row + column];
                                }
                            }
                        }
                    }
                }
            }
        }

        Self::from_data(
            input_grad,
            vec![batch, channels, height, width],
            grad_output.device,
        )
    }

    /// Gradients of a `torch.matmul`-style product for operands of any rank.
    ///
    /// 1-D operands are promoted the same way the forward pass promotes them
    /// (`[k] -> [1, k]` on the left, `[k] -> [k, 1]` on the right), batch axes are
    /// broadcast, and each operand's gradient is accumulated over the batch axes
    /// it was broadcast along:
    ///
    /// ```text
    /// dL/dlhs[b] = grad[b] @ rhs[b]^T
    /// dL/drhs[b] = lhs[b]^T @ grad[b]
    /// ```
    fn matmul_gradients(
        lhs: &Arc<Self>,
        rhs: &Arc<Self>,
        grad_output: &Self,
    ) -> Result<Vec<(Arc<Self>, Self)>> {
        let lhs_shape_binding = lhs.shape();
        let lhs_shape = lhs_shape_binding.dims().to_vec();
        let rhs_shape_binding = rhs.shape();
        let rhs_shape = rhs_shape_binding.dims().to_vec();
        if lhs_shape.is_empty() || rhs_shape.is_empty() {
            return Err(TorshError::AutogradError(
                "matmul backward requires operands with at least one dimension".to_string(),
            ));
        }

        let mut lhs_dims = lhs_shape.clone();
        if lhs_dims.len() == 1 {
            lhs_dims.insert(0, 1);
        }
        let mut rhs_dims = rhs_shape.clone();
        if rhs_dims.len() == 1 {
            rhs_dims.push(1);
        }

        let m = lhs_dims[lhs_dims.len() - 2];
        let k = lhs_dims[lhs_dims.len() - 1];
        let k_rhs = rhs_dims[rhs_dims.len() - 2];
        let n = rhs_dims[rhs_dims.len() - 1];
        if k != k_rhs {
            return Err(TorshError::ShapeMismatch {
                expected: lhs_shape,
                got: rhs_shape,
            });
        }

        let lhs_batch = lhs_dims[..lhs_dims.len() - 2].to_vec();
        let rhs_batch = rhs_dims[..rhs_dims.len() - 2].to_vec();
        let batch_shape = broadcast_batch_dims(&lhs_batch, &rhs_batch)?;
        let batch_count: usize = batch_shape.iter().product();

        if grad_output.numel() != batch_count * m * n {
            return Err(TorshError::ShapeMismatch {
                expected: vec![batch_count, m, n],
                got: grad_output.shape().dims().to_vec(),
            });
        }

        let needs_lhs = lhs.requires_grad;
        let needs_rhs = rhs.requires_grad;
        if !needs_lhs && !needs_rhs {
            return Ok(Vec::new());
        }

        let zero = <T as TensorElement>::zero();
        let grad_data = grad_output.to_vec()?;
        // Materialise one operand at a time: both may be backed by the same lock.
        let lhs_data = lhs.to_vec()?;
        let rhs_data = rhs.to_vec()?;
        let mut grad_lhs = if needs_lhs {
            vec![zero; lhs_data.len()]
        } else {
            Vec::new()
        };
        let mut grad_rhs = if needs_rhs {
            vec![zero; rhs_data.len()]
        } else {
            Vec::new()
        };

        let mut coords = vec![0usize; batch_shape.len()];
        for batch in 0..batch_count {
            let lhs_offset = batch_offset(&coords, &lhs_batch, m * k);
            let rhs_offset = batch_offset(&coords, &rhs_batch, k * n);
            let grad_offset = batch * m * n;

            if needs_lhs {
                // grad_lhs[i, p] += sum_j grad[i, j] * rhs[p, j]
                for i in 0..m {
                    for p in 0..k {
                        let mut acc = zero;
                        for j in 0..n {
                            acc = acc
                                + grad_data[grad_offset + i * n + j]
                                    * rhs_data[rhs_offset + p * n + j];
                        }
                        let slot = lhs_offset + i * k + p;
                        grad_lhs[slot] = grad_lhs[slot] + acc;
                    }
                }
            }
            if needs_rhs {
                // grad_rhs[p, j] += sum_i lhs[i, p] * grad[i, j]
                for p in 0..k {
                    for j in 0..n {
                        let mut acc = zero;
                        for i in 0..m {
                            acc = acc
                                + lhs_data[lhs_offset + i * k + p]
                                    * grad_data[grad_offset + i * n + j];
                        }
                        let slot = rhs_offset + p * n + j;
                        grad_rhs[slot] = grad_rhs[slot] + acc;
                    }
                }
            }

            if !coords.is_empty() {
                increment_coords(&mut coords, &batch_shape);
            }
        }

        let mut contributions = Vec::new();
        if needs_lhs {
            contributions.push((
                Arc::clone(lhs),
                Self::from_data(grad_lhs, lhs_shape, lhs.device)?,
            ));
        }
        if needs_rhs {
            contributions.push((
                Arc::clone(rhs),
                Self::from_data(grad_rhs, rhs_shape, rhs.device)?,
            ));
        }
        Ok(contributions)
    }

    /// Gather one contiguous slab `[start, start + length)` along `dim` out of
    /// `grad`, returning a fresh contiguous tensor.
    ///
    /// With `keep_dim` the sliced axis is retained (used by `cat`, whose inputs
    /// keep their extent along `dim`); without it the axis is dropped, which is
    /// the per-index slice `stack` needs (it always slices `length == 1`).
    fn slab_along_dim(
        grad: &Self,
        dim: usize,
        start: usize,
        length: usize,
        keep_dim: bool,
    ) -> Result<Self> {
        let shape = grad.shape();
        let dims = shape.dims();
        if dim >= dims.len() {
            return Err(TorshError::AutogradError(format!(
                "cat/stack backward: dim {dim} is out of range for a {dims:?} gradient"
            )));
        }
        let dim_size = dims[dim];
        if start + length > dim_size {
            return Err(TorshError::AutogradError(format!(
                "cat/stack backward: slice [{start}, {}) exceeds axis {dim} of size {dim_size}",
                start + length
            )));
        }
        let outer: usize = dims[..dim].iter().product();
        let inner: usize = dims[dim + 1..].iter().product();
        let data = grad.to_vec()?;
        let mut out = Vec::with_capacity(outer * length * inner);
        for o in 0..outer {
            for d in start..start + length {
                let base = (o * dim_size + d) * inner;
                out.extend_from_slice(&data[base..base + inner]);
            }
        }
        let mut new_dims = dims.to_vec();
        if keep_dim {
            new_dims[dim] = length;
        } else {
            new_dims.remove(dim);
        }
        Self::from_data(out, new_dims, grad.device)
    }

    /// Backward rule of [`Tensor::log_softmax`] along `dim`.
    ///
    /// `dL/dx = g - softmax(x) * sum(g, dim, keepdim)`, computed slice-by-slice
    /// on detached data with a stable (max-shifted) softmax so no intermediate
    /// under/overflows and no node is recorded.
    fn log_softmax_gradient(input: &Arc<Self>, dim: usize, grad_output: &Self) -> Result<Self> {
        let shape = input.shape();
        let dims = shape.dims();
        if dim >= dims.len() {
            return Err(TorshError::AutogradError(format!(
                "log_softmax backward: dim {dim} is out of range for a {dims:?} input"
            )));
        }
        let dim_size = dims[dim];
        let outer: usize = dims[..dim].iter().product();
        let inner: usize = dims[dim + 1..].iter().product();
        let x = input.to_vec()?;
        let g = grad_output.to_vec()?;
        if g.len() != x.len() {
            return Err(TorshError::AutogradError(format!(
                "log_softmax backward: gradient has {} elements but the input has {}",
                g.len(),
                x.len()
            )));
        }
        let zero = <T as TensorElement>::zero();
        let mut out = vec![zero; x.len()];
        for o in 0..outer {
            for n in 0..inner {
                let base = o * dim_size * inner + n;
                // Stable softmax and the per-slice gradient sum.
                let mut max = x[base];
                for d in 1..dim_size {
                    let v = x[base + d * inner];
                    if v > max {
                        max = v;
                    }
                }
                let mut denom = zero;
                let mut grad_sum = zero;
                for d in 0..dim_size {
                    denom = denom + (x[base + d * inner] - max).exp();
                    grad_sum = grad_sum + g[base + d * inner];
                }
                for d in 0..dim_size {
                    let idx = base + d * inner;
                    let softmax = (x[idx] - max).exp() / denom;
                    out[idx] = g[idx] - softmax * grad_sum;
                }
            }
        }
        Self::from_data(out, dims.to_vec(), input.device)
    }

    /// Local derivative `df/dx` of a recorded [`UnaryKind`], evaluated at every
    /// element of `input`. The caller multiplies it by the upstream gradient.
    fn unary_local_gradient(input: &Arc<Self>, kind: UnaryKind) -> Result<Self> {
        let shape = input.shape();
        let x = input.to_vec()?;
        let zero = <T as TensorElement>::zero();
        let one = Self::scalar_from_f64(1.0)?;
        let two = Self::scalar_from_f64(2.0)?;
        let local: Vec<T> = x
            .iter()
            .map(|&v| match kind {
                UnaryKind::Exp => v.exp(),
                UnaryKind::Ln => one / v,
                UnaryKind::Sqrt => one / (two * v.sqrt()),
                UnaryKind::Sin => v.cos(),
                UnaryKind::Cos => zero - v.sin(),
                UnaryKind::Tanh => {
                    let t = v.tanh();
                    one - t * t
                }
                UnaryKind::Sigmoid => {
                    let s = one / (one + (zero - v).exp());
                    s * (one - s)
                }
                UnaryKind::Relu => {
                    if v > zero {
                        one
                    } else {
                        zero
                    }
                }
            })
            .collect();
        Self::from_data(local, shape.dims().to_vec(), input.device)
    }

    /// Convert an `f64` constant into the tensor's element type.
    fn scalar_from_f64(value: f64) -> Result<T> {
        <T as TensorElement>::from_f64(value).ok_or_else(|| {
            TorshError::AutogradError(format!(
                "element type {:?} cannot represent the constant {value}",
                <T as TensorElement>::dtype()
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    fn tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
        Tensor::from_data(data, shape, DeviceType::Cpu).expect("tensor creation should succeed")
    }

    #[test]
    fn reduce_grad_to_shape_sums_leading_and_unit_axes() {
        let grad = tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        let to_bias = reduce_grad_to_shape(&grad, &[3]).expect("reduce should succeed");
        assert_eq!(to_bias.shape().dims(), &[3]);
        assert_eq!(to_bias.to_vec().expect("to_vec"), vec![5.0, 7.0, 9.0]);

        let keepdim = reduce_grad_to_shape(&grad, &[2, 1]).expect("reduce should succeed");
        assert_eq!(keepdim.shape().dims(), &[2, 1]);
        assert_eq!(keepdim.to_vec().expect("to_vec"), vec![6.0, 15.0]);

        let scalar = reduce_grad_to_shape(&grad, &[]).expect("reduce should succeed");
        assert_eq!(scalar.to_vec().expect("to_vec"), vec![21.0]);
    }

    #[test]
    fn expand_grad_to_shape_replicates_along_broadcast_axes() {
        let grad = tensor(vec![1.0, 2.0], vec![2, 1]);
        let expanded = expand_grad_to_shape(&grad, &[2, 3]).expect("expand should succeed");
        assert_eq!(expanded.shape().dims(), &[2, 3]);
        assert_eq!(
            expanded.to_vec().expect("to_vec"),
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
        );

        // A gradient whose axes cannot be right-aligned is rejected.
        let ambiguous = tensor(vec![1.0, 2.0], vec![2]);
        assert!(expand_grad_to_shape(&ambiguous, &[2, 3]).is_err());
    }

    /// F063: batched matmul must produce real gradients, not silence.
    ///
    /// The node is assembled by hand so this test exercises the backward rule
    /// alone; `Tensor::matmul` records the same node for every supported rank
    /// (see `matmul_ops::tests::matmul_batched_autograd_is_recorded`).
    #[test]
    fn matmul_backward_handles_batched_operands() {
        let lhs = tensor((1..=12).map(|v| v as f32).collect(), vec![2, 2, 3]).requires_grad_(true);
        let rhs = tensor((1..=6).map(|v| v as f32).collect(), vec![3, 2]).requires_grad_(true);

        let mut product = lhs
            .basic_matmul(&rhs)
            .expect("batched matmul should succeed");
        assert_eq!(product.shape().dims(), &[2, 2, 2]);
        product.requires_grad = true;
        product.operation = Operation::MatMul {
            lhs: Arc::new(lhs.clone()),
            rhs: Arc::new(rhs.clone()),
        };

        product.sum().expect("sum").backward().expect("backward");

        // d(sum(lhs @ rhs))/dlhs = ones @ rhs^T -> row sums of rhs, per position.
        let lhs_grad = lhs.grad().expect("lhs gradient").to_vec().expect("to_vec");
        assert_eq!(lhs_grad.len(), 12);
        for (index, value) in lhs_grad.iter().enumerate() {
            let column = index % 3;
            let expected = (1..=2).map(|j| (column * 2 + j) as f32).sum::<f32>();
            assert!(
                (value - expected).abs() < 1e-4,
                "lhs grad[{index}] = {value}, expected {expected}"
            );
        }

        // rhs is broadcast across both batches, so its gradient accumulates the
        // column sums of lhs over the whole batch.
        let rhs_grad = rhs.grad().expect("rhs gradient").to_vec().expect("to_vec");
        let lhs_data: Vec<f32> = (1..=12).map(|v| v as f32).collect();
        for p in 0..3usize {
            let expected: f32 = (0..4).map(|row| lhs_data[row * 3 + p]).sum();
            for j in 0..2usize {
                let got = rhs_grad[p * 2 + j];
                assert!(
                    (got - expected).abs() < 1e-4,
                    "rhs grad[{p},{j}] = {got}, expected {expected}"
                );
            }
        }
    }

    /// 1-D operands are promoted, so `[k] @ [k, n]` must give a `[k]` gradient.
    #[test]
    fn matmul_backward_handles_vector_operands() {
        let lhs = tensor(vec![1.0, 2.0, 3.0], vec![3]).requires_grad_(true);
        let rhs = tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).requires_grad_(true);

        let mut product = lhs.basic_matmul(&rhs).expect("matmul should succeed");
        assert_eq!(product.shape().dims(), &[2]);
        product.requires_grad = true;
        product.operation = Operation::MatMul {
            lhs: Arc::new(lhs.clone()),
            rhs: Arc::new(rhs.clone()),
        };

        product.sum().expect("sum").backward().expect("backward");

        let lhs_grad = lhs.grad().expect("lhs gradient").to_vec().expect("to_vec");
        assert_eq!(lhs_grad, vec![3.0, 7.0, 11.0]);
        let rhs_grad = rhs.grad().expect("rhs gradient").to_vec().expect("to_vec");
        assert_eq!(rhs_grad, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    }

    /// F060/F064: convolution records a `Custom` node whose derivative does not
    /// exist; backward must say so instead of returning a zero gradient.
    #[test]
    fn conv_backward_reports_the_missing_derivative() {
        let input = tensor(vec![1.0; 9], vec![1, 1, 3, 3]).requires_grad_(true);
        let weight = tensor(vec![1.0; 4], vec![1, 1, 2, 2]).requires_grad_(true);
        let output = input
            .conv2d(&weight, None, (1, 1), (0, 0), (1, 1), 1)
            .expect("conv2d should succeed");
        assert!(output.requires_grad());

        let error = output
            .sum()
            .expect("sum")
            .backward()
            .expect_err("conv backward must not silently succeed");
        let message = error.to_string();
        assert!(
            message.contains("conv2d"),
            "error should name the operation, got: {message}"
        );
    }
}
