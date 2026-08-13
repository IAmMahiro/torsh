//! In-place parameter update primitives.
//!
//! # Why optimizers must not rebind `*param`
//!
//! The obvious way to write an optimizer update is
//!
//! ```ignore
//! *param = param.sub(&update)?;
//! ```
//!
//! but that is wrong in two independent ways:
//!
//! 1. **The gradient is thrown away.** [`Tensor::sub`] builds a *new* tensor via
//!    `from_data`, which gets a fresh, empty gradient slot. After the first
//!    `step()` the parameter's `has_grad()` is `false`, so every subsequent
//!    `step()` skips it entirely — training silently freezes after one update.
//! 2. **The autograd graph grows without bound.** `sub` records
//!    `Operation::Sub { lhs: Arc::new(self.clone()), .. }`, so the post-step
//!    parameter holds a strong reference to the pre-step parameter. After `N`
//!    steps the optimizer is pinning `N` full copies of every parameter, and the
//!    live parameter is no longer an `Operation::Leaf`, so `backward()` routes
//!    its gradient to a discarded clone instead of accumulating into it.
//!
//! PyTorch avoids both by mutating parameter *data* under `torch.no_grad()`.
//! The helpers in this module do the same: they mutate the parameter's storage
//! and leave its shape, device, gradient slot and graph node untouched.
//!
//! # How the no-grad mutation is performed
//!
//! [`Tensor`]'s in-place ops deliberately refuse to run on a tensor with
//! `requires_grad == true` (autograd cannot be tracked through a destructive
//! write). An optimizer update is by definition outside the graph, so these
//! helpers temporarily clear the flag, perform the write, and restore it. The
//! tensor is *moved* out of its slot rather than cloned, so the storage handle,
//! the `Arc` holding the gradient and the recorded operation all survive.

use torsh_core::error::Result;
use torsh_tensor::Tensor;

/// Run `f` against `param` with gradient tracking temporarily disabled, then
/// restore the original tracking flag.
///
/// The parameter is moved (not cloned) into the scratch binding, so its storage
/// handle, gradient slot and graph node identity are preserved across the call.
/// If `f` fails, the flag is still restored before the error is propagated.
fn with_grad_tracking_disabled<F>(param: &mut Tensor, f: F) -> Result<()>
where
    F: FnOnce(&mut Tensor) -> Result<()>,
{
    if !param.requires_grad() {
        return f(param);
    }

    // A one-element stand-in keeps `param` in a valid state while its real
    // tensor is moved out; it is dropped again before this function returns.
    let placeholder = Tensor::zeros(&[1], param.device())?;
    let taken = std::mem::replace(param, placeholder);
    let mut scratch = taken.requires_grad_(false);
    let outcome = f(&mut scratch);
    *param = scratch.requires_grad_(true);
    outcome
}

/// `param -= update`, in place.
///
/// The parameter keeps its identity: same storage handle, same gradient, same
/// `requires_grad` flag, and no new autograd graph node.
///
/// # Errors
/// Returns an error if `update` is not broadcast-compatible with `param`, or if
/// the parameter's storage does not support mutation.
pub fn sub_assign(param: &mut Tensor, update: &Tensor) -> Result<()> {
    with_grad_tracking_disabled(param, |p| p.sub_(update).map(|_| ()))
}

/// `param += update`, in place. See [`sub_assign`] for the identity guarantees.
///
/// # Errors
/// Returns an error if `update` is not broadcast-compatible with `param`, or if
/// the parameter's storage does not support mutation.
pub fn add_assign(param: &mut Tensor, update: &Tensor) -> Result<()> {
    with_grad_tracking_disabled(param, |p| p.add_(update).map(|_| ()))
}

/// `param *= factor`, in place. See [`sub_assign`] for the identity guarantees.
///
/// # Errors
/// Returns an error if the parameter's storage does not support mutation.
pub fn scale(param: &mut Tensor, factor: f32) -> Result<()> {
    with_grad_tracking_disabled(param, |p| p.mul_scalar_(factor))
}

/// Overwrite `param`'s data with `src`'s data, in place.
///
/// Used by optimizers that compute a whole new parameter value (proximal
/// operators, weight clamping, slow-weight interpolation, flattened-vector
/// writeback) instead of an additive update.
///
/// Implemented as "zero, then add" rather than `param -= (param - src)`: the
/// subtractive round trip loses precision through catastrophic cancellation
/// whenever `param` and `src` differ in magnitude (overwriting `1000.0` with
/// `0.001` lands roughly 2% off in f32), and an overwrite must be exact. Both
/// passes go through the same identity-preserving, copy-on-write path as
/// [`sub_assign`].
///
/// # Non-finite values
/// A parameter that already holds `NaN` or an infinity is *not* cleanly
/// overwritten: `0.0 * NaN` and `0.0 * inf` are both `NaN`, so the result stays
/// `NaN`. Optimizers must not let a parameter go non-finite in the first place.
///
/// # Errors
/// Returns an error if `src`'s shape does not match `param`'s, or if the
/// parameter's storage does not support mutation.
pub fn assign(param: &mut Tensor, src: &Tensor) -> Result<()> {
    if param.shape().dims() != src.shape().dims() {
        return Err(torsh_core::error::TorshError::ShapeMismatch {
            expected: param.shape().dims().to_vec(),
            got: src.shape().dims().to_vec(),
        });
    }
    with_grad_tracking_disabled(param, |p| {
        p.mul_scalar_(0.0)?;
        p.add_(src).map(|_| ())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    fn tensor(data: Vec<f32>) -> Tensor {
        let len = data.len();
        Tensor::from_data(data, vec![len], DeviceType::Cpu).expect("tensor creation")
    }

    #[test]
    fn sub_assign_preserves_gradient_and_flag() -> Result<()> {
        let mut param = tensor(vec![1.0, 2.0, 3.0]).requires_grad_(true);
        param.set_grad(Some(tensor(vec![7.0, 7.0, 7.0])));

        sub_assign(&mut param, &tensor(vec![0.5, 0.5, 0.5]))?;

        assert_eq!(param.to_vec()?, vec![0.5, 1.5, 2.5]);
        assert!(param.requires_grad());
        assert!(param.has_grad());
        assert_eq!(
            param.grad().expect("gradient preserved").to_vec()?,
            vec![7.0, 7.0, 7.0]
        );
        Ok(())
    }

    #[test]
    fn repeated_updates_accumulate() -> Result<()> {
        let mut param = tensor(vec![0.0]).requires_grad_(true);
        for _ in 0..4 {
            sub_assign(&mut param, &tensor(vec![0.25]))?;
        }
        assert_eq!(param.to_vec()?, vec![-1.0]);
        Ok(())
    }

    #[test]
    fn assign_overwrites_values() -> Result<()> {
        let mut param = tensor(vec![1.0, 2.0]).requires_grad_(true);
        assign(&mut param, &tensor(vec![-3.0, 9.0]))?;
        assert_eq!(param.to_vec()?, vec![-3.0, 9.0]);
        assert!(param.requires_grad());
        Ok(())
    }

    #[test]
    fn assign_is_exact_across_magnitudes() -> Result<()> {
        // A subtractive round trip (`param -= param - src`) loses ~2% here.
        let mut param = tensor(vec![1000.0, -1e6, 3.0]).requires_grad_(true);
        assign(&mut param, &tensor(vec![0.001, 1e-7, -2.5]))?;
        assert_eq!(param.to_vec()?, vec![0.001, 1e-7, -2.5]);
        Ok(())
    }

    #[test]
    fn assign_leaves_snapshots_untouched() -> Result<()> {
        // In-place writes are copy-on-write: an earlier `clone()` keeps the old
        // values (this is what optimizer state snapshots rely on).
        let mut param = tensor(vec![1.0, 2.0]).requires_grad_(true);
        let snapshot = param.clone();
        assign(&mut param, &tensor(vec![9.0, 9.0]))?;
        assert_eq!(param.to_vec()?, vec![9.0, 9.0]);
        assert_eq!(snapshot.to_vec()?, vec![1.0, 2.0]);
        Ok(())
    }

    #[test]
    fn assign_rejects_shape_mismatch() {
        let mut param = tensor(vec![1.0, 2.0]);
        assert!(assign(&mut param, &tensor(vec![1.0])).is_err());
    }

    #[test]
    fn add_assign_and_scale_round_trip() -> Result<()> {
        let mut param = tensor(vec![1.0, -1.0]).requires_grad_(true);
        add_assign(&mut param, &tensor(vec![1.0, 1.0]))?;
        assert_eq!(param.to_vec()?, vec![2.0, 0.0]);
        scale(&mut param, 3.0)?;
        assert_eq!(param.to_vec()?, vec![6.0, 0.0]);
        assert!(param.requires_grad());
        Ok(())
    }
}
