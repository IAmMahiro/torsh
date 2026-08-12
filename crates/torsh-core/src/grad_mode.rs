//! Global gradient-recording mode.
//!
//! This is the single source of truth that decides whether tensor operations
//! record autograd graph nodes. It lives in `torsh-core` — the crate at the
//! bottom of the dependency graph — precisely so that both ends of the autograd
//! stack can see it:
//!
//! * `torsh-autograd` owns the user-facing guards (`no_grad()`, `enable_grad()`,
//!   `inference_mode()`), which push and pop states here.
//! * `torsh-tensor` consults [`is_grad_enabled`] before recording an operation,
//!   so a guard actually suppresses graph construction instead of only flipping
//!   a flag nobody reads.
//!
//! # Semantics
//!
//! The mode is **process-global** (not thread-local) and nestable: guards push
//! the requested state and pop back to the previous one, so
//! `no_grad()`-inside-`enable_grad()` restores correctly. [`set_grad_enabled`]
//! is the non-scoped escape hatch and clears the nesting stack, matching the
//! behaviour the autograd crate has always exposed.
//!
//! # Cost
//!
//! [`is_grad_enabled`] sits on the hot path of every tensor operation, so it
//! reads a mirrored [`AtomicBool`] with [`Ordering::Relaxed`] — a single plain
//! load on every real target, no lock acquisition. The lock is taken only by
//! the (rare) mutating entry points, which also refresh the mirror.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::RwLock;

/// Nesting stack for gradient mode.
///
/// Only touched by the mutating entry points; readers use [`GRAD_ENABLED`].
struct GradMode {
    /// Saved states of enclosing scopes, innermost last.
    enabled_stack: Vec<bool>,
    /// The state currently in effect.
    current_enabled: bool,
}

impl GradMode {
    const fn new() -> Self {
        Self {
            enabled_stack: Vec::new(),
            current_enabled: true,
        }
    }
}

/// Authoritative nesting state.
static GRAD_MODE: RwLock<GradMode> = RwLock::new(GradMode::new());

/// Lock-free mirror of `GRAD_MODE.current_enabled` for the read hot path.
static GRAD_ENABLED: AtomicBool = AtomicBool::new(true);

/// Run `f` against the gradient-mode stack and re-publish the resulting state.
///
/// A poisoned lock cannot leave the stack in an unusable state here (the guards
/// that touch it never panic while holding it), so the poison is recovered from
/// rather than propagated: gradient mode must never become an error source for
/// unrelated tensor code.
fn with_mode<R>(f: impl FnOnce(&mut GradMode) -> R) -> R {
    let mut mode = match GRAD_MODE.write() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    let result = f(&mut mode);
    GRAD_ENABLED.store(mode.current_enabled, Ordering::Relaxed);
    result
}

/// Check whether gradient recording is currently enabled.
///
/// Tensor operations must consult this before recording a graph node, so that
/// `no_grad()` scopes genuinely produce detached results.
#[inline]
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.load(Ordering::Relaxed)
}

/// Set gradient mode unconditionally, discarding any nesting.
///
/// Prefer the scoped guards in `torsh-autograd`; this is the flat
/// `torch.set_grad_enabled(mode)` equivalent.
pub fn set_grad_enabled(enabled: bool) {
    with_mode(|mode| {
        mode.current_enabled = enabled;
        mode.enabled_stack.clear();
    });
}

/// Enter a nested gradient-mode scope, remembering the current state.
pub fn push_grad_enabled(enabled: bool) {
    with_mode(|mode| {
        mode.enabled_stack.push(mode.current_enabled);
        mode.current_enabled = enabled;
    });
}

/// Leave the innermost gradient-mode scope, restoring the previous state.
///
/// Popping with an empty stack leaves the current state untouched.
pub fn pop_grad_enabled() {
    with_mode(|mode| {
        if let Some(previous) = mode.enabled_stack.pop() {
            mode.current_enabled = previous;
        }
    });
}

/// Run `f` with gradient mode temporarily set to `enabled`.
pub fn with_grad_mode<F, R>(enabled: bool, f: F) -> R
where
    F: FnOnce() -> R,
{
    push_grad_enabled(enabled);
    let result = f();
    pop_grad_enabled();
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, MutexGuard};

    /// Gradient mode is global, so these tests must not interleave.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn serial() -> MutexGuard<'static, ()> {
        let guard = match TEST_LOCK.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        set_grad_enabled(true);
        guard
    }

    #[test]
    fn default_is_enabled() {
        let _serial = serial();
        assert!(is_grad_enabled());
    }

    #[test]
    fn push_pop_restores_previous_state() {
        let _serial = serial();
        push_grad_enabled(false);
        assert!(!is_grad_enabled());
        push_grad_enabled(true);
        assert!(is_grad_enabled());
        pop_grad_enabled();
        assert!(!is_grad_enabled());
        pop_grad_enabled();
        assert!(is_grad_enabled());
    }

    #[test]
    fn set_clears_nesting() {
        let _serial = serial();
        push_grad_enabled(false);
        set_grad_enabled(true);
        pop_grad_enabled();
        assert!(is_grad_enabled());
    }

    #[test]
    fn with_grad_mode_scopes_the_change() {
        let _serial = serial();
        let observed = with_grad_mode(false, is_grad_enabled);
        assert!(!observed);
        assert!(is_grad_enabled());
    }

    #[test]
    fn state_is_visible_across_threads() {
        let _serial = serial();
        push_grad_enabled(false);
        let observed = std::thread::spawn(is_grad_enabled)
            .join()
            .unwrap_or_else(|_| panic!("observer thread panicked"));
        pop_grad_enabled();
        assert!(!observed, "gradient mode must stay process-global");
    }
}
