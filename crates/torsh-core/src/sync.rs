//! Poison-recovery helpers for `std::sync::{Mutex, RwLock}`.
//!
//! # Policy
//!
//! A [`std::sync::Mutex`] or [`std::sync::RwLock`] is *poisoned* when a
//! thread panics while holding the guard. By default the next `.lock()` /
//! `.read()` / `.write()` call returns `Err`, and the historical pattern in
//! this codebase has been to `.expect("... should not be poisoned")` that
//! `Err` away — which turns an unrelated panic on *any* thread that ever
//! touched the lock into a hard abort for every *other* thread that later
//! tries to acquire it, even though the guarded data itself is untouched.
//!
//! For the mutexes and rwlocks in ToRSh's hot paths (pools, caches, metrics,
//! device registries, ...) a panic while holding the guard never leaves the
//! protected data structurally invalid: the panicking paths validate their
//! arguments *before* mutating shared state, or the state is a plain
//! `HashMap`/`Vec`/counter that has no invariant a partial mutation could
//! break. `torsh-tensor`'s `memory_pool.rs` already relies on this via
//! `unwrap_or_else(|poisoned| poisoned.into_inner())` in over a dozen call
//! sites; this module gives that idiom a name and a single definition so it
//! can be applied uniformly instead of hand-rolled per call site.
//!
//! Recovering via [`std::sync::PoisonError::into_inner`] does **not** clear
//! the lock's poisoned flag — every *subsequent* `.lock()` on that same
//! mutex will still return `Err` too, and this module recovers those the
//! same way. That is intentional: poisoning is a one-shot signal ("a panic
//! happened while this was held at least once"), not a value that should be
//! consumed by only the first caller to observe it.
//!
//! Do **not** use these helpers for a lock whose documented contract is that
//! poisoning must stay fatal (for example a CSPRNG or another invariant that
//! genuinely cannot be trusted after a panic mid-mutation). Such call sites
//! should keep their `.expect(...)` and a `# Panics` doc comment explaining
//! why recovery is unsafe there.
//!
//! # Two ways to call this
//!
//! Free functions ([`lock_or_recover`], [`read_or_recover`],
//! [`write_or_recover`]) take `&Mutex<T>` / `&RwLock<T>` directly.
//! [`MutexExt`] and [`RwLockExt`] provide the same operations as methods on
//! the lock itself, which is what most call sites in this codebase use
//! (`self.field.lock_or_recover()`) since it is a drop-in replacement for
//! `self.field.lock().expect("...")` that never needs to re-parenthesize the
//! receiver expression.

use std::sync::{LockResult, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Recover a [`LockResult`] by taking the guard even if the lock was
/// poisoned.
///
/// This is the single primitive the other helpers in this module are built
/// from: `PoisonError::into_inner()` hands back the guard that was being
/// held at the moment of the panic, and since ToRSh's guarded data never
/// becomes structurally invalid mid-panic (see the [module docs](self)),
/// treating a poisoned lock the same as a healthy one is safe here.
#[inline]
pub fn recover<Guard>(result: LockResult<Guard>) -> Guard {
    result.unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Lock `mutex`, recovering the guard even if a prior panic poisoned it.
///
/// Equivalent to `mutex.lock().expect("... should not be poisoned")`, but
/// never aborts: a poisoned lock's guard is still returned so callers keep
/// working instead of propagating an unrelated panic.
#[inline]
pub fn lock_or_recover<T: ?Sized>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    recover(mutex.lock())
}

/// Acquire `lock` for reading, recovering the guard even if a prior panic
/// poisoned it.
///
/// Equivalent to `lock.read().expect("... should not be poisoned")`, but
/// never aborts.
#[inline]
pub fn read_or_recover<T: ?Sized>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    recover(lock.read())
}

/// Acquire `lock` for writing, recovering the guard even if a prior panic
/// poisoned it.
///
/// Equivalent to `lock.write().expect("... should not be poisoned")`, but
/// never aborts.
#[inline]
pub fn write_or_recover<T: ?Sized>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    recover(lock.write())
}

/// Method-call sugar for [`lock_or_recover`] on [`Mutex<T>`].
///
/// Imported alongside [`RwLockExt`], this lets an existing
/// `expr.lock().expect("...")` call site convert to
/// `expr.lock_or_recover()` without touching `expr` itself, regardless of
/// how complex that receiver expression is.
pub trait MutexExt<T: ?Sized> {
    /// Lock `self`, recovering the guard even if poisoned. See
    /// [`lock_or_recover`].
    fn lock_or_recover(&self) -> MutexGuard<'_, T>;
}

impl<T: ?Sized> MutexExt<T> for Mutex<T> {
    #[inline]
    fn lock_or_recover(&self) -> MutexGuard<'_, T> {
        lock_or_recover(self)
    }
}

/// Method-call sugar for [`read_or_recover`] / [`write_or_recover`] on
/// [`RwLock<T>`].
///
/// See [`MutexExt`] for the matching `Mutex` helper.
pub trait RwLockExt<T: ?Sized> {
    /// Acquire `self` for reading, recovering the guard even if poisoned.
    /// See [`read_or_recover`].
    fn read_or_recover(&self) -> RwLockReadGuard<'_, T>;

    /// Acquire `self` for writing, recovering the guard even if poisoned.
    /// See [`write_or_recover`].
    fn write_or_recover(&self) -> RwLockWriteGuard<'_, T>;
}

impl<T: ?Sized> RwLockExt<T> for RwLock<T> {
    #[inline]
    fn read_or_recover(&self) -> RwLockReadGuard<'_, T> {
        read_or_recover(self)
    }

    #[inline]
    fn write_or_recover(&self) -> RwLockWriteGuard<'_, T> {
        write_or_recover(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// A poisoned `Mutex` recovers through `lock_or_recover` (free function
    /// and trait method) instead of the process aborting the way
    /// `.lock().expect(...)` would.
    #[test]
    fn mutex_recovers_via_free_function_and_ext_trait() {
        let mutex = Arc::new(Mutex::new(vec![1, 2, 3]));

        // Poison the mutex: panic while holding the guard on another thread.
        let poisoner = Arc::clone(&mutex);
        let joined = std::thread::spawn(move || {
            let _guard = poisoner.lock().expect("thread must acquire the lock first");
            panic!("intentional poison for test");
        })
        .join();
        assert!(joined.is_err(), "poisoning thread should have panicked");

        // A plain `.lock()` now observes the poison...
        assert!(
            mutex.lock().is_err(),
            "mutex should be poisoned after the panic"
        );

        // ...but `lock_or_recover` still returns a usable guard with the
        // data intact (the panic happened before any mutation).
        let guard = lock_or_recover(&mutex);
        assert_eq!(*guard, vec![1, 2, 3]);
        drop(guard);

        // The trait method form behaves identically.
        let guard = mutex.lock_or_recover();
        assert_eq!(*guard, vec![1, 2, 3]);
    }

    /// A poisoned `RwLock` recovers through both `read_or_recover` and
    /// `write_or_recover` (free functions and trait methods), and the
    /// recovered write guard can still mutate the protected data.
    #[test]
    fn rwlock_recovers_read_and_write_via_free_function_and_ext_trait() {
        let lock = Arc::new(RwLock::new(10_i32));

        let poisoner = Arc::clone(&lock);
        let joined = std::thread::spawn(move || {
            let _guard = poisoner
                .write()
                .expect("thread must acquire the lock first");
            panic!("intentional poison for test");
        })
        .join();
        assert!(joined.is_err(), "poisoning thread should have panicked");

        assert!(lock.read().is_err(), "rwlock should be poisoned");

        // Free-function recovery for both read and write.
        assert_eq!(*read_or_recover(&lock), 10);
        *write_or_recover(&lock) += 5;
        assert_eq!(*read_or_recover(&lock), 15);

        // Trait-method recovery sees the mutation and can keep mutating.
        assert_eq!(*lock.read_or_recover(), 15);
        *lock.write_or_recover() += 1;
        assert_eq!(*lock.read_or_recover(), 16);
    }
}
