//! Scope-based profiling utilities
//!
//! This module provides RAII-based profiling scope guards that automatically
//! capture timing information when entering and leaving scopes.

use crate::{core::profiler::global_profiler, ProfileEvent};
use backtrace::Backtrace;
use std::cell::RefCell;
use std::time::Instant;

thread_local! {
    /// Stack of "time already attributed to children" accumulators, one
    /// entry per currently-open `ScopeGuard`/`MetricsScope` on this
    /// thread (mirrors the real RAII call stack exactly, so cross-thread
    /// concurrency never confuses one thread's nesting with another's).
    ///
    /// Used to compute each scope's *exclusive* (self) time: its own
    /// inclusive duration minus whatever its direct children already
    /// contributed. See [`push_scope_frame`] / [`pop_scope_frame`].
    static SCOPE_CHILDREN_US: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
}

/// Push a new "children time" accumulator frame when a scope starts.
fn push_scope_frame() {
    SCOPE_CHILDREN_US.with(|stack| stack.borrow_mut().push(0));
}

/// Pop this scope's accumulator frame, returning how much of its own
/// inclusive duration was already attributed to directly-nested children.
///
/// If a parent frame remains on the stack, it is credited with this
/// scope's full `inclusive_duration_us` so that the parent can, in turn,
/// subtract that back out of its own exclusive time once it is dropped.
fn pop_scope_frame(inclusive_duration_us: u64) -> u64 {
    SCOPE_CHILDREN_US.with(|stack| {
        let mut stack = stack.borrow_mut();
        let children_us = stack.pop().unwrap_or(0);
        if let Some(parent) = stack.last_mut() {
            *parent += inclusive_duration_us;
        }
        children_us
    })
}

/// RAII scope guard for automatic profiling
pub struct ScopeGuard {
    name: String,
    category: String,
    start: Instant,
    /// Set by an owning [`MetricsScope`] to indicate this guard's `Drop`
    /// has already been fully handled (scope-stack pop *and* profiler
    /// event) by the `MetricsScope` that wraps it. `MetricsScope` embeds a
    /// plain `ScopeGuard` purely for its name/category/timing bookkeeping
    /// but must emit exactly one combined event carrying its
    /// operation_count/flops/bytes_transferred -- without this flag, the
    /// inner guard's own automatic `Drop` (which Rust always runs, right
    /// after `MetricsScope::drop`'s body, once a struct has a manual
    /// `Drop` impl) would record a second, metric-less duplicate event for
    /// the same logical scope and pop the nesting stack a second time.
    suppressed: bool,
}

impl ScopeGuard {
    /// Create a new scope guard with default category
    pub fn new(name: &str) -> Self {
        Self::with_category(name, "general")
    }

    /// Create a new scope guard with specified category
    pub fn with_category(name: &str, category: &str) -> Self {
        push_scope_frame();
        Self {
            name: name.to_string(),
            category: category.to_string(),
            start: Instant::now(),
            suppressed: false,
        }
    }

    /// Get the current duration of this scope
    pub fn elapsed(&self) -> std::time::Duration {
        self.start.elapsed()
    }

    /// Get the scope name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the scope category
    pub fn category(&self) -> &str {
        &self.category
    }
}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        // An enclosing `MetricsScope` already popped this scope's nesting
        // frame and recorded a combined profiler event; nothing left to do.
        if self.suppressed {
            return;
        }

        let duration = self.start.elapsed();
        let duration_us = duration.as_micros() as u64;
        let children_us = pop_scope_frame(duration_us);
        let thread_id = get_thread_id();

        // Capture stack trace (and this scope's real start time, relative
        // to the profiler's epoch) under one lock acquisition.
        let profiler_arc = global_profiler();
        let (stack_trace, stack_trace_overhead_ns, start_us) = {
            let profiler = profiler_arc.lock();
            let start_us = profiler
                .start_time
                .map(|profiler_start| {
                    self.start
                        .saturating_duration_since(profiler_start)
                        .as_micros() as u64
                })
                .unwrap_or(0);
            let (trace, overhead_ns) = if profiler.are_stack_traces_enabled() {
                if profiler.is_overhead_tracking_enabled() {
                    capture_stack_trace_with_overhead()
                } else {
                    (capture_stack_trace(), 0)
                }
            } else {
                (None, 0)
            };
            (trace, overhead_ns, start_us)
        };

        let event = ProfileEvent {
            name: self.name.clone(),
            category: self.category.clone(),
            start_us,
            duration_us,
            thread_id,
            operation_count: None,
            flops: None,
            bytes_transferred: None,
            stack_trace,
        };

        // Update overhead stats if tracking is enabled
        {
            let mut profiler = profiler_arc.lock();
            if profiler.is_overhead_tracking_enabled() && stack_trace_overhead_ns > 0 {
                profiler.overhead_stats.stack_trace_time_ns += stack_trace_overhead_ns;
                profiler.overhead_stats.stack_trace_count += 1;
                profiler.overhead_stats.total_overhead_ns += stack_trace_overhead_ns;
            }
            // add_event() credits this event with its full duration as
            // exclusive time by default; correct that for whatever was
            // already attributed to this scope's own direct children.
            profiler.add_event(event);
            profiler.exclusive_total_us = profiler.exclusive_total_us.saturating_sub(children_us);
        }
    }
}

/// Get the current thread ID as a number
fn get_thread_id() -> usize {
    let thread_id = std::thread::current().id();
    format!("{thread_id:?}")
        .chars()
        .filter(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse::<usize>()
        .unwrap_or(0)
}

/// Capture stack trace if enabled
fn capture_stack_trace() -> Option<String> {
    // Only capture if backtrace is available and not in release mode for performance
    #[cfg(debug_assertions)]
    {
        let bt = Backtrace::new();
        Some(format!("{:?}", bt))
    }
    #[cfg(not(debug_assertions))]
    None
}

/// Capture stack trace with overhead measurement
fn capture_stack_trace_with_overhead() -> (Option<String>, u64) {
    let start = Instant::now();
    let stack_trace = capture_stack_trace();
    let overhead_ns = start.elapsed().as_nanos() as u64;
    (stack_trace, overhead_ns)
}

/// Profile a function with automatic scope management
pub fn profile_function<F, R>(name: &str, func: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = ScopeGuard::new(name);
    func()
}

/// Profile a function with category and automatic scope management
pub fn profile_function_with_category<F, R>(name: &str, category: &str, func: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = ScopeGuard::with_category(name, category);
    func()
}

/// Enhanced scope guard with custom metrics
pub struct MetricsScope {
    guard: ScopeGuard,
    operation_count: Option<u64>,
    flops: Option<u64>,
    bytes_transferred: Option<u64>,
}

impl MetricsScope {
    /// Create a new metrics scope
    pub fn new(name: &str) -> Self {
        Self {
            guard: ScopeGuard::new(name),
            operation_count: None,
            flops: None,
            bytes_transferred: None,
        }
    }

    /// Create a new metrics scope with category
    pub fn with_category(name: &str, category: &str) -> Self {
        Self {
            guard: ScopeGuard::with_category(name, category),
            operation_count: None,
            flops: None,
            bytes_transferred: None,
        }
    }

    /// Set the number of operations performed
    pub fn set_operation_count(&mut self, count: u64) {
        self.operation_count = Some(count);
    }

    /// Set the number of floating point operations
    pub fn set_flops(&mut self, flops: u64) {
        self.flops = Some(flops);
    }

    /// Set the number of bytes transferred
    pub fn set_bytes_transferred(&mut self, bytes: u64) {
        self.bytes_transferred = Some(bytes);
    }

    /// Add to operation count
    pub fn add_operations(&mut self, count: u64) {
        self.operation_count = Some(self.operation_count.unwrap_or(0) + count);
    }

    /// Add to FLOPS count
    pub fn add_flops(&mut self, flops: u64) {
        self.flops = Some(self.flops.unwrap_or(0) + flops);
    }

    /// Add to bytes transferred
    pub fn add_bytes(&mut self, bytes: u64) {
        self.bytes_transferred = Some(self.bytes_transferred.unwrap_or(0) + bytes);
    }

    /// Get current metrics
    pub fn metrics(&self) -> (Option<u64>, Option<u64>, Option<u64>) {
        (self.operation_count, self.flops, self.bytes_transferred)
    }
}

impl Drop for MetricsScope {
    fn drop(&mut self) {
        let duration = self.guard.start.elapsed();
        let duration_us = duration.as_micros() as u64;
        let children_us = pop_scope_frame(duration_us);
        let thread_id = get_thread_id();

        // Capture stack trace (and this scope's real start time) under one
        // lock acquisition, mirroring `ScopeGuard::drop`.
        let profiler_arc = global_profiler();
        let (stack_trace, stack_trace_overhead_ns, start_us) = {
            let profiler = profiler_arc.lock();
            let start_us = profiler
                .start_time
                .map(|profiler_start| {
                    self.guard
                        .start
                        .saturating_duration_since(profiler_start)
                        .as_micros() as u64
                })
                .unwrap_or(0);
            let (trace, overhead_ns) = if profiler.are_stack_traces_enabled() {
                if profiler.is_overhead_tracking_enabled() {
                    capture_stack_trace_with_overhead()
                } else {
                    (capture_stack_trace(), 0)
                }
            } else {
                (None, 0)
            };
            (trace, overhead_ns, start_us)
        };

        let event = ProfileEvent {
            name: self.guard.name.clone(),
            category: self.guard.category.clone(),
            start_us,
            duration_us,
            thread_id,
            operation_count: self.operation_count,
            flops: self.flops,
            bytes_transferred: self.bytes_transferred,
            stack_trace,
        };

        // Update overhead stats and add event
        {
            let mut profiler = profiler_arc.lock();
            if profiler.is_overhead_tracking_enabled() && stack_trace_overhead_ns > 0 {
                profiler.overhead_stats.stack_trace_time_ns += stack_trace_overhead_ns;
                profiler.overhead_stats.stack_trace_count += 1;
                profiler.overhead_stats.total_overhead_ns += stack_trace_overhead_ns;
            }
            profiler.add_event(event);
            profiler.exclusive_total_us = profiler.exclusive_total_us.saturating_sub(children_us);
        }

        // This scope's nesting frame is already popped and its (combined,
        // metrics-carrying) event already recorded above; prevent the
        // inner `ScopeGuard`'s own `Drop` -- which Rust will run
        // automatically right after this function returns -- from doing
        // either a second time. See the `suppressed` field's doc comment.
        self.guard.suppressed = true;
    }
}

/// Macros for convenient scope profiling
#[macro_export]
macro_rules! profile_scope {
    ($name:expr) => {
        let _guard = $crate::core::scope::ScopeGuard::new($name);
    };
    ($name:expr, $category:expr) => {
        let _guard = $crate::core::scope::ScopeGuard::with_category($name, $category);
    };
}

#[macro_export]
macro_rules! profile_function {
    ($name:expr, $func:expr) => {
        $crate::core::scope::profile_function($name, $func)
    };
    ($name:expr, $category:expr, $func:expr) => {
        $crate::core::scope::profile_function_with_category($name, $category, $func)
    };
}

#[macro_export]
macro_rules! profile_metrics {
    ($name:expr) => {
        $crate::core::scope::MetricsScope::new($name)
    };
    ($name:expr, $category:expr) => {
        $crate::core::scope::MetricsScope::with_category($name, $category)
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::profiler::{
        clear_global_events, get_global_stats, start_profiling, stop_profiling,
    };
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_scope_guard_basic() {
        start_profiling();
        clear_global_events();

        {
            let _guard = ScopeGuard::new("test_scope");
            thread::sleep(Duration::from_millis(10));
        }

        let stats = get_global_stats().expect("get global stats should succeed");
        assert!(stats.event_count > 0); // Should have at least one event

        stop_profiling();
    }

    #[test]
    fn test_scope_guard_with_category() {
        start_profiling();
        clear_global_events();

        {
            let _guard = ScopeGuard::with_category("test_scope", "testing");
            thread::sleep(Duration::from_millis(5));
        }

        let stats = get_global_stats().expect("get global stats should succeed");
        assert!(stats.event_count > 0);

        stop_profiling();
    }

    #[test]
    fn test_profile_function() {
        start_profiling();
        clear_global_events();

        let result = profile_function("test_function", || {
            thread::sleep(Duration::from_millis(5));
            42
        });

        assert_eq!(result, 42);
        let stats = get_global_stats().expect("get global stats should succeed");
        assert!(stats.event_count > 0);

        stop_profiling();
    }

    #[test]
    fn test_metrics_scope() {
        start_profiling();
        clear_global_events();

        {
            let mut scope = MetricsScope::new("test_metrics");
            scope.set_operation_count(100);
            scope.set_flops(500);
            scope.set_bytes_transferred(1024);

            thread::sleep(Duration::from_millis(5));

            let (ops, flops, bytes) = scope.metrics();
            assert_eq!(ops, Some(100));
            assert_eq!(flops, Some(500));
            assert_eq!(bytes, Some(1024));
        }

        let stats = get_global_stats().expect("get global stats should succeed");
        // Exactly one event: MetricsScope must not also let its embedded
        // ScopeGuard record a second, metric-less duplicate when it is
        // auto-dropped right after MetricsScope::drop's body finishes.
        assert_eq!(stats.event_count, 1);

        stop_profiling();
    }

    #[test]
    fn test_metrics_scope_accumulation() {
        let mut scope = MetricsScope::new("test_accumulation");

        scope.add_operations(50);
        scope.add_operations(75);
        scope.add_flops(100);
        scope.add_flops(200);
        scope.add_bytes(512);
        scope.add_bytes(256);

        let (ops, flops, bytes) = scope.metrics();
        assert_eq!(ops, Some(125));
        assert_eq!(flops, Some(300));
        assert_eq!(bytes, Some(768));
    }

    #[test]
    fn test_profile_scope_macro() {
        start_profiling();
        clear_global_events();

        {
            profile_scope!("macro_test");
            thread::sleep(Duration::from_millis(5));
        }

        {
            profile_scope!("macro_test_with_category", "testing");
            thread::sleep(Duration::from_millis(5));
        }

        let stats = get_global_stats().expect("get global stats should succeed");
        assert!(stats.event_count >= 2); // Should have at least 2 events

        stop_profiling();
    }

    #[test]
    fn test_thread_id_extraction() {
        let id1 = get_thread_id();
        let id2 = thread::spawn(|| get_thread_id())
            .join()
            .expect("join should succeed");

        // Thread IDs should be different
        assert_ne!(id1, id2);
        // Both should be valid numbers
        assert!(id1 > 0);
        assert!(id2 > 0);
    }
}
