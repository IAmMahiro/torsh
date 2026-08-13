//! Core profiler implementation
//!
//! This module provides the main Profiler struct and associated functionality
//! extracted from the massive lib.rs file to improve maintainability.

use crate::{OverheadStats, ProfileEvent, TorshResult};
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Instant;
use torsh_core::TorshError;

/// Main profiler implementation that manages events and statistics
#[derive(Debug, Clone)]
pub struct Profiler {
    /// Collected profiling events
    pub events: Vec<ProfileEvent>,
    /// Whether profiling is currently active
    pub enabled: bool,
    /// Whether stack traces are enabled
    pub stack_traces_enabled: bool,
    /// Whether overhead tracking is enabled
    pub overhead_tracking_enabled: bool,
    /// Overhead statistics
    pub overhead_stats: OverheadStats,
    /// Profiler start time
    pub start_time: Option<Instant>,
    /// Running total of every event's *exclusive* (self) time, in
    /// microseconds.
    ///
    /// `add_event` credits a newly-added event with its full `duration_us`
    /// as a default (correct when the event has no known children).
    /// `ScopeGuard`/`MetricsScope` (see `core::scope`) then subtract back
    /// out any directly-nested child scopes' durations once those children
    /// have already been counted, via a thread-local stack that mirrors
    /// real RAII nesting. This avoids double-counting time spent in nested
    /// scopes, unlike [`Profiler::total_duration_us`], which sums every
    /// event's inclusive duration and so can exceed 100% of wall time once
    /// scopes nest.
    pub exclusive_total_us: u64,
}

impl Profiler {
    /// Create a new profiler instance
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            enabled: false,
            stack_traces_enabled: false,
            overhead_tracking_enabled: false,
            overhead_stats: OverheadStats::default(),
            start_time: None,
            exclusive_total_us: 0,
        }
    }

    /// Start profiling
    pub fn start(&mut self) {
        self.enabled = true;
        self.start_time = Some(Instant::now());
    }

    /// Stop profiling
    pub fn stop(&mut self) {
        self.enabled = false;
    }

    /// Clear all collected events
    pub fn clear(&mut self) {
        self.events.clear();
        self.overhead_stats = OverheadStats::default();
        self.exclusive_total_us = 0;
    }

    /// Add a profiling event.
    ///
    /// `event.start_us` is trusted as supplied by the caller: it must
    /// already be a real clock reading relative to [`Profiler::start_time`]
    /// (that's what `ScopeGuard`/`MetricsScope` and [`add_global_event`] do).
    /// This method used to unconditionally overwrite `start_us` with
    /// "now" (`profiler_start.elapsed()`), which silently replaced every
    /// event's true start time with its own *end* time -- every event's
    /// recorded start equaled its own end, destroying nesting order in
    /// exports like the Chrome trace.
    pub fn add_event(&mut self, event: ProfileEvent) {
        if !self.enabled {
            return;
        }

        let start_overhead = if self.overhead_tracking_enabled {
            Some(Instant::now())
        } else {
            None
        };

        let duration_us = event.duration_us;
        self.events.push(event);

        // Every event starts out credited with its full duration as
        // "exclusive" time; nested scopes correct this for their ancestors
        // once dropped (see core::scope::pop_scope_frame).
        self.exclusive_total_us += duration_us;

        // Track overhead if enabled
        if let Some(start) = start_overhead {
            let overhead_ns = start.elapsed().as_nanos() as u64;
            self.overhead_stats.add_event_time_ns += overhead_ns;
            self.overhead_stats.add_event_count += 1;
            self.overhead_stats.total_overhead_ns += overhead_ns;
        }
    }

    /// Get current statistics as a named, self-documenting struct.
    ///
    /// Previously this returned an unlabeled `(u64, u64, u64, f64, f64)`
    /// tuple whose second and third elements were both the same
    /// "inclusive total" value (a bug: the third position was evidently
    /// meant to carry a different aggregate). See [`ProfilerEventStats`].
    pub fn get_stats(&self) -> ProfilerEventStats {
        let event_count = self.events.len() as u64;
        let inclusive_total_us: u64 = self.events.iter().map(|e| e.duration_us).sum();
        let avg_duration_us = if event_count > 0 {
            inclusive_total_us as f64 / event_count as f64
        } else {
            0.0
        };

        let min_duration_us = self.events.iter().map(|e| e.duration_us).min().unwrap_or(0) as f64;
        let max_duration_us = self.events.iter().map(|e| e.duration_us).max().unwrap_or(0) as f64;

        ProfilerEventStats {
            event_count,
            inclusive_total_us,
            exclusive_total_us: self.exclusive_total_us,
            wall_clock_us: self.elapsed_us(),
            avg_duration_us,
            min_duration_us,
            max_duration_us,
        }
    }

    /// Microseconds elapsed since [`Profiler::start`] was called, or `0` if
    /// the profiler has never been started.
    ///
    /// Unlike [`Profiler::total_duration_us`] (a sum of recorded event
    /// durations that can exceed wall-clock time once scopes nest), this is
    /// a direct `Instant::elapsed()` reading against the profiler's own
    /// epoch -- a genuine wall-clock counterpart.
    pub fn elapsed_us(&self) -> u64 {
        self.start_time
            .map(|t| t.elapsed().as_micros() as u64)
            .unwrap_or(0)
    }

    /// Check if profiler is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enable or disable stack traces
    pub fn set_stack_traces_enabled(&mut self, enabled: bool) {
        self.stack_traces_enabled = enabled;
    }

    /// Check if stack traces are enabled
    pub fn are_stack_traces_enabled(&self) -> bool {
        self.stack_traces_enabled
    }

    /// Enable or disable overhead tracking
    pub fn set_overhead_tracking_enabled(&mut self, enabled: bool) {
        self.overhead_tracking_enabled = enabled;
    }

    /// Check if overhead tracking is enabled
    pub fn is_overhead_tracking_enabled(&self) -> bool {
        self.overhead_tracking_enabled
    }

    /// Get overhead statistics
    pub fn get_overhead_stats(&self) -> &OverheadStats {
        &self.overhead_stats
    }

    /// Reset overhead statistics
    pub fn reset_overhead_stats(&mut self) {
        self.overhead_stats = OverheadStats::default();
    }

    /// Get reference to events
    pub fn events(&self) -> &[ProfileEvent] {
        &self.events
    }

    /// Get number of events
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Get total *inclusive* profiling duration: the sum of every recorded
    /// event's own `duration_us`.
    ///
    /// A scope and every scope nested inside it are each counted in full,
    /// so this figure can exceed wall-clock time once scopes nest (e.g. a
    /// 100us outer scope containing an 80us inner scope reports 180us
    /// here). See [`Profiler::exclusive_total_us`] / [`Profiler::get_stats`]
    /// for the non-double-counting alternative, and [`Profiler::elapsed_us`]
    /// for a true wall-clock reading.
    pub fn total_duration_us(&self) -> u64 {
        self.events.iter().map(|e| e.duration_us).sum()
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregated statistics for a [`Profiler`]'s recorded events.
///
/// Replaces a previous unlabeled `(u64, u64, u64, f64, f64)` tuple whose
/// third element duplicated the second exactly. This struct instead makes
/// each aggregate's meaning explicit, and separates *inclusive* time (which
/// double-counts nested scopes) from *exclusive* time (which does not).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProfilerEventStats {
    /// Number of recorded events.
    pub event_count: u64,
    /// Sum of every event's own `duration_us` ("inclusive" time). A scope
    /// and all of its nested children are each counted in full, so this
    /// can exceed wall-clock time once scopes nest.
    pub inclusive_total_us: u64,
    /// Sum of each event's *exclusive* (self) time: its own duration minus
    /// time attributed to directly-nested child scopes. Never double-counts
    /// nested work.
    pub exclusive_total_us: u64,
    /// Microseconds elapsed since the profiler was started
    /// (`Profiler::start`), independent of how many events were recorded.
    pub wall_clock_us: u64,
    /// Mean of `duration_us` across all events.
    pub avg_duration_us: f64,
    /// Minimum recorded event duration, in microseconds.
    pub min_duration_us: f64,
    /// Maximum recorded event duration, in microseconds.
    pub max_duration_us: f64,
}

/// Global profiler instance
static GLOBAL_PROFILER: once_cell::sync::Lazy<Arc<Mutex<Profiler>>> =
    once_cell::sync::Lazy::new(|| Arc::new(Mutex::new(Profiler::new())));

/// Get reference to global profiler
pub fn global_profiler() -> Arc<Mutex<Profiler>> {
    GLOBAL_PROFILER.clone()
}

/// Start global profiling
pub fn start_profiling() {
    global_profiler().lock().start();
}

/// Stop global profiling
pub fn stop_profiling() {
    global_profiler().lock().stop();
}

/// Clear global profiler events
pub fn clear_global_events() {
    global_profiler().lock().clear();
}

/// Add event to global profiler.
///
/// This function only receives a pre-measured `duration_us`, not a real
/// start `Instant` (unlike `ScopeGuard`/`MetricsScope`, which capture one).
/// `start_us` is therefore approximated as "now minus duration", i.e. it
/// assumes this function is called immediately after the measured
/// operation completes -- a real derivation from the profiler's own clock,
/// not a fabricated value. Prefer `ScopeGuard`/`MetricsScope`/
/// `profile_scope!` when a precise start time matters.
pub fn add_global_event(name: &str, category: &str, duration_us: u64, thread_id: usize) {
    let profiler_arc = global_profiler();
    let start_us = profiler_arc.lock().elapsed_us().saturating_sub(duration_us);

    let event = ProfileEvent {
        name: name.to_string(),
        category: category.to_string(),
        start_us,
        duration_us,
        thread_id,
        operation_count: None,
        flops: None,
        bytes_transferred: None,
        stack_trace: None,
    };

    profiler_arc.lock().add_event(event);
}

/// Get global profiler statistics
pub fn get_global_stats() -> TorshResult<ProfilerEventStats> {
    Ok(global_profiler().lock().get_stats())
}

/// Set global stack traces enabled
pub fn set_global_stack_traces_enabled(enabled: bool) {
    global_profiler().lock().set_stack_traces_enabled(enabled);
}

/// Check if global stack traces are enabled
pub fn are_global_stack_traces_enabled() -> bool {
    global_profiler().lock().are_stack_traces_enabled()
}

/// Set global overhead tracking enabled
pub fn set_global_overhead_tracking_enabled(enabled: bool) {
    global_profiler()
        .lock()
        .set_overhead_tracking_enabled(enabled);
}

/// Check if global overhead tracking is enabled
pub fn is_global_overhead_tracking_enabled() -> bool {
    global_profiler().lock().is_overhead_tracking_enabled()
}

/// Get global overhead statistics
pub fn get_global_overhead_stats() -> OverheadStats {
    global_profiler().lock().get_overhead_stats().clone()
}

/// Reset global overhead statistics
pub fn reset_global_overhead_stats() {
    global_profiler().lock().reset_overhead_stats();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_profiler_lifecycle() {
        let mut profiler = Profiler::new();
        assert!(!profiler.is_enabled());

        profiler.start();
        assert!(profiler.is_enabled());

        profiler.stop();
        assert!(!profiler.is_enabled());
    }

    #[test]
    fn test_event_collection() {
        let mut profiler = Profiler::new();
        profiler.start();

        let event = ProfileEvent {
            name: "test_event".to_string(),
            category: "test".to_string(),
            start_us: 0,
            duration_us: 1000,
            thread_id: 1,
            operation_count: None,
            flops: None,
            bytes_transferred: None,
            stack_trace: None,
        };

        profiler.add_event(event);
        assert_eq!(profiler.event_count(), 1);
        assert_eq!(profiler.total_duration_us(), 1000);
    }

    #[test]
    #[ignore = "Flaky test - passes individually but may fail in full suite"]
    fn test_overhead_tracking() {
        let mut profiler = Profiler::new();
        profiler.set_overhead_tracking_enabled(true);
        profiler.start();

        let event = ProfileEvent {
            name: "test_overhead".to_string(),
            category: "test".to_string(),
            start_us: 0,
            duration_us: 500,
            thread_id: 1,
            operation_count: None,
            flops: None,
            bytes_transferred: None,
            stack_trace: None,
        };

        profiler.add_event(event);

        let stats = profiler.get_overhead_stats();
        assert_eq!(stats.add_event_count, 1);
        assert!(stats.add_event_time_ns > 0);
        assert!(stats.total_overhead_ns > 0);
    }

    #[test]
    fn test_global_profiler() {
        start_profiling();
        add_global_event("global_test", "test", 2000, 1);

        let stats = get_global_stats().expect("get global stats should succeed");
        assert!(stats.event_count > 0);
        assert!(stats.inclusive_total_us > 0);

        clear_global_events();
        stop_profiling();
    }

    #[test]
    fn test_stack_trace_settings() {
        let mut profiler = Profiler::new();
        assert!(!profiler.are_stack_traces_enabled());

        profiler.set_stack_traces_enabled(true);
        assert!(profiler.are_stack_traces_enabled());

        profiler.set_stack_traces_enabled(false);
        assert!(!profiler.are_stack_traces_enabled());
    }

    #[test]
    fn test_profiler_statistics() {
        let mut profiler = Profiler::new();
        profiler.start();

        // Add multiple events with different durations
        for i in 1..=5 {
            let event = ProfileEvent {
                name: format!("event_{}", i),
                category: "test".to_string(),
                start_us: 0,
                duration_us: i * 100,
                thread_id: 1,
                operation_count: None,
                flops: None,
                bytes_transferred: None,
                stack_trace: None,
            };
            profiler.add_event(event);
        }

        let stats = profiler.get_stats();
        assert_eq!(stats.event_count, 5);
        assert_eq!(stats.inclusive_total_us, 1500); // 100 + 200 + 300 + 400 + 500
        assert_eq!(stats.min_duration_us, 100.0);
        assert_eq!(stats.max_duration_us, 500.0);
        assert_eq!(stats.avg_duration_us, 300.0); // 1500 / 5

        // These 5 events were added directly (not via ScopeGuard/MetricsScope),
        // so none of them are known to be nested inside one another --
        // exclusive_total_us falls back to crediting each with its full
        // duration, same as inclusive_total_us.
        assert_eq!(stats.exclusive_total_us, 1500);
    }

    #[test]
    fn test_get_stats_struct_fields_are_independent() {
        // Regression test for the historical bug where `get_stats` returned
        // a 5-tuple whose 2nd and 3rd elements were the exact same value
        // (`total_duration` duplicated instead of a distinct aggregate).
        let mut profiler = Profiler::new();
        profiler.start();
        profiler.add_event(ProfileEvent {
            name: "a".to_string(),
            category: "test".to_string(),
            start_us: 0,
            duration_us: 42,
            thread_id: 1,
            operation_count: None,
            flops: None,
            bytes_transferred: None,
            stack_trace: None,
        });

        let before_us = profiler.elapsed_us();
        let stats = profiler.get_stats();
        let after_us = profiler.elapsed_us();

        // inclusive_total_us and exclusive_total_us are separate fields
        // that happen to be equal here (single flat event), but
        // wall_clock_us is sourced independently: a live `Instant::elapsed`
        // reading against the profiler's start time, not summed from event
        // durations at all. Bracketing it between two direct calls to
        // `elapsed_us()` proves it is a real, monotonically-increasing
        // clock reading rather than a value copied from the other fields.
        assert_eq!(stats.inclusive_total_us, 42);
        assert_eq!(stats.exclusive_total_us, 42);
        assert!(
            stats.wall_clock_us >= before_us && stats.wall_clock_us <= after_us,
            "wall_clock_us ({}) should fall between two elapsed_us() readings ({}, {})",
            stats.wall_clock_us,
            before_us,
            after_us
        );
    }
}
