//! Regression tests for the production-hardening pass on torsh-profiler's
//! core scope/event timing (F242, F243) and the "no fabricated
//! measurements" cleanup (F244).
//!
//! These exercise the crate's public API end to end (global profiler +
//! `ScopeGuard`/`MetricsScope` + `WorkloadCharacterizer`), complementing
//! the inline `#[cfg(test)]` unit tests in `core/profiler.rs`,
//! `core/scope.rs`, and `workload_characterization.rs`, which cover
//! internals not reachable from here.
//!
//! Each `#[test]` below starts/clears/stops the *global* profiler. This is
//! safe under `cargo nextest`, which runs every test function in its own
//! process (the same pattern the crate's own inline tests already rely
//! on) -- it would NOT be safe under plain `cargo test`, which shares one
//! process across all tests in a binary.

use std::thread;
use std::time::Duration;
use torsh_profiler::workload_characterization::WorkloadCharacterizer;
use torsh_profiler::{
    clear_global_events, get_global_stats, start_profiling, stop_profiling, MetricsScope,
    ProfileEvent, ScopeGuard,
};

/// F242: nested scopes must record real, distinct start times such that
/// the outer scope's interval fully contains the inner scope's interval.
/// Before the fix, `Profiler::add_event` overwrote every event's
/// `start_us` with the time `add_event` was *called* (i.e. the event's own
/// end), so outer.start_us == outer's end and inner.start_us == inner's
/// end -- destroying nesting order (the inner scope would appear to start
/// AFTER the outer scope had already ended).
#[test]
fn f242_nested_scopes_record_real_start_times() {
    start_profiling();
    clear_global_events();

    {
        let _outer = ScopeGuard::new("f242_outer");
        thread::sleep(Duration::from_millis(5));
        {
            let _inner = ScopeGuard::new("f242_inner");
            thread::sleep(Duration::from_millis(5));
        }
        thread::sleep(Duration::from_millis(5));
    }

    let profiler = torsh_profiler::global_profiler();
    let events = profiler.lock().events().to_vec();
    stop_profiling();

    let outer = events
        .iter()
        .find(|e| e.name == "f242_outer")
        .expect("outer event should be recorded");
    let inner = events
        .iter()
        .find(|e| e.name == "f242_inner")
        .expect("inner event should be recorded");

    let outer_end = outer.start_us + outer.duration_us;
    let inner_end = inner.start_us + inner.duration_us;

    // The historical bug made outer.start_us equal outer's END (the
    // largest timestamp of the two events), which would put outer AFTER
    // inner and fail this assertion.
    assert!(
        outer.start_us <= inner.start_us,
        "outer scope should start no later than the inner scope it contains: \
         outer.start_us={}, inner.start_us={}",
        outer.start_us,
        inner.start_us
    );
    assert!(
        inner_end <= outer_end,
        "inner scope should end no later than the outer scope containing it: \
         inner_end={}, outer_end={}",
        inner_end,
        outer_end
    );
    // A meaningful (non-degenerate) nesting relationship: the inner scope
    // must not appear to start at time 0 relative to the outer.
    assert!(
        inner.start_us > outer.start_us,
        "inner scope should start strictly after the outer scope began \
         (outer slept 5ms before entering inner): outer.start_us={}, inner.start_us={}",
        outer.start_us,
        inner.start_us
    );
}

/// F242: a plain, non-nested `MetricsScope` should also carry a start time
/// consistent with real elapsed wall-clock time -- not a start equal to
/// its own end.
#[test]
fn f242_metrics_scope_records_real_start_time() {
    start_profiling();
    clear_global_events();

    {
        let mut scope = MetricsScope::new("f242_metrics");
        scope.set_flops(1234);
        thread::sleep(Duration::from_millis(5));
    }

    let profiler = torsh_profiler::global_profiler();
    let events = profiler.lock().events().to_vec();
    stop_profiling();

    let event = events
        .iter()
        .find(|e| e.name == "f242_metrics")
        .expect("event should be recorded");

    // duration_us must be a real, positive measurement of the ~5ms sleep.
    assert!(
        event.duration_us >= 1000,
        "duration_us should reflect the real ~5ms sleep, got {}",
        event.duration_us
    );
    // start_us + duration_us must be internally consistent with a single
    // real profiler run (both bounded by a sane upper limit -- a few
    // seconds -- rather than start_us wrapping around to equal the end).
    assert!(
        event.start_us < 5_000_000,
        "start_us should be a small, real offset from profiler start, got {}",
        event.start_us
    );
}

/// F243: `get_stats`/`get_global_stats` must expose distinct, correctly
/// computed inclusive vs. exclusive totals -- not the historical 5-tuple
/// whose 2nd and 3rd elements were the exact same "total_duration" value
/// duplicated (so any 3rd-element reader silently got a meaningless copy).
#[test]
fn f243_inclusive_and_exclusive_totals_differ_for_nested_scopes() {
    start_profiling();
    clear_global_events();

    {
        let _outer = ScopeGuard::new("f243_outer");
        thread::sleep(Duration::from_millis(10));
        {
            let _inner = ScopeGuard::new("f243_inner");
            thread::sleep(Duration::from_millis(10));
        }
    }

    let stats = get_global_stats().expect("get_global_stats should succeed");
    stop_profiling();

    assert_eq!(stats.event_count, 2);
    // Inclusive time double-counts the inner scope (once as its own event,
    // once again inside the outer scope's full duration), so it exceeds
    // the exclusive total once real nesting is involved.
    assert!(
        stats.inclusive_total_us > stats.exclusive_total_us,
        "inclusive_total_us ({}) should exceed exclusive_total_us ({}) once scopes nest",
        stats.inclusive_total_us,
        stats.exclusive_total_us
    );
    // The old bug returned the very same "total_duration" value as both
    // the 2nd and 3rd tuple elements; here that would mean
    // inclusive_total_us == exclusive_total_us, which we just proved false.
    assert_ne!(stats.inclusive_total_us, stats.exclusive_total_us);
}

/// F243: for a single, non-nested scope, exclusive time equals inclusive
/// time exactly (nothing to subtract), proving the fields are computed
/// independently rather than one being an arbitrary copy of the other.
#[test]
fn f243_exclusive_equals_inclusive_for_flat_scope() {
    start_profiling();
    clear_global_events();

    {
        let _solo = ScopeGuard::new("f243_solo");
        thread::sleep(Duration::from_millis(5));
    }

    let stats = get_global_stats().expect("get_global_stats should succeed");
    stop_profiling();

    assert_eq!(stats.event_count, 1);
    assert_eq!(stats.inclusive_total_us, stats.exclusive_total_us);
    assert!(stats.inclusive_total_us >= 1000);
}

/// F244: `WorkloadCharacterizer::add_samples_from_events` must not invent
/// CPU utilization / cache miss rate / I/O rate / thread count / energy
/// readings for metrics that `ProfileEvent` never carried in the first
/// place. Before the fix this always produced a plausible-looking
/// `cpu_utilization: 0.7` etc. regardless of what actually happened.
#[test]
fn f244_workload_samples_from_events_do_not_fabricate_unmeasured_fields() {
    let mut characterizer = WorkloadCharacterizer::new();
    let events = vec![ProfileEvent {
        name: "op".to_string(),
        category: "test".to_string(),
        start_us: 0,
        duration_us: 1000,
        thread_id: 1,
        operation_count: None,
        flops: Some(42),
        bytes_transferred: Some(4096),
        stack_trace: None,
    }];

    characterizer
        .add_samples_from_events(&events)
        .expect("add_samples_from_events should succeed");

    // We can only observe the samples indirectly via analyze(), which
    // requires reaching min_sample_count; instead, re-add enough copies to
    // pass that bar and confirm the derived resource pattern honestly
    // reports "unmeasured" (None) rather than a plausible-looking bogus
    // average.
    for _ in 0..200 {
        characterizer
            .add_samples_from_events(&events)
            .expect("add_samples_from_events should succeed");
    }

    let analysis = characterizer
        .analyze()
        .expect("analyze should succeed with enough samples");

    // cpu_utilization/cache_miss_rate/io_ops_per_sec were never present on
    // ProfileEvent, so every derived aggregate must honestly report "not
    // measured" rather than a fabricated value like the historical 0.7.
    assert_eq!(
        analysis.resource_patterns.avg_cpu_utilization, None,
        "CPU utilization was never measured from ProfileEvent data and must not be fabricated"
    );
    assert_eq!(
        analysis.resource_patterns.cache_efficiency_score, None,
        "cache efficiency was never measured from ProfileEvent data and must not be fabricated"
    );
    assert_eq!(
        analysis.resource_patterns.io_throughput_mbps, None,
        "I/O throughput was never measured from ProfileEvent data and must not be fabricated"
    );
    assert_eq!(analysis.resource_patterns.memory_locality_score, None);
    assert_eq!(
        analysis.compute_characteristics.vectorization_efficiency,
        None
    );
    assert_eq!(analysis.compute_characteristics.dominant_operations, None);
}
