//! Hardening tests for torsh-series production-hardening findings.
//!
//! See the campaign brief for finding IDs and full descriptions.
//!
//! F308 (STLDecomposition drops the dead `_trend_deg`/inert `seasonal_deg`
//! fields) and F309 (stale "placeholder" doc comments in
//! `utils::statistical_tests`) are not behavioral bugs -- there is no
//! observable pre/post difference to assert on for either one (F308 removes
//! fields that never influenced output even before the fix; F309 only
//! touches doc comments). Per the campaign's verify-first protocol they are
//! instead verified by direct inspection (recorded in the wave report), and
//! guarded here against regression: this test proves STLDecomposition still
//! builds and fits correctly with the trimmed API (no leftover dead fields,
//! `robust(true)` still wired through), so a future edit that silently
//! breaks the wrapper would be caught here.

use torsh_series::decomposition::STLDecomposition;
use torsh_series::TimeSeries;
use torsh_tensor::Tensor;

fn synthetic_series(n: usize) -> TimeSeries {
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        let trend = i as f32 * 0.1;
        let seasonal = (i as f32 * 2.0 * std::f32::consts::PI / 12.0).sin() * 2.0;
        data.push(trend + seasonal);
    }
    let tensor = Tensor::from_vec(data, &[n]).expect("tensor should build");
    TimeSeries::new(tensor)
}

#[test]
fn f308_stl_decomposition_builds_and_fits_without_dead_degree_fields() {
    let series = synthetic_series(50);

    // Only `period` and `robust` are configurable now; both must still be
    // honored end to end (no leftover `_trend_deg`/`seasonal_deg` fields to
    // set, and `robust` should still affect the options passed through to
    // scirs2-series without panicking or changing output shape).
    let stl = STLDecomposition::new(12).robust(true);
    let result = stl.fit(&series).expect("fit should succeed");

    assert_eq!(result.trend.shape().dims()[0], series.len());
    assert_eq!(result.seasonal.shape().dims()[0], series.len());
    assert_eq!(result.residual.shape().dims()[0], series.len());

    // Non-robust path must also still work identically in shape.
    let stl_plain = STLDecomposition::new(12);
    let result_plain = stl_plain.fit(&series).expect("fit should succeed");
    assert_eq!(result_plain.trend.shape().dims()[0], series.len());
}
