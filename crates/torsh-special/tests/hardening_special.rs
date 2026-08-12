//! Regression tests for the production-hardening campaign (torsh-special).
//!
//! Findings covered: F257 (a second, defective copy of the error functions in
//! `error_functions`) and F258 (`erfcx` overflowing and the Fresnel wrappers
//! swallowing errors).

use torsh_core::DeviceType;
use torsh_special::TorshResult;
use torsh_tensor::Tensor;

fn tensor(values: &[f32]) -> TorshResult<Tensor<f32>> {
    Tensor::from_data(values.to_vec(), vec![values.len()], DeviceType::Cpu)
}

// ---------------------------------------------------------------------------
// F257: error_functions must not shadow the accurate path
// ---------------------------------------------------------------------------

#[test]
fn f257_fresnel_small_argument_series_is_correct() -> TorshResult<()> {
    // S(0.5) = 0.0647, C(0.5) = 0.4923 (Abramowitz & Stegun 7.3).
    let x = tensor(&[0.5])?;
    let s = torsh_special::error_functions::fresnel_s(&x)?.data()?;
    let c = torsh_special::error_functions::fresnel_c(&x)?.data()?;
    assert!(
        (s[0] - 0.064732).abs() < 1e-4,
        "S(0.5) = {} but should be 0.064732",
        s[0]
    );
    assert!(
        (c[0] - 0.492344).abs() < 1e-4,
        "C(0.5) = {} but should be 0.492344",
        c[0]
    );
    Ok(())
}

#[test]
fn f257_fresnel_is_odd() -> TorshResult<()> {
    let x = tensor(&[-0.5, 0.5, -1.5, 1.5])?;
    let s = torsh_special::error_functions::fresnel_s(&x)?.data()?;
    let c = torsh_special::error_functions::fresnel_c(&x)?.data()?;
    assert!(
        (s[0] + s[1]).abs() < 1e-5,
        "S must be odd: {} {}",
        s[0],
        s[1]
    );
    assert!(
        (c[0] + c[1]).abs() < 1e-5,
        "C must be odd: {} {}",
        c[0],
        c[1]
    );
    assert!((s[2] + s[3]).abs() < 1e-5);
    assert!((c[2] + c[3]).abs() < 1e-5);
    // The sign must follow the argument, not be flipped twice.
    assert!(c[1] > 0.0 && c[0] < 0.0, "C(0.5) must be positive");
    Ok(())
}

#[test]
fn f257_erfcx_matches_reference_values() -> TorshResult<()> {
    // erfcx(1) = 0.4275836, erfcx(2) = 0.2553956, erfcx(0) = 1.
    let x = tensor(&[0.0, 1.0, 2.0])?;
    let values = torsh_special::error_functions::erfcx(&x)?.data()?;
    assert!((values[0] - 1.0).abs() < 1e-5);
    assert!(
        (values[1] - 0.4275836).abs() < 1e-4,
        "erfcx(1) = {} but should be 0.4275836",
        values[1]
    );
    assert!(
        (values[2] - 0.2553956).abs() < 1e-4,
        "erfcx(2) = {} but should be 0.2553956",
        values[2]
    );
    Ok(())
}

#[test]
fn f257_erfc_keeps_precision_in_the_tail() -> TorshResult<()> {
    // erfc(4) = 1.5417e-8: subtracting erf from one loses every digit.
    let x = tensor(&[4.0])?;
    let value = torsh_special::error_functions::erfc(&x)?.data()?[0];
    assert!(
        (value - 1.5417e-8).abs() < 1e-10,
        "erfc(4) = {value} but should be 1.5417e-8"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// F258: exported erfcx must not overflow; Fresnel must propagate errors
// ---------------------------------------------------------------------------

#[test]
fn f258_erfcx_does_not_overflow_for_large_arguments() -> TorshResult<()> {
    // erfcx(27) = 0.020887..., while exp(27^2) alone overflows f64.
    let x = tensor(&[27.0, 100.0, 1000.0])?;
    let values = torsh_special::erfcx(&x)?.data()?;
    assert!(
        values[0].is_finite() && (values[0] - 0.0208869).abs() < 1e-5,
        "erfcx(27) = {} but should be 0.0208869",
        values[0]
    );
    assert!(
        values[1].is_finite() && (values[1] - 0.00564161).abs() < 1e-6,
        "erfcx(100) = {} but should be 0.00564161",
        values[1]
    );
    assert!(
        values[2].is_finite() && values[2] > 0.0,
        "erfcx(1000) = {} must stay finite and positive",
        values[2]
    );
    Ok(())
}

#[test]
fn f258_erfcx_negative_branch() -> TorshResult<()> {
    // erfcx(-1) = 2*exp(1) - erfcx(1) = 5.0089...
    let x = tensor(&[-1.0])?;
    let value = torsh_special::erfcx(&x)?.data()?[0];
    assert!(
        (value - 5.0089).abs() < 1e-3,
        "erfcx(-1) = {value} but should be 5.0089"
    );
    Ok(())
}

#[test]
fn f258_fresnel_values_are_accurate() -> TorshResult<()> {
    // S(1) = 0.4382591, C(1) = 0.7798934.
    let x = tensor(&[1.0])?;
    let (s, c) = torsh_special::fresnel(&x)?;
    assert!((s.data()?[0] - 0.4382591).abs() < 1e-5);
    assert!((c.data()?[0] - 0.7798934).abs() < 1e-5);

    let s_only = torsh_special::fresnel_s(&x)?.data()?[0];
    let c_only = torsh_special::fresnel_c(&x)?.data()?[0];
    assert!((s_only - 0.4382591).abs() < 1e-5);
    assert!((c_only - 0.7798934).abs() < 1e-5);
    Ok(())
}

#[test]
fn f258_fresnel_propagates_errors_instead_of_returning_zero() -> TorshResult<()> {
    // scirs2-special rejects a NaN argument with a domain error. The old
    // wrappers turned that into (0.0, 0.0) — a perfectly plausible pair of
    // values (the exact answer at x = 0) and therefore indistinguishable from
    // success. All three entry points must surface the error.
    let x = tensor(&[f32::NAN])?;
    assert!(
        torsh_special::fresnel(&x).is_err(),
        "fresnel() swallowed a domain error"
    );
    assert!(
        torsh_special::fresnel_s(&x).is_err(),
        "fresnel_s() swallowed a domain error"
    );
    assert!(
        torsh_special::fresnel_c(&x).is_err(),
        "fresnel_c() swallowed a domain error"
    );
    Ok(())
}

#[test]
fn f257_error_functions_agree_with_the_scirs2_path() -> TorshResult<()> {
    let x = tensor(&[-2.0, -0.5, 0.0, 0.5, 2.0])?;

    let a = torsh_special::error_functions::erf(&x)?.data()?;
    let b = torsh_special::erf(&x)?.data()?;
    for (lhs, rhs) in a.iter().zip(b.iter()) {
        assert!((lhs - rhs).abs() < 1e-6, "erf mismatch: {lhs} vs {rhs}");
    }

    let a = torsh_special::error_functions::erfinv(&tensor(&[-0.5, 0.0, 0.5])?)?.data()?;
    let b = torsh_special::erfinv(&tensor(&[-0.5, 0.0, 0.5])?)?.data()?;
    for (lhs, rhs) in a.iter().zip(b.iter()) {
        assert!((lhs - rhs).abs() < 1e-6, "erfinv mismatch: {lhs} vs {rhs}");
    }
    Ok(())
}
