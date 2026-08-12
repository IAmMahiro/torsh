//! Production-hardening regression tests for `torsh-signal`.
//!
//! Every test in this file pins down behaviour that was previously fabricated
//! (all-zero outputs, invented filter coefficients, NaN windows, ...). The test
//! names carry the campaign finding IDs so a regression is easy to trace back.

use torsh_signal::prelude::*;
use torsh_tensor::Tensor;

const FS: f32 = 8000.0;

/// Build a unit-amplitude sine wave of `len` samples at `freq` Hz.
fn sine(freq: f32, len: usize) -> Tensor<f32> {
    let data: Vec<f32> = (0..len)
        .map(|n| (2.0 * std::f32::consts::PI * freq * n as f32 / FS).sin())
        .collect();
    Tensor::from_vec(data, &[len]).expect("tensor creation should succeed")
}

/// Measure the steady-state magnitude response of an IIR filter at `freq`
/// by filtering a long sine wave and comparing RMS values on the tail.
fn measure_gain(filter: &mut DigitalFilter, freq: f32) -> f32 {
    let len = 8192usize;
    let input = sine(freq, len);
    let output = filter.filter(&input).expect("filtering should succeed");
    let x = input.to_vec().expect("to_vec should succeed");
    let y = output.to_vec().expect("to_vec should succeed");
    let tail = len / 2;
    let rms = |v: &[f32]| -> f32 {
        (v[tail..].iter().map(|a| a * a).sum::<f32>() / (v.len() - tail) as f32).sqrt()
    };
    rms(&y) / rms(&x)
}

// ---------------------------------------------------------------------------
// F049: Butterworth designers used invented coefficients.
// ---------------------------------------------------------------------------

#[test]
fn f049_butterworth_lowpass_is_minus_3db_at_cutoff() {
    let designer = IIRFilterDesigner::new(FS);
    let mut filter = designer
        .butterworth(4, &[1000.0], FilterType::Lowpass)
        .expect("butterworth lowpass design should succeed");

    let g_pass = measure_gain(&mut filter, 200.0);
    let g_cut = measure_gain(&mut filter, 1000.0);
    let g_stop = measure_gain(&mut filter, 3000.0);

    assert!(
        (g_pass - 1.0).abs() < 0.02,
        "passband gain should be ~1, got {g_pass}"
    );
    assert!(
        (g_cut - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.03,
        "gain at cutoff should be -3 dB (0.7071), got {g_cut}"
    );
    assert!(g_stop < 0.05, "stopband gain should be small, got {g_stop}");
}

#[test]
fn f049_butterworth_highpass_is_minus_3db_at_cutoff() {
    let designer = IIRFilterDesigner::new(FS);
    let mut filter = designer
        .butterworth(4, &[1000.0], FilterType::Highpass)
        .expect("butterworth highpass design should succeed");

    let g_stop = measure_gain(&mut filter, 200.0);
    let g_cut = measure_gain(&mut filter, 1000.0);
    let g_pass = measure_gain(&mut filter, 3000.0);

    assert!(g_stop < 0.05, "stopband gain should be small, got {g_stop}");
    assert!(
        (g_cut - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.03,
        "gain at cutoff should be -3 dB, got {g_cut}"
    );
    assert!(
        (g_pass - 1.0).abs() < 0.03,
        "passband gain should be ~1, got {g_pass}"
    );
}

#[test]
fn f049_butterworth_bandpass_and_bandstop_track_their_bands() {
    let designer = IIRFilterDesigner::new(FS);

    let mut bp = designer
        .butterworth(4, &[800.0, 1600.0], FilterType::Bandpass)
        .expect("bandpass design should succeed");
    let centre = (800.0f32 * 1600.0).sqrt();
    let g_centre = measure_gain(&mut bp, centre);
    let g_low = measure_gain(&mut bp, 200.0);
    let g_high = measure_gain(&mut bp, 3500.0);
    assert!(
        (g_centre - 1.0).abs() < 0.05,
        "bandpass centre gain should be ~1, got {g_centre}"
    );
    assert!(g_low < 0.1, "bandpass DC-side gain too high: {g_low}");
    assert!(g_high < 0.1, "bandpass HF-side gain too high: {g_high}");

    let mut bs = designer
        .butterworth(4, &[800.0, 1600.0], FilterType::Bandstop)
        .expect("bandstop design should succeed");
    let g_notch = measure_gain(&mut bs, centre);
    let g_pass_low = measure_gain(&mut bs, 100.0);
    let g_pass_high = measure_gain(&mut bs, 3800.0);
    assert!(g_notch < 0.1, "bandstop notch gain too high: {g_notch}");
    assert!(
        (g_pass_low - 1.0).abs() < 0.05,
        "bandstop low passband gain should be ~1, got {g_pass_low}"
    );
    assert!(
        (g_pass_high - 1.0).abs() < 0.05,
        "bandstop high passband gain should be ~1, got {g_pass_high}"
    );
}

// ---------------------------------------------------------------------------
// F048: Chebyshev / Bessel / elliptic designers returned all-zero numerators.
// ---------------------------------------------------------------------------

#[test]
fn f048_chebyshev1_respects_ripple_and_cutoff() {
    let designer = IIRFilterDesigner::new(FS);
    let mut filter = designer
        .chebyshev1(4, 1.0, &[1000.0], FilterType::Lowpass)
        .expect("chebyshev1 design should succeed");

    // Chebyshev-I: |H| = -rp dB exactly at the passband edge.
    let expected_edge = 10f32.powf(-1.0 / 20.0);
    let g_edge = measure_gain(&mut filter, 1000.0);
    let g_pass = measure_gain(&mut filter, 300.0);
    let g_stop = measure_gain(&mut filter, 3000.0);

    assert!(
        (g_edge - expected_edge).abs() < 0.03,
        "gain at passband edge should be {expected_edge}, got {g_edge}"
    );
    assert!(
        g_pass >= expected_edge - 0.03 && g_pass <= 1.02,
        "passband gain should stay inside the ripple band, got {g_pass}"
    );
    assert!(g_stop < 0.05, "stopband gain should be small, got {g_stop}");
}

#[test]
fn f048_chebyshev2_reaches_requested_stopband_attenuation() {
    let designer = IIRFilterDesigner::new(FS);
    let mut filter = designer
        .chebyshev2(4, 40.0, &[1500.0], FilterType::Lowpass)
        .expect("chebyshev2 design should succeed");

    let g_pass = measure_gain(&mut filter, 100.0);
    let g_stop = measure_gain(&mut filter, 1500.0);

    assert!(
        (g_pass - 1.0).abs() < 0.05,
        "passband gain should be ~1, got {g_pass}"
    );
    assert!(
        g_stop < 10f32.powf(-40.0 / 20.0) * 1.15,
        "gain at the stopband edge should be <= -40 dB, got {g_stop}"
    );
}

#[test]
fn f048_bessel_is_minus_3db_at_cutoff() {
    let designer = IIRFilterDesigner::new(FS);
    let mut filter = designer
        .bessel(4, &[1000.0], FilterType::Lowpass)
        .expect("bessel design should succeed");

    let g_pass = measure_gain(&mut filter, 100.0);
    let g_cut = measure_gain(&mut filter, 1000.0);
    let g_stop = measure_gain(&mut filter, 3000.0);

    assert!(
        (g_pass - 1.0).abs() < 0.03,
        "passband gain should be ~1, got {g_pass}"
    );
    assert!(
        (g_cut - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.05,
        "magnitude-normalised Bessel should be -3 dB at cutoff, got {g_cut}"
    );
    assert!(g_stop < 0.2, "stopband gain should decay, got {g_stop}");
}

#[test]
fn f048_elliptic_respects_ripple_and_stopband() {
    let designer = IIRFilterDesigner::new(FS);
    let mut filter = designer
        .elliptic(4, 1.0, 40.0, &[1000.0], FilterType::Lowpass)
        .expect("elliptic design should succeed");

    let expected_edge = 10f32.powf(-1.0 / 20.0);
    let g_pass = measure_gain(&mut filter, 200.0);
    let g_edge = measure_gain(&mut filter, 1000.0);
    let g_stop = measure_gain(&mut filter, 2400.0);

    assert!(
        g_pass >= expected_edge - 0.03 && g_pass <= 1.02,
        "passband gain should stay inside the ripple band, got {g_pass}"
    );
    assert!(
        (g_edge - expected_edge).abs() < 0.03,
        "gain at passband edge should be {expected_edge}, got {g_edge}"
    );
    assert!(g_stop < 0.05, "stopband gain should be small, got {g_stop}");
}

// ---------------------------------------------------------------------------
// F050: decimate / interpolate / rational resampling returned zeros.
// ---------------------------------------------------------------------------

#[test]
fn f050_decimate_reproduces_the_subsampled_signal() {
    let len = 1024usize;
    let signal = sine(50.0, len);
    let out = decimate(&signal, 4).expect("decimate should succeed");
    let y = out.to_vec().expect("to_vec should succeed");
    assert_eq!(y.len(), len / 4);

    let mut max_err = 0.0f32;
    for (m, &value) in y.iter().enumerate().take(y.len() - 32).skip(32) {
        let expected = (2.0 * std::f32::consts::PI * 50.0 * (4 * m) as f32 / FS).sin();
        max_err = max_err.max((value - expected).abs());
    }
    assert!(
        max_err < 0.05,
        "decimated signal deviates from the subsampled original by {max_err}"
    );
}

#[test]
fn f050_interpolate_preserves_the_original_samples() {
    let len = 256usize;
    let signal = sine(50.0, len);
    let out = interpolate(&signal, 4).expect("interpolate should succeed");
    let y = out.to_vec().expect("to_vec should succeed");
    assert_eq!(y.len(), len * 4);

    let x = signal.to_vec().expect("to_vec should succeed");
    let mut max_err = 0.0f32;
    for m in 16..len - 16 {
        max_err = max_err.max((y[m * 4] - x[m]).abs());
    }
    assert!(
        max_err < 0.05,
        "interpolated signal does not pass through the original samples (err {max_err})"
    );
}

#[test]
fn f050_rational_resampler_produces_a_real_signal() {
    let len = 512usize;
    let signal = sine(100.0, len);
    let resampler = RationalResamplerProcessor::new(3, 2, 65);
    let out = resampler
        .resample(&signal)
        .expect("resample should succeed");
    let y = out.to_vec().expect("to_vec should succeed");
    assert_eq!(y.len(), len * 3 / 2);

    // Output rate is 1.5x the input rate, so sample m corresponds to t = m/(1.5*FS).
    let mut max_err = 0.0f32;
    for (m, &value) in y.iter().enumerate().take(y.len() - 48).skip(48) {
        let expected = (2.0 * std::f32::consts::PI * 100.0 * (m as f32 / 1.5) / FS).sin();
        max_err = max_err.max((value - expected).abs());
    }
    assert!(
        max_err < 0.1,
        "rational resampling deviates from the ideal resampled sine by {max_err}"
    );
}

// ---------------------------------------------------------------------------
// F051 / F146: simd_fft returned zeros on the default optimisation level.
// ---------------------------------------------------------------------------

#[test]
fn f051_simd_fft_finds_the_peak_bin_at_default_optimisation() {
    let config = PerformanceConfig::default();
    assert_ne!(
        config.optimization_level,
        OptimizationLevel::Maximum,
        "this test must exercise the default (non-Maximum) level"
    );
    let mut processor =
        SIMDSignalProcessor::new(config).expect("processor creation should succeed");

    let n = 64usize;
    let data: Vec<f32> = (0..n)
        .map(|k| (2.0 * std::f32::consts::PI * 8.0 * k as f32 / n as f32).sin())
        .collect();
    let signal = Tensor::from_vec(data, &[n]).expect("tensor creation should succeed");

    let spectrum = processor
        .simd_fft(&signal)
        .expect("simd_fft should succeed");
    let mags = spectrum.to_vec().expect("to_vec should succeed");
    assert_eq!(mags.len(), n);

    let (peak_bin, peak_val) =
        mags.iter()
            .take(n / 2)
            .enumerate()
            .fold(
                (0usize, 0.0f32),
                |(bi, bv), (i, &v)| {
                    if v > bv {
                        (i, v)
                    } else {
                        (bi, bv)
                    }
                },
            );
    assert_eq!(peak_bin, 8, "expected the peak at bin 8, got {peak_bin}");
    assert!(peak_val > 1.0, "peak magnitude should be non-trivial");
}

// ---------------------------------------------------------------------------
// F144: STFT frame count underflowed for short signals.
// ---------------------------------------------------------------------------

#[test]
fn f144_stft_rejects_signals_shorter_than_n_fft() {
    let signal = Tensor::from_vec(vec![0.5f32; 100], &[100]).expect("tensor creation");
    let params = StftParams {
        n_fft: 512,
        hop_length: Some(128),
        center: false,
        ..Default::default()
    };
    let result = stft(&signal, params);
    assert!(
        result.is_err(),
        "STFT of a signal shorter than n_fft must return an error, not panic or allocate"
    );
}

// ---------------------------------------------------------------------------
// F145: Savitzky-Golay used an ad-hoc distance weighting.
// ---------------------------------------------------------------------------

#[test]
fn f145_savgol_preserves_low_order_polynomials() {
    let len = 64usize;
    let data: Vec<f32> = (0..len)
        .map(|i| {
            let x = i as f32;
            2.0 - 0.5 * x + 0.1 * x * x
        })
        .collect();
    let signal = Tensor::from_vec(data.clone(), &[len]).expect("tensor creation");

    let out = torsh_signal::filters::savgol_filter(&signal, 7, 2).expect("savgol should succeed");
    let y = out.to_vec().expect("to_vec should succeed");

    let mut max_rel = 0.0f32;
    for i in 0..len {
        max_rel = max_rel.max((y[i] - data[i]).abs() / data[i].abs().max(1.0));
    }
    assert!(
        max_rel < 1e-3,
        "Savitzky-Golay must reproduce polynomials of degree <= polyorder exactly (rel err {max_rel})"
    );
}

// ---------------------------------------------------------------------------
// F251 / F252: window function defects.
// ---------------------------------------------------------------------------

#[test]
fn f251_tukey_window_is_always_finite() {
    for n in 2..40usize {
        for step in 1..20 {
            let alpha = step as f32 / 20.0;
            for periodic in [false, true] {
                let w = tukey_window(n, alpha, periodic).expect("tukey window should succeed");
                let values = w.to_vec().expect("to_vec should succeed");
                assert_eq!(values.len(), n);
                for (i, v) in values.iter().enumerate() {
                    assert!(
                        v.is_finite(),
                        "tukey_window({n}, {alpha}, {periodic})[{i}] = {v}"
                    );
                    assert!(
                        (-1e-6..=1.0 + 1e-6).contains(v),
                        "tukey_window({n}, {alpha}, {periodic})[{i}] = {v} out of range"
                    );
                }
            }
        }
    }
}

#[test]
fn f252_gaussian_window_std_is_in_samples() {
    // scipy.signal.windows.gaussian(7, 1.0) -> w[n] = exp(-0.5*((n-3)/1)^2)
    let w = gaussian_window(7, 1.0, false).expect("gaussian window should succeed");
    let values = w.to_vec().expect("to_vec should succeed");
    let expected: Vec<f32> = (0..7)
        .map(|i| {
            let x = i as f32 - 3.0;
            (-0.5 * x * x).exp()
        })
        .collect();
    for i in 0..7 {
        assert!(
            (values[i] - expected[i]).abs() < 1e-5,
            "gaussian_window(7, 1.0)[{i}] = {} expected {}",
            values[i],
            expected[i]
        );
    }
}

// ---------------------------------------------------------------------------
// F253: mel filterbank used integer bin truncation.
// ---------------------------------------------------------------------------

#[test]
fn f253_mel_filterbank_matches_the_reference_triangles() {
    let n_mels = 40usize;
    let n_fft = 512usize;
    let sample_rate = 16000.0f32;
    let n_freqs = n_fft / 2 + 1;

    let fb = torsh_signal::spectral::mel_filterbank(n_mels, n_fft, sample_rate, 0.0, None)
        .expect("mel filterbank should succeed");
    let values = fb.to_vec().expect("to_vec should succeed");
    assert_eq!(values.len(), n_mels * n_freqs);

    // Reference: librosa-style triangles over fractional FFT bin frequencies.
    let hz_to_mel = |hz: f64| 2595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |mel: f64| 700.0 * (10f64.powf(mel / 2595.0) - 1.0);
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(sample_rate as f64 / 2.0);
    let hz_points: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let fft_freqs: Vec<f64> = (0..n_freqs)
        .map(|k| k as f64 * sample_rate as f64 / n_fft as f64)
        .collect();

    let mut max_err = 0.0f64;
    for i in 0..n_mels {
        let mut row_max = 0.0f64;
        for (k, &f) in fft_freqs.iter().enumerate() {
            let lower = (f - hz_points[i]) / (hz_points[i + 1] - hz_points[i]);
            let upper = (hz_points[i + 2] - f) / (hz_points[i + 2] - hz_points[i + 1]);
            let expected = lower.min(upper).max(0.0);
            row_max = row_max.max(expected);
            let got = values[i * n_freqs + k] as f64;
            max_err = max_err.max((got - expected).abs());
        }
        assert!(row_max > 0.0, "reference filter {i} should not be empty");
    }
    assert!(
        max_err < 1e-5,
        "mel filterbank deviates from the reference triangles by {max_err}"
    );
}

// ---------------------------------------------------------------------------
// F254: STFT/ISTFT used symmetric windows, breaking COLA reconstruction.
// ---------------------------------------------------------------------------

#[test]
fn f254_istft_reconstructs_the_signal() {
    let n_fft = 64usize;
    let hop = n_fft / 4;
    let len = 512usize;
    let data: Vec<f32> = (0..len)
        .map(|n| {
            let t = n as f32;
            (0.05 * t).sin() + 0.5 * (0.31 * t).cos()
        })
        .collect();
    let signal = Tensor::from_vec(data.clone(), &[len]).expect("tensor creation");

    let params = StftParams {
        n_fft,
        hop_length: Some(hop),
        win_length: Some(n_fft),
        window: Some(Window::Hann),
        center: false,
        normalized: false,
        onesided: true,
        return_complex: true,
    };
    let spec = stft(&signal, params).expect("stft should succeed");
    let rec = istft(
        &spec,
        n_fft,
        Some(hop),
        Some(n_fft),
        Some(Window::Hann),
        false,
        false,
        true,
        None,
    )
    .expect("istft should succeed");
    let y = rec.to_vec().expect("to_vec should succeed");

    // Only the interior is covered by a full set of overlapping windows.
    let mut max_err = 0.0f32;
    for i in n_fft..y.len() - n_fft {
        max_err = max_err.max((y[i] - data[i]).abs());
    }
    assert!(
        max_err < 1e-3,
        "istft(stft(x)) should reconstruct the interior of x, max error {max_err}"
    );
}
