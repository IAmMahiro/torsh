//! Analog prototype design, frequency transformations and the bilinear transform.
//!
//! This module implements the textbook zero-pole-gain (ZPK) pipeline used by
//! every classical IIR filter design:
//!
//! 1. build a normalised analog *prototype* (cutoff at 1 rad/s),
//! 2. apply a lowpass → lowpass / highpass / bandpass / bandstop frequency
//!    transformation to the requested (pre-warped) edge frequencies,
//! 3. map the analog filter to the z-plane with the bilinear transform,
//! 4. expand the resulting zeros/poles into transfer-function coefficients.
//!
//! All arithmetic is performed in `f64` (and `scirs2_core::Complex64`) so that
//! the coefficients handed to the `f32` tensor API are as accurate as possible.

use scirs2_core::Complex64;
use torsh_core::error::{Result, TorshError};

/// Zero-pole-gain representation of an analog or digital filter.
#[derive(Debug, Clone)]
pub(crate) struct Zpk {
    /// Zeros of the transfer function.
    pub zeros: Vec<Complex64>,
    /// Poles of the transfer function.
    pub poles: Vec<Complex64>,
    /// Overall gain.
    pub gain: f64,
}

impl Zpk {
    fn new(zeros: Vec<Complex64>, poles: Vec<Complex64>, gain: f64) -> Self {
        Self { zeros, poles, gain }
    }

    /// Relative degree (number of poles in excess of the zeros).
    fn relative_degree(&self) -> Result<usize> {
        if self.poles.len() < self.zeros.len() {
            return Err(TorshError::InvalidArgument(
                "Improper transfer function: more zeros than poles".to_string(),
            ));
        }
        Ok(self.poles.len() - self.zeros.len())
    }
}

fn check_order(order: usize) -> Result<()> {
    if order == 0 {
        return Err(TorshError::InvalidArgument(
            "Filter order must be greater than zero".to_string(),
        ));
    }
    if order > 24 {
        return Err(TorshError::InvalidArgument(format!(
            "Filter order {order} is too high for a direct-form implementation (max 24)"
        )));
    }
    Ok(())
}

/// Butterworth analog lowpass prototype (cutoff at 1 rad/s).
pub(crate) fn butter_ap(order: usize) -> Result<Zpk> {
    check_order(order)?;
    let n = order as f64;
    let poles = (0..order)
        .map(|k| {
            let m = -(n - 1.0) + 2.0 * k as f64;
            -(Complex64::new(0.0, std::f64::consts::PI * m / (2.0 * n)).exp())
        })
        .collect();
    Ok(Zpk::new(Vec::new(), poles, 1.0))
}

/// Chebyshev type I analog lowpass prototype with `ripple_db` passband ripple.
///
/// The magnitude equals `-ripple_db` dB exactly at 1 rad/s.
pub(crate) fn cheb1_ap(order: usize, ripple_db: f64) -> Result<Zpk> {
    check_order(order)?;
    if ripple_db <= 0.0 {
        return Err(TorshError::InvalidArgument(
            "Chebyshev passband ripple must be positive".to_string(),
        ));
    }
    let n = order as f64;
    let eps = (10f64.powf(0.1 * ripple_db) - 1.0).sqrt();
    let mu = (1.0 / eps).asinh() / n;

    let poles: Vec<Complex64> = (0..order)
        .map(|k| {
            let m = -(n - 1.0) + 2.0 * k as f64;
            let theta = std::f64::consts::PI * m / (2.0 * n);
            -Complex64::new(mu, theta).sinh()
        })
        .collect();

    let mut gain: f64 = poles
        .iter()
        .map(|p| -p)
        .fold(Complex64::new(1.0, 0.0), |a, b| a * b)
        .re;
    if order % 2 == 0 {
        gain /= (1.0 + eps * eps).sqrt();
    }
    Ok(Zpk::new(Vec::new(), poles, gain))
}

/// Chebyshev type II (inverse Chebyshev) analog prototype.
///
/// The magnitude equals `-attenuation_db` dB at 1 rad/s, which is the
/// *stopband* edge for this filter family.
pub(crate) fn cheb2_ap(order: usize, attenuation_db: f64) -> Result<Zpk> {
    check_order(order)?;
    if attenuation_db <= 0.0 {
        return Err(TorshError::InvalidArgument(
            "Chebyshev stopband attenuation must be positive".to_string(),
        ));
    }
    let n = order as f64;
    let de = 1.0 / (10f64.powf(0.1 * attenuation_db) - 1.0).sqrt();
    let mu = (1.0 / de).asinh() / n;

    // Zero locations: purely imaginary, skipping the origin for odd orders.
    let mut zero_m: Vec<f64> = Vec::new();
    if order % 2 == 1 {
        let mut m = -(n - 1.0);
        while m < 0.0 {
            zero_m.push(m);
            m += 2.0;
        }
        let mut m = 2.0;
        while m < n {
            zero_m.push(m);
            m += 2.0;
        }
    } else {
        for k in 0..order {
            zero_m.push(-(n - 1.0) + 2.0 * k as f64);
        }
    }
    let zeros: Vec<Complex64> = zero_m
        .iter()
        .map(|&m| Complex64::new(0.0, 1.0 / (std::f64::consts::PI * m / (2.0 * n)).sin()))
        .collect();

    let poles: Vec<Complex64> = (0..order)
        .map(|k| {
            let m = -(n - 1.0) + 2.0 * k as f64;
            let p = -(Complex64::new(0.0, std::f64::consts::PI * m / (2.0 * n)).exp());
            let shaped = Complex64::new(mu.sinh() * p.re, mu.cosh() * p.im);
            Complex64::new(1.0, 0.0) / shaped
        })
        .collect();

    let num: Complex64 = poles
        .iter()
        .map(|p| -p)
        .fold(Complex64::new(1.0, 0.0), |a, b| a * b);
    let den: Complex64 = zeros
        .iter()
        .map(|z| -z)
        .fold(Complex64::new(1.0, 0.0), |a, b| a * b);
    Ok(Zpk::new(zeros, poles, (num / den).re))
}

/// Bessel (Thomson) analog lowpass prototype, magnitude-normalised so that the
/// response is -3 dB at 1 rad/s (scipy's `norm="mag"` convention).
pub(crate) fn bessel_ap(order: usize) -> Result<Zpk> {
    check_order(order)?;
    if order > 12 {
        return Err(TorshError::InvalidArgument(format!(
            "Bessel filter order {order} is not supported (max 12)"
        )));
    }

    // Reverse Bessel polynomial theta_n(s) = sum_k a_k s^k, computed by the
    // downward recurrence a_k = a_{k+1} * (2n-k)(k+1) / (2(n-k)) with a_n = 1.
    let n = order;
    let mut coeffs = vec![0.0f64; n + 1];
    coeffs[n] = 1.0;
    for k in (0..n).rev() {
        let kf = k as f64;
        let nf = n as f64;
        coeffs[k] = coeffs[k + 1] * (2.0 * nf - kf) * (kf + 1.0) / (2.0 * (nf - kf));
    }

    let poles = poly_roots(&coeffs)?;
    let gain = poles
        .iter()
        .map(|p| -p)
        .fold(Complex64::new(1.0, 0.0), |a, b| a * b)
        .re;
    let proto = Zpk::new(Vec::new(), poles, gain);

    // Find the -3 dB frequency of the delay-normalised prototype and rescale.
    let target = std::f64::consts::FRAC_1_SQRT_2;
    let mag = |w: f64| -> f64 { analog_magnitude(&proto, w) };
    let (mut lo, mut hi) = (1e-4f64, 1e4f64);
    if mag(hi) > target {
        return Err(TorshError::ComputeError(
            "Bessel prototype normalisation failed to bracket the -3 dB point".to_string(),
        ));
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mag(mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let w3 = 0.5 * (lo + hi);
    let poles: Vec<Complex64> = proto.poles.iter().map(|p| p / w3).collect();
    let gain = poles
        .iter()
        .map(|p| -p)
        .fold(Complex64::new(1.0, 0.0), |a, b| a * b)
        .re;
    Ok(Zpk::new(Vec::new(), poles, gain))
}

/// Complete elliptic integral of the first kind `K(m)` via the AGM.
fn ellipk(m: f64) -> f64 {
    if m >= 1.0 {
        return f64::INFINITY;
    }
    let mut a = 1.0f64;
    let mut b = (1.0 - m).sqrt();
    for _ in 0..60 {
        if (a - b).abs() < 1e-16 * a.abs() {
            break;
        }
        let next_a = 0.5 * (a + b);
        b = (a * b).sqrt();
        a = next_a;
    }
    std::f64::consts::PI / (2.0 * a)
}

/// Jacobi elliptic functions `(sn, cn, dn)` for real `u` and parameter `m`.
///
/// Uses the descending Landen (AGM) transformation of Abramowitz & Stegun 16.4.
fn ellipj(u: f64, m: f64) -> (f64, f64, f64) {
    if m < 1e-14 {
        return (u.sin(), u.cos(), 1.0);
    }
    if (1.0 - m).abs() < 1e-14 {
        let t = u.tanh();
        let sech = 1.0 / u.cosh();
        return (t, sech, sech);
    }

    let mut a = vec![1.0f64];
    let mut b = vec![(1.0 - m).sqrt()];
    let mut c = vec![m.sqrt()];
    let mut n = 0usize;
    while n < 30 && c[n].abs() > 1e-16 * a[n].abs() {
        let an = 0.5 * (a[n] + b[n]);
        let bn = (a[n] * b[n]).sqrt();
        let cn = 0.5 * (a[n] - b[n]);
        a.push(an);
        b.push(bn);
        c.push(cn);
        n += 1;
    }

    let mut phi = (2.0f64).powi(n as i32) * a[n] * u;
    for i in (1..=n).rev() {
        let arg = (c[i] / a[i] * phi.sin()).clamp(-1.0, 1.0);
        phi = 0.5 * (phi + arg.asin());
    }
    let sn = phi.sin();
    let cn = phi.cos();
    let dn = (1.0 - m * sn * sn).max(0.0).sqrt();
    (sn, cn, dn)
}

/// Inverse Jacobi `sn` for a complex argument, via descending Landen steps.
fn arc_jac_sn(w: Complex64, m: f64) -> Result<Complex64> {
    let complement = |k: f64| (1.0 - k * k).max(0.0).sqrt();
    let k = m.sqrt();
    if k > 1.0 {
        return Err(TorshError::InvalidArgument(
            "Elliptic modulus out of range".to_string(),
        ));
    }

    let mut ks = vec![k];
    let mut iterations = 0;
    while *ks.last().unwrap_or(&0.0) != 0.0 {
        let last = ks[ks.len() - 1];
        let kp = complement(last);
        ks.push((1.0 - kp) / (1.0 + kp));
        iterations += 1;
        if iterations > 10 {
            return Err(TorshError::ComputeError(
                "Landen descent for the inverse Jacobi sn did not converge".to_string(),
            ));
        }
    }

    let capital_k =
        ks[1..].iter().fold(1.0, |acc, k| acc * (1.0 + k)) * std::f64::consts::FRAC_PI_2;

    let mut wn = w;
    for idx in 0..ks.len() - 1 {
        let kn = ks[idx];
        let knext = ks[idx + 1];
        let root = (Complex64::new(1.0, 0.0) - (wn * kn) * (wn * kn)).sqrt();
        wn = 2.0 * wn / ((1.0 + knext) * (1.0 + root));
    }

    let u = 2.0 / std::f64::consts::PI * wn.asin();
    Ok(u * capital_k)
}

/// Solve the elliptic degree equation: given the order and `m1 = k1^2`,
/// return the modulus `m = k^2` of the corresponding elliptic filter.
fn ellipdeg(order: usize, m1: f64) -> f64 {
    let k1 = ellipk(m1);
    let k1p = ellipk(1.0 - m1);
    let q1 = (-std::f64::consts::PI * k1p / k1).exp();
    let q = q1.powf(1.0 / order as f64);

    let mut num = 0.0f64;
    for n in 0..=7u32 {
        num += q.powi((n * (n + 1)) as i32);
    }
    let mut den = 1.0f64;
    for n in 1..=8u32 {
        den += 2.0 * q.powi((n * n) as i32);
    }
    16.0 * q * (num / den).powi(4)
}

/// Elliptic (Cauer) analog lowpass prototype.
///
/// `ripple_db` is the passband ripple and `attenuation_db` the stopband
/// attenuation; the magnitude equals `-ripple_db` dB at 1 rad/s.
pub(crate) fn ellip_ap(order: usize, ripple_db: f64, attenuation_db: f64) -> Result<Zpk> {
    check_order(order)?;
    if ripple_db <= 0.0 || attenuation_db <= 0.0 {
        return Err(TorshError::InvalidArgument(
            "Elliptic ripple and attenuation must be positive".to_string(),
        ));
    }
    if attenuation_db <= ripple_db {
        return Err(TorshError::InvalidArgument(
            "Elliptic stopband attenuation must exceed the passband ripple".to_string(),
        ));
    }
    if order == 1 {
        // Degenerate case: a single real pole, identical to Chebyshev type I.
        return cheb1_ap(1, ripple_db);
    }

    let eps_sq = 10f64.powf(0.1 * ripple_db) - 1.0;
    let eps = eps_sq.sqrt();
    let ck1_sq = eps_sq / (10f64.powf(0.1 * attenuation_db) - 1.0);
    if ck1_sq <= 0.0 || ck1_sq >= 1.0 {
        return Err(TorshError::InvalidArgument(
            "Cannot design an elliptic filter with the given ripple/attenuation".to_string(),
        ));
    }

    let m = ellipdeg(order, ck1_sq);
    let capk = ellipk(m);

    // Sample points along the real axis of the elliptic prototype.
    let start = 1 - order % 2;
    let j_values: Vec<f64> = (start..order).step_by(2).map(|j| j as f64).collect();
    let n = order as f64;

    let mut zeros: Vec<Complex64> = Vec::new();
    let mut poles: Vec<Complex64> = Vec::new();

    let r = arc_jac_sn(Complex64::new(0.0, 1.0 / eps), ck1_sq)?;
    if r.re.abs() > 1e-8 * r.norm().max(1.0) {
        return Err(TorshError::ComputeError(
            "Inverse Jacobi sn returned an unexpected complex value".to_string(),
        ));
    }
    let v0 = capk * r.im / (n * ellipk(ck1_sq));
    let (sv, cv, dv) = ellipj(v0, 1.0 - m);

    for &j in &j_values {
        let (sn, cn, dn) = ellipj(j * capk / n, m);
        if sn.abs() > 1e-12 {
            let z = Complex64::new(0.0, 1.0 / (m.sqrt() * sn));
            zeros.push(z);
            zeros.push(z.conj());
        }
        let numer = Complex64::new(cn * dn * sv * cv, sn * dv);
        let denom = 1.0 - (dn * sv) * (dn * sv);
        if denom.abs() < 1e-15 {
            return Err(TorshError::ComputeError(
                "Elliptic pole placement produced a singular denominator".to_string(),
            ));
        }
        let p = -numer / denom;
        poles.push(p);
        if p.im.abs() > 1e-12 {
            poles.push(p.conj());
        }
    }

    let num: Complex64 = poles
        .iter()
        .map(|p| -p)
        .fold(Complex64::new(1.0, 0.0), |a, b| a * b);
    let den: Complex64 = zeros
        .iter()
        .map(|z| -z)
        .fold(Complex64::new(1.0, 0.0), |a, b| a * b);
    let mut gain = (num / den).re;
    if order % 2 == 0 {
        gain /= (1.0 + eps_sq).sqrt();
    }

    if poles.iter().any(|p| p.re >= 0.0) {
        return Err(TorshError::ComputeError(
            "Elliptic design produced unstable poles".to_string(),
        ));
    }

    Ok(Zpk::new(zeros, poles, gain))
}

/// Magnitude of an analog ZPK response at angular frequency `w`.
fn analog_magnitude(zpk: &Zpk, w: f64) -> f64 {
    let s = Complex64::new(0.0, w);
    let num = zpk
        .zeros
        .iter()
        .fold(Complex64::new(zpk.gain, 0.0), |acc, z| acc * (s - z));
    let den = zpk
        .poles
        .iter()
        .fold(Complex64::new(1.0, 0.0), |acc, p| acc * (s - p));
    (num / den).norm()
}

/// Roots of a real polynomial `sum coeffs[k] x^k` via the Durand-Kerner method.
fn poly_roots(coeffs: &[f64]) -> Result<Vec<Complex64>> {
    let degree = coeffs.len() - 1;
    if degree == 0 {
        return Ok(Vec::new());
    }
    let lead = coeffs[degree];
    if lead.abs() < f64::EPSILON {
        return Err(TorshError::InvalidArgument(
            "Leading polynomial coefficient must be non-zero".to_string(),
        ));
    }
    let monic: Vec<f64> = coeffs.iter().map(|c| c / lead).collect();

    let eval_poly = |x: Complex64| -> Complex64 {
        let mut acc = Complex64::new(0.0, 0.0);
        for c in monic.iter().rev() {
            acc = acc * x + Complex64::new(*c, 0.0);
        }
        acc
    };

    let seed = Complex64::new(0.4, 0.9);
    let mut roots: Vec<Complex64> = (0..degree).map(|i| seed.powu(i as u32)).collect();

    for _ in 0..1000 {
        let mut max_delta = 0.0f64;
        for i in 0..degree {
            let mut denom = Complex64::new(1.0, 0.0);
            for j in 0..degree {
                if i != j {
                    denom *= roots[i] - roots[j];
                }
            }
            if denom.norm() < 1e-300 {
                continue;
            }
            let delta = eval_poly(roots[i]) / denom;
            roots[i] -= delta;
            max_delta = max_delta.max(delta.norm());
        }
        if max_delta < 1e-14 {
            break;
        }
    }

    Ok(roots)
}

/// Lowpass → lowpass transformation to cutoff `wo` (rad/s).
pub(crate) fn lp2lp(zpk: &Zpk, wo: f64) -> Result<Zpk> {
    let degree = zpk.relative_degree()?;
    let zeros: Vec<Complex64> = zpk.zeros.iter().map(|z| z * wo).collect();
    let poles: Vec<Complex64> = zpk.poles.iter().map(|p| p * wo).collect();
    Ok(Zpk::new(zeros, poles, zpk.gain * wo.powi(degree as i32)))
}

/// Lowpass → highpass transformation with cutoff `wo` (rad/s).
pub(crate) fn lp2hp(zpk: &Zpk, wo: f64) -> Result<Zpk> {
    let degree = zpk.relative_degree()?;
    let zeros: Vec<Complex64> = zpk
        .zeros
        .iter()
        .map(|z| Complex64::new(wo, 0.0) / z)
        .collect();
    let poles: Vec<Complex64> = zpk
        .poles
        .iter()
        .map(|p| Complex64::new(wo, 0.0) / p)
        .collect();

    let num: Complex64 = zpk
        .zeros
        .iter()
        .map(|z| -z)
        .fold(Complex64::new(1.0, 0.0), |a, b| a * b);
    let den: Complex64 = zpk
        .poles
        .iter()
        .map(|p| -p)
        .fold(Complex64::new(1.0, 0.0), |a, b| a * b);

    let mut zeros = zeros;
    zeros.extend(std::iter::repeat_n(Complex64::new(0.0, 0.0), degree));
    Ok(Zpk::new(zeros, poles, zpk.gain * (num / den).re))
}

/// Lowpass → bandpass transformation (centre `wo`, bandwidth `bw`, rad/s).
pub(crate) fn lp2bp(zpk: &Zpk, wo: f64, bw: f64) -> Result<Zpk> {
    let degree = zpk.relative_degree()?;
    let half = Complex64::new(bw / 2.0, 0.0);
    let wo_sq = Complex64::new(wo * wo, 0.0);

    let expand = |roots: &[Complex64]| -> Vec<Complex64> {
        let mut out = Vec::with_capacity(roots.len() * 2);
        for r in roots {
            let scaled = r * half;
            let disc = (scaled * scaled - wo_sq).sqrt();
            out.push(scaled + disc);
            out.push(scaled - disc);
        }
        out
    };

    let mut zeros = expand(&zpk.zeros);
    let poles = expand(&zpk.poles);
    zeros.extend(std::iter::repeat_n(Complex64::new(0.0, 0.0), degree));
    Ok(Zpk::new(zeros, poles, zpk.gain * bw.powi(degree as i32)))
}

/// Lowpass → bandstop transformation (centre `wo`, bandwidth `bw`, rad/s).
pub(crate) fn lp2bs(zpk: &Zpk, wo: f64, bw: f64) -> Result<Zpk> {
    let degree = zpk.relative_degree()?;
    let half = Complex64::new(bw / 2.0, 0.0);
    let wo_sq = Complex64::new(wo * wo, 0.0);

    let expand = |roots: &[Complex64]| -> Vec<Complex64> {
        let mut out = Vec::with_capacity(roots.len() * 2);
        for r in roots {
            let inv = half / r;
            let disc = (inv * inv - wo_sq).sqrt();
            out.push(inv + disc);
            out.push(inv - disc);
        }
        out
    };

    let mut zeros = expand(&zpk.zeros);
    let poles = expand(&zpk.poles);
    for _ in 0..degree {
        zeros.push(Complex64::new(0.0, wo));
        zeros.push(Complex64::new(0.0, -wo));
    }

    let num: Complex64 = zpk
        .zeros
        .iter()
        .map(|z| -z)
        .fold(Complex64::new(1.0, 0.0), |a, b| a * b);
    let den: Complex64 = zpk
        .poles
        .iter()
        .map(|p| -p)
        .fold(Complex64::new(1.0, 0.0), |a, b| a * b);
    Ok(Zpk::new(zeros, poles, zpk.gain * (num / den).re))
}

/// Bilinear transform from the s-plane to the z-plane at sampling rate `fs`.
pub(crate) fn bilinear(zpk: &Zpk, fs: f64) -> Result<Zpk> {
    let degree = zpk.relative_degree()?;
    let fs2 = Complex64::new(2.0 * fs, 0.0);

    let zeros_z: Vec<Complex64> = zpk.zeros.iter().map(|z| (fs2 + z) / (fs2 - z)).collect();
    let poles_z: Vec<Complex64> = zpk.poles.iter().map(|p| (fs2 + p) / (fs2 - p)).collect();

    let num: Complex64 = zpk
        .zeros
        .iter()
        .fold(Complex64::new(1.0, 0.0), |acc, z| acc * (fs2 - z));
    let den: Complex64 = zpk
        .poles
        .iter()
        .fold(Complex64::new(1.0, 0.0), |acc, p| acc * (fs2 - p));

    let mut zeros_z = zeros_z;
    zeros_z.extend(std::iter::repeat_n(Complex64::new(-1.0, 0.0), degree));
    Ok(Zpk::new(zeros_z, poles_z, zpk.gain * (num / den).re))
}

/// Expand zeros/poles into real transfer-function coefficients `(b, a)`.
pub(crate) fn zpk2tf(zpk: &Zpk) -> (Vec<f64>, Vec<f64>) {
    let mut b = poly_from_roots(&zpk.zeros);
    for coeff in b.iter_mut() {
        *coeff *= zpk.gain;
    }
    let a = poly_from_roots(&zpk.poles);
    (b, a)
}

/// Monic polynomial coefficients (highest power first) from its roots.
///
/// The roots of a real filter come in conjugate pairs, so the imaginary parts
/// cancel and only the real parts are kept.
fn poly_from_roots(roots: &[Complex64]) -> Vec<f64> {
    let mut coeffs = vec![Complex64::new(1.0, 0.0)];
    for root in roots {
        let mut next = vec![Complex64::new(0.0, 0.0); coeffs.len() + 1];
        for (i, c) in coeffs.iter().enumerate() {
            next[i] += c;
            next[i + 1] -= c * root;
        }
        coeffs = next;
    }
    coeffs.into_iter().map(|c| c.re).collect()
}

/// Pre-warp a digital edge frequency (normalised to Nyquist = 1) for the
/// bilinear transform performed at `fs = 2`.
pub(crate) fn prewarp(normalized: f64) -> f64 {
    4.0 * (std::f64::consts::PI * normalized / 2.0).tan()
}

/// Evaluate the magnitude response |H(e^{j pi f})| of a digital filter,
/// where `f` is normalised to the Nyquist frequency.
pub(crate) fn digital_response(b: &[f64], a: &[f64], normalized_freq: f64) -> Complex64 {
    let w = std::f64::consts::PI * normalized_freq;
    let z_inv = Complex64::new(0.0, -w).exp();
    let mut num = Complex64::new(0.0, 0.0);
    for coeff in b.iter().rev() {
        num = num * z_inv + Complex64::new(*coeff, 0.0);
    }
    let mut den = Complex64::new(0.0, 0.0);
    for coeff in a.iter().rev() {
        den = den * z_inv + Complex64::new(*coeff, 0.0);
    }
    if den.norm() < 1e-300 {
        return Complex64::new(0.0, 0.0);
    }
    num / den
}

/// Roots of a real polynomial given in descending power order.
pub(crate) fn roots_descending(coeffs: &[f64]) -> Result<Vec<Complex64>> {
    let ascending: Vec<f64> = coeffs.iter().rev().copied().collect();
    poly_roots(&ascending)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn design_lowpass(zpk: Zpk, normalized_cutoff: f64) -> (Vec<f64>, Vec<f64>) {
        let warped = prewarp(normalized_cutoff);
        let analog = lp2lp(&zpk, warped).expect("lp2lp should succeed");
        let digital = bilinear(&analog, 2.0).expect("bilinear should succeed");
        zpk2tf(&digital)
    }

    #[test]
    fn butterworth_prototype_matches_known_poles() {
        let zpk = butter_ap(2).expect("prototype should succeed");
        let mut re: Vec<f64> = zpk.poles.iter().map(|p| p.re).collect();
        re.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        for value in re {
            assert!((value + std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
        }
    }

    #[test]
    fn butterworth_lowpass_is_minus_3db_at_cutoff() {
        let (b, a) = design_lowpass(butter_ap(4).expect("prototype"), 0.25);
        let mag = digital_response(&b, &a, 0.25).norm();
        assert!(
            (mag - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-6,
            "magnitude at cutoff was {mag}"
        );
        assert!((digital_response(&b, &a, 0.0).norm() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn chebyshev1_hits_the_ripple_at_the_edge() {
        let (b, a) = design_lowpass(cheb1_ap(5, 1.0).expect("prototype"), 0.3);
        let mag = digital_response(&b, &a, 0.3).norm();
        let expected = 10f64.powf(-1.0 / 20.0);
        assert!((mag - expected).abs() < 1e-6, "magnitude was {mag}");
    }

    #[test]
    fn chebyshev2_hits_the_attenuation_at_the_edge() {
        let (b, a) = design_lowpass(cheb2_ap(5, 40.0).expect("prototype"), 0.3);
        let mag = digital_response(&b, &a, 0.3).norm();
        let expected = 10f64.powf(-40.0 / 20.0);
        assert!((mag - expected).abs() < 1e-6, "magnitude was {mag}");
    }

    #[test]
    fn bessel_prototype_is_minus_3db_at_one_rad_per_second() {
        let zpk = bessel_ap(4).expect("prototype should succeed");
        let mag = analog_magnitude(&zpk, 1.0);
        assert!(
            (mag - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-9,
            "magnitude was {mag}"
        );
        // All poles must live in the open left half plane.
        assert!(zpk.poles.iter().all(|p| p.re < 0.0));
    }

    #[test]
    fn bandpass_transformation_keeps_the_centre_frequency() {
        let zpk = butter_ap(3).expect("prototype");
        let (low, high) = (0.2, 0.4);
        let (wl, wh) = (prewarp(low), prewarp(high));
        let analog = lp2bp(&zpk, (wl * wh).sqrt(), wh - wl).expect("lp2bp");
        let digital = bilinear(&analog, 2.0).expect("bilinear");
        let (b, a) = zpk2tf(&digital);

        assert!(
            (digital_response(&b, &a, low).norm() - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-6
        );
        assert!(
            (digital_response(&b, &a, high).norm() - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-6
        );
        assert!(digital_response(&b, &a, 0.0).norm() < 1e-6);
    }

    #[test]
    fn bandstop_transformation_notches_the_band() {
        let zpk = butter_ap(3).expect("prototype");
        let (low, high) = (0.2, 0.4);
        let (wl, wh) = (prewarp(low), prewarp(high));
        let analog = lp2bs(&zpk, (wl * wh).sqrt(), wh - wl).expect("lp2bs");
        let digital = bilinear(&analog, 2.0).expect("bilinear");
        let (b, a) = zpk2tf(&digital);

        assert!((digital_response(&b, &a, 0.0).norm() - 1.0).abs() < 1e-6);
        assert!((digital_response(&b, &a, 1.0).norm() - 1.0).abs() < 1e-6);
        assert!(
            (digital_response(&b, &a, low).norm() - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-6
        );
    }

    #[test]
    fn elliptic_prototype_meets_its_specification() {
        let zpk = ellip_ap(4, 1.0, 40.0).expect("elliptic prototype should succeed");
        let edge = analog_magnitude(&zpk, 1.0);
        let expected = 10f64.powf(-1.0 / 20.0);
        assert!(
            (edge - expected).abs() < 1e-6,
            "passband edge magnitude was {edge}"
        );
        assert!(zpk.poles.iter().all(|p| p.re < 0.0));

        // Deep in the stopband the equiripple level must stay below -40 dB.
        let mut worst: f64 = 0.0;
        for i in 0..200 {
            let w = 1.6 + i as f64 * 0.1;
            worst = worst.max(analog_magnitude(&zpk, w));
        }
        assert!(
            worst <= 10f64.powf(-40.0 / 20.0) * 1.001,
            "stopband peak {worst}"
        );
    }

    #[test]
    fn elliptic_integrals_match_reference_values() {
        // K(0.5) = 1.8540746773013719
        assert!((ellipk(0.5) - 1.854_074_677_301_372).abs() < 1e-12);
        // scipy.special.ellipj(0.6, 0.5)
        let (sn, cn, dn) = ellipj(0.6, 0.5);
        assert!((sn - 0.550_831_128_696_534_4).abs() < 1e-12, "sn = {sn}");
        assert!((cn - 0.834_616_719_014_723_6).abs() < 1e-12, "cn = {cn}");
        assert!((dn - 0.921_027_976_681_192_4).abs() < 1e-12, "dn = {dn}");
    }

    #[test]
    fn polynomial_roots_round_trip() {
        // (x - 1)(x + 2)(x^2 + 1) = x^4 + x^3 - x^2 + x - 2
        let roots = roots_descending(&[1.0, 1.0, -1.0, 1.0, -2.0]).expect("roots");
        assert_eq!(roots.len(), 4);
        let has = |re: f64, im: f64| {
            roots
                .iter()
                .any(|r| (r.re - re).abs() < 1e-8 && (r.im - im).abs() < 1e-8)
        };
        assert!(has(1.0, 0.0) && has(-2.0, 0.0) && has(0.0, 1.0) && has(0.0, -1.0));
    }
}
