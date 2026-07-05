# torsh-special

Special mathematical functions for ToRSh, leveraging scirs2-special for optimized implementations.

**Status**: Stable. **Tests**: 186/186 passing (`cargo nextest run --all-features`).

## Overview

This crate provides a comprehensive collection of 135+ special mathematical functions organized into 19 mathematical families:

- **Bessel Functions**: J₀, J₁, Jₙ, Y₀, Y₁, Yₙ, I₀, I₁, Iₙ, K₀, K₁, Kₙ (cylindrical), spherical Bessel, Hankel functions
- **Gamma Functions**: Gamma, log-gamma, digamma, polygamma, beta
- **Error Functions**: erf, erfc, erfcx, erfinv, Fresnel integrals
- **Elliptic Functions**: Complete and incomplete elliptic integrals (K, E, F), Jacobi functions (sn, cn, dn), Weierstrass functions, theta functions
- **Exponential Integrals**: Ei, Eₙ, logarithmic integral, sine/cosine integrals
- **Hypergeometric Functions**: ₁F₁, ₂F₁, pFq, Meijer G, Appell F₁
- **Orthogonal Polynomials**: Legendre, Chebyshev, Hermite, Laguerre, Jacobi, Gegenbauer
- **Advanced Functions**: Riemann zeta, polylogarithm, Hurwitz zeta, Dirichlet eta, Barnes G
- **Advanced Special**: Dawson, Kelvin (ber, bei, ker, kei), parabolic cylinder, Spence, Struve, Voigt
- **Airy Functions**: Ai, Bi and their derivatives
- **Coulomb Wave Functions**: F_L(η,ρ), G_L(η,ρ) for quantum scattering
- **Mathieu Functions**: ce_n, se_n, characteristic values for periodic boundaries
- **Lommel Functions**: s_μ,ν, S_μ,ν for diffraction theory
- **Spheroidal Wave Functions**: Prolate and oblate angular/radial functions for electromagnetic scattering
- **Lambert W Functions**: Principal and secondary branches with applications
- **Statistical Functions**: Normal, Student's t, chi-squared, F-distribution CDFs/PDFs, incomplete beta
- **Complex Functions**: Complex gamma, zeta, erf, Bessel functions with branch cuts
- **Performance Optimizations**: SIMD-accelerated, fast approximations, smart caching, lookup tables
- **Visualization Tools**: Function analysis, accuracy comparison, ASCII plotting

All functions are exported directly from the crate root (there is no `special::` namespace); the examples
below assume `use torsh_special::prelude::*;` (or `use torsh_special::*;`) unless noted otherwise.

## Usage

### Bessel Functions

```rust
use torsh_special::prelude::*;
use torsh_tensor::prelude::*;

// Bessel functions of the first kind
let x = tensor![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
let j0 = bessel_j0(&x)?;      // J₀(x)
let j1 = bessel_j1(&x)?;      // J₁(x)
let jn = bessel_jn(3, &x)?;   // J₃(x)

// Bessel functions of the second kind
let y0 = bessel_y0(&x)?;      // Y₀(x)
let y1 = bessel_y1(&x)?;      // Y₁(x)
let yn = bessel_yn(3, &x)?;   // Y₃(x)

// Modified Bessel functions
let i0 = bessel_i0(&x)?;      // I₀(x)
let i1 = bessel_i1(&x)?;      // I₁(x)
let k0 = bessel_k0(&x)?;      // K₀(x)
let k1 = bessel_k1(&x)?;      // K₁(x)

// Spherical Bessel functions (in the `trigonometric` module, not re-exported in `prelude`)
let j0_spherical = torsh_special::trigonometric::spherical_jn(0, &x)?;
let y0_spherical = torsh_special::trigonometric::spherical_yn(0, &x)?;
```

### Gamma and Related Functions

```rust
// Gamma function
let x = tensor![0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
let gamma_x = gamma(&x)?;

// Log-gamma function (more stable for large values)
let lgamma_x = lgamma(&x)?;

// Digamma (psi) function - derivative of log-gamma
let digamma_x = digamma(&x)?;

// Polygamma functions
let trigamma = polygamma(1, &x)?;   // ψ'(x)
let tetragamma = polygamma(2, &x)?; // ψ''(x)

// Beta function
let a = tensor![0.5, 1.0, 2.0];
let b = tensor![1.0, 2.0, 3.0];
let beta_ab = beta(&a, &b)?;
```

### Error Functions

```rust
// Error function and complementary error function
let x = tensor![-2.0, -1.0, 0.0, 1.0, 2.0];
let erf_x = erf(&x)?;
let erfc_x = erfc(&x)?;

// Scaled complementary error function (for large x)
let erfcx_x = erfcx(&x)?;  // exp(x²) * erfc(x)

// Inverse error function
let p = tensor![0.1, 0.5, 0.9];
let erfinv_p = erfinv(&p)?;

// Fresnel integrals (returned as a single (S, C) tuple)
let (s, c) = fresnel(&x)?;  // S(x) and C(x)
```

### Elliptic Functions

```rust
// Complete elliptic integrals
let m = tensor![0.0, 0.5, 0.9, 0.99]; // parameter m = k²
let ellipk = elliptic_k(&m)?;  // K(m)
let ellipe = elliptic_e(&m)?;  // E(m)

// Incomplete elliptic integrals
let phi = tensor![0.5, 1.0, 1.5];
let ellipf = elliptic_f(&phi, &m)?;            // F(φ,m)
let ellipe_inc = elliptic_e_incomplete(&phi, &m)?; // E(φ,m)

// Jacobi elliptic functions (returned individually, not as a combined tuple)
let u = tensor![0.0, 0.5, 1.0, 1.5];
let mm = tensor![0.0, 0.5, 0.9];
let sn = jacobi_sn(&u, &mm)?;
let cn = jacobi_cn(&u, &mm)?;
let dn = jacobi_dn(&u, &mm)?;
```

### Exponential and Logarithmic Integrals

```rust
// Exponential integral
let ei = exponential_integral_ei(&x)?;  // Ei(x)

// Exponential integral E_n(x)
let e1 = exponential_integral_e1(&x)?;     // E₁(x)
let e2 = exponential_integral_en(2, &x)?;  // E₂(x)

// Logarithmic integral
let li = logarithmic_integral(&x)?;  // li(x)

// Sine and cosine integrals (returned individually, not as a combined tuple)
let si = sine_integral(&x)?;    // Si(x)
let ci = cosine_integral(&x)?;  // Ci(x)

// Hyperbolic sine and cosine integrals
let shi = hyperbolic_sine_integral(&x)?;   // Shi(x)
let chi = hyperbolic_cosine_integral(&x)?; // Chi(x)
```

### Other Special Functions

```rust
// Riemann zeta function
let s = tensor![0.5, 1.5, 2.0, 3.0];
let zeta_s = riemann_zeta(&s)?;

// Airy functions (returned individually, not as a combined tuple)
let x = tensor![-2.0, -1.0, 0.0, 1.0, 2.0];
let ai = airy_ai(&x)?;
let aip = airy_ai_prime(&x)?;
let bi = airy_bi(&x)?;
let bip = airy_bi_prime(&x)?;

// Struve functions (tensor argument first, order second)
let h0 = struve_h(&x, 0)?;  // H₀(x)
let h1 = struve_h(&x, 1)?;  // H₁(x)

// Hypergeometric functions (a, b, c are plain scalars; only z is a tensor)
let (a, b, c) = (0.5_f32, 1.0_f32, 1.5_f32);
let z = tensor![0.1, 0.5, 0.9];
let hyp2f1 = hypergeometric_2f1(a, b, c, &z)?;

// Legendre polynomials
let n = 3;
let x = tensor![-1.0, -0.5, 0.0, 0.5, 1.0];
let pn = legendre_p(n, &x)?;

// Associated Legendre functions (order n, then degree m)
let m = 1;
let pmn = legendre_p_associated(n, m, &x)?;
```

### Spheroidal Wave Functions

```rust
use torsh_special::{prolate_angular, prolate_radial, oblate_angular, oblate_radial, spheroidal_eigenvalue};

// Prolate spheroidal wave functions (electromagnetic scattering)
let n = 2;  // Degree
let m = 0;  // Order
let c = 2.0; // Spheroidicity parameter

// Angular function S_nm(c, η) where η ∈ [-1, 1] — scalar in, scalar out
let eta = 0.5;
let s_value = prolate_angular(n, m, c, eta)?;

// Radial function R_nm(c, ξ) where ξ ∈ [1, ∞)
let xi = 2.0;
let r_value = prolate_radial(n, m, c, xi)?;

// Oblate spheroidal wave functions (acoustic cavities)
let xi_oblate = 0.5; // ξ ∈ [0, 1] for oblate
let s_oblate = oblate_angular(n, m, c, eta)?;
let r_oblate = oblate_radial(n, m, c, xi_oblate)?;

// Eigenvalues λ_nm(c)
let lambda_0 = spheroidal_eigenvalue(n, m, 0.0)?; // Spherical limit
let lambda = spheroidal_eigenvalue(n, m, c)?;    // Spheroidal

// Tensor-batched variants are also available: prolate_angular_tensor,
// prolate_radial_tensor, oblate_angular_tensor, oblate_radial_tensor.
```

### Batch Operations

All functions support batched operations:

```rust
// Batch computation on 2D tensors
let batch_x = randn(&[32, 100]);  // 32 batches of 100 elements
let batch_gamma = gamma(&batch_x)?;
let batch_erf = erf(&batch_x)?;

// Broadcasting
let x = randn(&[10, 1]);
let y = randn(&[1, 20]);
let beta_xy = beta(&x, &y)?;  // Shape: [10, 20]
```

### Complex Number Support

Complex-valued functions live in the `complex` module and are always compiled in (there is no
Cargo feature gate for them). They use `scirs2_core::Complex64`/`Complex32`, not `num_complex`
directly, per the SciRS2 POLICY:

```rust
use torsh_special::complex::{complex_gamma_c64, complex_zeta_c64};
use scirs2_core::Complex64;

let z = tensor![Complex64::new(1.0, 0.5), Complex64::new(2.0, -1.0)];
let gamma_z = complex_gamma_c64(&z)?;
let zeta_z = complex_zeta_c64(&z)?;
```

Complex Bessel, error, Airy, beta, polygamma and incomplete-gamma functions are also available
under the same `complex_*_c64`/`complex_*_c32` naming convention (e.g. `complex_bessel_j_c64`,
`complex_erf_c64`, `complex_airy_ai_c64`).

## Integration with SciRS2

This crate fully leverages scirs2-special for:
- Optimized implementations of all special functions
- Hardware acceleration where available
- Consistent numerical accuracy
- Efficient vectorized operations

## Numerical Considerations

- Functions are implemented with high numerical accuracy
- Appropriate algorithms are chosen for different input ranges
- Special care is taken near singularities and branch points
- Error bounds are documented for each function

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../../LICENSE) for details.
