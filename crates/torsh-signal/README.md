# torsh-signal

Signal processing operations for the ToRSh deep learning framework.

**Status**: Stable — DSP, filtering, windowing, resampling, audio features, and wavelet transforms
(including the Wavelet Packet Transform and a lifting-scheme DWT/IDWT, fixed 2026-07) are
implemented and tested. **Tests**: 138/138 passing, 3 skipped (`cargo nextest run --all-features`).

## Overview

This crate provides PyTorch-compatible signal processing functionality built on top of the **SciRS2 ecosystem** for high-performance scientific computing. It leverages scirs2-signal and scirs2-fft to deliver state-of-the-art signal processing operations with excellent performance and PyTorch compatibility.

## Features

### Core Signal Processing (Powered by SciRS2)
- **Advanced Spectral Operations**: STFT, ISTFT, spectrograms with scirs2-fft acceleration
- **Professional Window Functions**: Hamming, Hann, Blackman, Kaiser, Gaussian powered by scirs2-signal
- **High-Performance Filtering**: Convolution, correlation with scirs2-signal optimization
- **Wavelet Transforms**: Continuous (Morlet) and discrete wavelet transforms, a full recursive
  Wavelet Packet Transform (WPT/IWPT), a Sweldens (1996) lifting-scheme DWT/IDWT, and wavelet
  denoising, with a Torrence & Compo e-folding-time cone-of-influence calculation
- **SciRS2 Integration**: Full access to scirs2-signal's advanced signal processing algorithms

### PyTorch Compatibility
- **Drop-in Replacement**: Compatible API for torch.signal operations
- **Seamless Migration**: Easy transition from PyTorch signal processing
- **Performance Benefits**: Rust-native performance with Python-like API

## Modules

### ToRSh Signal Processing Modules
- `spectral`: STFT, ISTFT, spectrograms, mel-scale filtering (scirs2-fft powered)
- `windows`: Various window functions for signal processing (scirs2-signal powered)
- `filters`: Convolution and correlation operations (scirs2-signal powered)
- `wavelets`: CWT, DWT/IDWT, Wavelet Packet Transform (WPT/IWPT), lifting-scheme DWT/IDWT, and
  wavelet denoising — self-contained, does not depend on scirs2-signal

### SciRS2 Integration
- **scirs2-signal**: Advanced signal processing algorithms and optimizations
- **scirs2-fft**: High-performance FFT operations with SIMD acceleration
- **scirs2-core**: Scientific computing primitives and memory management

## Usage

### Basic Signal Processing with SciRS2 Integration

```rust
use torsh_signal::prelude::*;
use torsh_tensor::Tensor;

// Create a signal
let signal: Tensor<f32> = Tensor::randn(&[1000])?;

// Apply Hann window
let window = hann_window(256, false)?;
let windowed = signal.slice(0, 0, 256)? * &window;

// Compute STFT (takes a StftParams config struct, not positional arguments)
let stft_result = stft(
    &signal,
    StftParams {
        n_fft: 256,
        hop_length: Some(64),
        win_length: Some(256),
        window: Some(Window::Hann),
        center: true,
        normalized: false,
        onesided: true,
        return_complex: true,
    },
)?;

// Compute spectrogram (magnitude spectrogram via STFT under the hood)
let spec = spectrogram(
    &signal,
    512,         // n_fft
    Some(128),   // hop_length
    Some(512),   // win_length
    Some(Window::Hann),
    true,        // center
    "reflect",   // pad_mode (currently unused)
    false,       // normalized
    true,        // onesided
    Some(2.0),   // power (None = complex, Some(1.0) = magnitude, Some(2.0) = power)
)?;
```

### Wavelet Transforms

```rust
use torsh_signal::prelude::*;

let signal: Tensor<f32> = Tensor::randn(&[1024])?;

// Standard multi-level DWT / IDWT
let dwt = DiscreteWaveletProcessor::new(WaveletType::Daubechies(4), 3);
let (approximation, details) = dwt.dwt(&signal)?;
let reconstructed = dwt.idwt(&approximation, &details)?;

// Wavelet Packet Transform (WPT) — decomposes both approximation and detail
// branches at every level, unlike the standard DWT above
let wpt_processor = WaveletPacketProcessor::new(WaveletType::Haar, 3);
let packets = wpt_processor.wpt(&signal)?; // 2^3 = 8 leaf packets
let reconstructed_wpt = wpt_processor.iwpt(&packets)?;

// Sweldens (1996) lifting-scheme DWT / IDWT
let lifting = LiftingSchemeProcessor::new(WaveletType::Daubechies(4));
let (lifted_approx, lifted_detail) = lifting.lifting_dwt(&signal)?;
let reconstructed_lifting = lifting.lifting_idwt(&lifted_approx, &lifted_detail)?;

// Continuous Wavelet Transform (Morlet) + cone of influence
let scales = vec![1.0, 2.0, 4.0, 8.0, 16.0];
let cwt = ContinuousWaveletProcessor::new(WaveletType::Morlet, scales.clone(), 1000.0);
let cwt_result = cwt.cwt(&signal)?;
let scalogram = cwt.scalogram(&signal)?;
// e-folding-time based edge-effect boundary (Torrence & Compo, 1998)
let coi = WaveletUtils::cone_of_influence(&scales, signal.shape().dims()[0], WaveletType::Morlet);
```

## Dependencies

### Core ToRSh Dependencies
- `torsh-core`: Core types and device abstraction
- `torsh-tensor`: Tensor operations and storage
- `torsh-functional`: Functional API components
- `torsh-linalg`: Linear algebra operations

### SciRS2 Ecosystem Dependencies
- **`scirs2-signal`**: Advanced signal processing algorithms and optimizations
- **`scirs2-fft`**: High-performance FFT operations with SIMD acceleration
- **`scirs2-core`**: Scientific computing primitives and memory management

### Supporting Libraries
- `thiserror`: Error handling
- `approx`: Approximate floating-point comparisons (tests/benchmarks)

Per the SciRS2 POLICY, this crate does not depend directly on `num-complex`, `num-traits`, or
`ndarray` — numeric traits and complex/array types are accessed exclusively through `scirs2-core`.

## Performance

torsh-signal delivers exceptional performance through the SciRS2 ecosystem:

### SciRS2-Powered Optimizations
- **SIMD-accelerated operations** through scirs2-signal and scirs2-core
- **High-performance FFT** via scirs2-fft with CPU-specific optimizations
- **Memory-efficient processing** with scirs2-core's advanced memory management
- **Vectorized operations** for window functions and filtering

### Advanced Features
- **Automatic optimization selection** based on input size and hardware
- **Parallel processing** capabilities for large-scale signal processing
- **Zero-copy operations** where possible to minimize memory overhead
- **Complex number optimizations** with proper magnitude calculations

## Compatibility

Designed to be a drop-in replacement for PyTorch's signal processing operations:
- `torch.stft` → `torsh_signal::stft`
- `torch.istft` → `torsh_signal::istft`
- `torch.spectrogram` → `torsh_signal::spectrogram`
- Window functions match PyTorch's implementation

## Examples

See the `examples/` directory for comprehensive usage examples and PyTorch compatibility demonstrations.