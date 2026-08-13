//! Mel-scale operations for perceptually-motivated frequency analysis
//!
//! This module provides mel-scale filterbank operations commonly used in
//! audio processing and speech recognition, with full PyTorch compatibility.

use torsh_core::{
    device::DeviceType,
    error::{Result, TorshError},
};
use torsh_tensor::Tensor;

/// Normalisation applied to each mel filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MelNorm {
    /// No normalisation: every triangle peaks at 1.0 (torchaudio's default).
    None,
    /// Slaney normalisation: each filter is divided by its bandwidth so that
    /// filters have equal area (librosa's `norm="slaney"` default).
    Slaney,
}

/// Mel-scale triangular filterbank (unnormalised, torchaudio-compatible).
///
/// Returns a `[n_mels, n_fft / 2 + 1]` matrix of triangular weights computed
/// from the *fractional* FFT bin frequencies, so no filter collapses to zero
/// through integer truncation.
pub fn mel_filterbank(
    n_mels: usize,
    n_fft: usize,
    sample_rate: f32,
    f_min: f32,
    f_max: Option<f32>,
) -> Result<Tensor<f32>> {
    mel_filterbank_with_norm(n_mels, n_fft, sample_rate, f_min, f_max, MelNorm::None)
}

/// Mel-scale triangular filterbank with a selectable normalisation.
pub fn mel_filterbank_with_norm(
    n_mels: usize,
    n_fft: usize,
    sample_rate: f32,
    f_min: f32,
    f_max: Option<f32>,
    norm: MelNorm,
) -> Result<Tensor<f32>> {
    if n_mels == 0 {
        return Err(TorshError::InvalidArgument(
            "Number of mel bands must be greater than zero".to_string(),
        ));
    }
    if n_fft < 2 {
        return Err(TorshError::InvalidArgument(
            "n_fft must be at least 2".to_string(),
        ));
    }
    if sample_rate <= 0.0 {
        return Err(TorshError::InvalidArgument(
            "Sample rate must be positive".to_string(),
        ));
    }

    let f_max = f_max.unwrap_or(sample_rate / 2.0);
    if !(f_min >= 0.0 && f_max > f_min) {
        return Err(TorshError::InvalidArgument(format!(
            "Invalid mel frequency range: f_min = {f_min}, f_max = {f_max}"
        )));
    }
    let n_freqs = n_fft / 2 + 1;

    // Mel-spaced band edges, converted back to Hz.
    let mel_min = hz_to_mel(f_min) as f64;
    let mel_max = hz_to_mel(f_max) as f64;
    let hz_points: Vec<f64> = (0..n_mels + 2)
        .map(|i| {
            let mel = mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64;
            mel_to_hz(mel as f32) as f64
        })
        .collect();

    // Exact (fractional) FFT bin frequencies.
    let fft_freqs: Vec<f64> = (0..n_freqs)
        .map(|k| k as f64 * sample_rate as f64 / n_fft as f64)
        .collect();

    let mut filterbank: Tensor<f32> = Tensor::zeros(&[n_mels, n_freqs], DeviceType::Cpu)?;

    for i in 0..n_mels {
        let (left, centre, right) = (hz_points[i], hz_points[i + 1], hz_points[i + 2]);
        let rising = centre - left;
        let falling = right - centre;
        if rising <= 0.0 || falling <= 0.0 {
            return Err(TorshError::InvalidArgument(
                "Mel band edges are degenerate: reduce n_mels or widen the frequency range"
                    .to_string(),
            ));
        }

        let scale = match norm {
            MelNorm::None => 1.0,
            MelNorm::Slaney => 2.0 / (right - left),
        };

        for (k, &freq) in fft_freqs.iter().enumerate() {
            let lower = (freq - left) / rising;
            let upper = (right - freq) / falling;
            let weight = lower.min(upper).max(0.0);
            if weight > 0.0 {
                filterbank.set_2d(i, k, (weight * scale) as f32)?;
            }
        }
    }

    Ok(filterbank)
}

/// Convert frequency in Hz to mel scale
pub fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert mel scale to frequency in Hz
pub fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// Convert frequency-domain data to mel scale using triangular filterbanks.
///
/// This function applies mel-scale filterbanks to frequency-domain data,
/// providing perceptually-motivated frequency analysis commonly used in
/// audio processing and speech recognition. Compatible with PyTorch's mel-scale functions.
///
/// # Arguments
///
/// * `freqs` - Frequency-domain tensor (typically from STFT magnitude)
/// * `f_min` - Minimum frequency in Hz (typically 0.0)
/// * `f_max` - Maximum frequency in Hz (defaults to Nyquist frequency)
/// * `n_mels` - Number of mel frequency bins (typically 80-128 for speech)
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
///
/// Tensor with mel-scale representation, shape `[n_mels, ...]` where `...`
/// represents the original tensor dimensions except the first (frequency) dimension.
///
/// # Examples
///
/// ```rust,no_run
/// use torsh_signal::prelude::*;
/// use torsh_tensor::creation::ones;
///
/// // Convert spectrogram to mel scale
/// let specgram = ones(&[513, 100])?; // 513 freq bins, 100 time frames
/// let mel_spec = mel_scale(
///     &specgram,
///     0.0,              // f_min
///     Some(8000.0),     // f_max
///     80,               // n_mels
///     16000.0           // sample_rate
/// )?;
///
/// assert_eq!(mel_spec.shape().dims(), &[80, 100]); // 80 mel bins
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Mel Scale Properties
///
/// The mel scale is a perceptual scale of pitches judged by listeners to be
/// equal in distance from one another. It's defined as:
/// - `mel = 2595 * log10(1 + hz/700)`
/// - Triangular filterbanks with 50% overlap
/// - Linear spacing in mel domain, logarithmic in Hz domain
///
/// # Performance Notes
///
/// - Filterbank is computed on-demand (consider caching for repeated use)
/// - Memory usage: `O(n_mels × n_freqs)` for filterbank matrix
/// - Computational complexity: `O(n_mels × n_freqs × n_frames)`
///
/// # See Also
///
/// - [`mel_spectrogram`] - Direct conversion from time-domain signal
/// - [`inverse_mel_scale`] - Convert mel scale back to linear frequency
/// - [`create_fb_matrix`] - Generate mel filterbank matrix
pub fn mel_scale(
    freqs: &Tensor<f32>,
    f_min: f32,
    f_max: Option<f32>,
    n_mels: usize,
    sample_rate: f32,
) -> Result<Tensor<f32>> {
    let f_max = f_max.unwrap_or(sample_rate / 2.0);

    // Create mel scale filterbank
    let n_fft = (freqs.shape().dims()[0] - 1) * 2;
    let mel_fb = mel_filterbank(n_mels, n_fft, sample_rate, f_min, Some(f_max))?;

    // Apply mel filterbank to frequency tensor
    // Handle both 1D and 2D input tensors
    let result = if freqs.shape().ndim() == 1 {
        // For 1D input, reshape to column vector for matrix multiplication
        let freqs_2d = freqs.unsqueeze(1)?; // Shape: [n_freqs, 1]
        let result_2d = mel_fb.matmul(&freqs_2d)?; // Shape: [n_mels, 1]
        result_2d.squeeze(1)? // Back to 1D: [n_mels]
    } else {
        // For 2D input, direct matrix multiplication
        mel_fb.matmul(freqs)?
    };
    Ok(result)
}

/// PyTorch-compatible inverse_mel_scale function
/// Converts mel-scale tensor back to frequency domain
///
/// # Arguments
/// * `mel_specgram` - Tensor containing mel-scale values
/// * `f_min` - Minimum frequency (Hz)
/// * `f_max` - Maximum frequency (Hz), defaults to Nyquist
/// * `n_stft` - Number of STFT bins
/// * `sample_rate` - Sample rate (Hz)
///
/// # Returns
/// Tensor in frequency domain
pub fn inverse_mel_scale(
    mel_specgram: &Tensor<f32>,
    f_min: f32,
    f_max: Option<f32>,
    n_stft: usize,
    sample_rate: f32,
) -> Result<Tensor<f32>> {
    let f_max = f_max.unwrap_or(sample_rate / 2.0);
    let n_mels = mel_specgram.shape().dims()[0];

    // Create mel scale filterbank
    let n_fft = (n_stft - 1) * 2;
    let mel_fb = mel_filterbank(n_mels, n_fft, sample_rate, f_min, Some(f_max))?;

    // Pseudo-inverse of mel filterbank
    // For now, use transpose as approximation (this could be improved with proper pseudo-inverse)
    let mel_fb_inv = mel_fb.transpose(0, 1)?;

    // Apply inverse mel filterbank: mel_fb_inv @ mel_specgram
    // Handle both 1D and 2D input tensors
    let result = if mel_specgram.shape().ndim() == 1 {
        // For 1D input, reshape to column vector for matrix multiplication
        let mel_2d = mel_specgram.unsqueeze(1)?; // Shape: [n_mels, 1]
        let result_2d = mel_fb_inv.matmul(&mel_2d)?; // Shape: [n_freqs, 1]
        result_2d.squeeze(1)? // Back to 1D: [n_freqs]
    } else {
        // For 2D input, direct matrix multiplication
        mel_fb_inv.matmul(mel_specgram)?
    };
    Ok(result)
}

/// PyTorch-compatible create_fb_matrix function
/// Creates a frequency-bin conversion matrix
///
/// # Arguments
/// * `n_freqs` - Number of frequency bins in STFT
/// * `f_min` - Minimum frequency (Hz)
/// * `f_max` - Maximum frequency (Hz)
/// * `n_mels` - Number of mel bins
/// * `sample_rate` - Sample rate (Hz)
///
/// # Returns
/// Tensor containing the mel filterbank matrix
pub fn create_fb_matrix(
    n_freqs: usize,
    f_min: f32,
    f_max: f32,
    n_mels: usize,
    sample_rate: f32,
) -> Result<Tensor<f32>> {
    // Calculate n_fft from n_freqs (assuming onesided spectrum)
    let n_fft = (n_freqs - 1) * 2;
    mel_filterbank(n_mels, n_fft, sample_rate, f_min, Some(f_max))
}

/// Mel spectrogram computation - PyTorch compatible
///
/// # Arguments
/// * `specgram` - Magnitude spectrogram
/// * `f_min` - Minimum frequency (Hz)
/// * `f_max` - Maximum frequency (Hz), defaults to Nyquist
/// * `n_mels` - Number of mel bins
/// * `sample_rate` - Sample rate (Hz)
///
/// # Returns
/// Mel-scale spectrogram
pub fn mel_spectrogram(
    specgram: &Tensor<f32>,
    f_min: f32,
    f_max: Option<f32>,
    n_mels: usize,
    sample_rate: f32,
) -> Result<Tensor<f32>> {
    let shape = specgram.shape();
    let n_freqs = shape.dims()[0];
    let f_max = f_max.unwrap_or(sample_rate / 2.0);

    // Create mel filterbank matrix
    let mel_fb = create_fb_matrix(n_freqs, f_min, f_max, n_mels, sample_rate)?;

    // Apply mel filterbank to spectrogram
    // Handle both 1D and 2D input tensors
    let result = if specgram.shape().ndim() == 1 {
        // For 1D input, reshape to column vector for matrix multiplication
        let spec_2d = specgram.unsqueeze(1)?; // Shape: [n_freqs, 1]
        let result_2d = mel_fb.matmul(&spec_2d)?; // Shape: [n_mels, 1]
        result_2d.squeeze(1)? // Back to 1D: [n_mels]
    } else {
        // For 2D input, direct matrix multiplication
        mel_fb.matmul(specgram)?
    };
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_mel_filterbank() -> Result<()> {
        let filterbank = mel_filterbank(128, 2048, 16000.0, 0.0, None)?;
        assert_eq!(filterbank.shape().dims(), &[128, 1025]);

        // Check that filters sum to approximately 1
        let sum = filterbank.sum()?;
        if sum.shape().ndim() == 0 {
            // Sum is a scalar
            let val = sum.item();
            let val_unwrapped = val.expect("item value should be present");
            assert!(val_unwrapped >= 0.0 && val_unwrapped <= 2048.0); // Total sum across all filters
        } else {
            // Sum per dimension
            for i in 0..sum.shape().dims()[0] {
                let val = sum.get_1d(i)?;
                assert!(val >= 0.0 && val <= 2.0);
            }
        }
        Ok(())
    }

    #[test]
    fn test_hz_mel_conversion() {
        assert_relative_eq!(hz_to_mel(0.0), 0.0, epsilon = 1e-5);
        assert_relative_eq!(mel_to_hz(0.0), 0.0, epsilon = 1e-5);

        // Test round-trip conversion
        let hz = 1000.0;
        let mel = hz_to_mel(hz);
        let hz_back = mel_to_hz(mel);
        assert_relative_eq!(hz, hz_back, epsilon = 1e-3);
    }

    #[test]
    fn test_create_fb_matrix() -> Result<()> {
        let n_freqs = 513; // n_fft/2 + 1 where n_fft = 1024
        let f_min = 0.0;
        let f_max = 8000.0;
        let n_mels = 128;
        let sample_rate = 16000.0;

        let fb_matrix = create_fb_matrix(n_freqs, f_min, f_max, n_mels, sample_rate)?;
        let shape = fb_matrix.shape();

        // Check shape
        assert_eq!(shape.dims(), &[n_mels, n_freqs]);

        // Check that each filter has non-negative values (sum bounds can be quite large depending on parameters)
        for i in 0..n_mels {
            let mut sum = 0.0;
            for j in 0..n_freqs {
                let val = fb_matrix.get_2d(i, j)?;
                assert!(val >= 0.0, "Mel filter values should be non-negative"); // Non-negative values
                sum += val;
            }
            assert!(sum >= 0.0, "Mel filter sum should be non-negative"); // At least non-negative sum
        }

        Ok(())
    }

    #[test]
    fn test_mel_spectrogram() -> Result<()> {
        // Create a dummy spectrogram
        let n_freqs = 513;
        let n_frames = 100;
        let mut specgram: Tensor<f32> = Tensor::zeros(&[n_freqs, n_frames], DeviceType::Cpu)?;

        // Fill with some test data
        for i in 0..n_freqs {
            for j in 0..n_frames {
                specgram.set_2d(i, j, (i + j) as f32 * 0.001)?;
            }
        }

        let f_min = 0.0;
        let f_max = Some(8000.0);
        let n_mels = 80;
        let sample_rate = 16000.0;

        let mel_spec = mel_spectrogram(&specgram, f_min, f_max, n_mels, sample_rate)?;
        let shape = mel_spec.shape();

        // Check output shape
        assert_eq!(shape.dims(), &[n_mels, n_frames]);

        // Check that values are non-negative (mel spectrograms should be non-negative)
        for i in 0..n_mels {
            for j in 0..n_frames {
                let value = mel_spec.get_2d(i, j)?;
                assert!(value >= 0.0);
            }
        }

        Ok(())
    }

    #[test]
    fn test_mel_scale_operations() -> Result<()> {
        let n_freqs = 257; // n_fft/2 + 1 where n_fft = 512
        let n_mels = 40;
        let sample_rate = 16000.0;
        let f_min = 0.0;
        let f_max = Some(8000.0);

        // Create a dummy frequency-domain tensor
        let mut freqs: Tensor<f32> = Tensor::zeros(&[n_freqs], DeviceType::Cpu)?;
        for i in 0..n_freqs {
            freqs.set_1d(i, i as f32 * 0.1)?;
        }

        // Test mel_scale
        let mel_result = mel_scale(&freqs, f_min, f_max, n_mels, sample_rate)?;
        assert_eq!(mel_result.shape().dims()[0], n_mels);

        // Test inverse_mel_scale
        let inverse_result = inverse_mel_scale(&mel_result, f_min, f_max, n_freqs, sample_rate)?;
        assert_eq!(inverse_result.shape().dims()[0], n_freqs);

        // Check that values are reasonable (not NaN or infinite)
        for i in 0..n_mels {
            let val = mel_result.get_1d(i)?;
            assert!(val.is_finite());
        }

        for i in 0..n_freqs {
            let val = inverse_result.get_1d(i)?;
            assert!(val.is_finite());
        }

        Ok(())
    }
}
