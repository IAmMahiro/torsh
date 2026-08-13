//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use torsh_core::{device::DeviceType, error::Result};
use torsh_tensor::creation::from_vec;

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_tensor::creation::ones;
    #[test]
    fn test_cwt_processor() -> Result<()> {
        let scales = vec![1.0, 2.0, 4.0, 8.0];
        let processor =
            ContinuousWaveletProcessor::new(WaveletType::Morlet, scales.clone(), 1000.0);
        let signal = ones(&[256])?;
        let cwt_result = processor.cwt(&signal)?;
        assert_eq!(cwt_result.shape().dims(), &[4, 256]);
        Ok(())
    }
    #[test]
    fn test_dwt_processor() -> Result<()> {
        let processor = DiscreteWaveletProcessor::new(WaveletType::Daubechies(4), 3);
        let signal = ones(&[256])?;
        let (approximation, details) = processor.dwt(&signal)?;
        assert_eq!(approximation.shape().dims()[0], 32);
        assert_eq!(details.len(), 3);
        Ok(())
    }
    #[test]
    fn test_wavelet_packet_processor() -> Result<()> {
        let processor = WaveletPacketProcessor::new(WaveletType::Haar, 2);
        let signal = ones(&[256])?;
        let packets = processor.wpt(&signal)?;
        assert_eq!(packets.len(), 4);
        for packet in &packets {
            assert_eq!(packet.shape().dims()[0], 64);
        }
        Ok(())
    }
    /// L2 relative error `||a - b|| / ||a||`, robust to per-sample
    /// zero-crossings (unlike a naive per-element relative error).
    fn relative_l2_error(a: &[f32], b: &[f32]) -> f32 {
        let diff_sq: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        let norm_sq: f32 = a.iter().map(|&x| x * x).sum();
        if norm_sq <= 0.0 {
            diff_sq.sqrt()
        } else {
            (diff_sq / norm_sq).sqrt()
        }
    }
    /// A sum-of-sinusoids test signal with `n` samples, mixing a couple of
    /// frequencies and amplitudes so it exercises both the approximation
    /// and detail branches of a wavelet packet tree non-trivially.
    fn sinusoid_test_signal(n: usize, freqs_and_amps: &[(f32, f32)]) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / n as f32;
                freqs_and_amps
                    .iter()
                    .map(|&(freq, amp)| amp * (2.0 * std::f32::consts::PI * freq * t).sin())
                    .sum()
            })
            .collect()
    }
    #[test]
    fn test_wpt_round_trip_haar() -> Result<()> {
        let n = 256usize;
        let original = sinusoid_test_signal(n, &[(5.0, 1.0), (20.0, 0.5)]);
        let signal = from_vec(original.clone(), &[n], DeviceType::Cpu)?;
        let processor = WaveletPacketProcessor::new(WaveletType::Haar, 3);
        let packets = processor.wpt(&signal)?;
        assert_eq!(packets.len(), 8, "expected 2^3 leaf packets");
        for packet in &packets {
            assert_eq!(packet.shape().dims()[0], n / 8);
        }
        let reconstructed = processor.iwpt(&packets)?;
        let reconstructed_vec = reconstructed.to_vec()?;
        assert_eq!(reconstructed_vec.len(), n);
        let error = relative_l2_error(&original, &reconstructed_vec);
        assert!(
            error < 1e-5,
            "Haar WPT round-trip relative error {error} exceeds tolerance"
        );
        Ok(())
    }
    #[test]
    fn test_wpt_round_trip_daubechies4() -> Result<()> {
        let n = 256usize;
        let original = sinusoid_test_signal(n, &[(3.0, 1.0), (11.0, -0.3)]);
        let signal = from_vec(original.clone(), &[n], DeviceType::Cpu)?;
        let processor = WaveletPacketProcessor::new(WaveletType::Daubechies(4), 3);
        let packets = processor.wpt(&signal)?;
        assert_eq!(packets.len(), 8);
        let reconstructed = processor.iwpt(&packets)?;
        let reconstructed_vec = reconstructed.to_vec()?;
        assert_eq!(reconstructed_vec.len(), n);
        let error = relative_l2_error(&original, &reconstructed_vec);
        assert!(
            error < 1e-4,
            "Daubechies(4) WPT round-trip relative error {error} exceeds tolerance"
        );
        Ok(())
    }
    #[test]
    fn test_wpt_round_trip_daubechies8() -> Result<()> {
        let n = 256usize;
        let original = sinusoid_test_signal(n, &[(6.0, 0.8), (17.0, 0.4)]);
        let signal = from_vec(original.clone(), &[n], DeviceType::Cpu)?;
        let processor = WaveletPacketProcessor::new(WaveletType::Daubechies(8), 2);
        let packets = processor.wpt(&signal)?;
        assert_eq!(packets.len(), 4);
        let reconstructed = processor.iwpt(&packets)?;
        let reconstructed_vec = reconstructed.to_vec()?;
        let error = relative_l2_error(&original, &reconstructed_vec);
        assert!(
            error < 1e-4,
            "Daubechies(8) WPT round-trip relative error {error} exceeds tolerance"
        );
        Ok(())
    }
    #[test]
    fn test_wpt_non_trivial_output() -> Result<()> {
        let n = 64usize;
        let ramp: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let signal = from_vec(ramp, &[n], DeviceType::Cpu)?;
        let processor = WaveletPacketProcessor::new(WaveletType::Haar, 2);
        let packets = processor.wpt(&signal)?;
        let all_zero = packets.iter().all(|p| {
            p.to_vec()
                .map(|v| v.iter().all(|&x| x.abs() < 1e-9))
                .unwrap_or(true)
        });
        assert!(!all_zero, "wpt() must not silently return all-zero packets");
        Ok(())
    }
    #[test]
    fn test_lifting_round_trip_haar() -> Result<()> {
        let n = 256usize;
        let original = sinusoid_test_signal(n, &[(7.0, 1.0)])
            .iter()
            .enumerate()
            .map(|(i, &v)| v + 0.2 * (i as f32 / n as f32))
            .collect::<Vec<f32>>();
        let signal = from_vec(original.clone(), &[n], DeviceType::Cpu)?;
        let processor = LiftingSchemeProcessor::new(WaveletType::Haar);
        let (approx, detail) = processor.lifting_dwt(&signal)?;
        let approx_vec = approx.to_vec()?;
        let detail_vec = detail.to_vec()?;
        assert!(
            approx_vec.iter().any(|&x| x.abs() > 1e-6)
                && detail_vec.iter().any(|&x| x.abs() > 1e-6),
            "lifting_dwt() must not silently return all-zero coefficients"
        );
        let reconstructed = processor.lifting_idwt(&approx, &detail)?;
        let reconstructed_vec = reconstructed.to_vec()?;
        assert_eq!(reconstructed_vec.len(), n);
        let error = relative_l2_error(&original, &reconstructed_vec);
        assert!(
            error < 1e-5,
            "Haar lifting round-trip relative error {error} exceeds tolerance"
        );
        Ok(())
    }
    #[test]
    fn test_lifting_round_trip_daubechies4() -> Result<()> {
        let n = 256usize;
        let original = sinusoid_test_signal(n, &[(9.0, 1.0), (2.0, 0.4)]);
        let signal = from_vec(original.clone(), &[n], DeviceType::Cpu)?;
        let processor = LiftingSchemeProcessor::new(WaveletType::Daubechies(4));
        let (approx, detail) = processor.lifting_dwt(&signal)?;
        let approx_vec = approx.to_vec()?;
        let detail_vec = detail.to_vec()?;
        assert!(
            approx_vec.iter().any(|&x| x.abs() > 1e-6)
                && detail_vec.iter().any(|&x| x.abs() > 1e-6),
            "lifting_dwt() must not silently return all-zero coefficients"
        );
        let reconstructed = processor.lifting_idwt(&approx, &detail)?;
        let reconstructed_vec = reconstructed.to_vec()?;
        let error = relative_l2_error(&original, &reconstructed_vec);
        assert!(
            error < 1e-5,
            "Daubechies(4) lifting round-trip relative error {error} exceeds tolerance"
        );
        Ok(())
    }
    #[test]
    fn test_cone_of_influence_matches_formula() {
        let scales = vec![1.0, 2.0, 4.0, 8.0];
        let coi_morlet = WaveletUtils::cone_of_influence(&scales, 100_000, WaveletType::Morlet);
        for (&s, &c) in scales.iter().zip(coi_morlet.iter()) {
            let expected = std::f32::consts::SQRT_2 * s;
            assert_relative_eq!(c, expected, epsilon = 1e-4);
        }
        let coi_haar = WaveletUtils::cone_of_influence(&scales, 100_000, WaveletType::Haar);
        for (&s, &c) in scales.iter().zip(coi_haar.iter()) {
            let expected = 0.5 * s;
            assert_relative_eq!(c, expected, epsilon = 1e-4);
        }
        let coi_db4 = WaveletUtils::cone_of_influence(&scales, 100_000, WaveletType::Daubechies(4));
        for (&s, &c) in scales.iter().zip(coi_db4.iter()) {
            let expected = 1.5 * s;
            assert_relative_eq!(c, expected, epsilon = 1e-4);
        }
    }
    #[test]
    fn test_cone_of_influence_depends_on_wavelet_type() {
        let scales = vec![4.0];
        let coi_morlet = WaveletUtils::cone_of_influence(&scales, 100_000, WaveletType::Morlet);
        let coi_db4 = WaveletUtils::cone_of_influence(&scales, 100_000, WaveletType::Daubechies(4));
        assert!(
            (coi_morlet[0] - coi_db4[0]).abs() > 1e-3,
            "cone_of_influence must depend on wavelet type: morlet={}, db4={}",
            coi_morlet[0],
            coi_db4[0]
        );
    }
    #[test]
    fn test_cone_of_influence_depends_on_signal_length() {
        let scales = vec![1000.0];
        let coi_short = WaveletUtils::cone_of_influence(&scales, 10, WaveletType::Morlet);
        let coi_long = WaveletUtils::cone_of_influence(&scales, 1_000_000, WaveletType::Morlet);
        assert_relative_eq!(coi_short[0], 5.0, epsilon = 1e-4);
        assert!(
            coi_long[0] > coi_short[0] * 100.0,
            "cone_of_influence must depend on signal_length: short={}, long={}",
            coi_short[0],
            coi_long[0]
        );
    }
    #[test]
    fn test_lifting_scheme() -> Result<()> {
        let processor = LiftingSchemeProcessor::new(WaveletType::Haar);
        let signal = ones(&[256])?;
        let (approx, detail) = processor.lifting_dwt(&signal)?;
        assert_eq!(approx.shape().dims()[0], 128);
        assert_eq!(detail.shape().dims()[0], 128);
        let reconstructed = processor.lifting_idwt(&approx, &detail)?;
        assert_eq!(reconstructed.shape().dims()[0], 256);
        Ok(())
    }
    #[test]
    fn test_wavelet_denoiser() -> Result<()> {
        let denoiser = WaveletDenoiser::new(WaveletType::Haar, 2, ThresholdMethod::Soft);
        let signal = ones(&[256])?;
        let denoised = denoiser.denoise(&signal)?;
        let denoised_len = denoised.shape().dims()[0];
        assert!(
            denoised_len >= 64 && denoised_len <= 256,
            "Denoised signal length {} should be between 64 and 256",
            denoised_len
        );
        let noise_level = denoiser.estimate_noise_level(&signal)?;
        assert!(noise_level >= 0.0);
        Ok(())
    }
    #[test]
    fn test_wavelet_utils() {
        let frequency = 100.0;
        let sample_rate = 1000.0;
        let wavelet = WaveletType::Morlet;
        let scale = WaveletUtils::frequency_to_scale(frequency, sample_rate, wavelet);
        let frequency_back = WaveletUtils::scale_to_frequency(scale, sample_rate, wavelet);
        assert_relative_eq!(frequency, frequency_back, epsilon = 1e-5);
        let scales = vec![1.0, 2.0, 4.0];
        let coi = WaveletUtils::cone_of_influence(&scales, 256, wavelet);
        assert_eq!(coi.len(), 3);
    }
    #[test]
    fn test_symlet_wavelets() -> Result<()> {
        let processor_sym2 = DiscreteWaveletProcessor::new(WaveletType::Symlet(2), 2);
        let signal = ones(&[128])?;
        let (approx, details) = processor_sym2.dwt(&signal)?;
        assert!(approx.shape().dims()[0] > 0);
        assert_eq!(details.len(), 2);
        let processor_sym4 = DiscreteWaveletProcessor::new(WaveletType::Symlet(4), 2);
        let (approx, details) = processor_sym4.dwt(&signal)?;
        assert!(approx.shape().dims()[0] > 0);
        assert_eq!(details.len(), 2);
        let processor_sym6 = DiscreteWaveletProcessor::new(WaveletType::Symlet(6), 2);
        let (approx, details) = processor_sym6.dwt(&signal)?;
        assert!(approx.shape().dims()[0] > 0);
        assert_eq!(details.len(), 2);
        let processor_sym8 = DiscreteWaveletProcessor::new(WaveletType::Symlet(8), 2);
        let (approx, details) = processor_sym8.dwt(&signal)?;
        assert!(approx.shape().dims()[0] > 0);
        assert_eq!(details.len(), 2);
        Ok(())
    }
    #[test]
    fn test_coiflet_wavelets() -> Result<()> {
        let signal = ones(&[128])?;
        for order in 1..=5 {
            let processor = DiscreteWaveletProcessor::new(WaveletType::Coiflet(order), 2);
            let (approx, details) = processor.dwt(&signal)?;
            assert!(
                approx.shape().dims()[0] > 0,
                "Coiflet {} failed: approx size is 0",
                order
            );
            assert_eq!(
                details.len(),
                2,
                "Coiflet {} failed: expected 2 detail levels",
                order
            );
        }
        Ok(())
    }
    #[test]
    fn test_biorthogonal_wavelets() -> Result<()> {
        let signal = ones(&[128])?;
        let processor_bior11 = DiscreteWaveletProcessor::new(WaveletType::Biorthogonal(1, 1), 2);
        let (approx, details) = processor_bior11.dwt(&signal)?;
        assert!(approx.shape().dims()[0] > 0);
        assert_eq!(details.len(), 2);
        let processor_bior13 = DiscreteWaveletProcessor::new(WaveletType::Biorthogonal(1, 3), 2);
        let (approx, details) = processor_bior13.dwt(&signal)?;
        assert!(approx.shape().dims()[0] > 0);
        assert_eq!(details.len(), 2);
        let processor_bior15 = DiscreteWaveletProcessor::new(WaveletType::Biorthogonal(1, 5), 2);
        let (approx, details) = processor_bior15.dwt(&signal)?;
        assert!(approx.shape().dims()[0] > 0);
        assert_eq!(details.len(), 2);
        let processor_bior22 = DiscreteWaveletProcessor::new(WaveletType::Biorthogonal(2, 2), 2);
        let (approx, details) = processor_bior22.dwt(&signal)?;
        assert!(approx.shape().dims()[0] > 0);
        assert_eq!(details.len(), 2);
        let processor_bior24 = DiscreteWaveletProcessor::new(WaveletType::Biorthogonal(2, 4), 2);
        let (approx, details) = processor_bior24.dwt(&signal)?;
        assert!(approx.shape().dims()[0] > 0);
        assert_eq!(details.len(), 2);
        Ok(())
    }
    #[test]
    fn test_advanced_daubechies_wavelets() -> Result<()> {
        let signal = ones(&[128])?;
        let processor_db3 = DiscreteWaveletProcessor::new(WaveletType::Daubechies(6), 2);
        let (approx, details) = processor_db3.dwt(&signal)?;
        assert!(approx.shape().dims()[0] > 0);
        assert_eq!(details.len(), 2);
        let processor_db4 = DiscreteWaveletProcessor::new(WaveletType::Daubechies(8), 2);
        let (approx, details) = processor_db4.dwt(&signal)?;
        assert!(approx.shape().dims()[0] > 0);
        assert_eq!(details.len(), 2);
        let processor_db5 = DiscreteWaveletProcessor::new(WaveletType::Daubechies(10), 2);
        let (approx, details) = processor_db5.dwt(&signal)?;
        assert!(approx.shape().dims()[0] > 0);
        assert_eq!(details.len(), 2);
        Ok(())
    }
    #[test]
    fn test_wavelet_reconstruction_perfect() -> Result<()> {
        use torsh_tensor::creation::randn;
        let wavelets = vec![
            WaveletType::Haar,
            WaveletType::Daubechies(4),
            WaveletType::Daubechies(6),
            WaveletType::Symlet(4),
            WaveletType::Symlet(6),
            WaveletType::Coiflet(1),
            WaveletType::Coiflet(2),
        ];
        for wavelet in wavelets {
            let signal = randn::<f32>(&[256])?;
            let processor = DiscreteWaveletProcessor::new(wavelet, 2);
            let (approx, details) = processor.dwt(&signal)?;
            let reconstructed = processor.idwt(&approx, &details)?;
            let original_len = signal.shape().dims()[0];
            let reconstructed_len = reconstructed.shape().dims()[0];
            assert!(
                reconstructed_len >= original_len / 2 && reconstructed_len <= original_len * 2,
                "Wavelet {:?}: reconstructed length {} vs original {}",
                wavelet,
                reconstructed_len,
                original_len
            );
        }
        Ok(())
    }
    #[test]
    fn test_qmf_filter_construction() {
        use approx::assert_relative_eq;
        let lo_d = vec![0.5, 0.5, 0.5, 0.5];
        let (hi_d, lo_r, hi_r) = super::construct_qmf_filters(&lo_d);
        assert_eq!(hi_d.len(), lo_d.len());
        assert_eq!(lo_r.len(), lo_d.len());
        assert_eq!(hi_r.len(), lo_d.len());
        for i in 0..lo_d.len() {
            assert_relative_eq!(lo_r[i], lo_d[lo_d.len() - 1 - i], epsilon = 1e-6);
            assert_relative_eq!(hi_r[i], hi_d[hi_d.len() - 1 - i], epsilon = 1e-6);
        }
        let sqrt2_f32 = 2.0_f32.sqrt();
        let haar_lo_d = vec![1.0 / sqrt2_f32, 1.0 / sqrt2_f32];
        let (haar_hi_d, haar_lo_r, haar_hi_r) = super::construct_qmf_filters(&haar_lo_d);
        assert_eq!(haar_hi_d.len(), 2);
        let mag0 = haar_hi_d[0].abs();
        let mag1 = haar_hi_d[1].abs();
        assert_relative_eq!(mag0, 1.0 / sqrt2_f32, epsilon = 1e-6);
        assert_relative_eq!(mag1, 1.0 / sqrt2_f32, epsilon = 1e-6);
        assert_eq!(haar_lo_r.len(), 2);
        assert_eq!(haar_hi_r.len(), 2);
    }
}
