//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use torsh_core::{
    device::DeviceType,
    error::{Result, TorshError},
};
use torsh_tensor::{creation::from_vec, Tensor};

use super::core::{get_wavelet_filters, WaveletType};

/// Lifting Scheme processor
pub struct LiftingSchemeProcessor {
    pub wavelet: WaveletType,
}
impl LiftingSchemeProcessor {
    pub fn new(wavelet: WaveletType) -> Self {
        Self { wavelet }
    }
    /// Compute the forward lifting-scheme transform (Sweldens 1996).
    ///
    /// Splits the signal into even/odd polyphase components
    /// (`even[i] = s[2i]`, `odd[i] = s[2i+1]`) and applies a short,
    /// in-place sequence of predict (P) and update (U) steps. Every lifting
    /// step is individually invertible by construction (each forward step
    /// is an add, undone in reverse order by a subtract -- see
    /// [`Self::lifting_idwt`]), so the round trip is exact up to
    /// floating-point rounding regardless of the P/U coefficients used.
    ///
    /// The coefficients are selected from `self.wavelet`:
    /// - [`WaveletType::Daubechies`]`(4)` (the classic 4-tap / 2-vanishing-
    ///   moment Daubechies filter, matching `get_daubechies_filters`'s
    ///   `order = 4` case) uses the well-known two-predict/one-update "D4"
    ///   factorization of Daubechies & Sweldens.
    /// - Every other wavelet type uses the classic Haar lifting
    ///   factorization (difference then average), matching this module's
    ///   existing convention of defaulting unsupported wavelet types to
    ///   Haar (see `get_wavelet_filters`).
    ///
    /// Boundary samples use periodic (circular) wrap-around, which keeps
    /// the transform well-defined for any signal length without discarding
    /// information at the edges, and -- since lifting is invertible for
    /// *any* boundary rule as long as the inverse mirrors it -- does not
    /// compromise exact invertibility.
    pub fn lifting_dwt(&self, signal: &Tensor<f32>) -> Result<(Tensor<f32>, Tensor<f32>)> {
        let signal_shape = signal.shape();
        if signal_shape.ndim() != 1 {
            return Err(TorshError::InvalidArgument(
                "Lifting DWT requires 1D tensor".to_string(),
            ));
        }
        let signal_length = signal_shape.dims()[0];
        if signal_length == 0 || signal_length % 2 != 0 {
            return Err(TorshError::InvalidArgument(format!(
                "Lifting DWT requires a non-empty, even-length signal, got length {signal_length}"
            )));
        }
        let half_length = signal_length / 2;
        let data = signal.to_vec()?;
        let mut even: Vec<f64> = (0..half_length).map(|i| data[2 * i] as f64).collect();
        let mut odd: Vec<f64> = (0..half_length).map(|i| data[2 * i + 1] as f64).collect();
        match self.wavelet {
            WaveletType::Daubechies(4) => lifting_forward_d4(&mut even, &mut odd),
            _ => lifting_forward_haar(&mut even, &mut odd),
        }
        let approx_f32: Vec<f32> = even.iter().map(|&v| v as f32).collect();
        let detail_f32: Vec<f32> = odd.iter().map(|&v| v as f32).collect();
        let approximation = from_vec(approx_f32, &[half_length], DeviceType::Cpu)?;
        let detail = from_vec(detail_f32, &[half_length], DeviceType::Cpu)?;
        Ok((approximation, detail))
    }
    /// Compute the inverse lifting-scheme transform.
    ///
    /// This is the exact algebraic inverse of [`Self::lifting_dwt`]: it
    /// undoes each lifting step in reverse order (undo update, undo
    /// predict, ...) and then merges the resulting even/odd polyphase
    /// components back into a single signal (`s[2i] = even[i]`,
    /// `s[2i+1] = odd[i]`). Because every forward step is a simple
    /// add-then-store, the inverse is always an exact subtract-then-store
    /// applied in the opposite order -- this in-place invertibility by
    /// construction is what defines the "lifting scheme".
    pub fn lifting_idwt(
        &self,
        approximation: &Tensor<f32>,
        detail: &Tensor<f32>,
    ) -> Result<Tensor<f32>> {
        let approx_shape = approximation.shape();
        let detail_shape = detail.shape();
        if approx_shape.ndim() != 1 || detail_shape.ndim() != 1 {
            return Err(TorshError::InvalidArgument(
                "Lifting IDWT requires 1D tensors".to_string(),
            ));
        }
        let approx_len = approx_shape.dims()[0];
        let detail_len = detail_shape.dims()[0];
        if approx_len != detail_len {
            return Err(TorshError::InvalidArgument(format!(
                "Lifting IDWT requires matching approximation/detail lengths, got \
                 {approx_len} and {detail_len}"
            )));
        }
        if approx_len == 0 {
            return Err(TorshError::InvalidArgument(
                "Lifting IDWT requires non-empty approximation/detail tensors".to_string(),
            ));
        }
        let mut even: Vec<f64> = approximation.to_vec()?.iter().map(|&v| v as f64).collect();
        let mut odd: Vec<f64> = detail.to_vec()?.iter().map(|&v| v as f64).collect();
        match self.wavelet {
            WaveletType::Daubechies(4) => lifting_inverse_d4(&mut even, &mut odd),
            _ => lifting_inverse_haar(&mut even, &mut odd),
        }
        let reconstructed_length = approx_len * 2;
        let mut output = vec![0.0f32; reconstructed_length];
        for i in 0..approx_len {
            output[2 * i] = even[i] as f32;
            output[2 * i + 1] = odd[i] as f32;
        }
        from_vec(output, &[reconstructed_length], DeviceType::Cpu)
    }
}
/// Wavelet Packet Transform processor
pub struct WaveletPacketProcessor {
    pub wavelet: WaveletType,
    pub max_level: usize,
}
impl WaveletPacketProcessor {
    pub fn new(wavelet: WaveletType, max_level: usize) -> Self {
        Self { wavelet, max_level }
    }
    /// Compute the Wavelet Packet Transform (WPT).
    ///
    /// Unlike the standard DWT (which recurses only into the approximation
    /// branch at each level), the WPT recursively decomposes *both* the
    /// approximation and detail branches at every level, building a
    /// complete, balanced binary filter-bank tree. This function returns
    /// only the leaf packets at `self.max_level`: `2^max_level` packets of
    /// length `signal_length / 2^max_level` each, in natural (Paley) order
    /// -- packet `k`'s binary expansion (most significant bit = coarsest
    /// split) is exactly its approximation('0')/detail('1') path from the
    /// root, so consecutive pairs `(2i, 2i+1)` always share the same parent.
    ///
    /// Each level applies the wavelet's low-pass/high-pass decomposition
    /// filters (from `get_wavelet_filters`, the same filter tables used by
    /// [`crate::DiscreteWaveletProcessor`]) and downsamples by 2, treating the
    /// current node as *periodic* (circular convolution) rather than
    /// zero-padding at the boundary. This is the standard technique for
    /// finite-length perfect-reconstruction wavelet transforms: for an
    /// orthonormal analysis filter pair it makes the per-level transform an
    /// exact orthogonal matrix, so [`Self::iwpt`] (its transpose) is an
    /// exact inverse up to floating-point rounding, regardless of filter
    /// length or signal length.
    pub fn wpt(&self, signal: &Tensor<f32>) -> Result<Vec<Tensor<f32>>> {
        let signal_shape = signal.shape();
        if signal_shape.ndim() != 1 {
            return Err(TorshError::InvalidArgument(
                "WPT requires 1D tensor".to_string(),
            ));
        }
        let signal_length = signal_shape.dims()[0];
        if signal_length == 0 {
            return Err(TorshError::InvalidArgument(
                "WPT requires a non-empty signal".to_string(),
            ));
        }
        let max_level = self.max_level;
        let (lo_d, hi_d, _lo_r, _hi_r) = get_wavelet_filters(&self.wavelet)?;
        let filter_len = lo_d.len().max(hi_d.len());
        let mut projected_len = signal_length;
        for _ in 0..max_level {
            if projected_len < filter_len || projected_len % 2 != 0 {
                return Err(TorshError::InvalidArgument(format!(
                    "signal length {signal_length} cannot be decomposed to WPT level \
                     {max_level} with a {filter_len}-tap filter (node length \
                     {projected_len} became too short or odd); use a longer signal or \
                     a shallower level"
                )));
            }
            projected_len /= 2;
        }
        let lo_f64: Vec<f64> = lo_d.iter().map(|&v| v as f64).collect();
        let hi_f64: Vec<f64> = hi_d.iter().map(|&v| v as f64).collect();
        let root: Vec<f64> = signal.to_vec()?.iter().map(|&v| v as f64).collect();
        let mut nodes: Vec<Vec<f64>> = vec![root];
        for _ in 0..max_level {
            let mut next_nodes = Vec::with_capacity(nodes.len() * 2);
            for node in &nodes {
                let (approx, detail) = circular_analyze(node, &lo_f64, &hi_f64);
                next_nodes.push(approx);
                next_nodes.push(detail);
            }
            nodes = next_nodes;
        }
        nodes
            .into_iter()
            .map(|packet| {
                let packet_f32: Vec<f32> = packet.iter().map(|&v| v as f32).collect();
                let len = packet_f32.len();
                from_vec(packet_f32, &[len], DeviceType::Cpu)
            })
            .collect::<Result<Vec<_>>>()
    }
    /// Inverse Wavelet Packet Transform: reconstruct a signal from the leaf
    /// packets produced by [`Self::wpt`].
    ///
    /// `packets` must have a power-of-two length (any level's worth of leaf
    /// packets, not necessarily `self.max_level`, so partial-tree / best
    /// basis selections work too). Reconstruction proceeds level by level,
    /// from the leaves up to the root: at each level, packets are paired
    /// `(2i, 2i+1)` -- exactly the siblings produced by [`Self::wpt`] -- and
    /// merged with `circular_synthesize`, the exact transpose of the
    /// forward analysis step.
    pub fn iwpt(&self, packets: &[Tensor<f32>]) -> Result<Tensor<f32>> {
        let n_packets = packets.len();
        if n_packets == 0 {
            return Err(TorshError::InvalidArgument(
                "iwpt requires at least one packet".to_string(),
            ));
        }
        if !n_packets.is_power_of_two() {
            return Err(TorshError::InvalidArgument(format!(
                "iwpt requires a power-of-two number of packets, got {n_packets}"
            )));
        }
        for packet in packets {
            if packet.shape().ndim() != 1 {
                return Err(TorshError::InvalidArgument(
                    "iwpt requires 1D packet tensors".to_string(),
                ));
            }
        }
        let (lo_d, hi_d, _lo_r, _hi_r) = get_wavelet_filters(&self.wavelet)?;
        let lo_f64: Vec<f64> = lo_d.iter().map(|&v| v as f64).collect();
        let hi_f64: Vec<f64> = hi_d.iter().map(|&v| v as f64).collect();
        let mut nodes: Vec<Vec<f64>> = packets
            .iter()
            .map(|packet| -> Result<Vec<f64>> {
                Ok(packet.to_vec()?.iter().map(|&v| v as f64).collect())
            })
            .collect::<Result<Vec<_>>>()?;
        while nodes.len() > 1 {
            let mut parents = Vec::with_capacity(nodes.len() / 2);
            for pair in nodes.chunks(2) {
                parents.push(circular_synthesize(&pair[0], &pair[1], &lo_f64, &hi_f64));
            }
            nodes = parents;
        }
        let reconstructed = nodes.into_iter().next().ok_or_else(|| {
            TorshError::ComputeError("iwpt: reconstruction produced no output".to_string())
        })?;
        let reconstructed_f32: Vec<f32> = reconstructed.iter().map(|&v| v as f32).collect();
        let len = reconstructed_f32.len();
        from_vec(reconstructed_f32, &[len], DeviceType::Cpu)
    }
}
/// One level of circular (periodic) wavelet filter-bank analysis: convolve
/// `signal` with the low-pass filter `lo` and high-pass filter `hi` and
/// downsample each by 2, treating `signal` as periodic with period
/// `signal.len()` (which must be even). Returns `(approximation, detail)`,
/// each of length `signal.len() / 2`.
///
/// Using periodic (rather than zero-padded) boundary handling means no
/// signal energy is discarded at the edges regardless of filter length,
/// which is what allows [`circular_synthesize`] (its exact transpose) to be
/// a true inverse for orthonormal filter pairs -- the standard technique
/// for perfect-reconstruction transforms on finite-length signals. Used by
/// [`WaveletPacketProcessor::wpt`].
fn circular_analyze(signal: &[f64], lo: &[f64], hi: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = signal.len();
    let out_len = n / 2;
    let mut approx = vec![0.0f64; out_len];
    let mut detail = vec![0.0f64; out_len];
    for i in 0..out_len {
        let mut a_sum = 0.0f64;
        for (k, &coeff) in lo.iter().enumerate() {
            a_sum += coeff * signal[(2 * i + k) % n];
        }
        approx[i] = a_sum;
        let mut d_sum = 0.0f64;
        for (k, &coeff) in hi.iter().enumerate() {
            d_sum += coeff * signal[(2 * i + k) % n];
        }
        detail[i] = d_sum;
    }
    (approx, detail)
}
/// Exact transpose of [`circular_analyze`]: reconstructs a signal of length
/// `2 * approx.len()` from one level of approximation/detail coefficients.
/// Scattering each coefficient through the *same* (not time-reversed)
/// filter taps used for analysis is precisely the adjoint of the gather
/// performed by `circular_analyze`, which is the correct synthesis operator
/// whenever `lo`/`hi` form an orthonormal analysis pair -- true for the
/// Haar/Daubechies/Symlet/Coiflet families used elsewhere in this module.
/// Used by [`WaveletPacketProcessor::iwpt`].
fn circular_synthesize(approx: &[f64], detail: &[f64], lo: &[f64], hi: &[f64]) -> Vec<f64> {
    let out_len = approx.len() * 2;
    let mut output = vec![0.0f64; out_len];
    for (i, &a) in approx.iter().enumerate() {
        for (k, &coeff) in lo.iter().enumerate() {
            output[(2 * i + k) % out_len] += a * coeff;
        }
    }
    for (i, &d) in detail.iter().enumerate() {
        for (k, &coeff) in hi.iter().enumerate() {
            output[(2 * i + k) % out_len] += d * coeff;
        }
    }
    output
}
/// Haar lifting: predict (difference) then update (average), with the
/// standard orthonormal scaling so the coefficients match the Haar filter
/// bank convention used elsewhere in this module ([`get_haar_filters`]).
/// `even` and `odd` are modified in place.
fn lifting_forward_haar(even: &mut [f64], odd: &mut [f64]) {
    let sqrt2 = std::f64::consts::SQRT_2;
    let m = even.len();
    for i in 0..m {
        odd[i] -= even[i];
    }
    for i in 0..m {
        even[i] += odd[i] * 0.5;
    }
    for i in 0..m {
        even[i] *= sqrt2;
        odd[i] /= sqrt2;
    }
}
/// Exact inverse of [`lifting_forward_haar`]: undoes normalize, then
/// update, then predict, in that order.
fn lifting_inverse_haar(even: &mut [f64], odd: &mut [f64]) {
    let sqrt2 = std::f64::consts::SQRT_2;
    let m = even.len();
    for i in 0..m {
        even[i] /= sqrt2;
        odd[i] *= sqrt2;
    }
    for i in 0..m {
        even[i] -= odd[i] * 0.5;
    }
    for i in 0..m {
        odd[i] += even[i];
    }
}
/// Daubechies "D4" (4-tap, 2-vanishing-moment) lifting scheme: the
/// factorization of the D4 orthogonal filter bank into a predict step, an
/// update step, and a second predict step that also folds in the one-sample
/// alignment shift inherent to a 4-tap (as opposed to 2-tap Haar) filter,
/// followed by normalization (derived via the Euclidean algorithm on the
/// D4 polyphase matrix -- Daubechies & Sweldens, "Factoring wavelet
/// transforms into lifting steps", 1998 -- and verified to reproduce
/// [`circular_analyze`]'s db4 output exactly). Neighbor accesses use
/// periodic (circular) wrap-around so the scheme is well-defined for any
/// signal length. `even` and `odd` are modified in place.
fn lifting_forward_d4(even: &mut [f64], odd: &mut [f64]) {
    let m = even.len();
    if m == 0 {
        return;
    }
    let sqrt3 = 3.0_f64.sqrt();
    let k_even = (2.0 + sqrt3).sqrt();
    let k_odd = -(2.0 - sqrt3).sqrt();
    let coeff_a = sqrt3 / 4.0;
    let coeff_b = (sqrt3 - 2.0) / 4.0;
    for i in 0..m {
        odd[i] -= sqrt3 * even[i];
    }
    let p1 = odd.to_vec();
    for i in 0..m {
        let next = p1[(i + 1) % m];
        even[i] += coeff_a * p1[i] + coeff_b * next;
    }
    for i in 0..m {
        let next_p1 = p1[(i + 1) % m];
        odd[i] = next_p1 + even[i];
    }
    for i in 0..m {
        even[i] *= k_even;
        odd[i] *= k_odd;
    }
}
/// Exact inverse of [`lifting_forward_d4`]: undoes normalize, predict 2
/// (recovering the intermediate "p1" stage from its shifted form), update,
/// and predict 1, in that order.
fn lifting_inverse_d4(even: &mut [f64], odd: &mut [f64]) {
    let m = even.len();
    if m == 0 {
        return;
    }
    let sqrt3 = 3.0_f64.sqrt();
    let k_even = (2.0 + sqrt3).sqrt();
    let k_odd = -(2.0 - sqrt3).sqrt();
    let coeff_a = sqrt3 / 4.0;
    let coeff_b = (sqrt3 - 2.0) / 4.0;
    for i in 0..m {
        even[i] /= k_even;
        odd[i] /= k_odd;
    }
    let post_update_even = even.to_vec();
    let d3 = odd.to_vec();
    let mut p1 = vec![0.0f64; m];
    for j in 0..m {
        let prev = (j + m - 1) % m;
        p1[j] = d3[prev] - post_update_even[prev];
    }
    for i in 0..m {
        let next = p1[(i + 1) % m];
        even[i] = post_update_even[i] - coeff_a * p1[i] - coeff_b * next;
    }
    for i in 0..m {
        odd[i] = p1[i] + sqrt3 * even[i];
    }
}
