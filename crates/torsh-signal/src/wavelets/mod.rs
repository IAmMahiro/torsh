//! Wavelet transform capabilities with simplified SciRS2 integration
//!
//! This module provides wavelet analysis tools: continuous and discrete
//! wavelet transforms, a full recursive wavelet packet transform (WPT) with
//! its inverse, a Sweldens (1996) lifting-scheme DWT/IDWT, and wavelet
//! denoising. All transforms are self-contained (Haar, Daubechies, Symlet,
//! Coiflet and Biorthogonal filter banks are hard-coded below) and do not
//! depend on scirs2-signal.

pub mod packet_lifting;
pub mod core;

// Re-export all types
pub use packet_lifting::*;
pub use core::*;

#[cfg(test)]
mod tests;
