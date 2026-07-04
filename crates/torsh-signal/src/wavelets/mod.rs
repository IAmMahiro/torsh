//! Wavelet transform capabilities with simplified SciRS2 integration
//!
//! This module provides wavelet analysis tools: continuous and discrete
//! wavelet transforms, a full recursive wavelet packet transform (WPT) with
//! its inverse, a Sweldens (1996) lifting-scheme DWT/IDWT, and wavelet
//! denoising. All transforms are self-contained (Haar, Daubechies, Symlet,
//! Coiflet and Biorthogonal filter banks are hard-coded below) and do not
//! depend on scirs2-signal.

pub mod core;
pub mod packet_lifting;

// Re-export all types
pub use core::*;
pub use packet_lifting::*;

#[cfg(test)]
mod tests;
