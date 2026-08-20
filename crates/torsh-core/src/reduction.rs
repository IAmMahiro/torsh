//! Canonical reduction mode for loss functions.
//!
//! This enum is the single source of truth for reduction semantics
//! across all ToRSh crates. All loss functions should use this type
//! instead of string literals or crate-local enums.

use core::fmt;
use core::str::FromStr;

/// Specifies how to reduce per-element losses into a scalar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reduction {
    /// No reduction - return per-element losses.
    None,
    /// Mean over all elements (sum / numel). Matches PyTorch `"mean"`.
    Mean,
    /// Sum of all elements.
    Sum,
    /// Mean over batch dimension only (sum / batch_size).
    /// Equivalent to PyTorch `"batchmean"`.
    BatchMean,
}

impl FromStr for Reduction {
    type Err = ReductionParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "none" => Ok(Self::None),
            "mean" => Ok(Self::Mean),
            "sum" => Ok(Self::Sum),
            "batchmean" => Ok(Self::BatchMean),
            _ => Err(ReductionParseError(s.to_string())),
        }
    }
}

/// Error returned when parsing an invalid reduction string.
#[derive(Debug, Clone)]
pub struct ReductionParseError(String);

impl fmt::Display for ReductionParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid reduction mode '{}', expected one of: none, mean, sum, batchmean", self.0)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ReductionParseError {}