//! ToRSh CLI Library
//!
//! This library provides the core functionality for the ToRSh command-line interface,
//! including configuration management, command implementations, and utilities.

pub mod commands;
pub mod config;
mod tls;
pub mod utils;

// Re-export commonly used types
pub use config::Config;

/// CLI version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const VERSION_MAJOR: u32 = 0;
pub const VERSION_MINOR: u32 = 1;
pub const VERSION_PATCH: u32 = 0;

/// Error returned when the running CLI is older than a required version.
#[derive(Debug, thiserror::Error)]
#[error("ToRSh CLI version {required_major}.{required_minor} or higher required, but got {have_major}.{have_minor}")]
pub struct VersionMismatch {
    /// Required major version.
    pub required_major: u32,
    /// Required minor version.
    pub required_minor: u32,
    /// Running major version.
    pub have_major: u32,
    /// Running minor version.
    pub have_minor: u32,
}

/// Check if CLI version is compatible with required version.
pub fn check_version(required_major: u32, required_minor: u32) -> Result<(), VersionMismatch> {
    if VERSION_MAJOR < required_major
        || (VERSION_MAJOR == required_major && VERSION_MINOR < required_minor)
    {
        return Err(VersionMismatch {
            required_major,
            required_minor,
            have_major: VERSION_MAJOR,
            have_minor: VERSION_MINOR,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_check() {
        assert!(check_version(0, 1).is_ok());
        assert!(check_version(1, 0).is_err());
    }
}
