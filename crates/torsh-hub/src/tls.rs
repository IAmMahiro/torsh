//! Pure-Rust TLS wiring for `reqwest` clients.
//!
//! The workspace builds `reqwest` with the `rustls-no-provider` feature, so no
//! `aws-lc-rs` (BoringSSL / C) crypto backend is linked into the default build.
//! This module supplies the pure-Rust RustCrypto crypto provider together with
//! the Mozilla `webpki-roots` trust anchors and hands `reqwest` a fully
//! pre-configured [`rustls::ClientConfig`], so certificate verification stays
//! real (no `danger_accept_invalid_certs`, no disabled validation).

use std::sync::{Arc, Once};

use torsh_core::error::{Result, TorshError};

/// Install the RustCrypto provider as the process-wide default rustls crypto
/// provider (idempotent).
///
/// This is a safety net: any `reqwest` client built *without*
/// [`use_preconfigured_tls`](reqwest::ClientBuilder::use_preconfigured_tls)
/// would otherwise panic under the `rustls-no-provider` feature when it looks up
/// the (absent) default crypto provider.
fn install_default_provider() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // Ignore the error returned when a provider is already installed.
        let _ = oxitls_rustcrypto_provider::provider().install_default();
    });
}

/// Build a [`rustls::ClientConfig`] backed by the pure-Rust RustCrypto provider
/// and the Mozilla `webpki-roots` trust anchors, with full certificate
/// verification enabled.
fn rustls_config() -> Result<rustls::ClientConfig> {
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());

    let mut config = rustls::ClientConfig::builder_with_provider(Arc::new(
        oxitls_rustcrypto_provider::provider(),
    ))
    .with_safe_default_protocol_versions()
    .map_err(|e| TorshError::IoError(format!("failed to configure rustls TLS: {e}")))?
    .with_root_certificates(roots)
    .with_no_client_auth();

    // Match reqwest's default ALPN advertisement (HTTP/2 + HTTP/1.1); the
    // pre-configured TLS path does not set this automatically.
    config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
    Ok(config)
}

/// An async [`reqwest::ClientBuilder`] pre-configured with the pure-Rust TLS
/// stack.
pub(crate) fn client_builder() -> Result<reqwest::ClientBuilder> {
    install_default_provider();
    Ok(reqwest::Client::builder().use_preconfigured_tls(rustls_config()?))
}

/// A blocking [`reqwest::blocking::ClientBuilder`] pre-configured with the
/// pure-Rust TLS stack.
pub(crate) fn blocking_client_builder() -> Result<reqwest::blocking::ClientBuilder> {
    install_default_provider();
    Ok(reqwest::blocking::Client::builder().use_preconfigured_tls(rustls_config()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rustls_config_builds_with_webpki_roots() {
        let config = rustls_config().expect("pure-Rust TLS config must build");
        assert!(config.alpn_protocols.contains(&b"h2".to_vec()));
        assert!(config.alpn_protocols.contains(&b"http/1.1".to_vec()));
    }

    #[test]
    fn blocking_client_builder_builds_a_client() {
        let client = blocking_client_builder()
            .expect("builder")
            .build()
            .expect("blocking client must build with the pure-Rust TLS stack");
        drop(client);
    }
}
