//! Wave-3 production-hardening regression tests for torsh-hub crypto (F021/F024).
//!
//! Verifies that model signing/verification uses real Ed25519 cryptography and
//! that the old forgeable "placeholder" constant no longer authenticates.

use std::collections::HashMap;
use torsh_hub::security::{SecurityManager, SignatureAlgorithm};

fn write_temp_model(name: &str, contents: &[u8]) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("torsh_hardening_hub_crypto");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join(name);
    std::fs::write(&path, contents).expect("write temp model");
    path
}

/// F021/F024 money test: a signature file whose signature field is the old
/// literal placeholder constant must NOT verify as authentic.
#[test]
fn f024_placeholder_signature_is_rejected() {
    let mut mgr = SecurityManager::new();
    let key = SecurityManager::generate_key_pair("k1".to_string(), SignatureAlgorithm::Ed25519)
        .expect("generate ed25519 key");
    mgr.add_key(key);

    let model = write_temp_model("model_a.bin", b"the real model bytes");
    let mut sig = mgr
        .sign_model(&model, "k1", None)
        .expect("sign should succeed with a real key");

    // Attacker overwrites the signature with the historical forgeable constant.
    sig.signature = "ed25519_signature_placeholder".to_string();

    let verified = mgr
        .verify_model(&model, &sig, false)
        .expect("verify should not error");
    assert!(
        !verified,
        "the placeholder constant must never authenticate a model"
    );

    let _ = std::fs::remove_file(&model);
}

/// A genuine signature verifies, and any tampering with the model content fails.
#[test]
fn f024_real_roundtrip_and_tamper_detection() {
    let mut mgr = SecurityManager::new();
    let key = SecurityManager::generate_key_pair("k2".to_string(), SignatureAlgorithm::Ed25519)
        .expect("generate ed25519 key");
    mgr.add_key(key);

    let model = write_temp_model("model_b.bin", b"original contents");
    let sig = mgr.sign_model(&model, "k2", None).expect("sign");

    // Prove the sign path produced a real 64-byte Ed25519 signature (128 hex
    // chars, valid hex) rather than merely something that fails to verify.
    assert_eq!(
        sig.signature.len(),
        128,
        "Ed25519 signature must be 64 bytes / 128 hex chars, got {}",
        sig.signature.len()
    );
    assert!(
        hex::decode(&sig.signature).is_ok(),
        "signature must be valid hex"
    );

    assert!(
        mgr.verify_model(&model, &sig, false).expect("verify"),
        "a genuine signature must verify"
    );

    // Tamper with the model: same signature must now fail.
    std::fs::write(&model, b"malicious replacement").expect("overwrite");
    assert!(
        !mgr.verify_model(&model, &sig, false).expect("verify"),
        "a tampered model must fail verification"
    );

    let _ = std::fs::remove_file(&model);
}

/// Flipping a single hex nibble of a genuine signature (file unchanged) must
/// fail. This reaches `verify_ed25519` with a valid-length, valid-hex, wrong
/// signature — the only path that proves real cryptographic verification, since
/// the file-hash check passes.
#[test]
fn f024_flipped_signature_byte_fails() {
    let mut mgr = SecurityManager::new();
    let key = SecurityManager::generate_key_pair("k5".to_string(), SignatureAlgorithm::Ed25519)
        .expect("gen");
    mgr.add_key(key);

    let model = write_temp_model("model_e.bin", b"unchanged payload");
    let mut sig = mgr.sign_model(&model, "k5", None).expect("sign");

    // Flip the first hex character to a different valid hex digit.
    let mut chars: Vec<char> = sig.signature.chars().collect();
    chars[0] = if chars[0] == '0' { '1' } else { '0' };
    sig.signature = chars.into_iter().collect();

    let verified = mgr.verify_model(&model, &sig, false).expect("verify");
    assert!(
        !verified,
        "a signature with one flipped nibble must fail cryptographic verification"
    );

    let _ = std::fs::remove_file(&model);
}

/// A signature verified against a *different* key must fail (real key binding).
#[test]
fn f024_wrong_key_fails() {
    let mut mgr = SecurityManager::new();
    let k_sign =
        SecurityManager::generate_key_pair("signer".to_string(), SignatureAlgorithm::Ed25519)
            .expect("gen signer");
    // A second, independent keypair reusing the same key_id label as stored,
    // but different key bytes, replaces the signer's public key.
    let k_other =
        SecurityManager::generate_key_pair("signer".to_string(), SignatureAlgorithm::Ed25519)
            .expect("gen other");

    // Keys must actually differ (real randomness, not a constant).
    assert_ne!(
        k_sign.public_key, k_other.public_key,
        "generated keypairs must be unique"
    );

    mgr.add_key(k_sign.clone());
    let model = write_temp_model("model_c.bin", b"payload");
    let sig = mgr.sign_model(&model, "signer", None).expect("sign");

    // Replace stored key with an unrelated public key under the same id.
    mgr.add_key(k_other);
    let verified = mgr.verify_model(&model, &sig, false).expect("verify");
    assert!(!verified, "verification under a different key must fail");

    let _ = std::fs::remove_file(&model);
}

/// RSA and ECDSA are not implemented and must refuse rather than return a
/// forgeable placeholder.
#[test]
fn f024_rsa_ecdsa_keygen_refuses() {
    assert!(
        SecurityManager::generate_key_pair("r".to_string(), SignatureAlgorithm::RsaSha256).is_err(),
        "RSA key generation must return an error, not a constant key"
    );
    assert!(
        SecurityManager::generate_key_pair("e".to_string(), SignatureAlgorithm::EcdsaP256).is_err(),
        "ECDSA key generation must return an error, not a constant key"
    );
}

/// Sanity: metadata round-trips through signing.
#[test]
fn f024_signature_carries_metadata() {
    let mut mgr = SecurityManager::new();
    let key = SecurityManager::generate_key_pair("k3".to_string(), SignatureAlgorithm::Ed25519)
        .expect("gen");
    mgr.add_key(key);
    let model = write_temp_model("model_d.bin", b"xyz");
    let mut md = HashMap::new();
    md.insert("author".to_string(), "test".to_string());
    let sig = mgr.sign_model(&model, "k3", Some(md)).expect("sign");
    assert_eq!(sig.metadata.get("author").map(String::as_str), Some("test"));
    let _ = std::fs::remove_file(&model);
}
