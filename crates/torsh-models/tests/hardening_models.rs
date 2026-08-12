//! Wave-3 production-hardening regression tests for torsh-models.
//!
//! Covers: F030 (honest PyTorch-checkpoint errors), F031 (no fabricated
//! checksums), F126 (NaN-safe ordering), F228 (non-panicking global registry).

use std::collections::HashMap;
use torsh_models::registry::ModelRegistry;

/// F030: loading a `.pth` file must return an honest error rather than
/// fabricating a flat f32 tensor from raw pickle bytes.
#[test]
fn f030_load_pytorch_checkpoint_errors() {
    let dir = std::env::temp_dir().join("torsh_hardening_f030_load");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("fake.pth");
    // Bytes that would previously be reinterpreted as f32 (multiple of 4).
    std::fs::write(&path, [0u8; 16]).expect("write temp");

    let result = torsh_models::load_pytorch_checkpoint(&path, None);
    assert!(
        result.is_err(),
        "load_pytorch_checkpoint must error, not fabricate a tensor"
    );
    let _ = std::fs::remove_file(&path);
}

/// F030: saving a "PyTorch checkpoint" must error rather than writing a bespoke
/// format that PyTorch cannot read while claiming to be a checkpoint.
#[test]
fn f030_save_pytorch_checkpoint_errors() {
    let dir = std::env::temp_dir().join("torsh_hardening_f030_save");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("out.pth");
    let tensors = HashMap::new();

    let result = torsh_models::save_pytorch_checkpoint(&path, &tensors, None);
    assert!(
        result.is_err(),
        "save_pytorch_checkpoint must error, not write a fake format"
    );
    let _ = std::fs::remove_file(&path);
}

/// F031: no built-in model may carry a fabricated checksum. Every registered
/// checksum must be either empty (verification explicitly skipped) or a valid
/// 64-character lowercase-hex SHA-256, so short placeholders cannot reappear.
#[test]
fn f031_builtin_checksums_are_empty_or_valid_sha256() {
    let dir = std::env::temp_dir().join("torsh_hardening_f031_registry");
    let _ = std::fs::create_dir_all(&dir);
    let registry = ModelRegistry::new(&dir).expect("registry");
    registry
        .register_builtin_models()
        .expect("register builtin");

    for info in registry.list_models() {
        let c = &info.checksum;
        if c.is_empty() {
            continue; // verification skipped honestly
        }
        assert_eq!(
            c.len(),
            64,
            "model '{}' has a non-empty checksum of length {} (fabricated placeholder?)",
            info.name,
            c.len()
        );
        assert!(
            c.bytes().all(|b| b.is_ascii_hexdigit()),
            "model '{}' checksum is not valid hex: {}",
            info.name,
            c
        );
    }
}

/// F126: top-k token selection must not panic on NaN logits.
#[test]
fn f126_get_top_k_tokens_nan_no_panic() {
    use torsh_models::nlp::NlpModelUtils;
    let logits = vec![0.5f32, f32::NAN, 0.2, 0.9];
    let out = NlpModelUtils::get_top_k_tokens(&logits, 2, None);
    assert_eq!(
        out.len(),
        2,
        "should still return k results without panicking"
    );
}

/// F126: greedy decoding must return an error (not panic, not a silent token)
/// when the model produces NaN logits.
#[test]
fn f126_generate_text_greedy_nan_errors() {
    use torsh_models::nlp::NlpModelUtils;
    let model_fn = |_tokens: &[u32]| -> torsh_models::ModelResult<Vec<f32>> {
        Ok(vec![0.1f32, f32::NAN, 0.3])
    };
    let result = NlpModelUtils::generate_text_greedy(&[0u32], &model_fn, 4, None);
    assert!(
        result.is_err(),
        "greedy decoding must surface NaN logits as an error"
    );
}

/// F228: the global registry initializes without panicking and returns Ok.
#[test]
fn f228_global_registry_no_panic() {
    let result = torsh_models::registry::get_global_registry();
    assert!(result.is_ok(), "global registry init must not abort");
}
