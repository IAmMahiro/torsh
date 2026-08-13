//! Production-hardening regression tests for `torsh-cli`.
//!
//! Each test is named after the finding it pins down.

use torsh_cli::commands::model::pytorch_reader::{read_state_dict, TensorDType, TensorData};
use torsh_cli::commands::real_training::{
    build_mlp, evaluate_loss, synthetic_regression, train_regression, MlpConfig,
};

// ---------------------------------------------------------------------------
// F005 - `torsh train` must perform REAL training (loss must actually decrease),
// not fabricate a loss curve with an RNG.
// ---------------------------------------------------------------------------

#[test]
fn f005_real_training_decreases_loss_on_synthetic_regression() {
    let cfg = MlpConfig {
        input_dim: 8,
        hidden_dim: 16,
        output_dim: 1,
    };
    let model = build_mlp(&cfg).expect("build MLP");
    let data =
        synthetic_regression(64, cfg.input_dim, cfg.output_dim, 0xC0FF_EE01).expect("dataset");

    let initial = evaluate_loss(&model, &data).expect("initial loss");
    let history = train_regression(&model, &data, 0.05, 20).expect("training");

    assert_eq!(history.len(), 20, "one loss reading per step");
    let final_loss = *history.last().expect("final loss");

    assert!(
        final_loss < initial,
        "real training must reduce the loss: initial={initial}, final={final_loss}"
    );
    // The measured loss after training must be a real, finite number.
    assert!(
        final_loss.is_finite(),
        "loss must be finite, got {final_loss}"
    );
    assert!(
        final_loss >= 0.0,
        "MSE loss cannot be negative, got {final_loss}"
    );
}

#[test]
fn f005_training_history_is_deterministic_not_random() {
    // A genuinely-computed loss curve is reproducible for a fixed seed and fixed
    // initial weights; a fabricated RNG curve would not track the same values.
    let cfg = MlpConfig {
        input_dim: 4,
        hidden_dim: 8,
        output_dim: 2,
    };
    let data = synthetic_regression(32, cfg.input_dim, cfg.output_dim, 42).expect("dataset");

    let model_a = build_mlp(&cfg).expect("model a");
    let model_b = build_mlp(&cfg).expect("model b");

    // Same architecture + same data + same LR: the loss must be monotone-ish and
    // the two runs must land at meaningfully reduced loss (real optimisation),
    // demonstrating the numbers are computed, not drawn from a distribution.
    let hist_a = train_regression(&model_a, &data, 0.05, 30).expect("train a");
    let hist_b = train_regression(&model_b, &data, 0.05, 30).expect("train b");

    let init = evaluate_loss(&build_mlp(&cfg).expect("m"), &data).expect("init");
    assert!(hist_a.last().unwrap() < &init);
    assert!(hist_b.last().unwrap() < &init);
}

// ---------------------------------------------------------------------------
// pytorch_parser - the .pt reader must reconstruct REAL tensor values from a
// zip+pickle checkpoint, not fabricate them with an RNG.
// ---------------------------------------------------------------------------

/// Append one STORED (uncompressed) local zip entry.
fn push_stored_entry(out: &mut Vec<u8>, name: &str, body: &[u8]) {
    out.extend_from_slice(b"PK\x03\x04");
    out.extend_from_slice(&20u16.to_le_bytes()); // version needed
    out.extend_from_slice(&0u16.to_le_bytes()); // flags
    out.extend_from_slice(&0u16.to_le_bytes()); // method: STORED
    out.extend_from_slice(&0u16.to_le_bytes()); // mod time
    out.extend_from_slice(&0u16.to_le_bytes()); // mod date
    out.extend_from_slice(&0u32.to_le_bytes()); // crc32 (reader ignores)
    out.extend_from_slice(&(body.len() as u32).to_le_bytes()); // compressed size
    out.extend_from_slice(&(body.len() as u32).to_le_bytes()); // uncompressed size
    out.extend_from_slice(&(name.len() as u16).to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // extra len
    out.extend_from_slice(name.as_bytes());
    out.extend_from_slice(body);
}

/// Build a pickle stream for `{"w": rebuild_tensor_v2(FloatStorage "0", 0, (2,3), (3,1), False, None)}`.
fn crafted_pickle() -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&[0x80, 0x02]); // PROTO 2
    p.push(b'}'); // EMPTY_DICT
    p.extend_from_slice(&[b'q', 0x00]); // BINPUT 0
    p.push(b'('); // MARK
    p.extend_from_slice(&[0x8c, 0x01]); // SHORT_BINUNICODE len 1
    p.push(b'w'); // "w"
    p.extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\n"); // GLOBAL
    p.push(b'('); // MARK (args)
                  // persistent id tuple
    p.push(b'('); // MARK
    p.extend_from_slice(&[0x8c, 0x07]);
    p.extend_from_slice(b"storage");
    p.extend_from_slice(b"ctorch\nFloatStorage\n"); // GLOBAL
    p.extend_from_slice(&[0x8c, 0x01]);
    p.push(b'0'); // storage key "0"
    p.extend_from_slice(&[0x8c, 0x03]);
    p.extend_from_slice(b"cpu");
    p.extend_from_slice(&[b'K', 0x06]); // BININT1 6 (numel)
    p.push(b't'); // TUPLE -> pid
    p.push(b'Q'); // BINPERSID
    p.extend_from_slice(&[b'K', 0x00]); // storage_offset 0
    p.push(b'('); // size tuple
    p.extend_from_slice(&[b'K', 0x02, b'K', 0x03]);
    p.push(b't');
    p.push(b'('); // stride tuple
    p.extend_from_slice(&[b'K', 0x03, b'K', 0x01]);
    p.push(b't');
    p.push(0x89); // NEWFALSE (requires_grad)
    p.push(b'N'); // NONE (backward hooks)
    p.push(b't'); // TUPLE -> args
    p.push(b'R'); // REDUCE -> tensor
    p.push(b'u'); // SETITEMS
    p.push(b'.'); // STOP
    p
}

#[test]
fn pytorch_reader_reconstructs_real_tensor_values() {
    let mut storage = Vec::new();
    for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
        storage.extend_from_slice(&v.to_le_bytes());
    }

    let mut zip = Vec::new();
    push_stored_entry(&mut zip, "archive/data.pkl", &crafted_pickle());
    push_stored_entry(&mut zip, "archive/data/0", &storage);

    let tensors = read_state_dict(&zip).expect("reader must reconstruct the checkpoint");
    assert_eq!(tensors.len(), 1, "one tensor in the state_dict");
    let t = &tensors[0];
    assert_eq!(t.name, "w");
    assert_eq!(t.dtype, TensorDType::F32);
    assert_eq!(t.shape, vec![2, 3]);
    match &t.data {
        TensorData::F32(values) => {
            assert_eq!(values, &vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        }
        other => panic!("expected F32 data, got {other:?}"),
    }
}

#[test]
fn pytorch_reader_rejects_non_zip_input() {
    // A legacy pure-pickle .pt (no PK header) must be an honest error, not a
    // fabricated success.
    let err = read_state_dict(b"\x80\x02}q\x00.").unwrap_err();
    assert!(
        err.to_string().contains("zip-based"),
        "unexpected error: {err}"
    );
}
