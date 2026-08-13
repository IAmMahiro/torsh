//! Regression tests for [`WordEmbedding`]'s `padding_idx` handling.
//!
//! `WordEmbedding::new` used to zero the padding row of a *copy* of the
//! weight tensor: `index_select` returns a gather into fresh storage, so
//! `padding_row.fill_(0.0)` zeroed that temporary and dropped it, leaving the
//! real weight's padding row at its random `randn` initial values.
//! `WordEmbedding::new(5, 4, Some(2), ..)`'s row 2 was observably non-zero.
//! These tests pin the fixed behavior: the padding row is zero in the actual
//! weight storage, and every code path that reads it (`forward`, and the raw
//! weight tensor itself) sees the zero.

use torsh_core::device::DeviceType as Device;
use torsh_core::dtype::DType;
use torsh_text::{Module, WordEmbedding};

/// The padding row, read directly off `weight`, must be all-zero right after
/// construction — not just whatever the padding lookup happens to return.
#[test]
fn padding_row_is_zeroed_in_the_actual_weight_storage() {
    let embedding = WordEmbedding::new(
        5,
        4,
        Some(2),
        None,
        2.0,
        false,
        false,
        &Device::Cpu,
        DType::F32,
    )
    .expect("WordEmbedding::new should succeed");

    let weight_data = embedding
        .weight
        .tensor()
        .to_vec()
        .expect("to_vec should succeed");
    let padding_row = &weight_data[2 * 4..3 * 4];
    assert_eq!(
        padding_row,
        [0.0f32, 0.0, 0.0, 0.0],
        "padding_idx row must be zero in the weight tensor itself, got {padding_row:?}"
    );
}

/// Every non-padding row must still be genuinely random: the fix must zero
/// exactly the padding row, not the whole weight matrix (nor leave it
/// untouched under some accidental no-op).
#[test]
fn only_the_padding_row_is_zeroed() {
    let embedding = WordEmbedding::new(
        6,
        4,
        Some(2),
        None,
        2.0,
        false,
        false,
        &Device::Cpu,
        DType::F32,
    )
    .expect("WordEmbedding::new should succeed");

    let weight_data = embedding
        .weight
        .tensor()
        .to_vec()
        .expect("to_vec should succeed");
    for row in 0..6 {
        let this_row = &weight_data[row * 4..(row + 1) * 4];
        if row == 2 {
            continue;
        }
        assert!(
            this_row.iter().any(|&v| v != 0.0),
            "row {row} must keep its random `randn` initialization, got {this_row:?}"
        );
    }
}

/// `forward` looks up rows via `index_select` on the same `weight` tensor, so
/// it must observe the zeroed row too, not a stale copy.
#[test]
fn forward_returns_zeros_for_the_padding_index() {
    let embedding = WordEmbedding::new(
        5,
        4,
        Some(2),
        None,
        2.0,
        false,
        false,
        &Device::Cpu,
        DType::F32,
    )
    .expect("WordEmbedding::new should succeed");

    let input = torsh_tensor::Tensor::from_vec(vec![2.0f32], &[1]).expect("input tensor");
    let output = embedding.forward(&input).expect("forward should succeed");
    let values = output.to_vec().expect("to_vec should succeed");
    assert_eq!(
        values,
        vec![0.0f32; 4],
        "forward() must return the zeroed padding row for padding_idx, got {values:?}"
    );
}

/// A `padding_idx` at or beyond `vocab_size` is out of range for the "zero a
/// row" step; the constructor's own bounds check must skip it silently
/// rather than panicking or writing out of bounds.
#[test]
fn out_of_range_padding_idx_does_not_panic_or_write_out_of_bounds() {
    let embedding = WordEmbedding::new(
        4,
        4,
        Some(4), // == vocab_size, i.e. out of range
        None,
        2.0,
        false,
        false,
        &Device::Cpu,
        DType::F32,
    )
    .expect("WordEmbedding::new should succeed even with an out-of-range padding_idx");

    let weight_data = embedding
        .weight
        .tensor()
        .to_vec()
        .expect("to_vec should succeed");
    assert_eq!(
        weight_data.len(),
        4 * 4,
        "weight storage must be untouched in size"
    );
}

/// Without a `padding_idx`, no row is special: the zeroing branch must not
/// run, and the weight matrix stays fully random.
#[test]
fn no_padding_idx_leaves_every_row_random() {
    let embedding = WordEmbedding::new(
        6,
        4,
        None,
        None,
        2.0,
        false,
        false,
        &Device::Cpu,
        DType::F32,
    )
    .expect("WordEmbedding::new should succeed");

    let weight_data = embedding
        .weight
        .tensor()
        .to_vec()
        .expect("to_vec should succeed");
    let all_zero_rows = (0..6)
        .filter(|&row| {
            weight_data[row * 4..(row + 1) * 4]
                .iter()
                .all(|&v| v == 0.0)
        })
        .count();
    assert_eq!(
        all_zero_rows, 0,
        "with padding_idx = None, no row should be forced to zero"
    );
}
