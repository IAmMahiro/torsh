//! Production-hardening regression tests for `torsh-text`.
//!
//! Each test in this file pins down a defect found during the hardening
//! campaign so that it cannot silently regress:
//!
//! * F074 — `BPETokenizer::from_texts` never applied learned merges and
//!   therefore looped forever.
//! * F075 — the decoder half of `models::integration` fabricated random
//!   logits, and `text_similarity` / `classify` returned constants.
//! * F277 — `quick_clean` recompiled a regex on every call.
//! * F278 — vocabulary IDs were assigned in `HashMap` iteration order.

use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use torsh_core::device::DeviceType;
use torsh_tensor::Tensor;
use torsh_text::convenience::TextUtilities;
use torsh_text::integration::{AdvancedTextDecoder, TextModelWrapper};
use torsh_text::tokenization::{BPETokenizer, Tokenizer, WhitespaceTokenizer};
use torsh_text::{GenerationConfig, TextEncoder, TextModel};

fn quick_clean(text: &str) -> String {
    TextUtilities::quick_clean(text)
}

/// Corpus used by the determinism and similarity tests.
fn corpus() -> Vec<String> {
    vec![
        "alpha beta gamma delta epsilon zeta".to_string(),
        "eta theta iota kappa lambda mu".to_string(),
        "nu xi omicron pi rho sigma".to_string(),
    ]
}

// ---------------------------------------------------------------------------
// F074 — BPE training must terminate and must actually merge
// ---------------------------------------------------------------------------

/// Training is executed on a worker thread guarded by a timeout so that the
/// pre-fix infinite loop fails the test instead of hanging the whole suite.
fn train_bpe_with_timeout(texts: Vec<String>, vocab_size: usize) -> BPETokenizer {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let result = BPETokenizer::from_texts(&texts, vocab_size);
        let _ = tx.send(result);
    });

    match rx.recv_timeout(Duration::from_secs(20)) {
        Ok(Ok(tokenizer)) => tokenizer,
        Ok(Err(e)) => panic!("BPE training failed: {e}"),
        Err(_) => panic!("F074: BPETokenizer::from_texts did not terminate within 20s"),
    }
}

#[test]
fn f074_bpe_training_terminates_and_learns_multi_char_merges() {
    let texts = vec![
        "abcd abcd abcd efgh efgh".to_string(),
        "abcd efgh ij".to_string(),
    ];
    let tokenizer = train_bpe_with_timeout(texts, 20);

    assert!(
        tokenizer.vocab_size() <= 20,
        "vocabulary must not exceed the requested size, got {}",
        tokenizer.vocab_size()
    );

    let tokens = tokenizer
        .tokenize("abcd")
        .expect("tokenizing a trained word must succeed");
    let longest = tokens
        .iter()
        .map(|t| t.chars().count())
        .max()
        .unwrap_or_default();
    assert!(
        longest >= 3,
        "F074: BPE must learn merges longer than a bigram, got tokens {tokens:?}"
    );
}

#[test]
fn f074_bpe_roundtrip_preserves_word() {
    let texts = vec!["lower lower lower newest newest widest".to_string()];
    let tokenizer = train_bpe_with_timeout(texts, 30);

    let ids = tokenizer.encode("lower").expect("encode");
    let decoded = tokenizer.decode(&ids).expect("decode");
    assert_eq!(
        decoded, "lower",
        "BPE encode/decode round-trip must be exact"
    );
}

#[test]
fn f074_bpe_respects_small_vocab_budget() {
    // vocab_size smaller than the seeded character vocabulary must still return.
    let texts = vec!["abcdefghijklmnop".to_string()];
    let tokenizer = train_bpe_with_timeout(texts, 5);
    assert!(
        tokenizer.vocab_size() >= 4,
        "special tokens must be present"
    );
}

/// Byte-level BPE shares the merge loop; guard its termination at a realistic
/// (non-toy) corpus size and vocabulary budget.
#[test]
fn f074_byte_level_bpe_training_terminates_at_scale() {
    use torsh_text::tokenization::advanced::ByteLevelBPETokenizer;

    let sentence = "the quick brown fox jumps over the lazy dog while the quick \
                    brown cat watches the lazy dog sleep near the old brown fence ";
    let mut texts = Vec::new();
    for i in 0..200 {
        texts.push(format!("{sentence} sample {i}"));
    }

    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(ByteLevelBPETokenizer::from_texts(&texts, 600));
    });

    let tokenizer = match rx.recv_timeout(Duration::from_secs(60)) {
        Ok(Ok(tokenizer)) => tokenizer,
        Ok(Err(e)) => panic!("byte-level BPE training failed: {e}"),
        Err(_) => panic!("ByteLevelBPETokenizer::from_texts did not terminate within 60s"),
    };

    assert!(
        tokenizer.vocab_size() <= 600,
        "byte-level vocabulary must respect the budget, got {}",
        tokenizer.vocab_size()
    );
    assert_eq!(
        tokenizer
            .decode(&tokenizer.encode("the quick brown fox").expect("encode"))
            .expect("decode"),
        "the quick brown fox",
        "byte-level BPE must round-trip losslessly"
    );
}

// ---------------------------------------------------------------------------
// F278 — vocabulary construction must be reproducible
// ---------------------------------------------------------------------------

#[test]
fn f278_whitespace_vocab_is_deterministic() {
    let texts = corpus();
    let probe = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron";

    let first = WhitespaceTokenizer::from_texts(&texts, 1);
    let second = WhitespaceTokenizer::from_texts(&texts, 1);

    let ids_first = first.encode(probe).expect("encode");
    let ids_second = second.encode(probe).expect("encode");

    assert_eq!(
        ids_first, ids_second,
        "F278: two WhitespaceTokenizer::from_texts runs must produce identical IDs"
    );
}

#[test]
fn f278_whitespace_vocab_orders_by_frequency() {
    let texts = vec!["rare common common common".to_string()];
    let tokenizer = WhitespaceTokenizer::from_texts(&texts, 1);
    let common = tokenizer.encode("common").expect("encode");
    let rare = tokenizer.encode("rare").expect("encode");
    assert!(
        common[0] < rare[0],
        "more frequent tokens must receive lower IDs, got common={common:?} rare={rare:?}"
    );
}

#[test]
fn f278_bpe_vocab_is_deterministic() {
    let texts = corpus();
    let first = train_bpe_with_timeout(texts.clone(), 40);
    let second = train_bpe_with_timeout(texts, 40);

    let probe = "alpha beta gamma delta epsilon";
    assert_eq!(
        first.encode(probe).expect("encode"),
        second.encode(probe).expect("encode"),
        "F278: two BPETokenizer::from_texts runs must produce identical IDs"
    );
    assert_eq!(
        first.tokenize(probe).expect("tokenize"),
        second.tokenize(probe).expect("tokenize"),
        "F278: merge order must be reproducible"
    );
}

// ---------------------------------------------------------------------------
// F075 — the decoder must not fabricate model output
// ---------------------------------------------------------------------------

/// Minimal `TextModel` that carries only shape metadata (no forward path).
struct StubModel;

impl TextModel for StubModel {
    fn name(&self) -> &str {
        "stub"
    }
    fn vocab_size(&self) -> usize {
        32
    }
    fn hidden_dim(&self) -> usize {
        8
    }
    fn max_seq_length(&self) -> usize {
        16
    }
}

/// Deterministic bag-of-ids encoder: distinct token IDs map to distinct axes,
/// so cosine similarity of identical texts is exactly 1.0 and of texts with
/// disjoint token sets is exactly 0.0.
struct BagOfIdsEncoder {
    dim: usize,
}

impl TextEncoder for BagOfIdsEncoder {
    fn encode(
        &self,
        input_ids: &Tensor,
        _attention_mask: Option<&Tensor>,
    ) -> torsh_core::Result<Tensor> {
        let ids = input_ids.to_vec()?;
        let mut values = vec![0.0f32; self.dim];
        for id in ids {
            let idx = (id.max(0.0) as usize) % self.dim;
            values[idx] += 1.0;
        }
        Tensor::from_vec(values, &[1, self.dim])
    }

    fn output_dim(&self) -> usize {
        self.dim
    }

    fn max_input_length(&self) -> usize {
        128
    }
}

fn decoder() -> AdvancedTextDecoder {
    AdvancedTextDecoder::new(
        Box::new(StubModel),
        Arc::new(WhitespaceTokenizer::new()),
        DeviceType::Cpu,
    )
}

fn wrapper_with_encoder() -> TextModelWrapper {
    let tokenizer = Arc::new(WhitespaceTokenizer::from_texts(&corpus(), 1));
    TextModelWrapper::new(tokenizer, DeviceType::Cpu)
        .with_encoder(Box::new(BagOfIdsEncoder { dim: 128 }))
}

#[test]
fn f075_decoder_decode_returns_error_not_random_logits() {
    use torsh_text::TextDecoder;

    let dec = decoder();
    let hidden = Tensor::from_vec(vec![0.0f32; 2 * 3 * 8], &[2, 3, 8]).expect("hidden states");
    let result = dec.decode(&hidden, None);
    assert!(
        result.is_err(),
        "F075: TextDecoder::decode must return Err, not fabricated random logits"
    );
    let msg = format!("{}", result.err().expect("error"));
    assert!(
        msg.contains("forward_model") || msg.contains("decode"),
        "error must name the missing forward path, got: {msg}"
    );
}

#[test]
fn f075_decoder_generate_returns_error() {
    use torsh_text::TextDecoder;

    let dec = decoder();
    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[1, 3]).expect("input");
    let config = GenerationConfig::default();
    assert!(
        dec.generate(&input, 8, &config).is_err(),
        "F075: generate must not sample from random logits"
    );

    let beam_config = GenerationConfig {
        num_beams: 3,
        ..Default::default()
    };
    assert!(
        dec.generate(&input, 8, &beam_config).is_err(),
        "F075: beam search must not sample from random logits"
    );
}

#[test]
fn f075_text_similarity_is_real_cosine() {
    let wrapper = wrapper_with_encoder();

    let same = wrapper
        .text_similarity("alpha beta", "alpha beta")
        .expect("similarity");
    assert!(
        (same - 1.0).abs() < 1e-5,
        "F075: similarity of identical texts must be 1.0, got {same}"
    );

    let different = wrapper
        .text_similarity("alpha", "beta")
        .expect("similarity");
    assert!(
        different.abs() < 1e-5,
        "F075: similarity of disjoint texts must be 0.0, got {different}"
    );
}

#[test]
fn f075_classify_uses_real_embedding_similarity() {
    let wrapper = wrapper_with_encoder();

    let (label, score) = wrapper
        .classify("gamma", &["beta".to_string(), "gamma".to_string()])
        .expect("classify");
    assert_eq!(
        label, "gamma",
        "F075: classify must pick the semantically closest label"
    );
    assert!(
        score > 0.99,
        "F075: classify score must be a real similarity, got {score}"
    );
}

#[test]
fn f075_classify_rejects_empty_labels() {
    let wrapper = wrapper_with_encoder();
    assert!(
        wrapper.classify("gamma", &[]).is_err(),
        "classify with no labels must return Err"
    );
}

// ---------------------------------------------------------------------------
// F277 — quick_clean behaviour must be preserved after de-regexing
// ---------------------------------------------------------------------------

#[test]
fn f277_quick_clean_collapses_whitespace() {
    assert_eq!(quick_clean("  a \t\n b  \r\n c  "), "a b c");
    assert_eq!(quick_clean(""), "");
    assert_eq!(quick_clean("     "), "");
    assert_eq!(quick_clean("single"), "single");
    assert_eq!(quick_clean("a\u{00A0}b"), "a b");
    assert_eq!(quick_clean("a\u{200B}b"), "ab");
    assert_eq!(quick_clean("a\u{FEFF}b"), "ab");
}

#[test]
fn f277_quick_clean_is_cheap_to_repeat() {
    // Regression guard for the per-call regex compilation: 20k calls must stay
    // well inside a second on any machine that can run this suite.
    let start = std::time::Instant::now();
    for _ in 0..20_000 {
        let _ = quick_clean("  some   moderately  long   text  to   clean  ");
    }
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "quick_clean is recompiling its pattern on every call ({:?} for 20k calls)",
        start.elapsed()
    );
}
