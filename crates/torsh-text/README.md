# torsh-text

Text processing and NLP utilities for ToRSh, leveraging scirs2-text for efficient text operations.

## Overview

This crate provides comprehensive text processing capabilities:

- **Tokenization**: Various tokenization strategies (word, subword, character)
- **Vocabulary Management**: Efficient vocabulary building and management
- **Text Datasets**: Common NLP datasets and data loaders
- **Preprocessing**: Text normalization, cleaning, and augmentation
- **Embeddings**: Word embeddings and embedding layers
- **Utilities**: Text generation, metrics, and analysis tools

Note: This crate integrates with scirs2-text for optimized text processing operations.

## Usage

### Tokenization

```rust
use torsh_text::prelude::*;

// All tokenizers implement the same `Tokenizer` trait:
// tokenize(&str) -> Vec<String>, encode(&str) -> Vec<u32>, decode(&[u32]) -> String
let tokenizer = WhitespaceTokenizer::new();
let text = "Hello, World! This is a test.";
let tokens = tokenizer.tokenize(text)?;

// BPE tokenizer (trained from texts, not loaded from a vocab.json/merges.txt pair)
let bpe = BPETokenizer::from_texts(&training_texts, 8000)?;
let ids = bpe.encode("Hello, world!")?;
let decoded = bpe.decode(&ids)?;
```

Note: there is no `BasicTokenizer`, `WordPieceTokenizer`, or `SentencePieceTokenizer` in
this crate today — real tokenizer types are `WhitespaceTokenizer`, `CharTokenizer`,
`SubwordTokenizer`, `BPETokenizer`, `ByteLevelBPETokenizer`, and `UnigramTokenizer`.

### Vocabulary Management

```rust
use torsh_text::vocab::*;

// Build vocabulary from corpus texts (the real type is `Vocabulary`, not `Vocab`)
let corpus: Vec<String> = vec![
    "This is a sentence".to_string(),
    "This is another sentence".to_string(),
    "And one more".to_string(),
];

let vocab = Vocabulary::from_texts(&corpus, /* min_freq */ 1, /* max_size */ Some(10000));

// Convert between tokens and ids (method names are tokens_to_ids / ids_to_tokens)
let ids = vocab.tokens_to_ids(&["This".to_string(), "is".to_string(), "a".to_string()]);
let tokens = vocab.ids_to_tokens(&ids);

// Save and load vocabulary (plain-text format)
vocab.save_txt("vocab.txt")?;
let loaded = Vocabulary::load_txt("vocab.txt")?;
```

### Text Datasets

```rust
use torsh_text::datasets::*;

// IMDB sentiment dataset (real type is `ImdbDataset`; no built-in download)
let imdb = ImdbDataset::from_directory("./data", DatasetSplit::Train)?;
println!("{} positive, {} negative", imdb.num_positive(), imdb.num_negative());

// Text classification dataset (real type is `ClassificationDataset`; its
// `from_csv` expects a specific "class_idx,title,description" CSV layout,
// not arbitrary text/label columns)
let dataset = ClassificationDataset::from_csv("data.csv", DatasetSplit::Train)?;

// Language modeling dataset — plain text, split into fixed-length windows
// (there is no tokenizer/block_size keyword API; tokenization is a separate step)
let lm_dataset = LanguageModelingDataset::from_file("corpus.txt", 128, None)?;

// Translation dataset — parallel text files, one sentence per line
let translation = TranslationDataset::from_files(
    "train.en", "train.de", "en".to_string(), "de".to_string(),
)?;
```

### Text Preprocessing

```rust
use torsh_text::preprocessing::*;

// Text normalization
let normalizer = TextNormalizer::new()
    .lowercase(true)
    .remove_punctuation(true)
    .normalize_unicode(true)
    .expand_contractions(true);

let normalized = normalizer.normalize("Don't forget the café!")?;
// "do not forget the cafe"

// Text augmentation
let augmenter = TextAugmenter::new()
    .add_synonym_replacement(0.1)
    .add_random_insertion(0.1)
    .add_random_swap(0.1)
    .add_random_deletion(0.1);

let augmented = augmenter.augment("The quick brown fox")?;

// Padding and truncation (operates on one token-id sequence at a time)
let padded = pad_sequence(&tokens, 100, 0, PaddingStrategy::Right);
```

### Embeddings

```rust
use torsh_text::embeddings::*;

// There is no GloVe / pre-trained-vector loader in this crate today.
// Embedding is torsh-nn's layer type: Embedding::new(vocab_size, embedding_dim)
// (no padding_idx / from_pretrained here).
let embedding = torsh_nn::Embedding::new(vocab.len(), 300);

// Contextual embeddings — BertEmbeddings takes a config struct + device
let config = TextModelConfig::new(30522, 768, 12, 12)?; // vocab, hidden, layers, heads
let bert_embeddings = BertEmbeddings::new(&config, DeviceType::Cpu)?;
```

### Text Generation

```rust
use torsh_text::generation::*;

// Text generation utilities
let generator = TextGenerator::new(model)
    .temperature(0.8)
    .top_k(50)
    .top_p(0.95)
    .repetition_penalty(1.2);

let generated = generator.generate(
    prompt="Once upon a time",
    max_length=100,
    num_return_sequences=3,
)?;

// Beam search
let beam_output = generator.beam_search(
    input_ids,
    beam_size=5,
    max_length=50,
    length_penalty=0.6,
)?;

// Sampling strategies
let sampled = generator.sample(
    input_ids,
    do_sample=true,
    temperature=0.9,
    top_k=40,
)?;
```

### Text Analysis

```rust
use torsh_text::metrics::*;

// BLEU score (scorer object, not a calculate_bleu() free function)
let bleu = BleuScore::new().with_smoothing(true);
let score = bleu.calculate(&hypothesis, &references)?;

// ROUGE score
let rouge = RougeScore::new(RougeType::RougeL);
let rouge_result = rouge.calculate(&hypothesis, &reference)?;

// Perplexity (metrics::perplexity module, from probabilities or logits)
let ppl = metrics::perplexity::calculate(&probabilities)?;

// Text statistics (built from sentences, not a raw corpus)
let stats = TextStatistics::from_sentences(&sentences);
```

### Integration with Models

```rust
use torsh_text::models::*;
use torsh_nn::prelude::*;

// Text classification model
struct TextClassifier {
    embedding: Embedding,
    encoder: LSTM,
    classifier: Linear,
}

impl TextClassifier {
    fn new(vocab_size: usize, num_classes: usize) -> Self {
        Self {
            embedding: Embedding::new(vocab_size, 128, Some(0)),
            encoder: LSTM::new(128, 256, 2, true, true, 0.1, false),
            classifier: Linear::new(512, num_classes, true), // bidirectional
        }
    }
}

// Sequence-to-sequence model
struct Seq2Seq {
    encoder: Encoder,
    decoder: Decoder,
    attention: Attention,
}

// Transformer model
let transformer = TransformerModel::new(
    vocab_size=vocab.size(),
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
)?;
```

### Utilities

```rust
// N-gram extraction (returns frequency counts, not a plain list)
let ngrams = scirs2_ops::ngram_frequency(text, 3)?;

// TF-IDF
let tfidf = TfIdf::fit(&documents)?;
let tfidf_matrix = tfidf.transform(&documents)?;

// Text similarity
let sim = cosine_similarity(&text1_embedding, &text2_embedding)?;

// Sentence splitting
let sentences = split_sentences(text)?;

// Language detection
let language = detect_language(text)?;
```

## Integration with SciRS2

This crate leverages scirs2-text for:
- Efficient string operations
- Optimized tokenization algorithms
- Fast vocabulary lookups
- Vectorized text processing

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../../LICENSE) for details.