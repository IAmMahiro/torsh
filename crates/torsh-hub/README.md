# torsh-hub

Model hub integration for ToRSh, providing easy access to pre-trained models and datasets.

## Overview

ToRSh Hub is a repository of pre-trained models that facilitates research reproducibility. It provides:

- **Model Repository**: Access to pre-trained models
- **Easy Loading**: Simple API to load models with weights
- **Model Cards**: Detailed information about each model
- **Version Control**: Model versioning and updates
- **Caching**: Local caching for offline use

## Usage

### Loading Pre-trained Models

```rust
use torsh_hub::{load, HubConfig, load_state_dict_from_url};

// Load a model from a GitHub repository (PyTorch Hub-style hubconf discovery)
let model = load("pytorch/vision", "resnet18", true, None)?;

// Load with an explicit HubConfig (cache dir, retries, auth token, force_reload, ...)
let config = HubConfig {
    force_reload: true,
    ..Default::default()
};
let model = load("pytorch/vision", "resnet18", true, Some(config))?;

// Load a raw state dict from a URL into the hub cache
let state_dict = load_state_dict_from_url(
    "https://example.com/models/resnet50.pt",
    None,  // model_dir: defaults to the hub cache dir
    None,  // map_location: Option<torsh_core::DeviceType>
    true,  // show a progress bar
)?;
```

Note: there is no `hub::load_with_config`/`LoadConfig` in the current API — `load()` takes
an `Option<HubConfig>` directly, and `HubConfig` is a plain struct (no fluent
`.revision()`/`.map_location()` builder methods).

### Model Discovery

```rust
use torsh_hub::registry::{ModelRegistry, SearchQuery, ModelCategory};

let mut registry = ModelRegistry::new("./models")?;

let mut query = SearchQuery::default();
query.category = Some(ModelCategory::Vision);
query.tags = vec!["image-classification".to_string()];
query.limit = 10;

let results = registry.search(&query);
for entry in results {
    println!("{}: {} (downloads: {})", entry.name, entry.author, entry.downloads);
}
```

Note: there is no free-standing `hub::list_models`/`ModelFilter`/`hub::get_model_info` in the
current API; model discovery goes through `registry::ModelRegistry` and `SearchQuery` as shown
above.

### Publishing Models

```rust
use torsh_hub::model_info::{ModelCardBuilder, Version};
use torsh_hub::upload::{upload_model, UploadConfig};

// Fluent model card construction (fields currently supported by ModelCardBuilder)
let model_card = ModelCardBuilder::new()
    .developed_by("myusername".to_string())
    .model_type("image-classification".to_string())
    .architecture("resnet50".to_string())
    .build();

// Package + hash the model file and "upload" it.
// NOTE: the network transfer step (`perform_upload` in upload.rs) currently only
// simulates the upload and returns a placeholder URL — no bytes are actually sent
// to a remote server yet.
let upload_url = upload_model(
    std::path::Path::new("model.torsh"),
    "myusername/my-model",
    "my-model",
    &Version::new(1, 0, 0),
    model_info,
    UploadConfig::default(),
)?;
```

Note: there is no `torsh_hub::publish` module, `hub::publish()` function, or
`ModelCard::new().name(...)` fluent chain in the current API — `ModelCard::new()` takes
`(name, author, version)` directly; use `ModelCardBuilder` and `upload::upload_model`/
`upload_model_with_versioning` as shown above instead.

### Custom Model Registry

```rust
use torsh_hub::registry::{ModelRegistry, create_registry_entry};

let mut registry = ModelRegistry::new("./models")?;

// The registry is data-driven: there is no `#[derive(ModelRegistry)]` macro or a
// generic `hub::register_model::<T>()` in the current API. Instead, describe the
// model as a `RegistryEntry` and register that:
let entry = create_registry_entry(
    "custom_resnet".to_string(),
    "username".to_string(),
    "username/custom_resnet".to_string(),
    "A custom ResNet variant".to_string(),
);
registry.register_model(entry)?;
```

### Model Configuration

```rust
use torsh_hub::models::vision::ResNet;

// Direct construction: num_classes is passed to the constructor
let model = ResNet::resnet50(100);
```

Note: there is no `torsh_hub::config` module, `AutoConfig`/`AutoModel` (Hugging Face
`transformers`-style config-driven loading), or `ResNetConfig` type in the current API.
Each architecture instead exposes its own constructor (e.g. `ResNet::resnet50(num_classes)`,
`BertEncoder::bert_base(vocab_size)`).

### Caching and Offline Mode

```rust
use torsh_hub::{set_dir, load};

// Set cache directory
set_dir("~/my_cache/torsh/hub")?;

// load() downloads into the cache directory on first use and reuses the cache
// afterward (unless HubConfig::force_reload is set)
let model = load("pytorch/vision", "resnet18", true, None)?;
```

Note: there is currently no standalone `hub::download()` helper or a
`hub::set_offline_mode()`/`TORSH_HUB_OFFLINE` toggle in the codebase — caching is implicit
in `load()`.

### Integration with HuggingFace Hub

```rust
use torsh_hub::huggingface::HuggingFaceHub;

let hf = HuggingFaceHub::new();
let info = hf.model_info("microsoft/resnet-50")?;
let local_path = hf.download_model("microsoft/resnet-50", Some("main"))?;
```

Note: there is no `hub::from_huggingface()`/`hub::push_to_huggingface()` in the current API.
`HuggingFaceHub::load_torsh_model()` (auto-conversion into a ToRSh `Module`) only recognizes
`bert`/`gpt2`/`bart`/`t5` config types, and all four of those conversions currently return
`TorshError::NotImplemented(...)` — so end-to-end HuggingFace-to-ToRSh conversion does not
work yet for any model. `HuggingFaceHub::upload_model()` (pushing a model back to the Hub) is
likewise unimplemented and always returns
`TorshError::NotImplemented("Model upload not yet implemented")`.

### Model Versioning

```rust
use torsh_hub::model_info::{Version, VersionHistory};

let v1 = Version::new(1, 0, 0);
let v2 = Version::new(2, 0, 0);
assert!(v2 > v1);

let mut history = VersionHistory::new(v1.clone(), "cooljapan".to_string());
```

Note: `load()` repository strings must be exactly `owner/repo` or a GitHub URL — there is no
`owner/repo:tag` colon syntax for selecting a version. There is also no free-standing
`hub::update_model()` that pushes new weights; `ModelRegistry::update_model()` only updates an
existing `RegistryEntry`'s metadata, not the model file itself.

### Model Zoo

```rust
// Vision (also available via the generic GitHub-hubconf loader as `torsh_hub::sources::*`)
let resnet = torsh_hub::models::vision::pretrained::resnet50(true)?;
let efficientnet = torsh_hub::models::vision::pretrained::efficientnet_b0(true)?;
let vit = torsh_hub::models::vision::pretrained::vit_base_patch16_224(true)?;

// NLP
let bert = torsh_hub::models::nlp::pretrained::bert_base_uncased(true)?;
let gpt2 = torsh_hub::models::nlp::pretrained::gpt2_small(true)?;

// Audio / multimodal (no `pretrained` flag yet)
let wav2vec = torsh_hub::models::audio::wav2vec2_base();
let clip = torsh_hub::models::multimodal::clip_vit_b32();
```

Note: there is no `torsh_hub::zoo` module. More importantly, passing `pretrained: true` to
every factory function above currently just prints a warning
("pretrained weights not implemented yet") and returns a randomly-initialized model — no real
pretrained weights are downloaded for any model-zoo architecture yet.

### Security and Verification

```rust
use torsh_hub::security::{SecurityManager, SecurityConfig};

let manager = SecurityManager::new();
let signature = manager.sign_model("resnet50.torsh", "key-id", None)?;
let is_valid = manager.verify_model("resnet50.torsh", &signature, true)?;

let config = SecurityConfig {
    require_signatures: true,
    ..Default::default()
};
```

Note: there is no free `hub::verify_model()`/`hub::load_secure()`, and `SecurityConfig` has no
`.require_signature()`/`.allowed_ops()`/`.max_file_size()` builder methods — use
`SecurityManager` and the `SecurityConfig { require_signatures, .. }` struct as shown above.
Also, the RSA/Ed25519/ECDSA signing and verification behind
`SecurityManager::sign_model`/`verify_model` (in `security.rs`) are currently placeholder
implementations: they produce and compare fixed strings such as `"rsa_signature_placeholder"`
rather than performing real cryptographic operations.

## Implementation Notes

- **Streaming archive extraction**: `.tar.gz` model archives are extracted using `TarStreamReader<GzipStreamDecoder<File>>` (v0.1.2), requiring only O(512 B) memory per entry instead of O(archive size). No temporary buffering of the full archive.
- **Model-zoo pretrained weights are not implemented**: every `pretrained: true` factory function in `models::vision::pretrained`, `models::nlp::pretrained`, and the top-level `sources` module currently logs a warning and returns a randomly-initialized model; no real pretrained weights are downloaded yet. Real weights can only be loaded today via `load_state_dict_from_url` / applying a `StateDict` yourself, or through a GitHub `hubconf`-style repo passed to `load()`.
- **Cryptographic signing is a placeholder**: `SecurityManager::sign_model`/`verify_model` (`security.rs`) dispatch to RSA/Ed25519/ECDSA helper functions that return and compare fixed strings (e.g. `"rsa_signature_placeholder"`) instead of performing real signing/verification.
- **HuggingFace conversion and upload are unimplemented**: `HuggingFaceHub::load_torsh_model` only handles `bert`/`gpt2`/`bart`/`t5` config types, and all four conversions, plus `HuggingFaceHub::upload_model`, currently return `TorshError::NotImplemented(...)`.
- **Publishing performs a simulated upload**: `upload::upload_model`/`upload_model_with_versioning` package and hash the model file locally, but the final network transfer (`perform_upload`) is simulated and returns a placeholder URL rather than contacting a real server.

## Environment Variables

- `TORSH_HUB_DIR`: Override default cache directory
- `TORSH_HUB_TOKEN`: Authentication token for private models

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../../LICENSE) for details.