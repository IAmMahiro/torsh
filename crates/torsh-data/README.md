# torsh-data

Data loading and preprocessing framework for ToRSh with PyTorch-compatible API.

## Overview

This crate provides comprehensive data handling utilities including:

- **Datasets**: Abstract interfaces and common implementations
- **DataLoader**: Efficient multi-threaded data loading with batching
- **Samplers**: Various sampling strategies for data iteration
- **Transformations**: Common data preprocessing operations
- **Domain-specific support**: Vision, audio, and tabular data

Note: This crate can leverage scirs2-datasets for additional dataset utilities and sample datasets.

## Features

### Core Features
- `std` (default): Standard library support
- `image-support` (default): Enable image loading and vision datasets
- `mmap-support` (default): Enable memory-mapped file support for large datasets

### Data Format Support
- `audio-support`: Enable audio processing capabilities
- `dataframe`: Enable tabular data support with Polars integration
- `arrow-support`: Apache Arrow integration for efficient data interchange
- `hdf5-support`: HDF5 file format support for scientific datasets
- `parquet-support`: Apache Parquet columnar storage format

### Advanced Features
- `sparse`: Sparse tensor support for efficient handling of sparse data
- `serialize`: Serialization support with serde
- `async-support`: Asynchronous data loading with tokio
- `privacy`: Privacy-preserving data loading with differential privacy
- `federated`: Federated learning support for distributed datasets

### GPU Acceleration (Experimental)
- `gpu-acceleration`: Enable GPU-accelerated data preprocessing
- `cuda`: CUDA backend feature flag (placeholder — backend handle is a null/mock context, no real GPU dispatch yet)
- `opencl`: OpenCL backend support (placeholder for future)
- `vulkan`: Vulkan backend support (placeholder for future)
- `metal`: Metal backend support for Apple Silicon (placeholder for future)
- `webgpu`: WebGPU backend support for web and cross-platform (placeholder for future)

### WebAssembly
- `wasm`: WebAssembly support for browser-based data loading

### Convenience
- `full`: Enable all optional features

## Usage

### Basic Dataset and DataLoader

```rust
use torsh_data::prelude::*;
use torsh_tensor::prelude::*;

// Create a simple tensor dataset
let data = tensor![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
let labels = tensor![0, 1, 0];
let dataset = TensorDataset::new(vec![data, labels]);

// Create a dataloader
let dataloader = DataLoader::builder(dataset)
    .batch_size(2)
    .shuffle(true)
    .num_workers(4)
    .build()?;

// Iterate through batches
for batch in dataloader {
    let inputs = &batch[0];
    let targets = &batch[1];
    // Process batch...
}
```

### Custom Dataset Implementation

```rust
use torsh_data::dataset::Dataset;
use torsh_tensor::Tensor;
use std::path::PathBuf;

struct ImageDataset {
    image_paths: Vec<PathBuf>,
    labels: Vec<i64>,
    transform: Option<Box<dyn Transform>>,
}

impl Dataset for ImageDataset {
    fn len(&self) -> usize {
        self.image_paths.len()
    }
    
    fn get(&self, index: usize) -> Result<Vec<Tensor>> {
        // Load image from path
        let image = load_image(&self.image_paths[index])?;
        
        // Apply transformations if any
        let image = if let Some(transform) = &self.transform {
            transform.transform(image)?
        } else {
            image
        };
        
        // Return image and label
        Ok(vec![image, tensor![self.labels[index]]])
    }
}
```

### Samplers

```rust
// Sequential sampling
let sampler = SequentialSampler::new(dataset.len());

// Random sampling with replacement (dataset_size, num_samples, replacement)
let sampler = RandomSampler::new(dataset.len(), Some(10000), true);

// Batch sampling (inner sampler, batch_size, drop_last)
let batch_sampler = BatchSampler::new(sampler, 32, false);
```

### Data Transformations

```rust
use torsh_data::core_framework::{Compose, Normalize};
use torsh_data::tensor_transforms::RandomCrop;

// Compose chains same-type tensor transforms (e.g. crop -> crop)
let transform = Compose::new(vec![
    Box::new(RandomCrop::new((224, 224))),
]);
let cropped = transform.transform(image)?;

// Normalize::new validates mean/std lengths and returns a Result
let normalize = Normalize::new(
    vec![0.485, 0.456, 0.406],
    vec![0.229, 0.224, 0.225],
)?;
let normalized = normalize.transform(cropped)?;
```

### Collate Functions

```rust
use torsh_data::collate::{collate_fn, PadCollate};

// Default collate function (stacks tensors)
let dataloader = DataLoader::builder(dataset)
    .collate_fn(collate_fn)
    .build()?;

// Custom collate for variable-length sequences (padding_value)
let pad_collate = PadCollate::new(0.0);
let dataloader = DataLoader::builder(dataset)
    .collate_fn(move |batch| pad_collate.collate(batch))
    .build()?;
```

### Vision Datasets (with `image-support` feature)

```rust
#[cfg(feature = "image-support")]
use torsh_data::vision::{ImageFolder, CIFAR10, MNIST};

// Load from directory structure
let dataset = ImageFolder::new("path/to/images")?
    .with_transform(transform);

// Built-in datasets (root, train) — no transform/download params yet
let mnist = MNIST::new("./data", true)?;
let cifar = CIFAR10::new("./data", true)?;
```

### Tabular Data (with `dataframe` feature)

```rust
#[cfg(feature = "dataframe")]
use torsh_data::tabular::CSVDataset;

// new(path, target_column, has_header) — reads via the `csv` crate (no Polars dependency)
let dataset = CSVDataset::new("data.csv", Some("label"), true)?;
```

### Audio Support (with `audio-support` feature)

```rust
#[cfg(feature = "audio-support")]
use torsh_data::audio::AudioFolder;

// Loads class-labeled audio samples from a directory structure (like ImageFolder)
let dataset = AudioFolder::new("path/to/audio", Some(16000))?;
```

### Multi-Processing and Performance

```rust
// Parallel data loading
let dataloader = DataLoader::builder(dataset)
    .batch_size(64)
    .num_workers(8)  // Parallel loading threads
    .persistent_workers(true)  // Keep workers alive
    .pin_memory(true)  // Pin memory for faster GPU transfer
    .build()?;
```

## Tabular Data Backend

The `dataframe` feature reads CSV files directly through the `csv` crate (COOLJAPAN policy — Polars is intentionally not a dependency). `CSVDataset` parses the file eagerly into in-memory `f32` feature/target vectors:

```rust
use torsh_data::tabular::CSVDataset;

let dataset = CSVDataset::new("data.csv", Some("label"), true)?;
println!("{} features: {:?}", dataset.num_features(), dataset.feature_names());
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../../LICENSE) for details.