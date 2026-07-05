# torsh-cluster

Unsupervised learning and clustering algorithms for ToRSh, powered by SciRS2.

## Overview

This crate provides comprehensive clustering and unsupervised learning algorithms with a PyTorch-compatible API. It leverages `scirs2-cluster` for high-performance implementations while maintaining full integration with ToRSh's tensor operations (`torsh-tensor`).

## Features

- **Partitioning Methods**: K-Means (Lloyd, Elkan, and Mini-batch variants)
- **Hierarchical Clustering**: Agglomerative clustering with multiple linkage methods (Ward, Complete, Average, Single)
- **Density-Based Methods**: DBSCAN, HDBSCAN, OPTICS
- **Distribution-Based**: Gaussian Mixture Models (GMM) with Full/Diagonal/Spherical covariance, EM algorithm
- **Spectral Methods**: Spectral clustering with RBF/k-NN affinity
- **Streaming/Incremental**: Online K-Means and sliding-window K-Means with concept drift detection
- **Evaluation Metrics**: Silhouette, Davies-Bouldin, Calinski-Harabasz, Dunn, Xie-Beni, ARI, AMI, NMI, Fowlkes-Mallows, homogeneity/completeness/V-measure
- **Initialization Strategies**: K-means++, Forgy, Random Partition
- **Gap Statistic**: Automatic selection of the optimal number of clusters

GPU-accelerated clustering is not yet implemented (tracked in `TODO.md`); all algorithms currently run on CPU with SciRS2-backed SIMD/parallel acceleration.

## Usage

### K-Means Clustering

```rust
use torsh_cluster::prelude::*;
use torsh_tensor::prelude::*;

// Create sample data
let data = tensor![[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]];

// Initialize K-Means with 2 clusters
let kmeans = KMeans::new(2)
    .max_iters(100)
    .tolerance(1e-4)
    .init_method(InitMethod::KMeansPlusPlus)
    .random_state(42);

// Fit the model
let result = kmeans.fit(&data)?;

// Get cluster assignments
let labels = result.labels();
println!("Cluster assignments: {:?}", labels);

// Get cluster centers
let centers = result.centers();
println!("Cluster centers: {:?}", centers);

// Predict cluster for new data
let new_point = tensor![[2.0, 3.0]];
let predicted_cluster = kmeans.predict(&new_point)?;
```

### DBSCAN - Density-Based Clustering

```rust
use torsh_cluster::prelude::*;

let data = load_dataset("examples/clustering_data.csv")?;

// Initialize DBSCAN
let dbscan = DBSCAN::new(0.5, 5)  // eps=0.5, min_samples=5
    .metric("euclidean");

// Fit and predict
let labels = dbscan.fit_predict(&data)?;

// Points labeled -1 are considered noise
let noise_points = labels.iter().filter(|&&x| x == -1).count();
println!("Number of noise points: {}", noise_points);

// Get core samples
let core_samples = dbscan.core_sample_indices()?;
```

### Gaussian Mixture Models (GMM)

```rust
use torsh_cluster::prelude::*;

let data = generate_blobs(1000, 3, 5, 1.0, 42)?;  // 1000 samples, 3 features, 5 clusters

// Initialize GMM with 5 components
let gmm = GaussianMixture::new(5)
    .covariance_type(CovarianceType::Full)
    .max_iters(100)
    .n_init(10)
    .init_params(InitParams::KMeans);

// Fit the model
gmm.fit(&data)?;

// Predict cluster probabilities
let probabilities = gmm.predict_proba(&data)?;
println!("Cluster probabilities shape: {:?}", probabilities.shape());

// Get model parameters
let means = gmm.means();
let covariances = gmm.covariances();
let weights = gmm.weights();

// Compute BIC and AIC
let bic = gmm.bic(&data)?;
let aic = gmm.aic(&data)?;
```

### Hierarchical Clustering

```rust
use torsh_cluster::prelude::*;

let data = tensor![[1.0, 2.0], [2.0, 3.0], [10.0, 11.0], [11.0, 12.0]];

// Agglomerative clustering
let hierarchical = AgglomerativeClustering::new(2)
    .linkage(Linkage::Ward)
    .affinity("euclidean");

let labels = hierarchical.fit_predict(&data)?;

// Get dendrogram
let dendrogram = hierarchical.dendrogram();

// Compute cophenetic correlation
let cophenetic_corr = hierarchical.cophenetic_correlation(&data)?;
```

### Spectral Clustering

```rust
use torsh_cluster::prelude::*;

let data = make_moons(200, Some(0.05), Some(42))?;  // Non-convex clusters

// Spectral clustering works well with non-convex shapes
let spectral = SpectralClustering::new(2)
    .affinity(Affinity::NearestNeighbors(10))
    .assign_labels("kmeans")
    .random_state(42);

let labels = spectral.fit_predict(&data)?;

// Use custom affinity matrix
let affinity_matrix = compute_rbf_kernel(&data, gamma=1.0)?;
let spectral_custom = SpectralClustering::new(2)
    .affinity(Affinity::Precomputed(affinity_matrix));

let labels = spectral_custom.fit_predict(&data)?;
```

### OPTICS - Ordering Points To Identify Clustering Structure

```rust
use torsh_cluster::prelude::*;

let data = load_dataset("complex_shapes.csv")?;

// OPTICS can find clusters of varying densities
let optics = OPTICS::new()
    .min_samples(5)
    .max_eps(f32::INFINITY)
    .metric("euclidean")
    .cluster_method("xi");

let labels = optics.fit_predict(&data)?;

// Get reachability plot
let reachability = optics.reachability();
let ordering = optics.ordering();

// Extract clusters with different parameters
let labels_dbscan = optics.extract_dbscan(eps=0.5)?;
```

## Evaluation Metrics

### Clustering Quality

```rust
use torsh_cluster::evaluation::*;

let data = generate_blobs(1000, 2, 3, 1.0, 42)?;
let labels = kmeans.fit_predict(&data)?;

// Silhouette score (-1 to 1, higher is better)
let silhouette = silhouette_score(&data, &labels, "euclidean")?;
println!("Silhouette score: {:.4}", silhouette);

// Davies-Bouldin score (lower is better)
let db_index = davies_bouldin_score(&data, &labels)?;
println!("Davies-Bouldin index: {:.4}", db_index);

// Calinski-Harabasz index (higher is better)
let ch_index = calinski_harabasz_score(&data, &labels)?;
println!("Calinski-Harabasz score: {:.4}", ch_index);

// Dunn index (higher is better)
let dunn = dunn_index(&data, &labels)?;
println!("Dunn index: {:.4}", dunn);
```

### External Validation (when ground truth is available)

```rust
use torsh_cluster::evaluation::*;

let true_labels = tensor![0, 0, 1, 1, 2, 2];
let pred_labels = tensor![0, 0, 1, 2, 2, 1];

// Adjusted Rand Index (-1 to 1, 1 is perfect)
let ari = adjusted_rand_score(&true_labels, &pred_labels)?;

// Normalized Mutual Information (0 to 1, 1 is perfect)
let nmi = normalized_mutual_info_score(&true_labels, &pred_labels)?;

// Fowlkes-Mallows score (0 to 1, 1 is perfect)
let fmi = fowlkes_mallows_score(&true_labels, &pred_labels)?;

// V-measure (0 to 1, 1 is perfect)
let v_measure = v_measure_score(&true_labels, &pred_labels)?;

// Homogeneity and completeness
let (homogeneity, completeness, v_measure) = homogeneity_completeness_v_measure(&true_labels, &pred_labels)?;
```

## Initialization Methods

Initialization strategies implement the shared `InitializationStrategy` trait:

```rust
use torsh_cluster::initialization::{Forgy, InitializationStrategy, KMeansPlusPlus, RandomPartition};
use torsh_tensor::creation::randn;

let data = randn::<f32>(&[1000, 10])?;
let n_clusters = 5;

// K-means++ initialization (smart initialization)
let centers = KMeansPlusPlus.initialize(&data, n_clusters, Some(42))?;

// Forgy (random sample points as centers)
let centers = Forgy.initialize(&data, n_clusters, Some(42))?;

// Random partition
let centers = RandomPartition.initialize(&data, n_clusters, Some(42))?;
```

`KMeans` uses `InitMethod` (see the K-Means example above) to select one of these strategies internally.

## Advanced Features

### Mini-Batch K-Means for Large Datasets

```rust
use torsh_cluster::prelude::*;

let large_data = randn(&[1_000_000, 100])?;  // 1M samples

// Select the mini-batch variant of the K-Means algorithm for scalability
let mb_kmeans = KMeans::new(100)  // 100 clusters
    .algorithm(KMeansAlgorithm::MiniBatch)
    .max_iters(100);

let result = mb_kmeans.fit_predict(&large_data)?;
```

## Utilities

### Gap Statistic for Optimal K

```rust
use torsh_cluster::evaluation::*;

let data = generate_blobs(1000, 10, 5, 1.0, 42)?;

// Compare within-cluster dispersion against reference datasets to pick k
let mut gap_stat = GapStatistic::with_default().max_k(10).n_refs(10);
let result = gap_stat.compute(&data)?;
println!("Optimal number of clusters: {}", result.optimal_k);
```

## Integration with SciRS2

This crate leverages the SciRS2 ecosystem for:

- High-performance clustering algorithms through `scirs2-cluster`
- Optimized tensor operations via `scirs2-core`
- Statistical functions from `scirs2-stats`
- Evaluation metrics through `scirs2-metrics`
- Linear algebra operations via `scirs2-linalg`

All implementations follow the [SciRS2 POLICY](https://github.com/cool-japan/scirs/blob/master/SCIRS2_POLICY.md) for consistent APIs and optimal performance.

## Examples

See the `examples/` directory for more detailed examples:

- `comprehensive_clustering_demo.rs` - Overview of the core clustering algorithms
- `image_segmentation.rs` - Image segmentation with clustering
- `customer_segmentation.rs` - Customer segmentation use case
- `time_series_clustering.rs` - Clustering time-series data
- `adaptive_streaming_clustering.rs` - Streaming clustering with adaptive parameters and drift detection

## Performance Tips

1. **Use Mini-Batch K-Means** (`KMeansAlgorithm::MiniBatch`) for datasets with >100k samples
2. **Use K-means++ initialization** for better convergence
3. **Apply PCA** for dimensionality reduction before clustering high-dimensional data
4. Clustering automatically parallelizes for larger datasets (e.g. K-Means for n >= 1000) via `scirs2_core::parallel_ops`; GPU acceleration is not yet implemented

## Testing

This crate has 131 passing tests (`cargo nextest run -p torsh-cluster --all-features`).

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../../LICENSE) for details.
