//! Hardening tests for torsh-cluster production-hardening findings.
//!
//! Each test is a minimal reproducer for one assigned finding. See the
//! campaign brief for finding IDs and full descriptions.

use torsh_cluster::{ClusteringResult, Fit, KMeans, KMeansAlgorithm, OnlineKMeansResult};
use torsh_tensor::Tensor;

/// F196: `ClusteringResult::labels()` must not panic when a concrete
/// result's labels are genuinely absent (e.g. an online-clustering result
/// before any batch has been processed) -- it must report that honestly
/// via `Option`.
#[test]
fn f196_online_kmeans_labels_returns_none_instead_of_panicking() {
    let centroids = Tensor::from_vec(vec![0.0f32, 0.0, 1.0, 1.0], &[2, 2])
        .expect("centroids tensor should build");
    let result = OnlineKMeansResult {
        centroids,
        labels: None,
        cluster_counts: vec![0, 0],
        n_points_seen: 0,
        current_learning_rate: 0.1,
        drift_detected: false,
        avg_intra_cluster_distance: 0.0,
    };

    // Going through the trait (not an inherent method) is the crux of the
    // bug: the trait signature must be honest about labels being optional.
    let labels = ClusteringResult::labels(&result);
    assert!(
        labels.is_none(),
        "labels should be None, not a panic, when no batch has been processed"
    );
}

/// F197: an empty cluster's centroid must be reseeded (e.g. to the point
/// farthest from its assigned centroid), not silently left at the origin.
///
/// Using data where every point has the *same* coordinates forces every
/// initialization strategy (Forgy, k-means++, random partition) to pick
/// coincident initial centroids; the deterministic `dist < min_dist` tie
/// break in the assignment step then routes every point to cluster 0,
/// guaranteeing clusters 1 and 2 are empty after the first iteration --
/// regardless of `random_state`. The data value (5.0, 5.0) is chosen to be
/// far from the origin so "still at the origin" and "reseeded to the data"
/// are unambiguously distinguishable.
#[test]
fn f197_kmeans_empty_cluster_is_not_left_at_origin() {
    let data = Tensor::from_vec(vec![5.0f32; 12], &[6, 2]).expect("data tensor should build");

    let kmeans = KMeans::new(3)
        .max_iters(5)
        .n_init(1)
        .random_state(42)
        .algorithm(KMeansAlgorithm::Lloyd);
    let result = kmeans.fit(&data).expect("fit should succeed");

    let centroids = result
        .centroids
        .to_vec()
        .expect("centroids should be readable");
    assert_eq!(centroids.len(), 6, "3 clusters x 2 features");

    // Every centroid row must equal the data value (5.0, 5.0), never the
    // origin -- including whichever rows ended up empty this run.
    for k in 0..3 {
        let (x, y) = (centroids[k * 2], centroids[k * 2 + 1]);
        assert!(
            (x - 5.0).abs() < 1e-4 && (y - 5.0).abs() < 1e-4,
            "cluster {k} centroid ({x}, {y}) should have been reseeded to (5.0, 5.0), not left at the origin"
        );
    }

    // With identical data every initialization degenerates to coincident
    // centroids, so at least two of the three clusters are empty every
    // iteration and must be reseeded at least once.
    assert!(
        result.n_empty_cluster_reseeds > 0,
        "expected at least one empty-cluster reseed to be recorded"
    );
}
