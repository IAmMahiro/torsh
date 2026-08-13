//! Core sampling traits and utilities
//!
//! This module provides the fundamental building blocks for all sampling strategies.

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

// ✅ SciRS2 Policy Compliant - Using scirs2_core for all random operations
use scirs2_core::random::Random;

/// Common RNG utilities for samplers
pub(crate) mod rng_utils {
    use super::*;

    /// Create a seeded or default RNG
    pub fn create_rng(seed: Option<u64>) -> Random<scirs2_core::rngs::StdRng> {
        if let Some(seed) = seed {
            Random::seed(seed)
        } else {
            Random::seed(42) // Default seed for reproducible behavior
        }
    }

    /// Generic shuffle function
    pub fn shuffle_indices<T: Clone>(indices: &mut [T], seed: Option<u64>) {
        let mut rng = create_rng(seed);

        // Fisher-Yates shuffle
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }
    }

    /// Generate random range
    pub fn gen_range(
        rng: &mut Random<scirs2_core::rngs::StdRng>,
        range: std::ops::Range<usize>,
    ) -> usize {
        rng.gen_range(range)
    }
}

/// Unified trait for sampling from a dataset
///
/// This trait provides a consistent interface for both individual sample
/// and batch sampling strategies.
pub trait Sampler: Send {
    /// Iterator type returned by the sampler
    type Iter: Iterator<Item = usize> + Send;

    /// Create an iterator over indices
    fn iter(&self) -> Self::Iter;

    /// Total number of samples that will be yielded
    fn len(&self) -> usize;

    /// Check if sampler is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Convert this sampler into a batch sampler
    fn into_batch_sampler(
        self,
        batch_size: usize,
        drop_last: bool,
    ) -> super::batch::BatchingSampler<Self>
    where
        Self: Sized,
    {
        super::batch::BatchingSampler::new(self, batch_size, drop_last)
    }

    /// Create a distributed version of this sampler
    fn into_distributed(
        self,
        num_replicas: usize,
        rank: usize,
    ) -> super::distributed::DistributedWrapper<Self>
    where
        Self: Sized,
    {
        super::distributed::DistributedWrapper::new(self, num_replicas, rank)
    }
}

/// Trait for batch samplers that yield batches of indices
pub trait BatchSampler: Send {
    /// Iterator type returned by the batch sampler
    type Iter: Iterator<Item = Vec<usize>> + Send;

    /// Create an iterator over batches of indices
    fn iter(&self) -> Self::Iter;

    /// Total number of batches that will be yielded
    fn num_batches(&self) -> usize;

    /// Total number of batches that will be yielded (alias for num_batches)
    fn len(&self) -> usize {
        self.num_batches()
    }

    /// Check if batch sampler is empty
    fn is_empty(&self) -> bool {
        self.num_batches() == 0
    }
}

/// Iterator wrapper that provides additional functionality
pub struct SamplerIterator {
    indices: Vec<usize>,
    position: usize,
}

impl SamplerIterator {
    /// Create a new sampler iterator
    pub fn new(indices: Vec<usize>) -> Self {
        Self {
            indices,
            position: 0,
        }
    }

    /// Create from a range
    pub fn from_range(start: usize, end: usize) -> Self {
        Self::new((start..end).collect())
    }

    /// Create shuffled indices
    pub fn shuffled(mut indices: Vec<usize>, seed: Option<u64>) -> Self {
        // ✅ SciRS2 Policy Compliant - Using scirs2_core for random operations
        let mut rng = match seed {
            Some(s) => Random::seed(s),
            None => Random::seed(42), // Use fixed seed instead of Random::new()
        };

        // Fisher-Yates shuffle
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }

        Self::new(indices)
    }

    /// Get remaining items count
    pub fn remaining(&self) -> usize {
        self.indices.len() - self.position
    }
}

impl Iterator for SamplerIterator {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.indices.len() {
            let item = self.indices[self.position];
            self.position += 1;
            Some(item)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining();
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for SamplerIterator {
    fn len(&self) -> usize {
        self.remaining()
    }
}

/// Utility functions for sampling operations
pub mod utils {
    use super::*;

    /// Generate random indices without replacement
    ///
    /// # Panics
    ///
    /// Panics if `k > n`. See [`try_random_indices`] for a non-panicking variant.
    pub fn random_indices(n: usize, k: usize, seed: Option<u64>) -> Vec<usize> {
        assert!(k <= n, "Cannot sample more items than available");

        // ✅ SciRS2 Policy Compliant - Using scirs2_core for random operations
        let mut rng = match seed {
            Some(s) => Random::seed(s),
            None => Random::seed(42),
        };

        random_indices_from_rng(n, k, &mut rng)
    }

    /// Fallible variant of [`random_indices`] that returns an error instead of
    /// panicking when `k > n`.
    pub fn try_random_indices(
        n: usize,
        k: usize,
        seed: Option<u64>,
    ) -> torsh_core::error::Result<Vec<usize>> {
        if k > n {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Cannot sample {k} items without replacement from {n} available"
            )));
        }

        // ✅ SciRS2 Policy Compliant - Using scirs2_core for random operations
        let mut rng = match seed {
            Some(s) => Random::seed(s),
            None => Random::seed(42),
        };

        Ok(random_indices_from_rng(n, k, &mut rng))
    }

    /// Shared sampling core for [`random_indices`]/[`try_random_indices`], run
    /// once `k <= n` is already known to hold.
    fn random_indices_from_rng(
        n: usize,
        k: usize,
        rng: &mut Random<scirs2_core::rngs::StdRng>,
    ) -> Vec<usize> {
        if k == n {
            // Return all indices shuffled
            let mut indices: Vec<usize> = (0..n).collect();
            for i in (1..indices.len()).rev() {
                let j = rng.gen_range(0..=i);
                indices.swap(i, j);
            }
            indices
        } else if k <= n / 2 {
            // Use rejection sampling for small k, then shuffle the result: the
            // HashSet-derived order is canonicalized via sort_unstable() first
            // (HashSet iteration order is randomized per-process by RandomState
            // and must not leak into the sampled order), and only then is it
            // Fisher-Yates shuffled with the seeded RNG so the returned subset
            // comes back in random order rather than ascending order.
            let mut selected = std::collections::HashSet::new();
            while selected.len() < k {
                let idx = rng.gen_range(0..n);
                selected.insert(idx);
            }
            let mut result: Vec<usize> = selected.into_iter().collect();
            result.sort_unstable(); // Canonicalize HashSet iteration order first
            for i in (1..result.len()).rev() {
                let j = rng.gen_range(0..=i);
                result.swap(i, j);
            }
            result
        } else {
            // Use exclusion method for large k, then shuffle (same reasoning as above).
            let mut excluded = std::collections::HashSet::new();
            while excluded.len() < n - k {
                let idx = rng.gen_range(0..n);
                excluded.insert(idx);
            }
            let mut result: Vec<usize> = (0..n).filter(|&i| !excluded.contains(&i)).collect();
            result.sort_unstable(); // Canonicalize before shuffling
            for i in (1..result.len()).rev() {
                let j = rng.gen_range(0..=i);
                result.swap(i, j);
            }
            result
        }
    }

    /// Stratified split of indices
    pub fn stratified_split(
        indices: &[usize],
        labels: &[usize],
        test_ratio: f32,
        seed: Option<u64>,
    ) -> (Vec<usize>, Vec<usize>) {
        use std::collections::HashMap;

        // Group indices by label
        let mut label_groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for &idx in indices {
            if idx < labels.len() {
                label_groups
                    .entry(labels[idx])
                    .or_insert_with(Vec::new)
                    .push(idx);
            }
        }

        // ✅ SciRS2 Policy Compliant - Using scirs2_core for random operations
        let mut rng = match seed {
            Some(s) => Random::seed(s),
            None => Random::seed(42),
        };

        let mut train_indices = Vec::new();
        let mut test_indices = Vec::new();

        // Split each label group
        for (_, mut group_indices) in label_groups {
            // Shuffle group indices
            for i in (1..group_indices.len()).rev() {
                let j = rng.gen_range(0..=i);
                group_indices.swap(i, j);
            }

            let test_size = ((group_indices.len() as f32) * test_ratio).round() as usize;
            let test_size = test_size.min(group_indices.len());

            test_indices.extend(group_indices.iter().take(test_size));
            train_indices.extend(group_indices.iter().skip(test_size));
        }

        (train_indices, test_indices)
    }

    /// Calculate class weights for balanced sampling
    pub fn calculate_class_weights(labels: &[usize], num_classes: usize) -> Vec<f32> {
        let mut class_counts = vec![0usize; num_classes];

        // Count occurrences
        for &label in labels {
            if label < num_classes {
                class_counts[label] += 1;
            }
        }

        // Calculate weights (inverse frequency)
        let total_samples = labels.len() as f32;
        class_counts
            .iter()
            .map(|&count| {
                if count > 0 {
                    total_samples / (num_classes as f32 * count as f32)
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Validate sampling configuration
    pub fn validate_sampling_params(
        dataset_size: usize,
        num_samples: Option<usize>,
        replacement: bool,
    ) -> Result<usize, String> {
        let actual_num_samples = num_samples.unwrap_or(dataset_size);

        // Allow empty datasets (dataset_size == 0) with 0 samples
        if dataset_size == 0 {
            if actual_num_samples == 0 {
                return Ok(0);
            } else {
                return Err("Cannot sample from empty dataset".to_string());
            }
        }

        if !replacement && actual_num_samples > dataset_size {
            return Err(format!(
                "Cannot sample {} items without replacement from dataset of size {}",
                actual_num_samples, dataset_size
            ));
        }

        if actual_num_samples == 0 && !replacement {
            return Err(
                "Number of samples cannot be zero for non-empty dataset without replacement"
                    .to_string(),
            );
        }

        Ok(actual_num_samples)
    }

    /// Simple train-validation split
    pub fn train_val_split(
        dataset_size: usize,
        val_ratio: f32,
        seed: Option<u64>,
    ) -> (Vec<usize>, Vec<usize>) {
        let val_size = (dataset_size as f32 * val_ratio).round() as usize;
        let indices = random_indices(dataset_size, dataset_size, seed);

        let (val_indices, train_indices) = indices.split_at(val_size);
        (train_indices.to_vec(), val_indices.to_vec())
    }

    /// Generate k-fold cross-validation splits
    ///
    /// # Panics
    ///
    /// Panics if `k <= 1` or `k > dataset_size`. See [`try_kfold_splits`] for a
    /// non-panicking variant.
    pub fn kfold_splits(
        dataset_size: usize,
        k: usize,
        seed: Option<u64>,
    ) -> Vec<(Vec<usize>, Vec<usize>)> {
        assert!(k > 1, "K must be greater than 1");
        assert!(k <= dataset_size, "K cannot be larger than dataset size");
        kfold_splits_unchecked(dataset_size, k, seed)
    }

    /// Fallible variant of [`kfold_splits`] that returns an error instead of
    /// panicking on an invalid `k`.
    pub fn try_kfold_splits(
        dataset_size: usize,
        k: usize,
        seed: Option<u64>,
    ) -> torsh_core::error::Result<Vec<(Vec<usize>, Vec<usize>)>> {
        if k <= 1 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "K must be greater than 1".to_string(),
            ));
        }
        if k > dataset_size {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "K ({k}) cannot be larger than dataset size ({dataset_size})"
            )));
        }
        Ok(kfold_splits_unchecked(dataset_size, k, seed))
    }

    /// Shared fold-building core for [`kfold_splits`]/[`try_kfold_splits`], run
    /// once `1 < k <= dataset_size` is already known to hold.
    fn kfold_splits_unchecked(
        dataset_size: usize,
        k: usize,
        seed: Option<u64>,
    ) -> Vec<(Vec<usize>, Vec<usize>)> {
        let indices = random_indices(dataset_size, dataset_size, seed);
        let fold_size = dataset_size / k;
        let mut splits = Vec::new();

        for i in 0..k {
            let start = i * fold_size;
            let end = if i == k - 1 {
                dataset_size // Last fold gets remaining samples
            } else {
                (i + 1) * fold_size
            };

            let val_indices = indices[start..end].to_vec();
            let train_indices = [&indices[..start], &indices[end..]].concat();
            splits.push((train_indices, val_indices));
        }

        splits
    }

    /// Three-way split: train, validation, test
    ///
    /// # Panics
    ///
    /// Panics if the ratios are invalid. See [`try_train_val_test_split`] for a
    /// non-panicking variant.
    pub fn train_val_test_split(
        dataset_size: usize,
        train_ratio: f32,
        val_ratio: f32,
        seed: Option<u64>,
    ) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        assert!(
            train_ratio + val_ratio < 1.0,
            "Train and val ratios must sum to less than 1.0"
        );
        assert!(
            train_ratio > 0.0 && val_ratio > 0.0,
            "Ratios must be positive"
        );
        train_val_test_split_unchecked(dataset_size, train_ratio, val_ratio, seed)
    }

    /// Fallible variant of [`train_val_test_split`] that returns an error instead
    /// of panicking on invalid ratios.
    pub fn try_train_val_test_split(
        dataset_size: usize,
        train_ratio: f32,
        val_ratio: f32,
        seed: Option<u64>,
    ) -> torsh_core::error::Result<(Vec<usize>, Vec<usize>, Vec<usize>)> {
        if !(train_ratio + val_ratio < 1.0) {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "Train and val ratios must sum to less than 1.0".to_string(),
            ));
        }
        if !(train_ratio > 0.0 && val_ratio > 0.0) {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "Ratios must be positive".to_string(),
            ));
        }
        Ok(train_val_test_split_unchecked(
            dataset_size,
            train_ratio,
            val_ratio,
            seed,
        ))
    }

    /// Shared split core for [`train_val_test_split`]/[`try_train_val_test_split`],
    /// run once the ratios are already known to be valid.
    fn train_val_test_split_unchecked(
        dataset_size: usize,
        train_ratio: f32,
        val_ratio: f32,
        seed: Option<u64>,
    ) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let train_size = (dataset_size as f32 * train_ratio).round() as usize;
        let val_size = (dataset_size as f32 * val_ratio).round() as usize;
        let _test_size = dataset_size - train_size - val_size;

        let indices = random_indices(dataset_size, dataset_size, seed);

        let train_indices = indices[..train_size].to_vec();
        let val_indices = indices[train_size..train_size + val_size].to_vec();
        let test_indices = indices[train_size + val_size..].to_vec();

        (train_indices, val_indices, test_indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampler_iterator_basic() {
        let indices = vec![0, 1, 2, 3, 4];
        let iter = SamplerIterator::new(indices.clone());

        assert_eq!(iter.len(), 5);
        assert_eq!(iter.remaining(), 5);

        let collected: Vec<usize> = iter.collect();
        assert_eq!(collected, indices);
    }

    #[test]
    fn test_sampler_iterator_from_range() {
        let iter = SamplerIterator::from_range(0, 5);
        let collected: Vec<usize> = iter.collect();
        assert_eq!(collected, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_sampler_iterator_shuffled() {
        let indices = vec![0, 1, 2, 3, 4];
        let iter = SamplerIterator::shuffled(indices.clone(), Some(42));
        let collected: Vec<usize> = iter.collect();

        // Should contain same elements but in different order
        assert_eq!(collected.len(), indices.len());
        for &idx in &indices {
            assert!(collected.contains(&idx));
        }
    }

    #[test]
    fn test_utils_random_indices() {
        let indices = utils::random_indices(10, 5, Some(42));
        assert_eq!(indices.len(), 5);

        // All indices should be unique and in range
        let mut sorted_indices = indices.clone();
        sorted_indices.sort();
        sorted_indices.dedup();
        assert_eq!(sorted_indices.len(), 5);

        for &idx in &indices {
            assert!(idx < 10);
        }
    }

    #[test]
    fn test_utils_random_indices_all() {
        let indices = utils::random_indices(5, 5, Some(42));
        assert_eq!(indices.len(), 5);

        let mut sorted_indices = indices.clone();
        sorted_indices.sort();
        assert_eq!(sorted_indices, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_utils_stratified_split() {
        let indices = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let labels = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2];

        let (train, test) = utils::stratified_split(&indices, &labels, 0.3, Some(42));

        // Check that we have roughly the right proportions
        assert!(train.len() + test.len() == indices.len());
        assert!(test.len() >= 2); // Should have at least some test samples

        // Verify all indices are accounted for
        let mut all_indices = train.clone();
        all_indices.extend(test.clone());
        all_indices.sort();
        assert_eq!(all_indices, indices);
    }

    #[test]
    fn test_utils_calculate_class_weights() {
        let labels = vec![0, 0, 1, 1, 1, 2]; // Imbalanced: 2, 3, 1
        let weights = utils::calculate_class_weights(&labels, 3);

        assert_eq!(weights.len(), 3);

        // Class 2 (1 sample) should have highest weight
        // Class 1 (3 samples) should have lowest weight
        assert!(weights[2] > weights[1]);
        assert!(weights[0] > weights[1]);
    }

    #[test]
    fn test_utils_validate_sampling_params() {
        // Valid cases
        assert!(utils::validate_sampling_params(10, Some(5), false).is_ok());
        assert!(utils::validate_sampling_params(10, Some(15), true).is_ok());
        assert!(utils::validate_sampling_params(10, None, false).is_ok());

        // Empty dataset cases (now valid)
        assert!(utils::validate_sampling_params(0, Some(0), false).is_ok());
        assert!(utils::validate_sampling_params(0, None, false).is_ok());

        // Zero samples with replacement (now valid)
        assert!(utils::validate_sampling_params(10, Some(0), true).is_ok());

        // Invalid cases
        assert!(utils::validate_sampling_params(0, Some(5), false).is_err()); // Can't sample 5 from empty dataset
        assert!(utils::validate_sampling_params(10, Some(0), false).is_err()); // Can't sample 0 from non-empty dataset without replacement
        assert!(utils::validate_sampling_params(10, Some(15), false).is_err()); // Can't sample 15 without replacement from dataset of 10
    }

    #[test]
    fn test_size_hints() {
        let iter = SamplerIterator::new(vec![0, 1, 2]);
        assert_eq!(iter.size_hint(), (3, Some(3)));

        let mut iter = SamplerIterator::new(vec![0, 1, 2]);
        iter.next();
        assert_eq!(iter.size_hint(), (2, Some(2)));
    }
}
