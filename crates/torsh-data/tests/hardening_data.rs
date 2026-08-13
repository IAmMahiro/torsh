//! Production-hardening regression tests for torsh-data.
//!
//! Each test is tied to a specific finding ID from the production-hardening
//! campaign audit. Tests are grouped by finding and encode the *correct*
//! (post-fix) behavior; see the wave report for which tests were confirmed
//! red against the pre-fix baseline.

use std::collections::HashSet;

use torsh_core::device::DeviceType;
use torsh_data::collate::{Collate, DynamicBatchCollate, TensorStacker};
use torsh_data::dataloader::{DataLoader, MultiProcessIterator, WorkerPool};
use torsh_data::sampler::{
    self, BatchSampler, BatchingSampler, DistributedSampler, RandomSampler, Sampler,
    SequentialSampler,
};
use torsh_data::{Dataset, TensorDataset};
use torsh_tensor::Tensor;

// ---------------------------------------------------------------------
// F097: RandomSampler subset must be shuffled, not left in ascending order
// ---------------------------------------------------------------------

#[test]
fn f097_random_sampler_subset_not_ascending() {
    let sampler = RandomSampler::new(50, Some(10), false).with_generator(7);
    let indices: Vec<usize> = sampler.iter().collect();
    assert_eq!(indices.len(), 10);

    let is_ascending = indices.windows(2).all(|w| w[0] < w[1]);
    assert!(
        !is_ascending,
        "RandomSampler subset must not come back in ascending order: {indices:?}"
    );

    // All indices still unique and in range.
    let mut sorted = indices.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), 10);
    assert!(indices.iter().all(|&i| i < 50));
}

#[test]
fn f097_random_sampler_subset_reproducible_with_fixed_seed() {
    let a: Vec<usize> = RandomSampler::new(50, Some(10), false)
        .with_generator(7)
        .iter()
        .collect();
    let b: Vec<usize> = RandomSampler::new(50, Some(10), false)
        .with_generator(7)
        .iter()
        .collect();
    assert_eq!(a, b, "same seed must give the same subset+order");
}

// ---------------------------------------------------------------------
// F098: DistributedSampler(drop_last) must not permanently exclude the tail
// ---------------------------------------------------------------------

#[test]
fn f098_distributed_sampler_drop_last_can_reach_tail_indices() {
    // dataset_size=10, num_replicas=3 -> effective_size=9; index 9 is the tail
    // sample that a "truncate before shuffle" bug would drop for every seed.
    let mut saw_index_9 = false;
    for seed in 0..40u64 {
        let mut union = HashSet::new();
        for rank in 0..3 {
            let sampler = DistributedSampler::new(10, 3, rank, true)
                .with_generator(seed)
                .with_drop_last(true);
            union.extend(sampler.iter());
        }
        if union.contains(&9) {
            saw_index_9 = true;
            break;
        }
    }
    assert!(
        saw_index_9,
        "index 9 must be reachable for some seed once truncation happens AFTER shuffling"
    );
}

#[test]
fn f098_distributed_sampler_no_shuffle_drop_last_unchanged() {
    // Non-regression: shuffle=false behavior must stay exactly as documented.
    let sampler = DistributedSampler::new(10, 3, 0, false).with_drop_last(true);
    let indices: Vec<usize> = sampler.iter().collect();
    assert_eq!(indices, vec![0, 1, 2]);
}

// ---------------------------------------------------------------------
// F099: TensorStacker::stack must be dim-aware, not just dim-0 layout
// ---------------------------------------------------------------------

#[test]
fn f099_tensor_stacker_dim1_interleaves_correctly() {
    let stacker = TensorStacker::new();
    let t0 = Tensor::from_data(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        DeviceType::Cpu,
    )
    .unwrap();
    let t1 = Tensor::from_data(
        vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0],
        vec![2, 3],
        DeviceType::Cpu,
    )
    .unwrap();

    let stacked = stacker.stack(&[t0, t1], 1).unwrap();
    assert_eq!(stacked.shape().dims(), &[2, 2, 3]);

    let data = stacked.to_vec().unwrap();
    let expected = vec![
        1.0, 2.0, 3.0, // outer=0, source tensor 0
        10.0, 20.0, 30.0, // outer=0, source tensor 1
        4.0, 5.0, 6.0, // outer=1, source tensor 0
        40.0, 50.0, 60.0, // outer=1, source tensor 1
    ];
    assert_eq!(data, expected);
}

#[test]
fn f099_tensor_stacker_dim0_unchanged() {
    // Non-regression for the previously-working case.
    let stacker = TensorStacker::new();
    let t0 = Tensor::from_data(vec![1.0f32, 2.0], vec![2], DeviceType::Cpu).unwrap();
    let t1 = Tensor::from_data(vec![3.0f32, 4.0], vec![2], DeviceType::Cpu).unwrap();
    let stacked = stacker.stack(&[t0, t1], 0).unwrap();
    assert_eq!(stacked.shape().dims(), &[2, 2]);
    assert_eq!(stacked.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

// F099 (extended): `collate::stack_tensors` is a separate public free
// function (`collate/optimized.rs`) that duplicated the exact same
// dim-only-affects-shape bug as `TensorStacker::stack`, plus its own copy of
// the unsafe set_len-before-fully-initialized pattern (F298). It is fixed by
// delegating to the already-hardened `TensorStacker`.
#[test]
fn f099_stack_tensors_free_fn_dim1_interleaves_correctly() {
    use torsh_data::collate::stack_tensors;

    let t0 = Tensor::from_data(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        DeviceType::Cpu,
    )
    .unwrap();
    let t1 = Tensor::from_data(
        vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0],
        vec![2, 3],
        DeviceType::Cpu,
    )
    .unwrap();

    let stacked = stack_tensors(&[t0, t1], 1).unwrap();
    assert_eq!(stacked.shape().dims(), &[2, 2, 3]);

    let data = stacked.to_vec().unwrap();
    let expected = vec![
        1.0, 2.0, 3.0, // outer=0, source tensor 0
        10.0, 20.0, 30.0, // outer=0, source tensor 1
        4.0, 5.0, 6.0, // outer=1, source tensor 0
        40.0, 50.0, 60.0, // outer=1, source tensor 1
    ];
    assert_eq!(data, expected);
}

#[test]
fn f099_stack_tensors_free_fn_dim0_unchanged() {
    // Non-regression for the previously-working case.
    use torsh_data::collate::stack_tensors;

    let t0 = Tensor::from_data(vec![1.0f32, 2.0], vec![2], DeviceType::Cpu).unwrap();
    let t1 = Tensor::from_data(vec![3.0f32, 4.0], vec![2], DeviceType::Cpu).unwrap();
    let stacked = stack_tensors(&[t0, t1], 0).unwrap();
    assert_eq!(stacked.shape().dims(), &[2, 2]);
    assert_eq!(stacked.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

// ---------------------------------------------------------------------
// F100: DynamicBatchCollate must report post-truncation lengths
// ---------------------------------------------------------------------

#[test]
fn f100_dynamic_batch_collate_lengths_match_truncation() {
    let long = Tensor::from_data(vec![1.0f32; 50], vec![50], DeviceType::Cpu).unwrap();
    let short = Tensor::from_data(vec![2.0f32; 5], vec![5], DeviceType::Cpu).unwrap();

    let collate = DynamicBatchCollate::new(0.0f32).with_max_length(10);
    let (padded, lengths) = collate.collate(vec![long, short]).unwrap();

    assert_eq!(padded.size(1).unwrap(), 10);

    let lengths_vec = lengths.to_vec().unwrap();
    assert!(
        lengths_vec.iter().all(|&l| l <= 10),
        "reported lengths must not exceed the padded/truncated size: {lengths_vec:?}"
    );
    assert_eq!(
        lengths_vec.iter().copied().max().unwrap(),
        10,
        "the 50-step sequence must report its truncated length (10), not 50"
    );
}

// ---------------------------------------------------------------------
// F101: multi-worker DataLoader must yield batches in submission order
// ---------------------------------------------------------------------

#[derive(Clone)]
struct DelayedDataset {
    n: usize,
}

impl Dataset for DelayedDataset {
    type Item = f32;

    fn len(&self) -> usize {
        self.n
    }

    fn get(&self, index: usize) -> torsh_core::error::Result<f32> {
        // Earlier indices take longer, so a naive "first result wins" iterator
        // would deliver results out of submission order.
        let delay_ms = (self.n - index) as u64 * 20;
        std::thread::sleep(std::time::Duration::from_millis(delay_ms));
        Ok(index as f32)
    }
}

#[derive(Clone)]
struct IdentityCollate;

impl Collate<f32> for IdentityCollate {
    type Output = Vec<f32>;

    fn collate(&self, batch: Vec<f32>) -> torsh_core::error::Result<Vec<f32>> {
        Ok(batch)
    }
}

#[test]
fn f101_multiprocess_iterator_preserves_submission_order() {
    use std::sync::Arc;

    let dataset = Arc::new(DelayedDataset { n: 6 });
    let collate_fn = Arc::new(IdentityCollate);
    let pool = WorkerPool::new(dataset, collate_fn, 4);

    let base = SequentialSampler::new(6);
    let batch_sampler = BatchingSampler::new(base, 1, false);
    let sampler_iter = batch_sampler.iter();

    let mut mp_iter: MultiProcessIterator<
        DelayedDataset,
        BatchingSampler<SequentialSampler>,
        IdentityCollate,
    > = MultiProcessIterator::new(sampler_iter, &pool);

    let mut results = Vec::new();
    while let Some(item) = mp_iter.next() {
        let batch = item.expect("batch should collate successfully");
        results.push(batch[0]);
    }

    assert_eq!(
        results,
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "multi-worker batches must be reordered back to submission order: {results:?}"
    );
}

// ---------------------------------------------------------------------
// F208: train_val_split must shuffle even when seed is None
// ---------------------------------------------------------------------

#[test]
fn f208_train_val_split_shuffles_when_seed_is_none() {
    let (train1, val1) = sampler::train_val_split(20, 0.5, None);
    let (train2, val2) = sampler::train_val_split(20, 0.5, None);

    assert_eq!(train1.len(), 10);
    assert_eq!(val1.len(), 10);

    assert!(
        train1 != train2 || val1 != val2,
        "train_val_split(seed=None) must draw fresh entropy each call, not repeat a fixed split"
    );
}

// ---------------------------------------------------------------------
// F209: default_distributed_sampler must forward its seed argument
// ---------------------------------------------------------------------

#[test]
fn f209_default_distributed_sampler_forwards_seed() {
    let with_default_helper: Vec<usize> = sampler::default_distributed_sampler(20, 2, 0, Some(123))
        .iter()
        .collect();
    let with_explicit_generator: Vec<usize> = sampler::distributed_sampler(20, 2, 0, true)
        .with_generator(123)
        .iter()
        .collect();

    assert_eq!(
        with_default_helper, with_explicit_generator,
        "default_distributed_sampler(seed=Some(123)) must match a sampler built with .with_generator(123)"
    );
}

// ---------------------------------------------------------------------
// F096: DataLoaderBuilder::build() must not silently ignore shuffle(true)
// ---------------------------------------------------------------------

#[test]
fn f096_dataloader_builder_build_rejects_shuffle_true() {
    let tensor = torsh_tensor::creation::ones::<f32>(&[5]).expect("tensor creation");
    let dataset = TensorDataset::from_tensor(tensor);

    let result = DataLoader::builder(dataset)
        .batch_size(2)
        .shuffle(true)
        .build();

    assert!(
        result.is_err(),
        "build() must not silently produce a sequential loader when shuffle(true) was requested"
    );
}

#[test]
fn f096_dataloader_builder_build_still_works_for_shuffle_false() {
    // Non-regression: the common, correct call pattern must keep working.
    let tensor = torsh_tensor::creation::ones::<f32>(&[5]).expect("tensor creation");
    let dataset = TensorDataset::from_tensor(tensor);

    let dataloader = DataLoader::builder(dataset)
        .batch_size(2)
        .shuffle(false)
        .build()
        .expect("build() with shuffle(false) must still succeed");
    assert_eq!(dataloader.len(), 3);
}

// ---------------------------------------------------------------------
// F007: random image transforms must not be deterministic across calls
// ---------------------------------------------------------------------

#[cfg(feature = "image-support")]
fn make_test_image(w: u32, h: u32) -> image::DynamicImage {
    let img = image::RgbImage::from_fn(w, h, |x, y| image::Rgb([x as u8, y as u8, 0]));
    image::DynamicImage::ImageRgb8(img)
}

#[cfg(feature = "image-support")]
#[test]
fn f007_random_horizontal_flip_varies_across_calls() {
    use torsh_data::{RandomHorizontalFlip, Transform};

    let img = make_test_image(4, 4);
    let flip = RandomHorizontalFlip::new(0.5);

    let mut outcomes = HashSet::new();
    for _ in 0..40 {
        let out = flip.transform(img.clone()).expect("flip should succeed");
        outcomes.insert(out.to_rgb8().into_raw());
    }

    assert!(
        outcomes.len() > 1,
        "RandomHorizontalFlip::transform must not draw the identical decision on every call"
    );
}

#[cfg(feature = "image-support")]
#[test]
fn f007_random_horizontal_flip_with_seed_is_reproducible() {
    use torsh_data::{RandomHorizontalFlip, Transform};

    let img = make_test_image(4, 4);

    let flip_a = RandomHorizontalFlip::new(0.5).with_seed(123);
    let seq_a: Vec<Vec<u8>> = (0..10)
        .map(|_| flip_a.transform(img.clone()).unwrap().to_rgb8().into_raw())
        .collect();

    let flip_b = RandomHorizontalFlip::new(0.5).with_seed(123);
    let seq_b: Vec<Vec<u8>> = (0..10)
        .map(|_| flip_b.transform(img.clone()).unwrap().to_rgb8().into_raw())
        .collect();

    assert_eq!(
        seq_a, seq_b,
        "with_seed(..) must make the sequence of decisions reproducible"
    );
}

// ---------------------------------------------------------------------
// F008: set_epoch must exist and advance the permutation per epoch,
// reproducibly, while epoch 0 matches the pre-existing (no-epoch) behavior.
// ---------------------------------------------------------------------

#[test]
fn f008_random_sampler_set_epoch_changes_permutation_reproducibly() {
    let mut sampler = RandomSampler::new(30, None, false).with_generator(7);
    let epoch0: Vec<usize> = sampler.iter().collect();

    sampler.set_epoch(1);
    let epoch1: Vec<usize> = sampler.iter().collect();

    sampler.set_epoch(0);
    let epoch0_again: Vec<usize> = sampler.iter().collect();

    assert_ne!(
        epoch0, epoch1,
        "different epochs must draw different permutations"
    );
    assert_eq!(
        epoch0, epoch0_again,
        "returning to the same epoch must reproduce the same permutation"
    );
}

#[test]
fn f008_random_sampler_epoch0_matches_pre_epoch_behavior() {
    // epoch 0 (the default) must reproduce the exact seed used before
    // per-epoch reseeding existed, so existing callers who never call
    // set_epoch see no behavior change.
    let baseline: Vec<usize> = RandomSampler::new(20, None, false)
        .with_generator(99)
        .iter()
        .collect();

    let mut explicit_epoch0 = RandomSampler::new(20, None, false).with_generator(99);
    explicit_epoch0.set_epoch(0);
    let explicit: Vec<usize> = explicit_epoch0.iter().collect();

    assert_eq!(baseline, explicit);
}

#[test]
fn f008_distributed_sampler_set_epoch_changes_permutation() {
    let mut sampler = DistributedSampler::new(20, 2, 0, true).with_generator(7);
    let epoch0: Vec<usize> = sampler.iter().collect();

    sampler.set_epoch(1);
    let epoch1: Vec<usize> = sampler.iter().collect();

    assert_ne!(epoch0, epoch1);
}

#[test]
fn f008_dataloader_set_epoch_forwards_to_random_sampler() {
    let tensor = torsh_tensor::creation::ones::<f32>(&[10]).expect("tensor creation");
    let dataset = TensorDataset::from_tensor(tensor);
    let mut dataloader = DataLoader::builder(dataset)
        .batch_size(2)
        .generator(11)
        .build_with_random_sampling()
        .expect("build_with_random_sampling");

    assert_eq!(dataloader.sampler().sampler().epoch(), 0);
    dataloader.set_epoch(5);
    assert_eq!(dataloader.sampler().sampler().epoch(), 5);
}

// ---------------------------------------------------------------------
// F095: constructors get a fallible try_* sibling instead of only
// panicking via assert!. `new()`/`with_*()` are left untouched (existing
// #[should_panic] tests keep pinning their exact messages); these tests
// only exercise the new Result-returning API.
// ---------------------------------------------------------------------

#[test]
fn f095_augmentation_pipeline_try_constructors() {
    use torsh_data::augmentation_pipeline::{
        AugmentationPipeline, ConditionalTransform, GaussianNoise, RandomBrightness, RandomErasing,
        RandomHue,
    };

    assert!(AugmentationPipeline::<i32>::new()
        .try_with_probability(0.5)
        .is_ok());
    assert!(AugmentationPipeline::<i32>::new()
        .try_with_probability(1.5)
        .is_err());

    let ok: Result<ConditionalTransform<i32, _>, _> =
        ConditionalTransform::try_new(torsh_data::transforms::lambda(|x: i32| Ok(x)), 0.5);
    assert!(ok.is_ok());
    let err: Result<ConditionalTransform<i32, _>, _> =
        ConditionalTransform::try_new(torsh_data::transforms::lambda(|x: i32| Ok(x)), -0.1);
    assert!(err.is_err());

    assert!(RandomBrightness::try_new((0.8, 1.2)).is_ok());
    assert!(RandomBrightness::try_new((1.2, 0.8)).is_err());

    assert!(RandomHue::try_new((-0.1, 0.1)).is_ok());
    assert!(RandomHue::try_new((0.1, -0.1)).is_err()); // inverted range
    assert!(RandomHue::try_new((-2.0, 2.0)).is_err()); // out of [-1, 1]

    assert!(GaussianNoise::try_new(0.0, 0.1).is_ok());
    assert!(GaussianNoise::try_new(0.0, -0.1).is_err());

    assert!(RandomErasing::try_new(0.5, (0.02, 0.33), (0.3, 3.3)).is_ok());
    assert!(RandomErasing::try_new(1.5, (0.02, 0.33), (0.3, 3.3)).is_err());
}

#[test]
fn f095_sampler_try_constructors() {
    use torsh_data::sampler::importance::ImportanceSampler;

    // sampler::core::utils, re-exported as sampler::utils
    assert!(sampler::utils::try_random_indices(10, 5, Some(1)).is_ok());
    assert!(sampler::utils::try_random_indices(5, 10, Some(1)).is_err());

    assert!(sampler::utils::try_kfold_splits(10, 3, Some(1)).is_ok());
    assert!(sampler::utils::try_kfold_splits(10, 1, Some(1)).is_err());
    assert!(sampler::utils::try_kfold_splits(3, 10, Some(1)).is_err());

    assert!(sampler::utils::try_train_val_test_split(10, 0.6, 0.2, Some(1)).is_ok());
    assert!(sampler::utils::try_train_val_test_split(10, 0.6, 0.5, Some(1)).is_err());

    assert!(BatchingSampler::try_new(SequentialSampler::new(10), 3, false).is_ok());
    assert!(BatchingSampler::try_new(SequentialSampler::new(10), 0, false).is_err());

    assert!(ImportanceSampler::try_new(vec![1.0, 2.0, 3.0], 2, false).is_ok());
    assert!(ImportanceSampler::try_new(vec![], 2, false).is_err());
    assert!(ImportanceSampler::try_new(vec![-1.0, 2.0], 1, false).is_err());
    assert!(ImportanceSampler::try_new(vec![1.0], 1, false)
        .unwrap()
        .try_with_temperature(0.5)
        .is_ok());
    assert!(ImportanceSampler::try_new(vec![1.0], 1, false)
        .unwrap()
        .try_with_temperature(-0.5)
        .is_err());
}

#[test]
fn f095_dataset_kfold_try_new() {
    use torsh_data::KFold;
    assert!(KFold::try_new(5, true, Some(1)).is_ok());
    assert!(KFold::try_new(1, true, Some(1)).is_err());
}

#[test]
fn f095_ngram_try_new() {
    use torsh_data::text::transforms::NGrams;
    use torsh_data::text_processing::NGramGenerator;

    assert!(NGrams::try_new(2).is_ok());
    assert!(NGrams::try_new(0).is_err());

    assert!(NGramGenerator::try_new(2).is_ok());
    assert!(NGramGenerator::try_new(0).is_err());
}

#[test]
fn f095_tensor_transforms_random_horizontal_flip_try_new() {
    use torsh_data::tensor_transforms::RandomHorizontalFlip;
    assert!(RandomHorizontalFlip::try_new(0.5).is_ok());
    assert!(RandomHorizontalFlip::try_new(1.5).is_err());
}

#[test]
fn f095_zero_copy_tensor_try_constructors() {
    use torsh_data::zero_copy::ZeroCopyTensor;

    let data = vec![1i32, 2, 3, 4];
    assert!(ZeroCopyTensor::try_from_slice(&data, vec![2, 2]).is_ok());
    assert!(ZeroCopyTensor::try_from_slice(&data, vec![2, 3]).is_err());

    assert!(ZeroCopyTensor::try_from_vec(vec![1i32, 2, 3], vec![3]).is_ok());
    assert!(ZeroCopyTensor::try_from_vec(vec![1i32, 2, 3], vec![2, 2]).is_err());
}

#[cfg(feature = "audio-support")]
#[test]
fn f095_audio_add_noise_try_new() {
    use torsh_data::audio::AddNoise;
    assert!(AddNoise::try_new(0.1).is_ok());
    assert!(AddNoise::try_new(-0.1).is_err());
}

#[cfg(feature = "privacy")]
#[test]
fn f095_privacy_budget_try_new() {
    use torsh_data::PrivacyBudget;
    assert!(PrivacyBudget::try_new(1.0, 1e-5).is_ok());
    assert!(PrivacyBudget::try_new(-1.0, 1e-5).is_err());
    assert!(PrivacyBudget::try_new(1.0, 1.0).is_err());
}

// ---------------------------------------------------------------------
// F102: VideoFolder must fail loudly instead of fabricating frames; IMDB
// parsing itself is covered by torsh_data::text's own test suite
// (test_imdb_dataset / test_imdb_dataset_missing_layout_errors), since it
// needed a `#[cfg(test)]`-only tempdir fixture colocated with the type.
// ---------------------------------------------------------------------

#[cfg(feature = "image-support")]
#[test]
fn f102_video_folder_load_video_errors_instead_of_fabricating() {
    use torsh_data::VideoFolder;

    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let class_dir = tmp.path().join("class0");
    std::fs::create_dir_all(&class_dir).expect("create class dir");
    // A .mp4 "file" that is not actually a decodable video - real decoding
    // is not attempted, so content doesn't matter for this test.
    std::fs::write(class_dir.join("clip.mp4"), b"not a real video").expect("write clip");

    let dataset = VideoFolder::new(tmp.path()).expect("scanning the directory must succeed");
    assert_eq!(dataset.num_samples(), 1);

    // Enumeration (real filesystem data) succeeds, but decoding a sample
    // must fail loudly rather than silently returning random noise.
    let result = dataset.get(0);
    assert!(
        result.is_err(),
        "VideoFolder::get must error instead of fabricating frames"
    );
}

// ---------------------------------------------------------------------
// F007 (extended sweep): F007's suggested_fix explicitly calls for sweeping
// "every `Random::seed(<literal>)` occurrence inside a per-call code path
// across torsh-data". `privacy.rs`'s differential-privacy noise generators
// had the identical pattern (fresh `Random::seed(42)` / `Random::seed(self.seed)`
// reconstructed on every call), which is far more severe than a
// deterministic image transform: noise that repeats on every query provides
// no differential-privacy protection at all.
// ---------------------------------------------------------------------

#[cfg(feature = "privacy")]
#[test]
fn f007_laplace_noise_varies_across_repeated_calls() {
    use torsh_data::{LaplaceNoise, NoiseGenerator};

    let mut gen = LaplaceNoise::with_seed(7);
    let a = gen.generate_laplace_tensor(&[8, 8], 1.0).unwrap();
    let b = gen.generate_laplace_tensor(&[8, 8], 1.0).unwrap();
    assert_ne!(
        a.to_vec().unwrap(),
        b.to_vec().unwrap(),
        "two calls on the same LaplaceNoise instance must not draw identical noise"
    );
}

#[cfg(feature = "privacy")]
#[test]
fn f007_laplace_noise_with_seed_first_draw_reproducible() {
    // Non-regression: the first draw from a freshly-seeded instance must
    // stay reproducible (this is the epoch-0-equivalent contract).
    use torsh_data::{LaplaceNoise, NoiseGenerator};

    let a = LaplaceNoise::with_seed(99)
        .generate_laplace_tensor(&[4, 4], 1.0)
        .unwrap();
    let b = LaplaceNoise::with_seed(99)
        .generate_laplace_tensor(&[4, 4], 1.0)
        .unwrap();
    assert_eq!(a.to_vec().unwrap(), b.to_vec().unwrap());
}

#[cfg(feature = "privacy")]
#[test]
fn f007_gaussian_noise_varies_across_repeated_calls() {
    use torsh_data::{GaussianNoise, NoiseGenerator};

    let mut gen = GaussianNoise::with_seed(7);
    let a = gen.generate_gaussian_tensor(&[8, 8], 0.0, 1.0).unwrap();
    let b = gen.generate_gaussian_tensor(&[8, 8], 0.0, 1.0).unwrap();
    assert_ne!(
        a.to_vec().unwrap(),
        b.to_vec().unwrap(),
        "two calls on the same GaussianNoise instance must not draw identical noise"
    );
}

#[cfg(feature = "privacy")]
#[test]
fn f007_private_sampler_report_noisy_max_varies_across_calls() {
    use torsh_data::sampler::SequentialSampler;
    use torsh_data::{DPMechanism, PrivacyBudget, PrivateSampler};

    let base = SequentialSampler::new(100);
    let budget = PrivacyBudget::new(1000.0, 0.0);
    let mut sampler =
        PrivateSampler::new(base, budget, DPMechanism::ReportNoisyMax { epsilon: 1.0 });

    let first = sampler.private_iter().unwrap();
    let second = sampler.private_iter().unwrap();
    assert_ne!(
        first, second,
        "repeated ReportNoisyMax draws on the same PrivateSampler must not \
         reshuffle into the identical order every time"
    );
}
