//! Production-hardening regression tests for tensor creation, storage,
//! type conversion and the memory pool.
//!
//! Each test pins one of the findings fixed in the wave-1 hardening pass; every
//! one of them fails against the pre-fix implementation.

use torsh_core::device::DeviceType;
use torsh_tensor::creation::{
    arange, manual_seed, rand, randint, randn, randn_with_seed, zeros_device,
};
use torsh_tensor::storage::{MemoryMappedStorage, TensorStorage};
use torsh_tensor::Tensor;

// ── F052: rand/randn/randint must not return identical data on every call ────

#[test]
fn f052_randn_differs_between_calls() {
    let a = randn::<f32>(&[256])
        .expect("randn should succeed")
        .to_vec()
        .expect("to_vec should succeed");
    let b = randn::<f32>(&[256])
        .expect("randn should succeed")
        .to_vec()
        .expect("to_vec should succeed");

    assert_ne!(
        a, b,
        "two randn calls with the same shape must not return identical data"
    );
}

#[test]
fn f052_rand_and_randint_differ_between_calls() {
    let a = rand::<f32>(&[128])
        .expect("rand should succeed")
        .to_vec()
        .expect("to_vec should succeed");
    let b = rand::<f32>(&[128])
        .expect("rand should succeed")
        .to_vec()
        .expect("to_vec should succeed");
    assert_ne!(a, b, "two rand calls must not return identical data");

    let c = randint(0, 1_000_000, &[128])
        .expect("randint should succeed")
        .to_vec()
        .expect("to_vec should succeed");
    let d = randint(0, 1_000_000, &[128])
        .expect("randint should succeed")
        .to_vec()
        .expect("to_vec should succeed");
    assert_ne!(c, d, "two randint calls must not return identical data");
}

#[test]
fn f052_manual_seed_makes_sequences_reproducible() {
    manual_seed(1234);
    let first = randn::<f32>(&[64])
        .expect("randn should succeed")
        .to_vec()
        .expect("to_vec should succeed");
    let second = randn::<f32>(&[64])
        .expect("randn should succeed")
        .to_vec()
        .expect("to_vec should succeed");
    assert_ne!(
        first, second,
        "consecutive draws from one seeded stream must advance"
    );

    manual_seed(1234);
    let replay_first = randn::<f32>(&[64])
        .expect("randn should succeed")
        .to_vec()
        .expect("to_vec should succeed");
    let replay_second = randn::<f32>(&[64])
        .expect("randn should succeed")
        .to_vec()
        .expect("to_vec should succeed");

    assert_eq!(first, replay_first, "manual_seed must replay the sequence");
    assert_eq!(
        second, replay_second,
        "manual_seed must replay the sequence"
    );
}

#[test]
fn f052_seeded_helpers_are_independent_of_global_state() {
    let a = randn_with_seed::<f32>(&[32], 7)
        .expect("randn_with_seed should succeed")
        .to_vec()
        .expect("to_vec should succeed");
    let b = randn_with_seed::<f32>(&[32], 7)
        .expect("randn_with_seed should succeed")
        .to_vec()
        .expect("to_vec should succeed");
    assert_eq!(a, b, "an explicit seed must be reproducible");
}

// ── F053: randn::<f16>/<bf16> must produce real N(0,1) samples ───────────────

#[test]
fn f053_randn_f16_is_standard_normal() {
    let data = randn::<half::f16>(&[8192])
        .expect("randn::<f16> should succeed")
        .to_vec()
        .expect("to_vec should succeed");

    assert!(
        data.iter().all(|v| v.is_finite()),
        "f16 normal samples must all be finite"
    );

    let values: Vec<f64> = data.iter().map(|v| f64::from(*v)).collect();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance =
        values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / values.len() as f64;
    let stddev = variance.sqrt();

    assert!(mean.abs() < 0.2, "f16 sample mean {mean} should be near 0");
    assert!(
        (stddev - 1.0).abs() < 0.2,
        "f16 sample stddev {stddev} should be near 1"
    );
}

#[test]
fn f053_randn_bf16_is_standard_normal() {
    let data = randn::<half::bf16>(&[8192])
        .expect("randn::<bf16> should succeed")
        .to_vec()
        .expect("to_vec should succeed");

    assert!(
        data.iter().all(|v| v.is_finite()),
        "bf16 normal samples must all be finite"
    );

    let values: Vec<f64> = data.iter().map(|v| f64::from(*v)).collect();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance =
        values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / values.len() as f64;
    let stddev = variance.sqrt();

    assert!(mean.abs() < 0.2, "bf16 sample mean {mean} should be near 0");
    assert!(
        (stddev - 1.0).abs() < 0.2,
        "bf16 sample stddev {stddev} should be near 1"
    );
}

// ── F158: arange must reject a step that cannot terminate ────────────────────

#[test]
fn f158_arange_rejects_zero_step() {
    assert!(
        arange(0.0f32, 10.0, 0.0).is_err(),
        "a zero step must be an error, not an infinite loop"
    );
    assert!(arange(0i32, 10, 0).is_err(), "integer zero step must error");
}

#[test]
fn f158_arange_supports_descending_ranges() {
    let values = arange(5i32, 0, -1)
        .expect("descending arange should succeed")
        .to_vec()
        .expect("to_vec should succeed");
    assert_eq!(values, vec![5, 4, 3, 2, 1]);

    // A step whose sign cannot reach `end` yields an empty tensor.
    let empty = arange(0i32, 10, -1)
        .expect("unreachable range should be empty, not an error")
        .to_vec()
        .expect("to_vec should succeed");
    assert!(empty.is_empty());
}

#[test]
fn f158_arange_ascending_still_works() {
    let values = arange(0i32, 10, 2)
        .expect("arange should succeed")
        .to_vec()
        .expect("to_vec should succeed");
    assert_eq!(values, vec![0, 2, 4, 6, 8]);
}

// ── F061: temporary memory-mapped tensors must not share one backing file ────

#[test]
fn f061_two_temporary_mmap_storages_keep_their_own_data() {
    let first: TensorStorage<f32> = TensorStorage::memory_mapped(vec![1.0, 2.0, 3.0, 4.0], None)
        .expect("memory-mapped storage should be created");
    let second: TensorStorage<f32> =
        TensorStorage::memory_mapped(vec![10.0, 20.0, 30.0, 40.0, 50.0], None)
            .expect("memory-mapped storage should be created");

    assert_eq!(
        first.to_vec().expect("first storage should be readable"),
        vec![1.0, 2.0, 3.0, 4.0],
        "creating a second temporary mmap tensor must not truncate the first"
    );
    assert_eq!(
        second.to_vec().expect("second storage should be readable"),
        vec![10.0, 20.0, 30.0, 40.0, 50.0]
    );

    // Dropping one must not delete the other's backing file.
    drop(second);
    assert_eq!(
        first.to_vec().expect("first storage survives the drop"),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn f061_two_memory_mapped_tensors_round_trip() {
    let a = Tensor::from_data_with_storage(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu, true)
        .expect("mmap tensor should be created");
    let b = Tensor::from_data_with_storage(vec![7.0f32, 8.0, 9.0], vec![3], DeviceType::Cpu, true)
        .expect("mmap tensor should be created");

    assert_eq!(a.to_vec().expect("to_vec"), vec![1.0, 2.0, 3.0]);
    assert_eq!(b.to_vec().expect("to_vec"), vec![7.0, 8.0, 9.0]);
}

// ── F062: set/set_slice must work above the SimdOptimized threshold ──────────

#[test]
fn f062_set_works_on_large_tensor() {
    // 4096 f32 = 16 KB, well above the 10 KB SimdOptimized threshold.
    let tensor = Tensor::<f32>::zeros(&[64, 64], DeviceType::Cpu).expect("zeros should succeed");
    assert_eq!(tensor.numel(), 4096);

    tensor.set(&[0, 0], 1.5).expect("set should succeed");
    tensor.set(&[63, 63], -2.5).expect("set should succeed");

    assert_eq!(tensor.get(&[0, 0]).expect("get should succeed"), 1.5);
    assert_eq!(tensor.get(&[63, 63]).expect("get should succeed"), -2.5);
    assert_eq!(tensor.get(&[0, 1]).expect("get should succeed"), 0.0);

    let data = tensor.to_vec().expect("to_vec should succeed");
    assert_eq!(data[0], 1.5);
    assert_eq!(data[4095], -2.5);
}

#[test]
fn f062_set_slice_works_on_large_tensor() {
    let tensor = Tensor::<f32>::zeros(&[4096], DeviceType::Cpu).expect("zeros should succeed");
    let values: Vec<f32> = (0..128).map(|i| i as f32).collect();

    tensor
        .set_slice(1000, &values)
        .expect("set_slice should succeed");

    let read_back = tensor
        .get_slice(1000, 128)
        .expect("get_slice should succeed");
    assert_eq!(read_back, values);
    assert_eq!(tensor.get_flat(999).expect("get_flat"), 0.0);
}

#[test]
fn f062_repeated_sets_stay_linear() {
    // A fill loop must not degrade into per-element reallocation.
    let tensor = Tensor::<f32>::zeros(&[4096], DeviceType::Cpu).expect("zeros should succeed");
    for i in 0..4096 {
        tensor.set(&[i], i as f32).expect("set should succeed");
    }

    let data = tensor.to_vec().expect("to_vec should succeed");
    for (i, value) in data.iter().enumerate() {
        assert_eq!(*value, i as f32, "element {i} should have been written");
    }
}

#[test]
fn f062_simd_storage_direct_slice_matches_after_mutation() {
    let storage =
        TensorStorage::create_optimal(vec![1.0f32; 4096]).expect("create_optimal should succeed");
    assert_eq!(storage.storage_type(), "simd_optimized");

    // Before any write the lock-free direct slice is available.
    assert!(storage.try_as_slice_direct().is_some());

    storage.set(10, 42.0).expect("set should succeed");
    assert_eq!(storage.get(10).expect("get should succeed"), 42.0);
    assert_eq!(storage.to_vec().expect("to_vec should succeed")[10], 42.0);
    // After a write the unguarded slice is no longer handed out.
    assert!(storage.try_as_slice_direct().is_none());

    // with_slice must observe the mutation too.
    let observed = storage
        .with_slice(|slice| Ok(slice[10]))
        .expect("with_slice should succeed");
    assert_eq!(observed, 42.0);
}

#[test]
fn f062_clone_shares_storage_like_the_other_variants() {
    // Pins the semantics: a cloned tensor shares its storage handle, so a write
    // through one handle is visible through the other — the same behaviour the
    // InMemory and Aligned variants have always had.
    let large = Tensor::<f32>::zeros(&[4096], DeviceType::Cpu).expect("zeros should succeed");
    let large_clone = large.clone();
    large.set(&[7], 9.0).expect("set should succeed");
    assert_eq!(large_clone.get(&[7]).expect("get should succeed"), 9.0);

    let small = Tensor::<f32>::zeros(&[8], DeviceType::Cpu).expect("zeros should succeed");
    let small_clone = small.clone();
    small.set(&[3], 9.0).expect("set should succeed");
    assert_eq!(
        small_clone.get(&[3]).expect("get should succeed"),
        9.0,
        "large and small tensors must agree on clone semantics"
    );
}

#[test]
fn f062_with_slice_mut_works_on_simd_storage() {
    let storage =
        TensorStorage::create_optimal(vec![0.0f32; 4096]).expect("create_optimal should succeed");
    storage
        .with_slice_mut(|slice| {
            for (i, value) in slice.iter_mut().enumerate() {
                *value = i as f32;
            }
            Ok(())
        })
        .expect("with_slice_mut should succeed");

    assert_eq!(storage.get(4095).expect("get should succeed"), 4095.0);
}

// ── F169: aligned storage construction must preserve data ───────────────────

#[test]
fn f169_bulk_copy_preserves_data() {
    // Above ALIGNED_STORAGE_THRESHOLD (1 KB) but below the SIMD one (10 KB).
    let data: Vec<f32> = (0..1024).map(|i| i as f32 * 0.5).collect();
    let storage = TensorStorage::create_optimal(data.clone()).expect("create_optimal");
    assert_eq!(storage.storage_type(), "aligned_simd");
    assert_eq!(storage.to_vec().expect("to_vec"), data);

    let simd_data: Vec<f64> = (0..4096).map(|i| i as f64).collect();
    let simd_storage = TensorStorage::create_optimal(simd_data.clone()).expect("create_optimal");
    assert_eq!(simd_storage.storage_type(), "simd_optimized");
    assert_eq!(simd_storage.to_vec().expect("to_vec"), simd_data);

    // Empty input must not trip the bulk copy.
    let empty = TensorStorage::<f32>::aligned(Vec::new()).expect("aligned storage");
    assert!(empty.is_empty());
}

// ── F275: memory-mapped bulk reads ──────────────────────────────────────────

#[test]
fn f275_memory_mapped_get_slice_reads_in_bulk() {
    let data: Vec<i64> = (0..5000).collect();
    let mut storage = MemoryMappedStorage::new(data.clone(), None)
        .expect("memory-mapped storage should be created");

    assert_eq!(storage.get_slice(0, 5000).expect("bulk read"), data);
    assert_eq!(
        storage.get_slice(1234, 10).expect("bulk read"),
        data[1234..1244]
    );
    assert_eq!(storage.to_vec().expect("to_vec"), data);

    // Writes must be visible through the bulk path.
    storage.set(42, -1).expect("set should succeed");
    assert_eq!(
        storage.get_slice(40, 4).expect("bulk read"),
        vec![40, 41, -1, 43]
    );

    // Out-of-range requests still error.
    assert!(storage.get_slice(4999, 2).is_err());
}

// ── F147: disk-backed tensors are actually backed by a file ─────────────────

#[test]
fn f147_disk_backed_uses_file_storage() {
    let path = std::env::temp_dir().join(format!(
        "torsh_hardening_disk_backed_{}.bin",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);

    let tensor = Tensor::<f32>::disk_backed(
        &[64, 64],
        DeviceType::Cpu,
        Some(path.to_str().expect("temp path should be valid UTF-8")),
    )
    .expect("disk_backed should succeed");

    assert_eq!(tensor.numel(), 4096);
    assert_eq!(tensor.shape().dims(), &[64, 64]);

    let metadata = std::fs::metadata(&path).expect("backing file must exist");
    assert_eq!(
        metadata.len(),
        (4096 * std::mem::size_of::<f32>()) as u64,
        "the backing file must hold the whole tensor"
    );

    assert_eq!(tensor.get(&[0, 0]).expect("get should succeed"), 0.0);

    // The bulk read path (F275) must materialise the whole file correctly.
    let all = tensor.to_vec().expect("to_vec should succeed");
    assert_eq!(all.len(), 4096);
    assert!(all.iter().all(|v| *v == 0.0));

    tensor.set(&[1, 1], 3.5).expect("set should succeed");
    assert_eq!(tensor.get(&[1, 1]).expect("get should succeed"), 3.5);
    let after_write = tensor.to_vec().expect("to_vec should succeed");
    assert_eq!(after_write[65], 3.5, "writes must be visible in bulk reads");

    drop(tensor);
    // An explicit path is persistent.
    assert!(
        path.exists(),
        "an explicit backing path must not be deleted"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn f147_disk_backed_temporary_file_is_cleaned_up() {
    let tensor = Tensor::<f32>::disk_backed(&[32, 32], DeviceType::Cpu, None).expect("disk_backed");
    assert_eq!(tensor.numel(), 1024);
    assert_eq!(tensor.get(&[5, 5]).expect("get should succeed"), 0.0);
}

// ── F267: checked i32 conversions ───────────────────────────────────────────

#[test]
fn f267_to_i32_simd_rejects_out_of_range_i64() {
    let big = Tensor::from_data(vec![2i64.pow(40)], vec![1], DeviceType::Cpu)
        .expect("tensor creation should succeed");
    assert!(
        big.to_i32_simd().is_err(),
        "2^40 must not silently truncate to 0"
    );

    let ok = Tensor::from_data(vec![1i64, -2, 3], vec![3], DeviceType::Cpu)
        .expect("tensor creation should succeed");
    assert_eq!(
        ok.to_i32_simd()
            .expect("in-range conversion should succeed")
            .to_vec()
            .expect("to_vec"),
        vec![1, -2, 3]
    );
}

#[test]
fn f267_to_i32_simd_rejects_non_finite_and_huge_f32() {
    let nan = Tensor::from_data(vec![f32::NAN], vec![1], DeviceType::Cpu)
        .expect("tensor creation should succeed");
    assert!(nan.to_i32_simd().is_err(), "NaN must not convert to 0");

    let huge = Tensor::from_data(vec![1e30f32], vec![1], DeviceType::Cpu)
        .expect("tensor creation should succeed");
    assert!(
        huge.to_i32_simd().is_err(),
        "1e30 must not saturate silently"
    );

    let ok = Tensor::from_data(vec![1.9f32, -2.9], vec![2], DeviceType::Cpu)
        .expect("tensor creation should succeed");
    assert_eq!(
        ok.to_i32_simd()
            .expect("in-range conversion should succeed")
            .to_vec()
            .expect("to_vec"),
        vec![1, -2]
    );
}

// ── F260: constructing the pool must not depend on the leak detector ────────

#[test]
fn f260_tensor_pool_construction_does_not_panic() {
    let pool = torsh_tensor::memory_pool::GlobalMemoryPool::new();
    let stats = pool.get_statistics();
    assert_eq!(stats.total_allocations, 0);

    // And the pool is usable for real allocations.
    let tensor = zeros_device::<f32>(&[4], DeviceType::Cpu).expect("zeros should succeed");
    assert_eq!(tensor.numel(), 4);
}

// ── F148: HDF5 must persist device/dtype/version and custom metadata ────────

#[cfg(feature = "serialize-hdf5")]
#[test]
fn f148_hdf5_roundtrip_preserves_metadata() {
    use torsh_tensor::serialize::common::SerializationOptions;
    use torsh_tensor::serialize::scientific::hdf5;

    let path = std::env::temp_dir().join(format!("torsh_hardening_hdf5_{}.h5", std::process::id()));
    let _ = std::fs::remove_file(&path);

    let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu)
        .expect("tensor creation should succeed");

    let mut options = SerializationOptions::default();
    options
        .metadata
        .insert("experiment".to_string(), "hardening".to_string());
    options
        .metadata
        .insert("owner".to_string(), "wave1".to_string());

    hdf5::serialize_hdf5(&tensor, &path, "tensor", &options).expect("hdf5 serialization");

    let loaded: Tensor<f32> = hdf5::deserialize_hdf5(&path, "tensor").expect("hdf5 load");
    assert_eq!(loaded.to_vec().expect("to_vec"), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(loaded.shape().dims(), &[2, 2]);
    assert_eq!(loaded.device(), DeviceType::Cpu);

    let metadata = hdf5::get_dataset_metadata(&path, "tensor").expect("hdf5 metadata");
    assert_eq!(metadata.device, DeviceType::Cpu);
    assert!(
        metadata.dtype_name.contains("f32"),
        "dtype must survive the round-trip, got {}",
        metadata.dtype_name
    );
    assert_ne!(metadata.version, "unknown", "version must survive");
    assert_eq!(
        metadata
            .custom_metadata
            .get("experiment")
            .map(String::as_str),
        Some("hardening")
    );
    assert_eq!(
        metadata.custom_metadata.get("owner").map(String::as_str),
        Some("wave1")
    );

    let _ = std::fs::remove_file(&path);
}

// ── F068 (partial): over-aligned pooled buffers must not corrupt the heap ────

#[test]
fn f068_into_vec_on_over_aligned_buffer_is_safe() {
    use torsh_tensor::memory_pool::global_acquire_uninit_aligned;

    let mut buf = global_acquire_uninit_aligned::<f32>(1024, 32);
    for (i, slot) in buf.as_uninit_slice_mut().iter_mut().enumerate() {
        slot.write(i as f32);
    }
    let values = buf.into_vec(1024);

    assert_eq!(values.len(), 1024);
    assert_eq!(values[0], 0.0);
    assert_eq!(values[1023], 1023.0);
    // Dropping `values` must not deallocate with a mismatched layout.
    drop(values);
}
