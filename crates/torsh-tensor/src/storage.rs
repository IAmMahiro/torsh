//! Storage management for tensor data
//!
//! This module provides storage abstractions for tensor data, including both
//! in-memory and memory-mapped storage options with automatic optimization
//! based on data size.
//!
//! # Features
//!
//! - **In-memory storage**: Fast access for smaller tensors
//! - **Memory-mapped storage**: Efficient for large tensors with caching
//! - **Automatic optimization**: Chooses optimal storage based on size
//! - **Cross-platform support**: Works on Unix, Windows, and other platforms
//! - **LRU cache management**: Optimizes memory usage for memory-mapped storage

use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
#[cfg(feature = "simd")]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

#[cfg(feature = "gpu")]
use torsh_core::sync::RwLockExt;
use torsh_core::{
    dtype::TensorElement,
    error::{Result, TorshError},
};

use crate::memory_pool::global_acquire_uninit;

// 🚀 SciRS2 AlignedVec integration for SIMD-optimized storage
#[cfg(feature = "simd")]
use scirs2_core::simd_aligned::AlignedVec;

#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;

/// Threshold for switching to memory-mapped storage (1 GB)
const MEMORY_MAPPING_THRESHOLD: usize = 1024 * 1024 * 1024;

/// Threshold for using aligned storage for SIMD optimization (1 KB)
/// Arrays larger than this benefit from cache-line aligned memory for SIMD operations
#[cfg(feature = "simd")]
const ALIGNED_STORAGE_THRESHOLD: usize = 1024;

/// Threshold for using lock-free SIMD storage (10 KB)
/// Arrays larger than this benefit from lock-free access patterns
#[cfg(feature = "simd")]
const SIMD_OPTIMIZED_THRESHOLD: usize = 10240;

// ============================================================================
// PHASE 5: SIMD-OPTIMIZED LOCK-FREE STORAGE
// ============================================================================
// Reads are lock-free (a single atomic flag check) for as long as the storage
// has never been written to; the first write copies the buffer once into a
// guarded copy-on-write buffer.
// Benefits:
// - No lock acquisition overhead for reads of read-only tensors (~20ns savings)
// - Direct slice access for SIMD operations
// - Mutation support at every tensor size, without ever mutating memory a
//   previously handed-out `&[T]` still points into
// ============================================================================

/// SIMD-optimized storage with Copy-on-Write semantics (Phase 5)
///
/// This storage variant eliminates lock overhead for read operations *while the
/// tensor has never been written to*, which is the dominant case for SIMD
/// workloads:
///
/// - The buffer handed to [`SimdStorage::new`] is **never mutated in place**, so
///   [`SimdStorage::try_as_slice`] can hand out a plain `&[T]` with no guard at
///   all (that is the "lock-free read" this variant exists for).
/// - The first write copies that buffer once into a private copy-on-write buffer
///   guarded by an `RwLock`; every later write mutates it in place, so a
///   `for i in 0..n { t.set(i, v) }` loop is O(n), not O(n²).
/// - Once the copy exists, readers go through the same `RwLock`, exactly like
///   [`TensorStorage::Aligned`]. Slices handed out earlier stay valid because the
///   original buffer is kept alive and untouched for the storage's lifetime.
///
/// This is what makes `set`/`set_slice`/`with_slice_mut` work on tensors of every
/// size instead of failing above the 10 KB `SimdOptimized` threshold.
#[cfg(feature = "simd")]
pub struct SimdStorage<T> {
    /// The buffer published at construction. Immutable for the whole lifetime of
    /// this storage, which is what makes lock-free slice hand-out sound.
    original: AlignedVec<T>,
    /// Copy-on-write buffer holding the authoritative data once a write happened.
    cow: RwLock<Option<AlignedVec<T>>>,
    /// Lock-free flag: `true` once `cow` holds the authoritative data.
    mutated: AtomicBool,
    /// Whether this storage is shared with another `TensorStorage` handle
    shared: AtomicBool,
}

#[cfg(feature = "simd")]
impl<T> SimdStorage<T> {
    /// Create new SIMD storage from data
    pub fn new(data: AlignedVec<T>) -> Self {
        Self {
            original: data,
            cow: RwLock::new(None),
            mutated: AtomicBool::new(false),
            shared: AtomicBool::new(false),
        }
    }

    /// Get the length of the storage
    ///
    /// Mutation never changes the element count, so this stays lock-free.
    pub fn len(&self) -> usize {
        self.original.len()
    }

    /// Check if storage is empty
    pub fn is_empty(&self) -> bool {
        self.original.is_empty()
    }

    /// Get the capacity
    pub fn capacity(&self) -> usize {
        self.original.capacity()
    }

    /// Whether this storage has been written to (and therefore keeps its
    /// authoritative data in the guarded copy-on-write buffer).
    pub fn is_mutated(&self) -> bool {
        self.mutated.load(Ordering::Acquire)
    }

    /// Get the immutable, lock-free slice.
    ///
    /// Returns `None` once the storage has been written to: after that the
    /// authoritative data lives behind a lock and cannot be exposed as an
    /// unguarded borrow. Callers should fall back to
    /// [`SimdStorage::with_slice`].
    pub fn try_as_slice(&self) -> Option<&[T]> {
        if self.is_mutated() {
            None
        } else {
            Some(self.original.as_slice())
        }
    }

    /// Mark as shared (for Clone)
    pub fn mark_shared(&self) {
        self.shared.store(true, Ordering::SeqCst);
    }

    /// Check if shared
    pub fn is_shared(&self) -> bool {
        self.shared.load(Ordering::SeqCst)
    }
}

#[cfg(feature = "simd")]
impl<T: Copy> SimdStorage<T> {
    /// Read the storage contents, lock-free while it has never been written to.
    pub fn with_slice<R>(&self, f: impl FnOnce(&[T]) -> R) -> R {
        if !self.is_mutated() {
            return f(self.original.as_slice());
        }
        // A poisoned lock still holds structurally valid data (the only panic
        // path while it is held is an allocation failure before publication),
        // so recover rather than fail every subsequent read.
        let guard = self.cow.read().unwrap_or_else(|e| e.into_inner());
        match guard.as_ref() {
            Some(buffer) => f(buffer.as_slice()),
            // `mutated` is only set after `cow` is populated, so this is
            // unreachable; fall back to the original rather than panic.
            None => f(self.original.as_slice()),
        }
    }

    /// Mutate the storage contents, promoting to the copy-on-write buffer on the
    /// first call.
    pub fn with_slice_mut<R>(&self, f: impl FnOnce(&mut [T]) -> R) -> Result<R> {
        let mut guard = self.cow.write().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            let source = self.original.as_slice();
            let mut buffer = AlignedVec::with_capacity(source.len()).map_err(|e| {
                TorshError::InvalidArgument(format!("Failed to create SIMD COW buffer: {e}"))
            })?;
            if !source.is_empty() {
                // SAFETY: `buffer` has capacity for `source.len()` elements of
                // `T` and the regions do not overlap; `T: Copy` so a bitwise
                // copy is a valid initialization.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        source.as_ptr(),
                        buffer.as_mut_ptr(),
                        source.len(),
                    );
                    buffer.set_len(source.len());
                }
            }
            *guard = Some(buffer);
            // Publish only after the buffer is populated: a reader that observes
            // `mutated == true` is guaranteed to find `cow` initialized.
            self.mutated.store(true, Ordering::Release);
        }

        match guard.as_mut() {
            Some(buffer) => Ok(f(buffer.as_mut_slice())),
            None => Err(TorshError::SynchronizationError(
                "SIMD copy-on-write buffer disappeared".to_string(),
            )),
        }
    }

    /// Convert to Vec (copying data)
    pub fn to_vec(&self) -> Vec<T> {
        self.with_slice(|slice| slice.to_vec())
    }
}

#[cfg(feature = "simd")]
impl<T> std::fmt::Debug for SimdStorage<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimdStorage")
            .field("len", &self.original.len())
            .field("mutated", &self.mutated.load(Ordering::Relaxed))
            .field("shared", &self.shared.load(Ordering::Relaxed))
            .finish()
    }
}

// ============================================================================
// DEVICE-RESIDENT STORAGE
// ============================================================================
// A tensor whose data lives in device memory keeps only a pointer here; the
// host copy is materialised lazily and at most once. That is what turns a chain
// of GPU ops into "one upload, one download" instead of a host round trip per
// operation.
// ============================================================================

/// An owned device allocation.
///
/// **A device buffer is immutable for its whole life.** ToRSh writes it exactly
/// once, as the output of a backend op, and every mutation path on a
/// device-resident tensor demotes the tensor to host storage first (see
/// [`crate::Tensor::make_unique`]). Two properties follow, and the rest of the
/// design rests on them:
///
/// - the lazily downloaded host copy in [`TensorStorage::Device`] never needs
///   invalidating, and
/// - no device-to-device copy is required — which matters, because
///   [`oxicuda_backend::ComputeBackend`] does not provide one.
///
/// The buffer owns its pointer: `adopt` is the only constructor,
/// and `Drop` releases the allocation. It also owns a handle on the backend that
/// allocated it, so freeing consults no registry, takes no ToRSh lock, and stays
/// correct even after a different backend has been installed.
#[cfg(feature = "gpu")]
pub struct DeviceBuffer {
    /// Device pointer owned by this buffer.
    ptr: u64,
    /// Size of the allocation in bytes.
    bytes: usize,
    /// Element type the bytes encode.
    dtype: torsh_core::dtype::DType,
    /// The backend that allocated `ptr`, and the only one that may free it.
    backend: Arc<dyn oxicuda_backend::ComputeBackend>,
}

#[cfg(feature = "gpu")]
impl DeviceBuffer {
    /// Take ownership of `ptr`, which `backend` allocated with `bytes` bytes.
    ///
    /// The caller must not free `ptr` afterwards and must never hand the same
    /// pointer to a second `DeviceBuffer`: `Drop` frees it exactly once.
    pub(crate) fn adopt(
        ptr: u64,
        bytes: usize,
        dtype: torsh_core::dtype::DType,
        backend: Arc<dyn oxicuda_backend::ComputeBackend>,
    ) -> Self {
        Self {
            ptr,
            bytes,
            dtype,
            backend,
        }
    }

    /// The device pointer this buffer owns.
    pub fn ptr(&self) -> u64 {
        self.ptr
    }

    /// Size of the allocation in bytes.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Element type the buffer's bytes encode.
    pub fn dtype(&self) -> torsh_core::dtype::DType {
        self.dtype
    }

    /// The backend that owns this allocation.
    pub fn backend(&self) -> &Arc<dyn oxicuda_backend::ComputeBackend> {
        &self.backend
    }
}

#[cfg(feature = "gpu")]
impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        // Deliberately infallible: `drop` may run while unwinding, so a failed
        // free must never panic, and no ToRSh lock is taken here.
        let _ = self.backend.free(self.ptr);
    }
}

#[cfg(feature = "gpu")]
impl std::fmt::Debug for DeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("ptr", &format_args!("{:#x}", self.ptr))
            .field("bytes", &self.bytes)
            .field("dtype", &self.dtype)
            .field("backend", &self.backend.name())
            .finish()
    }
}

/// Storage abstraction for tensor data
pub enum TensorStorage<T: TensorElement> {
    /// In-memory storage for smaller tensors
    InMemory(Arc<RwLock<Vec<T>>>),
    /// Memory-mapped storage for large tensors
    MemoryMapped(Arc<RwLock<MemoryMappedStorage<T>>>),
    /// Cache-line aligned storage for SIMD-optimized operations (14.17x speedup)
    #[cfg(feature = "simd")]
    Aligned(Arc<RwLock<AlignedVec<T>>>),
    /// 🚀 Lock-free SIMD storage with Copy-on-Write semantics (Phase 5)
    ///
    /// Benefits:
    /// - Lock-free read access (~20ns savings per operation)
    /// - Direct slice access for SIMD
    /// - Thread-safe through atomic COW
    #[cfg(feature = "simd")]
    SimdOptimized(Arc<SimdStorage<T>>),
    /// Device-resident storage: the data lives in GPU memory.
    ///
    /// Produced by the residency path in [`crate::gpu_dispatch`], so a chain of
    /// device ops never returns to the host between operations.
    #[cfg(feature = "gpu")]
    Device {
        /// The device allocation holding this tensor's data.
        buffer: Arc<DeviceBuffer>,
        /// Host copy, downloaded on the first host-side read and kept
        /// afterwards.
        ///
        /// It never needs invalidating: the device buffer is immutable, because
        /// every mutation path demotes the tensor to host storage first.
        host_cache: Arc<RwLock<Option<Vec<T>>>>,
    },
}

impl<T: TensorElement> std::fmt::Debug for TensorStorage<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InMemory(data) => f.debug_tuple("InMemory").field(data).finish(),
            Self::MemoryMapped(storage) => f.debug_tuple("MemoryMapped").field(storage).finish(),
            #[cfg(feature = "simd")]
            Self::Aligned(_) => f.debug_tuple("Aligned").field(&"<AlignedVec>").finish(),
            #[cfg(feature = "simd")]
            Self::SimdOptimized(storage) => f.debug_tuple("SimdOptimized").field(storage).finish(),
            #[cfg(feature = "gpu")]
            Self::Device { buffer, .. } => f.debug_tuple("Device").field(buffer).finish(),
        }
    }
}

/// Memory-mapped storage implementation
#[derive(Debug)]
pub struct MemoryMappedStorage<T: TensorElement> {
    /// File backing the memory mapping
    file: File,
    /// Path to the backing file
    file_path: PathBuf,
    /// Number of elements stored
    num_elements: usize,
    /// Cache for frequently accessed elements
    cache: HashMap<usize, T>,
    /// Maximum cache size
    max_cache_size: usize,
    /// Access pattern tracking for cache optimization
    access_pattern: VecDeque<usize>,
    /// Whether the storage is temporary (should be deleted on drop)
    is_temporary: bool,
}

impl<T: TensorElement + Copy> TensorStorage<T> {
    /// Create in-memory storage
    pub fn in_memory(data: Vec<T>) -> Self {
        Self::InMemory(Arc::new(RwLock::new(data)))
    }

    /// Create memory-mapped storage
    pub fn memory_mapped(data: Vec<T>, file_path: Option<PathBuf>) -> Result<Self> {
        let storage = MemoryMappedStorage::new(data, file_path)?;
        Ok(Self::MemoryMapped(Arc::new(RwLock::new(storage))))
    }

    /// Create disk-backed storage of `num_elements` copies of `value` without
    /// ever holding the whole tensor in RAM.
    ///
    /// This is the storage behind [`crate::Tensor::disk_backed`]: the backing
    /// file is filled in bounded chunks, so datasets larger than available
    /// memory can be created.
    pub fn memory_mapped_filled(
        num_elements: usize,
        value: T,
        file_path: Option<PathBuf>,
    ) -> Result<Self> {
        let storage = MemoryMappedStorage::new_filled(num_elements, value, file_path)?;
        Ok(Self::MemoryMapped(Arc::new(RwLock::new(storage))))
    }

    /// Create cache-line aligned storage for SIMD operations (14.17x speedup potential)
    #[cfg(feature = "simd")]
    pub fn aligned(data: Vec<T>) -> Result<Self> {
        Ok(Self::Aligned(Arc::new(RwLock::new(Self::to_aligned_vec(
            &data,
        )?))))
    }

    /// Create cache-line aligned storage directly from a borrowed slice.
    ///
    /// Same result as [`TensorStorage::aligned`] without the intermediate `Vec`:
    /// callers that already hold (or can borrow) the source data pay one
    /// allocation and one bulk copy instead of two of each.
    #[cfg(feature = "simd")]
    pub(crate) fn aligned_from_slice(data: &[T]) -> Result<Self> {
        Ok(Self::Aligned(Arc::new(RwLock::new(Self::to_aligned_vec(
            data,
        )?))))
    }

    /// Bulk-copy `data` into a freshly allocated [`AlignedVec`].
    ///
    /// This is one `copy_nonoverlapping` rather than a per-element `push` with a
    /// capacity check per iteration, which is what every tensor ≥ 1 KB pays on
    /// construction.
    #[cfg(feature = "simd")]
    fn to_aligned_vec(data: &[T]) -> Result<AlignedVec<T>> {
        let mut aligned_vec = AlignedVec::with_capacity(data.len()).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to create aligned storage: {e}"))
        })?;

        if !data.is_empty() {
            // SAFETY: `aligned_vec` was allocated with capacity for `data.len()`
            // elements of `T`, the two regions cannot overlap (the destination
            // was just allocated), and `T: Copy` so a bitwise copy is a valid
            // initialization of the destination elements.
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), aligned_vec.as_mut_ptr(), data.len());
                aligned_vec.set_len(data.len());
            }
        }

        Ok(aligned_vec)
    }

    /// 🚀 **Phase 7**: Create fast result storage (skips alignment copy)
    ///
    /// For SIMD operation results where we already have the data in a Vec,
    /// uses InMemory storage to avoid the ~10µs alignment copy overhead.
    ///
    /// # Performance
    /// - Skips AlignedVec copy (saves ~10µs for 50K elements)
    /// - Uses InMemory storage (has RwLock but we just created it)
    /// - Optimal for result tensors that won't be immediately used in SIMD ops
    pub fn fast_result(data: Vec<T>) -> Self {
        Self::InMemory(Arc::new(RwLock::new(data)))
    }

    /// 🚀 **Phase 5**: Create lock-free SIMD-optimized storage
    ///
    /// This storage variant eliminates RwLock overhead for reads:
    /// - Lock-free read access (~20ns savings per operation)
    /// - Direct slice access for SIMD operations
    /// - Thread-safe through Copy-on-Write semantics
    ///
    /// # Performance
    /// - Best for medium-to-large tensors (> 10KB)
    /// - Optimal for SIMD operations that read but rarely write
    /// - **Note**: Has ~10µs alignment copy overhead for 50K elements
    #[cfg(feature = "simd")]
    pub fn simd_optimized(data: Vec<T>) -> Result<Self> {
        let aligned_vec = Self::to_aligned_vec(&data)?;
        let simd_storage = SimdStorage::new(aligned_vec);
        Ok(Self::SimdOptimized(Arc::new(simd_storage)))
    }

    /// Create storage automatically based on size and performance characteristics
    ///
    /// **Storage Selection Strategy**:
    /// - Very large (>1GB): Memory-mapped for virtual memory efficiency
    /// - Large (>10KB, SIMD enabled): SimdOptimized (lock-free reads)
    /// - Medium (>1KB, SIMD enabled): Aligned storage for SIMD alignment
    /// - Small (<1KB): In-memory with RwLock
    pub fn create_optimal(data: Vec<T>) -> Result<Self> {
        let size_bytes = data.len() * std::mem::size_of::<T>();

        if size_bytes >= MEMORY_MAPPING_THRESHOLD {
            // Very large data: use memory mapping
            Self::memory_mapped(data, None)
        } else {
            #[cfg(feature = "simd")]
            {
                if size_bytes >= SIMD_OPTIMIZED_THRESHOLD {
                    // Large data: use lock-free SimdOptimized storage
                    // This eliminates RwLock overhead for read operations
                    return Self::simd_optimized(data);
                } else if size_bytes >= ALIGNED_STORAGE_THRESHOLD {
                    // Medium data: use aligned storage for SIMD alignment
                    return Self::aligned(data);
                }
            }
            // Small data: use regular in-memory storage
            Ok(Self::in_memory(data))
        }
    }

    /// Wrap an owned device allocation as device-resident storage.
    ///
    /// The host cache starts empty and is filled by the first host-side read.
    #[cfg(feature = "gpu")]
    pub(crate) fn device(buffer: Arc<DeviceBuffer>) -> Self {
        Self::Device {
            buffer,
            host_cache: Arc::new(RwLock::new(None)),
        }
    }

    /// Whether this storage's data lives in device memory.
    ///
    /// Always `false` without the `gpu` feature, so callers that must demote
    /// before a write can test it unconditionally.
    pub fn is_device(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            matches!(self, Self::Device { .. })
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }

    /// The device allocation backing this storage, if it is device-resident.
    #[cfg(feature = "gpu")]
    pub(crate) fn device_buffer(&self) -> Option<&Arc<DeviceBuffer>> {
        match self {
            Self::Device { buffer, .. } => Some(buffer),
            _ => None,
        }
    }

    /// Run `f` against the host copy of a device buffer, downloading it first if
    /// this is the first host-side read.
    ///
    /// This is the **single** download point for device-resident storage: every
    /// later read is served from the cache, which is what keeps a materialised
    /// view or a per-element `get` loop from re-transferring the whole tensor.
    #[cfg(feature = "gpu")]
    fn with_host_cache<R, F>(
        buffer: &Arc<DeviceBuffer>,
        host_cache: &RwLock<Option<Vec<T>>>,
        f: F,
    ) -> Result<R>
    where
        F: FnOnce(&[T]) -> Result<R>,
        T: Copy,
    {
        // Fast path: serve from the cache under a read guard, exactly like the
        // `InMemory` arm does.
        {
            let guard = host_cache.read_or_recover();
            if let Some(cached) = guard.as_ref() {
                return f(cached);
            }
        }

        // Cold path: download while holding no guard at all, then publish. The
        // write guard is never held across user code, so a closure that reads
        // this storage again cannot dead-lock against the download.
        let downloaded = Self::download(buffer)?;
        {
            let mut guard = host_cache.write_or_recover();
            if guard.is_none() {
                *guard = Some(downloaded);
            }
        }

        let guard = host_cache.read_or_recover();
        match guard.as_ref() {
            Some(cached) => f(cached),
            None => Err(TorshError::SynchronizationError(
                "device host cache disappeared".to_string(),
            )),
        }
    }

    /// Copy a device buffer's contents into a freshly allocated host `Vec<T>`.
    #[cfg(feature = "gpu")]
    fn download(buffer: &Arc<DeviceBuffer>) -> Result<Vec<T>>
    where
        T: Copy,
    {
        let element_size = std::mem::size_of::<T>();
        if element_size == 0 || buffer.bytes() % element_size != 0 {
            return Err(TorshError::InvalidOperation(format!(
                "device buffer of {} bytes does not hold whole {}-byte elements",
                buffer.bytes(),
                element_size
            )));
        }
        // Checked rather than assumed: this is what makes the reinterpret below
        // sound without relying on a guard in another module.
        if buffer.dtype() != T::dtype() {
            return Err(TorshError::InvalidOperation(format!(
                "device buffer holds {} but the tensor element type is {}",
                buffer.dtype(),
                T::dtype()
            )));
        }

        let count = buffer.bytes() / element_size;
        let mut raw = vec![0u8; buffer.bytes()];
        buffer
            .backend()
            .copy_dtoh(&mut raw, buffer.ptr())
            .map_err(|e| TorshError::InvalidOperation(format!("device download failed: {e}")))?;

        let mut out: Vec<T> = Vec::with_capacity(count);
        if count > 0 {
            // SAFETY: `out` was allocated by `Vec<T>` with capacity for `count`
            // elements, so it is `T`-aligned and spans exactly `buffer.bytes()`
            // bytes; `raw` was just allocated, so the regions cannot overlap.
            // The dtype check above establishes that the downloaded bytes are a
            // valid representation of `T`, and `T: Copy` makes a bitwise copy a
            // valid initialization.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    raw.as_ptr(),
                    out.as_mut_ptr().cast::<u8>(),
                    buffer.bytes(),
                );
                out.set_len(count);
            }
        }
        Ok(out)
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        match self {
            Self::InMemory(data) => {
                data.read().map(|guard| guard.len()).unwrap_or(0) // If lock is poisoned, return 0 (safe fallback)
            }
            Self::MemoryMapped(storage) => {
                storage.read().map(|guard| guard.num_elements).unwrap_or(0) // If lock is poisoned, return 0 (safe fallback)
            }
            #[cfg(feature = "simd")]
            Self::Aligned(data) => {
                data.read().map(|guard| guard.len()).unwrap_or(0) // If lock is poisoned, return 0 (safe fallback)
            }
            #[cfg(feature = "simd")]
            Self::SimdOptimized(storage) => storage.len(), // Lock-free!
            #[cfg(feature = "gpu")]
            Self::Device { buffer, .. } => {
                // Derived from the allocation size: no download, no lock.
                let element_size = std::mem::size_of::<T>();
                if element_size == 0 {
                    0
                } else {
                    buffer.bytes() / element_size
                }
            }
        }
    }

    /// Check if storage is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get element at index
    pub fn get(&self, index: usize) -> Result<T>
    where
        T: Copy,
    {
        match self {
            Self::InMemory(data) => {
                let data_guard = data.read().map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during read".to_string())
                })?;
                data_guard
                    .get(index)
                    .copied()
                    .ok_or_else(|| TorshError::IndexOutOfBounds {
                        index,
                        size: data_guard.len(),
                    })
            }
            Self::MemoryMapped(storage) => storage
                .write()
                .map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during write".to_string())
                })?
                .get(index),
            #[cfg(feature = "simd")]
            Self::Aligned(data) => {
                let data_guard = data.read().map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during read".to_string())
                })?;
                if index >= data_guard.len() {
                    Err(TorshError::IndexOutOfBounds {
                        index,
                        size: data_guard.len(),
                    })
                } else {
                    Ok(data_guard.as_slice()[index])
                }
            }
            #[cfg(feature = "simd")]
            Self::SimdOptimized(storage) => {
                // Lock-free while the storage has never been written to.
                storage.with_slice(|slice| {
                    slice
                        .get(index)
                        .copied()
                        .ok_or_else(|| TorshError::IndexOutOfBounds {
                            index,
                            size: slice.len(),
                        })
                })
            }
            #[cfg(feature = "gpu")]
            Self::Device { buffer, host_cache } => {
                Self::with_host_cache(buffer, host_cache, |slice| {
                    slice
                        .get(index)
                        .copied()
                        .ok_or_else(|| TorshError::IndexOutOfBounds {
                            index,
                            size: slice.len(),
                        })
                })
            }
        }
    }

    /// Set element at index
    pub fn set(&self, index: usize, value: T) -> Result<()>
    where
        T: Copy,
    {
        match self {
            Self::InMemory(data) => {
                let mut data_guard = data.write().map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during write".to_string())
                })?;
                if index >= data_guard.len() {
                    return Err(TorshError::IndexOutOfBounds {
                        index,
                        size: data_guard.len(),
                    });
                }
                data_guard[index] = value;
                Ok(())
            }
            Self::MemoryMapped(storage) => storage
                .write()
                .map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during write".to_string())
                })?
                .set(index, value),
            #[cfg(feature = "simd")]
            Self::Aligned(data) => {
                let mut data_guard = data.write().map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during write".to_string())
                })?;
                if index >= data_guard.len() {
                    return Err(TorshError::IndexOutOfBounds {
                        index,
                        size: data_guard.len(),
                    });
                }
                // Use the new set() method from AlignedVec
                (*data_guard).set(index, value);
                Ok(())
            }
            #[cfg(feature = "simd")]
            Self::SimdOptimized(storage) => {
                // Copy-on-write: the first write promotes the immutable buffer
                // into a guarded mutable copy, later writes go straight in.
                storage.with_slice_mut(|slice| {
                    let size = slice.len();
                    match slice.get_mut(index) {
                        Some(slot) => {
                            *slot = value;
                            Ok(())
                        }
                        None => Err(TorshError::IndexOutOfBounds { index, size }),
                    }
                })?
            }
            #[cfg(feature = "gpu")]
            Self::Device { .. } => Err(Self::device_is_immutable()),
        }
    }

    /// The error every in-place write on device-resident storage returns.
    ///
    /// Device buffers are immutable by construction (see [`DeviceBuffer`]), so
    /// the tensor must be demoted to host storage before it can be written.
    #[cfg(feature = "gpu")]
    fn device_is_immutable() -> TorshError {
        TorshError::InvalidOperation(
            "device-resident storage is immutable; call make_unique() or to_device(DeviceType::Cpu) first"
                .to_string(),
        )
    }

    /// Get multiple elements
    pub fn get_slice(&self, start: usize, len: usize) -> Result<Vec<T>>
    where
        T: Copy,
    {
        match self {
            Self::InMemory(data) => {
                let data_guard = data.read().map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during read".to_string())
                })?;
                if start + len > data_guard.len() {
                    return Err(TorshError::IndexOutOfBounds {
                        index: start + len - 1,
                        size: data_guard.len(),
                    });
                }
                Ok(data_guard[start..start + len].to_vec())
            }
            Self::MemoryMapped(storage) => storage
                .write()
                .map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during write".to_string())
                })?
                .get_slice(start, len),
            #[cfg(feature = "simd")]
            Self::Aligned(data) => {
                let data_guard = data.read().map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during read".to_string())
                })?;
                if start + len > data_guard.len() {
                    return Err(TorshError::IndexOutOfBounds {
                        index: start + len - 1,
                        size: data_guard.len(),
                    });
                }
                let slice = data_guard.as_slice();
                Ok(slice[start..start + len].to_vec())
            }
            #[cfg(feature = "simd")]
            Self::SimdOptimized(storage) => storage.with_slice(|slice| {
                if start + len > slice.len() {
                    return Err(TorshError::IndexOutOfBounds {
                        index: start + len - 1,
                        size: slice.len(),
                    });
                }
                Ok(slice[start..start + len].to_vec())
            }),
            #[cfg(feature = "gpu")]
            Self::Device { buffer, host_cache } => {
                Self::with_host_cache(buffer, host_cache, |slice| {
                    if start + len > slice.len() {
                        return Err(TorshError::IndexOutOfBounds {
                            index: start + len - 1,
                            size: slice.len(),
                        });
                    }
                    Ok(slice[start..start + len].to_vec())
                })
            }
        }
    }

    /// Set multiple elements
    pub fn set_slice(&self, start: usize, values: &[T]) -> Result<()>
    where
        T: Copy,
    {
        match self {
            Self::InMemory(data) => {
                let mut data_guard = data.write().map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during write".to_string())
                })?;
                if start + values.len() > data_guard.len() {
                    return Err(TorshError::IndexOutOfBounds {
                        index: start + values.len() - 1,
                        size: data_guard.len(),
                    });
                }
                data_guard[start..start + values.len()].copy_from_slice(values);
                Ok(())
            }
            Self::MemoryMapped(storage) => storage
                .write()
                .map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during write".to_string())
                })?
                .set_slice(start, values),
            #[cfg(feature = "simd")]
            Self::Aligned(data) => {
                let mut data_guard = data.write().map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during write".to_string())
                })?;
                if start + values.len() > data_guard.len() {
                    return Err(TorshError::IndexOutOfBounds {
                        index: start + values.len() - 1,
                        size: data_guard.len(),
                    });
                }
                // Use as_mut_slice() and copy
                let slice = data_guard.as_mut_slice();
                slice[start..start + values.len()].copy_from_slice(values);
                Ok(())
            }
            #[cfg(feature = "simd")]
            Self::SimdOptimized(storage) => storage.with_slice_mut(|slice| {
                let size = slice.len();
                if start + values.len() > size {
                    return Err(TorshError::IndexOutOfBounds {
                        index: start + values.len() - 1,
                        size,
                    });
                }
                slice[start..start + values.len()].copy_from_slice(values);
                Ok(())
            })?,
            #[cfg(feature = "gpu")]
            Self::Device { .. } => Err(Self::device_is_immutable()),
        }
    }

    /// Convert to vector (useful for small tensors or debugging)
    pub fn to_vec(&self) -> Result<Vec<T>>
    where
        T: Copy,
    {
        match self {
            Self::InMemory(data) => Ok(data
                .read()
                .map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during read".to_string())
                })?
                .clone()),
            Self::MemoryMapped(storage) => storage
                .write()
                .map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during write".to_string())
                })?
                .to_vec(),
            #[cfg(feature = "simd")]
            Self::Aligned(data) => {
                let data_guard = data.read().map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during read".to_string())
                })?;
                Ok(data_guard.as_slice().to_vec())
            }
            #[cfg(feature = "simd")]
            Self::SimdOptimized(storage) => Ok(storage.to_vec()),
            #[cfg(feature = "gpu")]
            Self::Device { buffer, host_cache } => {
                Self::with_host_cache(buffer, host_cache, |slice| Ok(slice.to_vec()))
            }
        }
    }

    /// Get storage type information
    pub fn storage_type(&self) -> &'static str {
        match self {
            Self::InMemory(_) => "in_memory",
            Self::MemoryMapped(_) => "memory_mapped",
            #[cfg(feature = "simd")]
            Self::Aligned(_) => "aligned_simd",
            #[cfg(feature = "simd")]
            Self::SimdOptimized(_) => "simd_optimized",
            #[cfg(feature = "gpu")]
            Self::Device { .. } => "device",
        }
    }

    /// Get estimated memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        match self {
            Self::InMemory(data) => {
                data.read()
                    .map(|guard| guard.len() * std::mem::size_of::<T>())
                    .unwrap_or(0) // If lock is poisoned, return 0 (safe fallback)
            }
            Self::MemoryMapped(storage) => {
                storage
                    .read()
                    .map(|storage_guard| {
                        // Memory usage is just the cache size plus metadata
                        storage_guard.cache.len() * std::mem::size_of::<T>()
                            + std::mem::size_of::<MemoryMappedStorage<T>>()
                    })
                    .unwrap_or(std::mem::size_of::<MemoryMappedStorage<T>>()) // Fallback to metadata size
            }
            #[cfg(feature = "simd")]
            Self::Aligned(data) => {
                data.read()
                    .map(|data_guard| {
                        // AlignedVec uses more memory due to alignment padding
                        data_guard.capacity() * std::mem::size_of::<T>()
                    })
                    .unwrap_or(0) // If lock is poisoned, return 0 (safe fallback)
            }
            #[cfg(feature = "simd")]
            Self::SimdOptimized(storage) => {
                // After the first write the copy-on-write buffer is held
                // alongside the (retained) original one.
                let buffers = if storage.is_mutated() { 2 } else { 1 };
                storage.capacity() * std::mem::size_of::<T>() * buffers
            }
            #[cfg(feature = "gpu")]
            Self::Device { buffer, host_cache } => {
                // The device allocation, plus the host copy once one exists.
                let cached = host_cache
                    .read_or_recover()
                    .as_ref()
                    .map_or(0, |cache| cache.len() * std::mem::size_of::<T>());
                buffer.bytes() + cached
            }
        }
    }

    /// Execute a function with immutable access to data slice (zero-copy within scope)
    ///
    /// This enables zero-copy SIMD operations by providing direct `&[T]` access
    /// within the closure scope while the lock is held.
    ///
    /// # Arguments
    /// * `f` - Closure that receives `&[T]` and returns `Result<R>`
    ///
    /// # Returns
    /// Result from the closure
    ///
    /// # Performance
    /// - Zero allocations for in-memory and aligned storage
    /// - Converts memory-mapped storage to Vec (one allocation)
    ///
    /// # Examples
    /// ```ignore
    /// storage.with_slice(|data| {
    ///     // Direct SIMD access to data
    ///     f32::simd_add(&data, &other_data)
    /// })?;
    /// ```
    pub fn with_slice<R, F>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&[T]) -> Result<R>,
        T: Copy,
    {
        match self {
            Self::InMemory(data) => {
                let data_guard = data.read().map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during read".to_string())
                })?;
                f(data_guard.as_slice())
            }
            Self::MemoryMapped(storage) => {
                // Memory-mapped storage requires conversion to Vec
                let vec = storage
                    .write()
                    .map_err(|_| {
                        TorshError::SynchronizationError("Lock poisoned during write".to_string())
                    })?
                    .to_vec()?;
                f(&vec)
            }
            #[cfg(feature = "simd")]
            Self::Aligned(data) => {
                let data_guard = data.read().map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during read".to_string())
                })?;
                f(data_guard.as_slice())
            }
            #[cfg(feature = "simd")]
            Self::SimdOptimized(storage) => {
                // 🚀 Lock-free access while the storage has never been written to.
                storage.with_slice(f)
            }
            #[cfg(feature = "gpu")]
            Self::Device { buffer, host_cache } => Self::with_host_cache(buffer, host_cache, f),
        }
    }

    /// Try to get direct slice access without closures (only works for SimdOptimized)
    ///
    /// Returns `Some(&[T])` if storage is SimdOptimized **and** has never been
    /// written to (in which case its buffer is immutable and an unguarded borrow
    /// is sound). Returns `None` for every other storage type, and for a
    /// SimdOptimized storage that has been mutated — callers must fall back to
    /// [`TensorStorage::with_slice`].
    ///
    /// # Performance
    /// - SimdOptimized (unmutated): Direct slice access, zero overhead
    /// - Others: Returns None (use with_slice instead)
    #[cfg(feature = "simd")]
    pub fn try_as_slice_direct(&self) -> Option<&[T]> {
        match self {
            Self::SimdOptimized(storage) => storage.try_as_slice(),
            _ => None,
        }
    }

    /// Execute a function with mutable access to data slice (zero-copy within scope)
    ///
    /// This enables zero-copy in-place operations by providing direct `&mut [T]` access
    /// within the closure scope while the lock is held.
    ///
    /// # Arguments
    /// * `f` - Closure that receives `&mut [T]` and returns `Result<R>`
    ///
    /// # Returns
    /// Result from the closure
    ///
    /// # Performance
    /// - Zero allocations for in-memory storage
    /// - Aligned storage not yet supported (returns error)
    /// - Memory-mapped storage not supported for mutable access (returns error)
    ///
    /// # Examples
    /// ```ignore
    /// storage.with_slice_mut(|data| {
    ///     // In-place SIMD operation
    ///     f32::simd_add_inplace(data, &other_data)
    /// })?;
    /// ```
    pub fn with_slice_mut<R, F>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&mut [T]) -> Result<R>,
        T: Copy,
    {
        match self {
            Self::InMemory(data) => {
                let mut data_guard = data.write().map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during write".to_string())
                })?;
                f(data_guard.as_mut_slice())
            }
            Self::MemoryMapped(_) => {
                // Memory-mapped storage doesn't support mutable slice access
                Err(TorshError::InvalidArgument(
                    "Memory-mapped storage does not support mutable slice access".to_string(),
                ))
            }
            #[cfg(feature = "simd")]
            Self::Aligned(data) => {
                let mut data_guard = data.write().map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during write".to_string())
                })?;
                f(data_guard.as_mut_slice())
            }
            #[cfg(feature = "simd")]
            Self::SimdOptimized(storage) => {
                // Copy-on-write promotion, then a direct mutable slice.
                storage.with_slice_mut(f)?
            }
            #[cfg(feature = "gpu")]
            Self::Device { .. } => Err(Self::device_is_immutable()),
        }
    }
}

/// Build a backing-file path that is unique per storage instance.
///
/// A single per-process name would make two temporary memory-mapped tensors
/// share (and truncate) one file, so the name carries the pid, a monotonic
/// counter and a nanosecond timestamp.
fn unique_backing_path() -> PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "torsh_tensor_{pid}_{nanos}_{seq}.mmap",
        pid = std::process::id()
    ))
}

impl<T: TensorElement> MemoryMappedStorage<T> {
    /// Open (or create) the backing file for a storage instance.
    fn open_backing_file(file_path: Option<PathBuf>) -> Result<(File, PathBuf, bool)> {
        let (file_path, is_temporary) = match file_path {
            Some(path) => (path, false),
            // Unique per storage instance: sharing one name per process made
            // every new temporary tensor truncate the previous one's data.
            None => (unique_backing_path(), true),
        };

        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&file_path)
            .map_err(|e| {
                TorshError::IoError(format!("Failed to create memory-mapped file: {e}"))
            })?;

        Ok((file, file_path, is_temporary))
    }

    /// Create new memory-mapped storage
    pub fn new(data: Vec<T>, file_path: Option<PathBuf>) -> Result<Self> {
        let (mut file, file_path, is_temporary) = Self::open_backing_file(file_path)?;

        // Write data to file
        let data_bytes = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                std::mem::size_of_val(data.as_slice()),
            )
        };
        file.write_all(data_bytes).map_err(|e| {
            TorshError::IoError(format!("Failed to write to memory-mapped file: {e}"))
        })?;
        file.flush()
            .map_err(|e| TorshError::IoError(format!("Failed to flush memory-mapped file: {e}")))?;

        Ok(Self {
            file,
            file_path,
            num_elements: data.len(),
            cache: HashMap::new(),
            max_cache_size: 10000, // Cache up to 10k elements
            access_pattern: VecDeque::new(),
            is_temporary,
        })
    }

    /// Create memory-mapped storage of `num_elements` copies of `value` **without
    /// materialising the tensor in RAM**.
    ///
    /// The backing file is written in bounded chunks, so a tensor larger than
    /// available memory can be created: peak resident memory is the chunk size,
    /// not the tensor size.
    pub fn new_filled(num_elements: usize, value: T, file_path: Option<PathBuf>) -> Result<Self>
    where
        T: Copy,
    {
        let (mut file, file_path, is_temporary) = Self::open_backing_file(file_path)?;

        let element_size = std::mem::size_of::<T>();
        if element_size > 0 && num_elements > 0 {
            // Write in ~1 MiB chunks so peak RAM stays bounded.
            const TARGET_CHUNK_BYTES: usize = 1024 * 1024;
            let chunk_elements = (TARGET_CHUNK_BYTES / element_size).clamp(1, num_elements);
            let chunk = vec![value; chunk_elements];
            let chunk_bytes = unsafe {
                std::slice::from_raw_parts(
                    chunk.as_ptr() as *const u8,
                    std::mem::size_of_val(chunk.as_slice()),
                )
            };

            let mut written = 0usize;
            while written < num_elements {
                let remaining = num_elements - written;
                let this_chunk = remaining.min(chunk_elements);
                file.write_all(&chunk_bytes[..this_chunk * element_size])
                    .map_err(|e| {
                        TorshError::IoError(format!("Failed to write to memory-mapped file: {e}"))
                    })?;
                written += this_chunk;
            }
        }

        file.flush()
            .map_err(|e| TorshError::IoError(format!("Failed to flush memory-mapped file: {e}")))?;

        Ok(Self {
            file,
            file_path,
            num_elements,
            cache: HashMap::new(),
            max_cache_size: 10000,
            access_pattern: VecDeque::new(),
            is_temporary,
        })
    }

    /// Path of the file backing this storage.
    pub fn file_path(&self) -> &std::path::Path {
        &self.file_path
    }

    /// Get element at index with caching
    pub fn get(&mut self, index: usize) -> Result<T>
    where
        T: Copy,
    {
        if index >= self.num_elements {
            return Err(TorshError::IndexOutOfBounds {
                index,
                size: self.num_elements,
            });
        }

        // Check cache first
        if let Some(&value) = self.cache.get(&index) {
            self.update_access_pattern(index);
            return Ok(value);
        }

        // Read from file
        let value = self.read_element_from_file(index)?;

        // Add to cache if there's space
        if self.cache.len() < self.max_cache_size {
            self.cache.insert(index, value);
        } else {
            // Evict least recently used element
            self.evict_lru();
            self.cache.insert(index, value);
        }

        self.update_access_pattern(index);
        Ok(value)
    }

    /// Set element at index
    pub fn set(&mut self, index: usize, value: T) -> Result<()>
    where
        T: Copy,
    {
        if index >= self.num_elements {
            return Err(TorshError::IndexOutOfBounds {
                index,
                size: self.num_elements,
            });
        }

        // Update cache
        self.cache.insert(index, value);

        // Write to file
        self.write_element_to_file(index, value)?;
        self.update_access_pattern(index);
        Ok(())
    }

    /// Get slice of elements
    ///
    /// The whole range is fetched with a **single** positional read into the
    /// destination buffer, instead of one syscall (and one heap allocation) per
    /// element via the LRU cache. The cache is bypassed on purpose: every write
    /// goes through to the file, so the file is authoritative.
    pub fn get_slice(&mut self, start: usize, len: usize) -> Result<Vec<T>>
    where
        T: Copy,
    {
        if start + len > self.num_elements {
            return Err(TorshError::IndexOutOfBounds {
                index: start + len - 1,
                size: self.num_elements,
            });
        }

        if len == 0 {
            return Ok(Vec::new());
        }

        let element_size = std::mem::size_of::<T>();
        let mut buf = global_acquire_uninit::<T>(len);

        if element_size == 0 {
            // Zero-sized elements carry no bytes: nothing to read.
            let uninit = buf.as_uninit_slice_mut();
            for slot in uninit.iter_mut().take(len) {
                // SAFETY: a zero-sized type has exactly one value; reading the
                // (empty) representation back is a no-op.
                slot.write(unsafe { std::mem::zeroed() });
            }
            return Ok(buf.into_vec(len));
        }

        {
            let byte_len = len * element_size;
            let ptr = buf.as_uninit_slice_mut().as_mut_ptr() as *mut u8;
            // SAFETY: the buffer is allocated for `len` elements of `T`, so it
            // spans `byte_len` bytes and is correctly aligned for `T`. The bytes
            // are zeroed before a `&mut [u8]` is formed so no uninitialized
            // memory is ever exposed as an initialized reference; the read then
            // overwrites exactly that range, which is what `into_vec(len)`
            // claims as initialized.
            let byte_buf = unsafe {
                std::ptr::write_bytes(ptr, 0, byte_len);
                std::slice::from_raw_parts_mut(ptr, byte_len)
            };
            self.read_bytes_at(byte_buf, (start * element_size) as u64)?;
        }

        Ok(buf.into_vec(len))
    }

    /// Read exactly `buffer.len()` bytes at `offset` from the backing file.
    fn read_bytes_at(&mut self, buffer: &mut [u8], offset: u64) -> Result<()> {
        #[cfg(unix)]
        {
            self.file.read_exact_at(buffer, offset).map_err(|e| {
                TorshError::IoError(format!("Failed to read from memory-mapped file: {e}"))
            })?;
        }

        #[cfg(windows)]
        {
            let mut read_total = 0usize;
            while read_total < buffer.len() {
                let n = self
                    .file
                    .seek_read(&mut buffer[read_total..], offset + read_total as u64)
                    .map_err(|e| {
                        TorshError::IoError(format!("Failed to read from memory-mapped file: {e}"))
                    })?;
                if n == 0 {
                    return Err(TorshError::IoError(
                        "Unexpected end of memory-mapped file".to_string(),
                    ));
                }
                read_total += n;
            }
        }

        #[cfg(not(any(unix, windows)))]
        {
            use std::io::{Read, Seek, SeekFrom};
            self.file.seek(SeekFrom::Start(offset)).map_err(|e| {
                TorshError::IoError(format!("Failed to seek in memory-mapped file: {e}"))
            })?;
            self.file.read_exact(buffer).map_err(|e| {
                TorshError::IoError(format!("Failed to read from memory-mapped file: {e}"))
            })?;
        }

        Ok(())
    }

    /// Set slice of elements
    pub fn set_slice(&mut self, start: usize, values: &[T]) -> Result<()>
    where
        T: Copy,
    {
        if start + values.len() > self.num_elements {
            return Err(TorshError::IndexOutOfBounds {
                index: start + values.len() - 1,
                size: self.num_elements,
            });
        }

        for (i, &value) in values.iter().enumerate() {
            self.set(start + i, value)?;
        }
        Ok(())
    }

    /// Convert entire storage to vector
    pub fn to_vec(&mut self) -> Result<Vec<T>>
    where
        T: Copy,
    {
        self.get_slice(0, self.num_elements)
    }

    /// Read element from file
    fn read_element_from_file(&mut self, index: usize) -> Result<T>
    where
        T: Copy,
    {
        let offset = index * std::mem::size_of::<T>();
        let mut buffer = vec![0u8; std::mem::size_of::<T>()];

        #[cfg(unix)]
        {
            self.file
                .read_exact_at(&mut buffer, offset as u64)
                .map_err(|e| {
                    TorshError::IoError(format!("Failed to read from memory-mapped file: {e}"))
                })?;
        }

        #[cfg(windows)]
        {
            self.file
                .seek_read(&mut buffer, offset as u64)
                .map_err(|e| {
                    TorshError::IoError(format!("Failed to read from memory-mapped file: {e}"))
                })?;
        }

        #[cfg(not(any(unix, windows)))]
        {
            self.file
                .seek(SeekFrom::Start(offset as u64))
                .map_err(|e| {
                    TorshError::IoError(format!("Failed to seek in memory-mapped file: {e}"))
                })?;
            self.file.read_exact(&mut buffer).map_err(|e| {
                TorshError::IoError(format!("Failed to read from memory-mapped file: {e}"))
            })?;
        }

        // Convert bytes to T. The byte buffer is a `Vec<u8>` (alignment 1), so
        // the read must be an unaligned one.
        let value = unsafe { std::ptr::read_unaligned(buffer.as_ptr() as *const T) };
        Ok(value)
    }

    /// Write element to file
    fn write_element_to_file(&mut self, index: usize, value: T) -> Result<()>
    where
        T: Copy,
    {
        let offset = index * std::mem::size_of::<T>();
        let buffer = unsafe {
            std::slice::from_raw_parts(&value as *const T as *const u8, std::mem::size_of::<T>())
        };

        #[cfg(unix)]
        {
            self.file.write_all_at(buffer, offset as u64).map_err(|e| {
                TorshError::IoError(format!("Failed to write to memory-mapped file: {e}"))
            })?;
        }

        #[cfg(windows)]
        {
            self.file.seek_write(buffer, offset as u64).map_err(|e| {
                TorshError::IoError(format!("Failed to write to memory-mapped file: {e}"))
            })?;
        }

        #[cfg(not(any(unix, windows)))]
        {
            self.file
                .seek(SeekFrom::Start(offset as u64))
                .map_err(|e| {
                    TorshError::IoError(format!("Failed to seek in memory-mapped file: {e}"))
                })?;
            self.file.write_all(buffer).map_err(|e| {
                TorshError::IoError(format!("Failed to write to memory-mapped file: {e}"))
            })?;
        }

        Ok(())
    }

    /// Update access pattern for cache management
    fn update_access_pattern(&mut self, index: usize) {
        self.access_pattern.push_back(index);
        if self.access_pattern.len() > self.max_cache_size {
            self.access_pattern.pop_front();
        }
    }

    /// Evict least recently used element from cache
    fn evict_lru(&mut self) {
        if let Some(lru_index) = self.access_pattern.front().copied() {
            self.cache.remove(&lru_index);
        }
    }
}

impl<T: TensorElement> Drop for MemoryMappedStorage<T> {
    fn drop(&mut self) {
        if self.is_temporary {
            // Clean up temporary file
            let _ = std::fs::remove_file(&self.file_path);
        }
    }
}

impl<T: TensorElement> Clone for TensorStorage<T> {
    fn clone(&self) -> Self {
        match self {
            Self::InMemory(data) => Self::InMemory(Arc::clone(data)),
            Self::MemoryMapped(storage) => Self::MemoryMapped(Arc::clone(storage)),
            #[cfg(feature = "simd")]
            Self::Aligned(data) => Self::Aligned(Arc::clone(data)),
            #[cfg(feature = "simd")]
            Self::SimdOptimized(storage) => {
                // Mark the storage as shared for COW semantics
                storage.mark_shared();
                Self::SimdOptimized(Arc::clone(storage))
            }
            #[cfg(feature = "gpu")]
            Self::Device { buffer, host_cache } => Self::Device {
                // Both handles are shared: the device allocation is immutable,
                // and sharing the cache means a clone never re-downloads.
                buffer: Arc::clone(buffer),
                host_cache: Arc::clone(host_cache),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_memory_storage() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let storage = TensorStorage::in_memory(data.clone());

        assert_eq!(storage.len(), 4);
        assert!(!storage.is_empty());
        assert_eq!(storage.storage_type(), "in_memory");

        assert_eq!(storage.get(0).expect("get(0) failed"), 1.0);
        assert_eq!(storage.get(3).expect("get(3) failed"), 4.0);

        let slice = storage.get_slice(1, 2).expect("get_slice failed");
        assert_eq!(slice, vec![2.0, 3.0]);
    }

    #[test]
    fn test_optimal_storage_selection() {
        // Small data should use in-memory storage (200 f32 = 800 bytes < 1024 threshold)
        let small_data = vec![1.0f32; 200];
        let small_storage =
            TensorStorage::create_optimal(small_data).expect("create_optimal failed");

        #[cfg(feature = "simd")]
        {
            // With SIMD enabled, small data below threshold should use in-memory
            assert_eq!(small_storage.storage_type(), "in_memory");
        }
        #[cfg(not(feature = "simd"))]
        {
            // Without SIMD, all data uses in-memory storage
            assert_eq!(small_storage.storage_type(), "in_memory");
        }
    }

    #[test]
    fn test_memory_usage_calculation() {
        let data = vec![1.0f32; 1000];
        let storage = TensorStorage::in_memory(data);
        let expected_size = 1000 * std::mem::size_of::<f32>();
        assert_eq!(storage.memory_usage(), expected_size);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_aligned_storage() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let storage =
            TensorStorage::aligned(data.clone()).expect("aligned storage creation failed");

        assert_eq!(storage.len(), 4);
        assert!(!storage.is_empty());
        assert_eq!(storage.storage_type(), "aligned_simd");

        // Test basic element access
        assert_eq!(storage.get(0).expect("get(0) failed"), 1.0);
        assert_eq!(storage.get(3).expect("get(3) failed"), 4.0);

        // Test slice access
        let slice = storage.get_slice(1, 2).expect("get_slice failed");
        assert_eq!(slice, vec![2.0, 3.0]);

        // Test conversion to vec
        let vec = storage.to_vec().expect("to_vec failed");
        assert_eq!(vec, data);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_optimal_storage_selection_with_aligned() {
        // Medium-size data should use aligned storage when SIMD is enabled
        let medium_data = vec![1.0f32; 2000]; // Above ALIGNED_STORAGE_THRESHOLD
        let medium_storage = TensorStorage::create_optimal(medium_data)
            .expect("create_optimal for medium data failed");
        assert_eq!(medium_storage.storage_type(), "aligned_simd");

        // Small data should still use in-memory storage
        let small_data = vec![1.0f32; 100]; // Below ALIGNED_STORAGE_THRESHOLD
        let small_storage = TensorStorage::create_optimal(small_data)
            .expect("create_optimal for small data failed");
        assert_eq!(small_storage.storage_type(), "in_memory");
    }
}
