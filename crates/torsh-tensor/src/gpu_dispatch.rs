//! GPU compute dispatch for ToRSh, backed by oxicuda's `ComputeBackend` trait.
//!
//! This module is the migration target that replaces the former
//! `scirs2_core::gpu` dependency.  ToRSh tensors keep owning all
//! dtype / shape / autograd semantics; this layer only marshals the `f32`
//! payload of CUDA-device tensors through the flat
//! [`oxicuda_backend::ComputeBackend`] op interface using the
//! host → device → op → host transfer model (the same per-op transfer model
//! the previous `scirs2_core::gpu` path used).
//!
//! ## Backend selection
//!
//! The active backend lives in a process-global handle that is **injectable**
//! through `install_backend` / `clear_backend`.  With `--features cuda` the
//! handle auto-initialises once from ToRSh's own thin `CudaBackend` over the
//! oxicuda leaf crates, and is adopted only when a physical device was found;
//! otherwise `active_backend` returns `None` and every dispatch declines, so
//! callers fall back to ToRSh's native CPU / SIMD implementations.
//!
//! Injection is not merely a test convenience: it is what lets the transfer
//! behaviour of the residency model be *observed* (a backend that counts its
//! `copy_htod` / `copy_dtoh` calls can be installed and interrogated), which is
//! how `tests/hardening_gpu_residency.rs` proves that a chain of device ops
//! performs exactly one upload and one download.
//!
//! The marshalling helpers (`run_unary_f32`, `run_binary_f32`) are
//! exercised against the real [`oxicuda_backend::CpuBackend`] in the unit
//! tests, so the op mapping and buffer handling are verified end-to-end even
//! without a GPU present.

use crate::{Tensor, TensorElement};
#[cfg(test)]
use oxicuda_backend::BackendResult;
use oxicuda_backend::ComputeBackend;
use std::sync::{Arc, Once, RwLock};
use torsh_core::sync::RwLockExt;

// Re-export the op vocabulary so call sites depend on `crate::gpu_dispatch::*`
// rather than naming the `oxicuda_backend` crate directly.
pub use oxicuda_backend::{BinaryOp, ReduceOp, UnaryOp};

/// View an `f32` slice as its raw native-endian bytes (plain-old-data reinterpret).
#[inline]
fn f32_as_bytes(data: &[f32]) -> &[u8] {
    // SAFETY: `f32` is plain-old-data; every byte of its representation is a
    // valid `u8`.  The returned slice borrows `data` for the same lifetime and
    // spans exactly `size_of_val(data)` bytes.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data)) }
}

// ---------------------------------------------------------------------------
// Per-op marshalling reference (test-only)
// ---------------------------------------------------------------------------
// `run_unary_f32` / `run_binary_f32` implement the *old* transfer model — one
// upload and one download per operation — which the residency path below has
// replaced in production. They are kept, with their signatures unchanged, as the
// direct end-to-end exercise of the backend op mapping: the tests call them with
// a real `CpuBackend` (and, under `--features cuda`, a real device), so the
// `UnaryOp`/`BinaryOp` wiring stays verified independently of residency.
// ---------------------------------------------------------------------------

/// Decode native-endian `f32` values from a byte buffer.
#[cfg(test)]
#[inline]
fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Allocate one device buffer per entry in `sizes`, run `f` with the resulting
/// pointers, then free every buffer (even on error).  If an allocation fails
/// partway through, the already-allocated buffers are freed before returning.
#[cfg(test)]
fn with_buffers<R>(
    backend: &dyn ComputeBackend,
    sizes: &[usize],
    f: impl FnOnce(&[u64]) -> BackendResult<R>,
) -> BackendResult<R> {
    let mut ptrs: Vec<u64> = Vec::with_capacity(sizes.len());
    for &size in sizes {
        match backend.alloc(size) {
            Ok(ptr) => ptrs.push(ptr),
            Err(err) => {
                for &ptr in &ptrs {
                    let _ = backend.free(ptr);
                }
                return Err(err);
            }
        }
    }
    let result = f(&ptrs);
    for &ptr in &ptrs {
        let _ = backend.free(ptr);
    }
    result
}

/// Execute a unary `f32` op on `input` through `backend`
/// (alloc → copy-in → op → copy-out → free).
#[cfg(test)]
fn run_unary_f32(
    backend: &dyn ComputeBackend,
    op: UnaryOp,
    input: &[f32],
) -> BackendResult<Vec<f32>> {
    let n = input.len();
    let bytes = std::mem::size_of_val(input);
    with_buffers(backend, &[bytes, bytes], |ptrs| {
        let (input_ptr, output_ptr) = (ptrs[0], ptrs[1]);
        backend.copy_htod(input_ptr, f32_as_bytes(input))?;
        backend.unary(op, input_ptr, output_ptr, n)?;
        let mut out = vec![0u8; bytes];
        backend.copy_dtoh(&mut out, output_ptr)?;
        Ok(bytes_to_f32(&out))
    })
}

/// Execute a binary elementwise `f32` op on `(a, b)` through `backend`.
///
/// `a` and `b` must have equal length.
#[cfg(test)]
fn run_binary_f32(
    backend: &dyn ComputeBackend,
    op: BinaryOp,
    a: &[f32],
    b: &[f32],
) -> BackendResult<Vec<f32>> {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let bytes = std::mem::size_of_val(a);
    with_buffers(backend, &[bytes, bytes, bytes], |ptrs| {
        let (a_ptr, b_ptr, out_ptr) = (ptrs[0], ptrs[1], ptrs[2]);
        backend.copy_htod(a_ptr, f32_as_bytes(a))?;
        backend.copy_htod(b_ptr, f32_as_bytes(b))?;
        backend.binary(op, a_ptr, b_ptr, out_ptr, n)?;
        let mut out = vec![0u8; bytes];
        backend.copy_dtoh(&mut out, out_ptr)?;
        Ok(bytes_to_f32(&out))
    })
}

/// The process-wide compute backend handle.
///
/// An `Arc` (rather than a `&'static`) is what lets a device buffer keep its
/// own backend alive: [`crate::storage::DeviceBuffer`] holds a clone of this
/// handle so that its `Drop` can free the allocation without consulting any
/// registry — and therefore without taking a ToRSh lock or risking a panic.
static BACKEND: RwLock<Option<Arc<dyn ComputeBackend>>> = RwLock::new(None);

/// Guards the one-shot auto-initialisation of the real CUDA backend.
static AUTO_INIT: Once = Once::new();

/// Install `backend` as the process-wide compute backend, returning the handle
/// it replaced (if any).
///
/// Device buffers allocated through the previous backend stay valid: each one
/// owns a handle on the backend that allocated it, and
/// [`try_unary_f32`] / [`try_binary_f32`] compare backend *identity* before
/// reusing a resident buffer, so a tensor from the old backend is uploaded
/// afresh rather than having its pointer handed to the new one.
pub fn install_backend(backend: Arc<dyn ComputeBackend>) -> Option<Arc<dyn ComputeBackend>> {
    BACKEND.write_or_recover().replace(backend)
}

/// Remove the process-wide compute backend, returning it.
///
/// Every subsequent dispatch declines (and callers take their CPU path) until a
/// backend is installed again.
pub fn clear_backend() -> Option<Arc<dyn ComputeBackend>> {
    BACKEND.write_or_recover().take()
}

/// Return the active GPU compute backend, or `None` when no GPU backend is
/// available in this build / environment (in which case callers fall back to
/// their native CPU implementation).
pub(crate) fn active_backend() -> Option<Arc<dyn ComputeBackend>> {
    // Fast path: an explicitly installed backend wins, and the read guard is
    // released at the `return` so the auto-init below can never dead-lock
    // against it.
    if let Some(backend) = BACKEND.read_or_recover().as_ref() {
        return Some(Arc::clone(backend));
    }

    // Real CUDA path: ToRSh's own thin `CudaBackend` over the oxicuda leaf
    // crates, built lazily exactly once. It is adopted only when a physical GPU
    // device was found during init; otherwise the handle stays empty and
    // callers use their native CPU path.
    AUTO_INIT.call_once(|| {
        #[cfg(feature = "cuda")]
        {
            use crate::cuda_backend::CudaBackend;

            // `ComputeBackend::init` takes `&mut self`, so initialisation must
            // happen before the backend is shared behind an `Arc`.
            let mut backend = CudaBackend::new();
            if backend.init().is_ok() && backend.has_gpu_context() {
                *BACKEND.write_or_recover() = Some(Arc::new(backend));
            }
        }
    });

    BACKEND.read_or_recover().as_ref().map(Arc::clone)
}

// ---------------------------------------------------------------------------
// Device residency
// ---------------------------------------------------------------------------
// The helpers below implement the residency model: an operand that already
// lives on the device is read in place, and every result *stays* on the device.
// A chain of ops therefore transfers its data exactly twice — once up, once
// down — instead of twice per operation.
// ---------------------------------------------------------------------------

/// Bytes per `f32` element, the only dtype this dispatch layer marshals.
const F32_BYTES: usize = std::mem::size_of::<f32>();

/// Byte size of `elements` `f32` values, or `None` if that would overflow.
///
/// Every allocation and every residency check goes through this, so an absurd
/// element count declines the dispatch instead of wrapping into a buffer that is
/// far too small.
#[inline]
fn f32_byte_size(elements: usize) -> Option<usize> {
    elements.checked_mul(F32_BYTES)
}

/// The device buffer `tensor` can be read from directly, if any.
///
/// Residency is only claimed when the buffer is genuinely interchangeable with
/// a fresh upload of the tensor: same backend *instance* (pointers are not
/// portable across backends), plain row-major layout at offset zero (the
/// backend's copy primitives have no offset parameter), and an `f32` payload of
/// exactly the expected size.
fn residency_of<T: TensorElement>(
    tensor: &Tensor<T>,
    backend: &Arc<dyn ComputeBackend>,
) -> Option<Arc<crate::storage::DeviceBuffer>> {
    if tensor.is_view() {
        return None;
    }
    let buffer = tensor.storage.device_buffer()?;
    if !Arc::ptr_eq(buffer.backend(), backend) {
        return None;
    }
    if buffer.dtype() != torsh_core::dtype::DType::F32 {
        return None;
    }
    if buffer.bytes() != f32_byte_size(tensor.numel())? {
        return None;
    }
    Some(Arc::clone(buffer))
}

/// Upload `tensor`'s contiguous `f32` payload to a fresh device allocation.
///
/// Returns the pointer, which the caller owns and must free (or adopt into a
/// [`crate::storage::DeviceBuffer`]).
///
/// # Safety contract
/// Callers must have established `T == f32`.
fn upload_f32<T: TensorElement>(backend: &dyn ComputeBackend, tensor: &Tensor<T>) -> Option<u64> {
    let ptr = backend.alloc(f32_byte_size(tensor.numel())?).ok()?;
    let uploaded = tensor.with_contiguous_data(|data| {
        // SAFETY: the caller's `TypeId` guard confirms `T == f32`, so the slice
        // can be reinterpreted without copying.
        let data_f32: &[f32] =
            unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<f32>(), data.len()) };
        backend
            .copy_htod(ptr, f32_as_bytes(data_f32))
            .map_err(|e| torsh_core::error::TorshError::InvalidOperation(format!("{e}")))
    });
    if uploaded.is_err() {
        let _ = backend.free(ptr);
        return None;
    }
    Some(ptr)
}

/// Adopt `ptr` — an allocation of exactly `bytes` bytes — as the storage of a
/// fresh device-resident tensor.
fn device_tensor<T: TensorElement>(
    ptr: u64,
    bytes: usize,
    shape: Vec<usize>,
    device: crate::DeviceType,
    backend: &Arc<dyn ComputeBackend>,
) -> Tensor<T> {
    let buffer = crate::storage::DeviceBuffer::adopt(
        ptr,
        bytes,
        torsh_core::dtype::DType::F32,
        Arc::clone(backend),
    );
    let storage = crate::storage::TensorStorage::device(Arc::new(buffer));
    Tensor::<T>::from_device_storage(storage, shape, device)
}

/// Whether this dispatch layer will marshal `tensor` at all.
fn is_dispatchable<T: TensorElement>(tensor: &Tensor<T>) -> bool {
    std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        && matches!(tensor.device, crate::DeviceType::Cuda(_))
        // `alloc(0)` is an error on every backend, so an empty tensor declines
        // here rather than failing mid-dispatch.
        && tensor.numel() > 0
}

/// Attempt a unary activation on the GPU for `f32` CUDA tensors.
///
/// Returns `Some(result)` only when **all** of the following hold:
/// 1. `T == f32`,
/// 2. the tensor lives on a CUDA device and is non-empty, and
/// 3. a GPU backend is available and the dispatch succeeds end-to-end.
///
/// Otherwise returns `None`, signalling the caller to use its CPU path.
///
/// # Transfers
/// The result **stays on the device**. An input that is already device-resident
/// costs zero transfers; a host input costs exactly one upload. Nothing is ever
/// downloaded here — that happens once, lazily, when the host first reads the
/// data (see [`crate::storage::TensorStorage::Device`]).
pub fn try_unary_f32<T: TensorElement>(input: &Tensor<T>, op: UnaryOp) -> Option<Tensor<T>> {
    if !is_dispatchable(input) {
        return None;
    }
    let backend = active_backend()?;
    let numel = input.numel();
    let output_bytes = f32_byte_size(numel)?;

    // Read a resident buffer in place; otherwise upload a scratch copy that this
    // call owns and must release before returning.
    let resident = residency_of(input, &backend);
    let (input_ptr, scratch) = match &resident {
        Some(buffer) => (buffer.ptr(), None),
        None => {
            let ptr = upload_f32(&*backend, input)?;
            (ptr, Some(ptr))
        }
    };

    let output_ptr = match backend.alloc(output_bytes) {
        Ok(ptr) => ptr,
        Err(_) => {
            free_scratch(&*backend, scratch);
            return None;
        }
    };

    if backend.unary(op, input_ptr, output_ptr, numel).is_err() {
        // Both this call's allocations must go back, or an unsupported op would
        // leak a buffer on every invocation.
        let _ = backend.free(output_ptr);
        free_scratch(&*backend, scratch);
        return None;
    }
    free_scratch(&*backend, scratch);

    Some(device_tensor(
        output_ptr,
        output_bytes,
        input.shape().dims().to_vec(),
        input.device,
        &backend,
    ))
}

/// Release a scratch upload buffer, if this call made one.
fn free_scratch(backend: &dyn ComputeBackend, scratch: Option<u64>) {
    if let Some(ptr) = scratch {
        let _ = backend.free(ptr);
    }
}

/// Attempt a binary elementwise op on the GPU for equal-shaped `f32` CUDA tensors.
///
/// Same gating as [`try_unary_f32`]; returns `None` (CPU fallback) unless both
/// tensors are `f32`, on the same CUDA device, equal length, and a GPU backend
/// dispatches successfully.
///
/// # Transfers
/// Mixed residency uploads the **host** operand and leaves the resident one in
/// place — a resident buffer is never downloaded to feed an op. Two resident
/// operands cost zero transfers.
pub fn try_binary_f32<T: TensorElement>(
    lhs: &Tensor<T>,
    rhs: &Tensor<T>,
    op: BinaryOp,
) -> Option<Tensor<T>> {
    if !is_dispatchable(lhs) || !is_dispatchable(rhs) {
        return None;
    }
    if lhs.device != rhs.device || lhs.numel() != rhs.numel() {
        return None;
    }
    let backend = active_backend()?;
    let numel = lhs.numel();
    let output_bytes = f32_byte_size(numel)?;

    let lhs_resident = residency_of(lhs, &backend);
    let (lhs_ptr, lhs_scratch) = match &lhs_resident {
        Some(buffer) => (buffer.ptr(), None),
        None => {
            let ptr = upload_f32(&*backend, lhs)?;
            (ptr, Some(ptr))
        }
    };

    let rhs_resident = residency_of(rhs, &backend);
    let (rhs_ptr, rhs_scratch) = match &rhs_resident {
        Some(buffer) => (buffer.ptr(), None),
        None => match upload_f32(&*backend, rhs) {
            Some(ptr) => (ptr, Some(ptr)),
            None => {
                free_scratch(&*backend, lhs_scratch);
                return None;
            }
        },
    };

    let output_ptr = match backend.alloc(output_bytes) {
        Ok(ptr) => ptr,
        Err(_) => {
            free_scratch(&*backend, lhs_scratch);
            free_scratch(&*backend, rhs_scratch);
            return None;
        }
    };

    if backend
        .binary(op, lhs_ptr, rhs_ptr, output_ptr, numel)
        .is_err()
    {
        let _ = backend.free(output_ptr);
        free_scratch(&*backend, lhs_scratch);
        free_scratch(&*backend, rhs_scratch);
        return None;
    }
    free_scratch(&*backend, lhs_scratch);
    free_scratch(&*backend, rhs_scratch);

    Some(device_tensor(
        output_ptr,
        output_bytes,
        lhs.shape().dims().to_vec(),
        lhs.device,
        &backend,
    ))
}

/// Upload `tensor` into device-resident storage tagged for `device`.
///
/// This is the eager half of `to_device(DeviceType::Cuda(_))`: it makes the
/// device tag mean something, so the ops that follow find their operand already
/// resident. Returns `None` — leaving the data on the host — unless `T == f32`,
/// the tensor is non-empty and a backend is active.
pub(crate) fn try_upload_f32<T: TensorElement>(
    tensor: &Tensor<T>,
    device: crate::DeviceType,
) -> Option<Tensor<T>> {
    if std::any::TypeId::of::<T>() != std::any::TypeId::of::<f32>() {
        return None;
    }
    let numel = tensor.numel();
    if numel == 0 {
        return None;
    }
    let bytes = f32_byte_size(numel)?;
    let backend = active_backend()?;
    let ptr = upload_f32(&*backend, tensor)?;
    Some(device_tensor(
        ptr,
        bytes,
        tensor.shape().dims().to_vec(),
        device,
        &backend,
    ))
}

/// Attempt a single-axis reduction on the GPU for `f32` CUDA tensors.
///
/// `output_shape` is the caller's own reduced shape, so `keepdim` stays where it
/// is already computed: the reduction always produces `outer * inner` elements,
/// and only the shape metadata differs between the two conventions.
///
/// # Transfers
/// Same contract as [`try_unary_f32`]: the result stays resident, and a resident
/// input costs zero transfers.
pub fn try_reduce_axis_f32<T: TensorElement>(
    input: &Tensor<T>,
    op: ReduceOp,
    axis: usize,
    output_shape: &[usize],
) -> Option<Tensor<T>> {
    if !is_dispatchable(input) {
        return None;
    }
    let shape_binding = input.shape();
    let dims = shape_binding.dims();
    if axis >= dims.len() {
        return None;
    }
    let outer: usize = dims[..axis].iter().product();
    let inner: usize = dims[axis + 1..].iter().product();
    let output_elements = outer * inner;
    if output_elements == 0 || output_elements != output_shape.iter().product::<usize>() {
        return None;
    }
    let output_bytes = f32_byte_size(output_elements)?;

    let backend = active_backend()?;
    let resident = residency_of(input, &backend);
    let (input_ptr, scratch) = match &resident {
        Some(buffer) => (buffer.ptr(), None),
        None => {
            let ptr = upload_f32(&*backend, input)?;
            (ptr, Some(ptr))
        }
    };

    let output_ptr = match backend.alloc(output_bytes) {
        Ok(ptr) => ptr,
        Err(_) => {
            free_scratch(&*backend, scratch);
            return None;
        }
    };

    if backend
        .reduce(op, input_ptr, output_ptr, dims, axis)
        .is_err()
    {
        let _ = backend.free(output_ptr);
        free_scratch(&*backend, scratch);
        return None;
    }
    free_scratch(&*backend, scratch);

    Some(device_tensor(
        output_ptr,
        output_bytes,
        output_shape.to_vec(),
        input.device,
        &backend,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_backend::CpuBackend;

    #[test]
    fn unary_relu_through_compute_backend() {
        let mut backend = CpuBackend::new();
        backend.init().expect("backend init");
        let out = run_unary_f32(&backend, UnaryOp::Relu, &[-2.0, -0.5, 0.0, 1.5, 3.0])
            .expect("relu dispatch");
        assert_eq!(out, vec![0.0, 0.0, 0.0, 1.5, 3.0]);
        // Every scratch buffer must have been released.
        assert_eq!(backend.live_allocations(), 0);
    }

    #[test]
    fn unary_sigmoid_through_compute_backend() {
        let mut backend = CpuBackend::new();
        backend.init().expect("backend init");
        let out = run_unary_f32(&backend, UnaryOp::Sigmoid, &[0.0]).expect("sigmoid dispatch");
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert_eq!(backend.live_allocations(), 0);
    }

    #[test]
    fn binary_add_and_mul_through_compute_backend() {
        let mut backend = CpuBackend::new();
        backend.init().expect("backend init");
        let a = [1.0f32, 2.0, 3.0];
        let b = [10.0f32, 20.0, 30.0];
        assert_eq!(
            run_binary_f32(&backend, BinaryOp::Add, &a, &b).expect("add dispatch"),
            vec![11.0, 22.0, 33.0]
        );
        assert_eq!(
            run_binary_f32(&backend, BinaryOp::Mul, &a, &b).expect("mul dispatch"),
            vec![10.0, 40.0, 90.0]
        );
        assert_eq!(backend.live_allocations(), 0);
    }

    #[test]
    fn cpu_tensor_declines_gpu_dispatch() {
        // An f32 CPU tensor must never take the GPU path (device guard).
        let tensor = Tensor::from_data(vec![1.0f32, -1.0], vec![2], crate::DeviceType::Cpu)
            .expect("tensor creation");
        assert!(try_unary_f32(&tensor, UnaryOp::Relu).is_none());
    }

    /// The injectable handle must round-trip: installing returns the previous
    /// backend, clearing returns the current one, and a cleared handle declines.
    ///
    /// Runs only without `--features cuda`, where nothing auto-initialises the
    /// handle and the "clean state is empty" half of the assertion is meaningful.
    #[cfg(not(feature = "cuda"))]
    #[test]
    fn backend_handle_installs_and_clears() {
        // Clean default state: nothing is wired, so every dispatch declines.
        assert!(clear_backend().is_none());
        assert!(active_backend().is_none());

        let mut first = CpuBackend::new();
        first.init().expect("backend init");
        let first: Arc<dyn ComputeBackend> = Arc::new(first);
        assert!(install_backend(Arc::clone(&first)).is_none());

        let active = active_backend().expect("installed backend must be active");
        assert!(Arc::ptr_eq(&active, &first));

        let mut second = CpuBackend::new();
        second.init().expect("backend init");
        let second: Arc<dyn ComputeBackend> = Arc::new(second);
        let previous = install_backend(second).expect("install must return the previous handle");
        assert!(Arc::ptr_eq(&previous, &first));

        assert!(clear_backend().is_some());
        assert!(active_backend().is_none());
    }

    // ── Real-GPU end-to-end tests (require an actual CUDA device) ────────────
    // These run only with `--features cuda`. When no device is present they
    // skip (the dispatch declines), so they are safe on CPU-only CI too.

    #[cfg(feature = "cuda")]
    #[test]
    fn unary_relu_runs_on_real_gpu() {
        let Some(backend) = active_backend() else {
            eprintln!("no CUDA device available; skipping real-GPU relu test");
            return;
        };
        let input = vec![-2.0f32, -0.5, 0.0, 1.5, 3.0, -7.0, 4.0, 0.25];
        // Call the helper directly so a backend error is surfaced (not swallowed).
        let got =
            run_unary_f32(&*backend, UnaryOp::Relu, &input).expect("GPU relu dispatch failed");
        let expect: Vec<f32> = input.iter().map(|&x| x.max(0.0)).collect();
        assert_eq!(got, expect, "GPU relu result must match CPU reference");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn tensor_path_unary_dispatches_to_gpu() {
        if active_backend().is_none() {
            return;
        }
        // Realistic path: f32 Cuda tensor -> try_unary_f32 -> GPU. `expect`
        // (not silent CPU fallback) proves the GPU dispatch actually succeeded.
        let t = Tensor::from_data(
            vec![-1.0f32, 2.0, -3.0, 4.0],
            vec![4],
            crate::DeviceType::Cuda(0),
        )
        .expect("cuda tensor");
        let out = try_unary_f32(&t, UnaryOp::Relu).expect("Tensor GPU path returned None");
        assert_eq!(out.to_vec().expect("vec"), vec![0.0, 2.0, 0.0, 4.0]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn binary_add_runs_on_real_gpu() {
        let Some(backend) = active_backend() else {
            return;
        };
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [10.0f32, 20.0, 30.0, 40.0];
        let got =
            run_binary_f32(&*backend, BinaryOp::Add, &a, &b).expect("GPU add dispatch failed");
        assert_eq!(got, vec![11.0, 22.0, 33.0, 44.0]);
    }
}
