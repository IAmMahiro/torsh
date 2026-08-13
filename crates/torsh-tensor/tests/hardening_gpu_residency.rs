//! Production-hardening regression tests for **device-resident tensor storage**.
//!
//! The property under test is not numeric but *transfer count*: a chain of
//! elementwise ops on a CUDA tensor must upload its operand once and download
//! the result once, instead of paying a host→device→host round trip per
//! operation. That is only observable with an instrumented backend, which is
//! what [`gpu_dispatch::install_backend`] exists for.
//!
//! Every test here mutates the process-wide backend handle, so they all take a
//! file-local mutex: `cargo nextest` gives each test its own process, but plain
//! `cargo test` runs them as threads in one.

#![cfg(feature = "gpu")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use oxicuda_backend::{
    BackendError, BackendResult, BackendTranspose, BinaryOp, ComputeBackend, CpuBackend, ReduceOp,
    UnaryOp,
};
use torsh_core::device::DeviceType;
use torsh_tensor::gpu_dispatch;
use torsh_tensor::Tensor;

/// Serialises the process-wide backend handle across tests.
static GUARD: Mutex<()> = Mutex::new(());

/// Take the backend guard, recovering from a previous test's panic rather than
/// cascading a poison error into every later test.
fn backend_guard() -> MutexGuard<'static, ()> {
    GUARD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

// ---------------------------------------------------------------------------
// Instrumentation backends
// ---------------------------------------------------------------------------

/// A [`CpuBackend`] that counts the transfers and allocations made through it.
///
/// `CpuBackend` never reuses a synthetic pointer, so a double free surfaces as
/// an `InvalidArgument` error rather than silent corruption, and
/// `live_allocations()` is an exact leak oracle.
#[derive(Debug)]
struct CountingBackend {
    inner: CpuBackend,
    h2d: AtomicUsize,
    d2h: AtomicUsize,
    allocs: AtomicUsize,
    frees: AtomicUsize,
}

impl CountingBackend {
    fn new() -> Self {
        Self {
            inner: CpuBackend::new(),
            h2d: AtomicUsize::new(0),
            d2h: AtomicUsize::new(0),
            allocs: AtomicUsize::new(0),
            frees: AtomicUsize::new(0),
        }
    }

    /// Host→device transfers performed so far.
    fn h2d(&self) -> usize {
        self.h2d.load(Ordering::SeqCst)
    }

    /// Device→host transfers performed so far.
    fn d2h(&self) -> usize {
        self.d2h.load(Ordering::SeqCst)
    }

    fn allocs(&self) -> usize {
        self.allocs.load(Ordering::SeqCst)
    }

    fn frees(&self) -> usize {
        self.frees.load(Ordering::SeqCst)
    }

    /// Allocations that are still live — the leak oracle.
    fn live(&self) -> usize {
        self.inner.live_allocations()
    }
}

impl ComputeBackend for CountingBackend {
    fn name(&self) -> &str {
        "counting"
    }

    fn init(&mut self) -> BackendResult<()> {
        self.inner.init()
    }

    fn is_initialized(&self) -> bool {
        self.inner.is_initialized()
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &self,
        trans_a: BackendTranspose,
        trans_b: BackendTranspose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a_ptr: u64,
        lda: usize,
        b_ptr: u64,
        ldb: usize,
        beta: f64,
        c_ptr: u64,
        ldc: usize,
    ) -> BackendResult<()> {
        self.inner.gemm(
            trans_a, trans_b, m, n, k, alpha, a_ptr, lda, b_ptr, ldb, beta, c_ptr, ldc,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn conv2d_forward(
        &self,
        input_ptr: u64,
        input_shape: &[usize],
        filter_ptr: u64,
        filter_shape: &[usize],
        output_ptr: u64,
        output_shape: &[usize],
        stride: &[usize],
        padding: &[usize],
    ) -> BackendResult<()> {
        self.inner.conv2d_forward(
            input_ptr,
            input_shape,
            filter_ptr,
            filter_shape,
            output_ptr,
            output_shape,
            stride,
            padding,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        q_ptr: u64,
        k_ptr: u64,
        v_ptr: u64,
        o_ptr: u64,
        batch: usize,
        heads: usize,
        seq_q: usize,
        seq_kv: usize,
        head_dim: usize,
        scale: f64,
        causal: bool,
    ) -> BackendResult<()> {
        self.inner.attention(
            q_ptr, k_ptr, v_ptr, o_ptr, batch, heads, seq_q, seq_kv, head_dim, scale, causal,
        )
    }

    fn reduce(
        &self,
        op: ReduceOp,
        input_ptr: u64,
        output_ptr: u64,
        shape: &[usize],
        axis: usize,
    ) -> BackendResult<()> {
        self.inner.reduce(op, input_ptr, output_ptr, shape, axis)
    }

    fn unary(&self, op: UnaryOp, input_ptr: u64, output_ptr: u64, n: usize) -> BackendResult<()> {
        self.inner.unary(op, input_ptr, output_ptr, n)
    }

    fn binary(
        &self,
        op: BinaryOp,
        a_ptr: u64,
        b_ptr: u64,
        output_ptr: u64,
        n: usize,
    ) -> BackendResult<()> {
        self.inner.binary(op, a_ptr, b_ptr, output_ptr, n)
    }

    fn synchronize(&self) -> BackendResult<()> {
        self.inner.synchronize()
    }

    fn alloc(&self, bytes: usize) -> BackendResult<u64> {
        self.allocs.fetch_add(1, Ordering::SeqCst);
        self.inner.alloc(bytes)
    }

    fn free(&self, ptr: u64) -> BackendResult<()> {
        self.frees.fetch_add(1, Ordering::SeqCst);
        self.inner.free(ptr)
    }

    fn copy_htod(&self, dst: u64, src: &[u8]) -> BackendResult<()> {
        self.h2d.fetch_add(1, Ordering::SeqCst);
        self.inner.copy_htod(dst, src)
    }

    fn copy_dtoh(&self, dst: &mut [u8], src: u64) -> BackendResult<()> {
        self.d2h.fetch_add(1, Ordering::SeqCst);
        self.inner.copy_dtoh(dst, src)
    }
}

/// A [`CpuBackend`] whose `unary` always refuses.
///
/// Everything else — including `alloc` — is forwarded, which is exactly what
/// makes it the right vehicle for the leak path: the dispatch allocates its
/// output buffer, *then* the op fails, so the buffer must be released before the
/// caller falls back to the CPU. (`NullBackend` refuses `alloc` too, so the
/// allocation would never happen and the leak would go untested.)
#[derive(Debug)]
struct FailingUnaryBackend {
    inner: CpuBackend,
}

impl FailingUnaryBackend {
    fn new() -> Self {
        Self {
            inner: CpuBackend::new(),
        }
    }

    fn live(&self) -> usize {
        self.inner.live_allocations()
    }
}

impl ComputeBackend for FailingUnaryBackend {
    fn name(&self) -> &str {
        "failing-unary"
    }

    fn init(&mut self) -> BackendResult<()> {
        self.inner.init()
    }

    fn is_initialized(&self) -> bool {
        self.inner.is_initialized()
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &self,
        trans_a: BackendTranspose,
        trans_b: BackendTranspose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a_ptr: u64,
        lda: usize,
        b_ptr: u64,
        ldb: usize,
        beta: f64,
        c_ptr: u64,
        ldc: usize,
    ) -> BackendResult<()> {
        self.inner.gemm(
            trans_a, trans_b, m, n, k, alpha, a_ptr, lda, b_ptr, ldb, beta, c_ptr, ldc,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn conv2d_forward(
        &self,
        input_ptr: u64,
        input_shape: &[usize],
        filter_ptr: u64,
        filter_shape: &[usize],
        output_ptr: u64,
        output_shape: &[usize],
        stride: &[usize],
        padding: &[usize],
    ) -> BackendResult<()> {
        self.inner.conv2d_forward(
            input_ptr,
            input_shape,
            filter_ptr,
            filter_shape,
            output_ptr,
            output_shape,
            stride,
            padding,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        q_ptr: u64,
        k_ptr: u64,
        v_ptr: u64,
        o_ptr: u64,
        batch: usize,
        heads: usize,
        seq_q: usize,
        seq_kv: usize,
        head_dim: usize,
        scale: f64,
        causal: bool,
    ) -> BackendResult<()> {
        self.inner.attention(
            q_ptr, k_ptr, v_ptr, o_ptr, batch, heads, seq_q, seq_kv, head_dim, scale, causal,
        )
    }

    fn reduce(
        &self,
        op: ReduceOp,
        input_ptr: u64,
        output_ptr: u64,
        shape: &[usize],
        axis: usize,
    ) -> BackendResult<()> {
        self.inner.reduce(op, input_ptr, output_ptr, shape, axis)
    }

    fn unary(&self, _op: UnaryOp, _input: u64, _output: u64, _n: usize) -> BackendResult<()> {
        Err(BackendError::Unsupported("test".into()))
    }

    fn binary(
        &self,
        op: BinaryOp,
        a_ptr: u64,
        b_ptr: u64,
        output_ptr: u64,
        n: usize,
    ) -> BackendResult<()> {
        self.inner.binary(op, a_ptr, b_ptr, output_ptr, n)
    }

    fn synchronize(&self) -> BackendResult<()> {
        self.inner.synchronize()
    }

    fn alloc(&self, bytes: usize) -> BackendResult<u64> {
        self.inner.alloc(bytes)
    }

    fn free(&self, ptr: u64) -> BackendResult<()> {
        self.inner.free(ptr)
    }

    fn copy_htod(&self, dst: u64, src: &[u8]) -> BackendResult<()> {
        self.inner.copy_htod(dst, src)
    }

    fn copy_dtoh(&self, dst: &mut [u8], src: u64) -> BackendResult<()> {
        self.inner.copy_dtoh(dst, src)
    }
}

/// Build an initialised [`CountingBackend`] and install it process-wide.
///
/// The concrete `Arc<CountingBackend>` is returned (rather than the erased
/// handle) so the test can read the counters afterwards.
fn install_counting() -> Arc<CountingBackend> {
    let mut backend = CountingBackend::new();
    backend.init().expect("counting backend init");
    let backend = Arc::new(backend);
    gpu_dispatch::install_backend(Arc::clone(&backend) as Arc<dyn ComputeBackend>);
    backend
}

/// Build an `f32` tensor tagged as living on CUDA device 0.
fn cuda_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
    Tensor::from_data(data, shape, DeviceType::Cuda(0)).expect("cuda tensor creation")
}

/// Build an `f32` host tensor.
fn cpu_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
    Tensor::from_data(data, shape, DeviceType::Cpu).expect("cpu tensor creation")
}

/// A deterministic spread of values that exercises both sides of `relu`.
fn sample_values(count: usize) -> Vec<f32> {
    (0..count)
        .map(|index| (index as f32) * 0.05 - (count as f32) * 0.025)
        .collect()
}

/// Assert two slices agree elementwise.
fn assert_all_close(got: &[f32], expected: &[f32], tolerance: f32, what: &str) {
    assert_eq!(got.len(), expected.len(), "{what}: length mismatch");
    for (index, (left, right)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (left - right).abs() <= tolerance,
            "{what}: element {index} was {left}, expected {right}"
        );
    }
}

// ---------------------------------------------------------------------------
// T1 — the headline residency property
// ---------------------------------------------------------------------------

/// A chain of unary ops on a CUDA tensor must transfer its data exactly twice:
/// one upload of the operand and one download of the final result.
///
/// The per-op `alloc → htod → op → dtoh → free` marshalling this replaces cost
/// one upload *and* one download per link in the chain.
#[test]
fn chained_unary_ops_do_one_upload_and_one_download() {
    let _guard = backend_guard();
    let counting = install_counting();

    let input = cuda_tensor(vec![0.5f32; 1024], vec![1024]);
    let output = input
        .relu()
        .expect("relu")
        .sigmoid()
        .expect("sigmoid")
        .tanh()
        .expect("tanh");
    let values = output.to_vec().expect("to_vec");

    let (h2d, d2h) = (counting.h2d(), counting.d2h());
    drop(output);
    gpu_dispatch::clear_backend();

    assert_eq!(values.len(), 1024);
    assert_eq!(
        h2d, 1,
        "a resident chain must upload the operand exactly once"
    );
    assert_eq!(
        d2h, 1,
        "a resident chain must download the result exactly once"
    );
}

// ---------------------------------------------------------------------------
// T2 — residency must not change the numbers
// ---------------------------------------------------------------------------

/// The device-resident chain must agree elementwise with the host chain.
///
/// 512 elements keeps the host side on its exact `parallel_map` path rather than
/// the vectorised `f32` approximations, so any difference here is the device
/// dispatch's fault and not a kernel-vs-kernel one.
#[test]
fn device_result_matches_cpu_within_1e6() {
    let _guard = backend_guard();
    let counting = install_counting();

    let values = sample_values(512);
    let device = cuda_tensor(values.clone(), vec![512])
        .relu()
        .expect("relu")
        .sigmoid()
        .expect("sigmoid")
        .tanh()
        .expect("tanh")
        .to_vec()
        .expect("to_vec");
    let host = cpu_tensor(values, vec![512])
        .relu()
        .expect("relu")
        .sigmoid()
        .expect("sigmoid")
        .tanh()
        .expect("tanh")
        .to_vec()
        .expect("to_vec");

    let dispatched = counting.h2d() > 0;
    gpu_dispatch::clear_backend();

    assert!(dispatched, "the device path must actually have run");
    assert_all_close(&device, &host, 1e-6, "device chain vs host chain");
}

// ---------------------------------------------------------------------------
// T3 — mixed residency uploads only the host operand
// ---------------------------------------------------------------------------

/// A binary op between a resident tensor and a host one uploads the host operand
/// and leaves the resident one alone: a resident buffer is never downloaded just
/// to feed an operation.
#[test]
fn binary_mixed_residency_uploads_only_the_host_operand() {
    let _guard = backend_guard();
    let counting = install_counting();

    let lhs = cuda_tensor(sample_values(256), vec![256])
        .relu()
        .expect("relu");
    assert_eq!(
        lhs.storage_type(),
        "device",
        "the relu result must be device-resident"
    );
    let rhs = cuda_tensor(vec![2.0f32; 256], vec![256]);

    let (h2d_before, d2h_before) = (counting.h2d(), counting.d2h());
    let sum = gpu_dispatch::try_binary_f32(&lhs, &rhs, BinaryOp::Add)
        .expect("binary dispatch must take the device path");
    let (h2d_after, d2h_after) = (counting.h2d(), counting.d2h());

    let got = sum.to_vec().expect("to_vec");
    let expected: Vec<f32> = lhs
        .to_vec()
        .expect("lhs to_vec")
        .iter()
        .map(|value| value + 2.0)
        .collect();
    gpu_dispatch::clear_backend();

    assert_eq!(
        h2d_after - h2d_before,
        1,
        "only the host operand may be uploaded"
    );
    assert_eq!(
        d2h_after - d2h_before,
        0,
        "a resident operand must never be downloaded to feed an op"
    );
    assert_all_close(&got, &expected, 1e-6, "mixed-residency add");
}

// ---------------------------------------------------------------------------
// T4 — the device allocation is released with the tensor
// ---------------------------------------------------------------------------

/// Every buffer a chain allocates must be freed once the tensors are dropped:
/// scratch uploads immediately, results through `DeviceBuffer::drop`.
#[test]
fn dropping_device_tensor_frees_buffer() {
    let _guard = backend_guard();
    let counting = install_counting();

    let output = cuda_tensor(sample_values(64), vec![64])
        .relu()
        .expect("relu")
        .sigmoid()
        .expect("sigmoid")
        .tanh()
        .expect("tanh");
    assert!(counting.allocs() > 0, "the device path must have allocated");
    assert!(counting.live() > 0, "the result must still hold its buffer");

    drop(output);

    let (allocs, frees, live) = (counting.allocs(), counting.frees(), counting.live());
    gpu_dispatch::clear_backend();

    assert_eq!(live, 0, "dropping the tensors must release device memory");
    assert_eq!(frees, allocs, "every allocation must be freed exactly once");
}

// ---------------------------------------------------------------------------
// T5 — a failing op falls back to the CPU without leaking
// ---------------------------------------------------------------------------

/// When the backend refuses the op *after* the output buffer was allocated, the
/// dispatch must free everything it allocated before declining.
#[test]
fn unsupported_unary_falls_back_to_cpu_without_leaking() {
    let _guard = backend_guard();

    let mut backend = FailingUnaryBackend::new();
    backend.init().expect("failing backend init");
    let backend = Arc::new(backend);
    gpu_dispatch::install_backend(Arc::clone(&backend) as Arc<dyn ComputeBackend>);

    let values = sample_values(128);
    let tensor = cuda_tensor(values.clone(), vec![128]);
    assert!(
        gpu_dispatch::try_unary_f32(&tensor, UnaryOp::Relu).is_none(),
        "a refused op must decline rather than return a bogus tensor"
    );

    let got = tensor.relu().expect("relu must fall back to the CPU");
    let live = backend.live();
    gpu_dispatch::clear_backend();

    assert_eq!(
        live, 0,
        "a failed dispatch must not leak the buffers it allocated"
    );
    assert_ne!(
        got.storage_type(),
        "device",
        "the fallback result lives on the host"
    );
    let expected: Vec<f32> = values.iter().map(|value| value.max(0.0)).collect();
    assert_all_close(&got.to_vec().expect("to_vec"), &expected, 1e-6, "cpu relu");
}

// ---------------------------------------------------------------------------
// T6 / T7 — the host copy is downloaded at most once
// ---------------------------------------------------------------------------

/// Reading a device tensor twice must download it once: the host copy is cached.
#[test]
fn to_vec_twice_downloads_once() {
    let _guard = backend_guard();
    let counting = install_counting();

    let tensor = cuda_tensor(sample_values(256), vec![256])
        .relu()
        .expect("relu");
    let before = counting.d2h();
    let first = tensor.to_vec().expect("first to_vec");
    let second = tensor.to_vec().expect("second to_vec");
    let after = counting.d2h();
    gpu_dispatch::clear_backend();

    assert_eq!(after - before, 1, "the host copy must be cached");
    assert_eq!(first, second);
}

/// Views of a device tensor materialise through the same cache, so a strided
/// read is one download rather than one per element.
#[test]
fn view_of_device_tensor_materializes_once() {
    let _guard = backend_guard();
    let counting = install_counting();

    let values = sample_values(24);
    let tensor = cuda_tensor(values.clone(), vec![4, 6])
        .relu()
        .expect("relu");

    let before = counting.d2h();
    let reshaped = tensor.reshape(&[6, 4]).expect("reshape");
    let reshaped_values = reshaped.to_vec().expect("reshape to_vec");
    let permuted = tensor.permute(&[1, 0]).expect("permute");
    let permuted_values = permuted.to_vec().expect("permute to_vec");
    let after = counting.d2h();
    gpu_dispatch::clear_backend();

    assert_eq!(
        after - before,
        1,
        "every view must read through the one cached download"
    );

    let host = cpu_tensor(values, vec![4, 6]).relu().expect("host relu");
    assert_all_close(
        &reshaped_values,
        &host
            .reshape(&[6, 4])
            .expect("host reshape")
            .to_vec()
            .expect("vec"),
        1e-6,
        "reshaped device view",
    );
    assert_all_close(
        &permuted_values,
        &host
            .permute(&[1, 0])
            .expect("host permute")
            .to_vec()
            .expect("vec"),
        1e-6,
        "permuted device view",
    );
}

// ---------------------------------------------------------------------------
// T8 — mutation demotes to host storage
// ---------------------------------------------------------------------------

/// A device buffer is immutable, so making a tensor unique must move it back to
/// host storage — after which in-place writes work normally.
#[test]
fn make_unique_demotes_device_to_host() {
    let _guard = backend_guard();
    let counting = install_counting();

    let mut tensor = cuda_tensor(sample_values(32), vec![32])
        .relu()
        .expect("relu");
    assert_eq!(tensor.storage_type(), "device");

    tensor.make_unique().expect("make_unique");
    assert_ne!(
        tensor.storage_type(),
        "device",
        "make_unique must demote device storage"
    );

    tensor.fill_(1.5).expect("fill_ after demotion");
    let filled = tensor.to_vec().expect("to_vec");

    // `fill_` must demote by itself too, straight from device storage.
    let mut resident = cuda_tensor(sample_values(32), vec![32])
        .relu()
        .expect("relu");
    resident.fill_(2.5).expect("fill_ on device storage");
    let refilled = resident.to_vec().expect("to_vec");

    let live = counting.live();
    gpu_dispatch::clear_backend();

    assert_eq!(filled, vec![1.5f32; 32]);
    assert_eq!(refilled, vec![2.5f32; 32]);
    assert_eq!(
        live, 0,
        "demoting to host storage must release the device buffer"
    );
}

// ---------------------------------------------------------------------------
// T9 / T10 — the dispatch declines cleanly
// ---------------------------------------------------------------------------

/// With no backend installed every dispatch declines and the CPU path answers.
#[test]
fn no_backend_installed_declines() {
    let _guard = backend_guard();
    gpu_dispatch::clear_backend();

    let values = sample_values(64);
    let tensor = cuda_tensor(values.clone(), vec![64]);
    assert!(gpu_dispatch::try_unary_f32(&tensor, UnaryOp::Relu).is_none());

    let got = tensor.relu().expect("relu");
    let expected: Vec<f32> = values.iter().map(|value| value.max(0.0)).collect();
    assert_all_close(&got.to_vec().expect("to_vec"), &expected, 1e-6, "cpu relu");
}

/// An empty tensor must decline before allocating: `alloc(0)` is an error on
/// every backend.
#[test]
fn empty_tensor_declines_device_path() {
    let _guard = backend_guard();
    let counting = install_counting();

    let empty = cuda_tensor(Vec::new(), vec![0]);
    assert!(gpu_dispatch::try_unary_f32(&empty, UnaryOp::Relu).is_none());
    let relu = empty.relu().expect("relu on an empty tensor");

    let allocs = counting.allocs();
    gpu_dispatch::clear_backend();

    assert_eq!(allocs, 0, "an empty tensor must not reach the allocator");
    assert!(relu.to_vec().expect("to_vec").is_empty());
}

// ---------------------------------------------------------------------------
// T11 / T12 — reductions and explicit transfers
// ---------------------------------------------------------------------------

/// A single-axis reduction keeps its result on the device: one upload for the
/// operand, one download for the answer.
#[test]
fn axis_reduce_stays_resident() {
    let _guard = backend_guard();
    let counting = install_counting();

    let values = sample_values(24);
    let tensor = cuda_tensor(values.clone(), vec![2, 3, 4]);
    let reduced = tensor.sum_dim(&[1], false).expect("sum_dim");
    assert_eq!(
        reduced.storage_type(),
        "device",
        "the reduction result must stay resident"
    );
    let got = reduced.to_vec().expect("to_vec");

    let (h2d, d2h) = (counting.h2d(), counting.d2h());
    gpu_dispatch::clear_backend();

    assert_eq!(h2d, 1, "the operand must be uploaded exactly once");
    assert_eq!(d2h, 1, "the result must be downloaded exactly once");
    assert_eq!(reduced.shape().dims(), &[2, 4]);

    let expected = cpu_tensor(values, vec![2, 3, 4])
        .sum_dim(&[1], false)
        .expect("host sum_dim")
        .to_vec()
        .expect("vec");
    assert_all_close(&got, &expected, 1e-6, "device axis sum");
}

/// Moving a device tensor back to the host downloads it exactly once and leaves
/// host storage behind.
#[test]
fn to_device_cpu_downloads_exactly_once() {
    let _guard = backend_guard();
    let counting = install_counting();

    let values = sample_values(128);
    let resident = cuda_tensor(values.clone(), vec![128]).relu().expect("relu");

    let before = counting.d2h();
    let host = resident.to_device(DeviceType::Cpu).expect("to_device(Cpu)");
    let after = counting.d2h();
    let got = host.to_vec().expect("to_vec");
    gpu_dispatch::clear_backend();

    assert_eq!(after - before, 1, "the move must download exactly once");
    assert_ne!(host.storage_type(), "device");
    assert_eq!(host.device(), DeviceType::Cpu);
    let expected: Vec<f32> = values.iter().map(|value| value.max(0.0)).collect();
    assert_all_close(&got, &expected, 1e-6, "downloaded values");
}

/// `to_device(Cuda(_))` must genuinely upload, so that the tensor's device tag
/// means something to the ops that follow.
#[test]
fn to_device_cuda_uploads_eagerly() {
    let _guard = backend_guard();
    let counting = install_counting();

    let values = sample_values(64);
    let uploaded = cpu_tensor(values.clone(), vec![64])
        .to_device(DeviceType::Cuda(0))
        .expect("to_device(Cuda)");
    let storage_type = uploaded.storage_type();

    // A resident operand costs no further transfer.
    let before = counting.h2d();
    let relu = uploaded.relu().expect("relu");
    let after = counting.h2d();
    let got = relu.to_vec().expect("to_vec");
    gpu_dispatch::clear_backend();

    assert_eq!(storage_type, "device", "to_device(Cuda) must upload");
    assert_eq!(after - before, 0, "a resident operand must not re-upload");
    let expected: Vec<f32> = values.iter().map(|value| value.max(0.0)).collect();
    assert_all_close(&got, &expected, 1e-6, "relu of an uploaded tensor");
}

// ---------------------------------------------------------------------------
// T13 — autograd through a device-resident forward pass
// ---------------------------------------------------------------------------

/// Central-difference gradient of `loss` at each element of `values`.
///
/// Same helper (and step size) as `tests/hardening_autograd.rs`: `backward()`
/// runs in single precision, so a comparatively large step keeps the truncation
/// error below the rounding error.
fn numerical_gradient<F>(values: &[f32], loss: F) -> Vec<f32>
where
    F: Fn(&[f32]) -> f32,
{
    const STEP: f32 = 1e-2;
    (0..values.len())
        .map(|index| {
            let mut plus = values.to_vec();
            plus[index] += STEP;
            let mut minus = values.to_vec();
            minus[index] -= STEP;
            (loss(&plus) - loss(&minus)) / (2.0 * STEP)
        })
        .collect()
}

/// A device-resident axis reduction must still join the autograd graph.
///
/// The dispatch replaces the tensor the host path would have built, so the
/// `SumDim` record has to be applied to whichever tensor the dispatch produced.
/// If it were applied only on the host branch, the reduction would silently
/// become a detached leaf and the gradient would vanish.
#[test]
fn device_axis_sum_backward_matches_host() {
    let _guard = backend_guard();
    let counting = install_counting();

    let values = sample_values(24);
    let device_input = cuda_tensor(values.clone(), vec![2, 3, 4]).requires_grad_(true);
    let device_reduced = device_input.sum_dim(&[1], false).expect("sum_dim");
    assert_eq!(
        device_reduced.storage_type(),
        "device",
        "the reduction must have taken the device path"
    );
    assert!(
        device_reduced.requires_grad(),
        "the device result must join the autograd graph"
    );
    device_reduced
        .sum()
        .expect("sum")
        .backward()
        .expect("backward");
    let device_grad = device_input
        .grad()
        .expect("device gradient")
        .to_vec()
        .expect("vec");

    let dispatched = counting.h2d() > 0;
    gpu_dispatch::clear_backend();

    let host_input = cpu_tensor(values, vec![2, 3, 4]).requires_grad_(true);
    host_input
        .sum_dim(&[1], false)
        .expect("sum_dim")
        .sum()
        .expect("sum")
        .backward()
        .expect("backward");
    let host_grad = host_input
        .grad()
        .expect("host gradient")
        .to_vec()
        .expect("vec");

    assert!(dispatched, "the reduction must have run on the device");
    assert_eq!(
        device_grad, host_grad,
        "residency must not perturb the SumDim gradient"
    );
    // d(sum(sum_dim(x)))/dx is 1 for every element, whichever path ran.
    assert_eq!(device_grad, vec![1.0f32; 24]);
}

/// Backward through a device-resident forward pass must produce the same
/// gradients as the host, and as finite differences.
///
/// `tanh` is deliberate: the device kernel and the host path are both `f32::tanh`,
/// so the two analytic gradients are directly comparable.
#[test]
fn device_resident_unary_backward_matches_finite_differences() {
    let _guard = backend_guard();
    let counting = install_counting();

    let values = sample_values(16);

    let device_input = cuda_tensor(values.clone(), vec![16]).requires_grad_(true);
    let device_loss = device_input.tanh().expect("tanh").sum().expect("sum");
    device_loss.backward().expect("backward");
    let device_grad = device_input
        .grad()
        .expect("device gradient")
        .to_vec()
        .expect("vec");

    let dispatched = counting.h2d() > 0;
    gpu_dispatch::clear_backend();

    let host_input = cpu_tensor(values.clone(), vec![16]).requires_grad_(true);
    let host_loss = host_input.tanh().expect("tanh").sum().expect("sum");
    host_loss.backward().expect("backward");
    let host_grad = host_input
        .grad()
        .expect("host gradient")
        .to_vec()
        .expect("vec");

    let numeric = numerical_gradient(&values, |probe| probe.iter().map(|x| x.tanh()).sum());

    assert!(dispatched, "the forward pass must have run on the device");
    assert_eq!(
        device_grad, host_grad,
        "residency must not perturb the gradient"
    );
    for (index, (got, expected)) in device_grad.iter().zip(numeric.iter()).enumerate() {
        let tolerance = 2e-2 * expected.abs().max(1.0);
        assert!(
            (got - expected).abs() <= tolerance,
            "gradient[{index}] = {got}, finite differences gave {expected}"
        );
    }
}
