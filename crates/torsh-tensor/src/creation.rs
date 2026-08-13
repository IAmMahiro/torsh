//! Tensor creation functions

use crate::{FloatElement, Tensor, TensorElement};
// ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
use scirs2_core::random::{Random, StdRng as SeededRng};
use scirs2_core::RngExt;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use torsh_core::{
    device::DeviceType,
    dtype::{Complex32, Complex64, ComplexElement},
    error::{Result, TorshError},
};

// ============================================================================
// Process-global random number generation (PyTorch-compatible semantics)
// ============================================================================
//
// Every random constructor draws from a *thread-local* generator that is seeded
// from OS entropy the first time it is used, so two calls never return the same
// data by accident. [`manual_seed`] installs a process-wide seed (mirroring
// `torch.manual_seed`) and bumps a generation counter; each thread notices the
// new generation on its next draw and re-seeds its generator deterministically
// from `(manual seed, thread stream index)`. That keeps single-threaded programs
// bit-reproducible after `manual_seed` while never handing two threads the same
// stream.

/// Generation counter, bumped by [`manual_seed`]. `0` means "no manual seed yet".
static SEED_GENERATION: AtomicU64 = AtomicU64::new(0);

/// Seed installed by the most recent [`manual_seed`] call.
static MANUAL_SEED: AtomicU64 = AtomicU64::new(0);

/// Hands out a distinct stream index to every thread that draws random numbers.
static NEXT_STREAM: AtomicU64 = AtomicU64::new(0);

/// Per-thread generator state.
struct ThreadRngState {
    /// Seed generation this RNG was created for.
    generation: u64,
    /// Stable per-thread stream index (survives re-seeding).
    stream: u64,
    /// The generator itself.
    rng: SeededRng,
}

thread_local! {
    static THREAD_RNG: RefCell<Option<ThreadRngState>> = const { RefCell::new(None) };
}

/// SplitMix64 finaliser — mixes a counter/seed pair into a well-distributed seed.
fn splitmix64(value: u64) -> u64 {
    let mut z = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Sets the seed of the ToRSh random number generator for the whole process.
///
/// This mirrors `torch.manual_seed`: after calling it, the sequence of values
/// produced by [`rand`], [`randn`], [`randint`] and the complex variants is
/// reproducible for a single-threaded program. Threads that draw random numbers
/// get distinct (but deterministically derived) streams, so results are
/// reproducible per thread rather than dependent on scheduling.
///
/// Without a call to `manual_seed`, every thread seeds itself from OS entropy,
/// so each process — and each tensor — gets genuinely different random data.
///
/// # Examples
///
/// ```
/// use torsh_tensor::creation::{manual_seed, randn};
///
/// manual_seed(42);
/// let a = randn::<f32>(&[4]).expect("operation should succeed");
/// manual_seed(42);
/// let b = randn::<f32>(&[4]).expect("operation should succeed");
/// assert_eq!(a.to_vec().expect("to_vec"), b.to_vec().expect("to_vec"));
/// ```
pub fn manual_seed(seed: u64) {
    MANUAL_SEED.store(seed, Ordering::SeqCst);
    // Bump last so that a thread observing the new generation also observes the
    // new seed value.
    SEED_GENERATION.fetch_add(1, Ordering::SeqCst);
}

/// Run `f` with exclusive access to the calling thread's generator.
fn with_rng<R>(f: impl FnOnce(&mut SeededRng) -> R) -> R {
    THREAD_RNG.with(|cell| {
        let mut slot = cell.borrow_mut();
        let generation = SEED_GENERATION.load(Ordering::SeqCst);
        let stale = slot
            .as_ref()
            .map(|state| state.generation != generation)
            .unwrap_or(true);

        if stale {
            let stream = match slot.as_ref() {
                Some(state) => state.stream,
                None => NEXT_STREAM.fetch_add(1, Ordering::Relaxed),
            };
            let seed = if generation == 0 {
                // No manual seed installed: draw a fresh seed from OS entropy.
                scirs2_core::random::random::<u64>()
            } else {
                splitmix64(MANUAL_SEED.load(Ordering::SeqCst) ^ splitmix64(stream))
            };
            *slot = Some(ThreadRngState {
                generation,
                stream,
                rng: Random::seed(seed),
            });
        }

        // `slot` is `Some` at this point; the fallback never runs.
        let state = slot.get_or_insert_with(|| ThreadRngState {
            generation,
            stream: 0,
            rng: Random::seed(0),
        });
        f(&mut state.rng)
    })
}

/// Draw a single standard-normal sample pair using the Box-Muller transform.
fn box_muller(rng: &mut SeededRng) -> (f64, f64) {
    // `u1` must be strictly positive: `ln(0)` is `-inf`, which would poison the
    // sample with NaN.
    let u1: f64 = rng.gen_range(f64::MIN_POSITIVE..1.0);
    let u2: f64 = rng.gen_range(0.0..1.0);
    let radius = (-2.0_f64 * u1.ln()).sqrt();
    let angle = 2.0_f64 * std::f64::consts::PI * u2;
    (radius * angle.cos(), radius * angle.sin())
}

/// Convert an `f64` sample into the tensor element type.
fn sample_to_element<T: TensorElement>(value: f64) -> Result<T> {
    T::from_f64(value).ok_or_else(|| {
        TorshError::InvalidArgument(format!(
            "cannot represent random sample {value} as {:?}",
            T::dtype()
        ))
    })
}

/// Create a tensor from a scalar value
pub fn tensor_scalar<T: TensorElement>(value: T) -> Result<Tensor<T>> {
    Tensor::from_data(vec![value], vec![], DeviceType::Cpu)
}

/// Create a 1D tensor from a slice
pub fn tensor_1d<T: TensorElement>(data: &[T]) -> Result<Tensor<T>> {
    Tensor::from_data(data.to_vec(), vec![data.len()], DeviceType::Cpu)
}

/// Create a 2D tensor from nested slices
pub fn tensor_2d<T: TensorElement>(data: &[&[T]]) -> Result<Tensor<T>> {
    let rows = data.len();
    let cols = if rows > 0 { data[0].len() } else { 0 };

    let mut flat_data = Vec::with_capacity(rows * cols);
    for row in data {
        flat_data.extend_from_slice(row);
    }

    Tensor::from_data(flat_data, vec![rows, cols], DeviceType::Cpu)
}

/// Create a 2D tensor from nested arrays (for macro use)
pub fn tensor_2d_arrays<T: TensorElement, const M: usize, const N: usize>(
    data: &[[T; N]; M],
) -> Result<Tensor<T>> {
    let rows = M;
    let cols = N;

    let mut flat_data = Vec::with_capacity(rows * cols);
    for row in data {
        flat_data.extend_from_slice(row);
    }

    Tensor::from_data(flat_data, vec![rows, cols], DeviceType::Cpu)
}

/// Creates a tensor filled with zeros.
///
/// This is one of the most common tensor creation functions, useful for initializing
/// tensors before filling them with computed values.
///
/// # Arguments
///
/// * `shape` - The shape of the tensor as a slice of dimensions
///
/// # Returns
///
/// A new tensor filled with zeros on the CPU device, or an error if creation fails.
///
/// # Examples
///
/// ```
/// use torsh_tensor::creation::zeros;
///
/// // Create a 1D tensor with 5 elements
/// let t = zeros::<f32>(&[5]).expect("operation should succeed");
/// assert_eq!(t.shape().dims(), &[5]);
///
/// // Create a 2D tensor (matrix)
/// let m = zeros::<f32>(&[3, 4]).expect("operation should succeed");
/// assert_eq!(m.shape().dims(), &[3, 4]);
/// assert_eq!(m.numel(), 12);
///
/// // Create a 3D tensor
/// let cube = zeros::<f32>(&[2, 3, 4]).expect("operation should succeed");
/// assert_eq!(cube.shape().dims(), &[2, 3, 4]);
/// ```
///
/// # See Also
///
/// * [`ones`] - Create a tensor filled with ones
/// * [`zeros_like`] - Create zeros matching another tensor's shape
/// * [`zeros_device`] - Create zeros on a specific device
pub fn zeros<T: TensorElement>(shape: &[usize]) -> Result<Tensor<T>> {
    let size = shape.iter().product();
    let data = vec![T::zero(); size];

    Tensor::from_data(data, shape.to_vec(), DeviceType::Cpu)
}

/// Create a tensor of zeros backed by plain in-memory storage
///
/// `SimdOptimized` storage now supports mutation too (through a copy-on-write
/// buffer), so this is no longer required for correctness — it just skips the
/// SIMD alignment copy for tensors that are going to be written element by
/// element anyway.
pub fn zeros_mut<T: TensorElement>(shape: &[usize]) -> Tensor<T> {
    let size = shape.iter().product();
    let data = vec![T::zero(); size];

    Tensor::from_data_fast(data, shape.to_vec(), DeviceType::Cpu)
}

/// Create a tensor of zeros on a specific device
pub fn zeros_device<T: TensorElement>(shape: &[usize], device: DeviceType) -> Result<Tensor<T>> {
    let size = shape.iter().product();
    let data = vec![T::zero(); size];

    Tensor::from_data(data, shape.to_vec(), device)
}

/// Creates a tensor filled with ones.
///
/// Commonly used for creating masks, initializing accumulators, or as a starting
/// point for mathematical operations.
///
/// # Arguments
///
/// * `shape` - The shape of the tensor as a slice of dimensions
///
/// # Returns
///
/// A new tensor filled with ones on the CPU device, or an error if creation fails.
///
/// # Examples
///
/// ```
/// use torsh_tensor::creation::ones;
///
/// // Create a 1D tensor
/// let t = ones::<f32>(&[5]).expect("operation should succeed");
/// assert_eq!(t.to_vec().expect("to_vec should succeed"), vec![1.0; 5]);
///
/// // Create a 2D tensor and use it as a mask
/// let mask = ones::<f32>(&[2, 3]).expect("operation should succeed");
/// assert_eq!(mask.shape().dims(), &[2, 3]);
///
/// // Different data types
/// let int_ones = ones::<i32>(&[4]).expect("operation should succeed");
/// assert_eq!(int_ones.to_vec().expect("to_vec should succeed"), vec![1; 4]);
/// ```
///
/// # See Also
///
/// * [`zeros`] - Create a tensor filled with zeros
/// * [`ones_like`] - Create ones matching another tensor's shape
/// * [`full`] - Create a tensor filled with any value
pub fn ones<T: TensorElement>(shape: &[usize]) -> Result<Tensor<T>> {
    let size = shape.iter().product();
    let data = vec![T::one(); size];

    Tensor::from_data(data, shape.to_vec(), DeviceType::Cpu)
}

/// Create a tensor of ones on a specific device  
pub fn ones_device<T: TensorElement>(shape: &[usize], device: DeviceType) -> Result<Tensor<T>> {
    let size = shape.iter().product();
    let data = vec![T::one(); size];

    Tensor::from_data(data, shape.to_vec(), device)
}

/// Creates a tensor filled with a specific value.
///
/// Useful for initializing tensors with custom values, creating constant tensors,
/// or setting initial biases in neural networks.
///
/// # Arguments
///
/// * `shape` - The shape of the tensor as a slice of dimensions
/// * `value` - The value to fill the tensor with
///
/// # Returns
///
/// A new tensor filled with the specified value.
///
/// # Examples
///
/// ```
/// use torsh_tensor::creation::full;
///
/// // Create a tensor filled with a specific value
/// let t = full(&[2, 3], 7.0f32).expect("operation should succeed");
/// assert_eq!(t.shape().dims(), &[2, 3]);
/// assert_eq!(t.to_vec().expect("to_vec should succeed"), vec![7.0; 6]);
///
/// // Initialize bias tensor
/// let bias = full(&[256], 0.01f32).expect("operation should succeed");
/// assert_eq!(bias.numel(), 256);
///
/// // Create constant tensor for operations
/// let pi_tensor = full(&[10], std::f32::consts::PI).expect("operation should succeed");
/// ```
///
/// # See Also
///
/// * [`zeros`] - Create a tensor filled with zeros
/// * [`ones`] - Create a tensor filled with ones
/// * [`full_like`] - Create full tensor matching another's shape
pub fn full<T: TensorElement>(shape: &[usize], value: T) -> Result<Tensor<T>> {
    let size = shape.iter().product();
    let data = vec![value; size];

    Tensor::from_data(data, shape.to_vec(), DeviceType::Cpu)
}

/// Creates an identity matrix (2D tensor with ones on the diagonal).
///
/// An identity matrix is a square matrix with ones on the main diagonal and
/// zeros elsewhere. It's fundamental in linear algebra and commonly used in
/// neural network operations.
///
/// # Arguments
///
/// * `n` - The size of the square identity matrix (n × n)
///
/// # Returns
///
/// A new n×n tensor with ones on the diagonal and zeros elsewhere.
///
/// # Examples
///
/// ```
/// use torsh_tensor::creation::eye;
///
/// // Create a 3x3 identity matrix
/// let identity = eye::<f32>(3).expect("operation should succeed");
/// assert_eq!(identity.shape().dims(), &[3, 3]);
///
/// let data = identity.to_vec().expect("to_vec should succeed");
/// // Expected: [1, 0, 0,
/// //            0, 1, 0,
/// //            0, 0, 1]
/// assert_eq!(data[0], 1.0);  // (0,0)
/// assert_eq!(data[1], 0.0);  // (0,1)
/// assert_eq!(data[4], 1.0);  // (1,1)
/// assert_eq!(data[8], 1.0);  // (2,2)
///
/// // Use in linear algebra operations
/// let matrix = eye::<f32>(4).expect("operation should succeed");
/// // matrix @ vector preserves the vector (identity property)
/// ```
///
/// # See Also
///
/// * [`zeros`] - Create a tensor filled with zeros
/// * [`ones`] - Create a tensor filled with ones
pub fn eye<T: TensorElement>(n: usize) -> Result<Tensor<T>> {
    let mut data = vec![T::zero(); n * n];
    for i in 0..n {
        data[i * n + i] = T::one();
    }

    Tensor::from_data(data, vec![n, n], DeviceType::Cpu)
}

/// Creates a 1D tensor with evenly spaced values within a given interval.
///
/// Similar to Python's `range()` or NumPy's `arange()`, this function generates
/// a sequence of values starting from `start` up to (but not including) `end`,
/// incrementing by `step`.
///
/// # Arguments
///
/// * `start` - The starting value (inclusive)
/// * `end` - The ending value (exclusive)
/// * `step` - The spacing between values
///
/// # Returns
///
/// A 1D tensor containing the sequence of values.
///
/// # Examples
///
/// ```
/// use torsh_tensor::creation::arange;
///
/// // Create a sequence from 0 to 9
/// let t = arange(0, 10, 1).expect("operation should succeed");
/// assert_eq!(t.to_vec().expect("to_vec should succeed"), vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
///
/// // Create a sequence with step size 2
/// let evens = arange(0, 10, 2).expect("operation should succeed");
/// assert_eq!(evens.to_vec().expect("to_vec should succeed"), vec![0, 2, 4, 6, 8]);
///
/// // Floating point sequences
/// let floats = arange(0.0f32, 1.0, 0.25).expect("operation should succeed");
/// assert_eq!(floats.shape().dims(), &[4]);
///
/// // Descending ranges use a negative step, like `torch.arange`
/// let down = arange(5, 0, -1).expect("operation should succeed");
/// assert_eq!(down.to_vec().expect("to_vec should succeed"), vec![5, 4, 3, 2, 1]);
///
/// // A zero step is rejected instead of looping forever
/// assert!(arange(0.0f32, 1.0, 0.0).is_err());
///
/// // Use for indexing or creating coordinate grids
/// let indices = arange(0, 100, 1).expect("operation should succeed");
/// ```
///
/// # Errors
///
/// Returns [`TorshError::InvalidArgument`] if `step` is zero (or NaN), since no
/// such sequence can terminate.
///
/// # See Also
///
/// * [`linspace`] - Create linearly spaced values with exact count
/// * [`zeros`] - Create a tensor filled with zeros
pub fn arange<T: TensorElement + std::cmp::PartialOrd + std::ops::Add<Output = T> + Copy>(
    start: T,
    end: T,
    step: T,
) -> Result<Tensor<T>> {
    let zero = <T as TensorElement>::zero();
    let ascending = step > zero;
    let descending = step < zero;

    // Catches both `step == 0` and a NaN step: neither comparison holds, and
    // neither can ever reach `end`.
    if !ascending && !descending {
        return Err(TorshError::InvalidArgument(
            "arange requires a non-zero, non-NaN step".to_string(),
        ));
    }

    let mut values = Vec::new();
    let mut current = start;

    loop {
        let in_range = if ascending {
            current < end
        } else {
            current > end
        };
        if !in_range {
            break;
        }
        values.push(current);
        let next = current + step;
        // Guard against a floating-point step too small to advance `current`
        // (e.g. a subnormal step next to a large start), which would otherwise
        // spin forever.
        if next == current {
            break;
        }
        current = next;
    }

    let len = values.len();
    Tensor::from_data(values, vec![len], DeviceType::Cpu)
}

/// Create a tensor with linearly spaced values
pub fn linspace<T: FloatElement>(start: T, end: T, steps: usize) -> Result<Tensor<T>> {
    if steps == 0 {
        return zeros(&[0]);
    }

    if steps == 1 {
        return tensor_scalar(start);
    }

    let mut values = Vec::with_capacity(steps);
    let step_size = (end - start) / T::from(steps - 1).expect("numeric conversion should succeed");

    for i in 0..steps {
        let value = start + step_size * T::from(i).expect("numeric conversion should succeed");
        values.push(value);
    }

    Tensor::from_data(values, vec![steps], DeviceType::Cpu)
}

/// Creates a tensor with random values from uniform distribution [0, 1).
///
/// Useful for generating random data, initialization schemes that require uniform
/// distribution, or Monte Carlo simulations.
///
/// **Randomness**: draws from the process-global generator, which seeds itself
/// from OS entropy on first use. Call [`manual_seed`] first when you need a
/// reproducible sequence.
///
/// # Arguments
///
/// * `shape` - The shape of the tensor as a slice of dimensions
///
/// # Returns
///
/// A new tensor with values uniformly distributed in [0, 1), or an error if creation fails.
///
/// # Examples
///
/// ```
/// use torsh_tensor::creation::rand;
///
/// // Create random tensor
/// let t = rand::<f32>(&[3, 3]).expect("operation should succeed");
/// assert_eq!(t.shape().dims(), &[3, 3]);
///
/// // Values should be in [0, 1)
/// let data = t.to_vec().expect("to_vec should succeed");
/// for &val in &data {
///     assert!(val >= 0.0 && val < 1.0);
/// }
///
/// // Use for random initialization
/// let weights = rand::<f32>(&[128, 64]).expect("operation should succeed");
/// assert_eq!(weights.numel(), 128 * 64);
/// ```
///
/// # See Also
///
/// * [`randn`] - Create tensor with normal distribution
/// * [`randint`] - Create tensor with random integers
/// * [`rand_like`] - Create random tensor matching another's shape
/// * [`manual_seed`] - Make the sequence reproducible
pub fn rand<T: FloatElement>(shape: &[usize]) -> Result<Tensor<T>>
where
    T: From<f32>,
{
    let size = shape.iter().product();
    let values: Vec<T> = with_rng(|rng| {
        (0..size)
            .map(|_| <T as From<f32>>::from(rng.random::<f32>()))
            .collect()
    });

    Tensor::from_data(values, shape.to_vec(), DeviceType::Cpu)
}

/// Create a uniform-random tensor from an explicit seed (reproducible).
///
/// Unlike [`rand`], this does not touch the process-global generator, so it is
/// safe to use in tests that must not perturb other threads' sequences.
pub fn rand_with_seed<T: FloatElement>(shape: &[usize], seed: u64) -> Result<Tensor<T>>
where
    T: From<f32>,
{
    let size = shape.iter().product();
    let mut rng = Random::seed(seed);
    let values: Vec<T> = (0..size)
        .map(|_| <T as From<f32>>::from(rng.random::<f32>()))
        .collect();

    Tensor::from_data(values, shape.to_vec(), DeviceType::Cpu)
}

/// Creates a tensor with random values from standard normal distribution N(0, 1).
///
/// This is the most common initialization method for neural network weights,
/// as it provides a good starting point for gradient-based optimization.
/// Uses the Box-Muller transform to generate normally distributed values.
///
/// **Randomness**: draws from the process-global generator, which seeds itself
/// from OS entropy on first use. Call [`manual_seed`] first when you need a
/// reproducible sequence.
///
/// # Arguments
///
/// * `shape` - The shape of the tensor as a slice of dimensions
///
/// # Returns
///
/// A new tensor with values from N(0, 1) distribution, or an error if creation fails.
///
/// # Examples
///
/// ```
/// use torsh_tensor::creation::{manual_seed, randn};
///
/// manual_seed(0);
///
/// // Create random normal tensor
/// let t = randn::<f32>(&[1000]).expect("operation should succeed");
/// assert_eq!(t.shape().dims(), &[1000]);
///
/// // Initialize neural network layer weights
/// let weights = randn::<f32>(&[512, 256]).expect("operation should succeed");
/// assert_eq!(weights.shape().dims(), &[512, 256]);
///
/// // The values follow normal distribution with mean~0 and std~1
/// let data = t.to_vec().expect("to_vec should succeed");
/// let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
/// assert!((mean.abs() < 0.2), "Mean should be close to 0");
/// ```
///
/// # Implementation Details
///
/// Uses the Box-Muller transform to convert uniform random numbers into
/// normally distributed values. Sampling happens in `f64` and the result is
/// converted through [`TensorElement::from_f64`], so narrow float types such as
/// `f16`/`bf16` get a correctly rounded sample rather than a reinterpreted bit
/// pattern.
///
/// # See Also
///
/// * [`rand`] - Create tensor with uniform distribution
/// * [`randn_like`] - Create normal random tensor matching another's shape
/// * [`zeros`] - Create tensor filled with zeros
/// * [`manual_seed`] - Make the sequence reproducible
pub fn randn<T: FloatElement>(shape: &[usize]) -> Result<Tensor<T>> {
    let size = shape.iter().product();
    let samples = with_rng(|rng| normal_samples(rng, size));
    let values = samples
        .into_iter()
        .map(sample_to_element::<T>)
        .collect::<Result<Vec<T>>>()?;

    Tensor::from_data(values, shape.to_vec(), DeviceType::Cpu)
}

/// Create a standard-normal tensor from an explicit seed (reproducible).
///
/// Unlike [`randn`], this does not touch the process-global generator.
pub fn randn_with_seed<T: FloatElement>(shape: &[usize], seed: u64) -> Result<Tensor<T>> {
    let size = shape.iter().product();
    let mut rng = Random::seed(seed);
    let values = normal_samples(&mut rng, size)
        .into_iter()
        .map(sample_to_element::<T>)
        .collect::<Result<Vec<T>>>()?;

    Tensor::from_data(values, shape.to_vec(), DeviceType::Cpu)
}

/// Draw `size` independent N(0, 1) samples (two per Box-Muller evaluation).
fn normal_samples(rng: &mut SeededRng, size: usize) -> Vec<f64> {
    let mut samples = Vec::with_capacity(size);
    while samples.len() < size {
        let (first, second) = box_muller(rng);
        samples.push(first);
        if samples.len() < size {
            samples.push(second);
        }
    }
    samples
}

impl<T: FloatElement> Tensor<T> {
    /// Fills this tensor in-place with samples from a normal (Gaussian)
    /// distribution `N(mean, std^2)`.
    ///
    /// # PyTorch Compatibility
    ///
    /// Equivalent to `Tensor.normal_(mean, std)`. Like every other in-place
    /// mutator on this type, it refuses to run on a tensor that
    /// `requires_grad`, since autograd cannot track through a buffer
    /// mutation.
    ///
    /// **Randomness**: draws from the process-global generator, which seeds
    /// itself from OS entropy on first use. Call [`manual_seed`] first when
    /// you need a reproducible sequence.
    ///
    /// # Errors
    ///
    /// Returns [`TorshError::InvalidArgument`] if:
    /// - `self.requires_grad()` is `true`,
    /// - `std` is negative or not finite,
    /// - `mean` is not finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_tensor::creation::{manual_seed, zeros};
    ///
    /// manual_seed(0);
    /// let mut t = zeros::<f32>(&[10, 10]).expect("operation should succeed");
    /// t.normal_(0.0, 1.0).expect("operation should succeed");
    ///
    /// let data = t.to_vec().expect("to_vec should succeed");
    /// assert!(
    ///     data.iter().any(|&x| x != 0.0),
    ///     "normal_ should fill the tensor with non-zero values"
    /// );
    /// ```
    ///
    /// # See Also
    ///
    /// * [`randn`] - Create a new standard-normal tensor
    /// * [`manual_seed`] - Make the sequence reproducible
    pub fn normal_(&mut self, mean: f64, std: f64) -> Result<()> {
        if self.requires_grad {
            return Err(TorshError::InvalidArgument(
                "In-place operation `normal_` on tensor that requires grad is not allowed"
                    .to_string(),
            ));
        }
        if !std.is_finite() || std < 0.0 {
            return Err(TorshError::InvalidArgument(format!(
                "normal_ expects a finite std >= 0.0, got {std}"
            )));
        }
        if !mean.is_finite() {
            return Err(TorshError::InvalidArgument(format!(
                "normal_ expects a finite mean, got {mean}"
            )));
        }

        let numel = self.numel();
        let raw = with_rng(|rng| normal_samples(rng, numel));
        let values = raw
            .into_iter()
            .map(|z| sample_to_element::<T>(mean + std * z))
            .collect::<Result<Vec<T>>>()?;

        // `data_mut_apply` takes an infallible `FnMut(&mut T)`, so the
        // fallible `f64 -> T` conversion happens eagerly above; this closure
        // only ever assigns already-converted values. `written` is checked
        // against `numel` afterwards so a mismatch surfaces as an honest
        // error instead of silently leaving some elements at their old
        // value.
        let mut written = 0usize;
        self.data_mut_apply(|item| {
            if let Some(&value) = values.get(written) {
                *item = value;
            }
            written += 1;
        })?;

        if written != numel {
            return Err(TorshError::InvalidArgument(format!(
                "normal_ wrote {written} elements but the tensor has {numel}"
            )));
        }

        Ok(())
    }
}

impl Tensor<f32> {
    /// Draws samples from a multinomial (categorical) distribution defined by
    /// `weights`.
    ///
    /// Each entry of `weights` is the unnormalized probability of drawing its
    /// index; `weights` does not need to sum to 1. Returns a 1-D `i64` tensor
    /// of `num_samples` category indices, on the same device as `weights`.
    ///
    /// # PyTorch Compatibility
    ///
    /// Equivalent to `weights.multinomial(num_samples, replacement)` for a
    /// 1-D `weights` tensor. Batched (2-D) `weights` is not yet supported and
    /// returns [`TorshError::InvalidArgument`] rather than silently
    /// misinterpreting the input.
    ///
    /// **Randomness**: draws from the process-global generator; see
    /// [`manual_seed`].
    ///
    /// # Errors
    ///
    /// Returns [`TorshError::InvalidArgument`] if:
    /// - `weights` is not 1-dimensional,
    /// - any weight is negative or not finite,
    /// - every weight is zero (there is nothing to sample),
    /// - `replacement` is `false` and `num_samples` exceeds the number of
    ///   strictly-positive weights.
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_tensor::creation::{manual_seed, tensor_1d};
    /// use torsh_tensor::Tensor;
    ///
    /// manual_seed(0);
    /// let weights = tensor_1d(&[0.1f32, 0.2, 0.3, 0.4]).expect("operation should succeed");
    /// let samples = Tensor::multinomial(&weights, 10, true).expect("operation should succeed");
    /// assert_eq!(samples.shape().dims(), &[10]);
    /// ```
    pub fn multinomial(
        weights: &Self,
        num_samples: usize,
        replacement: bool,
    ) -> Result<Tensor<i64>> {
        if weights.ndim() != 1 {
            return Err(TorshError::InvalidArgument(format!(
                "multinomial expects a 1-D weights tensor, got {} dimensions",
                weights.ndim()
            )));
        }

        let mut pool = weights.to_vec()?;
        for &w in &pool {
            if !w.is_finite() || w < 0.0 {
                return Err(TorshError::InvalidArgument(format!(
                    "multinomial: weights must be finite and non-negative, found {w}"
                )));
            }
        }

        let num_positive = pool.iter().filter(|&&w| w > 0.0).count();
        if num_positive == 0 {
            return Err(TorshError::InvalidArgument(
                "multinomial: weights sum must be positive".to_string(),
            ));
        }
        if !replacement && num_samples > num_positive {
            return Err(TorshError::InvalidArgument(format!(
                "multinomial: cannot sample {num_samples} indices without replacement \
                 from {num_positive} categories with positive weight"
            )));
        }

        let mut samples = Vec::with_capacity(num_samples);
        with_rng(|rng| -> Result<()> {
            for _ in 0..num_samples {
                let total: f32 = pool.iter().sum();
                if total <= 0.0 {
                    return Err(TorshError::InvalidArgument(
                        "multinomial: ran out of positive-weight categories while \
                         sampling without replacement"
                            .to_string(),
                    ));
                }
                let draw: f32 = rng.gen_range(0.0..total);

                // Find the first category whose cumulative weight passes `draw`.
                let mut cumulative = 0.0f32;
                let mut chosen = None;
                for (i, &w) in pool.iter().enumerate() {
                    if w <= 0.0 {
                        continue;
                    }
                    cumulative += w;
                    if draw < cumulative {
                        chosen = Some(i);
                        break;
                    }
                }
                // Floating-point rounding can (rarely) leave `draw` just past
                // the last positive category's cumulative weight; fall back
                // to that last positive category rather than risk selecting
                // a zero-weight one.
                let chosen = match chosen {
                    Some(i) => i,
                    None => pool
                        .iter()
                        .enumerate()
                        .rev()
                        .find(|&(_, &w)| w > 0.0)
                        .map(|(i, _)| i)
                        .ok_or_else(|| {
                            TorshError::InvalidArgument(
                                "multinomial: no positive-weight category available \
                                 to sample"
                                    .to_string(),
                            )
                        })?,
                };

                samples.push(chosen as i64);
                if !replacement {
                    pool[chosen] = 0.0;
                }
            }
            Ok(())
        })?;

        let len = samples.len();
        Tensor::from_data(samples, vec![len], weights.device())
    }
}

/// Create a tensor with random integers in `[low, high)`
///
/// **Randomness**: draws from the process-global generator; see [`manual_seed`].
pub fn randint(low: i32, high: i32, shape: &[usize]) -> Result<Tensor<i32>> {
    let size = shape.iter().product();
    use scirs2_core::random::Uniform;
    let dist = Uniform::new(low, high)
        .map_err(|e| TorshError::InvalidArgument(format!("Invalid range for randint: {}", e)))?;
    let values: Vec<i32> = with_rng(|rng| (0..size).map(|_| rng.sample(&dist)).collect());

    Tensor::from_data(values, shape.to_vec(), DeviceType::Cpu)
}

/// Create a tensor with random integers in `[low, high)` from an explicit seed.
pub fn randint_with_seed(low: i32, high: i32, shape: &[usize], seed: u64) -> Result<Tensor<i32>> {
    let size = shape.iter().product();
    use scirs2_core::random::Uniform;
    let dist = Uniform::new(low, high)
        .map_err(|e| TorshError::InvalidArgument(format!("Invalid range for randint: {}", e)))?;
    let mut rng = Random::seed(seed);
    let values: Vec<i32> = (0..size).map(|_| rng.sample(&dist)).collect();

    Tensor::from_data(values, shape.to_vec(), DeviceType::Cpu)
}

/// Create a tensor of zeros with the same shape as another tensor
pub fn zeros_like<T: TensorElement>(tensor: &Tensor<T>) -> Result<Tensor<T>> {
    zeros(tensor.shape().dims())
}

/// Create a tensor of ones with the same shape as another tensor
pub fn ones_like<T: TensorElement>(tensor: &Tensor<T>) -> Result<Tensor<T>> {
    ones(tensor.shape().dims())
}

/// Create a tensor filled with a value with the same shape as another tensor
pub fn full_like<T: TensorElement>(tensor: &Tensor<T>, value: T) -> Result<Tensor<T>> {
    full(tensor.shape().dims(), value)
}

/// Create a tensor with random values with the same shape as another tensor
pub fn rand_like<T: FloatElement>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: From<f32>,
{
    rand(tensor.shape().dims())
}

/// Create a tensor with random normal values with the same shape as another tensor
pub fn randn_like<T: FloatElement>(tensor: &Tensor<T>) -> Result<Tensor<T>> {
    randn(tensor.shape().dims())
}

// Complex tensor creation functions

/// Create a complex tensor from real and imaginary parts
pub fn complex_from_parts<T, C>(real: &Tensor<T>, imag: &Tensor<T>) -> Result<Tensor<C>>
where
    T: FloatElement,
    C: ComplexElement<Real = T> + TensorElement,
{
    if real.shape() != imag.shape() {
        return Err(torsh_core::error::TorshError::InvalidArgument(
            "Real and imaginary parts must have the same shape".to_string(),
        ));
    }

    let real_data = real.to_vec()?;
    let imag_data = imag.to_vec()?;

    let complex_data: Vec<C> = real_data
        .iter()
        .zip(imag_data.iter())
        .map(|(&r, &i)| C::new(r, i))
        .collect();

    Tensor::from_data(complex_data, real.shape().dims().to_vec(), real.device())
}

/// Create a complex tensor of zeros
pub fn complex_zeros<T, C>(shape: &[usize]) -> Result<Tensor<C>>
where
    T: FloatElement,
    C: ComplexElement<Real = T> + TensorElement,
{
    let size = shape.iter().product();
    let data = vec![C::new(<T as TensorElement>::zero(), <T as TensorElement>::zero()); size];

    Tensor::from_data(data, shape.to_vec(), DeviceType::Cpu)
}

/// Create a complex tensor of ones
pub fn complex_ones<T, C>(shape: &[usize]) -> Result<Tensor<C>>
where
    T: FloatElement,
    C: ComplexElement<Real = T> + TensorElement,
{
    let size = shape.iter().product();
    let data = vec![C::new(<T as TensorElement>::one(), <T as TensorElement>::zero()); size];

    Tensor::from_data(data, shape.to_vec(), DeviceType::Cpu)
}

/// Create a complex tensor filled with a specific value
pub fn complex_full<C: TensorElement>(shape: &[usize], value: C) -> Result<Tensor<C>> {
    let size = shape.iter().product();
    let data = vec![value; size];

    Tensor::from_data(data, shape.to_vec(), DeviceType::Cpu)
}

/// Create a complex tensor with random values (uniform distribution)
/// Real and imaginary parts are independently sampled from [0, 1)
pub fn complex_rand<T, C>(shape: &[usize]) -> Result<Tensor<C>>
where
    T: FloatElement + From<f32>,
    C: ComplexElement<Real = T> + TensorElement,
{
    let size = shape.iter().product();
    let values: Vec<C> = with_rng(|rng| {
        (0..size)
            .map(|_| {
                C::new(
                    <T as From<f32>>::from(rng.random::<f32>()),
                    <T as From<f32>>::from(rng.random::<f32>()),
                )
            })
            .collect()
    });

    Tensor::from_data(values, shape.to_vec(), DeviceType::Cpu)
}

/// Create a complex tensor with random values from standard normal distribution
/// Real and imaginary parts are independently sampled from N(0, 1)
pub fn complex_randn<T, C>(shape: &[usize]) -> Result<Tensor<C>>
where
    T: FloatElement,
    C: ComplexElement<Real = T> + TensorElement,
{
    let size = shape.iter().product();
    let pairs = with_rng(|rng| {
        (0..size)
            .map(|_| box_muller(rng))
            .collect::<Vec<(f64, f64)>>()
    });

    let mut values = Vec::with_capacity(size);
    for (real, imag) in pairs {
        values.push(C::new(
            sample_to_element::<T>(real)?,
            sample_to_element::<T>(imag)?,
        ));
    }

    Tensor::from_data(values, shape.to_vec(), DeviceType::Cpu)
}

/// Create complex tensor with the same shape as another tensor, filled with zeros
pub fn complex_zeros_like<T, C>(tensor: &Tensor<T>) -> Result<Tensor<C>>
where
    T: FloatElement,
    C: ComplexElement<Real = T> + TensorElement,
{
    complex_zeros(tensor.shape().dims())
}

/// Create complex tensor with the same shape as another tensor, filled with ones
pub fn complex_ones_like<T, C>(tensor: &Tensor<T>) -> Result<Tensor<C>>
where
    T: FloatElement,
    C: ComplexElement<Real = T> + TensorElement,
{
    complex_ones(tensor.shape().dims())
}

/// Create complex tensor with random values with the same shape as another tensor
pub fn complex_rand_like<T, C>(tensor: &Tensor<T>) -> Result<Tensor<C>>
where
    T: FloatElement + From<f32>,
    C: ComplexElement<Real = T> + TensorElement,
{
    complex_rand(tensor.shape().dims())
}

/// Create complex tensor with random normal values with the same shape as another tensor
pub fn complex_randn_like<T, C>(tensor: &Tensor<T>) -> Result<Tensor<C>>
where
    T: FloatElement,
    C: ComplexElement<Real = T> + TensorElement,
{
    complex_randn(tensor.shape().dims())
}

// Convenience functions for specific complex types

/// Create a Complex32 tensor of zeros
pub fn complex32_zeros(shape: &[usize]) -> Result<Tensor<Complex32>> {
    complex_zeros::<f32, Complex32>(shape)
}

/// Create a Complex32 tensor of ones
pub fn complex32_ones(shape: &[usize]) -> Result<Tensor<Complex32>> {
    complex_ones::<f32, Complex32>(shape)
}

/// Create a Complex32 tensor with random values
pub fn complex32_rand(shape: &[usize]) -> Result<Tensor<Complex32>> {
    complex_rand::<f32, Complex32>(shape)
}

/// Create a Complex32 tensor with random normal values
pub fn complex32_randn(shape: &[usize]) -> Result<Tensor<Complex32>> {
    complex_randn::<f32, Complex32>(shape)
}

/// Create a Complex64 tensor of zeros
pub fn complex64_zeros(shape: &[usize]) -> Result<Tensor<Complex64>> {
    complex_zeros::<f64, Complex64>(shape)
}

/// Create a Complex64 tensor of ones
pub fn complex64_ones(shape: &[usize]) -> Result<Tensor<Complex64>> {
    complex_ones::<f64, Complex64>(shape)
}

/// Create a Complex64 tensor with random values
pub fn complex64_rand(shape: &[usize]) -> Result<Tensor<Complex64>> {
    complex_rand::<f64, Complex64>(shape)
}

/// Create a Complex64 tensor with random normal values
pub fn complex64_randn(shape: &[usize]) -> Result<Tensor<Complex64>> {
    complex_randn::<f64, Complex64>(shape)
}

#[cfg(test)]
mod complex_tests {
    use super::*;
    use crate::tensor;

    #[test]
    fn test_complex_zeros() {
        let tensor = complex32_zeros(&[2, 3]).expect("operation should succeed");
        assert_eq!(tensor.shape().dims(), &[2, 3]);
        assert_eq!(tensor.numel(), 6);

        let data = tensor.to_vec().expect("to_vec should succeed");
        for &val in &data {
            assert_eq!(val.re, 0.0);
            assert_eq!(val.im, 0.0);
        }
    }

    #[test]
    fn test_complex_ones() {
        let tensor = complex32_ones(&[2, 2]).expect("operation should succeed");
        assert_eq!(tensor.shape().dims(), &[2, 2]);

        let data = tensor.to_vec().expect("to_vec should succeed");
        for &val in &data {
            assert_eq!(val.re, 1.0);
            assert_eq!(val.im, 0.0);
        }
    }

    #[test]
    fn test_complex_from_parts() {
        let real = tensor![1.0f32, 2.0, 3.0].expect("operation should succeed");
        let imag = tensor![4.0f32, 5.0, 6.0].expect("operation should succeed");

        let complex_tensor: Tensor<Complex32> =
            complex_from_parts(&real, &imag).expect("operation should succeed");
        assert_eq!(complex_tensor.shape().dims(), &[3]);

        let data = complex_tensor.to_vec().expect("to_vec should succeed");
        assert_eq!(data[0].re, 1.0);
        assert_eq!(data[0].im, 4.0);
        assert_eq!(data[1].re, 2.0);
        assert_eq!(data[1].im, 5.0);
        assert_eq!(data[2].re, 3.0);
        assert_eq!(data[2].im, 6.0);
    }

    #[test]
    fn test_complex_from_parts_shape_mismatch() {
        let real = tensor![1.0f32, 2.0].expect("operation should succeed");
        let imag = tensor![4.0f32, 5.0, 6.0].expect("operation should succeed");

        let result: Result<Tensor<Complex32>> = complex_from_parts(&real, &imag);
        assert!(result.is_err());
    }

    #[test]
    fn test_complex_rand() {
        let tensor = complex32_rand(&[10]).expect("operation should succeed");
        assert_eq!(tensor.shape().dims(), &[10]);

        let data = tensor.to_vec().expect("to_vec should succeed");
        // Check that we have some variation in real and imaginary parts
        let all_same_real = data.iter().all(|&c| c.re == data[0].re);
        let all_same_imag = data.iter().all(|&c| c.im == data[0].im);

        // With random data, this should be extremely unlikely
        assert!(!all_same_real || !all_same_imag);
    }

    #[test]
    fn test_complex_randn() {
        let tensor = complex64_randn(&[5, 5]).expect("operation should succeed");
        assert_eq!(tensor.shape().dims(), &[5, 5]);
        assert_eq!(tensor.numel(), 25);

        // Test that values are reasonably distributed (not all zeros)
        let data = tensor.to_vec().expect("to_vec should succeed");
        let has_nonzero_real = data.iter().any(|&c| c.re.abs() > 0.01);
        let has_nonzero_imag = data.iter().any(|&c| c.im.abs() > 0.01);

        assert!(has_nonzero_real);
        assert!(has_nonzero_imag);
    }

    #[test]
    fn test_complex_like_functions() {
        let base = tensor![1.0f32, 2.0, 3.0].expect("operation should succeed");

        let zeros: Tensor<Complex32> = complex_zeros_like(&base).expect("operation should succeed");
        assert_eq!(zeros.shape().dims(), base.shape().dims());

        let ones: Tensor<Complex32> = complex_ones_like(&base).expect("operation should succeed");
        assert_eq!(ones.shape().dims(), base.shape().dims());

        let random: Tensor<Complex32> = complex_rand_like(&base).expect("operation should succeed");
        assert_eq!(random.shape().dims(), base.shape().dims());

        let normal: Tensor<Complex32> =
            complex_randn_like(&base).expect("operation should succeed");
        assert_eq!(normal.shape().dims(), base.shape().dims());
    }
}

/// Create a tensor from a vector and shape.
///
/// Rejects a `data` length that does not equal `shape`'s element count, so a
/// mismatch surfaces here instead of as a confusing out-of-bounds error deep in
/// a later operation.
pub fn from_vec<T: TensorElement>(
    data: Vec<T>,
    shape: &[usize],
    device: DeviceType,
) -> Result<Tensor<T>> {
    let numel: usize = shape.iter().product();
    if data.len() != numel {
        return Err(TorshError::InvalidArgument(format!(
            "from_vec: data has {} elements but shape {:?} requires {}",
            data.len(),
            shape,
            numel
        )));
    }
    Tensor::from_data(data, shape.to_vec(), device)
}
