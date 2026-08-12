//! Regression tests for the autograd production-hardening pass.
//!
//! Each test pins one previously-broken behaviour: gradient clipping that
//! computed a coefficient and threw it away (F000), `no_grad()` guards that no
//! tensor operation consulted (F083), gradient checkpointing that did not exist
//! (F084), second-order hyperparameter gradients that returned zeros (F081),
//! GPU activation paths that returned their input verbatim (F001/F082), and a
//! quantum circuit builder that aborted the process on bad input (F184).

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};

use torsh_autograd::autograd_traits::AutogradTensor;
use torsh_autograd::clip;
use torsh_core::device::DeviceType;
use torsh_core::error::Result;
use torsh_tensor::Tensor;

/// Gradient mode is process-global, so tests that toggle it must not interleave
/// with tests that observe it.
static GRAD_MODE_LOCK: Mutex<()> = Mutex::new(());

fn grad_mode_guard() -> MutexGuard<'static, ()> {
    let guard = match GRAD_MODE_LOCK.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    torsh_core::grad_mode::set_grad_enabled(true);
    guard
}

/// Minimal `AutogradTensor` so the clipping API can be exercised on known data.
#[derive(Clone, Debug)]
struct GradBuffer {
    data: Vec<f32>,
}

impl GradBuffer {
    fn new(data: Vec<f32>) -> Self {
        Self { data }
    }
}

impl AutogradTensor<f32> for GradBuffer {
    fn shape(&self) -> torsh_core::shape::Shape {
        torsh_core::shape::Shape::new(vec![self.data.len()])
    }

    fn requires_grad(&self) -> bool {
        true
    }

    fn data(&self) -> Box<dyn std::ops::Deref<Target = [f32]> + '_> {
        Box::new(&self.data[..])
    }

    fn clone_tensor(&self) -> Box<dyn AutogradTensor<f32>> {
        Box::new(self.clone())
    }

    fn to_vec(&self) -> Vec<f32> {
        self.data.clone()
    }

    fn device(&self) -> &dyn torsh_core::Device {
        use std::sync::LazyLock;
        static CPU: LazyLock<torsh_core::device::CpuDevice> =
            LazyLock::new(torsh_core::device::CpuDevice::new);
        &*CPU
    }

    fn ones_like(&self) -> Box<dyn AutogradTensor<f32>> {
        Box::new(Self::new(vec![1.0; self.data.len()]))
    }

    fn zeros_like(&self) -> Box<dyn AutogradTensor<f32>> {
        Box::new(Self::new(vec![0.0; self.data.len()]))
    }

    fn with_data(&self, data: Vec<f32>) -> Result<Box<dyn AutogradTensor<f32>>> {
        Ok(Box::new(Self::new(data)))
    }
}

fn l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|v| v * v).sum::<f32>().sqrt()
}

// ---------------------------------------------------------------------------
// F000: clip_grad_norm must actually scale the gradients.
// ---------------------------------------------------------------------------

#[test]
fn f000_clip_grad_norm_scales_gradients_down_to_max_norm() -> Result<()> {
    // [6, 8] has L2 norm exactly 10.
    let grad = GradBuffer::new(vec![6.0, 8.0]);
    let gradients: Vec<&dyn AutogradTensor<f32>> = vec![&grad];

    let (total_norm, clipped) = clip::clip_grad_norm(&gradients, 1.0, 2.0)?;

    assert!(
        (total_norm - 10.0).abs() < 1e-5,
        "pre-clip norm must be reported, got {total_norm}"
    );
    assert_eq!(clipped.len(), 1);

    let clipped_norm = l2_norm(&clipped[0].to_vec());
    assert!(
        (clipped_norm - 1.0).abs() < 1e-3,
        "clipped norm must be max_norm, got {clipped_norm}"
    );
    Ok(())
}

#[test]
fn f000_clip_grad_norm_leaves_small_gradients_untouched() -> Result<()> {
    let grad = GradBuffer::new(vec![0.3, 0.4]); // norm 0.5
    let gradients: Vec<&dyn AutogradTensor<f32>> = vec![&grad];

    let (total_norm, clipped) = clip::clip_grad_norm(&gradients, 1.0, 2.0)?;

    assert!((total_norm - 0.5).abs() < 1e-6);
    assert_eq!(clipped[0].to_vec(), vec![0.3, 0.4]);
    Ok(())
}

#[test]
fn f000_clip_grad_norm_is_global_across_tensors() -> Result<()> {
    let first = GradBuffer::new(vec![3.0, 0.0]);
    let second = GradBuffer::new(vec![0.0, 4.0]);
    let gradients: Vec<&dyn AutogradTensor<f32>> = vec![&first, &second];

    let (total_norm, clipped) = clip::clip_grad_norm(&gradients, 2.5, 2.0)?;

    assert!((total_norm - 5.0).abs() < 1e-5);
    let joint = l2_norm(
        &clipped
            .iter()
            .flat_map(|g| g.to_vec())
            .collect::<Vec<f32>>(),
    );
    assert!(
        (joint - 2.5).abs() < 1e-3,
        "joint norm across tensors must be clipped, got {joint}"
    );
    Ok(())
}

#[test]
fn f000_clip_grad_value_tensors_clamps_elementwise() -> Result<()> {
    let grad = GradBuffer::new(vec![-5.0, 0.25, 5.0]);
    let gradients: Vec<&dyn AutogradTensor<f32>> = vec![&grad];

    let clipped = clip::clip_grad_value_tensors(&gradients, -1.0, 1.0)?;

    assert_eq!(clipped[0].to_vec(), vec![-1.0, 0.25, 1.0]);
    Ok(())
}

// ---------------------------------------------------------------------------
// F083: no_grad()/enable_grad() must suppress graph recording.
// ---------------------------------------------------------------------------

#[test]
fn f083_no_grad_suppresses_recording_for_binary_ops() -> Result<()> {
    let _serialized = grad_mode_guard();

    let a = Tensor::from_data(vec![1.0f32, 2.0], vec![2], DeviceType::Cpu)?.requires_grad_(true);
    let b = Tensor::from_data(vec![2.0f32, 3.0], vec![2], DeviceType::Cpu)?;

    {
        let _guard = torsh_autograd::guards::no_grad();
        assert!(!a.mul(&b)?.requires_grad(), "mul must not record");
        assert!(!a.add(&b)?.requires_grad(), "add must not record");
        assert!(!a.sub(&b)?.requires_grad(), "sub must not record");
        assert!(!a.div(&b)?.requires_grad(), "div must not record");
    }

    // Outside the guard the graph is built again.
    assert!(
        a.mul(&b)?.requires_grad(),
        "mul must record outside no_grad"
    );
    Ok(())
}

#[test]
fn f083_no_grad_suppresses_recording_for_unary_and_reduction_ops() -> Result<()> {
    let _serialized = grad_mode_guard();

    let a =
        Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu)?.requires_grad_(true);

    {
        let _guard = torsh_autograd::guards::no_grad();
        assert!(!a.sum()?.requires_grad(), "sum must not record");
        assert!(
            !a.mean(None, false)?.requires_grad(),
            "mean must not record"
        );
        assert!(
            !a.mul_scalar(2.0)?.requires_grad(),
            "mul_scalar must not record"
        );
        assert!(
            !a.reshape(&[3, 1])?.requires_grad(),
            "reshape must not record"
        );
    }
    Ok(())
}

#[test]
fn f083_no_grad_result_is_a_leaf_and_cannot_backward() -> Result<()> {
    let _serialized = grad_mode_guard();

    let a = Tensor::from_data(vec![1.0f32, 2.0], vec![2], DeviceType::Cpu)?.requires_grad_(true);
    let b = Tensor::from_data(vec![2.0f32, 3.0], vec![2], DeviceType::Cpu)?;

    let detached_sum = {
        let _guard = torsh_autograd::guards::no_grad();
        a.mul(&b)?.sum()?
    };

    assert!(!detached_sum.requires_grad());
    assert!(
        detached_sum.backward().is_err(),
        "a tensor built under no_grad has no graph to back-propagate"
    );
    Ok(())
}

#[test]
fn f083_enable_grad_restores_recording_inside_no_grad() -> Result<()> {
    let _serialized = grad_mode_guard();

    let a = Tensor::from_data(vec![1.0f32, 2.0], vec![2], DeviceType::Cpu)?.requires_grad_(true);
    let b = Tensor::from_data(vec![2.0f32, 3.0], vec![2], DeviceType::Cpu)?;

    let _outer = torsh_autograd::guards::no_grad();
    assert!(!a.mul(&b)?.requires_grad());
    {
        let _inner = torsh_autograd::guards::enable_grad();
        assert!(a.mul(&b)?.requires_grad());
    }
    assert!(!a.mul(&b)?.requires_grad());
    Ok(())
}

#[test]
fn f083_gradients_still_flow_when_grad_mode_is_enabled() -> Result<()> {
    let _serialized = grad_mode_guard();

    let a = Tensor::from_data(vec![3.0f32], vec![1], DeviceType::Cpu)?.requires_grad_(true);
    let b = Tensor::from_data(vec![4.0f32], vec![1], DeviceType::Cpu)?;

    let product = a.mul(&b)?.sum()?;
    product.backward()?;

    let grad = a.grad().expect("gradient must exist outside no_grad");
    assert_eq!(grad.to_vec()?, vec![4.0]);
    Ok(())
}

// ---------------------------------------------------------------------------
// F084: gradient checkpointing exists and recomputes correctly.
// ---------------------------------------------------------------------------

#[test]
fn f084_checkpoint_recomputes_gradients_matching_the_direct_graph() -> Result<()> {
    let _serialized = grad_mode_guard();
    use torsh_autograd::checkpoint::{checkpoint_with_seed, CheckpointRng};

    let x = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu)?;

    let segment = checkpoint_with_seed(
        |inputs: &[Tensor<f32>], _rng: &mut CheckpointRng| inputs[0].mul(&inputs[0]),
        &[x],
        7,
    )?;

    assert_eq!(segment.output().to_vec()?, vec![1.0, 4.0, 9.0]);
    assert!(
        !segment.output().requires_grad(),
        "checkpointed forward must not retain a graph"
    );

    let upstream = Tensor::from_data(vec![1.0f32, 1.0, 1.0], vec![3], DeviceType::Cpu)?;
    let grads = segment.backward(&upstream)?;

    // d(x^2)/dx = 2x
    assert_eq!(grads[0].to_vec()?, vec![2.0, 4.0, 6.0]);
    Ok(())
}

#[test]
fn f084_checkpoint_replays_stochastic_segments_identically() -> Result<()> {
    let _serialized = grad_mode_guard();
    use torsh_autograd::checkpoint::{checkpoint_with_seed, CheckpointRng};

    // A "dropout"-like segment: scales the input by a random factor.
    let segment = checkpoint_with_seed(
        |inputs: &[Tensor<f32>], rng: &mut CheckpointRng| {
            let scale: f32 = rng.random_range(0.1..0.9);
            inputs[0].mul_scalar(scale)
        },
        &[Tensor::from_data(vec![2.0f32], vec![1], DeviceType::Cpu)?],
        1234,
    )?;

    let forward_scale = segment.output().to_vec()?[0] / 2.0;

    let upstream = Tensor::from_data(vec![1.0f32], vec![1], DeviceType::Cpu)?;
    let grads = segment.backward(&upstream)?;

    // d(s*x)/dx = s, and the recompute must have drawn the same s.
    assert!(
        (grads[0].to_vec()?[0] - forward_scale).abs() < 1e-6,
        "recompute drew a different random factor: forward {forward_scale}, backward {:?}",
        grads[0].to_vec()?
    );
    Ok(())
}

#[test]
fn f084_checkpoint_sequential_backpropagates_through_every_segment() -> Result<()> {
    let _serialized = grad_mode_guard();
    use torsh_autograd::checkpoint::{checkpoint_sequential, CheckpointRng};

    // Four layers, each doubling: output = 16 * input, d(output)/d(input) = 16.
    let layers: Vec<_> = (0..4)
        .map(|_| {
            |input: &Tensor<f32>, _rng: &mut CheckpointRng| -> Result<Tensor<f32>> {
                input.add(input)
            }
        })
        .collect();

    let input = Tensor::from_data(vec![1.0f32], vec![1], DeviceType::Cpu)?;
    let sequence = checkpoint_sequential(layers, 2, &input)?;

    assert_eq!(sequence.len(), 2, "four layers must land in two segments");
    assert_eq!(sequence.output()?.to_vec()?, vec![16.0]);

    let upstream = Tensor::from_data(vec![1.0f32], vec![1], DeviceType::Cpu)?;
    let gradient = sequence.backward(&upstream)?;
    assert_eq!(gradient.to_vec()?, vec![16.0]);
    Ok(())
}

// ---------------------------------------------------------------------------
// F081: second-order hyperparameter gradients must move the hyperparameters.
// ---------------------------------------------------------------------------

#[test]
fn f081_second_order_optimization_moves_hyperparameters_towards_the_optimum() -> Result<()> {
    use torsh_autograd::hyperparameter_optimization::{
        HyperparameterConfig, HyperparameterOptimizer, OptimizableHyperparameter,
    };

    // Objective: (x - 5)^2, minimised at x = 5, starting from x = 0.
    let objective = |values: &HashMap<String, f64>| -> Result<Tensor> {
        let x = values.get("x").copied().unwrap_or_default();
        Tensor::scalar(((x - 5.0) * (x - 5.0)) as f32)
    };

    let config = HyperparameterConfig {
        second_order: true,
        meta_learning_rate: 0.5,
        ..HyperparameterConfig::default()
    };
    let mut optimizer = HyperparameterOptimizer::new(config);
    optimizer.add_hyperparameter(OptimizableHyperparameter::new(
        "x".to_string(),
        0.0,
        None,
        None,
        false,
    )?);

    let start = optimizer.get_hyperparameter("x")?;
    for _ in 0..8 {
        optimizer.step(&objective)?;
    }
    let end = optimizer.get_hyperparameter("x")?;

    assert!(
        (end - start).abs() > 1e-3,
        "second_order=true must not leave hyperparameters unchanged (start {start}, end {end})"
    );
    assert!(
        (end - 5.0).abs() < (start - 5.0).abs(),
        "second-order steps must approach the optimum (start {start}, end {end})"
    );
    Ok(())
}

#[test]
fn f081_second_order_newton_step_beats_a_single_first_order_step() -> Result<()> {
    use torsh_autograd::hyperparameter_optimization::{
        HyperparameterConfig, HyperparameterOptimizer, OptimizableHyperparameter,
    };

    let objective = |values: &HashMap<String, f64>| -> Result<Tensor> {
        let x = values.get("x").copied().unwrap_or_default();
        Tensor::scalar(((x - 5.0) * (x - 5.0)) as f32)
    };

    let run = |second_order: bool| -> Result<f64> {
        let config = HyperparameterConfig {
            second_order,
            // A full Newton step: x <- x - g/c lands exactly on the optimum of
            // a quadratic, while the same rate on the raw gradient does not.
            meta_learning_rate: 1.0,
            ..HyperparameterConfig::default()
        };
        let mut optimizer = HyperparameterOptimizer::new(config);
        optimizer.add_hyperparameter(OptimizableHyperparameter::new(
            "x".to_string(),
            0.0,
            None,
            None,
            false,
        )?);
        optimizer.step(&objective)?;
        optimizer.get_hyperparameter("x")
    };

    let newton = run(true)?;
    let gradient = run(false)?;

    assert!(
        (newton - 5.0).abs() < 5e-2,
        "one Newton step on a quadratic should reach the optimum, got {newton}"
    );
    assert!(
        (newton - 5.0).abs() < (gradient - 5.0).abs(),
        "second order must beat first order here (newton {newton}, gradient {gradient})"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// F001 / F082: activation paths must compute real values and real gradients.
// ---------------------------------------------------------------------------

#[test]
fn f001_gpu_activation_is_not_the_identity() -> Result<()> {
    use torsh_autograd::gpu_gradient::{ActivationType, GpuBackend, GpuGradientComputer};

    let mut computer =
        GpuGradientComputer::new(GpuBackend::Auto).map_err(torsh_core::error::TorshError::from)?;
    let input = [-2.0f32, -0.5, 0.0, 0.5, 2.0];

    let relu = computer
        .gpu_activation(&input, ActivationType::ReLU)
        .map_err(torsh_core::error::TorshError::from)?;
    assert_eq!(relu, vec![0.0, 0.0, 0.0, 0.5, 2.0]);

    let sigmoid = computer
        .gpu_activation(&input, ActivationType::Sigmoid)
        .map_err(torsh_core::error::TorshError::from)?;
    assert!((sigmoid[2] - 0.5).abs() < 1e-6, "sigmoid(0) = 0.5");
    assert_ne!(sigmoid, input.to_vec(), "sigmoid must not be the identity");

    for activation in [
        ActivationType::GELU,
        ActivationType::LeakyReLU,
        ActivationType::Swish,
        ActivationType::Tanh,
    ] {
        let out = computer
            .gpu_activation(&input, activation)
            .map_err(torsh_core::error::TorshError::from)?;
        assert_ne!(
            out,
            input.to_vec(),
            "{activation:?} must not return its input verbatim"
        );
    }
    Ok(())
}

#[test]
fn f001_activation_backward_matches_finite_differences() -> Result<()> {
    use torsh_autograd::gpu_gradient::{ActivationType, GpuBackend, GpuGradientComputer};

    let mut computer =
        GpuGradientComputer::new(GpuBackend::Auto).map_err(torsh_core::error::TorshError::from)?;
    let input = [-1.3f64, -0.2, 0.4, 1.7];
    let ones = [1.0f64; 4];
    let step = 1e-6;

    for activation in [
        ActivationType::GELU,
        ActivationType::Swish,
        ActivationType::Tanh,
        ActivationType::Sigmoid,
        ActivationType::LeakyReLU,
    ] {
        let analytic = computer
            .gpu_activation_backward(&input, &ones, activation)
            .map_err(torsh_core::error::TorshError::from)?;

        for (index, &x) in input.iter().enumerate() {
            let plus = computer
                .gpu_activation(&[x + step], activation)
                .map_err(torsh_core::error::TorshError::from)?[0];
            let minus = computer
                .gpu_activation(&[x - step], activation)
                .map_err(torsh_core::error::TorshError::from)?[0];
            let numeric = (plus - minus) / (2.0 * step);
            assert!(
                (analytic[index] - numeric).abs() < 1e-4,
                "{activation:?} gradient at {x}: analytic {} vs numeric {numeric}",
                analytic[index]
            );
        }
    }
    Ok(())
}

#[test]
fn f082_stats_never_report_kernel_launches_or_transfers_that_did_not_happen() -> Result<()> {
    use torsh_autograd::gpu_gradient::{ActivationType, GpuBackend, GpuGradientComputer};

    let mut computer =
        GpuGradientComputer::new(GpuBackend::CUDA).map_err(torsh_core::error::TorshError::from)?;
    assert!(!computer.is_available(), "no GPU transport exists here");

    let big = vec![1.0f32; 100_000];
    let _ = computer
        .gpu_activation(&big, ActivationType::ReLU)
        .map_err(torsh_core::error::TorshError::from)?;

    let stats = computer.stats();
    assert_eq!(stats.total_ops, 1, "the operation itself is counted");
    assert_eq!(stats.kernel_launches, 0, "no kernel was launched");
    assert_eq!(stats.memory_transferred, 0, "nothing was transferred");
    Ok(())
}

// ---------------------------------------------------------------------------
// F185: see the report — the distributed accumulator lives in an orphan file
// (`src/distributed/accumulator.rs`) that no `mod` declaration includes, so it
// is not part of the compiled crate and cannot be exercised from a test.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// F184: QuantumCircuit::add_gate rejects bad input instead of aborting.
// ---------------------------------------------------------------------------

#[test]
fn f184_add_gate_rejects_out_of_range_qubits() {
    use torsh_autograd::quantum_autograd::{PauliX, QuantumCircuit, Qubit};

    let mut circuit = QuantumCircuit::new(2);

    assert!(circuit
        .add_gate(Box::new(PauliX::new(Qubit::new(1))))
        .is_ok());

    let rejected = circuit.add_gate(Box::new(PauliX::new(Qubit::new(5))));
    let error = rejected.expect_err("qubit 5 is outside a 2-qubit circuit");
    let message = error.to_string();
    assert!(
        message.contains('5') && message.contains('2'),
        "the error must name the offending index and the circuit size: {message}"
    );
}
