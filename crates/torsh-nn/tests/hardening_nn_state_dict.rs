//! Production-hardening regression tests for buffer-aware `Module::state_dict`
//! / `Module::load_state_dict` (`core/mod.rs`) and for the container overrides
//! that make the recursion reach a child's buffers (`container/basic.rs`).
//! Campaign item W6-M.
//!
//! # The finding
//!
//! `Module::state_dict()` was built from `all_named_parameters()` alone, so it
//! carried *only* trainable parameters. Measured on the pre-W6-M tree, against
//! the plainest possible module and the plainest possible save/load path:
//!
//! | call on a bare `BatchNorm1d::new(3)` | result                                              |
//! |--------------------------------------|-----------------------------------------------------|
//! | `named_buffers()` keys               | `["num_batches_tracked", "running_mean", "running_var"]` |
//! | `state_dict()` keys                  | `["bias", "weight"]`                                |
//!
//! `running_mean` / `running_var` are the *entire* difference between a trained
//! normalization layer and a freshly constructed one: evaluation mode consumes
//! them instead of the batch statistics. A model saved through `state_dict()`
//! and reloaded through `load_state_dict()` therefore came back with
//! initialization statistics (mean 0, variance 1) and produced different
//! eval-mode outputs than the model that was saved — silently, with no error
//! and no missing-key report, because strict mode only ever compared parameter
//! names.
//!
//! Two independent blocks had to fall for this to work:
//!
//! 1. the trait had no `all_named_buffers()` to pair with
//!    `all_named_parameters()`, so there was nothing for `state_dict()` to
//!    recurse over even in principle; and
//! 2. `Sequential` / `ModuleList` / `ModuleDict` override `parameters()` and
//!    `children()` but overrode neither `named_buffers()` nor
//!    `named_children()`, so a buffer walk rooted at a container found nothing
//!    — the containers were invisible to any name-carrying recursion.
//!
//! # The decision pinned here
//!
//! PyTorch semantics: `state_dict()` carries parameters **and** buffers,
//! `load_state_dict()` restores both, and strict-mode key checking counts
//! both. No `num_batches_tracked` leniency is granted — PyTorch's legacy
//! exemption for that one key is deliberately *not* reproduced, so a missing
//! buffer key is an error like any other.
//!
//! # What is pinned here
//!
//! * a bare `BatchNorm1d`'s `state_dict()` contains its three buffers;
//! * a save -> mutate -> load round trip restores the statistics bit-for-bit
//!   *and* returns the eval-mode forward output to bit-identical values, with
//!   an explicit non-vacuity check that mutating the layer did not also mutate
//!   the saved snapshot (`Tensor` is `Clone` over `Arc`-shared storage, so that
//!   aliasing hazard is real);
//! * `Sequential` / `ModuleList` / `ModuleDict` publish child buffers under the
//!   same key scheme their parameters already use;
//! * strict mode reports a missing buffer key, an unexpected buffer key, and a
//!   buffer shape mismatch;
//! * the **parameter** key set of a container is byte-for-byte what it was
//!   before this change — buffers were added, parameters were not disturbed.

use std::collections::HashMap;

use torsh_core::error::Result;
use torsh_nn::container::{ModuleDict, ModuleList, Sequential};
use torsh_nn::layers::normalization::BatchNorm1d;
use torsh_nn::layers::Linear;
use torsh_nn::Module;
use torsh_tensor::Tensor;

/// Serializes every test in this file against the process-global grad-mode
/// `AtomicBool` (`torsh-core/src/grad_mode.rs`). Under plain `cargo test` a
/// whole binary's tests share one process across a thread pool, so a `no_grad`
/// window opened by any test transiently suppresses recording for every other
/// test's tensor ops. Every test below runs `BatchNorm` training forwards and
/// reads the resulting buffers, so each takes the lock for its full body.
/// Pattern copied from `torsh-tensor/tests/hardening_autograd_unary.rs:19-28`.
static GRAD_MODE_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Takes [`GRAD_MODE_GUARD`], recovering the guard if a previous test panicked
/// while holding it, so one genuine failure does not turn every later test in
/// the file into a `PoisonError` report.
fn serialize() -> std::sync::MutexGuard<'static, ()> {
    GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner())
}

fn sorted_keys<V>(map: &HashMap<String, V>) -> Vec<String> {
    let mut keys: Vec<String> = map.keys().cloned().collect();
    keys.sort();
    keys
}

/// `[2, 3]` input, distinct per column so the per-channel statistics separate.
fn batch_a() -> Result<Tensor> {
    Tensor::from_vec(vec![1.0f32, 4.0, 9.0, 3.0, 6.0, 11.0], &[2, 3])
}

/// A batch whose channel means and variances are far from [`batch_a`]'s, so
/// folding it into the running statistics moves them unmistakably.
fn batch_b() -> Result<Tensor> {
    Tensor::from_vec(vec![-20.0f32, 30.0, -5.0, -24.0, 34.0, -9.0], &[2, 3])
}

/// A fixed probe input for eval-mode forwards, so two outputs differ only if
/// the running statistics differ.
fn probe() -> Result<Tensor> {
    Tensor::from_vec(vec![0.5f32, 2.0, 7.0, 1.5, 5.0, 8.0], &[2, 3])
}

/// Runs `count` training forwards, folding `input` into the running statistics.
fn train_on(layer: &mut BatchNorm1d, input: &Tensor, count: usize) -> Result<()> {
    layer.train();
    for _ in 0..count {
        let _ = layer.forward(input)?;
    }
    Ok(())
}

/// Eval-mode forward, flattened. Eval mode consumes the running statistics
/// rather than the batch statistics, so this is the observable that a lost
/// checkpoint corrupts.
fn eval_forward(layer: &mut BatchNorm1d, input: &Tensor) -> Result<Vec<f32>> {
    layer.eval();
    layer.forward(input)?.to_vec()
}

fn values(state: &HashMap<String, Tensor>, key: &str) -> Result<Vec<f32>> {
    state
        .get(key)
        .ok_or_else(|| torsh_core::error::TorshError::Other(format!("missing key {key}")))?
        .to_vec()
}

// ---------------------------------------------------------------------------
// (a) A bare leaf module must not drop its buffers
// ---------------------------------------------------------------------------

/// The headline defect, at the smallest possible scale: no container, no box,
/// no recursion — one `BatchNorm1d` whose `named_buffers()` reports three
/// entries while `state_dict()` reported two parameters and nothing else.
#[test]
fn bare_batch_norm_state_dict_carries_its_running_statistics() {
    let _guard = serialize();

    let layer = BatchNorm1d::new(3).expect("BatchNorm1d::new");

    assert_eq!(
        sorted_keys(&layer.named_buffers()),
        vec!["num_batches_tracked", "running_mean", "running_var"],
        "precondition: the layer does publish three buffers"
    );

    assert_eq!(
        sorted_keys(&layer.state_dict()),
        vec![
            "bias",
            "num_batches_tracked",
            "running_mean",
            "running_var",
            "weight"
        ],
        "state_dict must carry buffers alongside parameters (PyTorch parity); \
         a checkpoint without running_mean/running_var restores a *different* model"
    );
}

/// `all_named_buffers()` is the buffer-side twin of `all_named_parameters()`:
/// on a leaf it degenerates to `named_buffers()`.
#[test]
fn all_named_buffers_on_a_leaf_equals_its_named_buffers() {
    let _guard = serialize();

    let layer = BatchNorm1d::new(3).expect("BatchNorm1d::new");
    assert_eq!(
        sorted_keys(&layer.all_named_buffers()),
        sorted_keys(&layer.named_buffers())
    );
    assert_eq!(layer.all_named_buffers().len(), 3);
}

// ---------------------------------------------------------------------------
// (b) The round trip that the defect broke
// ---------------------------------------------------------------------------

/// Save -> mutate -> load, checked three ways: the buffer values come back
/// bit-for-bit, the eval-mode forward output comes back bit-for-bit, and the
/// saved snapshot is proven not to have been mutated along with the layer.
///
/// That last check is not ceremony. `Tensor` derives `Clone` over an
/// `Arc`-shared `TensorStorage`, so a snapshot taken by cloning could in
/// principle alias the live buffer and make the whole round trip vacuously
/// "succeed". It holds today only because `RunningStats::update` *replaces* the
/// tensor in each slot rather than writing through it; this assertion is what
/// notices if that ever changes.
#[test]
fn state_dict_round_trip_restores_running_statistics_and_eval_output() {
    let _guard = serialize();

    let mut layer = BatchNorm1d::new(3).expect("BatchNorm1d::new");
    let batch_a = batch_a().expect("batch a");
    let batch_b = batch_b().expect("batch b");
    let probe = probe().expect("probe");

    // Train the layer into a state distinguishable from its initialization.
    train_on(&mut layer, &batch_a, 3).expect("train on batch a");
    let trained_output = eval_forward(&mut layer, &probe).expect("eval forward, trained");

    // Checkpoint.
    let saved = layer.state_dict();
    let saved_mean = values(&saved, "running_mean").expect("saved running_mean");
    let saved_var = values(&saved, "running_var").expect("saved running_var");
    let saved_count = values(&saved, "num_batches_tracked").expect("saved num_batches_tracked");
    assert_eq!(
        saved_count,
        vec![3.0],
        "three training forwards were folded in"
    );

    // Move the layer somewhere else entirely.
    train_on(&mut layer, &batch_b, 5).expect("train on batch b");
    let drifted_output = eval_forward(&mut layer, &probe).expect("eval forward, drifted");
    assert_ne!(
        drifted_output, trained_output,
        "precondition: the extra training must actually change the eval output, \
         otherwise the restore below would be untestable"
    );

    // Non-vacuity: the checkpoint must be a snapshot, not a live alias.
    assert_eq!(
        values(&saved, "running_mean").expect("saved running_mean, re-read"),
        saved_mean,
        "the saved state_dict entry must be independent of the live buffer; \
         if this fails the round trip below proves nothing"
    );

    // Restore.
    layer
        .load_state_dict(&saved, true)
        .expect("strict load of a self-produced state_dict must succeed");

    let restored = layer.state_dict();
    assert_eq!(
        values(&restored, "running_mean").expect("restored running_mean"),
        saved_mean,
        "running_mean must be restored bit-for-bit"
    );
    assert_eq!(
        values(&restored, "running_var").expect("restored running_var"),
        saved_var,
        "running_var must be restored bit-for-bit"
    );
    assert_eq!(
        values(&restored, "num_batches_tracked").expect("restored num_batches_tracked"),
        saved_count,
        "num_batches_tracked must be restored bit-for-bit"
    );

    let restored_output = eval_forward(&mut layer, &probe).expect("eval forward, restored");
    assert_eq!(
        restored_output, trained_output,
        "a reloaded model must produce bit-identical eval-mode outputs to the \
         model that was saved"
    );
}

/// The restore must reach the layer's *live* handles, not a detached copy:
/// the layer's own accessors have to agree with the reloaded checkpoint.
#[test]
fn load_state_dict_writes_through_to_the_live_buffer_handles() {
    let _guard = serialize();

    let mut layer = BatchNorm1d::new(3).expect("BatchNorm1d::new");
    train_on(&mut layer, &batch_a().expect("batch a"), 2).expect("train");
    let saved = layer.state_dict();
    let saved_mean = values(&saved, "running_mean").expect("saved running_mean");

    train_on(&mut layer, &batch_b().expect("batch b"), 4).expect("train");
    layer.load_state_dict(&saved, true).expect("strict load");

    let live_mean = layer
        .running_mean()
        .expect("the layer tracks running statistics")
        .to_vec()
        .expect("running_mean to_vec");
    assert_eq!(
        live_mean, saved_mean,
        "load_state_dict must write through the shared buffer handle, so the \
         layer's own accessor sees the restored value"
    );
    assert_eq!(
        layer
            .num_batches_tracked()
            .expect("num_batches_tracked")
            .expect("statistics are tracked"),
        2.0,
        "the batch counter is state like any other and must be restored"
    );
}

// ---------------------------------------------------------------------------
// (c) Containers
// ---------------------------------------------------------------------------

/// `Sequential(Linear, BatchNorm1d, Linear)` — the buffer keys must be indexed
/// exactly the way the parameter keys already are.
#[test]
fn sequential_state_dict_carries_child_buffers_with_index_prefixes() {
    let _guard = serialize();

    let model = Sequential::new()
        .add(Linear::new(4, 3, true))
        .add(BatchNorm1d::new(3).expect("BatchNorm1d::new"))
        .add(Linear::new(3, 2, true));

    assert_eq!(
        sorted_keys(&model.named_buffers()),
        vec!["1.num_batches_tracked", "1.running_mean", "1.running_var"],
        "a container must publish its children's buffers under `{{index}}.{{name}}`, \
         mirroring its own named_parameters() key scheme"
    );

    assert_eq!(
        sorted_keys(&model.state_dict()),
        vec![
            "0.bias",
            "0.weight",
            "1.bias",
            "1.num_batches_tracked",
            "1.running_mean",
            "1.running_var",
            "1.weight",
            "2.bias",
            "2.weight",
        ],
        "the whole checkpoint: six parameters from the three layers, two affine \
         parameters and three buffers from the normalization layer"
    );

    assert_eq!(
        model.buffers().len(),
        3,
        "the unnamed accessor must agree with the named one"
    );
}

/// Guard against collateral damage: making the walk buffer-aware must not add,
/// drop or rename a single *parameter* key. These are the exact keys measured
/// on the pre-W6-M tree.
#[test]
fn container_parameter_key_sets_are_unchanged_by_buffer_awareness() {
    let _guard = serialize();

    let sequential = Sequential::new()
        .add(Linear::new(4, 3, true))
        .add(BatchNorm1d::new(3).expect("BatchNorm1d::new"))
        .add(Linear::new(3, 2, true));
    let expected = vec![
        "0.bias", "0.weight", "1.bias", "1.weight", "2.bias", "2.weight",
    ];
    assert_eq!(sorted_keys(&sequential.named_parameters()), expected);
    assert_eq!(
        sorted_keys(&sequential.all_named_parameters()),
        expected,
        "adding named_children() must not make the parameter recursion produce \
         duplicate or extra keys"
    );
    assert_eq!(
        sequential.num_parameters(),
        37,
        "4*3+3 (linear) + 3+3 (affine) + 3*2+2 (linear) = 37, as measured before \
         this change"
    );

    let mut list = ModuleList::new();
    list.push(Linear::new(4, 3, true));
    list.push(BatchNorm1d::new(3).expect("BatchNorm1d::new"));
    assert_eq!(
        sorted_keys(&list.all_named_parameters()),
        vec!["0.bias", "0.weight", "1.bias", "1.weight"]
    );

    let mut dict = ModuleDict::new();
    dict.insert("lin".to_string(), Linear::new(4, 3, true));
    dict.insert(
        "bn".to_string(),
        BatchNorm1d::new(3).expect("BatchNorm1d::new"),
    );
    assert_eq!(
        sorted_keys(&dict.all_named_parameters()),
        vec!["bn.bias", "bn.weight", "lin.bias", "lin.weight"]
    );
}

#[test]
fn module_list_and_module_dict_state_dicts_carry_buffers() {
    let _guard = serialize();

    let mut list = ModuleList::new();
    list.push(Linear::new(4, 3, true));
    list.push(BatchNorm1d::new(3).expect("BatchNorm1d::new"));
    assert_eq!(
        sorted_keys(&list.state_dict()),
        vec![
            "0.bias",
            "0.weight",
            "1.bias",
            "1.num_batches_tracked",
            "1.running_mean",
            "1.running_var",
            "1.weight",
        ]
    );

    let mut dict = ModuleDict::new();
    dict.insert("lin".to_string(), Linear::new(4, 3, true));
    dict.insert(
        "bn".to_string(),
        BatchNorm1d::new(3).expect("BatchNorm1d::new"),
    );
    assert_eq!(
        sorted_keys(&dict.state_dict()),
        vec![
            "bn.bias",
            "bn.num_batches_tracked",
            "bn.running_mean",
            "bn.running_var",
            "bn.weight",
            "lin.bias",
            "lin.weight",
        ],
        "ModuleDict keys its children by name, so its buffers key the same way"
    );
}

/// A container's children are now enumerable *by name*, which is what the whole
/// name-carrying recursion (`all_named_parameters`, `all_named_buffers`,
/// `named_modules`) is built on. Before this change every container reported
/// zero named children while reporting three unnamed ones.
#[test]
fn containers_enumerate_their_children_by_name() {
    let _guard = serialize();

    let model = Sequential::new()
        .add(Linear::new(4, 3, true))
        .add(BatchNorm1d::new(3).expect("BatchNorm1d::new"));
    let named: Vec<String> = model
        .named_children()
        .into_iter()
        .map(|(name, _)| name)
        .collect();
    assert_eq!(
        named,
        vec!["0", "1"],
        "named_children must agree in count and order with children()"
    );
    assert_eq!(model.named_children().len(), model.children().len());

    let mut list = ModuleList::new();
    list.push(Linear::new(4, 3, true));
    let list_named: Vec<String> = list
        .named_children()
        .into_iter()
        .map(|(name, _)| name)
        .collect();
    assert_eq!(list_named, vec!["0"]);

    let mut dict = ModuleDict::new();
    dict.insert("head".to_string(), Linear::new(4, 3, true));
    let dict_named: Vec<String> = dict
        .named_children()
        .into_iter()
        .map(|(name, _)| name)
        .collect();
    assert_eq!(dict_named, vec!["head"]);
}

/// The end-to-end story for a container: a trained `Sequential` saved and
/// reloaded must evaluate identically.
#[test]
fn sequential_round_trip_preserves_eval_output() {
    let _guard = serialize();

    let mut model = Sequential::new()
        .add(BatchNorm1d::new(3).expect("BatchNorm1d::new"))
        .add(Linear::new(3, 2, true));

    let batch_a = batch_a().expect("batch a");
    let batch_b = batch_b().expect("batch b");
    let probe = probe().expect("probe");

    model.train();
    for _ in 0..3 {
        let _ = model.forward(&batch_a).expect("train forward");
    }
    model.eval();
    let trained = model.forward(&probe).expect("eval forward").to_vec();
    let trained = trained.expect("to_vec");

    let saved = model.state_dict();

    model.train();
    for _ in 0..5 {
        let _ = model.forward(&batch_b).expect("train forward");
    }
    model.eval();
    let drifted = model
        .forward(&probe)
        .expect("eval forward")
        .to_vec()
        .expect("to_vec");
    assert_ne!(drifted, trained, "precondition: the drift must be visible");

    model
        .load_state_dict(&saved, true)
        .expect("strict load through the container");
    model.eval();
    let restored = model
        .forward(&probe)
        .expect("eval forward")
        .to_vec()
        .expect("to_vec");
    assert_eq!(
        restored, trained,
        "a Sequential reloaded from its own state_dict must evaluate identically"
    );
}

// ---------------------------------------------------------------------------
// (d) Strict-mode key checking counts buffers
// ---------------------------------------------------------------------------

#[test]
fn strict_load_reports_a_missing_buffer_key() {
    let _guard = serialize();

    let mut layer = BatchNorm1d::new(3).expect("BatchNorm1d::new");
    let mut state = layer.state_dict();
    let _ = state.remove("running_mean");

    let err = layer
        .load_state_dict(&state, true)
        .expect_err("strict mode must reject a checkpoint that is missing a buffer");
    let message = err.to_string();
    assert!(
        message.contains("running_mean"),
        "the error must name the missing buffer, got: {message}"
    );
    assert!(
        message.contains("Missing keys"),
        "a missing buffer is reported as a missing key, not as something else: {message}"
    );
}

/// PyTorch grants `num_batches_tracked` a legacy exemption. This codebase
/// deliberately does not — the decision recorded for W6-M is "strict-mode key
/// checking counts both", with no per-key carve-outs.
#[test]
fn strict_load_reports_a_missing_num_batches_tracked_key_too() {
    let _guard = serialize();

    let mut layer = BatchNorm1d::new(3).expect("BatchNorm1d::new");
    let mut state = layer.state_dict();
    let _ = state.remove("num_batches_tracked");

    let err = layer
        .load_state_dict(&state, true)
        .expect_err("no per-key leniency: num_batches_tracked is state like any other");
    assert!(
        err.to_string().contains("num_batches_tracked"),
        "got: {err}"
    );
}

#[test]
fn strict_load_reports_an_unexpected_buffer_key() {
    let _guard = serialize();

    let mut layer = BatchNorm1d::new(3).expect("BatchNorm1d::new");
    let mut state = layer.state_dict();
    let _ = state.insert(
        "running_kurtosis".to_string(),
        Tensor::from_vec(vec![0.0f32; 3], &[3]).expect("tensor"),
    );

    let err = layer
        .load_state_dict(&state, true)
        .expect_err("strict mode must reject a key the module cannot consume");
    let message = err.to_string();
    assert!(message.contains("running_kurtosis"), "got: {message}");
    assert!(message.contains("Unexpected keys"), "got: {message}");
}

/// A wrong-shaped buffer is rejected the same way a wrong-shaped parameter is,
/// in strict *and* non-strict mode: a shape mismatch is a corrupt checkpoint,
/// not a tolerable key difference.
///
/// The rejection is also *atomic*. Shapes are validated across the whole
/// checkpoint before anything is written, so a bad entry cannot leave the model
/// half-loaded — a hybrid of two checkpoints is harder to notice than a clean
/// failure, and impossible to recover from.
#[test]
fn load_rejects_a_buffer_of_the_wrong_shape_without_writing_anything() {
    let _guard = serialize();

    let mut layer = BatchNorm1d::new(3).expect("BatchNorm1d::new");
    train_on(&mut layer, &batch_a().expect("batch a"), 2).expect("train");
    let before = layer.state_dict();
    let before_mean = values(&before, "running_mean").expect("running_mean");
    let before_weight = values(&before, "weight").expect("weight");

    // A checkpoint whose every *other* entry is different from the live layer,
    // so a partial write would be detectable.
    let mut state = layer.state_dict();
    let _ = state.insert(
        "running_var".to_string(),
        Tensor::from_vec(vec![7.0f32, 7.0, 7.0], &[3]).expect("tensor"),
    );
    let _ = state.insert(
        "weight".to_string(),
        Tensor::from_vec(vec![5.0f32, 5.0, 5.0], &[3]).expect("tensor"),
    );
    let _ = state.insert(
        "running_mean".to_string(),
        Tensor::from_vec(vec![0.0f32; 5], &[5]).expect("tensor"),
    );

    for strict in [true, false] {
        let err = layer
            .load_state_dict(&state, strict)
            .expect_err("a wrong-shaped buffer must never be written into the layer");
        let message = err.to_string();
        assert!(
            message.contains("running_mean") && message.contains("Shape mismatch"),
            "strict={strict}, got: {message}"
        );
    }

    let after = layer.state_dict();
    assert_eq!(
        values(&after, "running_mean").expect("running_mean"),
        before_mean,
        "the rejected checkpoint must not have been partially applied"
    );
    assert_eq!(
        values(&after, "weight").expect("weight"),
        before_weight,
        "a parameter that sorted before the bad buffer must not have been written either"
    );
    assert_ne!(
        before_weight,
        vec![5.0f32, 5.0, 5.0],
        "non-vacuous: the rejected checkpoint really did carry a different weight"
    );
}

/// Non-strict mode keeps its documented meaning: key differences are tolerated
/// and every key that *does* match is applied — buffers included.
#[test]
fn non_strict_load_applies_the_buffers_it_does_have() {
    let _guard = serialize();

    let mut layer = BatchNorm1d::new(3).expect("BatchNorm1d::new");
    train_on(&mut layer, &batch_a().expect("batch a"), 2).expect("train");
    let saved = layer.state_dict();
    let saved_var = values(&saved, "running_var").expect("saved running_var");

    let mut partial = saved.clone();
    let _ = partial.remove("running_mean");

    train_on(&mut layer, &batch_b().expect("batch b"), 4).expect("train");
    let drifted_mean = values(&layer.state_dict(), "running_mean").expect("drifted running_mean");

    layer
        .load_state_dict(&partial, false)
        .expect("non-strict load tolerates the missing key");

    let after = layer.state_dict();
    assert_eq!(
        values(&after, "running_var").expect("running_var"),
        saved_var,
        "the buffer that was present must be applied"
    );
    assert_eq!(
        values(&after, "running_mean").expect("running_mean"),
        drifted_mean,
        "the buffer that was absent must be left exactly as it was"
    );
}

// ---------------------------------------------------------------------------
// Boxing stays transparent
// ---------------------------------------------------------------------------

/// `Sequential` stores `Vec<Box<dyn Module>>` and every call it makes on a
/// child resolves through `impl Module for Box<dyn Module>`, so the new
/// recursion is only correct if the box forwards it.
#[test]
fn boxing_preserves_the_buffer_aware_state_dict() {
    let _guard = serialize();

    let bare = BatchNorm1d::new(3).expect("BatchNorm1d::new");
    let expected_state = sorted_keys(&bare.state_dict());
    let expected_buffers = sorted_keys(&bare.all_named_buffers());

    assert!(
        expected_state.contains(&"running_mean".to_string()),
        "non-vacuous: the checkpoint being compared must actually contain a buffer, \
         got {expected_state:?}"
    );

    let boxed: Box<dyn Module> = Box::new(BatchNorm1d::new(3).expect("BatchNorm1d::new"));
    assert_eq!(
        sorted_keys(&boxed.state_dict()),
        expected_state,
        "boxing a module must not change its checkpoint"
    );
    assert_eq!(
        sorted_keys(&boxed.all_named_buffers()),
        expected_buffers,
        "all_named_buffers must forward through the box like all_named_parameters does"
    );
    assert_eq!(
        expected_buffers,
        vec!["num_batches_tracked", "running_mean", "running_var"],
        "non-vacuous: the lists compared above are not both empty"
    );
}
