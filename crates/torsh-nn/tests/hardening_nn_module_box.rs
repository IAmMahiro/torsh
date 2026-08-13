//! Production-hardening regression tests for `impl Module for Box<dyn Module>`
//! and `impl Module for &mut Box<dyn Module>` (`core/mod.rs`), campaign item
//! W5-N(2).
//!
//! # The finding
//!
//! `Module` is a wide trait: exactly one method (`forward`) is required and
//! everything else has a default. Those defaults split into two families.
//!
//! * **Composed defaults** — `all_named_parameters`, `state_dict`,
//!   `num_parameters`, `residual_forward`, … — are written in terms of other
//!   trait methods. A blanket impl that forwards the primitives gets these for
//!   free.
//! * **Leaf defaults** — `buffers`, `named_buffers`, `name`, `zero_grad`,
//!   `freeze`, `unfreeze`, `extra_repr`, `register_hook`, `remove_hook`,
//!   `execute_hooks`, `has_hooks` — return "nothing" (`Vec::new()`,
//!   `HashMap::new()`, `None`, `false`, `Ok(())`, or an empty body). A blanket
//!   impl that *omits* one of these does not fall through to the inner module;
//!   it silently answers "nothing" on the inner module's behalf.
//!
//! The pre-W5-N `impl Module for Box<dyn Module>` forwarded nine methods
//! (`forward`, `parameters`, `train`, `eval`, `training`, `children`,
//! `named_children`, `set_training`, `to_device`) and omitted every leaf default
//! listed above. Measured on the pre-fix tree:
//!
//! | call on a boxed `BatchNorm1d`  | boxed  | direct |
//! |--------------------------------|--------|--------|
//! | `named_buffers().len()`        | `0`    | `3`    |
//! | `buffers().len()`              | `0`    | `3`    |
//!
//! That is not cosmetic. `Sequential` and `ModuleList` store their children as
//! `Vec<Box<dyn Module>>`, and every trait call they make on a child
//! (`module.train()`, `module.forward(..)`, `module.named_buffers()`) resolves
//! through this very impl. So a `BatchNorm`'s `running_mean` / `running_var` /
//! `num_batches_tracked` — the only state that separates a trained normalization
//! layer from a fresh one — became invisible the moment the layer was boxed:
//! any checkpoint writer, device migration or state walk that goes through the
//! box round-trips a model whose evaluation behaviour is reset to the
//! initialization values, with no error anywhere.
//!
//! # What is pinned here
//!
//! * the two calls in the table above, against a real `BatchNorm1d` whose
//!   statistics have been moved away from their initial values;
//! * every remaining leaf default, against a `SpyModule` that counts the calls
//!   that reach it — the only way to distinguish "forwarded and the inner module
//!   answered nothing" from "never forwarded";
//! * a checkpoint round-trip over `Vec<Box<dyn Module>>` (the exact storage and
//!   the exact call path `Sequential` uses) that must carry running statistics;
//! * the same coverage for `&mut Box<dyn Module>`, which had the identical gap.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;
use torsh_core::device::DeviceType;
use torsh_core::error::Result;
use torsh_nn::layers::normalization::BatchNorm1d;
use torsh_nn::{HookHandle, HookRegistry, HookType, Module, Parameter};
use torsh_tensor::Tensor;

/// Serializes every test in this file against the process-global grad-mode
/// `AtomicBool` (`torsh-core/src/grad_mode.rs`). Under plain `cargo test` a
/// whole binary's tests share one process across a thread pool, so a `no_grad`
/// window opened by any test transiently suppresses recording for every other
/// test's tensor ops. The `BatchNorm1d` cases below run a training forward and
/// read the resulting buffers, so they take the lock for their full body.
/// Pattern copied from `torsh-tensor/tests/hardening_autograd_unary.rs`.
static GRAD_MODE_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Takes [`GRAD_MODE_GUARD`], recovering the guard if a previous test panicked
/// while holding it, so one genuine failure does not turn every later test in
/// the file into a `PoisonError` report.
fn serialize() -> std::sync::MutexGuard<'static, ()> {
    GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner())
}

// ---------------------------------------------------------------------------
// Spy module
// ---------------------------------------------------------------------------

/// One counter per `Module` method whose default implementation is a *leaf*
/// (returns emptiness rather than delegating). A boxed call that never reaches
/// the inner module leaves its counter at zero while still producing a
/// plausible-looking answer, which is exactly the failure mode being pinned.
#[derive(Default)]
struct CallLog {
    parameters: AtomicUsize,
    named_parameters: AtomicUsize,
    buffers: AtomicUsize,
    named_buffers: AtomicUsize,
    zero_grad: AtomicUsize,
    freeze: AtomicUsize,
    unfreeze: AtomicUsize,
    extra_repr: AtomicUsize,
    name: AtomicUsize,
    register_hook: AtomicUsize,
    remove_hook: AtomicUsize,
    execute_hooks: AtomicUsize,
    has_hooks: AtomicUsize,
    state_dict: AtomicUsize,
    load_state_dict: AtomicUsize,
    to_device: AtomicUsize,
}

impl CallLog {
    fn bump(counter: &AtomicUsize) {
        let _ = counter.fetch_add(1, Ordering::SeqCst);
    }
}

/// A `Module` that answers every leaf default with something *distinguishable
/// from the default* and records that it was asked.
///
/// `parameters()` and `named_parameters()` deliberately disagree (empty vs. one
/// entry). The trait's default `named_parameters()` delegates to
/// `parameters()`, so an impl that forwards only `parameters()` reports zero
/// named parameters here — a direct probe for whether `named_parameters` itself
/// is forwarded, which no realistic layer can provide because realistic layers
/// define the two identically.
struct SpyModule {
    log: Arc<CallLog>,
    weight: Parameter,
    buffer: Arc<RwLock<Tensor>>,
    hooks: RwLock<HookRegistry>,
    child: Option<Box<SpyModule>>,
    training: bool,
    device: DeviceType,
}

impl SpyModule {
    fn new(log: Arc<CallLog>, child: Option<Box<SpyModule>>) -> Result<Self> {
        Ok(Self {
            log,
            weight: Parameter::new(Tensor::from_vec(vec![1.0f32, 2.0], &[2])?),
            buffer: Arc::new(RwLock::new(Tensor::from_vec(vec![3.0f32, 4.0], &[2])?)),
            hooks: RwLock::new(HookRegistry::new()),
            child,
            training: true,
            device: DeviceType::Cpu,
        })
    }
}

impl Module for SpyModule {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.mul_scalar(2.0)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        CallLog::bump(&self.log.parameters);
        HashMap::new()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        CallLog::bump(&self.log.named_parameters);
        let mut map = HashMap::new();
        let _ = map.insert("weight".to_string(), self.weight.clone());
        map
    }

    fn training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        CallLog::bump(&self.log.to_device);
        self.device = device;
        Ok(())
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        CallLog::bump(&self.log.state_dict);
        let mut map = HashMap::new();
        let _ = map.insert("spy".to_string(), self.buffer.read().clone());
        map
    }

    fn load_state_dict(
        &mut self,
        _state_dict: &HashMap<String, Tensor>,
        _strict: bool,
    ) -> Result<()> {
        CallLog::bump(&self.log.load_state_dict);
        Ok(())
    }

    fn name(&self) -> Option<&str> {
        CallLog::bump(&self.log.name);
        Some("SpyModule")
    }

    fn buffers(&self) -> Vec<Arc<RwLock<Tensor>>> {
        CallLog::bump(&self.log.buffers);
        vec![Arc::clone(&self.buffer)]
    }

    fn named_buffers(&self) -> HashMap<String, Arc<RwLock<Tensor>>> {
        CallLog::bump(&self.log.named_buffers);
        let mut map = HashMap::new();
        let _ = map.insert("spy_buffer".to_string(), Arc::clone(&self.buffer));
        map
    }

    fn children(&self) -> Vec<&dyn Module> {
        match &self.child {
            Some(child) => vec![&**child as &dyn Module],
            None => Vec::new(),
        }
    }

    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        match &self.child {
            Some(child) => vec![("child".to_string(), &**child as &dyn Module)],
            None => Vec::new(),
        }
    }

    fn zero_grad(&mut self) {
        CallLog::bump(&self.log.zero_grad);
    }

    fn freeze(&mut self) {
        CallLog::bump(&self.log.freeze);
    }

    fn unfreeze(&mut self) {
        CallLog::bump(&self.log.unfreeze);
    }

    fn extra_repr(&self) -> String {
        CallLog::bump(&self.log.extra_repr);
        "spy=1".to_string()
    }

    fn register_hook(
        &mut self,
        hook_type: HookType,
        callback: torsh_nn::HookCallback,
    ) -> Option<HookHandle> {
        CallLog::bump(&self.log.register_hook);
        Some(self.hooks.write().register_hook(hook_type, callback))
    }

    fn remove_hook(&mut self, hook_type: HookType, handle: HookHandle) -> bool {
        CallLog::bump(&self.log.remove_hook);
        self.hooks.write().remove_hook(hook_type, handle)
    }

    fn execute_hooks(
        &self,
        hook_type: HookType,
        input: &Tensor,
        output: Option<&Tensor>,
    ) -> Result<()> {
        CallLog::bump(&self.log.execute_hooks);
        self.hooks
            .read()
            .execute_hooks(hook_type, self, input, output)
    }

    fn has_hooks(&self, hook_type: HookType) -> bool {
        CallLog::bump(&self.log.has_hooks);
        self.hooks.read().has_hooks(hook_type)
    }
}

/// Builds a spy with one spy child and returns it boxed together with its log.
fn boxed_spy() -> Result<(Box<dyn Module>, Arc<CallLog>, Arc<CallLog>)> {
    let parent_log = Arc::new(CallLog::default());
    let child_log = Arc::new(CallLog::default());
    let child = SpyModule::new(Arc::clone(&child_log), None)?;
    let parent = SpyModule::new(Arc::clone(&parent_log), Some(Box::new(child)))?;
    Ok((Box::new(parent), parent_log, child_log))
}

// ---------------------------------------------------------------------------
// The named red: buffers on a boxed BatchNorm1d
// ---------------------------------------------------------------------------

/// Trains a `BatchNorm1d` for one step so its running statistics are provably
/// different from the values a freshly constructed layer would report, and
/// returns the layer.
fn trained_batch_norm() -> Result<BatchNorm1d> {
    let mut bn = BatchNorm1d::new(2)?;
    bn.train();
    let input = Tensor::from_vec(vec![1.0f32, 5.0, 3.0, 9.0], &[2, 2])?;
    let _ = bn.forward(&input)?;
    Ok(bn)
}

fn buffer_values(buffers: &HashMap<String, Arc<RwLock<Tensor>>>, key: &str) -> Vec<f32> {
    buffers
        .get(key)
        .unwrap_or_else(|| panic!("buffer `{key}` must be published"))
        .read()
        .to_vec()
        .expect("buffer values")
}

#[test]
fn boxed_batch_norm_publishes_the_same_named_buffers_as_the_bare_layer() {
    let _serial = serialize();

    let direct = trained_batch_norm().expect("BatchNorm1d");
    let direct_buffers = direct.named_buffers();
    let mut direct_keys: Vec<String> = direct_buffers.keys().cloned().collect();
    direct_keys.sort();
    assert_eq!(
        direct_keys,
        vec![
            "num_batches_tracked".to_string(),
            "running_mean".to_string(),
            "running_var".to_string()
        ],
        "control: a bare BatchNorm1d publishes its three running-statistic buffers"
    );

    let boxed: Box<dyn Module> = Box::new(direct);
    let boxed_buffers = boxed.named_buffers();
    let mut boxed_keys: Vec<String> = boxed_buffers.keys().cloned().collect();
    boxed_keys.sort();
    assert_eq!(
        boxed_keys, direct_keys,
        "boxing a module must not delete its buffers: `Sequential`/`ModuleList` \
         reach every child through exactly this impl"
    );

    // The handles must be the *same* `Arc`s, not copies -- a checkpoint loader
    // writes through them.
    let running_mean = buffer_values(&boxed_buffers, "running_mean");
    assert_eq!(
        running_mean.len(),
        2,
        "running_mean must have one entry per feature"
    );
    assert!(
        running_mean.iter().any(|v| v.abs() > 1e-6),
        "the fixture must have moved the statistics off their zero initialization, got {running_mean:?}"
    );
    {
        let mut guard = boxed_buffers
            .get("running_mean")
            .expect("running_mean handle")
            .write();
        *guard = Tensor::from_vec(vec![-1.0f32, -2.0], &[2]).expect("replacement");
    }
    assert_eq!(
        buffer_values(&boxed.named_buffers(), "running_mean"),
        vec![-1.0, -2.0],
        "the published handle must alias the layer's own buffer, not a snapshot"
    );
}

#[test]
fn boxed_batch_norm_publishes_the_same_buffer_vector_as_the_bare_layer() {
    let _serial = serialize();

    let direct = trained_batch_norm().expect("BatchNorm1d");
    let direct_len = direct.buffers().len();
    assert_eq!(direct_len, 3, "control: three running-statistic buffers");

    let boxed: Box<dyn Module> = Box::new(direct);
    assert_eq!(
        boxed.buffers().len(),
        direct_len,
        "`buffers()` must forward through the box"
    );
}

// ---------------------------------------------------------------------------
// Every remaining leaf default
// ---------------------------------------------------------------------------

#[test]
fn boxed_module_forwards_every_leaf_default_to_the_inner_module() {
    let (mut boxed, log, child_log) = boxed_spy().expect("spy");

    assert_eq!(
        boxed.name(),
        Some("SpyModule"),
        "`name()` must forward; the default answers `None`"
    );
    assert_eq!(
        log.name.load(Ordering::SeqCst),
        1,
        "`name()` must reach the inner module"
    );

    assert_eq!(
        boxed.extra_repr(),
        "spy=1",
        "`extra_repr()` must forward; the default answers an empty string"
    );

    assert_eq!(
        boxed.named_parameters().len(),
        1,
        "`named_parameters()` must forward, not fall back to the `parameters()`-delegating default"
    );
    assert_eq!(
        log.named_parameters.load(Ordering::SeqCst),
        1,
        "`named_parameters()` must reach the inner module"
    );

    assert_eq!(
        boxed.parameters().len(),
        0,
        "`parameters()` keeps forwarding (control)"
    );
    assert_eq!(log.parameters.load(Ordering::SeqCst), 1);

    assert_eq!(boxed.buffers().len(), 1, "`buffers()` must forward");
    assert_eq!(
        boxed.named_buffers().len(),
        1,
        "`named_buffers()` must forward"
    );
    assert_eq!(log.buffers.load(Ordering::SeqCst), 1);
    assert_eq!(log.named_buffers.load(Ordering::SeqCst), 1);

    assert_eq!(
        boxed.state_dict().len(),
        1,
        "`state_dict()` must forward, so a module with a custom checkpoint format keeps it"
    );
    assert_eq!(log.state_dict.load(Ordering::SeqCst), 1);

    boxed
        .load_state_dict(&HashMap::new(), false)
        .expect("load_state_dict");
    assert_eq!(
        log.load_state_dict.load(Ordering::SeqCst),
        1,
        "`load_state_dict()` must forward"
    );

    boxed.zero_grad();
    assert_eq!(
        log.zero_grad.load(Ordering::SeqCst),
        1,
        "`zero_grad()` must forward; the default is an empty body"
    );

    boxed.freeze();
    boxed.unfreeze();
    assert_eq!(
        log.freeze.load(Ordering::SeqCst),
        1,
        "`freeze()` must forward"
    );
    assert_eq!(
        log.unfreeze.load(Ordering::SeqCst),
        1,
        "`unfreeze()` must forward"
    );

    boxed
        .to_device(DeviceType::Cpu)
        .expect("to_device must succeed");
    assert_eq!(log.to_device.load(Ordering::SeqCst), 1);

    // The child is reached through `children()`, which already forwarded; its
    // log proves the recursion still terminates at the real leaf.
    assert_eq!(boxed.children().len(), 1);
    assert_eq!(boxed.named_children().len(), 1);
    assert_eq!(
        child_log.name.load(Ordering::SeqCst),
        0,
        "the child must not have been consulted yet"
    );
}

#[test]
fn boxed_module_forwards_the_whole_hook_protocol() {
    let (mut boxed, log, _child_log) = boxed_spy().expect("spy");

    assert!(
        !boxed.has_hooks(HookType::PreForward),
        "no hooks registered yet"
    );
    assert_eq!(
        log.has_hooks.load(Ordering::SeqCst),
        1,
        "`has_hooks()` must forward; the default answers `false` without asking"
    );

    let fired = Arc::new(AtomicUsize::new(0));
    let fired_in_callback = Arc::clone(&fired);
    let handle = boxed
        .register_hook(
            HookType::PreForward,
            Box::new(move |_module, _input, _output| {
                let _ = fired_in_callback.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }),
        )
        .expect("`register_hook()` must forward; the default answers `None`");
    assert_eq!(log.register_hook.load(Ordering::SeqCst), 1);

    assert!(
        boxed.has_hooks(HookType::PreForward),
        "the hook registered through the box must be visible through the box"
    );

    let input = Tensor::from_vec(vec![1.0f32, 2.0], &[2]).expect("input");
    let output = boxed
        .forward_with_hooks(&input)
        .expect("forward_with_hooks");
    assert_eq!(
        output.to_vec().expect("values"),
        vec![2.0, 4.0],
        "the forward itself is unchanged"
    );
    assert_eq!(
        fired.load(Ordering::SeqCst),
        1,
        "a hook registered on a boxed module must actually run: with `execute_hooks` \
         unforwarded, `forward_with_hooks` silently executes nothing"
    );
    assert!(log.execute_hooks.load(Ordering::SeqCst) >= 1);

    assert!(
        boxed.remove_hook(HookType::PreForward, handle),
        "`remove_hook()` must forward; the default answers `false`"
    );
    assert_eq!(log.remove_hook.load(Ordering::SeqCst), 1);
    assert!(
        !boxed.has_hooks(HookType::PreForward),
        "the hook must really be gone"
    );
}

#[test]
fn boxed_module_roots_its_module_walk_at_the_inner_module() {
    let (boxed, _log, _child_log) = boxed_spy().expect("spy");

    let modules = boxed.modules();
    assert_eq!(modules.len(), 2, "the spy plus its one child");
    assert_eq!(
        modules[0].name(),
        Some("SpyModule"),
        "the root of the walk must be the inner module, not the box: the default \
         `modules()` pushes `self` (the `Box`), whose own answers came from the \
         trait defaults"
    );

    let named = boxed.named_modules();
    assert_eq!(named.len(), 2);
    assert_eq!(
        named[0].0, "",
        "the root is unnamed, as in the trait default"
    );
    assert_eq!(named[0].1.name(), Some("SpyModule"));
    assert_eq!(named[1].0, "child");
}

// ---------------------------------------------------------------------------
// `&mut Box<dyn Module>`
// ---------------------------------------------------------------------------

#[test]
fn mut_ref_to_boxed_module_forwards_every_leaf_default() {
    let (mut owned, log, _child_log) = boxed_spy().expect("spy");
    let boxed = &mut owned;

    assert_eq!(boxed.name(), Some("SpyModule"));
    assert_eq!(boxed.extra_repr(), "spy=1");
    assert_eq!(boxed.buffers().len(), 1);
    assert_eq!(boxed.named_buffers().len(), 1);
    assert_eq!(boxed.named_parameters().len(), 1);
    assert_eq!(boxed.state_dict().len(), 1);

    boxed.zero_grad();
    boxed.freeze();
    boxed.unfreeze();

    assert_eq!(log.name.load(Ordering::SeqCst), 1);
    assert_eq!(log.extra_repr.load(Ordering::SeqCst), 1);
    assert_eq!(log.buffers.load(Ordering::SeqCst), 1);
    assert_eq!(log.named_buffers.load(Ordering::SeqCst), 1);
    assert_eq!(log.named_parameters.load(Ordering::SeqCst), 1);
    assert_eq!(log.state_dict.load(Ordering::SeqCst), 1);
    assert_eq!(log.zero_grad.load(Ordering::SeqCst), 1);
    assert_eq!(log.freeze.load(Ordering::SeqCst), 1);
    assert_eq!(log.unfreeze.load(Ordering::SeqCst), 1);
}

// ---------------------------------------------------------------------------
// The whole point: a checkpoint round-trip over boxed children
// ---------------------------------------------------------------------------

/// Collects a full checkpoint — parameters *and* buffers — from a boxed layer
/// stack, keyed exactly the way `Sequential::named_parameters` keys its own
/// (`"{index}.{name}"`).
///
/// The walk deliberately calls `named_parameters()` / `named_buffers()` on
/// `Box<dyn Module>` values rather than on `&dyn Module`, because that is the
/// resolution `Sequential` performs on its `Vec<Box<dyn Module>>` field: method
/// lookup finds `impl Module for Box<dyn Module>` before it ever derefs to the
/// inner `dyn Module`.
fn checkpoint(layers: &[Box<dyn Module>]) -> HashMap<String, Vec<f32>> {
    let mut state = HashMap::new();
    for (index, layer) in layers.iter().enumerate() {
        for (name, param) in layer.named_parameters() {
            let values = param.clone_data().to_vec().unwrap_or_default();
            let _ = state.insert(format!("{index}.{name}"), values);
        }
        for (name, buffer) in layer.named_buffers() {
            let values = buffer.read().to_vec().unwrap_or_default();
            let _ = state.insert(format!("{index}.{name}"), values);
        }
    }
    state
}

/// Writes a checkpoint produced by [`checkpoint`] back into a boxed layer stack.
fn restore(layers: &[Box<dyn Module>], state: &HashMap<String, Vec<f32>>) {
    for (index, layer) in layers.iter().enumerate() {
        for (name, buffer) in layer.named_buffers() {
            if let Some(values) = state.get(&format!("{index}.{name}")) {
                let len = values.len();
                let restored = Tensor::from_vec(values.clone(), &[len]).expect("checkpoint tensor");
                *buffer.write() = restored;
            }
        }
    }
}

#[test]
fn a_checkpoint_over_boxed_layers_round_trips_running_statistics() {
    let _serial = serialize();

    let layers: Vec<Box<dyn Module>> = vec![Box::new(trained_batch_norm().expect("BatchNorm1d"))];

    let saved = checkpoint(&layers);
    assert!(
        saved.contains_key("0.running_mean") && saved.contains_key("0.running_var"),
        "a checkpoint taken through the box must contain the running statistics; \
         it held {:?}",
        {
            let mut keys: Vec<&String> = saved.keys().collect();
            keys.sort();
            keys
        }
    );
    let saved_mean = saved
        .get("0.running_mean")
        .expect("running_mean in checkpoint")
        .clone();
    assert!(
        saved_mean.iter().any(|v| v.abs() > 1e-6),
        "fixture must have trained the statistics away from zero, got {saved_mean:?}"
    );

    // Corrupt the live statistics the way loading a *different* checkpoint would.
    for (_name, buffer) in layers[0].named_buffers() {
        let len = buffer.read().to_vec().expect("values").len();
        *buffer.write() = Tensor::from_vec(vec![0.0f32; len], &[len]).expect("zeroed");
    }
    let zeroed = checkpoint(&layers);
    assert_eq!(
        zeroed.get("0.running_mean"),
        Some(&vec![0.0f32; saved_mean.len()]),
        "the corruption must be observable through the box"
    );

    restore(&layers, &saved);
    let reloaded = checkpoint(&layers);
    assert_eq!(
        reloaded.get("0.running_mean"),
        Some(&saved_mean),
        "restoring through the box must bring the running statistics back"
    );
}
