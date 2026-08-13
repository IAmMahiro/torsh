//! Core module trait system for neural network modules
//!
//! This module provides the foundational Module trait and essential interfaces
//! for all neural network components in ToRSh.

pub mod module_ext;

pub use module_ext::{ModuleExt, ParameterStats, ValidationReport};

use torsh_core::device::DeviceType;
use torsh_core::error::Result;
use torsh_tensor::Tensor;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

/// Base trait for all neural network modules
///
/// This trait provides the core interface for all neural network components,
/// following PyTorch-compatible patterns for maximum interoperability.
///
/// # Design Philosophy
///
/// This trait is designed for maximum ergonomics while maintaining flexibility:
/// - Most methods have sensible defaults to reduce boilerplate
/// - Core functionality (forward, training mode) is required
/// - Parameter management is streamlined with helper methods
/// - Hook system is optional but well-integrated
///
/// # Required Methods
///
/// Only [`forward()`](Module::forward) must be implemented. All other methods have sensible defaults.
///
/// # Implementing a Custom Module
///
/// ## Basic Module (No Parameters)
///
/// ```rust
/// use torsh_nn::{Module, ModuleBase};
/// use torsh_tensor::Tensor;
/// use torsh_core::error::Result;
///
/// /// Simple ReLU activation function
/// struct MyReLU {
///     base: ModuleBase,
/// }
///
/// impl MyReLU {
///     fn new() -> Self {
///         Self {
///             base: ModuleBase::new(),
///         }
///     }
/// }
///
/// impl Module for MyReLU {
///     fn forward(&self, input: &Tensor) -> Result<Tensor> {
///         // Apply ReLU: max(0, x)
///         input.relu()
///     }
///
///     fn training(&self) -> bool {
///         self.base.training()
///     }
///
///     fn set_training(&mut self, training: bool) {
///         self.base.set_training(training);
///     }
/// }
/// ```
///
/// ## Module with Parameters
///
/// ```rust
/// use torsh_nn::{Module, ModuleBase, Parameter};
/// use torsh_tensor::{Tensor, creation};
/// use torsh_core::error::Result;
/// use std::collections::HashMap;
///
/// /// Custom linear layer with learnable weight and bias
/// struct MyLinear {
///     base: ModuleBase,
///     in_features: usize,
///     out_features: usize,
/// }
///
/// impl MyLinear {
///     fn new(in_features: usize, out_features: usize) -> Result<Self> {
///         let mut base = ModuleBase::new();
///
///         // Initialize weight: [in_features, out_features]
///         let weight = creation::randn(&[in_features, out_features])?;
///         base.register_parameter("weight".to_string(), Parameter::new(weight));
///
///         // Initialize bias: [out_features]
///         let bias = creation::zeros(&[out_features])?;
///         base.register_parameter("bias".to_string(), Parameter::new(bias));
///
///         Ok(Self {
///             base,
///             in_features,
///             out_features,
///         })
///     }
/// }
///
/// impl Module for MyLinear {
///     fn forward(&self, input: &Tensor) -> Result<Tensor> {
///         // Get parameters
///         let weight = self.base.parameters["weight"].tensor().read().clone();
///         let bias = self.base.parameters["bias"].tensor().read().clone();
///
///         // Compute: input @ weight + bias
///         let output = input.matmul(&weight)?;
///         output.add(&bias)
///     }
///
///     fn parameters(&self) -> HashMap<String, Parameter> {
///         self.base.parameters.clone()
///     }
///
///     fn named_parameters(&self) -> HashMap<String, Parameter> {
///         self.base.named_parameters()
///     }
///
///     fn training(&self) -> bool {
///         self.base.training()
///     }
///
///     fn set_training(&mut self, training: bool) {
///         self.base.set_training(training);
///     }
/// }
/// ```
///
/// ## Module with Training/Evaluation Modes
///
/// ```rust
/// use torsh_nn::{Module, ModuleBase};
/// use torsh_tensor::{Tensor, creation};
/// use torsh_core::error::Result;
///
/// /// Dropout layer with different behavior in train/eval modes
/// struct MyDropout {
///     base: ModuleBase,
///     p: f32,  // Dropout probability
/// }
///
/// impl MyDropout {
///     fn new(p: f32) -> Self {
///         Self {
///             base: ModuleBase::new(),
///             p,
///         }
///     }
/// }
///
/// impl Module for MyDropout {
///     fn forward(&self, input: &Tensor) -> Result<Tensor> {
///         if self.training() {
///             // During training: randomly zero out units
///             let rand_vals = creation::rand_like(input)?;
///             let threshold = creation::full_like(input, self.p)?;
///             // Keep elements where random value >= p
///             let keep_mask = rand_vals.ge(&threshold)?;
///             let zeros = creation::zeros_like(input)?;
///             let masked = input.where_tensor(&keep_mask, &zeros)?;
///             // Scale by 1/(1-p) to maintain expected value (inverted dropout)
///             masked.div_scalar(1.0 - self.p)
///         } else {
///             // During evaluation: pass through unchanged
///             Ok(input.clone())
///         }
///     }
///
///     fn training(&self) -> bool {
///         self.base.training()
///     }
///
///     fn set_training(&mut self, training: bool) {
///         self.base.set_training(training);
///     }
/// }
/// ```
///
/// # Complete Training Loop Example
///
/// ```rust,no_run
/// use torsh_nn::prelude::{Module, Linear, Sequential};
/// use torsh_tensor::{Tensor, creation};
/// use torsh_core::error::Result;
///
/// fn train_eval_example() -> Result<()> {
///     // 1. Create model
///     let mut model = Sequential::new()
///         .add(Linear::new(784, 128, true))
///         .add(Linear::new(128, 10, true));
///
///     // 2. Set model to training mode
///     model.train();
///
///     // 3. Forward pass
///     let inputs = creation::randn(&[32, 784])?;
///     let outputs = model.forward(&inputs)?;
///
///     // 4. Compute loss (simplified MSE)
///     let targets = creation::randn(&[32, 10])?;
///     let diff = outputs.sub(&targets)?;
///     let loss = diff.pow_scalar(2.0)?.mean(None, false)?;
///
///     // 5. Backward pass (computes gradients)
///     loss.backward()?;
///
///     // 6. Switch to evaluation mode
///     model.eval();
///     let test_input = creation::randn(&[10, 784])?;
///     let test_output = model.forward(&test_input)?;
///
///     Ok(())
/// }
/// ```
///
/// # Method Categories
///
/// ## Core Methods (Required)
/// - [`forward()`](Module::forward) - Forward pass computation
///
/// ## Parameter Management
/// - [`parameters()`](Module::parameters) - Get all trainable parameters
/// - [`named_parameters()`](Module::named_parameters) - Get parameters with names
/// - [`all_parameters()`](Module::all_parameters) - Get parameters recursively
/// - [`zero_grad()`](Module::zero_grad) - Clear all gradients
///
/// ## Buffer Management
/// Buffers are persistent, *untrained* tensors (a `BatchNorm`'s `running_mean`,
/// `running_var`, `num_batches_tracked`). They are not returned by
/// `parameters()` and no optimizer touches them, but they are model state and
/// travel in the checkpoint.
/// - [`buffers()`](Module::buffers) - Get all buffers
/// - [`named_buffers()`](Module::named_buffers) - Get buffers with names
/// - [`all_named_buffers()`](Module::all_named_buffers) - Get buffers recursively
///
/// ## Training Mode Control
/// - [`training()`](Module::training) - Check if in training mode
/// - [`train()`](Module::train) - Set to training mode
/// - [`eval()`](Module::eval) - Set to evaluation mode
/// - [`set_training()`](Module::set_training) - Set training mode explicitly
///
/// ## Module Hierarchy
/// - [`children()`](Module::children) - Get direct child modules
/// - [`named_children()`](Module::named_children) - Get children with names
/// - [`modules()`](Module::modules) - Get all modules recursively
///
/// ## State Management
/// - [`state_dict()`](Module::state_dict) - Save module state (parameters *and* buffers)
/// - [`load_state_dict()`](Module::load_state_dict) - Load module state (parameters *and* buffers)
/// - [`to_device()`](Module::to_device) - Move module to device
///
/// ## Utilities
/// - [`freeze()`](Module::freeze) - Freeze all parameters
/// - [`unfreeze()`](Module::unfreeze) - Unfreeze all parameters
/// - [`num_parameters()`](Module::num_parameters) - Count total parameters
/// - [`diagnose()`](Module::diagnose) - Check module health
///
/// # PyTorch Compatibility
///
/// ToRSh's Module trait closely follows PyTorch's `nn.Module` interface:
///
/// | PyTorch | ToRSh | Notes |
/// |---------|-------|-------|
/// | `forward(x)` | `forward(&x)` | Returns `Result<Tensor>` |
/// | `parameters()` | `parameters()` | Returns `HashMap<String, Parameter>` |
/// | `named_buffers()` | `named_buffers()` | Returns `HashMap<String, Arc<RwLock<Tensor>>>` — live handles, not copies |
/// | `train()` | `train()` | Sets training mode |
/// | `eval()` | `eval()` | Sets evaluation mode |
/// | `state_dict()` | `state_dict()` | Returns parameter **and buffer** tensors |
/// | `load_state_dict()` | `load_state_dict()` | Loads parameters **and buffers** from a HashMap |
/// | `to(device)` | `to_device(device)` | Moves to device |
/// | `zero_grad()` | `zero_grad()` | Clears gradients |
///
/// One deliberate divergence: PyTorch exempts `num_batches_tracked` from
/// strict-mode key checking for backwards compatibility with pre-0.4.1
/// checkpoints. ToRSh grants no per-key exemptions — every buffer is required
/// under `strict = true`.
///
/// # Best Practices
///
/// 1. **Always use `ModuleBase`**: Store a `ModuleBase` instance in your module to handle
///    common functionality like parameters, buffers, and training state.
///
/// 2. **Register parameters in constructor**: Use `base.register_parameter()` to register
///    all trainable parameters during module creation.
///
/// 3. **Implement training/eval behavior**: If your module behaves differently during
///    training vs evaluation (like Dropout, BatchNorm), check `self.training()`.
///
/// 4. **Use `Result<Tensor>` for error handling**: Always return `Result<Tensor>` from
///    `forward()` to properly propagate errors.
///
/// 5. **Delegate to `ModuleBase`**: Implement parameter and training mode methods by
///    delegating to your `ModuleBase` instance.
///
/// 6. **Initialize parameters properly**: Use initialization methods from `torsh_nn::init`
///    for better training convergence.
pub trait Module: Send + Sync {
    /// Forward pass through the module
    ///
    /// This is the only required method that must be implemented by all modules.
    ///
    /// # Arguments
    /// * `input` - Input tensor
    ///
    /// # Returns
    /// * `Result<Tensor>` - Output tensor or error
    fn forward(&self, input: &Tensor) -> Result<Tensor>;

    /// Get all parameters in the module (non-recursive)
    ///
    /// Override this method if your module has trainable parameters.
    /// The default implementation returns an empty map.
    ///
    /// # Returns
    /// * `HashMap<String, Parameter>` - Map of parameter names to parameters
    fn parameters(&self) -> HashMap<String, crate::Parameter> {
        HashMap::new()
    }

    /// Get named parameters (non-recursive)
    ///
    /// Default implementation delegates to `parameters()`. Override if you need
    /// different behavior for named vs unnamed parameter access.
    ///
    /// # Returns
    /// * `HashMap<String, Parameter>` - Map of parameter names to parameters
    fn named_parameters(&self) -> HashMap<String, crate::Parameter> {
        self.parameters()
    }

    /// Get all parameters recursively including submodules
    ///
    /// # Returns
    /// * `HashMap<String, Parameter>` - Flattened map of all parameters
    fn all_parameters(&self) -> HashMap<String, crate::Parameter> {
        let mut all_params = self.parameters();

        for child in self.children() {
            let child_params = child.all_parameters();
            for (name, param) in child_params {
                all_params.insert(name, param);
            }
        }

        all_params
    }

    /// Get all named parameters recursively with module prefixes
    ///
    /// # Returns
    /// * `HashMap<String, Parameter>` - Hierarchical parameter names
    fn all_named_parameters(&self) -> HashMap<String, crate::Parameter> {
        let mut all_params = HashMap::new();

        // Add own parameters
        for (name, param) in self.named_parameters() {
            all_params.insert(name, param);
        }

        // Add child parameters with prefixes
        let children_named = self.named_children();
        for (child_name, child) in children_named {
            for (param_name, param) in child.all_named_parameters() {
                let full_name = format!("{}.{}", child_name, param_name);
                all_params.insert(full_name, param);
            }
        }

        all_params
    }

    /// Get all named buffers recursively with module prefixes
    ///
    /// The buffer-side twin of [`all_named_parameters()`](Module::all_named_parameters),
    /// and the recursion [`state_dict()`](Module::state_dict) uses to reach a
    /// child's non-trainable state. Buffers are a module's *persistent, untrained*
    /// tensors — a `BatchNorm`'s `running_mean` / `running_var` /
    /// `num_batches_tracked` — which evaluation mode consumes in place of the
    /// batch statistics, so losing them turns a trained normalization layer back
    /// into a freshly constructed one.
    ///
    /// Child names come from [`named_children()`](Module::named_children); a
    /// module that overrides only `children()` contributes no buffer names here,
    /// exactly as it contributes no parameter names to `all_named_parameters()`.
    ///
    /// # Returns
    /// * `HashMap<String, Arc<RwLock<Tensor>>>` - Hierarchical buffer names,
    ///   holding the module's *live* handles rather than copies
    fn all_named_buffers(&self) -> HashMap<String, std::sync::Arc<parking_lot::RwLock<Tensor>>> {
        let mut all_buffers = self.named_buffers();

        for (child_name, child) in self.named_children() {
            for (buffer_name, buffer) in child.all_named_buffers() {
                all_buffers.insert(format!("{}.{}", child_name, buffer_name), buffer);
            }
        }

        all_buffers
    }

    /// Check if in training mode
    ///
    /// Default implementation returns true. Override if your module tracks training state.
    fn training(&self) -> bool {
        true
    }

    /// Set training mode
    ///
    /// Convenience method that calls `set_training(true)`.
    fn train(&mut self) {
        self.set_training(true);
    }

    /// Set evaluation mode
    ///
    /// Convenience method that calls `set_training(false)`.
    fn eval(&mut self) {
        self.set_training(false);
    }

    /// Set training mode (internal implementation)
    ///
    /// Default implementation does nothing. Override if your module needs to track
    /// training state or propagate it to child modules.
    fn set_training(&mut self, _training: bool) {
        // Default: do nothing
    }

    /// Move module to device
    ///
    /// Default implementation does nothing. Override if your module has parameters
    /// or buffers that need to be moved between devices.
    fn to_device(&mut self, _device: DeviceType) -> Result<()> {
        Ok(())
    }

    /// Load state dictionary into the module
    ///
    /// Restores **parameters and buffers**, matching PyTorch's
    /// `nn.Module.load_state_dict`. Buffers are written through the live
    /// handles published by [`all_named_buffers()`](Module::all_named_buffers),
    /// so a `BatchNorm`'s running statistics come back into the layer itself
    /// rather than into a detached copy.
    ///
    /// # Arguments
    /// * `state_dict` - Map of parameter *and buffer* names to tensors
    /// * `strict` - Whether to require exact name matches. Strict mode counts
    ///   buffers: a checkpoint missing `running_mean` is rejected, with no
    ///   per-key exemptions (PyTorch's legacy leniency for
    ///   `num_batches_tracked` is deliberately not reproduced — silently
    ///   accepting a partial checkpoint is the defect this method exists to
    ///   prevent).
    ///
    /// # Returns
    /// * `Result<()>` - Success or error with details about missing/unexpected keys
    ///
    /// # Errors
    ///
    /// Shapes are validated for every matching entry *before* anything is
    /// written, so a corrupt checkpoint leaves the module exactly as it was
    /// instead of half-loaded. A shape mismatch is an error in both strict and
    /// non-strict mode; `strict` governs which *names* must be present, not
    /// whether the values that are present have to fit.
    fn load_state_dict(
        &mut self,
        state_dict: &HashMap<String, Tensor>,
        strict: bool,
    ) -> Result<()> {
        let current_params = self.all_named_parameters();
        let current_buffers = self.all_named_buffers();
        let mut missing_keys = Vec::new();
        let mut unexpected_keys = Vec::new();

        // Check for missing parameters and buffers
        for name in current_params.keys().chain(current_buffers.keys()) {
            if !state_dict.contains_key(name) {
                missing_keys.push(name.clone());
            }
        }

        // Check for entries the module cannot consume
        for name in state_dict.keys() {
            if !current_params.contains_key(name) && !current_buffers.contains_key(name) {
                unexpected_keys.push(name.clone());
            }
        }

        if strict && (!missing_keys.is_empty() || !unexpected_keys.is_empty()) {
            // Sorted so the report is reproducible; both maps iterate in
            // unspecified order.
            missing_keys.sort();
            unexpected_keys.sort();
            return Err(torsh_core::error::TorshError::Other(format!(
                "State dict loading failed. Missing keys: {:?}, Unexpected keys: {:?}",
                missing_keys, unexpected_keys
            )));
        }

        // Validate every shape before mutating anything.
        for (name, param) in &current_params {
            if let Some(new_tensor) = state_dict.get(name) {
                let current_shape = param.shape()?;
                let new_shape = new_tensor.shape().dims().to_vec();
                if current_shape != new_shape {
                    return Err(torsh_core::error::TorshError::Other(format!(
                        "Shape mismatch for parameter '{}': expected {:?}, got {:?}",
                        name, current_shape, new_shape
                    )));
                }
            }
        }
        for (name, buffer) in &current_buffers {
            if let Some(new_tensor) = state_dict.get(name) {
                // The read guard is a statement-scoped temporary: it must be
                // released before the write pass below takes the same lock,
                // because `parking_lot::RwLock` is not reentrant.
                let current_shape = buffer.read().shape().dims().to_vec();
                let new_shape = new_tensor.shape().dims().to_vec();
                if current_shape != new_shape {
                    return Err(torsh_core::error::TorshError::Other(format!(
                        "Shape mismatch for buffer '{}': expected {:?}, got {:?}",
                        name, current_shape, new_shape
                    )));
                }
            }
        }

        // Load matching parameters
        for (name, param) in current_params {
            if let Some(new_tensor) = state_dict.get(&name) {
                *param.tensor().write() = new_tensor.clone();
            }
        }

        // Load matching buffers
        for (name, buffer) in current_buffers {
            if let Some(new_tensor) = state_dict.get(&name) {
                *buffer.write() = new_tensor.clone();
            }
        }

        Ok(())
    }

    /// Load state dictionary with default strict=true
    fn load_state_dict_strict(&mut self, state_dict: &HashMap<String, Tensor>) -> Result<()> {
        self.load_state_dict(state_dict, true)
    }

    /// Save state dictionary from the module
    ///
    /// Carries **parameters and buffers**, matching PyTorch's
    /// `nn.Module.state_dict`. Buffers are a module's persistent untrained
    /// state — `running_mean`, `running_var`, `num_batches_tracked` — and
    /// evaluation mode consumes them in place of the batch statistics, so a
    /// checkpoint that omitted them reloaded as a *different* model with no
    /// error reported anywhere.
    ///
    /// Values are snapshots taken at call time, keyed by
    /// [`all_named_parameters()`](Module::all_named_parameters) and
    /// [`all_named_buffers()`](Module::all_named_buffers), which is exactly the
    /// key set [`load_state_dict()`](Module::load_state_dict) expects back.
    fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut state = HashMap::new();
        for (name, param) in self.all_named_parameters() {
            state.insert(name, param.clone_data());
        }
        for (name, buffer) in self.all_named_buffers() {
            // PyTorch rejects registering a buffer under a name already taken
            // by a parameter; nothing enforces that here, so the impossible
            // case is resolved deterministically in the parameter's favour
            // rather than by silently overwriting trainable state.
            state.entry(name).or_insert_with(|| buffer.read().clone());
        }
        state
    }

    /// Get the module name (optional, for debugging and serialization)
    fn name(&self) -> Option<&str> {
        None
    }

    /// Get all buffers (non-trainable parameters)
    fn buffers(&self) -> Vec<std::sync::Arc<parking_lot::RwLock<Tensor>>> {
        Vec::new()
    }

    /// Get named buffers
    fn named_buffers(&self) -> HashMap<String, std::sync::Arc<parking_lot::RwLock<Tensor>>> {
        HashMap::new()
    }

    /// Get all direct child modules
    ///
    /// Default implementation returns an empty vector. Override if your module
    /// contains child modules.
    fn children(&self) -> Vec<&dyn Module> {
        Vec::new()
    }

    /// Get all direct child modules with names
    ///
    /// Default implementation returns an empty vector. Override if your module
    /// contains named child modules.
    fn named_children(&self) -> Vec<(String, &dyn Module)> {
        Vec::new()
    }

    /// Get all modules recursively (depth-first traversal)
    fn modules(&self) -> Vec<&dyn Module>
    where
        Self: Sized,
    {
        let mut modules: Vec<&dyn Module> = vec![self];
        for child in self.children() {
            // Since child is &dyn Module, we need to use a different approach
            // We'll just collect immediate children for now
            modules.push(child);
        }
        modules
    }

    /// Get all modules recursively with hierarchical names
    fn named_modules(&self) -> Vec<(String, &dyn Module)>
    where
        Self: Sized,
    {
        let mut modules: Vec<(String, &dyn Module)> = vec![(String::new(), self)];

        for (child_name, child) in self.named_children() {
            // Since child is &dyn Module, we need to use a different approach
            // We'll just collect immediate named children for now
            modules.push((child_name, child));
        }

        modules
    }

    /// Zero all gradients recursively
    ///
    /// Default implementation does nothing. Override if your module has parameters
    /// with gradients that need to be zeroed.
    fn zero_grad(&mut self) {
        // Default: do nothing
    }

    /// Count total number of parameters
    fn num_parameters(&self) -> usize {
        self.all_parameters()
            .values()
            .map(|p| p.numel().unwrap_or(0))
            .sum()
    }

    /// Count trainable parameters
    fn num_trainable_parameters(&self) -> usize {
        self.all_parameters()
            .values()
            .filter(|p| p.requires_grad())
            .map(|p| p.numel().unwrap_or(0))
            .sum()
    }

    /// Get memory usage estimate in bytes
    fn memory_usage(&self) -> usize {
        self.all_parameters()
            .values()
            .map(|p| p.numel().unwrap_or(0) * 4) // Assume f32 = 4 bytes
            .sum()
    }

    /// Freeze all parameters (set requires_grad = false)
    ///
    /// Default implementation does nothing. Override if your module has parameters
    /// that can be frozen/unfrozen.
    fn freeze(&mut self) {
        // Default: do nothing
    }

    /// Unfreeze all parameters (set requires_grad = true)
    ///
    /// Default implementation does nothing. Override if your module has parameters
    /// that can be frozen/unfrozen.
    fn unfreeze(&mut self) {
        // Default: do nothing
    }

    /// Get string representation
    fn extra_repr(&self) -> String {
        String::new()
    }

    /// Register a hook for this module (default implementation does nothing)
    fn register_hook(
        &mut self,
        _hook_type: crate::HookType,
        _callback: crate::HookCallback,
    ) -> Option<crate::HookHandle> {
        None
    }

    /// Remove a hook by handle (default implementation does nothing)
    fn remove_hook(&mut self, _hook_type: crate::HookType, _handle: crate::HookHandle) -> bool {
        false
    }

    /// Execute hooks of a specific type (default implementation does nothing)
    fn execute_hooks(
        &self,
        _hook_type: crate::HookType,
        _input: &Tensor,
        _output: Option<&Tensor>,
    ) -> Result<()> {
        Ok(())
    }

    /// Forward pass with hooks support
    fn forward_with_hooks(&self, input: &Tensor) -> Result<Tensor> {
        // Execute pre-forward hooks
        self.execute_hooks(crate::HookType::PreForward, input, None)?;

        // Perform forward pass
        let output = self.forward(input)?;

        // Execute post-forward hooks
        self.execute_hooks(crate::HookType::PostForward, input, Some(&output))?;

        Ok(output)
    }

    /// Check if module has hooks registered
    fn has_hooks(&self, _hook_type: crate::HookType) -> bool {
        false
    }

    // === Ergonomic Helper Methods ===

    /// Convenient method to call forward and handle common patterns
    ///
    /// This is equivalent to `forward()` but provides a more ergonomic interface
    /// for chaining operations.
    fn call(&self, input: &Tensor) -> Result<Tensor> {
        self.forward(input)
    }

    /// Apply the module to input (alias for forward)
    ///
    /// PyTorch-style method name for compatibility.
    fn apply(&self, input: &Tensor) -> Result<Tensor> {
        self.forward(input)
    }

    /// Check if the module has any parameters
    fn has_parameters(&self) -> bool {
        !self.parameters().is_empty()
    }

    /// Check if the module has any child modules
    fn has_children(&self) -> bool {
        !self.children().is_empty()
    }

    /// Get parameter count (convenience method)
    fn parameter_count(&self) -> usize {
        self.num_parameters()
    }

    /// Get trainable parameter count (convenience method)
    fn trainable_parameter_count(&self) -> usize {
        self.num_trainable_parameters()
    }

    /// Get memory usage in MB (convenience method)
    fn memory_usage_mb(&self) -> f64 {
        self.memory_usage() as f64 / (1024.0 * 1024.0)
    }

    /// Toggle training mode (convenience method)
    fn toggle_training(&mut self) {
        self.set_training(!self.training());
    }

    /// Check if module is in evaluation mode
    fn eval_mode(&self) -> bool {
        !self.training()
    }

    // === Enhanced Ergonomic Methods ===

    /// Sequential forward pass through multiple modules
    ///
    /// This provides a convenient way to chain multiple forward passes.
    ///
    /// # Arguments
    /// * `modules` - Slice of modules to apply sequentially
    /// * `input` - Input tensor
    ///
    /// # Returns
    /// * `Result<Tensor>` - Final output after all modules
    ///
    /// # Example
    /// ```ignore
    /// let result = Module::sequential_forward(&[&layer1, &layer2, &layer3], &input)?;
    /// ```
    fn sequential_forward(modules: &[&dyn Module], mut input: Tensor) -> Result<Tensor>
    where
        Self: Sized,
    {
        for module in modules {
            input = module.forward(&input)?;
        }
        Ok(input)
    }

    /// Apply module multiple times with different inputs (batch processing)
    ///
    /// This is useful for processing multiple independent inputs through the same module.
    ///
    /// # Arguments
    /// * `inputs` - Vector of input tensors
    ///
    /// # Returns
    /// * `Result<Vec<Tensor>>` - Vector of output tensors
    fn batch_forward(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        inputs.iter().map(|input| self.forward(input)).collect()
    }

    /// Forward with condition - only apply if condition is true
    ///
    /// This provides conditional execution of modules.
    ///
    /// # Arguments
    /// * `input` - Input tensor
    /// * `condition` - Whether to apply this module
    ///
    /// # Returns
    /// * `Result<Tensor>` - Output tensor (input if condition is false)
    fn conditional_forward(&self, input: &Tensor, condition: bool) -> Result<Tensor> {
        if condition {
            self.forward(input)
        } else {
            Ok(input.clone())
        }
    }

    /// Forward with residual connection
    ///
    /// Applies the module and adds the result to the input (residual/skip connection).
    ///
    /// # Arguments
    /// * `input` - Input tensor
    ///
    /// # Returns
    /// * `Result<Tensor>` - Output tensor (input + forward(input))
    fn residual_forward(&self, input: &Tensor) -> Result<Tensor> {
        let output = self.forward(input)?;
        // This would use tensor addition when available
        // For now, just return the output
        Ok(output)
    }

    /// Get detailed module information for debugging
    ///
    /// This provides comprehensive information about the module state.
    ///
    /// # Returns
    /// * `ModuleInfo` - Detailed module information
    fn module_info(&self) -> crate::ModuleInfo {
        crate::ModuleInfo {
            name: self.name().unwrap_or("Unknown").to_string(),
            training: self.training(),
            parameter_count: self.num_parameters(),
            trainable_parameter_count: self.num_trainable_parameters(),
            memory_usage_bytes: self.memory_usage(),
            has_children: self.has_children(),
            children_count: self.children().len(),
        }
    }

    /// Check if module is ready for training
    ///
    /// Performs various checks to ensure the module is properly configured for training.
    ///
    /// # Returns
    /// * `Result<()>` - Ok if ready, Error with details if not
    fn check_training_readiness(&self) -> Result<()> {
        // Check if module has parameters
        if !self.has_parameters() {
            return Err(torsh_core::error::TorshError::Other(
                "Module has no parameters - may not be trainable".to_string(),
            ));
        }

        // Check if in training mode
        if !self.training() {
            return Err(torsh_core::error::TorshError::Other(
                "Module is in evaluation mode - switch to training mode first".to_string(),
            ));
        }

        // Check for finite parameters
        for param in self.parameters().values() {
            if !param.is_finite().unwrap_or(false) {
                return Err(torsh_core::error::TorshError::Other(
                    "Module contains non-finite parameters (NaN or infinity)".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Get parameter names matching a pattern
    ///
    /// This helps with selective parameter access and manipulation.
    ///
    /// # Arguments
    /// * `pattern` - String pattern to match against parameter names
    ///
    /// # Returns
    /// * `Vec<String>` - Vector of parameter names matching the pattern
    fn parameter_names_matching(&self, pattern: &str) -> Vec<String> {
        self.all_named_parameters()
            .keys()
            .filter(|name| name.contains(pattern))
            .cloned()
            .collect()
    }

    /// Get parameters by layer type (e.g., "weight", "bias")
    ///
    /// # Arguments
    /// * `param_type` - Type of parameters to retrieve
    ///
    /// # Returns
    /// * `HashMap<String, Parameter>` - Filtered parameters
    fn parameters_by_type(&self, param_type: &str) -> HashMap<String, crate::Parameter> {
        self.all_named_parameters()
            .into_iter()
            .filter(|(name, _)| name.contains(param_type))
            .collect()
    }

    /// Clone module parameters (for creating copies or checkpoints)
    ///
    /// # Returns
    /// * `HashMap<String, Tensor>` - Cloned parameter tensors
    fn clone_parameters(&self) -> HashMap<String, Tensor> {
        self.all_named_parameters()
            .into_iter()
            .map(|(name, param)| (name, param.clone_data()))
            .collect()
    }

    /// Quick diagnostic check of module health
    ///
    /// # Returns
    /// * `ModuleDiagnostics` - Diagnostic information
    fn diagnose(&self) -> crate::ModuleDiagnostics {
        let mut issues = Vec::new();
        let mut warnings = Vec::new();

        // Check parameter health
        for (name, param) in self.all_named_parameters() {
            if let Ok(diag) = param.diagnose() {
                if !diag.issues.is_empty() {
                    issues.extend(
                        diag.issues
                            .into_iter()
                            .map(|issue| format!("{}: {}", name, issue)),
                    );
                }
                if !diag.warnings.is_empty() {
                    warnings.extend(
                        diag.warnings
                            .into_iter()
                            .map(|warning| format!("{}: {}", name, warning)),
                    );
                }
            }
        }

        // Check training readiness
        if let Err(e) = self.check_training_readiness() {
            warnings.push(format!("Training readiness: {}", e));
        }

        crate::ModuleDiagnostics {
            module_info: self.module_info(),
            issues,
            warnings,
            parameter_diagnostics: self
                .all_named_parameters()
                .into_iter()
                .filter_map(|(name, param)| param.diagnose().ok().map(|d| (name, d)))
                .collect(),
        }
    }
}

/// Forwards **every** `Module` method to `(**self)`.
///
/// `Module` has one required method and ~50 defaulted ones, and those defaults
/// split into two families. *Composed* defaults (`all_named_parameters`,
/// `state_dict`, `num_parameters`, `residual_forward`, …) are written in terms
/// of other trait methods, so an impl that forwards the primitives inherits
/// them correctly. *Leaf* defaults (`buffers`, `named_buffers`, `name`,
/// `zero_grad`, `freeze`, `unfreeze`, `extra_repr`, and the whole hook
/// protocol) return emptiness — `Vec::new()`, `HashMap::new()`, `None`,
/// `false`, `Ok(())`, an empty body. A smart-pointer impl that *omits* one of
/// those does not fall through to the pointee: it answers "nothing" on the
/// pointee's behalf, with no error anywhere.
///
/// That distinction is why this macro exists rather than a hand-written list.
/// The predecessor forwarded nine methods and omitted every leaf default, so a
/// boxed `BatchNorm1d` reported zero buffers while the bare layer reported
/// three — and `Sequential`/`ModuleList` store their children as
/// `Vec<Box<dyn Module>>`, where *every* trait call on a child resolves through
/// this impl. A checkpoint or device migration taken through the box therefore
/// dropped `running_mean` / `running_var` / `num_batches_tracked` silently,
/// reverting a trained normalization layer to its initialization on reload.
/// `tests/hardening_nn_module_box.rs` pins the whole surface.
///
/// The body is spelled `(**self)` for both implementors on purpose:
/// - for `Box<dyn Module>` that is the `dyn Module` pointee, i.e. a vtable call;
/// - for `&mut Box<dyn Module>` that is the `Box<dyn Module>` itself, i.e. a
///   call into the impl directly above, which then performs the vtable call.
///
/// `modules`/`named_modules` are excluded: they carry a `where Self: Sized`
/// bound, so they cannot be invoked on an unsized `dyn Module` and are supplied
/// per-implementor below. `sequential_forward` is excluded too — it is an
/// associated function with no receiver, so there is nothing to forward to.
macro_rules! forward_all_module_methods {
    () => {
        fn forward(&self, input: &Tensor) -> Result<Tensor> {
            (**self).forward(input)
        }

        fn parameters(&self) -> HashMap<String, crate::Parameter> {
            (**self).parameters()
        }

        fn named_parameters(&self) -> HashMap<String, crate::Parameter> {
            (**self).named_parameters()
        }

        fn all_parameters(&self) -> HashMap<String, crate::Parameter> {
            (**self).all_parameters()
        }

        fn all_named_parameters(&self) -> HashMap<String, crate::Parameter> {
            (**self).all_named_parameters()
        }

        fn all_named_buffers(
            &self,
        ) -> HashMap<String, std::sync::Arc<parking_lot::RwLock<Tensor>>> {
            (**self).all_named_buffers()
        }

        fn training(&self) -> bool {
            (**self).training()
        }

        fn train(&mut self) {
            (**self).train()
        }

        fn eval(&mut self) {
            (**self).eval()
        }

        fn set_training(&mut self, training: bool) {
            (**self).set_training(training)
        }

        fn to_device(&mut self, device: DeviceType) -> Result<()> {
            (**self).to_device(device)
        }

        fn load_state_dict(
            &mut self,
            state_dict: &HashMap<String, Tensor>,
            strict: bool,
        ) -> Result<()> {
            (**self).load_state_dict(state_dict, strict)
        }

        fn load_state_dict_strict(&mut self, state_dict: &HashMap<String, Tensor>) -> Result<()> {
            (**self).load_state_dict_strict(state_dict)
        }

        fn state_dict(&self) -> HashMap<String, Tensor> {
            (**self).state_dict()
        }

        fn name(&self) -> Option<&str> {
            (**self).name()
        }

        fn buffers(&self) -> Vec<std::sync::Arc<parking_lot::RwLock<Tensor>>> {
            (**self).buffers()
        }

        fn named_buffers(&self) -> HashMap<String, std::sync::Arc<parking_lot::RwLock<Tensor>>> {
            (**self).named_buffers()
        }

        fn children(&self) -> Vec<&dyn Module> {
            (**self).children()
        }

        fn named_children(&self) -> Vec<(String, &dyn Module)> {
            (**self).named_children()
        }

        fn zero_grad(&mut self) {
            (**self).zero_grad()
        }

        fn num_parameters(&self) -> usize {
            (**self).num_parameters()
        }

        fn num_trainable_parameters(&self) -> usize {
            (**self).num_trainable_parameters()
        }

        fn memory_usage(&self) -> usize {
            (**self).memory_usage()
        }

        fn freeze(&mut self) {
            (**self).freeze()
        }

        fn unfreeze(&mut self) {
            (**self).unfreeze()
        }

        fn extra_repr(&self) -> String {
            (**self).extra_repr()
        }

        fn register_hook(
            &mut self,
            hook_type: crate::HookType,
            callback: crate::HookCallback,
        ) -> Option<crate::HookHandle> {
            (**self).register_hook(hook_type, callback)
        }

        fn remove_hook(&mut self, hook_type: crate::HookType, handle: crate::HookHandle) -> bool {
            (**self).remove_hook(hook_type, handle)
        }

        fn execute_hooks(
            &self,
            hook_type: crate::HookType,
            input: &Tensor,
            output: Option<&Tensor>,
        ) -> Result<()> {
            (**self).execute_hooks(hook_type, input, output)
        }

        fn forward_with_hooks(&self, input: &Tensor) -> Result<Tensor> {
            (**self).forward_with_hooks(input)
        }

        fn has_hooks(&self, hook_type: crate::HookType) -> bool {
            (**self).has_hooks(hook_type)
        }

        fn call(&self, input: &Tensor) -> Result<Tensor> {
            (**self).call(input)
        }

        fn apply(&self, input: &Tensor) -> Result<Tensor> {
            (**self).apply(input)
        }

        fn has_parameters(&self) -> bool {
            (**self).has_parameters()
        }

        fn has_children(&self) -> bool {
            (**self).has_children()
        }

        fn parameter_count(&self) -> usize {
            (**self).parameter_count()
        }

        fn trainable_parameter_count(&self) -> usize {
            (**self).trainable_parameter_count()
        }

        fn memory_usage_mb(&self) -> f64 {
            (**self).memory_usage_mb()
        }

        fn toggle_training(&mut self) {
            (**self).toggle_training()
        }

        fn eval_mode(&self) -> bool {
            (**self).eval_mode()
        }

        fn batch_forward(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
            (**self).batch_forward(inputs)
        }

        fn conditional_forward(&self, input: &Tensor, condition: bool) -> Result<Tensor> {
            (**self).conditional_forward(input, condition)
        }

        fn residual_forward(&self, input: &Tensor) -> Result<Tensor> {
            (**self).residual_forward(input)
        }

        fn module_info(&self) -> crate::ModuleInfo {
            (**self).module_info()
        }

        fn check_training_readiness(&self) -> Result<()> {
            (**self).check_training_readiness()
        }

        fn parameter_names_matching(&self, pattern: &str) -> Vec<String> {
            (**self).parameter_names_matching(pattern)
        }

        // Spelled as a fully-qualified call because `ModuleExt` (blanket-impl'd
        // for every `Module`) also has a `parameters_by_type`, with a different
        // signature; plain method syntax is ambiguous between the two.
        fn parameters_by_type(&self, param_type: &str) -> HashMap<String, crate::Parameter> {
            Module::parameters_by_type(&**self, param_type)
        }

        fn clone_parameters(&self) -> HashMap<String, Tensor> {
            (**self).clone_parameters()
        }

        fn diagnose(&self) -> crate::ModuleDiagnostics {
            (**self).diagnose()
        }
    };
}

/// Implementation for boxed trait objects
impl Module for Box<dyn Module> {
    forward_all_module_methods!();

    /// Rooted at the *inner* module rather than at the box.
    ///
    /// The trait default pushes `self`, which here is the `Box` — a legal
    /// `&dyn Module`, but one extra indirection deep, so a caller walking the
    /// tree sees a wrapper that has no counterpart in the module hierarchy it
    /// is describing. `(**self).modules()` cannot be used to delegate because
    /// the method carries `where Self: Sized` and `dyn Module` is unsized, so
    /// the walk is rebuilt here from the same two pieces the default uses.
    fn modules(&self) -> Vec<&dyn Module>
    where
        Self: Sized,
    {
        let inner: &dyn Module = &**self;
        let mut modules: Vec<&dyn Module> = vec![inner];
        modules.extend(inner.children());
        modules
    }

    /// Rooted at the inner module, for the same reason as [`Module::modules`].
    fn named_modules(&self) -> Vec<(String, &dyn Module)>
    where
        Self: Sized,
    {
        let inner: &dyn Module = &**self;
        let mut modules: Vec<(String, &dyn Module)> = vec![(String::new(), inner)];
        modules.extend(inner.named_children());
        modules
    }
}

/// Implementation for mutable references to boxed trait objects
impl Module for &mut Box<dyn Module> {
    forward_all_module_methods!();

    /// Delegates to the `Box<dyn Module>` impl: `**self` *is* a `Box`, which is
    /// `Sized`, so unlike the boxed case this one can simply forward.
    fn modules(&self) -> Vec<&dyn Module>
    where
        Self: Sized,
    {
        (**self).modules()
    }

    /// Delegates to the `Box<dyn Module>` impl, as [`Module::modules`] does.
    fn named_modules(&self) -> Vec<(String, &dyn Module)>
    where
        Self: Sized,
    {
        (**self).named_modules()
    }
}
