//! Gradient-based hyperparameter optimization for automatic tuning of learning rates,
//! regularization parameters, and other hyperparameters using differentiation.

use crate::context::AutogradContext;
use std::collections::HashMap;
use torsh_core::{Result, TorshError};
use torsh_tensor::Tensor;

/// Configuration for hyperparameter optimization
#[derive(Debug, Clone)]
pub struct HyperparameterConfig {
    /// Learning rate for hyperparameter updates
    pub meta_learning_rate: f64,
    /// Maximum number of optimization steps
    pub max_steps: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Whether to use second-order gradients
    pub second_order: bool,
    /// Validation frequency (steps)
    pub validation_frequency: usize,
    /// Early stopping patience
    pub early_stopping_patience: usize,
}

impl Default for HyperparameterConfig {
    fn default() -> Self {
        Self {
            meta_learning_rate: 0.01,
            max_steps: 1000,
            tolerance: 1e-6,
            second_order: true,
            validation_frequency: 10,
            early_stopping_patience: 50,
        }
    }
}

/// Hyperparameter that can be optimized
#[derive(Debug, Clone)]
pub struct OptimizableHyperparameter {
    /// Current value of the hyperparameter
    pub value: Tensor,
    /// Name/identifier for the hyperparameter
    pub name: String,
    /// Lower bound for the parameter
    pub lower_bound: Option<f64>,
    /// Upper bound for the parameter
    pub upper_bound: Option<f64>,
    /// Whether to use log scale (e.g., for learning rates)
    pub log_scale: bool,
}

impl OptimizableHyperparameter {
    /// Create a new optimizable hyperparameter
    pub fn new(
        name: String,
        initial_value: f64,
        lower_bound: Option<f64>,
        upper_bound: Option<f64>,
        log_scale: bool,
    ) -> Result<Self> {
        let value = if log_scale {
            Tensor::scalar(initial_value.ln() as f32)
        } else {
            Tensor::scalar(initial_value as f32)
        }?;

        Ok(Self {
            value,
            name,
            lower_bound,
            upper_bound,
            log_scale,
        })
    }

    /// Get the actual hyperparameter value (handling log scale)
    pub fn get_value(&self) -> Result<f64> {
        let raw_value = self.value.item()? as f64;
        if self.log_scale {
            Ok(raw_value.exp())
        } else {
            Ok(raw_value)
        }
    }

    /// Apply bounds and constraints to the hyperparameter
    pub fn apply_constraints(&mut self) -> Result<()> {
        let mut value = self.value.item()? as f64;

        // Apply bounds
        if let Some(lower) = self.lower_bound {
            let bound = if self.log_scale { lower.ln() } else { lower };
            value = value.max(bound);
        }

        if let Some(upper) = self.upper_bound {
            let bound = if self.log_scale { upper.ln() } else { upper };
            value = value.min(bound);
        }

        self.value = Tensor::scalar(value as f32)?;
        Ok(())
    }
}

/// Gradient-based hyperparameter optimizer
pub struct HyperparameterOptimizer {
    config: HyperparameterConfig,
    hyperparameters: HashMap<String, OptimizableHyperparameter>,
    #[allow(dead_code)]
    context: AutogradContext,
    step_count: usize,
    best_validation_loss: Option<f64>,
    patience_counter: usize,
}

impl HyperparameterOptimizer {
    /// Create a new hyperparameter optimizer
    pub fn new(config: HyperparameterConfig) -> Self {
        Self {
            config,
            hyperparameters: HashMap::new(),
            context: AutogradContext::new(),
            step_count: 0,
            best_validation_loss: None,
            patience_counter: 0,
        }
    }

    /// Add a hyperparameter to optimize
    pub fn add_hyperparameter(&mut self, hyperparameter: OptimizableHyperparameter) {
        let name = hyperparameter.name.clone();
        self.hyperparameters.insert(name, hyperparameter);
    }

    /// Get hyperparameter value by name
    pub fn get_hyperparameter(&self, name: &str) -> Result<f64> {
        self.hyperparameters
            .get(name)
            .ok_or_else(|| {
                TorshError::AutogradError(format!("Hyperparameter '{}' not found", name))
            })?
            .get_value()
    }

    /// Perform one step of hyperparameter optimization
    pub fn step<F>(&mut self, objective_fn: F) -> Result<f64>
    where
        F: Fn(&HashMap<String, f64>) -> Result<Tensor>,
    {
        // Get current hyperparameter values
        let mut current_values = HashMap::new();
        for (name, hyperparam) in &self.hyperparameters {
            current_values.insert(name.clone(), hyperparam.get_value()?);
        }

        // Compute objective and gradients
        let objective = objective_fn(&current_values)?;
        let objective_value = objective.item()? as f64;

        // Compute gradients with respect to hyperparameters
        let gradients =
            self.compute_hyperparameter_gradients(&objective, &objective_fn, &current_values)?;

        // Update hyperparameters using gradients
        self.update_hyperparameters(&gradients)?;

        // Apply constraints
        for hyperparam in self.hyperparameters.values_mut() {
            hyperparam.apply_constraints()?;
        }

        self.step_count += 1;
        Ok(objective_value)
    }

    /// Optimize hyperparameters using validation loss
    pub fn optimize<F, V>(
        &mut self,
        objective_fn: F,
        validation_fn: V,
    ) -> Result<HyperparameterOptimizationResult>
    where
        F: Fn(&HashMap<String, f64>) -> Result<Tensor>,
        V: Fn(&HashMap<String, f64>) -> Result<f64>,
    {
        let mut history = Vec::new();
        let mut converged = false;

        for step in 0..self.config.max_steps {
            // Perform optimization step
            let train_loss = self.step(&objective_fn)?;

            // Validate periodically
            let mut validation_loss = None;
            if step % self.config.validation_frequency == 0 {
                let current_values = self.get_current_values()?;
                let val_loss = validation_fn(&current_values)?;
                validation_loss = Some(val_loss);

                // Early stopping check
                if let Some(best_loss) = self.best_validation_loss {
                    if val_loss < best_loss - self.config.tolerance {
                        self.best_validation_loss = Some(val_loss);
                        self.patience_counter = 0;
                    } else {
                        self.patience_counter += 1;
                        if self.patience_counter >= self.config.early_stopping_patience {
                            converged = true;
                        }
                    }
                } else {
                    self.best_validation_loss = Some(val_loss);
                }
            }

            // Record history
            history.push(OptimizationStep {
                step,
                train_loss,
                validation_loss,
                hyperparameters: self.get_current_values()?,
            });

            // Check convergence
            if converged {
                break;
            }
        }

        Ok(HyperparameterOptimizationResult {
            converged,
            final_hyperparameters: self.get_current_values()?,
            best_validation_loss: self.best_validation_loss,
            history,
        })
    }

    /// Compute gradients with respect to hyperparameters
    fn compute_hyperparameter_gradients<F>(
        &self,
        objective: &Tensor,
        objective_fn: &F,
        current_values: &HashMap<String, f64>,
    ) -> Result<HashMap<String, Tensor>>
    where
        F: Fn(&HashMap<String, f64>) -> Result<Tensor>,
    {
        let mut gradients = HashMap::new();

        for (name, hyperparam) in &self.hyperparameters {
            let grad = if self.config.second_order {
                // Second-order gradient computation (Hessian-vector product).
                // Not yet implemented; tracked separately from this fix (see
                // `compute_second_order_gradient` and `higher_order_gradients.rs`).
                let param_with_grad = hyperparam.value.clone();
                self.compute_second_order_gradient(objective, &param_with_grad)?
            } else {
                // First-order gradient via central finite differences.
                self.compute_first_order_gradient(objective_fn, current_values, name, hyperparam)?
            };

            gradients.insert(name.clone(), grad);
        }

        Ok(gradients)
    }

    /// Compute the first-order gradient of the objective with respect to a
    /// single hyperparameter using central finite differences.
    ///
    /// `objective_fn` only exposes the objective as a black-box function of
    /// concrete hyperparameter values (`&HashMap<String, f64> -> Tensor`): the
    /// values it receives are plain `f64`s extracted via
    /// `OptimizableHyperparameter::get_value`, fully detached from any
    /// computation graph. That means there is no recorded tape for
    /// `self.context` (`AutogradContext`) -- or for the tensor's own
    /// `requires_grad` tape -- to run reverse-mode differentiation through, no
    /// matter how the objective is invoked. Central finite differences is the
    /// principled technique for differentiating exactly this kind of
    /// black-box scalar objective; it is not a placeholder, it is the same
    /// numerical method this crate already trusts as a reference oracle for
    /// gradient checking elsewhere (see `gradient_checking.rs`).
    ///
    /// The difference is taken on the *raw* hyperparameter tensor value
    /// (`hyperparam.value`, pre-`exp` for log-scale parameters) because that
    /// is the quantity `update_hyperparameters` actually updates. For
    /// log-scale hyperparameters we re-apply the same `exp` transform used by
    /// `OptimizableHyperparameter::get_value` before invoking `objective_fn`,
    /// so the chain rule through the log transform falls out automatically.
    fn compute_first_order_gradient<F>(
        &self,
        objective_fn: &F,
        current_values: &HashMap<String, f64>,
        name: &str,
        hyperparam: &OptimizableHyperparameter,
    ) -> Result<Tensor>
    where
        F: Fn(&HashMap<String, f64>) -> Result<Tensor>,
    {
        let raw_value = hyperparam.value.item()? as f64;
        let step = Self::finite_difference_step(raw_value);

        let evaluate_at = |raw: f64| -> Result<f64> {
            let actual_value = if hyperparam.log_scale { raw.exp() } else { raw };
            let mut perturbed = current_values.clone();
            perturbed.insert(name.to_string(), actual_value);
            Ok(objective_fn(&perturbed)?.item()? as f64)
        };

        let objective_plus = evaluate_at(raw_value + step)?;
        let objective_minus = evaluate_at(raw_value - step)?;
        let gradient = (objective_plus - objective_minus) / (2.0 * step);

        Tensor::scalar(gradient as f32)
    }

    /// Adaptive step size for central-difference numerical differentiation.
    ///
    /// Uses a relative step (scaled by the magnitude of the evaluation point,
    /// floored at `1.0` so the step stays well-defined near zero) sized to
    /// `f32::EPSILON.cbrt()`. Hyperparameter values round-trip through
    /// `f32`-backed `Tensor`s here, so a step much smaller than that would be
    /// swallowed by `f32` rounding noise when `objective_fn` casts its result
    /// down to `f32`, while a much larger step would introduce unnecessary
    /// truncation error for non-quadratic objectives.
    fn finite_difference_step(x: f64) -> f64 {
        (f32::EPSILON as f64).cbrt() * x.abs().max(1.0)
    }

    /// Compute second-order gradient (for more accurate optimization)
    fn compute_second_order_gradient(
        &self,
        _objective: &Tensor,
        parameter: &Tensor,
    ) -> Result<Tensor> {
        // TODO: Implement second-order gradient computation when autograd API is available
        // For now, return zeros as placeholder
        Ok(Tensor::zeros_like(parameter)?)
    }

    /// Update hyperparameters using computed gradients
    fn update_hyperparameters(&mut self, gradients: &HashMap<String, Tensor>) -> Result<()> {
        for (name, gradient) in gradients {
            if let Some(hyperparam) = self.hyperparameters.get_mut(name) {
                let grad_value = gradient.item()? as f64;
                let current_value = hyperparam.value.item()? as f64;

                // Gradient ascent (we want to maximize the objective, which is typically negative loss)
                let new_value =
                    current_value - (self.config.meta_learning_rate as f64 * grad_value);
                hyperparam.value = Tensor::scalar(new_value as f32)?;
            }
        }
        Ok(())
    }

    /// Get current hyperparameter values
    fn get_current_values(&self) -> Result<HashMap<String, f64>> {
        let mut values = HashMap::new();
        for (name, hyperparam) in &self.hyperparameters {
            values.insert(name.clone(), hyperparam.get_value()?);
        }
        Ok(values)
    }
}

/// Result of hyperparameter optimization
#[derive(Debug, Clone)]
pub struct HyperparameterOptimizationResult {
    /// Whether optimization converged
    pub converged: bool,
    /// Final optimized hyperparameters
    pub final_hyperparameters: HashMap<String, f64>,
    /// Best validation loss achieved
    pub best_validation_loss: Option<f64>,
    /// Optimization history
    pub history: Vec<OptimizationStep>,
}

/// Single step in optimization history
#[derive(Debug, Clone)]
pub struct OptimizationStep {
    /// Step number
    pub step: usize,
    /// Training loss at this step
    pub train_loss: f64,
    /// Validation loss at this step (if computed)
    pub validation_loss: Option<f64>,
    /// Hyperparameter values at this step
    pub hyperparameters: HashMap<String, f64>,
}

/// Convenience functions for common hyperparameter optimization scenarios
impl HyperparameterOptimizer {
    /// Create optimizer for learning rate optimization
    pub fn for_learning_rate(
        initial_lr: f64,
        config: Option<HyperparameterConfig>,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();
        let mut optimizer = Self::new(config);

        let lr_param = OptimizableHyperparameter::new(
            "learning_rate".to_string(),
            initial_lr,
            Some(1e-6), // Lower bound
            Some(1.0),  // Upper bound
            true,       // Log scale
        )?;

        optimizer.add_hyperparameter(lr_param);
        Ok(optimizer)
    }

    /// Create optimizer for regularization strength
    pub fn for_regularization(
        initial_reg: f64,
        config: Option<HyperparameterConfig>,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();
        let mut optimizer = Self::new(config);

        let reg_param = OptimizableHyperparameter::new(
            "regularization".to_string(),
            initial_reg,
            Some(0.0), // Lower bound
            Some(1.0), // Upper bound
            true,      // Log scale
        )?;

        optimizer.add_hyperparameter(reg_param);
        Ok(optimizer)
    }

    /// Create optimizer for multiple hyperparameters
    pub fn for_multiple_params(
        params: Vec<(&str, f64, Option<f64>, Option<f64>, bool)>,
        config: Option<HyperparameterConfig>,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();
        let mut optimizer = Self::new(config);

        for (name, initial_value, lower_bound, upper_bound, log_scale) in params {
            let param = OptimizableHyperparameter::new(
                name.to_string(),
                initial_value,
                lower_bound,
                upper_bound,
                log_scale,
            )?;
            optimizer.add_hyperparameter(param);
        }

        Ok(optimizer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizable_hyperparameter_creation() {
        let param = OptimizableHyperparameter::new(
            "learning_rate".to_string(),
            0.01,
            Some(1e-6),
            Some(1.0),
            true,
        )
        .unwrap();

        assert_eq!(param.name, "learning_rate");
        assert!((param.get_value().unwrap() - 0.01).abs() < 1e-6);
        assert_eq!(param.lower_bound, Some(1e-6));
        assert_eq!(param.upper_bound, Some(1.0));
        assert!(param.log_scale);
    }

    #[test]
    fn test_hyperparameter_bounds() {
        let mut param = OptimizableHyperparameter::new(
            "test".to_string(),
            10.0, // Initial value too high
            Some(1e-6),
            Some(1.0),
            false,
        )
        .unwrap();

        param.apply_constraints().unwrap();
        assert!((param.get_value().unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_optimizer_creation() {
        let config = HyperparameterConfig::default();
        let optimizer = HyperparameterOptimizer::new(config);
        assert_eq!(optimizer.step_count, 0);
    }

    #[test]
    fn test_learning_rate_optimizer() {
        let optimizer = HyperparameterOptimizer::for_learning_rate(0.01, None).unwrap();
        let lr = optimizer.get_hyperparameter("learning_rate").unwrap();
        assert!((lr - 0.01).abs() < 1e-6);
    }

    /// Regression test for the "no-op optimizer" bug: `compute_first_order_gradient`
    /// used to unconditionally return `Tensor::zeros_like(...)`, so gradient
    /// descent never moved the hyperparameter no matter how far it was from the
    /// optimum. This must fail against that stub (the parameter never moves,
    /// so `x_after == x_before` and neither assertion below can hold) and pass
    /// once `compute_first_order_gradient` returns a real gradient.
    #[test]
    fn test_first_order_gradient_moves_toward_minimum() {
        // Minimize f(x) = (x - 5)^2, whose unique minimum is x = 5.
        fn objective(values: &HashMap<String, f64>) -> Result<Tensor> {
            let x = values["x"];
            Tensor::scalar(((x - 5.0) * (x - 5.0)) as f32)
        }

        let config = HyperparameterConfig {
            meta_learning_rate: 0.1,
            max_steps: 1,
            tolerance: 1e-6,
            second_order: false,
            validation_frequency: 1,
            early_stopping_patience: 50,
        };

        let mut optimizer = HyperparameterOptimizer::new(config);
        optimizer.add_hyperparameter(
            OptimizableHyperparameter::new("x".to_string(), 0.0, None, None, false).unwrap(),
        );

        let x_before = optimizer.get_hyperparameter("x").unwrap();
        assert!((x_before - 0.0).abs() < 1e-6);

        for _ in 0..20 {
            optimizer.step(objective).unwrap();
        }

        let x_after = optimizer.get_hyperparameter("x").unwrap();

        // The parameter must have moved strictly closer to the true minimum.
        assert!(
            (x_after - 5.0).abs() < (x_before - 5.0).abs(),
            "expected x to move toward the minimum at 5.0 (from {x_before}), got {x_after}"
        );
        // 20 gradient-descent steps on a well-conditioned convex quadratic
        // should land close to the minimum, not just move a little.
        assert!(
            (x_after - 5.0).abs() < 0.5,
            "expected x to converge near the minimum 5.0 after 20 steps, got {x_after}"
        );
    }

    /// `compute_first_order_gradient` is central-difference exact for a
    /// quadratic objective, so its output should closely match the true
    /// analytic gradient df/dx = 2(x - 5), not just be "non-zero".
    #[test]
    fn test_first_order_gradient_matches_analytic_gradient() {
        fn objective(values: &HashMap<String, f64>) -> Result<Tensor> {
            let x = values["x"];
            Tensor::scalar(((x - 5.0) * (x - 5.0)) as f32)
        }

        let config = HyperparameterConfig {
            second_order: false,
            ..HyperparameterConfig::default()
        };
        let mut optimizer = HyperparameterOptimizer::new(config);
        optimizer.add_hyperparameter(
            OptimizableHyperparameter::new("x".to_string(), 0.0, None, None, false).unwrap(),
        );

        let current_values = optimizer.get_current_values().unwrap();
        let objective_tensor = objective(&current_values).unwrap();
        let gradients = optimizer
            .compute_hyperparameter_gradients(&objective_tensor, &objective, &current_values)
            .unwrap();

        let grad_x = gradients["x"].item().unwrap() as f64;

        // Analytic gradient at x = 0 is 2 * (0 - 5) = -10.
        assert!(
            (grad_x - (-10.0)).abs() < 1e-2,
            "expected gradient close to -10.0, got {grad_x}"
        );
        assert!(grad_x.abs() > 1e-6, "gradient must not be (near-)zero");
    }
}
