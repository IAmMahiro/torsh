//! Updated for ToRSh 0.2.0
//! 
//! # Tutorial 02: Automatic Differentiation (Autograd) Basics
//! 
//! This is the second tutorial in the ToRSh learning series.
//! Learn how automatic differentiation works and how to compute gradients.
//! 
//! ## What you'll learn:
//! - Enabling gradient computation on tensors
//! - Computing gradients with backward()
//! - Understanding the computational graph
//! - Gradient accumulation and zeroing
//! - Using gradients for optimization
//! 
//! ## Prerequisites:
//! - Complete Tutorial 01: Tensor Basics
//! - Basic understanding of derivatives (calculus)
//! 
//! Run with: `cargo run --example 02_autograd_basics`

use std::result::Result as StdResult;
use std::error::Error;

use torsh::prelude::*;


fn main() -> StdResult<(), Box<dyn Error>> {
    println!("=== ToRSh Tutorial 02: Automatic Differentiation Basics ===\n");
    
    // 1. Enabling gradient computation
    println!("1. Enabling Gradient Computation");
    println!("================================");
    
    // Create a tensor that requires gradients
    // Note: requires_grad_() returns a new Tensor, so we reassign rather than mutate
    let x = Tensor::from_vec(vec![2.0], &[1])?.requires_grad_(true);

    println!("Input tensor x = {:?}", x.data()?);
    println!("Requires gradient: {}\n", x.requires_grad());
    
    // Simple function: y = x^2
    let y = &x * &x;
    println!("Function: y = x^2");
    println!("y = {:?}", y.data()?);  // use .data()? to see tensor values
    println!("y requires gradient: {}\n", y.requires_grad()); // it does!
    
    // Compute gradient dy/dx = 2x
    y.backward()?;
    
    if let Some(grad) = x.grad() {
        println!("Gradient dy/dx = {:?}", grad.data()?);
        println!("Expected: 2 * x = 2 * 2 = 4 ✓\n");
    }
    
    // 2. More complex functions
    println!("2. More Complex Functions");
    println!("=========================");
    
    // Reset gradients for new computation
    let a = Tensor::from_vec(vec![3.0], &[1])?.requires_grad_(true);
    let b = Tensor::from_vec(vec![4.0], &[1])?.requires_grad_(true);
    
    println!("a = {:?}\nb = {:?}", a.data()?, b.data()?);
    
    // Function: z = a^2 + 2*a*b + b^2 = (a + b)^2
    let a_squared = &a * &a;
    let b_squared = &b * &b;
    let ab_term = (&a * &b).mul_scalar(2.0)?;
    let z = &(&a_squared + &ab_term) + &b_squared;
    
    println!("Function: z = a^2 + 2*a*b + b^2");
    println!("z = {:?}\n", z.data()?);
    
    z.backward()?;
    
    if let Some(grad_a) = a.grad() {
        println!("∂z/∂a = {:?}", grad_a.data()?);
        println!("Expected: 2a + 2b = 2*3 + 2*4 = 14 ✓");
    }
    
    if let Some(grad_b) = b.grad() {
        println!("∂z/∂b = {:?}", grad_b.data()?);
        println!("Expected: 2a + 2b = 2*3 + 2*4 = 14 ✓\n");
    }
    
    // 3. Vector functions and Jacobians
    println!("3. Vector Functions");
    println!("===================");
    
    let vec_x = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?.requires_grad_(true);
    
    println!("Input vector x = {:?}", vec_x.data()?);
    
    // Function: f(x) = x^2 (element-wise)
    let f_x = &vec_x * &vec_x;
    println!("Function: f(x) = x^2 (element-wise)");
    println!("f(x) = {:?}", f_x.data()?);
    
    // To compute gradients for vector functions, we need to specify which output
    // element we want to differentiate. Let's sum all outputs first.
    let scalar_output = f_x.sum()?;
    println!("Sum of f(x) = {:?}\n", scalar_output.data()?);
    
    scalar_output.backward()?;
    
    if let Some(grad) = vec_x.grad() {
        println!("Gradient: {:?}", grad.data()?);
        println!("Expected: [2*1, 2*2, 2*3] = [2, 4, 6] ✓\n");
    }
    
    // 4. Gradient accumulation
    println!("4. Gradient Accumulation");
    println!("========================");
    
    let mut param = Tensor::from_vec(vec![1.0], &[1])?.requires_grad_(true);
    
    println!("Parameter = {:?}\n", param.data()?);
    
    // First computation: y1 = param^2
    let y1 = &param * &param;
    y1.backward()?;
    
    if let Some(grad) = param.grad() {
        println!("After first backward (y1 = param^2):");
        println!("Gradient = {:?} (should be 2*1 = 2)", grad.data()?);
    }
    
    // Second computation without zeroing gradients: y2 = param^3
    let y2 = &(&param * &param) * &param;
    y2.backward()?;
    
    if let Some(grad) = param.grad() {
        println!("After second backward (y2 = param^3, accumulated):");
        println!("Gradient = {:?} (should be 2 + 3*1^2 = 5)", grad.data()?);
    }
    
    // Zero gradients and compute again
    param.zero_grad();
    let y3 = &(&param * &param) * &param;
    y3.backward()?;
    
    if let Some(grad) = param.grad() {
        println!("After zeroing gradients and computing y3 = param^3:");
        println!("Gradient = {:?} (should be 3*1^2 = 3)\n", grad.data()?);
    }
    
    // 5. Chain rule in action
    println!("5. Chain Rule in Action");
    println!("=======================");
    
    let input = Tensor::from_vec(vec![0.5], &[1])?.requires_grad_(true);
    
    println!("Input = {:?}", input.data()?);
    
    // Multi-step computation: final = sin(x^2 + 1)
    let step1 = &input * &input;                            // x^2
    let step2 = step1.add_scalar(1.0)?;                     // x^2 + 1
    let final_result = step2.sin()?;                        // sin(x^2 + 1)
    
    println!("Step 1: x^2 = {:?}", step1.data()?);
    println!("Step 2: x^2 + 1 = {:?}", step2.data()?);
    println!("Final: sin(x^2 + 1) = {:?}\n", final_result.data()?);
    
    final_result.backward()?;
    
    if let Some(grad) = input.grad() {
        println!("Gradient d/dx[sin(x^2 + 1)] = {:?}", grad.data()?);
        // Chain rule: d/dx[sin(x^2 + 1)] = cos(x^2 + 1) * 2x
        let x_val: f64 = 0.5;
        let expected = (x_val * x_val + 1.0).cos() * 2.0 * x_val;
        println!("Expected: cos(x² + 1) * 2x = cos({:.3}) * {:.1} = {:.6}", 
                 x_val * x_val + 1.0, 2.0 * x_val, expected);
    }
    
    // 6. Practical example: Simple linear regression
    println!("\n6. Practical Example: Simple Linear Regression");
    println!("===============================================");
    
    // Training data: y = 2x + 1 + noise
    let x_data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5])?;
    let y_data = Tensor::from_vec(vec![3.1, 4.9, 7.2, 9.1, 10.8], &[5])?;
    
    // Parameters to learn
    let mut weight = Tensor::from_vec(vec![0.0], &[1])?.requires_grad_(true);
    let mut bias = Tensor::from_vec(vec![0.0], &[1])?.requires_grad_(true);
    
    println!("Training data:");
    println!("X: {:?}", x_data.data()?);
    println!("Y: {:?}\n", y_data.data()?);
    
    let learning_rate = 0.01;
    let epochs = 100;
    
    println!("Training simple linear regression (y = wx + b):");
    println!("Initial parameters: w = {:.3}, b = {:.3}", 
             weight.item()?, bias.item()?);
    
    for epoch in 0..epochs {
        // Zero gradients
        weight.zero_grad();
        bias.zero_grad();
        
        // Forward pass: predictions = weight * x_data + bias
        let predictions = &(&weight * &x_data) + &bias;
        
        // Compute loss: Mean Squared Error
        let diff = &predictions - &y_data;
        let squared_diff = &diff * &diff;
        let loss = squared_diff.mean(None, false)?;
        
        // Backward pass
        loss.backward()?;
        
        // Update parameters using gradients
        if let (Some(w_grad), Some(b_grad)) = (weight.grad(), bias.grad()) {
            // w = w - learning_rate * gradient
            // For single-element tensors, .item() is cleaner than .to_vec()?[0]
            let w_val = weight.item()?;
            let b_val = bias.item()?;
            let w_grad_val = w_grad.item()?;
            let b_grad_val = b_grad.item()?;
            
            let new_w = w_val - learning_rate * w_grad_val;
            let new_b = b_val - learning_rate * b_grad_val;

            // Note that here we create new tensors for clarity.
            // In production we do optimizer.step(), which we'll see in action in the next tutorial

            weight = Tensor::from_vec(vec![new_w], &[1])?.requires_grad_(true);
            bias = Tensor::from_vec(vec![new_b], &[1])?.requires_grad_(true);
        }
        
        // Print progress
        if epoch % 20 == 0 || epoch == epochs - 1 {
            println!("Epoch {}: Loss = {:.6}, w = {:.3}, b = {:.3}", 
                     epoch, loss.item()?, weight.item()?, bias.item()?);
        }
    }
    
    println!("\n✅ Training completed!");
    println!("Final parameters: w = {:.3}, b = {:.3}", 
             weight.item()?, bias.item()?);
    println!("Target parameters: w = 2.0, b = 1.0");
    println!("(Difference due to noise in training data)\n");
    
    // 7. Key concepts summary
    println!("7. Key Concepts Summary");
    println!("=======================");
    println!("✓ requires_grad_(true): Enables gradient computation for a tensor");
    println!("✓ backward(): Computes gradients via backpropagation");
    println!("✓ grad(): Access computed gradients");
    println!("✓ zero_grad(): Reset gradients to zero (important for training loops)");
    println!("✓ Chain rule: Automatic differentiation handles complex function compositions");
    println!("✓ Gradient accumulation: Gradients add up across multiple backward() calls");
    println!("✓ Optimization: Use gradients to update parameters (gradient descent)");
    println!("✓ item(): Extract scalar value from single-element tensor\n");
    
    
    println!("🎉 Congratulations! You've completed Tutorial 02: Autograd Basics");
    println!("📚 Next: Run `cargo run --example 03_neural_networks` to learn about neural networks");
    

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simple_gradient() {
        let x = Tensor::from_vec(vec![2.0], &[1]).unwrap().requires_grad_(true);
        
        let y = &x * &x; // y = x^2
        y.backward().unwrap();
        
        if let Some(grad) = x.grad() {
            let grad_val = grad.item::<f32>().unwrap();
            assert!((grad_val - 4.0).abs() < 1e-6); // dy/dx = 2x = 2*2 = 4
        }
    }
    
    #[test]
    fn test_gradient_accumulation() {
        let x = Tensor::from_vec(vec![1.0], &[1]).unwrap().requires_grad_(true);
        
        // First computation
        let y1 = &x * &x;
        y1.backward().unwrap();
        
        // Second computation (gradients should accumulate)
        let y2 = x.mul_scalar(3.0).unwrap();
        y2.backward().unwrap();
        
        if let Some(grad) = x.grad() {
            let grad_val = grad.item::<f32>().unwrap();
            // Should be 2*1 (from x^2) + 3 (from 3*x) = 5
            assert!((grad_val - 5.0).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_zero_grad() {
        let x = Tensor::from_vec(vec![1.0], &[1]).unwrap().requires_grad_(true);
        
        // First computation
        let y1 = &x * &x;
        y1.backward().unwrap();
        
        // Zero gradients
        x.zero_grad();
        
        // Second computation
        let y2 = x.mul_scalar(3.0).unwrap();
        y2.backward().unwrap();
        
        if let Some(grad) = x.grad() {
            let grad_val = grad.item::<f32>().unwrap();
            // Should be only 3 (from 3*x), not accumulated
            assert!((grad_val - 3.0).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_chain_rule() {
        let x = Tensor::from_vec(vec![0.5], &[1]).unwrap().requires_grad_(true);
        
        // sin(x^2 + 1)
        let step1 = &x * &x;
        let step2 = step1.add_scalar(1.0).unwrap();
        let result = step2.sin().unwrap();
        
        result.backward().unwrap();
        
        if let Some(grad) = x.grad() {
            let grad_val = grad.item::<f32>().unwrap();
            // Expected: cos(x^2 + 1) * 2x
            let x_val: f32 = 0.5;
            let expected = (x_val * x_val + 1.0).cos() * 2.0 * x_val;
            assert!((grad_val - expected).abs() < 1e-5);
        }
    }
}