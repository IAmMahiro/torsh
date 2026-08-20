//! Updated for ToRSh 0.2.0
//! 
//! # Tutorial 03: Neural Networks
//! 
//! This is the third tutorial in the ToRSh learning series.
//! Learn how to build, train, and use neural networks with ToRSh.
//! 
//! ## What you'll learn:
//! - Creating neural network layers (Linear, Activation)
//! - Forward and backward passes
//! - Loss functions and optimization
//! - Training a complete neural network
//! - Making predictions with trained models
//! 
//! ## Prerequisites:
//! - Complete Tutorial 01: Tensor Basics
//! - Complete Tutorial 02: Autograd Basics
//! - Understanding of basic machine learning concepts
//! 
//! Run with: `cargo run --example 03_neural_networks`
 

use std::result::Result as StdResult;
use std::error::Error;

use torsh::F::CustomLoss;
use torsh::prelude::*;
use torsh::{Tensor};
use torsh::nn::{Module, layers::{Linear, ReLU}, functional::MSELoss};
use torsh_optim::adam::AdamBuilder;

fn main() -> StdResult<(), Box<dyn Error>> {
    println!("=== ToRSh Tutorial 03: Neural Networks ===\n");
    
    // 1. Understanding neural network layers
    println!("1. Neural Network Layers");
    println!("========================");
    
    // Create a simple linear layer: y = Wx + b
    let linear_layer = Linear::new(3, 2, true); // 3 inputs, 2 outputs
    
    println!("Linear layer: 3 inputs → 2 outputs");
    if let Some(weight) = linear_layer.all_named_parameters().get("weight") {
        println!("Weight shape: {:?}", weight.shape()?);
    }
    if let Some(bias) = linear_layer.all_named_parameters().get("bias")  {
        println!("Bias shape: {:?}", bias.shape()?);
    }
    
    // Forward pass through the layer
    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3])?; // Batch size 1, 3 features
    let output = linear_layer.forward(&input)?;
    
    println!("Input shape: {:?}", input.shape());
    println!("Output shape: {:?}", output.shape());
    println!("Input: {:?}", input.data()?);
    println!("Output: {:?}\n", output.data()?);
    
    // 2. Activation functions
    println!("2. Activation Functions");
    println!("=======================");
    
    let relu = ReLU::new();
    let test_input = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5])?;
    let relu_output = relu.forward(&test_input)?;
    
    println!("ReLU activation function:");
    println!("Input:  {:?}", test_input.data()?);
    println!("Output: {:?}", relu_output.data()?);
    println!("ReLU sets negative values to 0, keeps positive values unchanged\n");
    
    // Manual activation functions
    let sigmoid_input = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5])?;
    let sigmoid_output = sigmoid_input.sigmoid()?;
    
    println!("Sigmoid activation function:");
    println!("Input:  {:?}", sigmoid_input.data()?);
    println!("Output: {:?}", sigmoid_output.data()?);
    println!("Sigmoid maps values to (0, 1) range\n");
    
    // 3. Building a multi-layer network
    println!("3. Multi-Layer Neural Network");
    println!("=============================");
    
    // Create a simple 2-layer network for binary classification
    struct SimpleNet {
        layer1: Linear,
        relu: ReLU,
        layer2: Linear,
    }

    impl SimpleNet {
        fn new() -> Self {
            Self {
                layer1: Linear::new(2, 4, true), // 2 inputs, 4 hidden units
                relu: ReLU::new(),
                layer2: Linear::new(4, 1, true), // 4 hidden units, 1 output
            }
        }        
    }
    
    impl Module for  SimpleNet {
        fn forward(&self, x: &Tensor) -> Result<Tensor> {
            let h1 = self.layer1.forward(x)?;
            let h1_activated = self.relu.forward(&h1)?;
            let output = self.layer2.forward(&h1_activated)?;
            Ok(output)
        }
        
        fn named_children(&self) -> Vec<(String, &dyn Module)> {
            vec![
                ("layer1".to_string(), &self.layer1),
                ("layer2".to_string(), &self.layer2),
            ]
        }
        
    }
    
    let network = SimpleNet::new();
    
    // Test forward pass
    let test_input = Tensor::from_vec(vec![0.5, -0.3], &[1, 2])?;
    let prediction = network.forward(&test_input)?;
    
    println!("Network architecture: 2 → 4 → 1");
    println!("Test input: {:?}", test_input.data()?);
    println!("Network output: {:?}\n", prediction.data()?);
    
    // 4. Training data generation (XOR problem)
    println!("4. Training Data: XOR Problem");
    println!("=============================");
    
    // XOR truth table: A XOR B = (A AND !B) OR (!A AND B)
    let training_inputs = Tensor::from_vec(
        vec![
            0.0, 0.0,  // 0 XOR 0 = 0
            0.0, 1.0,  // 0 XOR 1 = 1
            1.0, 0.0,  // 1 XOR 0 = 1
            1.0, 1.0,  // 1 XOR 1 = 0
        ],
        &[4, 2] // 4 samples, 2 features each
    )?;
    
    let training_targets = Tensor::from_vec(
        vec![0.0, 1.0, 1.0, 0.0], // Expected XOR outputs
        &[4, 1] // 4 samples, 1 output each
    )?;
    
    println!("XOR Training Data:");
    println!("Inputs:  {:?}", training_inputs.data()?);
    println!("Targets: {:?}", training_targets.data()?);
    println!("This is a classic non-linearly separable problem\n");
    
    // 5. Loss function
    println!("5. Loss Function");
    println!("================");
    
    // Note: F::Reduction::Mean divides by numel (PyTorch "mean")
    // Use F::Reduction::BatchMean to divide by batch_size instead
    let loss_fn = MSELoss::new(F::Reduction::Mean);
    
    // Example loss calculation
    let example_predictions = Tensor::from_vec(vec![0.2, 0.8, 0.7, 0.1], &[4, 1])?;
    let example_loss = loss_fn.compute_loss(&example_predictions, &training_targets)?;
    
    println!("Mean Squared Error (MSE) Loss Function:");
    println!("Predictions: {:?}", example_predictions.data()?);
    println!("Targets:     {:?}", training_targets.data()?);
    println!("Loss:        {:?}", example_loss.data()?);
    println!("Lower loss = better predictions\n");
    
    // 6. Training loop
    println!("6. Training the Neural Network");
    println!("==============================");

    // The best way to update parameters is to use an optimizer
    // If you are not sure which one? just use AdamW. It's good for almost everything

    // First thing you need is to extract parameters for the optimizer
    let params = network.all_named_parameters().values().map(|p| {p.tensor()}).collect(); // In current version there is a bit of boilerplate you need to remember

    let mut optimizer = AdamBuilder::new().build_adamw(params); // Building an optimizer with default settings
    
    let learning_rate = 0.01;
    optimizer.set_lr(learning_rate);

    let epochs = 1000;
    
    println!("Training XOR neural network...");
    println!("Learning rate: {}", learning_rate);
    println!("Epochs: {}\n", epochs);
    
    for epoch in 0..epochs {
        // Zero gradients
        optimizer.zero_grad(); // Optimizer will manage it itself
        
        // Forward pass
        let predictions = network.forward(&training_inputs)?;
        
        // Compute loss
        let loss = loss_fn.compute_loss(&predictions, &training_targets)?;
        
        // Backward pass
        loss.backward()?;
        
        optimizer.step()?; // Optimizer will update params automatically
        
        // Print progress
        if epoch % 200 == 0 || epoch == epochs - 1 {
            let loss_val = loss.to_vec()?[0];
            println!("Epoch {}: Loss = {:.6}", epoch, loss_val);
            
            // Show current predictions
            if epoch == epochs - 1 {
                let final_predictions = network.forward(&training_inputs)?;
                println!("Final predictions: {:?}", final_predictions.data()?);
                println!("Targets:           {:?}", training_targets.data()?);
            }
        }
    }
    
    // 7. Testing the trained network
    println!("\n7. Testing the Trained Network");
    println!("==============================");

    let _no_grad = no_grad(); // To be sure gradients are not getting accumulated. (Best practice)
    
    // Test each XOR combination
    let test_cases = vec![
        (vec![0.0, 0.0], "0 XOR 0"),
        (vec![0.0, 1.0], "0 XOR 1"),
        (vec![1.0, 0.0], "1 XOR 0"),
        (vec![1.0, 1.0], "1 XOR 1"),
    ];
    
    println!("Testing XOR function:");
    for (input_vals, description) in test_cases {
        let test_input = Tensor::from_vec(input_vals.clone(), &[1, 2])?;
        let prediction = network.forward(&test_input)?;
        let pred_val = prediction.to_vec()?[0];
        let rounded = if pred_val > 0.5 { 1 } else { 0 };
        
        println!("{}: Input {:?} → Prediction {:.3} → Rounded {}",
                 description, input_vals, pred_val, rounded);
    }

    drop(_no_grad);
    
    // 8. Understanding what the network learned
    println!("\n8. Understanding the Network");
    println!("============================");
    
    // Visualize decision boundary (simplified)
    println!("The network learned to separate the XOR function by:");
    println!("1. First layer: Creates features that can distinguish the patterns");
    println!("2. ReLU activation: Introduces non-linearity (essential for XOR)");
    println!("3. Second layer: Combines features to produce final classification");
    println!("\nWithout the hidden layer and ReLU, this problem would be impossible!");
    println!("This demonstrates why deep networks can solve complex problems.\n");
    
    // 9. Saving and loading models (conceptual)
    println!("9. Model Persistence (Conceptual)");
    println!("==================================");
    println!("In practice, you would:");
    println!("✓ Save model parameters after training");
    println!("✓ Load parameters to restore a trained model");
    println!("✓ Use the model for inference on new data");
    println!("✓ Continue training from a checkpoint\n");
    
    // 10. Key concepts summary
    println!("10. Key Neural Network Concepts");
    println!("===============================");
    println!("✓ Layers: Transform input data (Linear, Convolution, etc.)");
    println!("✓ Activation Functions: Add non-linearity (ReLU, Sigmoid, Tanh)");
    println!("✓ Forward Pass: Data flows through network to produce predictions");
    println!("✓ Loss Function: Measures how wrong predictions are");
    println!("✓ Backward Pass: Computes gradients via backpropagation");
    println!("✓ Optimization: Updates parameters using gradients (SGD, Adam, etc.)");
    println!("✓ Training Loop: Repeat forward→loss→backward→update cycle");
    println!("✓ Non-linearity: Essential for learning complex patterns (like XOR)\n");
    
    println!("🎉 Congratulations! You've completed Tutorial 03: Neural Networks");
    println!("📚 Next: Run `cargo run --example 04_cnn_basics` to learn about Convolutional Neural Networks");
    
    Ok(())
}

/// Helper function for creating synthetic classification data
fn create_spiral_data(n_samples: usize, n_classes: usize) -> Result<(Tensor, Tensor)> {
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    
    for class_id in 0..n_classes {
        for i in 0..n_samples {
            let r = i as f32 / n_samples as f32;
            let t = (class_id as f32 * 4.0) + (r * 4.0) + 
                    (rand::random::<f32>() - 0.5) * 0.2;
            
            let x = r * t.cos();
            let y = r * t.sin();
            
            inputs.extend_from_slice(&[x, y]);
            targets.push(class_id as f32);
        }
    }
    
    let input_tensor = Tensor::from_vec(inputs, &[n_samples * n_classes, 2])?;
    let target_tensor = Tensor::from_vec(targets, &[n_samples * n_classes])?;
    
    Ok((input_tensor, target_tensor))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_linear_layer() {
        let layer = Linear::new(3, 2, true); // 3 inputs, 2 outputs, with bias
        let input = Tensor::ones(&[1, 3], DeviceType::Cpu).unwrap();
        let output = layer.forward(&input).unwrap();
        
        assert_eq!(output.shape().dims(), &[1, 2]);
    }
    
    #[test]
    fn test_linear_layer_no_bias() {
        let layer = Linear::new(3, 2, false); // without bias
        let input = Tensor::ones(&[1, 3], DeviceType::Cpu).unwrap();
        let output = layer.forward(&input).unwrap();
        
        assert_eq!(output.shape().dims(), &[1, 2]);
        
        // Check that bias is not in parameters
        assert!(layer.all_named_parameters().get("bias").is_none());
    }
    
    #[test]
    fn test_relu_activation() {
        let relu = ReLU::new();
        let input = Tensor::from_vec(vec![-1.0, 0.0, 1.0, -0.5, 2.0], &[5]).unwrap();
        let output = relu.forward(&input).unwrap();
        
        let output_data = output.to_vec().unwrap();
        assert_eq!(output_data[0], 0.0); // -1.0 → 0.0
        assert_eq!(output_data[1], 0.0); // 0.0 → 0.0
        assert_eq!(output_data[2], 1.0); // 1.0 → 1.0
        assert_eq!(output_data[3], 0.0); // -0.5 → 0.0
        assert_eq!(output_data[4], 2.0); // 2.0 → 2.0
    }
    
    #[test]
    fn test_mse_loss_none_reduction() {
        let loss_fn = MSELoss::new(F::Reduction::None);
        let predictions = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let targets = Tensor::from_vec(vec![1.5, 2.5, 3.5], &[3]).unwrap();
        let loss = loss_fn.compute_loss(&predictions, &targets).unwrap();
        
        // Each element: (pred - target)^2 = 0.25
        let loss_data = loss.to_vec().unwrap();
        assert_eq!(loss_data.len(), 3);
        for val in loss_data {
            assert!((val - 0.25).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_mse_loss_mean_reduction() {
        let loss_fn = MSELoss::new(F::Reduction::Mean);
        let predictions = Tensor::from_vec(vec![0.0, 2.0], &[2]).unwrap();
        let targets = Tensor::from_vec(vec![1.0, 1.0], &[2]).unwrap();
        let loss = loss_fn.compute_loss(&predictions, &targets).unwrap();
        
        // ((0-1)^2 + (2-1)^2) / 2 = (1 + 1) / 2 = 1.0
        let loss_val = loss.to_vec().unwrap()[0];
        assert!((loss_val - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_mse_loss_sum_reduction() {
        let loss_fn = MSELoss::new(F::Reduction::Sum);
        let predictions = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let targets = Tensor::from_vec(vec![2.0, 3.0, 4.0], &[3]).unwrap();
        let loss = loss_fn.compute_loss(&predictions, &targets).unwrap();
        
        // Each element: (pred - target)^2 = 1.0, sum = 3.0
        let loss_val = loss.to_vec().unwrap()[0];
        assert!((loss_val - 3.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_mse_loss_perfect_prediction() {
        let loss_fn = MSELoss::new(F::Reduction::Mean);
        let predictions = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let targets = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let loss = loss_fn.compute_loss(&predictions, &targets).unwrap();
        
        // Perfect predictions should have zero loss
        let loss_val = loss.to_vec().unwrap()[0];
        assert!(loss_val < 1e-6);
    }
    
    #[test]
    fn test_simple_network_forward() {
        struct TestNet {
            layer1: Linear,
            relu: ReLU,
            layer2: Linear,
        }
        
        impl TestNet {
            fn new() -> Self {
                Self {
                    layer1: Linear::new(2, 3, true),
                    relu: ReLU::new(),
                    layer2: Linear::new(3, 1, true),
                }
            }
        }
        
        impl Module for TestNet {
            fn forward(&self, x: &Tensor) -> Result<Tensor> {
                let h = self.layer1.forward(x)?;
                let h = self.relu.forward(&h)?;
                self.layer2.forward(&h)
            }
            
            fn named_children(&self) -> Vec<(String, &dyn Module)> {
                vec![
                    ("layer1".to_string(), &self.layer1),
                    ("layer2".to_string(), &self.layer2),
                ]
            }
        }
        
        let network = TestNet::new();
        let input = Tensor::from_vec(vec![1.0, 2.0], &[1, 2]).unwrap();
        let output = network.forward(&input).unwrap();
        
        assert_eq!(output.shape().dims(), &[1, 1]);
    }
}