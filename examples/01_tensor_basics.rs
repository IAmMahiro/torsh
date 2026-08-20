//! Updated for ToRSh 0.2.0
//! 
//! # Tutorial 01: Tensor Basics
//! 
//! This is the first tutorial in the ToRSh learning series.
//! Learn the fundamental concepts of tensors, shapes, and basic operations.
//! 
//! ## What you'll learn:
//! - Creating tensors from data
//! - Understanding tensor shapes and dimensions
//! - Basic tensor operations (add, multiply, reshape)
//! - Device management (CPU/GPU)
//! - Tensor indexing, slicing, and transposing
//! - Reduction operations and broadcasting
//! 
//! ## Prerequisites:
//! - Basic Rust knowledge
//! - No prior ML/tensor experience needed
//! 
//! Run with: `cargo run --example 01_tensor_basics`

use std::result::Result as StdResult;
use std::error::Error;

use torsh::core::device::implementations::CudaDevice;
use torsh::prelude::*;
use torsh::{Tensor, DeviceType};


fn main() -> StdResult<(), Box<dyn Error>> {
    println!("=== ToRSh Tutorial 01: Tensor Basics ===\n");
    
    // 1. Creating your first tensor
    println!("1. Creating Tensors");
    println!("==================");
    
    // Create a 1D tensor from a vector
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let tensor1d = Tensor::from_vec(data, &[5])?;
    println!("1D tensor: {:?}", tensor1d); // Printing out a Tensor will not show you its data. It will show you the shape, dtype and a device.
    println!("1D tensor data: {:?}", tensor1d.data()?); // use .data()? (will return Vec<T>) or .as_ndarray()? (will return ArrayBase) to see what's inside
    println!("Shape: {:?}", tensor1d.shape());
    println!("Number of elements: {}\n", tensor1d.numel());

    // Create a 2D tensor (matrix)
    let matrix_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor2d = Tensor::from_vec(matrix_data, &[2, 3])?; // 2 rows, 3 columns
    println!("2D tensor (2x3 matrix):");
    println!("{:?}", tensor2d.as_ndarray()?);
    println!("Shape: {:?}\n", tensor2d.shape());

    // Create a 3D tensor
    let tensor3d = Tensor::from_vec(
        (1..=24).map(|x| x as f32).collect(),
        &[2, 3, 4] // 2x3x4 tensor
    )?;
    println!("3D tensor (2x3x4):");
    println!("Shape: {:?}", tensor3d.shape());
    println!("Total elements: {}\n", tensor3d.numel());
    
    // 2. Tensor creation utilities
    println!("2. Tensor Creation Utilities");
    println!("============================");
    
    // Create tensors filled with specific values
    let zeros_tensor: Tensor<f64> = Tensor::zeros(&[3, 3], DeviceType::Cpu)?;  // Or let zeros_tensor: Tensor<f64> = zeros(&[3, 3])?;
    println!("3x3 tensor of zeros:");
    println!("{:?}\n", zeros_tensor.as_ndarray()?);
    
    let ones_tensor: Tensor<f64> = Tensor::ones(&[2, 4], DeviceType::Cpu)?; // Or let ones_tensor: Tensor<f64> = ones(&[2, 4])?;
    println!("2x4 tensor of ones:");
    println!("{:?}\n", ones_tensor.as_ndarray()?);
    
    // Create tensor with random values
    let random_tensor: Tensor<f64> = randn(&[3, 3])?;
    println!("3x3 tensor with random values (normal distribution):");
    println!("{:?}\n", random_tensor.as_ndarray()?);
    
    // Create tensor with a range of values
    let range_tensor = arange(0.0, 10.0, 1.0)?;
    println!("Range tensor (0 to 9):");
    println!("{:?}\n", range_tensor.as_ndarray()?);
    
    // 3. Basic tensor operations
    println!("3. Basic Tensor Operations");
    println!("==========================");
    
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2])?;
    
    println!("Tensor A:");
    println!("{:?}", a.as_ndarray()?);
    println!("Tensor B:");
    println!("{:?}\n", b.as_ndarray()?);

    // Element-wise addition
    let sum = &a + &b;
    println!("A + B (element-wise):");
    println!("{:?}\n", sum.as_ndarray()?);
    
    // Element-wise multiplication
    let product = &a * &b;
    println!("A * B (element-wise):");
    println!("{:?}\n", product.as_ndarray()?);
    
    // Scalar operations
    let scaled = a.mul_scalar(2.0)?; // &a * 2.0 doesn't work anymore, please use scalar ops directly through Tensor functions
    println!("A * 2 (scalar multiplication):");
    println!("{:?}\n", scaled.as_ndarray()?);

    // 4. Tensor reshaping
    println!("4. Tensor Reshaping");
    println!("===================");
    
    let original = arange(0.0, 12.0, 1.0)?;
    println!("Original tensor (12 elements):");
    println!("{:?}", original.as_ndarray()?);
    println!("Shape: {:?}\n", original.shape());
    
    // Reshape to 3x4 matrix
    let reshaped_3x4 = original.reshape(&[3, 4])?;
    println!("Reshaped to 3x4:");
    println!("{:?}", reshaped_3x4.as_ndarray()?);
    println!("Shape: {:?}\n", reshaped_3x4.shape());
    
    // Reshape to 2x6 matrix
    let reshaped_2x6 = original.reshape(&[2, 6])?;
    println!("Reshaped to 2x6:");
    println!("{:?}", reshaped_2x6.as_ndarray()?);
    println!("Shape: {:?}\n", reshaped_2x6.shape());
    
    // 5. Device management
    println!("5. Device Management");
    println!("====================");

    // Create tensor on CPU (default)
    let mut tensor: Tensor<f64> = Tensor::ones(&[2, 2], DeviceType::Cpu)?;
    println!("CPU tensor device: {:?}", tensor.device());
    
    // Check available devices
    println!("Available devices:");
    println!("- CPU: Always available");

    match CudaDevice::new(0) {
        Ok(device) => {
            if device.is_available()? {
                println!("- CUDA GPU: Available");
                
                // Move tensor to GPU (if available)
                tensor = tensor.to_device(DeviceType::Cuda(0))?;
                println!("GPU tensor device: {:?}", tensor.device());
            } else {
                println!("- CUDA GPU: Not available");
            }
        },
        Err(_) => {
            println!("- CUDA GPU: Not available");
            println!("You should compile Cuda support by enabling cuda feature")
        }
    }
    
    // 6. Tensor indexing and slicing
    println!("\n6. Tensor Indexing and Slicing");
    println!("===============================");
    
    let matrix = Tensor::from_vec(
        (1..=20).map(|x| x as f32).collect(),
        &[4, 5] // 4x5 matrix
    )?;
    println!("4x5 matrix:");
    println!("{:?}\n", matrix.as_ndarray()?);
    
    // Get a single element (row 1, column 2)
    let element = matrix.get(&[1, 2])?;
    println!("Element at [1, 2]: {}", element);
    
    // Get a row (row 0)
    let row = matrix.select(0, 0)?;
    println!("First row: {:?}", row.as_ndarray()?);
    
    // Get a column (column 2)
    let col = matrix.select(1, 2)?;
    println!("Third column: {:?}\n", col.as_ndarray()?);
    
    // 7. Reduction operations
    println!("7. Reduction Operations");
    println!("=======================");
    
    let data_tensor = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3]
    )?;
    println!("Data tensor:");
    println!("{:?}\n", data_tensor.as_ndarray()?);
    
    // Sum all elements
    let total_sum = data_tensor.sum()?;
    println!("Sum of all elements: {}", total_sum.as_ndarray()?);
    
    // Mean of all elements
    let mean = data_tensor.mean(None, false)?;
    println!("Mean of all elements: {}", mean.as_ndarray()?);
    
    // Maximum value
    let max_val = data_tensor.max(None, false)?;
    println!("Maximum value: {}", max_val.as_ndarray()?);
    
    // Minimum value
    let min_val = data_tensor.min()?; // currently doesn't require dim and keepdim options
    println!("Minimum value: {}\n", min_val.as_ndarray()?);
    
    // Sum along specific dimension
    let row_sums = data_tensor.sum_dim(&[1], false)?; // Sum along columns
    println!("Sum along columns (for each row): {:?}", row_sums.as_ndarray()?);
    
    let col_sums = data_tensor.sum_dim(&[0], false)?; // Sum along rows
    println!("Sum along rows (for each column): {:?}\n", col_sums.as_ndarray()?);
    
    // 8. Understanding tensor broadcasting
    println!("8. Tensor Broadcasting");
    println!("======================");
    
    let matrix_a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let vector_b = Tensor::from_vec(vec![10.0, 20.0], &[2])?;
    
    println!("Matrix A (2x2): {:?}", matrix_a.data()?);
    println!("Vector B (2,): {:?}\n", vector_b.data()?);
    
    // Broadcasting: vector is automatically expanded to match matrix dimensions
    let broadcast_result = &matrix_a + &vector_b;
    println!("A + B (with broadcasting): {:?}\n", broadcast_result.as_ndarray()?);
    
    // 9. Tensor Transpose
    println!("9. Tensor Transpose");
    println!("===================");
    
    let matrix = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
    println!("Original (2x3):");
    println!("{:?}\n", matrix.as_ndarray()?);
    
    let transposed = matrix.transpose(0, 1)?;
    println!("Transposed (3x2):");
    println!("{:?}\n", transposed.as_ndarray()?);
    
    // 10. Tensor Concatenation
    println!("10. Tensor Concatenation");
    println!("========================");
    
    let a = Tensor::from_vec(vec![1.0, 2.0], &[1, 2])?;
    let b = Tensor::from_vec(vec![3.0, 4.0], &[1, 2])?;
    
    println!("Tensor A (1x2): {:?}", a.as_ndarray()?);
    println!("Tensor B (1x2): {:?}\n", b.as_ndarray()?);
    
    // Concatenate along dimension 0 (rows)
    let cat_rows = Tensor::cat(&[&a, &b], 0)?;
    println!("Concatenated along rows (2x2):");
    println!("{:?}\n", cat_rows.as_ndarray()?);
    
    // Concatenate along dimension 1 (columns)
    let cat_cols = Tensor::cat(&[&a, &b], 1)?;
    println!("Concatenated along columns (1x4):");
    println!("{:?}\n", cat_cols.as_ndarray()?);
    
    println!("🎉 Congratulations! You've completed Tutorial 01: Tensor Basics");
    println!("\n📖 Additional Operations Available:");
    println!("====================================");
    println!("✓ Comparison: a.eq(&b), a.lt(&b), a.gt(&b)");
    println!("✓ Clamping: tensor.clamp(min, max)");
    println!("✓ Sorting: tensor.sort(dim), tensor.argsort(dim)");
    println!("✓ Type casting: tensor.cast::<f32>()");
    println!("✓ Stacking: Tensor::stack(&[&a, &b], dim)");
    println!("\n📚 Next: Run `cargo run --example 02_autograd_basics` to learn about automatic differentiation");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tensor_creation() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        assert_eq!(tensor.numel(), 3);
        assert_eq!(tensor.shape().dims(), &[3]);
    }
    
    #[test]
    fn test_tensor_operations() {
        let a: Tensor<f64> = ones(&[2, 2]).unwrap();
        let b: Tensor<f64> = ones(&[2, 2]).unwrap();
        let result = &a + &b;
        
        // Result should be a 2x2 tensor of twos
        assert_eq!(result.shape().dims(), &[2, 2]);
    }
    
    #[test]
    fn test_tensor_reshape() {
        let tensor: Tensor<f64> = arange(0.0, 12.0, 1.0).unwrap();
        let reshaped: Tensor<f64> = tensor.reshape(&[3, 4]).unwrap();
        assert_eq!(reshaped.shape().dims(), &[3, 4]);
        assert_eq!(reshaped.numel(), 12);
    }
    
    #[test]
    fn test_tensor_transpose() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let transposed = tensor.transpose(0, 1).unwrap();
        assert_eq!(transposed.shape().dims(), &[3, 2]);
    }
    
    #[test]
    fn test_tensor_concat() {
        let a = Tensor::from_vec(vec![1.0, 2.0], &[1, 2]).unwrap();
        let b = Tensor::from_vec(vec![3.0, 4.0], &[1, 2]).unwrap();
        
        let cat_rows = Tensor::cat(&[&a, &b], 0).unwrap();
        assert_eq!(cat_rows.shape().dims(), &[2, 2]);
        
        let cat_cols = Tensor::cat(&[&a, &b], 1).unwrap();
        assert_eq!(cat_cols.shape().dims(), &[1, 4]);
    }
}