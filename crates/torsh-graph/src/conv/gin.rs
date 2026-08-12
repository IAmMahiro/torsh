//! Graph Isomorphism Network (GIN) layer implementation
//! Based on the paper "How Powerful are Graph Neural Networks?"
// Framework infrastructure - components designed for future use
#![allow(dead_code)]
/// Crate-local result alias: the error type defaults to [`TorshError`],
/// so both `Result<T>` and `Result<T, OtherError>` stay valid.
type Result<T, E = torsh_core::error::TorshError> = std::result::Result<T, E>;

use crate::parameter::Parameter;
use crate::{GraphData, GraphLayer};
use torsh_tensor::{
    creation::{randn, zeros},
    Tensor,
};

/// Graph Isomorphism Network (GIN) layer
#[derive(Debug)]
pub struct GINConv {
    in_features: usize,
    out_features: usize,
    eps: f64,
    train_eps: bool,
    eps_param: Option<Parameter>,
    mlp: Vec<Parameter>, // Simple MLP: Linear -> ReLU -> Linear
    bias: Option<Parameter>,
}

impl GINConv {
    /// Create a new GIN convolution layer
    pub fn new(
        in_features: usize,
        out_features: usize,
        eps: f64,
        train_eps: bool,
        bias: bool,
    ) -> Result<Self> {
        let eps_param = if train_eps {
            Some(Parameter::new(torsh_tensor::creation::tensor_scalar(
                eps as f32,
            )?))
        } else {
            None
        };

        // Create a simple 2-layer MLP
        let hidden_dim = (in_features + out_features) / 2;
        let mlp = vec![
            Parameter::new(randn(&[in_features, hidden_dim])?),
            Parameter::new(randn(&[hidden_dim, out_features])?),
        ];

        let bias = if bias {
            Some(Parameter::new(zeros(&[out_features])?))
        } else {
            None
        };

        Ok(Self {
            in_features,
            out_features,
            eps,
            train_eps,
            eps_param,
            mlp,
            bias,
        })
    }

    /// Apply GIN convolution
    pub fn forward(&self, graph: &GraphData) -> Result<GraphData> {
        let num_nodes = graph.num_nodes;
        let edge_flat = graph.edge_index.to_vec()?;
        let num_edges = edge_flat.len() / 2;
        let edge_data = vec![
            edge_flat[0..num_edges].to_vec(),
            edge_flat[num_edges..].to_vec(),
        ];

        // Build adjacency list for efficient neighbor aggregation
        let mut adjacency_list: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];
        for j in 0..edge_data[0].len() {
            let src = edge_data[0][j] as usize;
            let dst = edge_data[1][j] as usize;
            if src < num_nodes && dst < num_nodes {
                adjacency_list[dst].push(src);
            }
        }

        // Aggregate neighbor features (sum aggregation for GIN)
        let neighbor_features = zeros(&[num_nodes, self.in_features])?;

        for node in 0..num_nodes {
            let mut aggregated = zeros(&[self.in_features])?;

            // Sum all neighbor features
            for &neighbor in &adjacency_list[node] {
                let neighbor_feat = graph
                    .x
                    .slice_tensor(0, neighbor, neighbor + 1)?
                    .squeeze_tensor(0)?;
                aggregated = aggregated.add(&neighbor_feat)?;
            }

            let mut node_slice = neighbor_features.slice_tensor(0, node, node + 1)?;
            let _ = node_slice.copy_(&aggregated.unsqueeze_tensor(0)?);
        }

        // Get epsilon value
        let epsilon = if let Some(ref eps_param) = self.eps_param {
            eps_param.clone_data().to_vec()?[0] as f64
        } else {
            self.eps
        };

        // Combine self and neighbor features: (1 + eps) * h_i + sum(h_j)
        let self_weighted = graph.x.mul_scalar((1.0 + epsilon) as f32)?;
        let combined_features = self_weighted.add(&neighbor_features)?;

        // Apply MLP
        let mut output = combined_features.matmul(&self.mlp[0].clone_data())?;

        // Apply ReLU activation (using max with zero tensor)
        let zero_tensor = zeros(output.shape().dims())?;
        output = output.maximum(&zero_tensor)?;

        // Second layer
        output = output.matmul(&self.mlp[1].clone_data())?;

        // Add bias if present
        if let Some(ref bias) = self.bias {
            output = output.add(&bias.clone_data())?;
        }

        // Create output graph
        Ok(GraphData {
            x: output,
            edge_index: graph.edge_index.clone(),
            edge_attr: graph.edge_attr.clone(),
            batch: graph.batch.clone(),
            num_nodes: graph.num_nodes,
            num_edges: graph.num_edges,
        })
    }
}

impl GraphLayer for GINConv {
    fn forward(&self, graph: &GraphData) -> Result<GraphData> {
        self.forward(graph)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.mlp[0].clone_data(), self.mlp[1].clone_data()];

        if let Some(ref eps_param) = self.eps_param {
            params.push(eps_param.clone_data());
        }

        if let Some(ref bias) = self.bias {
            params.push(bias.clone_data());
        }

        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;
    use torsh_tensor::creation::from_vec;

    #[test]
    fn test_gin_creation() {
        let gin = GINConv::new(8, 16, 0.5, true, true);
        let params = gin.expect("operation should succeed").parameters();
        assert!(params.len() >= 2); // At least MLP weights
        assert!(params.len() <= 4); // At most MLP + eps + bias
    }

    #[test]
    fn test_gin_forward() {
        let gin = GINConv::new(4, 6, 0.0, false, false);

        // Create test graph
        let x = from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, // node 0
                5.0, 6.0, 7.0, 8.0, // node 1
                9.0, 10.0, 11.0, 12.0, // node 2
            ],
            &[3, 4],
            DeviceType::Cpu,
        )
        .expect("operation should succeed");
        let edge_index = from_vec(vec![0.0, 1.0, 2.0, 1.0, 2.0, 0.0], &[2, 3], DeviceType::Cpu)
            .expect("from vec should succeed");
        let graph = GraphData::new(x, edge_index);

        let output = gin
            .expect("operation should succeed")
            .forward(&graph)
            .expect("operation should succeed");
        assert_eq!(output.x.shape().dims(), &[3, 6]);
        assert_eq!(output.num_nodes, 3);
    }

    #[test]
    fn test_gin_trainable_eps() {
        let gin_fixed = GINConv::new(4, 8, 1.0, false, false);
        let gin_trainable = GINConv::new(4, 8, 1.0, true, false);

        let fixed_params = gin_fixed.expect("operation should succeed").parameters();
        let trainable_params = gin_trainable
            .expect("operation should succeed")
            .parameters();

        // Trainable eps version should have one more parameter
        assert_eq!(trainable_params.len(), fixed_params.len() + 1);
    }
}
