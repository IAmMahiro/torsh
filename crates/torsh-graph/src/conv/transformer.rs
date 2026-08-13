//! Graph Transformer Networks layer implementation
/// Crate-local result alias: the error type defaults to [`TorshError`],
/// so both `Result<T>` and `Result<T, OtherError>` stay valid.
type Result<T, E = torsh_core::error::TorshError> = std::result::Result<T, E>;

use crate::parameter::Parameter;
use crate::{GraphData, GraphLayer};
use torsh_tensor::{
    creation::{randn, zeros},
    Tensor,
};

/// Graph Transformer Networks layer
#[derive(Debug)]
pub struct GraphTransformer {
    in_features: usize,
    out_features: usize,
    heads: usize,
    edge_dim: usize,
    query_weight: Parameter,
    key_weight: Parameter,
    value_weight: Parameter,
    edge_weight: Parameter,
    output_weight: Parameter,
    bias: Option<Parameter>,
    dropout: f32,
}

impl GraphTransformer {
    /// Create a new Graph Transformer layer
    ///
    /// # Errors
    /// Returns [`torsh_core::error::TorshError::InvalidArgument`] when `heads`
    /// is zero or does not divide `out_features` (the per-head dimension would
    /// not be well defined), and propagates tensor-allocation failures.
    pub fn new(
        in_features: usize,
        out_features: usize,
        heads: usize,
        edge_dim: usize,
        dropout: f32,
        bias: bool,
    ) -> Result<Self> {
        // Validate the head split once, so the forward path stays cheap.
        if heads == 0 || out_features % heads != 0 {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "out_features ({out_features}) must be a positive multiple of heads ({heads})"
            )));
        }

        let query_weight = Parameter::new(randn(&[in_features, out_features])?);
        let key_weight = Parameter::new(randn(&[in_features, out_features])?);
        let value_weight = Parameter::new(randn(&[in_features, out_features])?);
        let edge_weight = Parameter::new(randn(&[edge_dim, heads])?);
        let output_weight = Parameter::new(randn(&[out_features, out_features])?);

        let bias = if bias {
            Some(Parameter::new(zeros(&[out_features])?))
        } else {
            None
        };

        Ok(Self {
            in_features,
            out_features,
            heads,
            edge_dim,
            query_weight,
            key_weight,
            value_weight,
            edge_weight,
            output_weight,
            bias,
            dropout,
        })
    }

    /// Get input feature dimension
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get output feature dimension
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Get number of attention heads
    pub fn heads(&self) -> usize {
        self.heads
    }

    /// Get edge feature dimension
    pub fn edge_dim(&self) -> usize {
        self.edge_dim
    }

    /// Get dropout rate
    pub fn dropout(&self) -> f32 {
        self.dropout
    }

    /// Apply graph transformer convolution
    pub fn forward(&self, graph: &GraphData) -> Result<GraphData> {
        let num_nodes = graph.num_nodes;
        let head_dim = self.out_features / self.heads;

        // Linear transformations for Q, K, V
        let queries = graph.x.matmul(&self.query_weight.clone_data())?;
        let keys = graph.x.matmul(&self.key_weight.clone_data())?;
        let values = graph.x.matmul(&self.value_weight.clone_data())?;

        // Reshape for multi-head attention
        let q = queries.view(&[num_nodes as i32, self.heads as i32, head_dim as i32])?;
        let k = keys.view(&[num_nodes as i32, self.heads as i32, head_dim as i32])?;
        let v = values.view(&[num_nodes as i32, self.heads as i32, head_dim as i32])?;

        // Initialize output
        let mut output_features = zeros(&[num_nodes, self.out_features])?;

        // For simplicity, use a basic attention mechanism
        for head in 0..self.heads {
            let head_dim_start = head * head_dim;
            let head_dim_end = (head + 1) * head_dim;

            let q_head = q.slice(1, head, head + 1)?;
            let k_head = k.slice(1, head, head + 1)?;
            let v_head = v.slice(1, head, head + 1)?;

            // Basic self-attention computation
            let scale = 1.0 / (head_dim as f64).sqrt();
            let k_head_tensor = k_head.to_tensor()?.squeeze_tensor(1)?;
            let q_head_tensor = q_head.to_tensor()?.squeeze_tensor(1)?;
            let v_head_tensor = v_head.to_tensor()?.squeeze_tensor(1)?;

            let k_transposed = k_head_tensor.transpose(0, 1)?;
            let attention_scores = q_head_tensor
                .matmul(&k_transposed)?
                .mul_scalar(scale as f32)?;
            let attention_weights = attention_scores.softmax(-1)?;
            let head_output = attention_weights.matmul(&v_head_tensor)?;

            // Copy to output
            let output_slice = output_features.slice(1, head_dim_start, head_dim_end)?;
            // head_output is already [num_nodes, head_dim] - no need to squeeze
            let mut output_slice_tensor = output_slice.to_tensor()?;
            output_slice_tensor.copy_(&head_output)?;
        }

        // Apply output projection
        output_features = output_features.matmul(&self.output_weight.clone_data())?;

        // Add bias if present
        if let Some(ref bias) = self.bias {
            output_features = output_features.add(&bias.clone_data())?;
        }

        Ok(GraphData {
            x: output_features,
            edge_index: graph.edge_index.clone(),
            edge_attr: graph.edge_attr.clone(),
            batch: graph.batch.clone(),
            num_nodes: graph.num_nodes,
            num_edges: graph.num_edges,
        })
    }
}

impl GraphLayer for GraphTransformer {
    fn forward(&self, graph: &GraphData) -> Result<GraphData> {
        self.forward(graph)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![
            self.query_weight.clone_data(),
            self.key_weight.clone_data(),
            self.value_weight.clone_data(),
            self.edge_weight.clone_data(),
            self.output_weight.clone_data(),
        ];
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
    fn test_transformer_creation() {
        let transformer =
            GraphTransformer::new(16, 32, 8, 4, 0.1, true).expect("operation should succeed");
        let params = transformer.parameters();
        assert_eq!(params.len(), 6); // Q, K, V, edge, output weights + bias
        assert_eq!(transformer.heads, 8);
    }

    #[test]
    fn test_transformer_forward() {
        let transformer = GraphTransformer::new(6, 12, 3, 2, 0.0, false);

        // Create test graph
        let x = from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // node 0
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // node 1
                13.0, 14.0, 15.0, 16.0, 17.0, 18.0, // node 2
            ],
            &[3, 6],
            DeviceType::Cpu,
        )
        .expect("operation should succeed");
        let edge_index = from_vec(vec![0.0, 1.0, 2.0, 1.0, 2.0, 0.0], &[2, 3], DeviceType::Cpu)
            .expect("from vec should succeed");
        let graph = GraphData::new(x, edge_index);

        let output = transformer
            .expect("operation should succeed")
            .forward(&graph)
            .expect("operation should succeed");
        assert_eq!(output.x.shape().dims(), &[3, 12]);
        assert_eq!(output.num_nodes, 3);
    }
}
