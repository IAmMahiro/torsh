//! Graph Convolutional Network (GCN) layer implementation
/// Crate-local result alias: the error type defaults to [`TorshError`],
/// so both `Result<T>` and `Result<T, OtherError>` stay valid.
type Result<T, E = torsh_core::error::TorshError> = std::result::Result<T, E>;

use crate::parameter::Parameter;
use crate::{GraphData, GraphLayer};
use torsh_tensor::{
    creation::{randn, zeros},
    Tensor,
};

/// Graph Convolutional Network (GCN) layer
#[derive(Debug)]
pub struct GCNConv {
    in_features: usize,
    out_features: usize,
    weight: Parameter,
    bias: Option<Parameter>,
}

impl GCNConv {
    /// Create a new GCN convolution layer
    ///
    /// # Errors
    /// Returns an error when the weight or bias tensors cannot be allocated.
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Result<Self> {
        let weight = Parameter::new(randn(&[in_features, out_features])?);
        let bias = if bias {
            Some(Parameter::new(zeros(&[out_features])?))
        } else {
            None
        };

        Ok(Self {
            in_features,
            out_features,
            weight,
            bias,
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

    /// Apply graph convolution
    ///
    /// Propagates with the Kipf & Welling operator
    /// `H' = D~^(-1/2) (A + I) D~^(-1/2) X W`, i.e. the renormalized adjacency,
    /// not the graph Laplacian.
    ///
    /// # Errors
    /// Returns an error when `graph.x` does not have `in_features` columns, or
    /// when `graph.edge_index` is malformed.
    pub fn forward(&self, graph: &GraphData) -> Result<GraphData> {
        let feature_dim = graph.x.shape().dims().get(1).copied().unwrap_or(0);
        if feature_dim != self.in_features {
            return Err(torsh_core::error::TorshError::ShapeMismatch {
                expected: vec![graph.num_nodes, self.in_features],
                got: graph.x.shape().dims().to_vec(),
            });
        }

        // Kipf-Welling propagation operator A_hat = D~^-1/2 (A + I) D~^-1/2
        let adjacency_hat = crate::utils::gcn_norm(&graph.edge_index, graph.num_nodes)?;

        // Apply graph convolution: A_hat @ X @ W
        let x_transformed = graph.x.matmul(&self.weight.clone_data())?;
        let mut output_features = adjacency_hat.matmul(&x_transformed)?;

        // Add bias if present
        if let Some(ref bias) = self.bias {
            output_features = output_features.add(&bias.clone_data())?;
        }

        // Create output graph with transformed features
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

impl GraphLayer for GCNConv {
    fn forward(&self, graph: &GraphData) -> Result<GraphData> {
        self.forward(graph)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone_data()];
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
    fn test_gcn_creation() {
        let gcn = GCNConv::new(8, 16, true).expect("gcn");
        let params = gcn.parameters();
        assert_eq!(params.len(), 2); // weight + bias
    }

    #[test]
    fn test_gcn_forward() {
        let gcn = GCNConv::new(3, 8, false).expect("gcn");

        // Create simple test graph
        let x = from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DeviceType::Cpu)
            .expect("from vec should succeed");
        let edge_index = from_vec(vec![0.0, 1.0, 1.0, 0.0], &[2, 2], DeviceType::Cpu)
            .expect("from vec should succeed");
        let graph = GraphData::new(x, edge_index);

        let output = gcn.forward(&graph).expect("forward");
        assert_eq!(output.x.shape().dims(), &[2, 8]);
        assert_eq!(output.num_nodes, 2);
    }
}
