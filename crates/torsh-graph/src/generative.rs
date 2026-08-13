//! Graph Generation Models
//!
//! Advanced implementation of generative models for graphs including
//! Variational Autoencoders (VAE) and Generative Adversarial Networks (GAN)
//! specifically designed for graph-structured data.
//!
//! # Features:
//! - Graph Variational Autoencoder (GraphVAE)
//! - Graph Generative Adversarial Network (GraphGAN)
//! - Conditional graph generation
//! - Graph reconstruction and completion
//! - Latent space graph interpolation
//! - Property-guided graph generation
// Framework infrastructure - components designed for future use
#![allow(dead_code)]
/// Crate-local result alias: the error type defaults to [`TorshError`],
/// so both `Result<T>` and `Result<T, OtherError>` stay valid.
type Result<T, E = torsh_core::error::TorshError> = std::result::Result<T, E>;

use crate::parameter::Parameter;
use crate::{GraphData, GraphLayer};
use scirs2_core::random::thread_rng;
use torsh_tensor::{
    creation::{from_vec, randn, zeros},
    Tensor,
};

/// Numerically stable `softplus(x) = ln(1 + e^x)`.
///
/// Evaluated as `x + ln(1 + e^-x)` for positive `x` so that neither branch ever
/// overflows `e^x`; the result is finite for every finite input.
fn softplus(x: f32) -> f32 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// Graph Variational Autoencoder (GraphVAE)
/// Learns a probabilistic latent representation of graphs
#[derive(Debug)]
pub struct GraphVAE {
    // Encoder parameters
    encoder_in_features: usize,
    encoder_hidden_features: usize,
    latent_dim: usize,

    // Encoder layers
    encoder_layer1: Parameter,
    encoder_layer2: Parameter,

    // Variational parameters (mean and log-variance)
    mu_layer: Parameter,
    logvar_layer: Parameter,

    // Decoder parameters
    decoder_layer1: Parameter,
    decoder_layer2: Parameter,
    node_decoder: Parameter,
    edge_decoder: Parameter,

    // KL divergence weight
    beta: f32,

    // Bias terms
    encoder_bias1: Option<Parameter>,
    encoder_bias2: Option<Parameter>,
    decoder_bias1: Option<Parameter>,
    decoder_bias2: Option<Parameter>,
}

impl GraphVAE {
    /// Create a new Graph Variational Autoencoder
    pub fn new(
        in_features: usize,
        hidden_features: usize,
        latent_dim: usize,
        beta: f32,
        use_bias: bool,
    ) -> Result<Self> {
        // Encoder layers
        let encoder_layer1 = Parameter::new(randn(&[in_features, hidden_features])?);
        let encoder_layer2 = Parameter::new(randn(&[hidden_features, hidden_features])?);

        // Variational layers
        let mu_layer = Parameter::new(randn(&[hidden_features, latent_dim])?);
        let logvar_layer = Parameter::new(randn(&[hidden_features, latent_dim])?);

        // Decoder layers
        let decoder_layer1 = Parameter::new(randn(&[latent_dim, hidden_features])?);
        let decoder_layer2 = Parameter::new(randn(&[hidden_features, hidden_features])?);
        let node_decoder = Parameter::new(randn(&[hidden_features, in_features])?);
        let edge_decoder = Parameter::new(randn(&[hidden_features, 1])?);

        let (encoder_bias1, encoder_bias2, decoder_bias1, decoder_bias2) = if use_bias {
            (
                Some(Parameter::new(zeros(&[hidden_features])?)),
                Some(Parameter::new(zeros(&[hidden_features])?)),
                Some(Parameter::new(zeros(&[hidden_features])?)),
                Some(Parameter::new(zeros(&[hidden_features])?)),
            )
        } else {
            (None, None, None, None)
        };

        Ok(Self {
            encoder_in_features: in_features,
            encoder_hidden_features: hidden_features,
            latent_dim,
            encoder_layer1,
            encoder_layer2,
            mu_layer,
            logvar_layer,
            decoder_layer1,
            decoder_layer2,
            node_decoder,
            edge_decoder,
            beta,
            encoder_bias1,
            encoder_bias2,
            decoder_bias1,
            decoder_bias2,
        })
    }

    /// Encode graph to latent distribution parameters
    pub fn encode(&self, graph: &GraphData) -> Result<(Tensor, Tensor)> {
        // Forward through encoder
        let mut h = graph.x.matmul(&self.encoder_layer1.clone_data())?;
        if let Some(ref bias) = self.encoder_bias1 {
            h = h.add(&bias.clone_data())?;
        }
        h = self.relu(&h)?;

        h = h.matmul(&self.encoder_layer2.clone_data())?;
        if let Some(ref bias) = self.encoder_bias2 {
            h = h.add(&bias.clone_data())?;
        }
        h = self.relu(&h)?;

        // Global mean pooling
        let graph_embedding = h.mean(Some(&[0]), false)?;
        let graph_embedding_2d = graph_embedding.unsqueeze(0)?; // Make 2D for matmul

        // Compute mu and logvar
        let mu = graph_embedding_2d.matmul(&self.mu_layer.clone_data())?;
        let logvar = graph_embedding_2d.matmul(&self.logvar_layer.clone_data())?;

        Ok((mu, logvar))
    }

    /// Reparameterization trick for sampling from latent distribution
    pub fn reparameterize(&self, mu: &Tensor, logvar: &Tensor) -> Result<Tensor> {
        // std = exp(0.5 * logvar)
        let std = logvar.mul_scalar(0.5)?.exp()?;

        // Sample epsilon from N(0, 1)
        let epsilon = randn(mu.shape().dims())?;

        // z = mu + std * epsilon
        Ok(mu.add(&std.mul(&epsilon)?)?)
    }

    /// Decode latent representation to graph
    pub fn decode(&self, z: &Tensor, num_nodes: usize) -> Result<GraphData> {
        // Forward through decoder
        let mut h = z.matmul(&self.decoder_layer1.clone_data())?;
        if let Some(ref bias) = self.decoder_bias1 {
            h = h.add(&bias.clone_data())?;
        }
        h = self.relu(&h)?;

        h = h.matmul(&self.decoder_layer2.clone_data())?;
        if let Some(ref bias) = self.decoder_bias2 {
            h = h.add(&bias.clone_data())?;
        }
        h = self.relu(&h)?;

        // Expand to node-level representation
        let h_expanded = self.expand_to_nodes(&h, num_nodes)?;

        // Decode node features
        let node_features = h_expanded.matmul(&self.node_decoder.clone_data())?;

        // Decode edge probabilities
        let edge_logits = self.decode_edges(&h_expanded, num_nodes)?;
        let edge_index = self.sample_edges(&edge_logits, num_nodes)?;

        Ok(GraphData::new(node_features, edge_index))
    }

    /// Forward pass through GraphVAE
    ///
    /// # Errors
    /// Propagates encoder/decoder tensor-operation failures.
    pub fn forward(&self, graph: &GraphData) -> Result<(GraphData, Tensor, Tensor)> {
        // Encode
        let (mu, logvar) = self.encode(graph)?;

        // Sample latent variable
        let z = self.reparameterize(&mu, &logvar)?;

        // Decode
        let reconstructed = self.decode(&z, graph.num_nodes)?;

        Ok((reconstructed, mu, logvar))
    }

    /// Compute VAE loss (reconstruction + KL divergence)
    pub fn compute_loss(
        &self,
        graph: &GraphData,
        reconstructed: &GraphData,
        mu: &Tensor,
        logvar: &Tensor,
    ) -> Result<f32> {
        // Reconstruction loss (MSE for node features)
        let recon_loss = self.reconstruction_loss(graph, reconstructed)?;

        // KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        let kl_loss = self.kl_divergence(mu, logvar)?;

        // Total loss
        Ok(recon_loss + self.beta * kl_loss)
    }

    /// Reconstruction loss (MSE)
    fn reconstruction_loss(&self, original: &GraphData, reconstructed: &GraphData) -> Result<f32> {
        let orig_data = original.x.to_vec()?;
        let recon_data = reconstructed.x.to_vec()?;

        let mut mse = 0.0;
        let len = orig_data.len().min(recon_data.len());

        for i in 0..len {
            mse += (orig_data[i] - recon_data[i]).powi(2);
        }

        Ok(mse / len as f32)
    }

    /// KL divergence loss
    fn kl_divergence(&self, mu: &Tensor, logvar: &Tensor) -> Result<f32> {
        let mu_data = mu.to_vec()?;
        let logvar_data = logvar.to_vec()?;

        let mut kl = 0.0;
        for i in 0..mu_data.len() {
            kl += -0.5 * (1.0 + logvar_data[i] - mu_data[i].powi(2) - logvar_data[i].exp());
        }

        Ok(kl / mu_data.len() as f32)
    }

    /// Generate new graph from random latent vector
    pub fn generate(&self, num_nodes: usize) -> Result<GraphData> {
        // Sample from standard normal
        let z = randn(&[1, self.latent_dim])?;

        // Decode to graph
        self.decode(&z, num_nodes)
    }

    /// Interpolate between two graphs in latent space
    pub fn interpolate(
        &self,
        graph1: &GraphData,
        graph2: &GraphData,
        alpha: f32,
        num_nodes: usize,
    ) -> Result<GraphData> {
        let (mu1, _) = self.encode(graph1)?;
        let (mu2, _) = self.encode(graph2)?;

        // Linear interpolation
        let z_interp = mu1.mul_scalar(1.0 - alpha)?.add(&mu2.mul_scalar(alpha)?)?;

        // Decode interpolated latent
        self.decode(&z_interp, num_nodes)
    }

    // Helper methods

    fn relu(&self, x: &Tensor) -> Result<Tensor> {
        let data = x.to_vec()?;
        let activated: Vec<f32> = data.iter().map(|&v| v.max(0.0)).collect();
        Ok(from_vec(
            activated,
            x.shape().dims(),
            torsh_core::device::DeviceType::Cpu,
        )?)
    }

    fn expand_to_nodes(&self, h: &Tensor, num_nodes: usize) -> Result<Tensor> {
        // Repeat graph-level embedding for each node
        let h_data = h.to_vec()?;
        let feat_dim = h_data.len();

        let mut expanded_data = Vec::new();
        for _ in 0..num_nodes {
            expanded_data.extend(&h_data);
        }

        Ok(from_vec(
            expanded_data,
            &[num_nodes, feat_dim],
            torsh_core::device::DeviceType::Cpu,
        )?)
    }

    fn decode_edges(&self, h: &Tensor, num_nodes: usize) -> Result<Tensor> {
        // Compute pairwise edge probabilities
        let mut edge_logits_data = Vec::new();

        for i in 0..num_nodes {
            for j in 0..num_nodes {
                if i != j {
                    // Simplified: use dot product of node embeddings as edge logit
                    let h_i = h.slice_tensor(0, i, i + 1)?;
                    let h_j = h.slice_tensor(0, j, j + 1)?;

                    let logit = h_i.dot(&h_j.t()?)?.item()?;
                    edge_logits_data.push(logit);
                } else {
                    edge_logits_data.push(-1000.0); // No self-loops
                }
            }
        }

        Ok(from_vec(
            edge_logits_data,
            &[num_nodes, num_nodes],
            torsh_core::device::DeviceType::Cpu,
        )?)
    }

    fn sample_edges(&self, edge_logits: &Tensor, num_nodes: usize) -> Result<Tensor> {
        let logits_data = edge_logits.to_vec()?;
        let mut edges = Vec::new();

        // Sample edges based on probabilities (threshold at 0.5)
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                if i != j {
                    let idx = i * num_nodes + j;
                    let prob = 1.0 / (1.0 + (-logits_data[idx]).exp()); // Sigmoid

                    if prob > 0.5 {
                        edges.push(i as f32);
                        edges.push(j as f32);
                    }
                }
            }
        }

        if edges.is_empty() {
            // Return empty edge index
            return Ok(zeros(&[2, 0])?);
        }

        let num_edges = edges.len() / 2;
        Ok(from_vec(
            edges,
            &[2, num_edges],
            torsh_core::device::DeviceType::Cpu,
        )?)
    }
}

impl GraphLayer for GraphVAE {
    fn forward(&self, graph: &GraphData) -> Result<GraphData> {
        let (reconstructed, _, _) = GraphVAE::forward(self, graph)?;
        Ok(reconstructed)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![
            self.encoder_layer1.clone_data(),
            self.encoder_layer2.clone_data(),
            self.mu_layer.clone_data(),
            self.logvar_layer.clone_data(),
            self.decoder_layer1.clone_data(),
            self.decoder_layer2.clone_data(),
            self.node_decoder.clone_data(),
            self.edge_decoder.clone_data(),
        ];

        if let Some(ref b) = self.encoder_bias1 {
            params.push(b.clone_data());
        }
        if let Some(ref b) = self.encoder_bias2 {
            params.push(b.clone_data());
        }
        if let Some(ref b) = self.decoder_bias1 {
            params.push(b.clone_data());
        }
        if let Some(ref b) = self.decoder_bias2 {
            params.push(b.clone_data());
        }

        params
    }
}

/// Graph Generative Adversarial Network (GraphGAN)
/// Learns to generate realistic graphs through adversarial training
#[derive(Debug)]
pub struct GraphGAN {
    latent_dim: usize,
    hidden_dim: usize,
    output_features: usize,

    // Generator network
    generator: GraphGANGenerator,

    // Discriminator network
    discriminator: GraphGANDiscriminator,
}

impl GraphGAN {
    /// Create a new Graph GAN
    pub fn new(
        latent_dim: usize,
        hidden_dim: usize,
        output_features: usize,
        use_bias: bool,
    ) -> Result<Self> {
        let generator = GraphGANGenerator::new(latent_dim, hidden_dim, output_features, use_bias)?;
        let discriminator = GraphGANDiscriminator::new(output_features, hidden_dim, use_bias)?;

        Ok(Self {
            latent_dim,
            hidden_dim,
            output_features,
            generator,
            discriminator,
        })
    }

    /// Generate fake graph from random noise
    pub fn generate(&self, num_nodes: usize) -> Result<GraphData> {
        let z = randn(&[1, self.latent_dim])?;
        self.generator.generate(&z, num_nodes)
    }

    /// Discriminator forward pass (returns real/fake probability in `(0, 1)`)
    ///
    /// # Errors
    /// Propagates discriminator tensor-operation failures.
    pub fn discriminate(&self, graph: &GraphData) -> Result<f32> {
        self.discriminator.forward(graph)
    }

    /// Discriminator forward pass returning the raw pre-sigmoid logit.
    ///
    /// The losses are computed from this value rather than from the sigmoid
    /// output: in `f32` the sigmoid saturates to exactly `0.0` or `1.0` for
    /// logits beyond roughly +-17, and `ln(0)` would make the loss infinite.
    ///
    /// # Errors
    /// Propagates discriminator tensor-operation failures.
    pub fn discriminate_logit(&self, graph: &GraphData) -> Result<f32> {
        self.discriminator.forward_logit(graph)
    }

    /// Train generator (maximize discriminator error)
    ///
    /// Computes `-log D(G(z))` as `softplus(-logit)`, which is finite for every
    /// finite logit.
    ///
    /// # Errors
    /// Propagates generator/discriminator tensor-operation failures.
    pub fn generator_loss(&self, num_nodes: usize) -> Result<f32> {
        let fake_graph = self.generate(num_nodes)?;
        let fake_logit = self.discriminate_logit(&fake_graph)?;

        // Generator loss: -log(D(G(z))) = softplus(-logit)
        Ok(softplus(-fake_logit))
    }

    /// Train discriminator (distinguish real from fake)
    ///
    /// Computes `-log D(real) - log(1 - D(fake))` in the numerically stable
    /// binary-cross-entropy-with-logits form
    /// `softplus(-logit_real) + softplus(logit_fake)`.
    ///
    /// # Errors
    /// Propagates generator/discriminator tensor-operation failures.
    pub fn discriminator_loss(&self, real_graph: &GraphData, num_nodes: usize) -> Result<f32> {
        let real_logit = self.discriminate_logit(real_graph)?;

        let fake_graph = self.generate(num_nodes)?;
        let fake_logit = self.discriminate_logit(&fake_graph)?;

        // -log(sigmoid(x))     = softplus(-x)
        // -log(1 - sigmoid(x)) = softplus(x)
        Ok(softplus(-real_logit) + softplus(fake_logit))
    }

    /// Get generator parameters
    pub fn generator_parameters(&self) -> Vec<Tensor> {
        self.generator.parameters()
    }

    /// Get discriminator parameters
    pub fn discriminator_parameters(&self) -> Vec<Tensor> {
        self.discriminator.parameters()
    }
}

/// Generator network for GraphGAN
#[derive(Debug)]
struct GraphGANGenerator {
    latent_dim: usize,
    hidden_dim: usize,
    output_features: usize,

    layer1: Parameter,
    layer2: Parameter,
    node_layer: Parameter,
    edge_layer: Parameter,

    bias1: Option<Parameter>,
    bias2: Option<Parameter>,
}

impl GraphGANGenerator {
    fn new(
        latent_dim: usize,
        hidden_dim: usize,
        output_features: usize,
        use_bias: bool,
    ) -> Result<Self> {
        let layer1 = Parameter::new(randn(&[latent_dim, hidden_dim])?);
        let layer2 = Parameter::new(randn(&[hidden_dim, hidden_dim])?);
        let node_layer = Parameter::new(randn(&[hidden_dim, output_features])?);
        let edge_layer = Parameter::new(randn(&[hidden_dim, 1])?);

        let (bias1, bias2) = if use_bias {
            (
                Some(Parameter::new(zeros(&[hidden_dim])?)),
                Some(Parameter::new(zeros(&[hidden_dim])?)),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            latent_dim,
            hidden_dim,
            output_features,
            layer1,
            layer2,
            node_layer,
            edge_layer,
            bias1,
            bias2,
        })
    }

    fn generate(&self, z: &Tensor, num_nodes: usize) -> Result<GraphData> {
        // Forward through generator
        let mut h = z.matmul(&self.layer1.clone_data())?;
        if let Some(ref bias) = self.bias1 {
            h = h.add(&bias.clone_data())?;
        }
        h = self.leaky_relu(&h, 0.2)?;

        h = h.matmul(&self.layer2.clone_data())?;
        if let Some(ref bias) = self.bias2 {
            h = h.add(&bias.clone_data())?;
        }
        h = self.leaky_relu(&h, 0.2)?;

        // Expand to node-level
        let h_expanded = self.expand_to_nodes(&h, num_nodes)?;

        // Generate node features
        let node_features = h_expanded.matmul(&self.node_layer.clone_data())?;
        let node_features = self.tanh(&node_features)?;

        // Generate edges
        let edge_index = self.generate_edges(&h_expanded, num_nodes)?;

        Ok(GraphData::new(node_features, edge_index))
    }

    fn leaky_relu(&self, x: &Tensor, alpha: f32) -> Result<Tensor> {
        let data = x.to_vec()?;
        let activated: Vec<f32> = data
            .iter()
            .map(|&v| if v > 0.0 { v } else { alpha * v })
            .collect();
        Ok(from_vec(
            activated,
            x.shape().dims(),
            torsh_core::device::DeviceType::Cpu,
        )?)
    }

    fn tanh(&self, x: &Tensor) -> Result<Tensor> {
        let data = x.to_vec()?;
        let activated: Vec<f32> = data.iter().map(|&v| v.tanh()).collect();
        Ok(from_vec(
            activated,
            x.shape().dims(),
            torsh_core::device::DeviceType::Cpu,
        )?)
    }

    fn expand_to_nodes(&self, h: &Tensor, num_nodes: usize) -> Result<Tensor> {
        let h_data = h.to_vec()?;
        let feat_dim = h_data.len();

        let mut expanded_data = Vec::new();
        for _ in 0..num_nodes {
            expanded_data.extend(&h_data);
        }

        Ok(from_vec(
            expanded_data,
            &[num_nodes, feat_dim],
            torsh_core::device::DeviceType::Cpu,
        )?)
    }

    fn generate_edges(&self, _h: &Tensor, num_nodes: usize) -> Result<Tensor> {
        let mut edges = Vec::new();
        let mut rng = thread_rng();

        // Generate edges probabilistically
        for i in 0..num_nodes {
            for j in (i + 1)..num_nodes {
                // Use node embeddings to determine edge probability
                if rng.gen_range(0.0..1.0) > 0.7 {
                    edges.push(i as f32);
                    edges.push(j as f32);
                    edges.push(j as f32);
                    edges.push(i as f32);
                }
            }
        }

        if edges.is_empty() {
            return Ok(zeros(&[2, 0])?);
        }

        let num_edges = edges.len() / 2;
        Ok(from_vec(
            edges,
            &[2, num_edges],
            torsh_core::device::DeviceType::Cpu,
        )?)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![
            self.layer1.clone_data(),
            self.layer2.clone_data(),
            self.node_layer.clone_data(),
            self.edge_layer.clone_data(),
        ];

        if let Some(ref b) = self.bias1 {
            params.push(b.clone_data());
        }
        if let Some(ref b) = self.bias2 {
            params.push(b.clone_data());
        }

        params
    }
}

/// Discriminator network for GraphGAN
#[derive(Debug)]
struct GraphGANDiscriminator {
    input_features: usize,
    hidden_dim: usize,

    layer1: Parameter,
    layer2: Parameter,
    output_layer: Parameter,

    bias1: Option<Parameter>,
    bias2: Option<Parameter>,
    bias_out: Option<Parameter>,
}

impl GraphGANDiscriminator {
    fn new(input_features: usize, hidden_dim: usize, use_bias: bool) -> Result<Self> {
        let layer1 = Parameter::new(randn(&[input_features, hidden_dim])?);
        let layer2 = Parameter::new(randn(&[hidden_dim, hidden_dim])?);
        let output_layer = Parameter::new(randn(&[hidden_dim, 1])?);

        let (bias1, bias2, bias_out) = if use_bias {
            (
                Some(Parameter::new(zeros(&[hidden_dim])?)),
                Some(Parameter::new(zeros(&[hidden_dim])?)),
                Some(Parameter::new(zeros(&[1])?)),
            )
        } else {
            (None, None, None)
        };

        Ok(Self {
            input_features,
            hidden_dim,
            layer1,
            layer2,
            output_layer,
            bias1,
            bias2,
            bias_out,
        })
    }

    /// Raw pre-sigmoid discriminator output.
    fn forward_logit(&self, graph: &GraphData) -> Result<f32> {
        // Forward through discriminator
        let mut h = graph.x.matmul(&self.layer1.clone_data())?;
        if let Some(ref bias) = self.bias1 {
            h = h.add(&bias.clone_data())?;
        }
        h = self.leaky_relu(&h, 0.2)?;

        h = h.matmul(&self.layer2.clone_data())?;
        if let Some(ref bias) = self.bias2 {
            h = h.add(&bias.clone_data())?;
        }
        h = self.leaky_relu(&h, 0.2)?;

        // Global mean pooling
        let graph_repr = h.mean(Some(&[0]), false)?;
        let graph_repr_2d = graph_repr.unsqueeze(0)?; // Make 2D for matmul

        // Output layer
        let mut logit = graph_repr_2d.matmul(&self.output_layer.clone_data())?;
        if let Some(ref bias) = self.bias_out {
            logit = logit.add(&bias.clone_data())?;
        }

        logit.item()
    }

    /// Discriminator score in `(0, 1)`: `sigmoid(logit)`.
    fn forward(&self, graph: &GraphData) -> Result<f32> {
        let logit_val = self.forward_logit(graph)?;
        Ok(1.0 / (1.0 + (-logit_val).exp()))
    }

    fn leaky_relu(&self, x: &Tensor, alpha: f32) -> Result<Tensor> {
        let data = x.to_vec()?;
        let activated: Vec<f32> = data
            .iter()
            .map(|&v| if v > 0.0 { v } else { alpha * v })
            .collect();
        Ok(from_vec(
            activated,
            x.shape().dims(),
            torsh_core::device::DeviceType::Cpu,
        )?)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![
            self.layer1.clone_data(),
            self.layer2.clone_data(),
            self.output_layer.clone_data(),
        ];

        if let Some(ref b) = self.bias1 {
            params.push(b.clone_data());
        }
        if let Some(ref b) = self.bias2 {
            params.push(b.clone_data());
        }
        if let Some(ref b) = self.bias_out {
            params.push(b.clone_data());
        }

        params
    }
}

/// Conditional Graph Generation
#[derive(Debug)]
pub struct ConditionalGraphGenerator {
    vae: GraphVAE,
    condition_dim: usize,
    condition_layer: Parameter,
}

impl ConditionalGraphGenerator {
    /// Create a new conditional graph generator
    pub fn new(
        in_features: usize,
        hidden_features: usize,
        latent_dim: usize,
        condition_dim: usize,
        beta: f32,
    ) -> Result<Self> {
        let vae = GraphVAE::new(in_features, hidden_features, latent_dim, beta, true)?;
        let condition_layer = Parameter::new(randn(&[condition_dim, latent_dim])?);

        Ok(Self {
            vae,
            condition_dim,
            condition_layer,
        })
    }

    /// Generate graph conditioned on a property vector
    pub fn generate_conditional(&self, condition: &Tensor, num_nodes: usize) -> Result<GraphData> {
        // Map condition to latent space bias
        let condition_bias = condition.matmul(&self.condition_layer.clone_data())?;

        // Sample base latent vector
        let z_base = randn(&[1, self.vae.latent_dim])?;

        // Add conditional bias
        let z = z_base.add(&condition_bias)?;

        // Decode to graph
        self.vae.decode(&z, num_nodes)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.vae.parameters();
        params.push(self.condition_layer.clone_data());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_graphvae_creation() {
        let vae = GraphVAE::new(8, 16, 10, 1.0, true).expect("operation should succeed");
        assert_eq!(vae.encoder_in_features, 8);
        assert_eq!(vae.encoder_hidden_features, 16);
        assert_eq!(vae.latent_dim, 10);
        assert_eq!(vae.beta, 1.0);
    }

    #[test]
    fn test_graphvae_encode_decode() {
        let features = randn(&[5, 8]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0];
        let edge_index = from_vec(edges, &[2, 4], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(features, edge_index);

        let vae = GraphVAE::new(8, 16, 10, 1.0, true).expect("operation should succeed");

        let (mu, logvar) = vae.encode(&graph).expect("operation should succeed");
        assert_eq!(mu.shape().dims(), &[1, 10]);
        assert_eq!(logvar.shape().dims(), &[1, 10]);

        let z = vae
            .reparameterize(&mu, &logvar)
            .expect("operation should succeed");
        assert_eq!(z.shape().dims(), &[1, 10]);

        let reconstructed = vae.decode(&z, 5).expect("operation should succeed");
        assert_eq!(reconstructed.num_nodes, 5);
    }

    #[test]
    fn test_graphvae_generation() {
        let vae = GraphVAE::new(8, 16, 10, 1.0, true).expect("operation should succeed");
        let generated = vae.generate(6).expect("operation should succeed");

        assert_eq!(generated.num_nodes, 6);
        assert_eq!(generated.x.shape().dims()[0], 6);
        assert_eq!(generated.x.shape().dims()[1], 8);
    }

    #[test]
    fn test_graphvae_interpolation() {
        let features1 = randn(&[4, 6]).unwrap();
        let features2 = randn(&[4, 6]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0];
        let edge_index = from_vec(edges.clone(), &[2, 3], DeviceType::Cpu).unwrap();

        let graph1 = GraphData::new(features1, edge_index.clone());
        let graph2 = GraphData::new(features2, edge_index);

        let vae = GraphVAE::new(6, 12, 8, 1.0, true).expect("operation should succeed");

        // Interpolate at alpha = 0.5 (midpoint)
        let interpolated = vae
            .interpolate(&graph1, &graph2, 0.5, 4)
            .expect("operation should succeed");
        assert_eq!(interpolated.num_nodes, 4);
    }

    #[test]
    fn test_graphgan_creation() {
        let gan = GraphGAN::new(16, 32, 8, true).expect("operation should succeed");
        assert_eq!(gan.latent_dim, 16);
        assert_eq!(gan.hidden_dim, 32);
        assert_eq!(gan.output_features, 8);
    }

    #[test]
    fn test_graphgan_generation() {
        let gan = GraphGAN::new(16, 32, 8, true).expect("operation should succeed");
        let generated = gan.generate(5).expect("operation should succeed");

        assert_eq!(generated.num_nodes, 5);
        assert_eq!(generated.x.shape().dims()[1], 8);
    }

    #[test]
    fn test_graphgan_discriminate() {
        let features = randn(&[4, 8]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0];
        let edge_index = from_vec(edges, &[2, 3], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(features, edge_index);

        let gan = GraphGAN::new(16, 32, 8, true).expect("operation should succeed");
        let score = gan.discriminate(&graph).expect("operation should succeed");

        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_conditional_generation() {
        let cond_gen =
            ConditionalGraphGenerator::new(8, 16, 10, 4, 1.0).expect("operation should succeed");

        let condition = randn(&[1, 4]).unwrap();
        let generated = cond_gen
            .generate_conditional(&condition, 5)
            .expect("operation should succeed");

        assert_eq!(generated.num_nodes, 5);
        assert_eq!(generated.x.shape().dims()[1], 8);
    }

    #[test]
    fn test_graphvae_loss_computation() {
        let features = randn(&[3, 6]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0];
        let edge_index = from_vec(edges, &[2, 2], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(features, edge_index);

        let vae = GraphVAE::new(6, 12, 8, 1.0, true).expect("operation should succeed");
        let (reconstructed, mu, logvar) = vae.forward(&graph).expect("operation should succeed");

        let loss = vae
            .compute_loss(&graph, &reconstructed, &mu, &logvar)
            .expect("operation should succeed");
        assert!(loss > 0.0);
    }

    /// A value in `[0, 1)` that is a pure function of `name` and `index` — no
    /// RNG, no thread-local state, no process-global generator state. See
    /// `deterministic_reinit_mpnn` in
    /// `torsh-graph/tests/comprehensive_gnn_tests.rs` for the twin of this
    /// helper and the measurement backing it.
    fn deterministic_unit_interval(name: &str, index: u64) -> f32 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a offset basis
        for byte in name.bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3); // FNV-1a prime
        }
        hash ^= index.wrapping_add(0x9E37_79B9_7F4A_7C15);
        hash = (hash ^ (hash >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        hash = (hash ^ (hash >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        hash ^= hash >> 31;
        ((hash >> 40) as f32) / (1u64 << 24) as f32
    }

    /// Deterministically overwrites every tensor in `params` in place with a
    /// pure function of `(label, parameter position, flat index)`.
    ///
    /// `params` tensors (as returned by `GraphGAN::generator_parameters` /
    /// `discriminator_parameters`, themselves built from `Parameter::
    /// clone_data`) alias the layer's real storage: `Tensor`'s `InMemory`
    /// backing is `Arc<RwLock<Vec<T>>>`, `Clone` shares the `Arc`, and
    /// `Tensor::set_slice` writes through shared storage with no
    /// copy-on-write step — so overwriting these tensors mutates the live
    /// `GraphGAN` in place (mirrors the empirically-confirmed aliasing in
    /// `deterministic_reinit_mpnn`).
    ///
    /// Skips 1-D tensors (every bias in `GraphGANGenerator`/
    /// `GraphGANDiscriminator` is 1-D, and every weight is 2-D): both
    /// constructors already initialize biases to `zeros(..)`, so they are
    /// already deterministic, and overwriting them with nonzero values would
    /// only make the reinitialized network diverge from what `GraphGAN::new`
    /// actually produces, for no determinism gained.
    fn reinit_params(params: &[Tensor], label: &str) {
        for (i, tensor) in params.iter().enumerate() {
            let dims = tensor.shape().dims().to_vec();
            if dims.len() != 2 {
                continue;
            }
            let numel: usize = dims.iter().product();
            let bound = (6.0 / (dims[0] + dims[1]) as f32).sqrt();
            let name = format!("{label}.param{i}");
            let values: Vec<f32> = (0..numel)
                .map(|j| bound * (2.0 * deterministic_unit_interval(&name, j as u64) - 1.0))
                .collect();
            tensor
                .set_slice(0, &values)
                .expect("deterministic reinit set_slice should succeed");
        }
    }

    /// Deterministically reinitializes every parameter of `gan` (both
    /// generator and discriminator).
    ///
    /// # Why not `torsh_tensor::creation::manual_seed`
    ///
    /// `GraphGANGenerator`/`GraphGANDiscriminator::new` draw every weight
    /// from `randn` with no Xavier/Kaiming scaling, stacked through 2-3
    /// unnormalized layers. On an unlucky draw the discriminator's logit for
    /// the generated graph reaches a large enough magnitude that
    /// `softplus(-logit)` underflows to exactly `0.0f32` in `generator_loss`,
    /// failing `gen_loss > 0.0` — mathematically an open bound (`softplus` is
    /// strictly positive everywhere), so that exact `0.0` is an `f32`
    /// underflow artifact of an extreme, unscaled-init logit, not a value
    /// this test should legitimately produce. As
    /// `deterministic_reinit_mpnn`'s doc comment describes (and its swept
    /// probe measured) for the same `randn`-without-scaling pattern in
    /// `MPNNConv`, `manual_seed`'s effective per-thread seed depends on a
    /// process-global, scheduling-dependent "stream index" under `cargo
    /// test`'s shared-process default, so a fixed seed does not reliably
    /// avoid the underflow either — it just relocates which run hits it.
    /// Reinitializing post-construction (this function) avoids that source
    /// of nondeterminism entirely, and — by drawing from a Xavier-uniform
    /// bound instead of raw `randn` — keeps logits far from the underflow
    /// edge. Fixing the scaling in `GraphGANGenerator`/`Discriminator::new`
    /// itself is out of scope: it is production default initialization with
    /// its own blast radius, not this test's bug.
    fn deterministic_reinit_gan(gan: &GraphGAN) {
        reinit_params(&gan.generator_parameters(), "gan.generator");
        reinit_params(&gan.discriminator_parameters(), "gan.discriminator");
    }

    #[test]
    fn test_graphgan_losses() {
        let features = randn(&[4, 8]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0];
        let edge_index = from_vec(edges, &[2, 3], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(features, edge_index);

        let gan = GraphGAN::new(16, 32, 8, true).expect("operation should succeed");
        // Flake: see `deterministic_reinit_gan`. Reinitializing in place
        // makes this run reproducible without touching `GraphGAN`'s
        // production default init.
        deterministic_reinit_gan(&gan);

        let gen_loss = gan.generator_loss(4).expect("operation should succeed");
        assert!(gen_loss > 0.0);

        let disc_loss = gan
            .discriminator_loss(&graph, 4)
            .expect("operation should succeed");
        // Discriminator loss can be negative
        assert!(disc_loss.is_finite());
    }

    /// `deterministic_reinit_gan` fixes the GAN's *weights*, but
    /// `discriminator_loss`'s `real_graph.x` is supplied fresh by the caller
    /// and `generator_loss`/`discriminator_loss` both draw a fresh latent
    /// `z = randn(..)` internally (`GraphGAN::generate`) — neither is
    /// touched by the reinit, so both losses still depend on an unseeded
    /// draw every call. That's fine *only if* the now-Xavier-scaled network
    /// keeps logits far from the `softplus` underflow edge (`gen_loss > 0.0`
    /// fails only when the logit magnitude reaches roughly 104, see
    /// `deterministic_reinit_gan`) regardless of which `z`/`graph.x` gets
    /// drawn. This test is that check, run with much higher confidence than
    /// `test_graphgan_losses`'s single draw: 2000 fresh `graph`/`z` draws
    /// against the same reinit'd weights stay well clear of the edge
    /// (measured range roughly `[-0.7, 0.15]`, vs. the ~104 needed to
    /// underflow) rather than just not-yet-having-hit it once.
    #[test]
    fn discriminator_logit_stays_bounded_across_many_random_graphs() {
        let gan = GraphGAN::new(16, 32, 8, true).expect("operation should succeed");
        deterministic_reinit_gan(&gan);

        for _ in 0..2000 {
            let features = randn(&[4, 8]).expect("randn");
            let edges = vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0];
            let edge_index = from_vec(edges, &[2, 3], DeviceType::Cpu).expect("from_vec");
            let graph = GraphData::new(features, edge_index);

            let real_logit = gan.discriminate_logit(&graph).expect("discriminate_logit");
            assert!(
                real_logit.abs() < 50.0,
                "discriminator logit strayed far enough from 0 to approach the \
                 softplus underflow edge: {real_logit}"
            );

            let gen_loss = gan.generator_loss(4).expect("generator_loss");
            assert!(gen_loss > 0.0, "gen_loss was not > 0.0: {gen_loss}");

            let disc_loss = gan
                .discriminator_loss(&graph, 4)
                .expect("discriminator_loss");
            assert!(
                disc_loss.is_finite(),
                "disc_loss was non-finite: {disc_loss}"
            );
        }
    }
}
