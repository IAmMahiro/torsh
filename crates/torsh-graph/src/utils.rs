//! Graph utilities and algorithms

use torsh_core::device::DeviceType;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::{
    creation::{from_vec, zeros},
    Tensor,
};
// Direct implementation of graph algorithms for now

/// Helper function to convert tensor to 2D Vec format
/// Assumes tensor is of shape \[rows, cols\] and returns `Vec<Vec<T>>`
///
/// # Errors
/// Returns [`TorshError::InvalidArgument`] when the tensor is not 2-dimensional.
pub fn tensor_to_vec2<T: Clone + torsh_core::TensorElement>(
    tensor: &Tensor<T>,
) -> Result<Vec<Vec<T>>> {
    let data = tensor.to_vec()?;
    let tensor_shape = tensor.shape();
    let shape = tensor_shape.dims();

    if shape.len() != 2 {
        return Err(TorshError::InvalidArgument(format!(
            "tensor_to_vec2 requires a 2D tensor, got {shape:?}"
        )));
    }

    let rows = shape[0];
    let cols = shape[1];
    let mut result = Vec::with_capacity(rows);

    for i in 0..rows {
        let start = i * cols;
        let end = start + cols;
        result.push(data[start..end].to_vec());
    }

    Ok(result)
}

/// Extract the `(source, target)` rows of an edge index tensor.
///
/// The tensor must have shape `[2, num_edges]`; anything else is a hard error
/// rather than a panic, because `edge_index` is usually caller-supplied data.
pub(crate) fn edge_rows(edge_index: &Tensor) -> Result<(Vec<f32>, Vec<f32>)> {
    let edge_data = tensor_to_vec2::<f32>(edge_index)?;
    if edge_data.len() < 2 {
        return Err(TorshError::InvalidArgument(format!(
            "edge_index must have shape [2, num_edges], got {:?}",
            edge_index.shape().dims()
        )));
    }
    let src = edge_data[0].clone();
    let dst = edge_data[1].clone();
    Ok((src, dst))
}

/// Build a dense, symmetric, de-duplicated adjacency matrix together with the
/// degree vector obtained as the **row sums of that matrix**.
///
/// Deriving degrees from the finished adjacency matrix (instead of counting
/// `edge_index` columns) makes the result identical whether the caller lists
/// each undirected edge once or in both directions, which is the convention
/// PyTorch-Geometric-style `edge_index` tensors use.
///
/// `add_self_loops` adds the identity before the row sums are taken, which is
/// exactly the renormalization trick of Kipf & Welling.
fn dense_adjacency(
    edge_index: &Tensor,
    num_nodes: usize,
    add_self_loops: bool,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let (src_row, dst_row) = edge_rows(edge_index)?;

    let mut adjacency = vec![0.0f32; num_nodes * num_nodes];
    for (&src, &dst) in src_row.iter().zip(dst_row.iter()) {
        if src < 0.0 || dst < 0.0 {
            return Err(TorshError::InvalidArgument(format!(
                "edge_index contains a negative node id ({src}, {dst})"
            )));
        }
        let src = src as usize;
        let dst = dst as usize;
        if src < num_nodes && dst < num_nodes {
            adjacency[src * num_nodes + dst] = 1.0;
            adjacency[dst * num_nodes + src] = 1.0;
        }
    }

    if add_self_loops {
        for i in 0..num_nodes {
            adjacency[i * num_nodes + i] = 1.0;
        }
    }

    let mut degrees = vec![0.0f32; num_nodes];
    for (i, degree) in degrees.iter_mut().enumerate() {
        *degree = adjacency[i * num_nodes..(i + 1) * num_nodes].iter().sum();
    }

    Ok((adjacency, degrees))
}

/// Compute graph Laplacian
///
/// * `normalized == false` returns the combinatorial Laplacian `L = D - A`.
/// * `normalized == true` returns `L = I - D^(-1/2) A D^(-1/2)`.
///
/// Degrees are the row sums of the symmetrized, de-duplicated adjacency matrix,
/// so listing an undirected edge once or in both directions yields the same
/// operator.
///
/// # Errors
/// Returns an error when `edge_index` is not a `[2, num_edges]` tensor or
/// contains negative node ids.
pub fn graph_laplacian(edge_index: &Tensor, num_nodes: usize, normalized: bool) -> Result<Tensor> {
    let (adjacency, degrees) = dense_adjacency(edge_index, num_nodes, false)?;
    let mut laplacian_data = vec![0.0f32; num_nodes * num_nodes];

    if normalized {
        // Normalized Laplacian: L = I - D^(-1/2) @ A @ D^(-1/2)
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                let a_ij = adjacency[i * num_nodes + j];
                if i == j {
                    // Identity diagonal minus the (possible) normalized self-loop.
                    // The normalized self-loop term is A[i][i] / d_i, not
                    // A[i][i] / sqrt(d_i).
                    laplacian_data[i * num_nodes + j] = if degrees[i] > 0.0 {
                        1.0 - a_ij / degrees[i]
                    } else {
                        1.0
                    };
                } else if a_ij > 0.0 && degrees[i] > 0.0 && degrees[j] > 0.0 {
                    laplacian_data[i * num_nodes + j] =
                        -a_ij / (degrees[i].sqrt() * degrees[j].sqrt());
                }
            }
        }
    } else {
        // Unnormalized Laplacian: L = D - A (row sums are exactly zero)
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                let diagonal = if i == j { degrees[i] } else { 0.0 };
                laplacian_data[i * num_nodes + j] = diagonal - adjacency[i * num_nodes + j];
            }
        }
    }

    Ok(from_vec(
        laplacian_data,
        &[num_nodes, num_nodes],
        DeviceType::Cpu,
    )?)
}

/// Compute the Kipf & Welling GCN propagation operator
/// `A_hat = D~^(-1/2) (A + I) D~^(-1/2)`.
///
/// This is the operator a graph convolution propagates with; it is **not** the
/// Laplacian. Self-loops are added before the degrees are computed (the
/// "renormalization trick"), so `A_hat` has non-negative entries and spectral
/// radius at most one.
///
/// # Errors
/// Returns an error when `edge_index` is not a `[2, num_edges]` tensor or
/// contains negative node ids.
pub fn gcn_norm(edge_index: &Tensor, num_nodes: usize) -> Result<Tensor> {
    let (adjacency, degrees) = dense_adjacency(edge_index, num_nodes, true)?;
    let mut normalized = vec![0.0f32; num_nodes * num_nodes];

    for i in 0..num_nodes {
        for j in 0..num_nodes {
            let a_ij = adjacency[i * num_nodes + j];
            if a_ij > 0.0 && degrees[i] > 0.0 && degrees[j] > 0.0 {
                normalized[i * num_nodes + j] = a_ij / (degrees[i].sqrt() * degrees[j].sqrt());
            }
        }
    }

    Ok(from_vec(
        normalized,
        &[num_nodes, num_nodes],
        DeviceType::Cpu,
    )?)
}

/// Compute degree matrix
///
/// The diagonal holds the row sums of the symmetrized, de-duplicated adjacency
/// matrix, matching [`graph_laplacian`].
///
/// # Errors
/// Returns an error when `edge_index` is malformed.
pub fn degree_matrix(edge_index: &Tensor, num_nodes: usize) -> Result<Tensor> {
    let (_, degrees) = dense_adjacency(edge_index, num_nodes, false)?;

    // Create diagonal degree matrix
    let mut degree_matrix = zeros(&[num_nodes, num_nodes])?;
    for (i, &degree) in degrees.iter().enumerate() {
        degree_matrix.set_item(&[i, i], degree)?;
    }

    Ok(degree_matrix)
}

/// Graph connectivity utilities
pub mod connectivity {
    use super::*;
    use std::collections::VecDeque;

    /// Check if graph is connected
    pub fn is_connected(edge_index: &Tensor, num_nodes: usize) -> Result<bool> {
        if num_nodes <= 1 {
            return Ok(true);
        }

        // Build adjacency list
        let adjacency_list = build_adjacency_list(edge_index, num_nodes)?;

        // Perform BFS from node 0
        let mut visited = vec![false; num_nodes];
        let mut queue = VecDeque::new();
        queue.push_back(0);
        visited[0] = true;
        let mut visited_count = 1;

        while let Some(node) = queue.pop_front() {
            for &neighbor in &adjacency_list[node] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    visited_count += 1;
                    queue.push_back(neighbor);
                }
            }
        }

        Ok(visited_count == num_nodes)
    }

    /// Get connected components using DFS
    pub fn connected_components(edge_index: &Tensor, num_nodes: usize) -> Result<Vec<Vec<usize>>> {
        let adjacency_list = build_adjacency_list(edge_index, num_nodes)?;
        let mut visited = vec![false; num_nodes];
        let mut components = Vec::new();

        for start_node in 0..num_nodes {
            if !visited[start_node] {
                let mut component = Vec::new();
                let mut stack = vec![start_node];

                while let Some(node) = stack.pop() {
                    if !visited[node] {
                        visited[node] = true;
                        component.push(node);

                        for &neighbor in &adjacency_list[node] {
                            if !visited[neighbor] {
                                stack.push(neighbor);
                            }
                        }
                    }
                }

                components.push(component);
            }
        }

        Ok(components)
    }

    /// Get largest connected component
    pub fn largest_component(edge_index: &Tensor, num_nodes: usize) -> Result<Vec<usize>> {
        let components = connected_components(edge_index, num_nodes)?;
        Ok(components
            .into_iter()
            .max_by_key(|component| component.len())
            .unwrap_or_default())
    }

    /// Build adjacency list from edge index
    pub(crate) fn build_adjacency_list(
        edge_index: &Tensor,
        num_nodes: usize,
    ) -> Result<Vec<Vec<usize>>> {
        let mut adjacency_list = vec![Vec::new(); num_nodes];
        let (src_row, dst_row) = super::edge_rows(edge_index)?;

        for (&src, &dst) in src_row.iter().zip(dst_row.iter()) {
            if src < 0.0 || dst < 0.0 {
                continue;
            }
            let src = src as usize;
            let dst = dst as usize;

            if src < num_nodes && dst < num_nodes {
                adjacency_list[src].push(dst);
                adjacency_list[dst].push(src); // Undirected graph
            }
        }

        // Remove duplicates and sort
        for neighbors in &mut adjacency_list {
            neighbors.sort();
            neighbors.dedup();
        }

        Ok(adjacency_list)
    }
}

/// Graph metrics and statistics
pub mod metrics {
    use super::*;
    use std::collections::VecDeque;

    /// Compute node centrality measures
    pub fn node_centrality(edge_index: &Tensor, num_nodes: usize) -> Result<CentralityMeasures> {
        let adjacency_list = super::connectivity::build_adjacency_list(edge_index, num_nodes)?;

        // Degree centrality
        let degree_values: Vec<f64> = adjacency_list
            .iter()
            .map(|neighbors| neighbors.len() as f64)
            .collect();
        let degree_f32: Vec<f32> = degree_values.into_iter().map(|x| x as f32).collect();
        let degree = from_vec(degree_f32, &[num_nodes], DeviceType::Cpu)?;

        // Betweenness centrality (simplified implementation)
        let betweenness_values = compute_betweenness_centrality(&adjacency_list, num_nodes);
        let betweenness_f32: Vec<f32> = betweenness_values.into_iter().map(|x| x as f32).collect();
        let betweenness = from_vec(betweenness_f32, &[num_nodes], DeviceType::Cpu)?;

        // Closeness centrality
        let closeness_values = compute_closeness_centrality(&adjacency_list, num_nodes);
        let closeness_f32: Vec<f32> = closeness_values.into_iter().map(|x| x as f32).collect();
        let closeness = from_vec(closeness_f32, &[num_nodes], DeviceType::Cpu)?;

        // Eigenvector centrality (power iteration approximation)
        let eigenvector_values = compute_eigenvector_centrality(&adjacency_list, num_nodes);
        let eigenvector_f32: Vec<f32> = eigenvector_values.into_iter().map(|x| x as f32).collect();
        let eigenvector = from_vec(eigenvector_f32, &[num_nodes], DeviceType::Cpu)?;

        Ok(CentralityMeasures {
            degree,
            betweenness,
            closeness,
            eigenvector,
        })
    }

    /// Centrality measures container
    pub struct CentralityMeasures {
        pub degree: Tensor,
        pub betweenness: Tensor,
        pub closeness: Tensor,
        pub eigenvector: Tensor,
    }

    /// Compute clustering coefficient
    pub fn clustering_coefficient(edge_index: &Tensor, num_nodes: usize) -> Result<Tensor> {
        let adjacency_list = super::connectivity::build_adjacency_list(edge_index, num_nodes)?;
        let mut clustering_coeffs = Vec::with_capacity(num_nodes);

        for node in 0..num_nodes {
            let neighbors = &adjacency_list[node];
            let degree = neighbors.len();

            if degree < 2 {
                clustering_coeffs.push(0.0);
                continue;
            }

            // Count triangles
            let mut triangles = 0;
            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    let neighbor1 = neighbors[i];
                    let neighbor2 = neighbors[j];

                    // Check if neighbor1 and neighbor2 are connected
                    if adjacency_list[neighbor1].contains(&neighbor2) {
                        triangles += 1;
                    }
                }
            }

            // Clustering coefficient = 2 * triangles / (degree * (degree - 1))
            let max_edges = degree * (degree - 1) / 2;
            let clustering = if max_edges > 0 {
                triangles as f64 / max_edges as f64
            } else {
                0.0
            };

            clustering_coeffs.push(clustering);
        }

        let coeffs_f32: Vec<f32> = clustering_coeffs.into_iter().map(|x| x as f32).collect();
        Ok(from_vec(coeffs_f32, &[num_nodes], DeviceType::Cpu)?)
    }

    /// Compute graph diameter
    pub fn graph_diameter(edge_index: &Tensor, num_nodes: usize) -> Result<usize> {
        let adjacency_list = super::connectivity::build_adjacency_list(edge_index, num_nodes)?;
        let mut max_distance = 0;

        for start_node in 0..num_nodes {
            let distances = bfs_distances(&adjacency_list, start_node, num_nodes);
            for &distance in &distances {
                if distance != usize::MAX && distance > max_distance {
                    max_distance = distance;
                }
            }
        }

        Ok(max_distance)
    }

    /// Compute betweenness centrality using Brandes' algorithm (simplified)
    fn compute_betweenness_centrality(adjacency_list: &[Vec<usize>], num_nodes: usize) -> Vec<f64> {
        let mut betweenness = vec![0.0; num_nodes];

        for s in 0..num_nodes {
            // Single-source shortest-path problem
            let mut stack = Vec::new();
            let mut paths = vec![Vec::new(); num_nodes];
            let mut sigma = vec![0.0; num_nodes];
            let mut distances = vec![-1; num_nodes];
            let mut delta = vec![0.0; num_nodes];

            sigma[s] = 1.0;
            distances[s] = 0;

            let mut queue = VecDeque::new();
            queue.push_back(s);

            while let Some(v) = queue.pop_front() {
                stack.push(v);

                for &w in &adjacency_list[v] {
                    // First time we found shortest path to w?
                    if distances[w] < 0 {
                        queue.push_back(w);
                        distances[w] = distances[v] + 1;
                    }
                    // Shortest path to w via v?
                    if distances[w] == distances[v] + 1 {
                        sigma[w] += sigma[v];
                        paths[w].push(v);
                    }
                }
            }

            // Accumulation
            while let Some(w) = stack.pop() {
                for &v in &paths[w] {
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                }
                if w != s {
                    betweenness[w] += delta[w];
                }
            }
        }

        // Normalize
        let normalization = if num_nodes > 2 {
            2.0 / ((num_nodes - 1) * (num_nodes - 2)) as f64
        } else {
            1.0
        };

        betweenness.iter().map(|&x| x * normalization).collect()
    }

    /// Compute closeness centrality
    fn compute_closeness_centrality(adjacency_list: &[Vec<usize>], num_nodes: usize) -> Vec<f64> {
        let mut closeness = Vec::with_capacity(num_nodes);

        for node in 0..num_nodes {
            let distances = bfs_distances(adjacency_list, node, num_nodes);
            let mut sum_distances = 0.0;
            let mut reachable_nodes = 0;

            for &distance in &distances {
                if distance != usize::MAX {
                    sum_distances += distance as f64;
                    reachable_nodes += 1;
                }
            }

            let closeness_value = if sum_distances > 0.0 && reachable_nodes > 1 {
                (reachable_nodes - 1) as f64 / sum_distances
            } else {
                0.0
            };

            closeness.push(closeness_value);
        }

        closeness
    }

    /// Compute eigenvector centrality using power iteration
    fn compute_eigenvector_centrality(adjacency_list: &[Vec<usize>], num_nodes: usize) -> Vec<f64> {
        let mut centrality = vec![1.0; num_nodes];
        let iterations = 100;
        let tolerance = 1e-6;

        for _ in 0..iterations {
            let mut new_centrality = vec![0.0; num_nodes];

            // Matrix-vector multiplication with adjacency matrix
            for node in 0..num_nodes {
                for &neighbor in &adjacency_list[node] {
                    new_centrality[node] += centrality[neighbor];
                }
            }

            // Normalize
            let norm: f64 = new_centrality.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                for value in &mut new_centrality {
                    *value /= norm;
                }
            }

            // Check convergence
            let diff: f64 = centrality
                .iter()
                .zip(&new_centrality)
                .map(|(&old, &new)| (old - new).abs())
                .sum();

            centrality = new_centrality;

            if diff < tolerance {
                break;
            }
        }

        centrality
    }

    /// Compute shortest distances from a source node using BFS
    fn bfs_distances(adjacency_list: &[Vec<usize>], source: usize, num_nodes: usize) -> Vec<usize> {
        let mut distances = vec![usize::MAX; num_nodes];
        let mut queue = VecDeque::new();

        distances[source] = 0;
        queue.push_back(source);

        while let Some(node) = queue.pop_front() {
            for &neighbor in &adjacency_list[node] {
                if distances[neighbor] == usize::MAX {
                    distances[neighbor] = distances[node] + 1;
                    queue.push_back(neighbor);
                }
            }
        }

        distances
    }
}

/// Graph sampling utilities
pub mod sampling {
    use super::*;
    use scirs2_core::random::Random;
    use scirs2_core::RngExt;

    /// Sample neighbors for GraphSAGE
    pub fn neighbor_sampling(
        edge_index: &Tensor,
        node_idx: &[usize],
        num_neighbors: usize,
    ) -> Result<Vec<Vec<usize>>> {
        let num_nodes = node_idx.iter().max().unwrap_or(&0) + 1;
        let adjacency_list = super::connectivity::build_adjacency_list(edge_index, num_nodes)?;
        let mut rng = Random::seed(42);
        let mut sampled_neighbors = Vec::with_capacity(node_idx.len());

        for &node in node_idx {
            let neighbors = &adjacency_list[node];

            if neighbors.len() <= num_neighbors {
                // Return all neighbors if we have fewer than requested
                sampled_neighbors.push(neighbors.clone());
            } else {
                // Randomly sample without replacement
                let mut indices: Vec<usize> = (0..neighbors.len()).collect();

                // Fisher-Yates shuffle for first num_neighbors elements
                for i in 0..num_neighbors {
                    let j = rng.gen_range(i..indices.len());
                    indices.swap(i, j);
                }

                let sampled: Vec<usize> = indices[..num_neighbors]
                    .iter()
                    .map(|&i| neighbors[i])
                    .collect();

                sampled_neighbors.push(sampled);
            }
        }

        Ok(sampled_neighbors)
    }

    /// Random walk sampling
    pub fn random_walk(
        edge_index: &Tensor,
        start_nodes: &[usize],
        walk_length: usize,
    ) -> Result<Vec<Vec<usize>>> {
        let num_nodes = start_nodes.iter().max().unwrap_or(&0) + 1;
        let adjacency_list = super::connectivity::build_adjacency_list(edge_index, num_nodes)?;
        let mut rng = Random::seed(42);
        let mut walks = Vec::with_capacity(start_nodes.len());

        for &start_node in start_nodes {
            let mut walk = vec![start_node];
            let mut current_node = start_node;

            for _ in 1..walk_length {
                let neighbors = &adjacency_list[current_node];

                if neighbors.is_empty() {
                    break; // Dead end
                }

                // Randomly select next node
                let next_idx = rng.gen_range(0..neighbors.len());
                current_node = neighbors[next_idx];
                walk.push(current_node);
            }

            walks.push(walk);
        }

        Ok(walks)
    }

    /// Subgraph sampling using random node sampling
    pub fn subgraph_sampling(
        edge_index: &Tensor,
        num_nodes: usize,
        sample_size: usize,
    ) -> Result<(Tensor, Vec<usize>)> {
        let mut rng = Random::seed(42);

        // Sample nodes randomly
        let mut sampled_nodes: Vec<usize> = (0..num_nodes).collect();

        // Fisher-Yates shuffle
        for i in 0..sample_size.min(num_nodes) {
            let j = rng.gen_range(i..sampled_nodes.len());
            sampled_nodes.swap(i, j);
        }
        sampled_nodes.truncate(sample_size.min(num_nodes));
        sampled_nodes.sort();

        // Create node mapping
        let mut node_map = std::collections::HashMap::new();
        for (new_idx, &old_idx) in sampled_nodes.iter().enumerate() {
            node_map.insert(old_idx, new_idx);
        }

        // Extract subgraph edges
        let (src_row, dst_row) = super::edge_rows(edge_index)?;
        let mut subgraph_edges = Vec::new();

        for (&src, &dst) in src_row.iter().zip(dst_row.iter()) {
            if src < 0.0 || dst < 0.0 {
                continue;
            }
            let src = src as usize;
            let dst = dst as usize;

            // Include edge if both nodes are in the sampled set
            if let (Some(&new_src), Some(&new_dst)) = (node_map.get(&src), node_map.get(&dst)) {
                subgraph_edges.push([new_src as i64, new_dst as i64]);
            }
        }

        // Create edge tensor
        let subgraph_edge_index = if subgraph_edges.is_empty() {
            zeros(&[2, 0])?
        } else {
            let num_edges = subgraph_edges.len();
            let mut edge_vec = Vec::with_capacity(2 * num_edges);

            for edge in &subgraph_edges {
                edge_vec.push(edge[0] as f32);
            }
            for edge in &subgraph_edges {
                edge_vec.push(edge[1] as f32);
            }

            from_vec(edge_vec, &[2, num_edges], DeviceType::Cpu)?
        };

        Ok((subgraph_edge_index, sampled_nodes))
    }

    /// FastGCN sampling for efficient graph convolution
    pub fn fastgcn_sampling(
        edge_index: &Tensor,
        layer_sizes: &[usize],
        num_nodes: usize,
    ) -> Result<Vec<Vec<usize>>> {
        let mut rng = Random::seed(42);
        let adjacency_list = super::connectivity::build_adjacency_list(edge_index, num_nodes)?;

        // Compute importance sampling probabilities based on node degrees
        let degrees: Vec<f64> = adjacency_list
            .iter()
            .map(|neighbors| neighbors.len() as f64)
            .collect();

        let total_degree: f64 = degrees.iter().sum();
        let probabilities: Vec<f64> = degrees.iter().map(|&d| d / total_degree).collect();

        let mut sampled_layers = Vec::with_capacity(layer_sizes.len());

        for &layer_size in layer_sizes {
            let mut sampled_nodes = Vec::new();

            // Importance sampling
            for _ in 0..layer_size.min(num_nodes) {
                let mut cumsum = 0.0;
                let random_val = rng.random::<f64>();

                for (node, &prob) in probabilities.iter().enumerate() {
                    cumsum += prob;
                    if random_val <= cumsum {
                        sampled_nodes.push(node);
                        break;
                    }
                }
            }

            sampled_nodes.sort();
            sampled_nodes.dedup();
            sampled_layers.push(sampled_nodes);
        }

        Ok(sampled_layers)
    }
}

/// Memory-efficient graph operations: sparse COO representations, graph
/// Laplacians, adaptive coarsening, and chunked neighbor aggregation.
pub mod memory_efficient;
