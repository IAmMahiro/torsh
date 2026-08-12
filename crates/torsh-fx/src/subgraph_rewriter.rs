//! Subgraph pattern matching and rewriting

use crate::{FxGraph, Node, TorshResult};
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use std::collections::HashSet;
use torsh_core::error::TorshError;

/// Pattern matcher for subgraphs
pub struct PatternMatcher {
    /// Pattern to match
    pattern: SubgraphPattern,
}

/// Represents a subgraph pattern to match
#[derive(Debug, Clone)]
pub struct SubgraphPattern {
    /// Pattern name
    pub name: String,
    /// Sequence of operations in the pattern
    pub operations: Vec<String>,
    /// Replacement operation
    pub replacement: String,
}

impl SubgraphPattern {
    /// Create a new pattern
    pub fn new(name: String, operations: Vec<String>, replacement: String) -> Self {
        Self {
            name,
            operations,
            replacement,
        }
    }

    /// Create a linear activation fusion pattern
    pub fn linear_relu_fusion() -> Self {
        Self::new(
            "linear_relu_fusion".to_string(),
            vec!["linear".to_string(), "relu".to_string()],
            "linear_relu".to_string(),
        )
    }

    /// Create a conv activation fusion pattern
    pub fn conv_relu_fusion() -> Self {
        Self::new(
            "conv_relu_fusion".to_string(),
            vec!["conv2d".to_string(), "relu".to_string()],
            "conv2d_relu".to_string(),
        )
    }

    /// Create a batch norm fusion pattern
    pub fn conv_bn_fusion() -> Self {
        Self::new(
            "conv_bn_fusion".to_string(),
            vec!["conv2d".to_string(), "batch_norm".to_string()],
            "conv2d_bn".to_string(),
        )
    }

    /// Create a three-operation fusion pattern
    pub fn conv_bn_relu_fusion() -> Self {
        Self::new(
            "conv_bn_relu_fusion".to_string(),
            vec![
                "conv2d".to_string(),
                "batch_norm".to_string(),
                "relu".to_string(),
            ],
            "conv2d_bn_relu".to_string(),
        )
    }
}

/// Match result for a pattern
#[derive(Debug)]
pub struct PatternMatch {
    /// Matched node indices in order
    pub nodes: Vec<NodeIndex>,
    /// Pattern that was matched
    pub pattern: SubgraphPattern,
}

impl PatternMatcher {
    /// Create a new pattern matcher
    pub fn new(pattern: SubgraphPattern) -> Self {
        Self { pattern }
    }

    /// Find all matches of the pattern in the graph
    pub fn find_matches(&self, graph: &FxGraph) -> Vec<PatternMatch> {
        let mut matches = Vec::new();

        // Iterate through all nodes to find potential starting points
        for (start_idx, start_node) in graph.nodes() {
            if let Some(pattern_match) = self.match_pattern_at(graph, start_idx, start_node) {
                matches.push(pattern_match);
            }
        }

        matches
    }

    /// Try to match pattern starting at given node
    fn match_pattern_at(
        &self,
        graph: &FxGraph,
        start_idx: NodeIndex,
        start_node: &Node,
    ) -> Option<PatternMatch> {
        // Check if the first operation matches
        if let Node::Call(op_name, _) = start_node {
            if self.pattern.operations.is_empty() || &self.pattern.operations[0] != op_name {
                return None;
            }
        } else {
            return None;
        }

        // Try to match the complete pattern
        if let Some(matched_nodes) = self.match_sequence(graph, start_idx, &self.pattern.operations)
        {
            return Some(PatternMatch {
                nodes: matched_nodes,
                pattern: self.pattern.clone(),
            });
        }

        None
    }

    /// Match a sequence of operations starting from a node
    fn match_sequence(
        &self,
        graph: &FxGraph,
        start_idx: NodeIndex,
        operations: &[String],
    ) -> Option<Vec<NodeIndex>> {
        if operations.is_empty() {
            return Some(vec![]);
        }

        let mut current_nodes = vec![start_idx];
        let mut matched_nodes = vec![start_idx];

        // Match subsequent operations
        for expected_op in &operations[1..] {
            let mut next_nodes = Vec::new();

            for &current_idx in &current_nodes {
                // Find successors of current node
                let successors: Vec<_> = graph
                    .graph
                    .neighbors_directed(current_idx, petgraph::Direction::Outgoing)
                    .collect();

                for successor_idx in successors {
                    if let Some(Node::Call(op_name, _)) = graph.get_node(successor_idx) {
                        if op_name == expected_op {
                            next_nodes.push(successor_idx);
                            matched_nodes.push(successor_idx);
                        }
                    }
                }
            }

            if next_nodes.is_empty() {
                return None; // Pattern doesn't match
            }

            current_nodes = next_nodes;
        }

        Some(matched_nodes)
    }
}

/// Subgraph rewriter for applying pattern transformations
pub struct SubgraphRewriter {
    patterns: Vec<SubgraphPattern>,
}

impl SubgraphRewriter {
    /// Create a new rewriter
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
        }
    }

    /// Add a pattern to the rewriter
    pub fn add_pattern(&mut self, pattern: SubgraphPattern) {
        self.patterns.push(pattern);
    }

    /// Create a rewriter with common fusion patterns
    pub fn with_common_fusions() -> Self {
        let mut rewriter = Self::new();
        rewriter.add_pattern(SubgraphPattern::linear_relu_fusion());
        rewriter.add_pattern(SubgraphPattern::conv_relu_fusion());
        rewriter.add_pattern(SubgraphPattern::conv_bn_fusion());
        rewriter.add_pattern(SubgraphPattern::conv_bn_relu_fusion());
        rewriter
    }

    /// Apply all patterns to the graph
    pub fn apply(&self, graph: &mut FxGraph) -> TorshResult<usize> {
        let mut total_replacements = 0;

        for pattern in &self.patterns {
            let replacements = self.apply_pattern(graph, pattern)?;
            total_replacements += replacements;
        }

        Ok(total_replacements)
    }

    /// Apply a specific pattern to the graph
    ///
    /// Matches are recomputed after every rewrite: applying one rewrite rebuilds the
    /// graph, which would leave any previously collected `NodeIndex` dangling.
    fn apply_pattern(&self, graph: &mut FxGraph, pattern: &SubgraphPattern) -> TorshResult<usize> {
        let matcher = PatternMatcher::new(pattern.clone());
        let mut replacements = 0;
        // The replacement operation never matches the pattern's first operation, so
        // this terminates; the budget only guards against pathological patterns.
        let mut budget = graph.node_count() + 1;

        while budget > 0 {
            budget -= 1;

            let next_match = matcher
                .find_matches(graph)
                .into_iter()
                .find(|candidate| Self::is_legal_match(graph, candidate));

            match next_match {
                Some(pattern_match) => {
                    self.replace_match(graph, &pattern_match)?;
                    replacements += 1;
                }
                None => break,
            }
        }

        Ok(replacements)
    }

    /// Check that a match can be fused without changing the meaning of the graph
    ///
    /// A match is legal when it is a simple chain of distinct nodes whose length
    /// equals the pattern length, every consecutive pair is connected, and no node
    /// except the last one is consumed from outside the match (fusing a value that
    /// other operations still read would silently feed them the fused result).
    fn is_legal_match(graph: &FxGraph, pattern_match: &PatternMatch) -> bool {
        let nodes = &pattern_match.nodes;
        if nodes.len() != pattern_match.pattern.operations.len() || nodes.is_empty() {
            return false;
        }

        let unique: HashSet<NodeIndex> = nodes.iter().copied().collect();
        if unique.len() != nodes.len() {
            return false;
        }

        for (position, &node_idx) in nodes.iter().enumerate() {
            match graph.get_node(node_idx) {
                Some(Node::Call(op_name, _))
                    if *op_name == pattern_match.pattern.operations[position] => {}
                _ => return false,
            }

            if position + 1 < nodes.len() {
                let next_idx = nodes[position + 1];
                if graph.graph.find_edge(node_idx, next_idx).is_none() {
                    return false;
                }
                // Everything the fused nodes produce is consumed inside the match.
                let consumers: HashSet<NodeIndex> = graph
                    .graph
                    .neighbors_directed(node_idx, petgraph::Direction::Outgoing)
                    .collect();
                if consumers.len() != 1 || !consumers.contains(&next_idx) {
                    return false;
                }
            }
        }

        true
    }

    /// Replace a matched pattern with the replacement operation
    fn replace_match(&self, graph: &mut FxGraph, pattern_match: &PatternMatch) -> TorshResult<()> {
        if pattern_match.nodes.is_empty() {
            return Ok(());
        }

        let first_node_idx = pattern_match.nodes[0];
        let matched: HashSet<NodeIndex> = pattern_match.nodes.iter().copied().collect();

        // Get the arguments from the first node
        let mut args = if let Some(Node::Call(_, args)) = graph.get_node(first_node_idx) {
            args.clone()
        } else {
            vec![]
        };

        // Rewire the folded nodes' external neighbours onto the fused node before
        // anything is deleted: incoming edges from outside the pattern (weights,
        // running statistics, ...) are real inputs of the fused operation and must
        // not be dropped along with the node that used to consume them.
        for &node_idx in &pattern_match.nodes[1..] {
            let incoming: Vec<(NodeIndex, crate::Edge)> = graph
                .graph
                .edges_directed(node_idx, petgraph::Direction::Incoming)
                .filter(|edge| !matched.contains(&edge.source()))
                .map(|edge| (edge.source(), edge.weight().clone()))
                .collect();
            for (source_idx, weight) in incoming {
                if !args.contains(&weight.name) {
                    args.push(weight.name.clone());
                }
                if graph.graph.find_edge(source_idx, first_node_idx).is_none() {
                    graph.graph.add_edge(source_idx, first_node_idx, weight);
                }
            }

            let outgoing: Vec<(NodeIndex, crate::Edge)> = graph
                .graph
                .edges_directed(node_idx, petgraph::Direction::Outgoing)
                .filter(|edge| !matched.contains(&edge.target()))
                .map(|edge| (edge.target(), edge.weight().clone()))
                .collect();
            for (target_idx, weight) in outgoing {
                if graph.graph.find_edge(first_node_idx, target_idx).is_none() {
                    graph.graph.add_edge(first_node_idx, target_idx, weight);
                }
            }

            // The fused node now produces what this node produced.
            graph.redirect_boundary_node(node_idx, first_node_idx);
        }

        // Replace the first node with the fused operation
        graph.graph[first_node_idx] = Node::Call(pattern_match.pattern.replacement.clone(), args);

        // Remove the folded nodes in one batch through FxGraph, which keeps the
        // input/output lists valid.
        let to_remove: HashSet<NodeIndex> = pattern_match.nodes[1..].iter().copied().collect();
        if !to_remove.is_empty() {
            graph.remove_nodes(&to_remove);
        }

        Ok(())
    }
}

impl Default for SubgraphRewriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function for replacing patterns
pub fn replace_pattern(graph: &mut FxGraph, pattern: &str, _replacement: &str) -> TorshResult<()> {
    let pattern_obj = match pattern {
        "linear->relu" => SubgraphPattern::linear_relu_fusion(),
        "conv2d->relu" => SubgraphPattern::conv_relu_fusion(),
        "conv2d->batch_norm" => SubgraphPattern::conv_bn_fusion(),
        "conv2d->batch_norm->relu" => SubgraphPattern::conv_bn_relu_fusion(),
        _ => {
            return Err(TorshError::InvalidArgument(format!(
                "Unknown pattern: {}",
                pattern
            )));
        }
    };

    let mut rewriter = SubgraphRewriter::new();
    rewriter.add_pattern(pattern_obj);
    rewriter.apply(graph)?;

    Ok(())
}

/// Apply common fusion optimizations
pub fn apply_fusion_optimizations(graph: &mut FxGraph) -> TorshResult<usize> {
    let rewriter = SubgraphRewriter::with_common_fusions();
    rewriter.apply(graph)
}

/// Replace specific operation sequences
pub fn replace_operation_sequence(
    graph: &mut FxGraph,
    sequence: &[&str],
    replacement: &str,
) -> TorshResult<()> {
    let operations: Vec<String> = sequence.iter().map(|s| s.to_string()).collect();
    let pattern = SubgraphPattern::new(
        "custom_pattern".to_string(),
        operations,
        replacement.to_string(),
    );

    let mut rewriter = SubgraphRewriter::new();
    rewriter.add_pattern(pattern);
    rewriter.apply(graph)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracer::ModuleTracer;

    #[test]
    fn test_pattern_creation() {
        let pattern = SubgraphPattern::linear_relu_fusion();
        assert_eq!(pattern.name, "linear_relu_fusion");
        assert_eq!(pattern.operations, vec!["linear", "relu"]);
        assert_eq!(pattern.replacement, "linear_relu");
    }

    #[test]
    fn test_pattern_matching() {
        let mut tracer = ModuleTracer::new();
        tracer.add_input("x");
        tracer.add_call("linear", vec!["x".to_string()]);
        tracer.add_call("relu", vec!["node_0".to_string()]);
        tracer.add_output("node_1");
        let graph = tracer.finalize();

        let pattern = SubgraphPattern::linear_relu_fusion();
        let matcher = PatternMatcher::new(pattern);
        let matches = matcher.find_matches(&graph);

        assert!(!matches.is_empty());
    }

    #[test]
    fn test_subgraph_rewriting() {
        let mut tracer = ModuleTracer::new();
        tracer.add_input("x");
        tracer.add_call("linear", vec!["x".to_string()]);
        tracer.add_call("relu", vec!["node_0".to_string()]);
        tracer.add_output("node_1");
        let mut graph = tracer.finalize();

        let original_node_count = graph.node_count();

        let mut rewriter = SubgraphRewriter::new();
        rewriter.add_pattern(SubgraphPattern::linear_relu_fusion());
        let replacements = rewriter.apply(&mut graph).unwrap();

        assert!(replacements > 0);
        // Node count should decrease due to fusion
        assert!(graph.node_count() < original_node_count);
    }

    #[test]
    fn test_convenience_functions() {
        let mut tracer = ModuleTracer::new();
        tracer.add_input("x");
        tracer.add_call("linear", vec!["x".to_string()]);
        tracer.add_call("relu", vec!["node_0".to_string()]);
        tracer.add_output("node_1");
        let mut graph = tracer.finalize();

        // Test string-based pattern replacement
        assert!(replace_pattern(&mut graph, "linear->relu", "linear_relu").is_ok());

        // Test operation sequence replacement
        let mut tracer2 = ModuleTracer::new();
        tracer2.add_input("x");
        tracer2.add_call("conv2d", vec!["x".to_string()]);
        tracer2.add_call("batch_norm", vec!["node_0".to_string()]);
        tracer2.add_call("relu", vec!["node_1".to_string()]);
        tracer2.add_output("node_2");
        let mut graph2 = tracer2.finalize();

        assert!(replace_operation_sequence(
            &mut graph2,
            &["conv2d", "batch_norm", "relu"],
            "conv2d_bn_relu"
        )
        .is_ok());
    }

    #[test]
    fn test_fusion_optimizations() {
        let mut tracer = ModuleTracer::new();
        tracer.add_input("x");
        tracer.add_call("conv2d", vec!["x".to_string()]);
        tracer.add_call("relu", vec!["node_0".to_string()]);
        tracer.add_output("node_1");
        let mut graph = tracer.finalize();

        let _replacements = apply_fusion_optimizations(&mut graph).unwrap();
        // Should run without error - replacements is a valid count
    }
}
