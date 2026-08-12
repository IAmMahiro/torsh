//! Graph transformation passes

use crate::{FxGraph, Node, TorshResult};
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use std::collections::{HashMap, HashSet};

/// Pass trait for graph transformations
pub trait Pass {
    /// Apply the pass to the graph
    fn apply(&self, graph: &mut FxGraph) -> TorshResult<()>;

    /// Get the name of this pass
    fn name(&self) -> &str;
}

/// Operation fusion pass
///
/// Fuses `linear -> relu` and `conv2d -> relu` chains into the `linear_relu` /
/// `conv2d_relu` operations understood by the interpreter. A fusion is only legal
/// when the producer feeds nothing but the activation and the activation consumes
/// nothing but the producer; otherwise the activation would be applied to values
/// that other consumers still expect unactivated.
pub struct OperationFusionPass;

impl OperationFusionPass {
    /// Find the next legal producer/activation pair to fuse.
    ///
    /// # Arguments
    /// * `graph` - Graph to scan
    ///
    /// # Returns
    /// `(producer, activation, fused_op_name)` for the first legal candidate.
    fn find_candidate(graph: &FxGraph) -> Option<(NodeIndex, NodeIndex, String)> {
        for (relu_idx, node) in graph.nodes() {
            let Node::Call(op_name, _) = node else {
                continue;
            };
            if op_name != "relu" {
                continue;
            }

            // The activation must consume exactly one value.
            let incoming: Vec<_> = graph
                .graph
                .edges_directed(relu_idx, petgraph::Direction::Incoming)
                .collect();
            if incoming.len() != 1 {
                continue;
            }
            let producer_idx = incoming[0].source();

            let Some(Node::Call(producer_op, _)) = graph.get_node(producer_idx) else {
                continue;
            };
            if producer_op != "linear" && producer_op != "conv2d" {
                continue;
            }

            // The producer must not be consumed by anything else, otherwise the
            // unrelated consumers would silently observe relu(producer(..)).
            let out_degree = graph
                .graph
                .edges_directed(producer_idx, petgraph::Direction::Outgoing)
                .count();
            if out_degree != 1 {
                continue;
            }

            return Some((producer_idx, relu_idx, format!("{producer_op}_relu")));
        }

        None
    }
}

impl Pass for OperationFusionPass {
    fn apply(&self, graph: &mut FxGraph) -> TorshResult<()> {
        // Each fusion rebuilds the graph, so candidates are re-discovered every round.
        // The fused operation name never matches the patterns above, so the loop is
        // bounded by the number of nodes.
        let mut budget = graph.node_count() + 1;

        while let Some((producer_idx, relu_idx, fused_op)) = Self::find_candidate(graph) {
            if budget == 0 {
                break;
            }
            budget -= 1;

            let args = match graph.get_node(producer_idx) {
                Some(Node::Call(_, args)) => args.clone(),
                _ => Vec::new(),
            };
            graph.graph[producer_idx] = Node::Call(fused_op, args);

            // Rewire the activation's consumers onto the fused node.
            let successors: Vec<(NodeIndex, crate::Edge)> = graph
                .graph
                .edges_directed(relu_idx, petgraph::Direction::Outgoing)
                .map(|edge| (edge.target(), edge.weight().clone()))
                .collect();
            for (target, weight) in successors {
                if graph.graph.find_edge(producer_idx, target).is_none() {
                    graph.graph.add_edge(producer_idx, target, weight);
                }
            }

            // The fused node now produces what the activation produced.
            graph.redirect_boundary_node(relu_idx, producer_idx);
            graph.remove_node(relu_idx);
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "operation_fusion"
    }
}

/// Dead code elimination pass
pub struct DeadCodeEliminationPass;

impl Pass for DeadCodeEliminationPass {
    fn apply(&self, graph: &mut FxGraph) -> TorshResult<()> {
        // Without recorded outputs there is no reachability information at all;
        // treating every node as dead would erase the whole graph, so refuse.
        if graph.outputs().is_empty() {
            log::debug!("dead_code_elimination: graph has no outputs, nothing to prune");
            return Ok(());
        }

        // Mark all nodes reachable from outputs
        let mut reachable: HashSet<NodeIndex> = HashSet::new();
        let mut stack = graph.outputs().to_vec();

        while let Some(node_idx) = stack.pop() {
            if reachable.insert(node_idx) {
                // Add all predecessors to the stack
                let predecessors: Vec<_> = graph
                    .graph
                    .neighbors_directed(node_idx, petgraph::Direction::Incoming)
                    .collect();
                stack.extend(predecessors);
            }
        }

        // Declared inputs are part of the graph signature and are never dead.
        reachable.extend(graph.inputs().iter().copied());

        // Collect nodes to remove (those not reachable)
        let to_remove: HashSet<NodeIndex> = graph
            .graph
            .node_indices()
            .filter(|idx| !reachable.contains(idx))
            .collect();

        if !to_remove.is_empty() {
            // Removal goes through FxGraph so the input/output lists stay valid.
            graph.remove_nodes(&to_remove);
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "dead_code_elimination"
    }
}

/// Constant folding pass
///
/// Performs real constant propagation over the graph: nodes whose operands are all
/// known scalar constants are evaluated at compile time and replaced by a
/// `constant` node carrying the computed value. Propagation runs to a fixed point,
/// so chains such as `mul(add(2, 3), 4)` collapse completely.
///
/// A node produces a known constant when it is `constant(<literal>)`,
/// `constant_zero` or `constant_one` (the latter two are emitted by
/// [`GraphSimplificationPass`]). Operands may also be inline numeric literals in
/// the argument list.
pub struct ConstantFoldingPass;

impl ConstantFoldingPass {
    /// Scalar value produced by a node, if it is a constant.
    ///
    /// # Arguments
    /// * `node` - Node to inspect
    ///
    /// # Returns
    /// * `Option<f32>` - The literal value, or `None` for non-constant nodes
    pub fn constant_value(node: &Node) -> Option<f32> {
        match node {
            Node::Call(op_name, args) => match op_name.as_str() {
                "constant" => args.first().and_then(|arg| arg.parse::<f32>().ok()),
                "constant_zero" => Some(0.0),
                "constant_one" => Some(1.0),
                _ => None,
            },
            _ => None,
        }
    }

    /// Evaluate a pure operation on known scalar operands.
    ///
    /// # Arguments
    /// * `op_name` - Operation to evaluate
    /// * `operands` - Constant operand values in argument order
    ///
    /// # Returns
    /// * `Option<f32>` - The result, or `None` if the operation is not foldable
    fn evaluate(op_name: &str, operands: &[f32]) -> Option<f32> {
        match (op_name, operands) {
            ("add", [lhs, rhs]) => Some(lhs + rhs),
            ("sub", [lhs, rhs]) => Some(lhs - rhs),
            ("mul", [lhs, rhs]) => Some(lhs * rhs),
            ("div", [lhs, rhs]) if *rhs != 0.0 => Some(lhs / rhs),
            ("pow", [base, exponent]) => Some(base.powf(*exponent)),
            ("maximum", [lhs, rhs]) => Some(lhs.max(*rhs)),
            ("minimum", [lhs, rhs]) => Some(lhs.min(*rhs)),
            ("neg", [value]) => Some(-value),
            ("abs", [value]) => Some(value.abs()),
            ("sqrt", [value]) if *value >= 0.0 => Some(value.sqrt()),
            ("exp", [value]) => Some(value.exp()),
            ("log", [value]) if *value > 0.0 => Some(value.ln()),
            ("relu", [value]) => Some(value.max(0.0)),
            ("sigmoid", [value]) => Some(1.0 / (1.0 + (-value).exp())),
            ("tanh", [value]) => Some(value.tanh()),
            ("identity", [value]) => Some(*value),
            _ => None,
        }
    }

    /// Resolve the constant values of a node's arguments.
    ///
    /// An argument is constant when it is a numeric literal or when the incoming
    /// edge carrying that name comes from a node with a known constant value.
    fn resolve_operands(
        graph: &FxGraph,
        node_idx: NodeIndex,
        args: &[String],
        known: &HashMap<NodeIndex, f32>,
    ) -> Option<Vec<f32>> {
        let mut producers: HashMap<String, NodeIndex> = HashMap::new();
        for edge in graph
            .graph
            .edges_directed(node_idx, petgraph::Direction::Incoming)
        {
            producers.insert(edge.weight().name.clone(), edge.source());
        }

        // Every incoming value must be accounted for by an argument, otherwise the
        // node consumes something we cannot see.
        let incoming_count = graph
            .graph
            .edges_directed(node_idx, petgraph::Direction::Incoming)
            .count();
        if incoming_count > args.len() {
            return None;
        }

        let mut values = Vec::with_capacity(args.len());
        for arg in args {
            if let Ok(literal) = arg.parse::<f32>() {
                values.push(literal);
                continue;
            }
            let producer = producers.get(arg)?;
            values.push(*known.get(producer)?);
        }

        Some(values)
    }
}

impl Pass for ConstantFoldingPass {
    fn apply(&self, graph: &mut FxGraph) -> TorshResult<()> {
        loop {
            let known: HashMap<NodeIndex, f32> = graph
                .nodes()
                .filter_map(|(idx, node)| Self::constant_value(node).map(|value| (idx, value)))
                .collect();

            let mut foldable: Vec<(NodeIndex, f32)> = Vec::new();
            for (idx, node) in graph.nodes() {
                if known.contains_key(&idx) {
                    continue;
                }
                let Node::Call(op_name, args) = node else {
                    continue;
                };
                let Some(operands) = Self::resolve_operands(graph, idx, args, &known) else {
                    continue;
                };
                if let Some(value) = Self::evaluate(op_name, &operands) {
                    foldable.push((idx, value));
                }
            }

            if foldable.is_empty() {
                return Ok(());
            }

            for (idx, value) in foldable {
                log::debug!("constant_folding: folding node {idx:?} to {value}");
                graph.graph[idx] = Node::Call("constant".to_string(), vec![value.to_string()]);

                // The operands are no longer consumed by this node; dropping the
                // edges lets dead code elimination reclaim them.
                let mut incoming: Vec<_> = graph
                    .graph
                    .edges_directed(idx, petgraph::Direction::Incoming)
                    .map(|edge| edge.id())
                    .collect();
                // Removing the highest edge index first keeps the remaining ones valid.
                incoming.sort_by(|a, b| b.index().cmp(&a.index()));
                for edge_idx in incoming {
                    graph.graph.remove_edge(edge_idx);
                }
            }
        }
    }

    fn name(&self) -> &str {
        "constant_folding"
    }
}

/// Pass manager for organizing and running passes
pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
}

impl PassManager {
    /// Create a new pass manager
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    /// Add a pass to the manager
    pub fn add_pass(&mut self, pass: Box<dyn Pass>) {
        self.passes.push(pass);
    }

    /// Run all passes on the graph
    pub fn run(&self, graph: &mut FxGraph) -> TorshResult<()> {
        for pass in &self.passes {
            log::debug!("Running pass: {}", pass.name());
            pass.apply(graph)?;
        }
        Ok(())
    }

    /// Create a default pass manager with common optimization passes
    pub fn default_optimization_passes() -> Self {
        let mut manager = Self::new();
        manager.add_pass(Box::new(GraphSimplificationPass));
        manager.add_pass(Box::new(ConstantFoldingPass));
        manager.add_pass(Box::new(CommonSubexpressionEliminationPass));
        manager.add_pass(Box::new(DeadCodeEliminationPass));
        manager.add_pass(Box::new(OperationFusionPass));
        manager.add_pass(Box::new(MemoryOptimizationPass));
        manager.add_pass(Box::new(LoopOptimizationPass));
        manager
    }

    /// Create an aggressive optimization pass manager
    pub fn aggressive_optimization_passes() -> Self {
        let mut manager = Self::new();
        // Run multiple rounds of optimization
        manager.add_pass(Box::new(GraphSimplificationPass));
        manager.add_pass(Box::new(ConstantFoldingPass));
        manager.add_pass(Box::new(CommonSubexpressionEliminationPass));
        manager.add_pass(Box::new(DeadCodeEliminationPass));
        manager.add_pass(Box::new(OperationFusionPass));
        // Second round
        manager.add_pass(Box::new(GraphSimplificationPass));
        manager.add_pass(Box::new(CommonSubexpressionEliminationPass));
        manager.add_pass(Box::new(DeadCodeEliminationPass));
        manager.add_pass(Box::new(MemoryOptimizationPass));
        manager.add_pass(Box::new(LoopOptimizationPass));
        manager
    }
}

impl Default for PassManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function for operation fusion
pub fn fuse_operations(graph: &mut FxGraph) -> TorshResult<()> {
    let pass = OperationFusionPass;
    pass.apply(graph)
}

/// Convenience function for dead code elimination
pub fn eliminate_dead_code(graph: &mut FxGraph) -> TorshResult<()> {
    let pass = DeadCodeEliminationPass;
    pass.apply(graph)
}

/// Convenience function for constant folding
pub fn fold_constants(graph: &mut FxGraph) -> TorshResult<()> {
    let pass = ConstantFoldingPass;
    pass.apply(graph)
}

/// Common Subexpression Elimination (CSE) pass
pub struct CommonSubexpressionEliminationPass;

impl Pass for CommonSubexpressionEliminationPass {
    fn apply(&self, graph: &mut FxGraph) -> TorshResult<()> {
        // Map from operation signature to node index
        let mut expression_map: HashMap<String, NodeIndex> = HashMap::new();
        let mut nodes_to_replace: Vec<(NodeIndex, NodeIndex)> = Vec::new();

        // Find common subexpressions
        for (idx, node) in graph.nodes() {
            if let Node::Call(op_name, args) = node {
                // Create a signature for this operation
                let args_str = args.join(",");
                let signature = format!("{op_name}({args_str})");

                if let Some(&existing_idx) = expression_map.get(&signature) {
                    // Found a duplicate expression
                    nodes_to_replace.push((idx, existing_idx));
                } else {
                    // First occurrence of this expression
                    expression_map.insert(signature, idx);
                }
            }
        }

        // Redirect the consumers of every duplicate onto the original expression.
        // All rewiring happens first; the duplicates are then removed in a single
        // batch so no pending index can be invalidated in between.
        let mut duplicates: HashSet<NodeIndex> = HashSet::new();
        for (duplicate_idx, original_idx) in nodes_to_replace {
            let successors: Vec<(NodeIndex, crate::Edge)> = graph
                .graph
                .edges_directed(duplicate_idx, petgraph::Direction::Outgoing)
                .map(|edge| (edge.target(), edge.weight().clone()))
                .collect();

            for (successor_idx, weight) in successors {
                if graph.graph.find_edge(original_idx, successor_idx).is_none() {
                    graph.graph.add_edge(original_idx, successor_idx, weight);
                }
            }

            graph.redirect_boundary_node(duplicate_idx, original_idx);
            duplicates.insert(duplicate_idx);
        }

        if !duplicates.is_empty() {
            graph.remove_nodes(&duplicates);
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "common_subexpression_elimination"
    }
}

/// Memory optimization pass
pub struct MemoryOptimizationPass;

impl Pass for MemoryOptimizationPass {
    fn apply(&self, graph: &mut FxGraph) -> TorshResult<()> {
        // Analyze tensor lifetimes and identify opportunities for in-place operations
        let mut in_place_candidates = Vec::new();

        for (idx, node) in graph.nodes() {
            if let Node::Call(op_name, args) = node {
                // Check if this operation can be done in-place
                if self.can_be_inplace(op_name) && args.len() == 1 {
                    // Find the input node
                    let predecessors: Vec<_> = graph
                        .graph
                        .neighbors_directed(idx, petgraph::Direction::Incoming)
                        .collect();

                    if predecessors.len() == 1 {
                        let input_idx = predecessors[0];

                        // Check if input has only one use (this operation)
                        let input_uses: Vec<_> = graph
                            .graph
                            .neighbors_directed(input_idx, petgraph::Direction::Outgoing)
                            .collect();

                        if input_uses.len() == 1 {
                            in_place_candidates.push((idx, op_name.clone()));
                        }
                    }
                }
            }
        }

        // Mark operations as in-place (in practice, this would modify the operation metadata)
        for (idx, op_name) in in_place_candidates {
            // Replace operation with in-place version
            if let Some(Node::Call(ref mut current_op, ref _args)) =
                graph.graph.node_weight_mut(idx)
            {
                *current_op = format!("{op_name}_inplace");
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "memory_optimization"
    }
}

impl MemoryOptimizationPass {
    /// Check if an operation can be performed in-place
    fn can_be_inplace(&self, op_name: &str) -> bool {
        matches!(op_name, "relu" | "sigmoid" | "tanh" | "add" | "mul")
    }
}

/// Loop optimization pass
pub struct LoopOptimizationPass;

impl Pass for LoopOptimizationPass {
    fn apply(&self, graph: &mut FxGraph) -> TorshResult<()> {
        // Find loop nodes and optimize them
        let mut loop_optimizations = Vec::new();

        for (idx, node) in graph.nodes() {
            if let Node::Loop {
                condition: _,
                body,
                loop_vars: _,
            } = node
            {
                // Analyze loop for optimization opportunities
                if self.can_unroll_loop(body) {
                    loop_optimizations.push((idx, "unroll"));
                } else if self.can_vectorize_loop(body) {
                    loop_optimizations.push((idx, "vectorize"));
                }
            }
        }

        // Apply optimizations
        for (idx, optimization) in loop_optimizations {
            if let Some(Node::Loop { ref mut body, .. }) = graph.graph.node_weight_mut(idx) {
                match optimization {
                    "unroll" => {
                        // Mark loop for unrolling
                        body.push("unrolled".to_string());
                    }
                    "vectorize" => {
                        // Mark loop for vectorization
                        body.push("vectorized".to_string());
                    }
                    _ => {}
                }
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "loop_optimization"
    }
}

impl LoopOptimizationPass {
    fn can_unroll_loop(&self, _body: &[String]) -> bool {
        // Simplified heuristic: small loops can be unrolled
        true // For demonstration
    }

    fn can_vectorize_loop(&self, _body: &[String]) -> bool {
        // Simplified heuristic: element-wise operations can be vectorized
        true // For demonstration
    }
}

/// Graph simplification pass
pub struct GraphSimplificationPass;

impl Pass for GraphSimplificationPass {
    fn apply(&self, graph: &mut FxGraph) -> TorshResult<()> {
        let mut simplifications = Vec::new();

        // Find patterns that can be simplified
        for (idx, node) in graph.nodes() {
            if let Node::Call(op_name, args) = node {
                match op_name.as_str() {
                    "add" => {
                        // Check for add(x, 0) or add(0, x) patterns
                        if args.len() == 2 && (args[0] == "zero" || args[1] == "zero") {
                            simplifications.push((idx, "identity"));
                        }
                    }
                    "mul" => {
                        // Check for mul(x, 1) or mul(1, x) patterns
                        if args.len() == 2 && (args[0] == "one" || args[1] == "one") {
                            simplifications.push((idx, "identity"));
                        }
                        // Check for mul(x, 0) or mul(0, x) patterns
                        if args.len() == 2 && (args[0] == "zero" || args[1] == "zero") {
                            simplifications.push((idx, "zero"));
                        }
                    }
                    _ => {}
                }
            }
        }

        // Apply simplifications
        for (idx, simplification) in simplifications {
            match simplification {
                "identity" => {
                    // Replace with identity operation (just pass through the non-constant input)
                    if let Some(Node::Call(ref mut op_name, ref mut args)) =
                        graph.graph.node_weight_mut(idx)
                    {
                        *op_name = "identity".to_string();
                        args.retain(|arg| arg != "zero" && arg != "one");
                    }
                }
                "zero" => {
                    // Replace with constant zero
                    if let Some(node) = graph.graph.node_weight_mut(idx) {
                        *node = Node::Call("constant_zero".to_string(), vec![]);
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "graph_simplification"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracer::ModuleTracer;

    #[test]
    fn test_pass_manager() {
        let mut manager = PassManager::new();
        manager.add_pass(Box::new(DeadCodeEliminationPass));

        let mut tracer = ModuleTracer::new();
        tracer.add_input("x");
        tracer.add_call("relu", vec!["x".to_string()]);
        tracer.add_output("node_0");
        let mut graph = tracer.finalize();

        // Should run without error
        manager.run(&mut graph).unwrap();
    }

    #[test]
    fn test_operation_fusion_pass() {
        let pass = OperationFusionPass;

        let mut tracer = ModuleTracer::new();
        tracer.add_input("x");
        tracer.add_call("linear", vec!["x".to_string()]);
        tracer.add_call("relu", vec!["node_0".to_string()]);
        tracer.add_output("node_1");
        let mut graph = tracer.finalize();

        // Should run without error
        pass.apply(&mut graph).unwrap();
    }
}
