//! Production-hardening regression tests for torsh-fx.
//!
//! Each test corresponds to a finding from the hardening campaign:
//! F114 (node removal invalidates graph.inputs/outputs), F115 (subgraph rewriter
//! stale indices / dropped external in-edges), F116 (operation fusion leaves the
//! consumer node and ignores fan-out), F219 (generated deployment server echoed
//! its input) and F220 (constant folding was a no-op).

use std::collections::HashMap;
use torsh_fx::passes::{
    ConstantFoldingPass, DeadCodeEliminationPass, OperationFusionPass, Pass, PassManager,
};
use torsh_fx::subgraph_rewriter::{SubgraphPattern, SubgraphRewriter};
use torsh_fx::tracer::ModuleTracer;
use torsh_fx::{FxGraph, Node};

/// Every index kept in `inputs`/`outputs` must still address a node of the right kind.
fn assert_io_invariant(graph: &FxGraph) {
    for &idx in graph.inputs() {
        let node = graph
            .get_node(idx)
            .unwrap_or_else(|| panic!("input index {idx:?} no longer exists in the graph"));
        assert!(
            matches!(node, Node::Input(_)),
            "input index {idx:?} points at {node:?}, not an Input node"
        );
    }
    for &idx in graph.outputs() {
        let node = graph
            .get_node(idx)
            .unwrap_or_else(|| panic!("output index {idx:?} no longer exists in the graph"));
        assert!(
            matches!(node, Node::Output),
            "output index {idx:?} points at {node:?}, not an Output node"
        );
    }
}

fn op_names(graph: &FxGraph) -> Vec<String> {
    graph
        .nodes()
        .filter_map(|(_, node)| match node {
            Node::Call(name, _) => Some(name.clone()),
            _ => None,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// F114
// ---------------------------------------------------------------------------

#[test]
fn f114_dce_keeps_input_output_indices_valid() {
    let mut tracer = ModuleTracer::new();
    tracer.add_input("x");
    tracer.add_call("relu", vec!["x".to_string()]); // node_0 (live)
    tracer.add_call("sigmoid", vec!["x".to_string()]); // node_1 (dead)
    tracer.add_call("tanh", vec!["node_0".to_string()]); // node_2 (live)
    tracer.add_output("node_2");
    let mut graph = tracer.finalize();

    DeadCodeEliminationPass.apply(&mut graph).expect("dce");

    assert_io_invariant(&graph);
    let ops = op_names(&graph);
    assert!(
        ops.contains(&"relu".to_string()),
        "live relu was removed: {ops:?}"
    );
    assert!(
        ops.contains(&"tanh".to_string()),
        "live tanh was removed: {ops:?}"
    );
    assert!(
        !ops.contains(&"sigmoid".to_string()),
        "dead sigmoid survived: {ops:?}"
    );
}

#[test]
fn f114_dce_with_several_dead_nodes_does_not_delete_live_ones() {
    let mut tracer = ModuleTracer::new();
    tracer.add_input("x");
    tracer.add_call("relu", vec!["x".to_string()]); // node_0 live
    tracer.add_call("sigmoid", vec!["x".to_string()]); // node_1 dead
    tracer.add_call("tanh", vec!["x".to_string()]); // node_2 dead
    tracer.add_call("gelu", vec!["node_0".to_string()]); // node_3 live
    tracer.add_output("node_3");
    let mut graph = tracer.finalize();

    DeadCodeEliminationPass.apply(&mut graph).expect("dce");

    assert_io_invariant(&graph);
    let ops = op_names(&graph);
    assert!(
        ops.contains(&"relu".to_string()),
        "live relu removed: {ops:?}"
    );
    assert!(
        ops.contains(&"gelu".to_string()),
        "live gelu removed: {ops:?}"
    );
    assert_eq!(ops.len(), 2, "dead nodes should be gone, got {ops:?}");
    assert_eq!(
        graph.node_count(),
        4,
        "input + relu + gelu + output expected"
    );
}

#[test]
fn f114_dce_without_outputs_removes_nothing() {
    // No outputs recorded => no reachability information; erasing the whole graph
    // would be catastrophic, so the pass must be a no-op.
    let mut tracer = ModuleTracer::new();
    tracer.add_input("x");
    tracer.add_call("relu", vec!["x".to_string()]);
    tracer.add_call("sigmoid", vec!["node_0".to_string()]);
    let mut graph = tracer.finalize();
    let before = graph.node_count();

    DeadCodeEliminationPass.apply(&mut graph).expect("dce");

    assert_eq!(
        graph.node_count(),
        before,
        "graph with no outputs must not be erased"
    );
}

#[test]
fn f114_cse_keeps_input_output_indices_valid() {
    let mut tracer = ModuleTracer::new();
    tracer.add_input("x");
    tracer.add_call("relu", vec!["x".to_string()]); // node_0
    tracer.add_call("relu", vec!["x".to_string()]); // node_1 (duplicate)
    tracer.add_call("tanh", vec!["node_1".to_string()]); // node_2
    tracer.add_output("node_2");
    let mut graph = tracer.finalize();

    let manager = {
        let mut m = PassManager::new();
        m.add_pass(Box::new(
            torsh_fx::passes::CommonSubexpressionEliminationPass,
        ));
        m
    };
    manager.run(&mut graph).expect("cse");

    assert_io_invariant(&graph);
    let ops = op_names(&graph);
    assert_eq!(
        ops.iter().filter(|o| *o == "relu").count(),
        1,
        "duplicate relu should be eliminated: {ops:?}"
    );
    assert!(
        ops.contains(&"tanh".to_string()),
        "consumer must survive: {ops:?}"
    );
}

// ---------------------------------------------------------------------------
// F115
// ---------------------------------------------------------------------------

#[test]
fn f115_rewriter_preserves_external_in_edges_and_io() {
    // x, w -> conv2d ; conv2d + scale -> batch_norm ; batch_norm -> output
    let mut tracer = ModuleTracer::new();
    tracer.add_input("x");
    tracer.add_input("w");
    tracer.add_input("scale");
    tracer.add_call("conv2d", vec!["x".to_string(), "w".to_string()]); // node_0
    tracer.add_call(
        "batch_norm",
        vec!["node_0".to_string(), "scale".to_string()],
    ); // node_1
    tracer.add_output("node_1");
    let mut graph = tracer.finalize();

    let mut rewriter = SubgraphRewriter::new();
    rewriter.add_pattern(SubgraphPattern::conv_bn_fusion());
    let replacements = rewriter.apply(&mut graph).expect("rewrite");
    assert_eq!(replacements, 1);

    assert_io_invariant(&graph);

    let ops = op_names(&graph);
    assert!(
        ops.contains(&"conv2d_bn".to_string()),
        "fused op missing: {ops:?}"
    );
    assert!(
        !ops.contains(&"batch_norm".to_string()),
        "batch_norm survived: {ops:?}"
    );

    // The external `scale` input must still feed the fused node.
    let fused_idx = graph
        .nodes()
        .find(|(_, n)| matches!(n, Node::Call(name, _) if name == "conv2d_bn"))
        .map(|(idx, _)| idx)
        .expect("fused node");
    let preds: Vec<_> = graph
        .graph
        .neighbors_directed(fused_idx, petgraph::Direction::Incoming)
        .collect();
    assert_eq!(
        preds.len(),
        3,
        "fused node must keep x, w and the external scale input"
    );
}

#[test]
fn f115_rewritten_graph_computes_the_same_result() {
    use torsh_fx::interpreter::GraphInterpreter;
    use torsh_tensor::creation::full;

    let build = || {
        let mut tracer = ModuleTracer::new();
        tracer.add_input("x");
        tracer.add_input("w");
        tracer.add_input("scale");
        tracer.add_call("conv2d", vec!["x".to_string(), "w".to_string()]); // node_0
        tracer.add_call(
            "batch_norm",
            vec!["node_0".to_string(), "scale".to_string()],
        ); // node_1
        tracer.add_output("node_1");
        tracer.finalize()
    };

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), full(&[1, 1, 4, 4], 2.0f32).expect("x"));
    inputs.insert("w".to_string(), full(&[1, 1, 3, 3], 0.5f32).expect("w"));
    inputs.insert(
        "scale".to_string(),
        full(&[1, 1, 2, 2], 3.0f32).expect("scale"),
    );

    let reference = GraphInterpreter::new(torsh_core::device::DeviceType::Cpu)
        .run(&build(), inputs.clone())
        .expect("unfused graph must execute");

    let mut graph = build();
    let mut rewriter = SubgraphRewriter::new();
    rewriter.add_pattern(SubgraphPattern::conv_bn_fusion());
    rewriter.apply(&mut graph).expect("rewrite");

    let fused = GraphInterpreter::new(torsh_core::device::DeviceType::Cpu)
        .run(&graph, inputs)
        .expect("fused graph must stay executable");

    assert_eq!(reference.len(), fused.len());
    let expected = reference[0].to_vec().expect("reference data");
    let actual = fused[0].to_vec().expect("fused data");
    assert_eq!(expected.len(), actual.len());
    for (lhs, rhs) in expected.iter().zip(actual.iter()) {
        assert!(
            (lhs - rhs).abs() < 1e-5,
            "fusion changed the result: {expected:?} vs {actual:?}"
        );
    }
}

#[test]
fn interpreter_orders_operands_by_argument_list() {
    use torsh_fx::interpreter::GraphInterpreter;

    // sub is not commutative: relying on petgraph's reverse edge order computes 3-5.
    let mut tracer = ModuleTracer::new();
    tracer.add_call("constant", vec!["5.0".to_string()]); // node_0
    tracer.add_call("constant", vec!["3.0".to_string()]); // node_1
    tracer.add_call("sub", vec!["node_0".to_string(), "node_1".to_string()]); // node_2
    tracer.add_output("node_2");
    let graph = tracer.finalize();

    let outputs = GraphInterpreter::new(torsh_core::device::DeviceType::Cpu)
        .run(&graph, HashMap::new())
        .expect("graph must execute");
    let values = outputs[0].to_vec().expect("tensor data");
    assert!(
        (values[0] - 2.0).abs() < 1e-6,
        "sub(5, 3) must be 2, got {values:?}"
    );
}

#[test]
fn f115_rewriter_refuses_to_fuse_a_producer_with_extra_consumers() {
    // node_0 = linear(x); node_1 = relu(node_0); node_2 = sigmoid(node_0)
    let mut tracer = ModuleTracer::new();
    tracer.add_input("x");
    tracer.add_call("linear", vec!["x".to_string()]); // node_0
    tracer.add_call("relu", vec!["node_0".to_string()]); // node_1
    tracer.add_call("sigmoid", vec!["node_0".to_string()]); // node_2
    tracer.add_output("node_1");
    tracer.add_output("node_2");
    let mut graph = tracer.finalize();
    let before = graph.node_count();

    let mut rewriter = SubgraphRewriter::new();
    rewriter.add_pattern(SubgraphPattern::linear_relu_fusion());
    rewriter.apply(&mut graph).expect("rewrite");

    assert_io_invariant(&graph);
    let ops = op_names(&graph);
    assert!(
        ops.contains(&"linear".to_string()) && ops.contains(&"relu".to_string()),
        "fan-out producer must not be fused: {ops:?}"
    );
    assert!(
        !ops.contains(&"linear_relu".to_string()),
        "illegal fusion: {ops:?}"
    );
    assert_eq!(graph.node_count(), before);
}

// ---------------------------------------------------------------------------
// F116
// ---------------------------------------------------------------------------

#[test]
fn f116_operation_fusion_removes_the_consumer_and_rewires() {
    let mut tracer = ModuleTracer::new();
    tracer.add_input("x");
    tracer.add_call("linear", vec!["x".to_string()]); // node_0
    tracer.add_call("relu", vec!["node_0".to_string()]); // node_1
    tracer.add_output("node_1");
    let mut graph = tracer.finalize();

    OperationFusionPass.apply(&mut graph).expect("fusion");

    assert_io_invariant(&graph);
    let ops = op_names(&graph);
    assert!(
        ops.contains(&"linear_relu".to_string()),
        "no fused op: {ops:?}"
    );
    assert!(
        !ops.contains(&"relu".to_string()),
        "relu must be removed, otherwise the activation is applied twice: {ops:?}"
    );
    assert_eq!(graph.node_count(), 3, "input + fused + output");

    // The fused node must feed the output node.
    let fused_idx = graph
        .nodes()
        .find(|(_, n)| matches!(n, Node::Call(name, _) if name == "linear_relu"))
        .map(|(idx, _)| idx)
        .expect("fused node");
    let output_idx = graph.outputs()[0];
    assert!(
        graph.graph.find_edge(fused_idx, output_idx).is_some(),
        "fused node must be connected to the output node"
    );
}

#[test]
fn f116_operation_fusion_skips_producers_with_multiple_consumers() {
    let mut tracer = ModuleTracer::new();
    tracer.add_input("x");
    tracer.add_call("linear", vec!["x".to_string()]); // node_0
    tracer.add_call("relu", vec!["node_0".to_string()]); // node_1
    tracer.add_call("sigmoid", vec!["node_0".to_string()]); // node_2
    tracer.add_output("node_1");
    tracer.add_output("node_2");
    let mut graph = tracer.finalize();

    OperationFusionPass.apply(&mut graph).expect("fusion");

    let ops = op_names(&graph);
    assert!(
        ops.contains(&"linear".to_string()),
        "linear feeding two consumers must stay unfused: {ops:?}"
    );
    assert!(
        !ops.contains(&"linear_relu".to_string()),
        "illegal fusion: {ops:?}"
    );
    assert_io_invariant(&graph);
}

// ---------------------------------------------------------------------------
// F220
// ---------------------------------------------------------------------------

#[test]
fn f220_constant_folding_actually_folds() {
    let mut tracer = ModuleTracer::new();
    tracer.add_call("constant", vec!["2.0".to_string()]); // node_0
    tracer.add_call("constant", vec!["3.0".to_string()]); // node_1
    tracer.add_call("add", vec!["node_0".to_string(), "node_1".to_string()]); // node_2
    tracer.add_output("node_2");
    let mut graph = tracer.finalize();

    ConstantFoldingPass.apply(&mut graph).expect("fold");

    assert_io_invariant(&graph);
    let folded = graph
        .nodes()
        .filter_map(|(_, n)| match n {
            Node::Call(name, args) if name == "constant" => args.first().cloned(),
            _ => None,
        })
        .filter_map(|v| v.parse::<f32>().ok())
        .collect::<Vec<_>>();
    assert!(
        folded.iter().any(|v| (*v - 5.0).abs() < 1e-6),
        "add(2,3) must fold to 5, got {folded:?}"
    );
    let ops = op_names(&graph);
    assert!(
        !ops.contains(&"add".to_string()),
        "folded op should be gone: {ops:?}"
    );
}

#[test]
fn f220_folded_graph_stays_executable() {
    use torsh_fx::interpreter::GraphInterpreter;

    let mut tracer = ModuleTracer::new();
    tracer.add_call("constant", vec!["2.0".to_string()]); // node_0
    tracer.add_call("constant", vec!["3.0".to_string()]); // node_1
    tracer.add_call("mul", vec!["node_0".to_string(), "node_1".to_string()]); // node_2
    tracer.add_output("node_2");
    let mut graph = tracer.finalize();

    ConstantFoldingPass.apply(&mut graph).expect("fold");

    let mut interpreter = GraphInterpreter::new(torsh_core::device::DeviceType::Cpu);
    let outputs = interpreter
        .run(&graph, HashMap::new())
        .expect("folded graph must remain executable");
    assert_eq!(outputs.len(), 1);
    let values = outputs[0].to_vec().expect("tensor data");
    assert!(
        (values[0] - 6.0).abs() < 1e-5,
        "expected 6.0, got {values:?}"
    );
}

#[test]
fn f220_non_constant_graph_is_left_alone() {
    let mut tracer = ModuleTracer::new();
    tracer.add_input("x");
    tracer.add_call("constant", vec!["2.0".to_string()]); // node_0
    tracer.add_call("add", vec!["x".to_string(), "node_0".to_string()]); // node_1
    tracer.add_output("node_1");
    let mut graph = tracer.finalize();
    let before = graph.node_count();

    ConstantFoldingPass.apply(&mut graph).expect("fold");

    let ops = op_names(&graph);
    assert!(
        ops.contains(&"add".to_string()),
        "add with a runtime input must stay"
    );
    assert_eq!(graph.node_count(), before);
}

// ---------------------------------------------------------------------------
// F219
// ---------------------------------------------------------------------------

#[test]
fn f219_generated_server_never_echoes_its_input() {
    use torsh_fx::cloud_deployment::{CloudDeploymentPackager, DeploymentConfig};
    use torsh_fx::model_zoo::{
        ModelMetadataBuilder, ModelWeights, ModelZooEntry, WeightData, WeightFormat,
    };

    let mut tracer = ModuleTracer::new();
    tracer.add_input("x");
    tracer.add_call("relu", vec!["x".to_string()]);
    tracer.add_output("node_0");
    let graph = tracer.finalize();

    let metadata =
        ModelMetadataBuilder::new("test-id".to_string(), "test-model".to_string()).build();
    let weights = ModelWeights {
        format: WeightFormat::SafeTensors,
        data: WeightData::Embedded {
            data: String::new(),
        },
        shapes: HashMap::new(),
        dtypes: HashMap::new(),
        total_params: 0,
        trainable_params: 0,
    };
    let entry = ModelZooEntry::new(metadata, graph, weights).expect("entry");

    let temp_dir = std::env::temp_dir().join("torsh_fx_hardening_f219");
    let _ = std::fs::remove_dir_all(&temp_dir);
    let config = DeploymentConfig::aws_sagemaker("hardening".to_string(), "us-east-1".to_string());
    let packager = CloudDeploymentPackager::new(&temp_dir, config).expect("packager");
    let package = packager.package_model(&entry).expect("package");

    let server = std::fs::read_to_string(package.path().join("server.py")).expect("server.py");

    assert!(
        !server.contains("identity function as placeholder"),
        "generated server still ships the identity placeholder"
    );
    assert!(
        !server.contains("outputs = input_array.tolist()"),
        "generated server still echoes the request payload as predictions"
    );
    assert!(
        !server.contains("TODO: Load and use actual model"),
        "generated server still contains the inference TODO"
    );
    assert!(
        server.contains("NotImplementedError"),
        "generated server must refuse loudly for operations it cannot run"
    );
    assert!(
        server.contains("def run_graph("),
        "generated server must execute the serialized graph"
    );

    let _ = std::fs::remove_dir_all(&temp_dir);
}
