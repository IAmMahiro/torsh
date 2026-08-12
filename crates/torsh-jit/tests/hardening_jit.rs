//! Production-hardening regression tests for torsh-jit.
//!
//! F121 (Cranelift codegen returned kernels with no code), F122 (fusion merged
//! producers that still had external consumers), F123 (aggressive fusion never
//! de-duplicated overlapping groups) and F124 (`trace` returned an empty module
//! with `Ok`).

use torsh_core::{DType, DeviceType};
use torsh_jit::codegen::CodeGenerator;
use torsh_jit::fusion::KernelFusion;
use torsh_jit::graph::{shape_from_slice, ComputationGraph, Conv2dInfo, Edge, Node, Operation};
use torsh_jit::FusionStrategy;

fn node(op: Operation, name: &str) -> Node {
    Node::new(op, name.to_string())
        .with_output_shapes(vec![Some(shape_from_slice(&[4, 4]))])
        .with_dtypes(vec![DType::F32])
        .with_device(DeviceType::Cpu)
}

fn conv_info() -> Conv2dInfo {
    Conv2dInfo {
        in_channels: 4,
        out_channels: 4,
        kernel_size: (3, 3),
        stride: (1, 1),
        padding: (1, 1),
        dilation: (1, 1),
        groups: 1,
    }
}

/// Total number of primitive operations represented by a graph: a fused kernel
/// stands for all the operations it absorbed.
fn primitive_op_count(graph: &ComputationGraph) -> usize {
    graph
        .nodes()
        .map(|(_, n)| match &n.op {
            Operation::FusedKernel { ops, .. } => ops.len(),
            _ => 1,
        })
        .sum()
}

fn fused_kernels(graph: &ComputationGraph) -> Vec<Vec<Operation>> {
    graph
        .nodes()
        .filter_map(|(_, n)| match &n.op {
            Operation::FusedKernel { ops, .. } => Some(ops.clone()),
            _ => None,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// F121
// ---------------------------------------------------------------------------

#[test]
fn f121_cpu_codegen_never_returns_empty_kernel_code() {
    let mut graph = ComputationGraph::new();
    let input = graph.add_node(node(Operation::Input, "x"));
    let relu = graph.add_node(node(Operation::Relu, "relu"));
    graph.add_edge(input, relu, Edge::default());
    graph.add_input(input);
    graph.add_output(relu);

    let generator = CodeGenerator::new(DeviceType::Cpu);
    let kernels = generator
        .generate(&graph)
        .expect("an element-wise CPU graph must compile");

    assert!(
        !kernels.is_empty(),
        "CPU codegen produced no kernels at all"
    );
    for kernel in &kernels {
        assert!(
            !kernel.code.is_empty(),
            "kernel {} was reported as compiled but carries no machine code",
            kernel.id
        );
    }
}

// ---------------------------------------------------------------------------
// F122
// ---------------------------------------------------------------------------

#[test]
fn f122_fusion_skips_producers_with_external_consumers() {
    // conv -> relu and conv -> sigmoid: fusing conv into relu would silently feed
    // sigmoid the activated value.
    let mut graph = ComputationGraph::new();
    let input = graph.add_node(node(Operation::Input, "x"));
    let conv = graph.add_node(node(Operation::Conv2d(conv_info()), "conv"));
    let relu = graph.add_node(node(Operation::Relu, "relu"));
    let sigmoid = graph.add_node(node(Operation::Sigmoid, "sigmoid"));
    graph.add_edge(input, conv, Edge::default());
    graph.add_edge(conv, relu, Edge::default());
    graph.add_edge(conv, sigmoid, Edge::default());
    graph.add_input(input);
    graph.add_output(relu);
    graph.add_output(sigmoid);

    let fused = KernelFusion::new(FusionStrategy::Default)
        .apply(graph)
        .expect("fusion must succeed");

    for ops in fused_kernels(&fused) {
        assert!(
            !ops.iter().any(|op| matches!(op, Operation::Conv2d(_))),
            "conv with two consumers must not be fused: {ops:?}"
        );
    }
}

#[test]
fn f122_fusion_still_fuses_a_single_consumer_chain() {
    let mut graph = ComputationGraph::new();
    let input = graph.add_node(node(Operation::Input, "x"));
    let conv = graph.add_node(node(Operation::Conv2d(conv_info()), "conv"));
    let relu = graph.add_node(node(Operation::Relu, "relu"));
    graph.add_edge(input, conv, Edge::default());
    graph.add_edge(conv, relu, Edge::default());
    graph.add_input(input);
    graph.add_output(relu);

    let fused = KernelFusion::new(FusionStrategy::Default)
        .apply(graph)
        .expect("fusion must succeed");

    assert!(
        fused_kernels(&fused)
            .iter()
            .any(|ops| ops.iter().any(|op| matches!(op, Operation::Conv2d(_)))),
        "a legal conv->relu chain must still be fused"
    );
}

// ---------------------------------------------------------------------------
// F123
// ---------------------------------------------------------------------------

#[test]
fn f123_aggressive_fusion_does_not_duplicate_nodes_across_groups() {
    // conv -> relu -> exp: the conv+activation finder claims [conv, relu] while the
    // element-wise chain finder claims [relu, exp].
    let mut graph = ComputationGraph::new();
    let input = graph.add_node(node(Operation::Input, "x"));
    let conv = graph.add_node(node(Operation::Conv2d(conv_info()), "conv"));
    let relu = graph.add_node(node(Operation::Relu, "relu"));
    let exp = graph.add_node(node(Operation::Exp, "exp"));
    graph.add_edge(input, conv, Edge::default());
    graph.add_edge(conv, relu, Edge::default());
    graph.add_edge(relu, exp, Edge::default());
    graph.add_input(input);
    graph.add_output(exp);

    let before = primitive_op_count(&graph);

    let fused = KernelFusion::new(FusionStrategy::Aggressive)
        .apply(graph)
        .expect("aggressive fusion must succeed");

    assert_eq!(
        primitive_op_count(&fused),
        before,
        "an operation must belong to exactly one fusion group; fused kernels: {:?}",
        fused_kernels(&fused)
    );
}

// ---------------------------------------------------------------------------
// F124
// ---------------------------------------------------------------------------

#[test]
fn f124_trace_refuses_instead_of_returning_an_empty_module() {
    let result = torsh_jit::trace(|_inputs| vec![], &[]);
    let err = match result {
        Ok(_) => panic!("trace returned Ok with an empty module instead of refusing"),
        Err(err) => err,
    };
    let message = err.to_string();
    assert!(
        message.to_lowercase().contains("trac"),
        "the error must explain that tracing is unavailable, got: {message}"
    );
}
