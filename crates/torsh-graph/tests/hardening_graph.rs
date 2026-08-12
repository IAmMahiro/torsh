//! Regression tests for the wave2_graph production-hardening findings.
//!
//! Every test here failed against the pre-fix code; see the per-test comments
//! for the observed wrong value.

use torsh_core::device::DeviceType;
use torsh_graph::conv::gcn::GCNConv;
use torsh_graph::data::GraphDataLoader;
use torsh_graph::generative::GraphGAN;
use torsh_graph::utils::{gcn_norm, graph_laplacian};
use torsh_graph::{GraphData, GraphLayer};
use torsh_tensor::creation::from_vec;

fn edge_index(rows: Vec<f32>, cols: usize) -> torsh_tensor::Tensor {
    from_vec(rows, &[2, cols], DeviceType::Cpu).expect("edge index")
}

/// F119: `L = D - A` must have zero row sums.
///
/// Pre-fix the degrees were incremented once per `edge_index` column, so a
/// both-directions listing doubled them and the rows summed to 1, not 0.
#[test]
fn f119_unnormalized_laplacian_row_sums_are_zero() {
    // K2 with both directions listed (PyTorch-Geometric convention).
    let ei = edge_index(vec![0.0, 1.0, 1.0, 0.0], 2);
    let l = graph_laplacian(&ei, 2, false)
        .expect("laplacian")
        .to_vec()
        .expect("to_vec");
    for i in 0..2 {
        let row_sum: f32 = (0..2).map(|j| l[i * 2 + j]).sum();
        assert!(
            row_sum.abs() < 1e-6,
            "row {i} sums to {row_sum}, expected 0"
        );
    }
}

/// F119: the normalized Laplacian of K2 is `[[1,-1],[-1,1]]`.
///
/// Pre-fix: `[[1,-0.5],[-0.5,1]]` (degrees off by a factor of two).
#[test]
fn f119_normalized_laplacian_k2() {
    let ei = edge_index(vec![0.0, 1.0, 1.0, 0.0], 2);
    let l = graph_laplacian(&ei, 2, true)
        .expect("laplacian")
        .to_vec()
        .expect("to_vec");
    let want = [1.0f32, -1.0, -1.0, 1.0];
    for (got, expect) in l.iter().zip(want.iter()) {
        assert!((got - expect).abs() < 1e-5, "got {l:?} want {want:?}");
    }
}

/// F119: a self-looped node normalizes its diagonal by `d_i`, not `sqrt(d_i)`.
///
/// Pre-fix: `L[0][0] == 0.293` instead of `0`.
#[test]
fn f119_normalized_laplacian_self_loop_diagonal() {
    let ei = edge_index(vec![0.0, 1.0, 0.0, 1.0], 2);
    let l = graph_laplacian(&ei, 2, true)
        .expect("laplacian")
        .to_vec()
        .expect("to_vec");
    assert!(l[0].abs() < 1e-6, "L[0][0] = {} expected 0", l[0]);
    assert!(l[3].abs() < 1e-6, "L[1][1] = {} expected 0", l[3]);
}

/// F119: a malformed `edge_index` is an error, not an out-of-bounds panic.
#[test]
fn f119_malformed_edge_index_is_an_error() {
    let one_row = from_vec(vec![0.0, 1.0], &[1, 2], DeviceType::Cpu).expect("tensor");
    assert!(graph_laplacian(&one_row, 2, true).is_err());
    assert!(gcn_norm(&one_row, 2).is_err());
}

/// F118: `gcn_norm` returns `D~^-1/2 (A + I) D~^-1/2` exactly.
///
/// For K2 with self-loops every entry of `A_hat` is `1/2`.
#[test]
fn f118_gcn_norm_matches_kipf_welling() {
    let ei = edge_index(vec![0.0, 1.0, 1.0, 0.0], 2);
    let a_hat = gcn_norm(&ei, 2)
        .expect("gcn_norm")
        .to_vec()
        .expect("to_vec");
    for value in &a_hat {
        assert!((value - 0.5).abs() < 1e-6, "A_hat = {a_hat:?}");
    }

    // Path 0-1-2: d~ = [2, 3, 2].
    let path = edge_index(vec![0.0, 1.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0], 4);
    let a_hat = gcn_norm(&path, 3)
        .expect("gcn_norm")
        .to_vec()
        .expect("to_vec");
    let sqrt6 = 6.0f32.sqrt();
    let want = [
        0.5,
        1.0 / sqrt6,
        0.0,
        1.0 / sqrt6,
        1.0 / 3.0,
        1.0 / sqrt6,
        0.0,
        1.0 / sqrt6,
        0.5,
    ];
    for (got, expect) in a_hat.iter().zip(want.iter()) {
        assert!((got - expect).abs() < 1e-6, "got {a_hat:?} want {want:?}");
    }
}

/// F118: `GCNConv::forward` propagates with `A_hat`, not with the Laplacian.
///
/// With `X = 1` the output row `i` is `rowsum_i(A_hat) * W`, so the ratio
/// between two rows is weight-independent. Pre-fix the ratio was 2.207
/// (Laplacian propagation with doubled degrees) instead of 0.7899.
#[test]
fn f118_gcn_uses_kipf_welling_operator() {
    let ei = edge_index(vec![0.0, 1.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0], 4);
    let x = from_vec(vec![1.0, 1.0, 1.0], &[3, 1], DeviceType::Cpu).expect("x");
    let graph = GraphData::new(x, ei);

    let conv = GCNConv::new(1, 4, false).expect("gcn");
    let out = GCNConv::forward(&conv, &graph).expect("forward");
    let data = out.x.to_vec().expect("to_vec");

    let mut best = 0usize;
    for j in 0..4 {
        if data[j].abs() > data[best].abs() {
            best = j;
        }
    }
    assert!(data[best].abs() > 1e-8, "degenerate weights");
    let ratio = data[best] / data[4 + best];
    let expected = 0.908_248_3f32 / 1.149_829_9f32;
    assert!(
        (ratio - expected).abs() < 1e-4,
        "row0/row1 ratio {ratio} expected {expected}"
    );
}

/// F303/F020: a feature-dimension mismatch is an error, not a process abort.
#[test]
fn f303_gcn_forward_reports_shape_mismatch() {
    let ei = edge_index(vec![0.0, 1.0, 1.0, 0.0], 2);
    let x = from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], DeviceType::Cpu).expect("x");
    let graph = GraphData::new(x, ei);

    let conv = GCNConv::new(8, 4, true).expect("gcn");
    let err = GCNConv::forward(&conv, &graph).expect_err("shape mismatch must be an error");
    assert!(format!("{err}").contains("Shape mismatch"), "{err}");
}

/// F020: the `GraphLayer` trait itself is fallible, so a trait-object forward
/// pass over bad input returns `Err` instead of panicking.
#[test]
fn f020_graph_layer_trait_is_fallible() {
    let ei = edge_index(vec![0.0, 1.0, 1.0, 0.0], 2);
    let x = from_vec(vec![1.0, 2.0], &[2, 1], DeviceType::Cpu).expect("x");
    let graph = GraphData::new(x, ei);

    let layer: Box<dyn GraphLayer> = Box::new(GCNConv::new(5, 3, false).expect("gcn"));
    assert!(layer.forward(&graph).is_err());
}

/// F117: `from_directory` really loads the directory instead of returning an
/// empty `Vec`.
#[test]
fn f117_data_loader_reads_directory() {
    let dir = std::env::temp_dir().join("torsh_graph_hardening_f117");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    std::fs::write(dir.join("a.edgelist"), "0 1\n1 2\n2 0\n").expect("write");
    std::fs::write(dir.join("ignored.bin"), [0u8, 1, 2]).expect("write");

    let loader = GraphDataLoader::new(4, false)
        .expect("loader")
        .with_node_features(6);
    let graphs = loader
        .from_directory(dir.to_str().expect("utf8"))
        .expect("from_directory");
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(graphs.len(), 1, "expected exactly one loaded graph");
    assert_eq!(graphs[0].num_nodes, 3);
    assert_eq!(graphs[0].x.shape().dims(), &[3, 6]);
}

/// F117: a missing directory is an error, never a silent empty dataset.
#[test]
fn f117_missing_directory_is_an_error() {
    let dir = std::env::temp_dir().join("torsh_graph_hardening_f117_missing");
    let _ = std::fs::remove_dir_all(&dir);
    let loader = GraphDataLoader::new(4, false).expect("loader");
    assert!(loader.from_directory(dir.to_str().expect("utf8")).is_err());
}

/// GAN losses stay finite when the discriminator saturates.
///
/// Pre-fix the loss went through `ln(1 - sigmoid(logit))`; in `f32` the sigmoid
/// saturates to exactly 1.0, so the loss was `+inf`.
#[test]
fn graphgan_losses_finite_under_saturation() {
    let features = from_vec(vec![1.0e6f32; 4 * 8], &[4, 8], DeviceType::Cpu).expect("x");
    let ei = edge_index(vec![0.0, 1.0, 2.0, 1.0, 2.0, 3.0], 3);
    let graph = GraphData::new(features, ei);

    let gan = GraphGAN::new(16, 32, 8, true).expect("gan");

    let logit = gan.discriminate_logit(&graph).expect("logit");
    assert!(logit.is_finite());
    // The sigmoid itself does saturate: that is exactly why the losses must be
    // computed from the logit.
    let score = gan.discriminate(&graph).expect("score");
    assert!((0.0..=1.0).contains(&score));

    let disc_loss = gan.discriminator_loss(&graph, 4).expect("disc loss");
    assert!(disc_loss.is_finite(), "disc_loss = {disc_loss}");
    assert!(disc_loss >= 0.0, "BCE-with-logits is non-negative");

    let gen_loss = gan.generator_loss(4).expect("gen loss");
    assert!(gen_loss.is_finite(), "gen_loss = {gen_loss}");
}

/// F020: head divisibility is validated once, at layer construction, so the
/// forward path can never hit an ill-defined `view` reshape.
#[test]
fn f020_transformer_validates_head_split_at_construction() {
    use torsh_graph::conv::GraphTransformer;

    assert!(GraphTransformer::new(8, 10, 4, 2, 0.0, true).is_err());
    assert!(GraphTransformer::new(8, 16, 0, 2, 0.0, true).is_err());
    assert!(GraphTransformer::new(8, 16, 4, 2, 0.0, true).is_ok());
}
