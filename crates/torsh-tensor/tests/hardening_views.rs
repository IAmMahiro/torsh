//! Hardening regression tests for the tensor view / shape operations.
//!
//! Every test here pins down a behaviour that was previously wrong:
//! aliasing semantics of the view constructors (F152/F272/F157), the stride of
//! an inserted axis (F153), autograd continuity across reshape/transpose/expand
//! (F156/F269) and zero-sized-dimension handling in the broadcast helpers
//! (F314).

use torsh_core::device::DeviceType;
use torsh_tensor::broadcast::BroadcastOps;
use torsh_tensor::Tensor;

fn tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
    Tensor::from_data(data, shape, DeviceType::Cpu).expect("tensor creation should succeed")
}

fn iota(shape: Vec<usize>) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    tensor((1..=numel).map(|v| v as f32).collect(), shape)
}

// ---------------------------------------------------------------- F153 ------

/// F153: the stride of the axis inserted by `unsqueeze` must keep a contiguous
/// tensor contiguous.
#[test]
fn f153_unsqueeze_keeps_a_contiguous_tensor_contiguous() {
    let base = iota(vec![3, 4]);

    let front = base.unsqueeze(0).expect("unsqueeze(0) should succeed");
    assert_eq!(front.shape().dims(), &[1, 3, 4]);
    assert_eq!(front.strides(), vec![12, 4, 1]);
    assert!(front.is_contiguous(), "[1, 3, 4] view must be contiguous");

    let middle = base.unsqueeze(1).expect("unsqueeze(1) should succeed");
    assert_eq!(middle.strides(), vec![4, 4, 1]);
    assert!(middle.is_contiguous(), "[3, 1, 4] view must be contiguous");

    let back = base.unsqueeze(2).expect("unsqueeze(2) should succeed");
    assert_eq!(back.strides(), vec![4, 1, 1]);
    assert!(back.is_contiguous(), "[3, 4, 1] view must be contiguous");

    // Data is unaffected by the metadata fix.
    assert_eq!(
        front.to_vec().expect("to_vec"),
        base.to_vec().expect("to_vec")
    );
}

/// F153 consequence: a contiguous unsqueezed tensor can be re-viewed without a
/// copy instead of being rejected as "non-contiguous".
#[test]
fn f153_unsqueezed_tensor_can_be_viewed_again() {
    let base = iota(vec![3, 4]);
    let unsqueezed = base.unsqueeze(0).expect("unsqueeze should succeed");
    let reviewed = unsqueezed
        .view_as(&[3, 4])
        .expect("re-viewing a contiguous unsqueeze must not fail");
    assert_eq!(reviewed.shape().dims(), &[3, 4]);
    assert_eq!(
        reviewed.to_vec().expect("to_vec"),
        base.to_vec().expect("to_vec")
    );
}

// ---------------------------------------------------------- F152 / F272 -----

/// F152: `to_vec()` on a strided view must still return the data in view order.
///
/// This is the trap the `is_view()` redefinition could have broken.
#[test]
fn f152_strided_views_still_read_in_view_order() {
    let base = iota(vec![2, 3]);

    let transposed = base.transpose_view(0, 1).expect("transpose_view");
    assert_eq!(
        transposed.to_vec().expect("to_vec"),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );

    let sliced = base.slice_tensor(1, 1, 3).expect("slice_tensor");
    assert_eq!(sliced.to_vec().expect("to_vec"), vec![2.0, 3.0, 5.0, 6.0]);

    let permuted = iota(vec![2, 3, 4]).permute(&[2, 0, 1]).expect("permute");
    assert_eq!(permuted.shape().dims(), &[4, 2, 3]);
    assert_eq!(permuted.get(&[3, 1, 2]).expect("get"), 24.0);
}

/// F272: a view keeps its source alive and reachable instead of storing a
/// reference that is dead on arrival.
#[test]
fn f272_views_share_storage_with_their_source() {
    let base = iota(vec![2, 3]);
    let view = base.transpose_view(0, 1).expect("transpose_view");

    // Writing through the view is visible in the source: the view really does
    // alias the base's storage.
    view.set(&[0, 1], 40.0).expect("set through the view");
    assert_eq!(base.get(&[1, 0]).expect("get"), 40.0);

    // ... and vice versa.
    base.set(&[0, 2], 30.0).expect("set through the base");
    assert_eq!(view.get(&[2, 0]).expect("get"), 30.0);
}

// ---------------------------------------------------------------- F156 ------

/// F156: `reshape`/`view` must not sever the autograd graph.
#[test]
fn f156_reshape_preserves_requires_grad_and_gradient_flow() {
    let x = iota(vec![2, 3]).requires_grad_(true);
    let y = x.reshape(&[3, 2]).expect("reshape should succeed");
    assert!(
        y.requires_grad(),
        "reshape must propagate requires_grad from its input"
    );

    let seed = tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    y.backward_with_grad(Some(&seed))
        .expect("backward through reshape should succeed");

    let grad = x
        .grad()
        .expect("reshape must forward a gradient to its input");
    assert_eq!(grad.shape().dims(), &[2, 3]);
    assert_eq!(
        grad.to_vec().expect("to_vec"),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

/// F156: reshaping a contiguous tensor is a genuine zero-copy view.
#[test]
fn f156_reshape_of_a_contiguous_tensor_is_a_view() {
    let base = iota(vec![2, 3]);
    let flat = base.reshape(&[6]).expect("reshape should succeed");
    assert_eq!(flat.shape().dims(), &[6]);

    flat.set(&[0], 90.0).expect("set through the reshape");
    assert_eq!(
        base.get(&[0, 0]).expect("get"),
        90.0,
        "reshape must share storage with its source"
    );
}

/// F156: `squeeze`/`unsqueeze`/`squeeze_all` keep the graph connected too.
#[test]
fn f156_squeeze_and_unsqueeze_keep_the_graph_connected() {
    let x = iota(vec![1, 3]).requires_grad_(true);

    let squeezed = x.squeeze(0).expect("squeeze should succeed");
    assert!(squeezed.requires_grad());
    let seed = tensor(vec![1.0, 2.0, 3.0], vec![3]);
    squeezed
        .backward_with_grad(Some(&seed))
        .expect("backward through squeeze");
    let grad = x.grad().expect("squeeze must forward a gradient");
    assert_eq!(grad.shape().dims(), &[1, 3]);
    assert_eq!(grad.to_vec().expect("to_vec"), vec![1.0, 2.0, 3.0]);

    let y = iota(vec![3]).requires_grad_(true);
    let unsqueezed = y.unsqueeze(0).expect("unsqueeze should succeed");
    assert!(unsqueezed.requires_grad());
    let seed = tensor(vec![4.0, 5.0, 6.0], vec![1, 3]);
    unsqueezed
        .backward_with_grad(Some(&seed))
        .expect("backward through unsqueeze");
    let grad = y.grad().expect("unsqueeze must forward a gradient");
    assert_eq!(grad.to_vec().expect("to_vec"), vec![4.0, 5.0, 6.0]);

    let z = iota(vec![1, 3, 1]).requires_grad_(true);
    let all = z.squeeze_all().expect("squeeze_all should succeed");
    assert_eq!(all.shape().dims(), &[3]);
    assert!(
        all.requires_grad(),
        "squeeze_all must propagate requires_grad"
    );
    let seed = tensor(vec![7.0, 8.0, 9.0], vec![3]);
    all.backward_with_grad(Some(&seed))
        .expect("backward through squeeze_all");
    let grad = z.grad().expect("squeeze_all must forward a gradient");
    assert_eq!(grad.shape().dims(), &[1, 3, 1]);
    assert_eq!(grad.to_vec().expect("to_vec"), vec![7.0, 8.0, 9.0]);
}

// ---------------------------------------------------------------- F157 ------

/// F157: `transpose` must be a view for every rank, not a copy for 2-D.
#[test]
fn f157_transpose_is_a_view_at_every_rank() {
    let matrix = iota(vec![2, 3]);
    let transposed = matrix
        .transpose(0, 1)
        .expect("2-D transpose should succeed");
    assert_eq!(transposed.shape().dims(), &[3, 2]);
    assert_eq!(
        transposed.strides(),
        vec![1, 3],
        "2-D transpose must swap strides instead of repacking the buffer"
    );
    assert!(transposed.is_view(), "2-D transpose must be a view");
    assert_eq!(
        transposed.to_vec().expect("to_vec"),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );

    // Aliasing is the same as for the N-D path.
    transposed.set(&[0, 1], 40.0).expect("set through the view");
    assert_eq!(matrix.get(&[1, 0]).expect("get"), 40.0);

    let cube = iota(vec![2, 3, 4]);
    let swapped = cube.transpose(0, 2).expect("3-D transpose should succeed");
    assert!(swapped.is_view());
    swapped.set(&[0, 0, 1], 99.0).expect("set through the view");
    assert_eq!(cube.get(&[1, 0, 0]).expect("get"), 99.0);
}

/// F157: the transpose view carries the autograd record.
#[test]
fn f157_transpose_backward_transposes_the_gradient() {
    let x = iota(vec![2, 3]).requires_grad_(true);
    let y = x.transpose(0, 1).expect("transpose should succeed");
    assert!(y.requires_grad(), "transpose must propagate requires_grad");

    // A non-symmetric seed distinguishes the permutation from its inverse.
    let seed = tensor(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![3, 2]);
    y.backward_with_grad(Some(&seed))
        .expect("backward through transpose");

    let grad = x.grad().expect("transpose must forward a gradient");
    assert_eq!(grad.shape().dims(), &[2, 3]);
    assert_eq!(
        grad.to_vec().expect("to_vec"),
        vec![10.0, 30.0, 50.0, 20.0, 40.0, 60.0]
    );
}

/// F157/F156: `permute` records the axis order it applied.
#[test]
fn f157_permute_backward_undoes_the_permutation() {
    let x = iota(vec![2, 3, 4]).requires_grad_(true);
    let y = x.permute(&[2, 0, 1]).expect("permute should succeed");
    assert_eq!(y.shape().dims(), &[4, 2, 3]);
    assert!(y.requires_grad());

    let seed = iota(vec![4, 2, 3]);
    y.backward_with_grad(Some(&seed))
        .expect("backward through permute");

    let grad = x.grad().expect("permute must forward a gradient");
    assert_eq!(grad.shape().dims(), &[2, 3, 4]);
    // grad[i, j, k] == seed[k, i, j]
    let seed_data = seed.to_vec().expect("to_vec");
    let grad_data = grad.to_vec().expect("to_vec");
    for i in 0..2usize {
        for j in 0..3usize {
            for k in 0..4usize {
                let got = grad_data[(i * 3 + j) * 4 + k];
                let expected = seed_data[(k * 2 + i) * 3 + j];
                assert_eq!(got, expected, "grad[{i},{j},{k}]");
            }
        }
    }
}

// ---------------------------------------------------------------- F269 ------

/// F269: `expand` must keep `requires_grad` and sum the gradient back over the
/// broadcast axes.
#[test]
fn f269_expand_preserves_requires_grad_and_sums_the_gradient() {
    let x = tensor(vec![1.0, 2.0], vec![1, 2]).requires_grad_(true);
    let expanded = x.expand(&[3, 2]).expect("expand should succeed");
    assert!(
        expanded.requires_grad(),
        "expand must propagate requires_grad"
    );

    let seed = tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    expanded
        .backward_with_grad(Some(&seed))
        .expect("backward through expand");

    let grad = x.grad().expect("expand must forward a gradient");
    assert_eq!(grad.shape().dims(), &[1, 2]);
    assert_eq!(grad.to_vec().expect("to_vec"), vec![9.0, 12.0]);
}

/// F269: a brand-new leading axis is summed away as well.
#[test]
fn f269_expand_reduces_new_leading_axes() {
    let x = tensor(vec![1.0, 2.0], vec![2]).requires_grad_(true);
    let expanded = x
        .broadcast_to(&[3, 2])
        .expect("broadcast_to should succeed");
    assert!(expanded.requires_grad());

    let seed = tensor(vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0], vec![3, 2]);
    expanded
        .backward_with_grad(Some(&seed))
        .expect("backward through broadcast_to");

    let grad = x.grad().expect("broadcast_to must forward a gradient");
    assert_eq!(grad.shape().dims(), &[2]);
    assert_eq!(grad.to_vec().expect("to_vec"), vec![6.0, 6.0]);
}

// ---------------------------------------------------------------- F314 ------

/// F314: the broadcast helpers must not divide by a zero-sized dimension.
#[test]
fn f314_flat_to_multi_index_handles_zero_sized_dims() {
    // A zero-sized shape addresses no elements at all: the decomposition is the
    // all-zero index instead of a division-by-zero panic.
    assert_eq!(
        BroadcastOps::flat_to_multi_index(0, &[2, 0, 3]),
        vec![0, 0, 0]
    );
    assert_eq!(BroadcastOps::flat_to_multi_index(7, &[0]), vec![0]);

    // Non-degenerate shapes are unchanged.
    assert_eq!(
        BroadcastOps::flat_to_multi_index(5, &[2, 3, 4]),
        vec![0, 1, 1]
    );
}

/// F314: `get_broadcast_info` reports an error instead of panicking.
#[test]
fn f314_get_broadcast_info_rejects_zero_sized_dims() {
    assert!(
        BroadcastOps::get_broadcast_info(&[2, 0, 3], &[2, 0, 3]).is_err(),
        "zero-sized dimensions must be reported, not divided by"
    );
    assert!(BroadcastOps::get_broadcast_info(&[0], &[1]).is_err());

    // Scalars (empty shapes) still work: the empty product is 1.
    let info = BroadcastOps::get_broadcast_info(&[], &[2, 3]).expect("scalar broadcast");
    assert_eq!(info.broadcast_shape, vec![2, 3]);
}

/// F157 follow-through: now that a 2-D transpose is a view, the ops that used
/// to receive a freshly packed buffer must still read it in view order.
///
/// The three sizes straddle the thresholds the f32 SIMD dispatch uses. They all
/// come out right because `add`/`mul` materialise through `to_vec()`; the
/// `with_data_slice`-based SIMD entry points in `ops/simd` were confirmed *not*
/// to be reached from here, so this pins the public behaviour, not those paths.
#[test]
fn f157_elementwise_and_matmul_read_a_transposed_view_correctly() {
    for side in [8usize, 64, 256] {
        let a = iota(vec![side, side]);
        let b = iota(vec![side, side]);
        let sum = a
            .transpose(0, 1)
            .expect("transpose")
            .add(&b)
            .expect("adding a transposed view must succeed");

        let a_data = a.to_vec().expect("to_vec");
        let b_data = b.to_vec().expect("to_vec");
        let expected: Vec<f32> = (0..side * side)
            .map(|index| {
                let (row, col) = (index / side, index % side);
                a_data[col * side + row] + b_data[index]
            })
            .collect();
        assert_eq!(
            sum.to_vec().expect("to_vec"),
            expected,
            "elementwise add on a {side}x{side} transposed view"
        );
    }

    // The classic `x @ w^T` shape: matmul must see the transposed operand.
    let x = iota(vec![2, 3]);
    let w = iota(vec![4, 3]);
    let product = x
        .matmul(&w.transpose(0, 1).expect("transpose"))
        .expect("matmul with a transposed view");
    assert_eq!(product.shape().dims(), &[2, 4]);
    let x_data = x.to_vec().expect("to_vec");
    let w_data = w.to_vec().expect("to_vec");
    let expected: Vec<f32> = (0..2 * 4)
        .map(|index| {
            let (row, col) = (index / 4, index % 4);
            (0..3)
                .map(|k| x_data[row * 3 + k] * w_data[col * 3 + k])
                .sum()
        })
        .collect();
    assert_eq!(product.to_vec().expect("to_vec"), expected);
}
