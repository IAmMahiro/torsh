//! Production-hardening regression tests for dimension-aware tensor operations,
//! matrix multiplication and in-place/copy-on-write semantics.
//!
//! Every test in this file corresponds to a finding from the ToRSh production
//! hardening campaign (wave 1, `torsh-tensor` dim-ops / matmul scope).  Expected
//! values are hand-computed against PyTorch semantics.

use torsh_core::device::DeviceType;
use torsh_tensor::stats::StatMode;
use torsh_tensor::Tensor;

fn t2(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
    Tensor::from_data(data, shape, DeviceType::Cpu).expect("tensor creation should succeed")
}

// ---------------------------------------------------------------------------
// F054 - cumsum must respect `dim` and reset the accumulator per slice
// ---------------------------------------------------------------------------

#[test]
fn f054_cumsum_dim0_2d() {
    // [[1,2,3],[4,5,6]] -> cumsum(0) = [[1,2,3],[5,7,9]]
    let t = t2(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let out = t.cumsum(0).expect("cumsum should succeed");
    assert_eq!(out.shape().dims(), &[2, 3]);
    assert_eq!(
        out.to_vec().expect("to_vec"),
        vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0]
    );
}

#[test]
fn f054_cumsum_last_dim_2d() {
    // [[1,2,3],[4,5,6]] -> cumsum(-1) = [[1,3,6],[4,9,15]]
    let t = t2(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let out = t.cumsum(-1).expect("cumsum should succeed");
    assert_eq!(
        out.to_vec().expect("to_vec"),
        vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]
    );
}

#[test]
fn f054_cumsum_3d_middle_dim() {
    // shape [2,2,3] with 1..=12, cumsum along dim 1
    let t = t2((1..=12).map(|v| v as f32).collect(), vec![2, 2, 3]);
    let out = t.cumsum(1).expect("cumsum should succeed");
    assert_eq!(
        out.to_vec().expect("to_vec"),
        vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 7.0, 8.0, 9.0, 17.0, 19.0, 21.0]
    );
}

#[test]
fn f054_cumprod_dim0_2d() {
    // [[1,2,3],[4,5,6]] -> cumprod(0) = [[1,2,3],[4,10,18]]
    let t = t2(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let out = t.cumprod(0).expect("cumprod should succeed");
    assert_eq!(
        out.to_vec().expect("to_vec"),
        vec![1.0, 2.0, 3.0, 4.0, 10.0, 18.0]
    );
    let out_last = t.cumprod(-1).expect("cumprod should succeed");
    assert_eq!(
        out_last.to_vec().expect("to_vec"),
        vec![1.0, 2.0, 6.0, 4.0, 20.0, 120.0]
    );
}

// ---------------------------------------------------------------------------
// F055 - sort must honour `dim` and `descending`
// ---------------------------------------------------------------------------

#[test]
fn f055_sort_dim1_ascending_values() {
    let t = t2(vec![3.0, 1.0, 2.0, 0.0, 5.0, 4.0], vec![2, 3]);
    let (values, _indices) = t.sort(Some(1), false).expect("sort should succeed");
    assert_eq!(values.shape().dims(), &[2, 3]);
    assert_eq!(
        values.to_vec().expect("to_vec"),
        vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0]
    );
}

#[test]
fn f055_sort_dim0_ascending_values() {
    let t = t2(vec![3.0, 1.0, 2.0, 0.0, 5.0, 4.0], vec![2, 3]);
    let (values, _indices) = t.sort(Some(0), false).expect("sort should succeed");
    assert_eq!(
        values.to_vec().expect("to_vec"),
        vec![0.0, 1.0, 2.0, 3.0, 5.0, 4.0]
    );
}

#[test]
fn f055_sort_descending_values() {
    let t = t2(vec![3.0, 1.0, 2.0, 0.0, 5.0, 4.0], vec![2, 3]);
    let (values, _indices) = t.sort(Some(1), true).expect("sort should succeed");
    assert_eq!(
        values.to_vec().expect("to_vec"),
        vec![3.0, 2.0, 1.0, 5.0, 4.0, 0.0]
    );
}

#[test]
fn f055_sort_defaults_to_last_dim() {
    let t = t2(vec![3.0, 1.0, 2.0, 0.0, 5.0, 4.0], vec![2, 3]);
    let (values, _indices) = t.sort(None, false).expect("sort should succeed");
    assert_eq!(
        values.to_vec().expect("to_vec"),
        vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0]
    );
}

#[test]
fn f055_sort_returns_i64_positions_within_dim() {
    let t = t2(vec![3.0, 1.0, 2.0, 0.0, 5.0, 4.0], vec![2, 3]);
    let (_values, indices) = t.sort(Some(1), false).expect("sort should succeed");
    assert_eq!(indices.shape().dims(), &[2, 3]);
    let idx: Vec<i64> = indices.to_vec().expect("to_vec");
    assert_eq!(idx, vec![1i64, 2, 0, 0, 2, 1]);
}

#[test]
fn f055_sort_3d_along_dim0() {
    // shape [2,2,2]: sorting along dim 0 compares matching (row, col) cells.
    let t = t2(vec![5.0, 1.0, 8.0, 2.0, 3.0, 7.0, 0.0, 9.0], vec![2, 2, 2]);
    let (values, indices) = t.sort(Some(0), false).expect("sort should succeed");
    assert_eq!(values.shape().dims(), &[2, 2, 2]);
    assert_eq!(
        values.to_vec().expect("to_vec"),
        vec![3.0, 1.0, 0.0, 2.0, 5.0, 7.0, 8.0, 9.0]
    );
    assert_eq!(
        indices.to_vec().expect("to_vec"),
        vec![1i64, 0, 1, 0, 0, 1, 0, 1]
    );
}

#[test]
fn f055_median_uses_dim_aware_sort() {
    // Row-wise median of [[3,1,2],[9,5,7]] is [2, 7]; before the sort fix the
    // globally-sorted buffer produced [2, 7] only by accident on 1-D inputs.
    let t = t2(vec![3.0, 1.0, 2.0, 9.0, 5.0, 7.0], vec![2, 3]);
    let med = t.median(Some(1), false).expect("median should succeed");
    assert_eq!(med.shape().dims(), &[2]);
    assert_eq!(med.to_vec().expect("to_vec"), vec![2.0, 7.0]);
}

// ---------------------------------------------------------------------------
// F056 - argmin/argmax with Some(dim) must reduce that dimension
// ---------------------------------------------------------------------------

#[test]
fn f056_argmax_dim1_shape_and_values() {
    // [[1,5,2,3],[9,2,3,1],[0,0,7,2]] -> argmax(dim=1) = [1,0,2]
    let t = t2(
        vec![1.0, 5.0, 2.0, 3.0, 9.0, 2.0, 3.0, 1.0, 0.0, 0.0, 7.0, 2.0],
        vec![3, 4],
    );
    let out = t.argmax(Some(1)).expect("argmax should succeed");
    assert_eq!(out.shape().dims(), &[3]);
    assert_eq!(out.to_vec().expect("to_vec"), vec![1i64, 0, 2]);
}

#[test]
fn f056_argmin_dim0_shape_and_values() {
    let t = t2(
        vec![1.0, 5.0, 2.0, 3.0, 9.0, 2.0, 3.0, 1.0, 0.0, 0.0, 7.0, 2.0],
        vec![3, 4],
    );
    let out = t.argmin(Some(0)).expect("argmin should succeed");
    assert_eq!(out.shape().dims(), &[4]);
    assert_eq!(out.to_vec().expect("to_vec"), vec![2i64, 2, 0, 1]);
}

#[test]
fn f056_argmax_global_is_scalar() {
    let t = t2(vec![1.0, 5.0, 2.0, 3.0], vec![2, 2]);
    let out = t.argmax(None).expect("argmax should succeed");
    assert_eq!(out.shape().dims(), &[] as &[usize]);
    assert_eq!(out.to_vec().expect("to_vec"), vec![1i64]);
}

// ---------------------------------------------------------------------------
// F057 / F165 - multi-dimensional sum_dim / mean
// ---------------------------------------------------------------------------

#[test]
fn f057_sum_dim_multiple_dims() {
    // [2,3,4] with 0..23; sum over dims [0,1] -> shape [4] = [60,66,72,78]
    let t = t2((0..24).map(|v| v as f32).collect(), vec![2, 3, 4]);
    let out = t.sum_dim(&[0, 1], false).expect("sum_dim should succeed");
    assert_eq!(out.shape().dims(), &[4]);
    assert_eq!(out.to_vec().expect("to_vec"), vec![60.0, 66.0, 72.0, 78.0]);
}

#[test]
fn f057_sum_dim_multiple_dims_keepdim() {
    let t = t2((0..24).map(|v| v as f32).collect(), vec![2, 3, 4]);
    let out = t.sum_dim(&[0, 1], true).expect("sum_dim should succeed");
    assert_eq!(out.shape().dims(), &[1, 1, 4]);
    assert_eq!(out.to_vec().expect("to_vec"), vec![60.0, 66.0, 72.0, 78.0]);
}

#[test]
fn f057_sum_dim_negative_index() {
    let t = t2((0..24).map(|v| v as f32).collect(), vec![2, 3, 4]);
    let out = t.sum_dim(&[-1], false).expect("sum_dim should succeed");
    assert_eq!(out.shape().dims(), &[2, 3]);
    assert_eq!(
        out.to_vec().expect("to_vec"),
        vec![6.0, 22.0, 38.0, 54.0, 70.0, 86.0]
    );
}

#[test]
fn f057_mean_multiple_dims() {
    let t = t2((0..24).map(|v| v as f32).collect(), vec![2, 3, 4]);
    let out = t.mean(Some(&[0, 1]), false).expect("mean should succeed");
    assert_eq!(out.shape().dims(), &[4]);
    assert_eq!(out.to_vec().expect("to_vec"), vec![10.0, 11.0, 12.0, 13.0]);
}

#[test]
fn f057_mean_out_of_range_dim_returns_error() {
    let t = t2(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    assert!(
        t.mean(Some(&[5]), false).is_err(),
        "mean with an out-of-range dim must return Err, not panic"
    );
}

// ---------------------------------------------------------------------------
// F154 - var/std with dims=Some(..)
// ---------------------------------------------------------------------------

#[test]
fn f154_var_dim0_population() {
    // [[1,2],[3,4]] -> var over dim 0 (population) = [1,1]
    let t = t2(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let out = t
        .var(Some(&[0]), false, StatMode::Population)
        .expect("var should succeed");
    assert_eq!(out.shape().dims(), &[2]);
    let v = out.to_vec().expect("to_vec");
    assert!((v[0] - 1.0).abs() < 1e-6, "got {:?}", v);
    assert!((v[1] - 1.0).abs() < 1e-6, "got {:?}", v);
}

#[test]
fn f154_var_dim1_sample() {
    // [[1,2,3],[4,6,8]] -> sample var over dim 1 = [1, 4]
    let t = t2(vec![1.0, 2.0, 3.0, 4.0, 6.0, 8.0], vec![2, 3]);
    let out = t
        .var(Some(&[1]), false, StatMode::Sample)
        .expect("var should succeed");
    assert_eq!(out.shape().dims(), &[2]);
    let v = out.to_vec().expect("to_vec");
    assert!((v[0] - 1.0).abs() < 1e-6, "got {:?}", v);
    assert!((v[1] - 4.0).abs() < 1e-6, "got {:?}", v);
}

#[test]
fn f154_std_dim1_keepdim() {
    let t = t2(vec![1.0, 2.0, 3.0, 4.0, 6.0, 8.0], vec![2, 3]);
    let out = t
        .std(Some(&[1]), true, StatMode::Sample)
        .expect("std should succeed");
    assert_eq!(out.shape().dims(), &[2, 1]);
    let v = out.to_vec().expect("to_vec");
    assert!((v[0] - 1.0).abs() < 1e-6, "got {:?}", v);
    assert!((v[1] - 2.0).abs() < 1e-6, "got {:?}", v);
}

// ---------------------------------------------------------------------------
// F058 - in-place ops on strided views must not corrupt data
// ---------------------------------------------------------------------------

#[test]
fn f058_inplace_scalar_mul_on_transposed_view() {
    let base = t2(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let mut view = base
        .transpose_view(0, 1)
        .expect("transpose_view should succeed");
    assert_eq!(view.shape().dims(), &[3, 2]);
    assert_eq!(
        view.to_vec().expect("to_vec"),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        "view must expose transposed order before mutation"
    );

    view.mul_scalar_(2.0).expect("mul_scalar_ should succeed");
    assert_eq!(
        view.to_vec().expect("to_vec"),
        vec![2.0, 8.0, 4.0, 10.0, 6.0, 12.0],
        "in-place scalar multiply on a transposed view must double each element in view order"
    );
}

#[test]
fn f058_inplace_add_scalar_on_narrow_view() {
    let base = t2((1..=6).map(|v| v as f32).collect(), vec![2, 3]);
    let mut view = base.transpose_view(0, 1).expect("transpose_view");
    view.add_scalar_(1.0).expect("add_scalar_ should succeed");
    assert_eq!(
        view.to_vec().expect("to_vec"),
        vec![2.0, 5.0, 3.0, 6.0, 4.0, 7.0]
    );
}

#[test]
fn f058_inplace_detaches_the_view_metadata() {
    let base = t2((1..=6).map(|v| v as f32).collect(), vec![2, 3]);
    let mut view = base.transpose_view(0, 1).expect("transpose_view");
    assert!(view.is_view(), "transpose_view must produce a view");

    view.mul_scalar_(2.0).expect("mul_scalar_ should succeed");
    assert!(
        !view.is_view(),
        "after an in-place write the tensor must be a contiguous base, not a stale view"
    );
    // The original tensor is untouched (write-through aliasing is not supported).
    assert_eq!(
        base.to_vec().expect("to_vec"),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    // A second in-place op keeps operating on the right elements.
    view.mul_scalar_(0.5).expect("mul_scalar_ should succeed");
    assert_eq!(
        view.to_vec().expect("to_vec"),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
}

#[test]
fn f058_apply_on_view_uses_view_order() {
    let base = t2((1..=6).map(|v| v as f32).collect(), vec![2, 3]);
    let mut view = base.transpose_view(0, 1).expect("transpose_view");
    view.apply_(|x| x * 10.0).expect("apply_ should succeed");
    assert_eq!(view.shape().dims(), &[3, 2]);
    assert_eq!(
        view.to_vec().expect("to_vec"),
        vec![10.0, 40.0, 20.0, 50.0, 30.0, 60.0]
    );
}

// ---------------------------------------------------------------------------
// F065 - matmul: PyTorch shape semantics + correctness at BLAS-sized inputs
// ---------------------------------------------------------------------------

#[test]
fn f065_matmul_2d_2d() {
    let a = t2(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = t2(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let c = a.matmul(&b).expect("matmul should succeed");
    assert_eq!(c.shape().dims(), &[2, 2]);
    assert_eq!(c.to_vec().expect("to_vec"), vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn f065_matmul_1d_1d_dot() {
    let a = t2(vec![1.0, 2.0, 3.0], vec![3]);
    let b = t2(vec![4.0, 5.0, 6.0], vec![3]);
    let c = a.matmul(&b).expect("matmul should succeed");
    assert_eq!(c.shape().dims(), &[] as &[usize]);
    assert_eq!(c.to_vec().expect("to_vec"), vec![32.0]);
}

#[test]
fn f065_matmul_1d_2d() {
    let a = t2(vec![1.0, 2.0], vec![2]);
    let b = t2(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let c = a.matmul(&b).expect("matmul should succeed");
    assert_eq!(c.shape().dims(), &[3]);
    assert_eq!(c.to_vec().expect("to_vec"), vec![9.0, 12.0, 15.0]);
}

#[test]
fn f065_matmul_2d_1d() {
    let a = t2(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = t2(vec![1.0, 1.0], vec![2]);
    let c = a.matmul(&b).expect("matmul should succeed");
    assert_eq!(c.shape().dims(), &[2]);
    assert_eq!(c.to_vec().expect("to_vec"), vec![3.0, 7.0]);
}

#[test]
fn f065_matmul_batched_3d() {
    let a = t2((1..=12).map(|v| v as f32).collect(), vec![2, 2, 3]);
    let b = t2((1..=12).map(|v| v as f32).collect(), vec![2, 3, 2]);
    let c = a.matmul(&b).expect("matmul should succeed");
    assert_eq!(c.shape().dims(), &[2, 2, 2]);
    assert_eq!(
        c.to_vec().expect("to_vec"),
        vec![22.0, 28.0, 49.0, 64.0, 220.0, 244.0, 301.0, 334.0]
    );
}

#[test]
fn f065_matmul_batched_broadcast_rhs_2d() {
    let a = t2((1..=12).map(|v| v as f32).collect(), vec![2, 2, 3]);
    let b = t2(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let c = a.matmul(&b).expect("matmul should succeed");
    assert_eq!(c.shape().dims(), &[2, 2, 2]);
    assert_eq!(
        c.to_vec().expect("to_vec"),
        vec![22.0, 28.0, 49.0, 64.0, 76.0, 100.0, 103.0, 136.0]
    );
}

#[test]
fn f065_matmul_shape_mismatch_errors() {
    let a = t2(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = t2(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    assert!(a.matmul(&b).is_err(), "inner dimensions must be checked");
}

#[test]
fn f065_matmul_large_matches_reference() {
    // 96x96 @ 96x96: exercises the blocked/BLAS path and compares against a
    // straightforward reference implementation computed here.
    let n = 96usize;
    let a_data: Vec<f32> = (0..n * n).map(|i| ((i % 17) as f32) * 0.25 - 2.0).collect();
    let b_data: Vec<f32> = (0..n * n).map(|i| ((i % 13) as f32) * 0.5 - 3.0).collect();
    let a = t2(a_data.clone(), vec![n, n]);
    let b = t2(b_data.clone(), vec![n, n]);
    let c = a.matmul(&b).expect("matmul should succeed");
    let got = c.to_vec().expect("to_vec");

    let mut expected = vec![0.0f32; n * n];
    for (i, row) in expected.chunks_mut(n).enumerate() {
        for (kk, &a_ik) in a_data[i * n..(i + 1) * n].iter().enumerate() {
            for (j, out) in row.iter_mut().enumerate() {
                *out += a_ik * b_data[kk * n + j];
            }
        }
    }
    for (idx, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() <= 1e-3 * e.abs().max(1.0),
            "mismatch at {idx}: got {g}, expected {e}"
        );
    }
}

#[test]
fn f065_matmul_2d_autograd_still_recorded() {
    let a = t2(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).requires_grad_(true);
    let b = t2(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).requires_grad_(true);
    let c = a.matmul(&b).expect("matmul should succeed");
    assert!(c.requires_grad(), "2-D matmul must keep recording autograd");
    let loss = c.sum().expect("sum");
    loss.backward().expect("backward");
    assert_eq!(
        a.grad().expect("grad").to_vec().expect("to_vec"),
        vec![11.0, 15.0, 11.0, 15.0]
    );
}

// ---------------------------------------------------------------------------
// F265 - cat correctness on multi-dimensional inputs (perf fix regression guard)
// ---------------------------------------------------------------------------

#[test]
fn f265_cat_along_last_dim_3d() {
    let a = t2((1..=8).map(|v| v as f32).collect(), vec![2, 2, 2]);
    let b = t2((9..=12).map(|v| v as f32).collect(), vec![2, 2, 1]);
    let out = Tensor::<f32>::cat(&[&a, &b], 2).expect("cat should succeed");
    assert_eq!(out.shape().dims(), &[2, 2, 3]);
    assert_eq!(
        out.to_vec().expect("to_vec"),
        vec![1.0, 2.0, 9.0, 3.0, 4.0, 10.0, 5.0, 6.0, 11.0, 7.0, 8.0, 12.0]
    );
}

#[test]
fn f265_cat_of_view_inputs() {
    let base = t2((1..=6).map(|v| v as f32).collect(), vec![2, 3]);
    let view = base.transpose_view(0, 1).expect("transpose_view");
    let out = Tensor::<f32>::cat(&[&view, &view], 1).expect("cat should succeed");
    assert_eq!(out.shape().dims(), &[3, 4]);
    assert_eq!(
        out.to_vec().expect("to_vec"),
        vec![1.0, 4.0, 1.0, 4.0, 2.0, 5.0, 2.0, 5.0, 3.0, 6.0, 3.0, 6.0]
    );
}

// ---------------------------------------------------------------------------
// F168 - map / apply_ must keep producing correct values (alloc reduction guard)
// ---------------------------------------------------------------------------

#[test]
fn f168_apply_and_map_values() {
    let mut t = t2(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    t.apply_(|x| x * 3.0).expect("apply_ should succeed");
    assert_eq!(t.to_vec().expect("to_vec"), vec![3.0, 6.0, 9.0, 12.0]);

    let mapped = t.map(|x| x + 1.0).expect("map should succeed");
    assert_eq!(mapped.to_vec().expect("to_vec"), vec![4.0, 7.0, 10.0, 13.0]);
    assert_eq!(t.to_vec().expect("to_vec"), vec![3.0, 6.0, 9.0, 12.0]);
}

// ---------------------------------------------------------------------------
// F313 - min/max symmetry: amin/amax accept the same (dim, keepdim) arguments
// ---------------------------------------------------------------------------

#[test]
fn f313_amin_amax_symmetric_signatures() {
    let t = t2(vec![1.0, 5.0, 3.0, 2.0, 8.0, 0.0], vec![2, 3]);

    for keepdim in [false, true] {
        let mins = t.amin(Some(1), keepdim).expect("amin should succeed");
        let maxs = t.amax(Some(1), keepdim).expect("amax should succeed");
        assert_eq!(mins.shape().dims(), maxs.shape().dims());
        assert_eq!(mins.to_vec().expect("to_vec"), vec![1.0, 0.0]);
        assert_eq!(maxs.to_vec().expect("to_vec"), vec![5.0, 8.0]);
    }

    // Global forms agree with the parameter-less min() and with max(None, ..)
    assert_eq!(
        t.amin(None, false).expect("amin").item().expect("item"),
        t.min().expect("min").item().expect("item")
    );
    assert_eq!(
        t.amax(None, false).expect("amax").item().expect("item"),
        t.max(None, false).expect("max").item().expect("item")
    );

    // min_dim / max_dim agree with amin / amax on the same axis
    assert_eq!(
        t.min_dim(0, false).expect("min_dim").to_vec().expect("v"),
        t.amin(Some(0), false).expect("amin").to_vec().expect("v")
    );
    assert_eq!(
        t.max_dim(0, false).expect("max_dim").to_vec().expect("v"),
        t.amax(Some(0), false).expect("amax").to_vec().expect("v")
    );
}

#[test]
fn f313_dim_reductions_are_correct_on_views() {
    let base = t2((1..=6).map(|v| v as f32).collect(), vec![2, 3]);
    let view = base.transpose_view(0, 1).expect("transpose_view");
    // view = [[1,4],[2,5],[3,6]]
    assert_eq!(
        view.amax(Some(1), false)
            .expect("amax")
            .to_vec()
            .expect("v"),
        vec![4.0, 5.0, 6.0]
    );
    assert_eq!(
        view.amin(Some(0), false)
            .expect("amin")
            .to_vec()
            .expect("v"),
        vec![1.0, 4.0]
    );
    assert_eq!(
        view.sum_dim(&[0], false)
            .expect("sum_dim")
            .to_vec()
            .expect("v"),
        vec![6.0, 15.0]
    );
    assert_eq!(
        view.argmax(Some(0)).expect("argmax").to_vec().expect("v"),
        vec![2i64, 2]
    );
}

// ---------------------------------------------------------------------------
// F057 - dimension validation happens before any indexing
// ---------------------------------------------------------------------------

#[test]
fn f057_multi_dim_out_of_range_returns_error() {
    let t = t2((0..24).map(|v| v as f32).collect(), vec![2, 3, 4]);
    assert!(t.sum_dim(&[0, 9], false).is_err());
    assert!(t.mean(Some(&[0, 9]), false).is_err());
    assert!(t.var(Some(&[9]), false, StatMode::Population).is_err());
}

#[test]
fn f057_sum_dim_duplicate_dims_reduce_once() {
    let t = t2((0..24).map(|v| v as f32).collect(), vec![2, 3, 4]);
    let once = t.sum_dim(&[0], false).expect("sum_dim");
    let twice = t.sum_dim(&[0, 0], false).expect("sum_dim");
    assert_eq!(once.shape().dims(), twice.shape().dims());
    assert_eq!(once.to_vec().expect("v"), twice.to_vec().expect("v"));
}

#[test]
fn f168_apply_on_large_simd_tensor() {
    // >= 2560 f32 elements selects SimdOptimized storage in create_optimal.
    let n = 4096usize;
    let mut t = t2((0..n).map(|i| i as f32).collect(), vec![n]);
    t.apply_(|x| x + 1.0).expect("apply_ should succeed");
    let data = t.to_vec().expect("to_vec");
    assert_eq!(data.len(), n);
    assert_eq!(data[0], 1.0);
    assert_eq!(data[n - 1], n as f32);
}
