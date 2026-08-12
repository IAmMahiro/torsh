//! Wave-3 hardening: close the autograd-graph breaks where structural,
//! indexing, reduction and element-wise ops silently DETACHED the graph.
//!
//! Every test here fails on the pre-Wave-3 tree because the forward op rebuilt
//! its result through `from_data`/`Tensor::map` (→ `Operation::Leaf`) or a bare
//! strided view, so `requires_grad` inputs received `grad == None`. Each test
//! pins the real backward rule with an analytic or finite-difference check.

use torsh_core::device::DeviceType;
use torsh_tensor::Tensor;

fn t32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
    Tensor::from_data(data, shape, DeviceType::Cpu).expect("f32 tensor creation should succeed")
}

/// Distinct, non-uniform ramp so a mis-split / mis-scatter backward cannot pass
/// by symmetry.
fn ramp(shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 + 1.0).collect();
    t32(data, shape.to_vec())
}

/// Central finite-difference gradient of a scalar loss w.r.t. every element of
/// `base` (which carries `shape`).
fn fd_grad<F>(base: &[f32], shape: &[usize], loss: F) -> Vec<f32>
where
    F: Fn(&Tensor<f32>) -> f32,
{
    let eps = 1e-3_f32;
    let mut g = vec![0.0_f32; base.len()];
    for i in 0..base.len() {
        let mut plus = base.to_vec();
        let mut minus = base.to_vec();
        plus[i] += eps;
        minus[i] -= eps;
        let lp = loss(&t32(plus, shape.to_vec()));
        let lm = loss(&t32(minus, shape.to_vec()));
        g[i] = (lp - lm) / (2.0 * eps);
    }
    g
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, ctx: &str) {
    assert_eq!(a.len(), b.len(), "{ctx}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() <= tol,
            "{ctx}: index {i}: {x} vs {y} (tol {tol})"
        );
    }
}

// ---------------------------------------------------------------------------
// Item 1: cat / stack record a multi-input backward.
// ---------------------------------------------------------------------------

/// cat along dim=1 with a leading dim > 1: the discriminating case. A flat
/// (single-block) backward would scatter the wrong slabs.
#[test]
fn cat_backward_splits_grad_along_dim1() {
    let a = ramp(&[2, 3, 4]).requires_grad_(true);
    let b = ramp(&[2, 3, 4]).requires_grad_(true);
    let out = Tensor::cat(&[&a, &b], 1).expect("cat");
    assert_eq!(out.shape().dims(), &[2, 6, 4]);

    let seed = ramp(&[2, 6, 4]);
    out.backward_with_grad(Some(&seed)).expect("cat backward");

    let seed_v = seed.to_vec().expect("seed");
    // a is dim1 in [0,3), b in [3,6).
    let mut exp_a = vec![0.0f32; 2 * 3 * 4];
    let mut exp_b = vec![0.0f32; 2 * 3 * 4];
    for o in 0..2 {
        for d in 0..6 {
            for k in 0..4 {
                let s = ((o * 6 + d) * 4 + k) as usize;
                if d < 3 {
                    exp_a[(o * 3 + d) * 4 + k] = seed_v[s];
                } else {
                    exp_b[(o * 3 + (d - 3)) * 4 + k] = seed_v[s];
                }
            }
        }
    }
    let ga = a
        .grad()
        .expect("a grad must exist (was detached)")
        .to_vec()
        .unwrap();
    let gb = b
        .grad()
        .expect("b grad must exist (was detached)")
        .to_vec()
        .unwrap();
    assert_close(&ga, &exp_a, 1e-5, "cat grad a");
    assert_close(&gb, &exp_b, 1e-5, "cat grad b");
}

/// stack inserts a new dim=1; each input's grad is that slab of the seed.
#[test]
fn stack_backward_selects_grad_along_new_dim1() {
    let a = ramp(&[2, 4]).requires_grad_(true);
    let b = ramp(&[2, 4]).requires_grad_(true);
    let c = ramp(&[2, 4]).requires_grad_(true);
    let out = Tensor::stack(&[a.clone(), b.clone(), c.clone()], 1).expect("stack");
    assert_eq!(out.shape().dims(), &[2, 3, 4]);

    let seed = ramp(&[2, 3, 4]);
    out.backward_with_grad(Some(&seed)).expect("stack backward");
    let seed_v = seed.to_vec().unwrap();

    let mut exp = [vec![0.0f32; 8], vec![0.0f32; 8], vec![0.0f32; 8]];
    for o in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                exp[j][o * 4 + k] = seed_v[(o * 3 + j) * 4 + k];
            }
        }
    }
    for (idx, tns) in [&a, &b, &c].iter().enumerate() {
        let g = tns
            .grad()
            .expect("stack input grad must exist")
            .to_vec()
            .unwrap();
        assert_close(&g, &exp[idx], 1e-5, "stack grad");
    }
}

// ---------------------------------------------------------------------------
// Item 2: narrow / select / slice_tensor / slice_with_step scatter-grad.
// This is what unblocks RNN gradients (F133).
// ---------------------------------------------------------------------------

#[test]
fn narrow_backward_scatters_into_zeros_f133() {
    let x = ramp(&[4, 5]).requires_grad_(true);
    let y = x.narrow(0, 1, 2).expect("narrow"); // rows 1,2 -> [2,5]
    assert_eq!(y.shape().dims(), &[2, 5]);
    let seed = ramp(&[2, 5]);
    y.backward_with_grad(Some(&seed))
        .expect("narrow backward (F133 RNN unblock)");

    let seed_v = seed.to_vec().unwrap();
    let mut exp = vec![0.0f32; 20];
    for r in 0..2 {
        for c in 0..5 {
            exp[(r + 1) * 5 + c] = seed_v[r * 5 + c];
        }
    }
    let g = x
        .grad()
        .expect("narrow input grad must exist (was detached, broke RNN)");
    assert_close(&g.to_vec().unwrap(), &exp, 1e-5, "narrow grad");
}

#[test]
fn select_backward_scatters_row() {
    let x = ramp(&[4, 3]).requires_grad_(true);
    let y = x.select(0, 2).expect("select"); // row 2 -> [3]
    assert_eq!(y.shape().dims(), &[3]);
    let seed = t32(vec![10.0, 20.0, 30.0], vec![3]);
    y.backward_with_grad(Some(&seed)).expect("select backward");
    let mut exp = vec![0.0f32; 12];
    exp[2 * 3] = 10.0;
    exp[2 * 3 + 1] = 20.0;
    exp[2 * 3 + 2] = 30.0;
    let g = x.grad().expect("select input grad must exist");
    assert_close(&g.to_vec().unwrap(), &exp, 1e-5, "select grad");
}

#[test]
fn slice_tensor_backward_scatters_view() {
    let x = ramp(&[3, 4]).requires_grad_(true);
    let y = x.slice_tensor(1, 1, 3).expect("slice_tensor"); // cols 1,2 -> [3,2]
    assert_eq!(y.shape().dims(), &[3, 2]);
    let seed = ramp(&[3, 2]);
    y.backward_with_grad(Some(&seed))
        .expect("slice_tensor backward");
    let seed_v = seed.to_vec().unwrap();
    let mut exp = vec![0.0f32; 12];
    for r in 0..3 {
        for c in 0..2 {
            exp[r * 4 + (c + 1)] = seed_v[r * 2 + c];
        }
    }
    let g = x
        .grad()
        .expect("slice_tensor input grad must exist (bare strided view detached)");
    assert_close(&g.to_vec().unwrap(), &exp, 1e-5, "slice_tensor grad");
}

// ---------------------------------------------------------------------------
// Item 3: log_softmax records its own stable Jacobian (unblocks x-entropy F112).
// ---------------------------------------------------------------------------

#[test]
fn log_softmax_backward_matches_finite_difference_f112() {
    let shape = [2usize, 3];
    let base = ramp(&shape).to_vec().unwrap();
    let weight = t32(vec![0.3, -1.2, 0.7, 2.0, -0.5, 1.1], shape.to_vec());
    let weight_for_loss = weight.clone();

    let loss = move |xx: &Tensor<f32>| -> f32 {
        let y = xx.log_softmax(1).expect("log_softmax");
        y.mul(&weight_for_loss)
            .expect("mul")
            .sum()
            .expect("sum")
            .item()
            .expect("item")
    };

    let x = t32(base.clone(), shape.to_vec()).requires_grad_(true);
    let y = x.log_softmax(1).expect("log_softmax");
    y.mul(&weight)
        .expect("mul")
        .sum()
        .expect("sum")
        .backward()
        .expect("backward (F112)");
    let analytic = x
        .grad()
        .expect("log_softmax input grad must exist (x-entropy was inf/None)");

    let numeric = fd_grad(&base, &shape, loss);
    assert_close(
        &analytic.to_vec().unwrap(),
        &numeric,
        2e-2,
        "log_softmax grad vs FD",
    );
}

// ---------------------------------------------------------------------------
// Item 4: map-based element-wise activations record a real derivative.
// ---------------------------------------------------------------------------

#[test]
fn exp_log_tanh_sigmoid_backward_are_recorded() {
    // exp: d/dx = exp(x)
    let xv = vec![-1.0f32, 0.0, 0.5, 1.5];
    let x = t32(xv.clone(), vec![4]).requires_grad_(true);
    x.exp().unwrap().sum().unwrap().backward().unwrap();
    let g = x.grad().expect("exp grad").to_vec().unwrap();
    let xe: Vec<f32> = xv.iter().map(|v| v.exp()).collect();
    assert_close(&g, &xe, 1e-4, "exp grad");

    // ln: d/dx = 1/x
    let x = t32(vec![0.5, 1.0, 2.0, 4.0], vec![4]).requires_grad_(true);
    x.log().unwrap().sum().unwrap().backward().unwrap();
    let g = x.grad().expect("log grad").to_vec().unwrap();
    assert_close(&g, &[2.0, 1.0, 0.5, 0.25], 1e-4, "log grad");

    // tanh: d/dx = 1 - tanh(x)^2
    let xv = vec![-0.8, 0.2, 1.0, 2.0];
    let x = t32(xv.clone(), vec![4]).requires_grad_(true);
    x.tanh().unwrap().sum().unwrap().backward().unwrap();
    let g = x.grad().expect("tanh grad").to_vec().unwrap();
    let te: Vec<f32> = xv.iter().map(|v| 1.0 - v.tanh() * v.tanh()).collect();
    assert_close(&g, &te, 1e-4, "tanh grad");

    // sigmoid: d/dx = s(1-s)
    let xv = vec![-1.0, 0.0, 0.7, 2.0];
    let x = t32(xv.clone(), vec![4]).requires_grad_(true);
    x.sigmoid().unwrap().sum().unwrap().backward().unwrap();
    let g = x.grad().expect("sigmoid grad").to_vec().unwrap();
    let se: Vec<f32> = xv
        .iter()
        .map(|v| {
            let s = 1.0 / (1.0 + (-v).exp());
            s * (1.0 - s)
        })
        .collect();
    assert_close(&g, &se, 1e-4, "sigmoid grad");
}

/// The SIMD (numel > 1000) and parallel (101..1000) dispatch paths of
/// sigmoid/relu must record gradients too, not only the sequential fallback.
#[test]
fn sigmoid_relu_backward_on_simd_and_parallel_paths() {
    for &n in &[1024usize, 256usize] {
        // sigmoid over a large tensor
        let xv: Vec<f32> = (0..n).map(|i| ((i % 7) as f32) - 3.0).collect();
        let x = t32(xv.clone(), vec![n]).requires_grad_(true);
        x.sigmoid().unwrap().sum().unwrap().backward().unwrap();
        let g = x
            .grad()
            .unwrap_or_else(|| panic!("sigmoid grad missing at n={n}"))
            .to_vec()
            .unwrap();
        for (i, &v) in xv.iter().enumerate() {
            let s = 1.0 / (1.0 + (-v).exp());
            assert!((g[i] - s * (1.0 - s)).abs() < 1e-4, "sigmoid n={n} i={i}");
        }

        // relu over a large tensor (avoid exact zero)
        let xv: Vec<f32> = (0..n).map(|i| ((i % 5) as f32) - 2.5).collect();
        let x = t32(xv.clone(), vec![n]).requires_grad_(true);
        x.relu().unwrap().sum().unwrap().backward().unwrap();
        let g = x
            .grad()
            .unwrap_or_else(|| panic!("relu grad missing at n={n}"))
            .to_vec()
            .unwrap();
        for (i, &v) in xv.iter().enumerate() {
            let expected = if v > 0.0 { 1.0 } else { 0.0 };
            assert!(
                (g[i] - expected).abs() < 1e-5,
                "relu n={n} i={i}: {} vs {expected}",
                g[i]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Item 5: contiguous() preserves the graph on the copy branch.
// ---------------------------------------------------------------------------

#[test]
fn contiguous_backward_flows_through_copy() {
    let x = ramp(&[2, 3]).requires_grad_(true);
    let xt = x.transpose_view(0, 1).expect("transpose_view"); // [3,2], non-contiguous
    assert!(!xt.is_contiguous());
    let xc = xt.contiguous().expect("contiguous"); // copy branch
    let seed = ramp(&[3, 2]);
    xc.backward_with_grad(Some(&seed))
        .expect("contiguous backward");

    // grad wrt x = transpose of seed back to [2,3]
    let seed_v = seed.to_vec().unwrap();
    let mut exp = vec![0.0f32; 6];
    for i in 0..3 {
        for j in 0..2 {
            // xt[i,j] = x[j,i]; seed[i,j] -> x[j,i]
            exp[j * 3 + i] = seed_v[i * 2 + j];
        }
    }
    let g = x
        .grad()
        .expect("contiguous input grad must exist (copy detached the graph)");
    assert_close(&g.to_vec().unwrap(), &exp, 1e-5, "contiguous grad");
}

// ---------------------------------------------------------------------------
// Item 6: from_vec rejects a length that does not match the shape.
// ---------------------------------------------------------------------------

#[test]
fn from_vec_rejects_length_mismatch() {
    // 8 values into [4,1,2,2] (numel 16) must be an Err, not silent acceptance.
    assert!(
        Tensor::<f32>::from_vec(vec![0.0; 8], &[4, 1, 2, 2]).is_err(),
        "Tensor::from_vec must reject len != numel"
    );
    assert!(
        torsh_tensor::creation::from_vec(vec![0.0f32; 8], &[4, 1, 2, 2], DeviceType::Cpu).is_err(),
        "creation::from_vec must reject len != numel"
    );
    // A correct length still succeeds.
    assert!(Tensor::<f32>::from_vec(vec![0.0; 16], &[4, 1, 2, 2]).is_ok());
}

// ---------------------------------------------------------------------------
// Item 7: copy_ / copy_from / set_data are copy-on-write safe.
// ---------------------------------------------------------------------------

#[test]
fn copy_is_cow_safe_shared_clone_survives() {
    let mut a = t32(vec![1.0, 2.0, 3.0], vec![3]);
    let snap = a.clone(); // shallow clone shares storage
    a.copy_(&t32(vec![9.0, 9.0, 9.0], vec![3])).expect("copy_");
    assert_eq!(
        snap.to_vec().unwrap(),
        vec![1.0, 2.0, 3.0],
        "copy_ must not clobber a shared snapshot clone"
    );
    assert_eq!(a.to_vec().unwrap(), vec![9.0, 9.0, 9.0]);
}

#[test]
fn copy_from_and_set_data_are_cow_safe() {
    let mut a = t32(vec![1.0, 2.0, 3.0], vec![3]);
    let snap = a.clone();
    a.copy_from(&t32(vec![7.0, 8.0, 9.0], vec![3]))
        .expect("copy_from");
    assert_eq!(
        snap.to_vec().unwrap(),
        vec![1.0, 2.0, 3.0],
        "copy_from clobbered clone"
    );
    assert_eq!(a.to_vec().unwrap(), vec![7.0, 8.0, 9.0]);

    let mut b = t32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let snap_b = b.clone();
    b.set_data(&[5.0, 6.0, 7.0, 8.0]).expect("set_data");
    assert_eq!(
        snap_b.to_vec().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0],
        "set_data clobbered clone"
    );
    assert_eq!(b.to_vec().unwrap(), vec![5.0, 6.0, 7.0, 8.0]);
    assert!(
        b.set_data(&[1.0]).is_err(),
        "set_data must reject length mismatch"
    );
}

/// copy_from / set_data into a strided/offset view must write *through* to the
/// base tensor (PyTorch in-place-on-a-view semantics), which is what scatter-
/// style aggregation relies on — not silently detach the view and drop the write.
#[test]
fn copy_from_and_set_data_write_through_views() {
    // base = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]; view = columns 1..3 (a [3,2] strided view).
    let base = ramp(&[3, 4]);
    let mut view = base.slice_tensor(1, 1, 3).expect("slice_tensor");
    assert!(view.is_view());
    // ramp = i*0.5 + 1.0, so base = [[1,1.5,2,2.5],[3,3.5,4,4.5],[5,5.5,6,6.5]]
    // and the [3,2] ramp = [[1,1.5],[2,2.5],[3,3.5]] lands in columns 1..3.
    view.copy_from(&ramp(&[3, 2]))
        .expect("copy_from writes through a view");
    assert_eq!(
        base.to_vec().unwrap(),
        vec![1.0, 1.0, 1.5, 2.5, 3.0, 2.0, 2.5, 4.5, 5.0, 3.0, 3.5, 6.5],
        "copy_from must write through the view into the base's addressed elements"
    );

    // set_data follows the same write-through path.
    let base2 = ramp(&[3, 4]);
    let mut view2 = base2.slice_tensor(1, 1, 3).expect("slice_tensor");
    view2
        .set_data(&[100.0; 6])
        .expect("set_data writes through a view");
    assert_eq!(
        base2.to_vec().unwrap(),
        vec![1.0, 100.0, 100.0, 2.5, 3.0, 100.0, 100.0, 4.5, 5.0, 100.0, 100.0, 6.5],
        "set_data must write through the view into the base"
    );
    // A length mismatch is still rejected.
    assert!(
        view2.set_data(&[1.0]).is_err(),
        "set_data still rejects length mismatch"
    );
}

/// A tensor concatenated with itself is two graph edges; its gradient must be
/// the sum of both slabs, not just one.
#[test]
fn cat_backward_accumulates_duplicate_input() {
    let a = ramp(&[2, 3]).requires_grad_(true);
    let out = Tensor::cat(&[&a, &a], 0).expect("cat"); // [4,3]
    let seed = ramp(&[4, 3]);
    out.backward_with_grad(Some(&seed))
        .expect("cat duplicate-input backward");
    let seed_v = seed.to_vec().unwrap();
    let mut exp = vec![0.0f32; 6];
    for r in 0..2 {
        for c in 0..3 {
            exp[r * 3 + c] = seed_v[r * 3 + c] + seed_v[(r + 2) * 3 + c];
        }
    }
    let g = a.grad().expect("duplicate-input grad").to_vec().unwrap();
    assert_close(&g, &exp, 1e-5, "cat duplicate-input grad accumulation");
}
