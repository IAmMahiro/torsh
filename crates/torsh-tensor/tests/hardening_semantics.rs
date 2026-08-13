//! Task W5-T hardening: tensor value semantics and view-op gradient recording.
//!
//! Two independent contracts are pinned here.
//!
//! **Value semantics for element/bulk writers.** `Tensor::clone()` shares
//! storage (it is a cheap handle copy, not PyTorch's deep `clone()`), so an
//! in-place write through one handle used to be visible through every other
//! handle. The in-place arithmetic ops (`add_`, `mul_scalar_`, …) and the bulk
//! copies (`copy_from`, `set_data`) already isolate a shared *base* tensor with
//! `make_unique()` before writing; the element writers (`set_item`,
//! `set_item_flat`) and the fill family (`fill_`, `zero_`, `ones_`) did not, so
//! `let mut c = orig.clone(); c.set_item(&[0], 99.0)` wrote straight through to
//! `orig`. These tests pin the same rule everywhere: a **base** tensor is
//! copy-on-write, a **view** keeps PyTorch's write-through aliasing.
//!
//! **View ops must record.** `t()`, `gather()` and `index_select()` all built
//! their output with `from_data`, which produces a detached leaf: the output
//! came back with `requires_grad == false` even from a `requires_grad` input,
//! so a loss routed through any of them silently lost its gradient path. Each
//! is now recorded (`ViewKind::Permute` for `t()`, `Operation::Gather` for the
//! two index ops) and checked against finite differences, including the
//! discriminating duplicate-index case where the gradient must *accumulate*.

use torsh_core::device::DeviceType;
use torsh_tensor::Tensor;

/// Serializes every test in this file that records or asserts recording.
///
/// Grad mode is a process-global `AtomicBool` (`torsh-core/src/grad_mode.rs`),
/// and `cargo test` runs a whole binary's tests in one process across a thread
/// pool (unlike `cargo nextest`'s process-per-test), so a `with_grad_mode`
/// window opened by one test transiently suppresses recording for every other
/// test's tensor ops. `view_ops_do_not_record_under_no_grad` below opens
/// exactly such a window; every test that asserts on `requires_grad` or calls
/// `backward()` takes this lock for its full body. See
/// `hardening_autograd_unary.rs` for the measured failure this prevents.
static GRAD_MODE_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn t32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
    Tensor::from_data(data, shape, DeviceType::Cpu).expect("f32 tensor creation should succeed")
}

fn idx(data: Vec<i64>, shape: Vec<usize>) -> Tensor<i64> {
    Tensor::from_data(data, shape, DeviceType::Cpu).expect("i64 tensor creation should succeed")
}

/// Central finite-difference gradient of a scalar loss w.r.t. every element of
/// `base` (which carries `shape`). Mirrors hardening_autograd_unary.rs.
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
        g[i] = (loss(&t32(plus, shape.to_vec())) - loss(&t32(minus, shape.to_vec()))) / (2.0 * eps);
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
// (1) Copy-on-write for element and bulk writers
// ---------------------------------------------------------------------------

#[test]
fn set_item_on_a_clone_leaves_the_original_untouched() {
    // TODO.md:1374's follow-up, verbatim: the clone shares storage, so before
    // this fix `orig` read back [99, 2, 3].
    let orig = t32(vec![1.0, 2.0, 3.0], vec![3]);
    let mut cl = orig.clone();
    cl.set_item(&[0], 99.0).expect("set_item should succeed");

    assert_eq!(
        cl.to_vec().expect("clone to_vec"),
        vec![99.0, 2.0, 3.0],
        "the write must land on the clone"
    );
    assert_eq!(
        orig.to_vec().expect("original to_vec"),
        vec![1.0, 2.0, 3.0],
        "a clone is value-semantic: writing through it must not reach the original"
    );
}

#[test]
fn set_item_flat_on_a_clone_leaves_the_original_untouched() {
    let orig = t32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let mut cl = orig.clone();
    cl.set_item_flat(3, -7.0).expect("set_item_flat");

    assert_eq!(cl.to_vec().expect("clone"), vec![1.0, 2.0, 3.0, -7.0]);
    assert_eq!(orig.to_vec().expect("orig"), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn fill_on_a_clone_leaves_the_original_untouched() {
    let orig = t32(vec![1.0, 2.0, 3.0], vec![3]);
    let mut cl = orig.clone();
    cl.fill_(0.5).expect("fill_");

    assert_eq!(cl.to_vec().expect("clone"), vec![0.5, 0.5, 0.5]);
    assert_eq!(
        orig.to_vec().expect("orig"),
        vec![1.0, 2.0, 3.0],
        "fill_ must be copy-on-write like every other in-place writer"
    );
}

#[test]
fn zero_and_ones_on_a_clone_leave_the_original_untouched() {
    let orig = t32(vec![4.0, 5.0], vec![2]);

    let mut zeroed = orig.clone();
    zeroed.zero_().expect("zero_");
    assert_eq!(zeroed.to_vec().expect("zeroed"), vec![0.0, 0.0]);
    assert_eq!(orig.to_vec().expect("orig after zero_"), vec![4.0, 5.0]);

    let mut oned = orig.clone();
    oned.ones_().expect("ones_");
    assert_eq!(oned.to_vec().expect("oned"), vec![1.0, 1.0]);
    assert_eq!(orig.to_vec().expect("orig after ones_"), vec![4.0, 5.0]);
}

#[test]
fn set_item_on_a_uniquely_owned_tensor_still_mutates_in_place() {
    // The copy-on-write step must not turn an ordinary mutation into a no-op:
    // with no other handle alive there is nothing to isolate from.
    let mut solo = t32(vec![1.0, 2.0, 3.0], vec![3]);
    solo.set_item(&[2], 8.0).expect("set_item");
    assert_eq!(solo.to_vec().expect("solo"), vec![1.0, 2.0, 8.0]);
}

#[test]
fn repeated_set_item_accumulates_on_the_same_buffer() {
    // A loop of element writes must keep landing on the *same* isolated buffer;
    // isolating on every call would drop every write but the last if the copy
    // were taken from the pre-write original each time.
    let orig = t32(vec![0.0; 4], vec![4]);
    let mut cl = orig.clone();
    for i in 0..4 {
        cl.set_item(&[i], (i + 1) as f32).expect("set_item");
    }
    assert_eq!(cl.to_vec().expect("clone"), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(orig.to_vec().expect("orig"), vec![0.0; 4]);
}

#[test]
fn set_item_through_a_view_writes_through_to_the_base() {
    // Regression pin for the semantics the earlier campaign deliberately
    // restored for MPNN scatter: a *view* aliases its base, and PyTorch writes
    // through it. This is green today and must stay green.
    let base = t32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let mut row = base
        .slice_tensor(0, 1, 2)
        .expect("slice_tensor should succeed");
    row.set_item(&[0, 2], -1.0).expect("set_item on a view");

    assert_eq!(
        base.to_vec().expect("base"),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, -1.0],
        "an in-place element write through a view must reach the base tensor"
    );
}

#[test]
fn set_item_through_a_transposed_view_is_stride_aware() {
    // A transposed view has non-default strides, so the write must be mapped
    // back through them rather than indexing the base buffer row-major.
    let base = t32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let mut tr = base.transpose(0, 1).expect("transpose view");
    assert_eq!(tr.shape().dims(), &[3, 2]);
    // tr[2, 0] is base[0, 2] -> flat slot 2.
    tr.set_item(&[2, 0], -9.0)
        .expect("set_item on a strided view");

    assert_eq!(
        base.to_vec().expect("base"),
        vec![1.0, 2.0, -9.0, 4.0, 5.0, 6.0],
        "the write must be routed through the view's strides"
    );
}

#[test]
fn fill_through_a_strided_view_writes_only_the_view_elements() {
    // `fill_` used to write `storage[0..numel]` directly, ignoring both the
    // strides and the storage offset: filling a 3-element row of a [2, 3] base
    // clobbered the base's first three elements instead of its second row.
    let base = t32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let mut row = base.slice_tensor(0, 1, 2).expect("slice_tensor");
    row.fill_(0.0).expect("fill_ on a view");

    assert_eq!(
        base.to_vec().expect("base"),
        vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
        "fill_ through a view must write the view's elements, not the base's prefix"
    );
}

#[test]
fn optimizer_style_update_keeps_the_snapshot_intact() {
    // torsh-optim's `param_update::assign` writes through `mul_scalar_` +
    // `add_`, both of which route through `make_unique`. The element writers
    // must agree with that: a retained `.clone()` snapshot (what optimizer
    // state and checkpointing hold) keeps its pre-step values.
    let param = t32(vec![1.0, 2.0, 3.0], vec![3]);
    let snapshot = param.clone();

    let mut updated = param.clone();
    for i in 0..3 {
        let old = updated.get_item(&[i]).expect("get_item");
        updated.set_item(&[i], old - 0.5).expect("set_item");
    }

    assert_eq!(updated.to_vec().expect("updated"), vec![0.5, 1.5, 2.5]);
    assert_eq!(
        snapshot.to_vec().expect("snapshot"),
        vec![1.0, 2.0, 3.0],
        "an optimizer snapshot must survive an element-wise parameter update"
    );
    assert_eq!(
        param.to_vec().expect("param"),
        vec![1.0, 2.0, 3.0],
        "the source handle must survive too"
    );
}

// ---------------------------------------------------------------------------
// (2) t() / gather() / index_select() record for autograd
// ---------------------------------------------------------------------------

#[test]
fn t_records_and_matches_transpose() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    let xv = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = t32(xv.clone(), vec![2, 3]).requires_grad_(true);

    let via_t = x.t().expect("t() should succeed");
    let via_transpose = x.transpose(0, 1).expect("transpose should succeed");

    assert_eq!(via_t.shape().dims(), &[3, 2]);
    assert_eq!(
        via_t.to_vec().expect("t values"),
        via_transpose.to_vec().expect("transpose values"),
        "t() must produce exactly what transpose(0, 1) produces"
    );
    assert!(
        via_t.requires_grad(),
        "t() must propagate requires_grad (gatekeeper measured false)"
    );
}

#[test]
fn t_of_a_strided_view_matches_the_packed_transpose() {
    // The hardest layout `t()` now produces: a transpose *of a slice view*, so
    // the result carries both swapped strides and a non-zero storage offset.
    // torsh-graph's `decode_edges` builds exactly this
    // (`h.slice_tensor(0, j, j + 1)?.t()?`), so every consumer that reads it —
    // `to_vec`, broadcasting `mul`, `dot`, `matmul` — has to agree with the
    // packed tensor it used to get.
    let base = t32(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![3, 3],
    );
    let row = base.slice_tensor(0, 2, 3).expect("slice_tensor");

    let strided = row.t().expect("t of a slice view");
    let packed = strided.contiguous().expect("contiguous");
    assert_eq!(strided.shape().dims(), &[3, 1]);
    assert_close(
        &strided.to_vec().expect("strided"),
        &[7.0, 8.0, 9.0],
        1e-6,
        "t() of the third row",
    );
    assert_close(
        &strided.to_vec().expect("strided"),
        &packed.to_vec().expect("packed"),
        1e-6,
        "strided vs packed transpose",
    );

    // `dot` broadcasts [1, 3] × [3, 1] into the outer product and sums it, so
    // both layouts must land on the same scalar.
    let other = base.slice_tensor(0, 0, 1).expect("slice_tensor");
    let via_strided = other
        .dot(&strided)
        .expect("dot with a strided transpose")
        .item()
        .expect("item");
    let via_packed = other
        .dot(&packed)
        .expect("dot with a packed transpose")
        .item()
        .expect("item");
    assert_close(
        &[via_strided],
        &[via_packed],
        1e-5,
        "dot must not depend on the transpose's layout",
    );

    // …and the same for matmul: [1, 3] · [3, 1] = 1*7 + 2*8 + 3*9 = 50.
    let product = other
        .matmul(&strided)
        .expect("matmul with a strided transpose")
        .to_vec()
        .expect("to_vec");
    assert_close(&product, &[50.0], 1e-5, "[1,3] · [3,1]");
}

#[test]
fn t_output_feeds_matmul_correctly() {
    // `t()` used to hand matmul a freshly packed buffer; it now hands over a
    // strided view, which is the one behavioural change with reach outside this
    // crate (every `.t()?` call site in the workspace feeds matmul or dot).
    // Pin the product against the hand-computed reference so a stride-blind
    // consumer cannot regress silently.
    let a = t32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    // a · aᵀ = [[14, 32], [32, 77]]
    let gram = a
        .matmul(&a.t().expect("t"))
        .expect("matmul with a transposed operand");
    assert_eq!(gram.shape().dims(), &[2, 2]);
    assert_close(
        &gram.to_vec().expect("gram"),
        &[14.0, 32.0, 32.0, 77.0],
        1e-5,
        "a · aᵀ",
    );

    // aᵀ · a = [[17, 22, 27], [22, 29, 36], [27, 36, 45]]
    let cov = a
        .t()
        .expect("t")
        .matmul(&a)
        .expect("matmul from a transposed operand");
    assert_eq!(cov.shape().dims(), &[3, 3]);
    assert_close(
        &cov.to_vec().expect("cov"),
        &[17.0, 22.0, 27.0, 22.0, 29.0, 36.0, 27.0, 36.0, 45.0],
        1e-5,
        "aᵀ · a",
    );

    // (aᵀ)ᵀ round-trips to the original values.
    assert_close(
        &a.t()
            .expect("t")
            .t()
            .expect("t twice")
            .to_vec()
            .expect("round trip"),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        1e-6,
        "(aᵀ)ᵀ",
    );
}

#[test]
fn t_backward_matches_finite_difference() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    // Weight the transposed entries so the gradient is a genuine permutation
    // rather than the all-ones tensor `sum()` alone would produce.
    let xv = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let weights = t32(vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0], vec![3, 2]);
    let loss = |t: &Tensor<f32>| -> f32 {
        t.t()
            .expect("t")
            .mul(&weights)
            .expect("mul")
            .sum()
            .expect("sum")
            .item()
            .expect("item")
    };

    let x = t32(xv.clone(), vec![2, 3]).requires_grad_(true);
    let y = x
        .t()
        .expect("t")
        .mul(&weights)
        .expect("mul")
        .sum()
        .expect("sum");
    y.backward().expect("t() backward must be recorded");

    let analytic = x.grad().expect("t grad").to_vec().expect("grad to_vec");
    let numeric = fd_grad(&xv, &[2, 3], loss);
    assert_close(&analytic, &numeric, 2e-2, "t() grad vs FD");
}

#[test]
fn gather_records_and_backward_matches_finite_difference() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    let xv = vec![1.0f32, 2.0, 3.0, 4.0];
    let indices = idx(vec![2, 0, 3], vec![3]);
    let weights = t32(vec![1.0, 2.0, 5.0], vec![3]);
    let loss = |t: &Tensor<f32>| -> f32 {
        t.gather(0, &indices)
            .expect("gather")
            .mul(&weights)
            .expect("mul")
            .sum()
            .expect("sum")
            .item()
            .expect("item")
    };

    let x = t32(xv.clone(), vec![4]).requires_grad_(true);
    let g = x.gather(0, &indices).expect("gather");
    assert!(
        g.requires_grad(),
        "gather must propagate requires_grad (gatekeeper measured false)"
    );
    g.mul(&weights)
        .expect("mul")
        .sum()
        .expect("sum")
        .backward()
        .expect("gather backward must be recorded");

    let analytic = x.grad().expect("gather grad").to_vec().expect("to_vec");
    let numeric = fd_grad(&xv, &[4], loss);
    assert_close(&analytic, &numeric, 2e-2, "gather grad vs FD");
}

#[test]
fn gather_with_duplicate_indices_accumulates_gradient() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    // The discriminating case: source element 1 is read three times, so its
    // gradient is the *sum* of the three output gradients (1 + 2 + 4 = 7), and
    // element 2 is never read at all.
    let indices = idx(vec![1, 1, 0, 1], vec![4]);
    let weights = t32(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let x = t32(vec![5.0f32, 6.0, 7.0], vec![3]).requires_grad_(true);

    x.gather(0, &indices)
        .expect("gather")
        .mul(&weights)
        .expect("mul")
        .sum()
        .expect("sum")
        .backward()
        .expect("gather backward");

    assert_close(
        &x.grad().expect("grad").to_vec().expect("to_vec"),
        &[3.0, 7.0, 0.0],
        1e-6,
        "gather gradient must accumulate on duplicated source elements",
    );
}

#[test]
fn gather_2d_backward_matches_finite_difference() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    // dim=1 on a [2, 3] source with a duplicate inside each row.
    let xv = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let indices = idx(vec![2, 2, 0, 1], vec![2, 2]);
    let weights = t32(vec![1.0, 3.0, 5.0, 7.0], vec![2, 2]);
    let loss = |t: &Tensor<f32>| -> f32 {
        t.gather(1, &indices)
            .expect("gather")
            .mul(&weights)
            .expect("mul")
            .sum()
            .expect("sum")
            .item()
            .expect("item")
    };

    let x = t32(xv.clone(), vec![2, 3]).requires_grad_(true);
    x.gather(1, &indices)
        .expect("gather")
        .mul(&weights)
        .expect("mul")
        .sum()
        .expect("sum")
        .backward()
        .expect("gather backward");

    let analytic = x.grad().expect("grad").to_vec().expect("to_vec");
    let numeric = fd_grad(&xv, &[2, 3], loss);
    assert_close(&analytic, &numeric, 2e-2, "gather(dim=1) grad vs FD");
}

#[test]
fn index_select_records_and_backward_matches_finite_difference() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    let xv = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let sel = idx(vec![2, 0], vec![2]);
    let weights = t32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let loss = |t: &Tensor<f32>| -> f32 {
        t.index_select(0, &sel)
            .expect("index_select")
            .mul(&weights)
            .expect("mul")
            .sum()
            .expect("sum")
            .item()
            .expect("item")
    };

    let x = t32(xv.clone(), vec![3, 2]).requires_grad_(true);
    let s = x.index_select(0, &sel).expect("index_select");
    assert!(
        s.requires_grad(),
        "index_select must propagate requires_grad (gatekeeper measured false)"
    );
    s.mul(&weights)
        .expect("mul")
        .sum()
        .expect("sum")
        .backward()
        .expect("index_select backward must be recorded");

    let analytic = x.grad().expect("grad").to_vec().expect("to_vec");
    let numeric = fd_grad(&xv, &[3, 2], loss);
    assert_close(&analytic, &numeric, 2e-2, "index_select grad vs FD");
}

#[test]
fn index_select_with_duplicate_indices_accumulates_gradient() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    // Row 0 is selected twice; its gradient must be the sum of both output
    // rows' gradients, and row 2 (never selected) must stay zero.
    let sel = idx(vec![0, 1, 0], vec![3]);
    let weights = t32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let x = t32(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).requires_grad_(true);

    x.index_select(0, &sel)
        .expect("index_select")
        .mul(&weights)
        .expect("mul")
        .sum()
        .expect("sum")
        .backward()
        .expect("index_select backward");

    assert_close(
        &x.grad().expect("grad").to_vec().expect("to_vec"),
        &[6.0, 8.0, 3.0, 4.0, 0.0, 0.0],
        1e-6,
        "index_select gradient must accumulate on duplicated rows",
    );
}

#[test]
fn index_select_on_dim1_backward_matches_finite_difference() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    let xv = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let sel = idx(vec![2, 1, 2], vec![3]);
    let weights = t32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let loss = |t: &Tensor<f32>| -> f32 {
        t.index_select(1, &sel)
            .expect("index_select")
            .mul(&weights)
            .expect("mul")
            .sum()
            .expect("sum")
            .item()
            .expect("item")
    };

    let x = t32(xv.clone(), vec![2, 3]).requires_grad_(true);
    x.index_select(1, &sel)
        .expect("index_select")
        .mul(&weights)
        .expect("mul")
        .sum()
        .expect("sum")
        .backward()
        .expect("index_select backward");

    let analytic = x.grad().expect("grad").to_vec().expect("to_vec");
    let numeric = fd_grad(&xv, &[2, 3], loss);
    assert_close(&analytic, &numeric, 2e-2, "index_select(dim=1) grad vs FD");
}

#[test]
fn view_ops_do_not_record_under_no_grad() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    let x = t32(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).requires_grad_(true);
    let indices = idx(vec![1, 0], vec![2]);

    torsh_core::grad_mode::with_grad_mode(false, || {
        assert!(
            !x.t().expect("t").requires_grad(),
            "t() must not record inside a no_grad scope"
        );
        assert!(
            !x.gather(1, &idx(vec![0, 1], vec![2, 1]))
                .expect("gather")
                .requires_grad(),
            "gather must not record inside a no_grad scope"
        );
        assert!(
            !x.index_select(0, &indices)
                .expect("index_select")
                .requires_grad(),
            "index_select must not record inside a no_grad scope"
        );
    });

    // …and recording resumes once the scope closes.
    assert!(x.t().expect("t").requires_grad());
}

// ---------------------------------------------------------------------------
// (4) `narrow()` is an aliasing view, like PyTorch's
// ---------------------------------------------------------------------------
//
// Task W6-T. `narrow()` used to route through `index()`, which *gathers* the
// selected elements into a fresh buffer: the result reported `is_view() ==
// false`, carried no base handle, and a write through it was invisible to the
// source — while `slice_tensor()`, the very same single-axis contiguous range,
// returned a write-through alias. Two incompatible senses of "view" in one
// crate, and the copy was paid on every call (RNN gate splitting pays it per
// gate per timestep). `narrow()` now builds exactly the view `slice_tensor()`
// builds; only its argument handling (negative `dim`/`start`, `length` instead
// of `end`) and its error strings stay its own.

#[test]
fn narrow_shares_storage_with_its_base() {
    let base = t32((1..=12).map(|v| v as f32).collect(), vec![3, 4]);
    let rows = base.narrow(0, 1, 2).expect("narrow");

    assert!(
        rows.is_view(),
        "narrow must address its source through view metadata, not copy it"
    );
    assert!(
        rows.base_tensor().is_some(),
        "narrow must retain a handle to the tensor it is a view of"
    );
    assert!(
        base.shares_storage(&rows),
        "narrow must alias the source buffer instead of allocating a new slab"
    );
    assert_eq!(rows.shape().dims(), &[2, 4]);
    assert_eq!(
        rows.to_vec().expect("to_vec"),
        vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        "the view must still read the narrowed elements"
    );
}

#[test]
fn narrow_aliases_its_base() {
    // PyTorch parity: `t.narrow(0, 1, 2)[0, 0] = 99` is visible in `t`.
    let base = t32((1..=12).map(|v| v as f32).collect(), vec![3, 4]);
    let mut rows = base.narrow(0, 1, 2).expect("narrow");
    rows.set_item_flat(0, 99.0)
        .expect("set_item_flat on narrow");

    assert_eq!(
        base.to_vec().expect("base"),
        vec![1.0, 2.0, 3.0, 4.0, 99.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        "a write through a narrow must land in the base tensor"
    );
}

#[test]
fn narrow_on_a_middle_axis_writes_through_its_strides() {
    // Narrowing a non-leading axis yields a *strided* window: the store must be
    // mapped back through the view's strides, not indexed row-major.
    let base = t32((1..=12).map(|v| v as f32).collect(), vec![3, 4]);
    let mut cols = base.narrow(1, 1, 2).expect("narrow");
    assert_eq!(cols.shape().dims(), &[3, 2]);
    assert_eq!(
        cols.to_vec().expect("to_vec"),
        vec![2.0, 3.0, 6.0, 7.0, 10.0, 11.0]
    );
    // view element 3 is (row 1, col 2) -> base flat slot 6.
    cols.set_item_flat(3, -1.0).expect("set_item_flat");

    assert_eq!(
        base.to_vec().expect("base"),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        "a strided narrow must write through its own strides"
    );
}

#[test]
fn narrow_copy_into_a_padded_slab_writes_through() {
    // The pattern `torsh-vision`'s `pad_tensor` uses (advanced_transforms.rs):
    // narrow the destination to the source's extent and `copy_` into it. With
    // narrow-as-copy this wrote into a detached slab and the padded tensor
    // stayed all zeros — a silent no-op.
    let padded = t32(vec![0.0; 12], vec![3, 4]);
    let source = t32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let mut window = padded
        .narrow(0, 0, 2)
        .expect("narrow rows")
        .narrow(1, 0, 3)
        .expect("narrow cols");
    window.copy_(&source).expect("copy_ into the padded window");

    assert_eq!(
        padded.to_vec().expect("padded"),
        vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "copying into a narrow of a zero slab must fill the top-left corner"
    );
}

#[test]
fn narrow_of_narrow_composes() {
    let base = t32((1..=12).map(|v| v as f32).collect(), vec![3, 4]);
    let inner = base
        .narrow(0, 1, 2)
        .expect("narrow rows")
        .narrow(1, 2, 2)
        .expect("narrow cols");

    assert_eq!(inner.shape().dims(), &[2, 2]);
    assert_eq!(
        inner.to_vec().expect("to_vec"),
        vec![7.0, 8.0, 11.0, 12.0],
        "a narrow of a narrow must compose offsets, not re-index from zero"
    );
    assert!(
        base.shares_storage(&inner),
        "a view of a view must still alias the original buffer"
    );
    let recorded_base = inner.base_tensor().expect("base handle");
    assert_eq!(
        recorded_base.shape().dims(),
        &[3, 4],
        "a view of a view must point at the original base, not the intermediate"
    );
}

#[test]
fn narrow_matches_slice_tensor_for_the_same_range() {
    // The delegation contract: for a single-axis contiguous range the two APIs
    // are the same view, down to the layout metadata.
    let base = t32((1..=12).map(|v| v as f32).collect(), vec![3, 4]);
    for (dim, start, len) in [(0usize, 1usize, 2usize), (1, 1, 2), (0, 0, 3), (1, 3, 1)] {
        let narrowed = base.narrow(dim as i32, start as i64, len).expect("narrow");
        let sliced = base
            .slice_tensor(dim, start, start + len)
            .expect("slice_tensor");
        assert_eq!(
            narrowed.shape().dims(),
            sliced.shape().dims(),
            "shape mismatch for ({dim}, {start}, {len})"
        );
        assert_eq!(
            narrowed.to_vec().expect("narrow to_vec"),
            sliced.to_vec().expect("slice to_vec"),
            "values mismatch for ({dim}, {start}, {len})"
        );
        assert_eq!(
            narrowed.strides(),
            sliced.strides(),
            "strides mismatch for ({dim}, {start}, {len})"
        );
        assert_eq!(
            narrowed.is_view(),
            sliced.is_view(),
            "view flag mismatch for ({dim}, {start}, {len})"
        );
        assert!(
            base.shares_storage(&narrowed),
            "narrow({dim}, {start}, {len}) must alias the base like slice_tensor does"
        );
    }
}

#[test]
fn narrow_reads_the_same_values_as_the_gather_route() {
    // Value-level regression guard: aliasing must not disturb what `narrow`
    // reads, on either a leading or a trailing axis, and with a negative start.
    let base = t32((1..=12).map(|v| v as f32).collect(), vec![3, 4]);
    assert_eq!(
        base.narrow(0, 1, 2).expect("narrow").to_vec().expect("v"),
        vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    );
    assert_eq!(
        base.narrow(1, 0, 2).expect("narrow").to_vec().expect("v"),
        vec![1.0, 2.0, 5.0, 6.0, 9.0, 10.0]
    );
    assert_eq!(
        base.narrow(-1, -2, 2).expect("narrow").to_vec().expect("v"),
        vec![3.0, 4.0, 7.0, 8.0, 11.0, 12.0],
        "negative dim and negative start must still fold to the same window"
    );
}

#[test]
fn narrow_zero_length_keeps_an_empty_result() {
    // Pinned as it behaves today: `slice_tensor` rejects an empty range, so the
    // delegation must not start erroring here.
    let base = t32((1..=12).map(|v| v as f32).collect(), vec![3, 4]);
    let empty_rows = base.narrow(0, 1, 0).expect("zero-length narrow on dim 0");
    assert_eq!(empty_rows.shape().dims(), &[0, 4]);
    assert_eq!(empty_rows.numel(), 0);
    assert!(empty_rows.to_vec().expect("to_vec").is_empty());

    let empty_cols = base.narrow(1, 0, 0).expect("zero-length narrow on dim 1");
    assert_eq!(empty_cols.shape().dims(), &[3, 0]);
    assert!(empty_cols.to_vec().expect("to_vec").is_empty());
}

#[test]
fn narrow_error_contract_is_unchanged() {
    let base = t32((1..=12).map(|v| v as f32).collect(), vec![3, 4]);
    let start_oob = base
        .narrow(1, 5, 1)
        .expect_err("start past the axis must fail");
    assert!(
        start_oob
            .to_string()
            .contains("Start index 5 out of range for dimension 1 with size 4"),
        "unexpected start-out-of-range message: {start_oob}"
    );
    // `start == dim_size` is rejected even for a zero-length window (unlike
    // PyTorch); pinned so the delegation does not quietly widen the contract.
    let start_at_end = base
        .narrow(0, 3, 0)
        .expect_err("start == dim_size must fail");
    assert!(
        start_at_end
            .to_string()
            .contains("Start index 3 out of range for dimension 0 with size 3"),
        "unexpected start-at-end message: {start_at_end}"
    );
    let end_oob = base
        .narrow(1, 2, 5)
        .expect_err("end past the axis must fail");
    assert!(
        end_oob
            .to_string()
            .contains("End index 7 out of range for dimension 1 with size 4"),
        "unexpected end-out-of-range message: {end_oob}"
    );
    let dim_oob = base
        .narrow(7, 0, 1)
        .expect_err("dim past the rank must fail");
    assert!(
        dim_oob
            .to_string()
            .contains("Dimension 7 out of range for tensor with 2 dimensions"),
        "unexpected dim-out-of-range message: {dim_oob}"
    );
}

#[test]
fn narrow_backward_still_matches_finite_differences() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    let xv: Vec<f32> = (1..=12).map(|v| v as f32 * 0.25).collect();
    let weights = t32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let loss = |t: &Tensor<f32>| -> f32 {
        t.narrow(1, 1, 2)
            .expect("narrow")
            .mul(&weights)
            .expect("mul")
            .sum()
            .expect("sum")
            .item()
            .expect("item")
    };

    let x = t32(xv.clone(), vec![3, 4]).requires_grad_(true);
    let out = x.narrow(1, 1, 2).expect("narrow");
    assert!(
        out.requires_grad(),
        "narrow of a requires_grad tensor must stay in the graph"
    );
    out.mul(&weights)
        .expect("mul")
        .sum()
        .expect("sum")
        .backward()
        .expect("narrow backward");

    let analytic = x.grad().expect("grad").to_vec().expect("to_vec");
    let numeric = fd_grad(&xv, &[3, 4], loss);
    assert_close(&analytic, &numeric, 2e-2, "narrow(dim=1) grad vs FD");
}

#[test]
fn narrow_backward_is_identical_to_slice_tensor_backward() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    let xv: Vec<f32> = (1..=12).map(|v| v as f32 * 0.5).collect();
    let weights = t32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);

    let via_narrow = t32(xv.clone(), vec![3, 4]).requires_grad_(true);
    via_narrow
        .narrow(1, 1, 2)
        .expect("narrow")
        .mul(&weights)
        .expect("mul")
        .sum()
        .expect("sum")
        .backward()
        .expect("narrow backward");

    let via_slice = t32(xv, vec![3, 4]).requires_grad_(true);
    via_slice
        .slice_tensor(1, 1, 3)
        .expect("slice_tensor")
        .mul(&weights)
        .expect("mul")
        .sum()
        .expect("sum")
        .backward()
        .expect("slice_tensor backward");

    assert_eq!(
        via_narrow.grad().expect("narrow grad").to_vec().expect("v"),
        via_slice.grad().expect("slice grad").to_vec().expect("v"),
        "the two spellings of one slab must scatter bit-identically"
    );
}

#[test]
fn narrow_under_no_grad_stays_detached() {
    let _serial = GRAD_MODE_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    let x = t32((1..=12).map(|v| v as f32).collect(), vec![3, 4]).requires_grad_(true);
    torsh_core::grad_mode::with_grad_mode(false, || {
        assert!(
            !x.narrow(0, 1, 2).expect("narrow").requires_grad(),
            "narrow must not record inside a no_grad scope"
        );
    });
    assert!(x.narrow(0, 1, 2).expect("narrow").requires_grad());
}

/// A device-resident `narrow` must behave exactly like a device-resident
/// `slice_tensor`: the view shares the device allocation and reads go through
/// the storage's host cache (one download), rather than declining.
#[cfg(feature = "gpu")]
#[test]
fn narrow_on_device_storage_matches_slice_tensor() {
    use oxicuda_backend::{ComputeBackend, CpuBackend};
    use std::sync::Arc;

    let mut backend = CpuBackend::new();
    backend.init().expect("backend init");
    torsh_tensor::gpu_dispatch::install_backend(Arc::new(backend) as Arc<dyn ComputeBackend>);

    let base = t32((1..=12).map(|v| v as f32).collect(), vec![3, 4])
        .to_device(DeviceType::Cuda(0))
        .expect("upload to the device");
    assert_eq!(
        base.storage_type(),
        "device",
        "precondition: the source must be device-resident"
    );

    let narrowed = base.narrow(0, 1, 2).expect("narrow on device storage");
    let sliced = base.slice_tensor(0, 1, 3).expect("slice_tensor on device");

    let narrowed_values = narrowed.to_vec().expect("narrow to_vec");
    let sliced_values = sliced.to_vec().expect("slice to_vec");
    let (narrow_storage, slice_storage) = (narrowed.storage_type(), sliced.storage_type());
    let (narrow_is_view, slice_is_view) = (narrowed.is_view(), sliced.is_view());
    let narrow_shares = base.shares_storage(&narrowed);
    let slice_shares = base.shares_storage(&sliced);
    drop(narrowed);
    drop(sliced);
    drop(base);
    torsh_tensor::gpu_dispatch::clear_backend();

    assert_eq!(narrowed_values, sliced_values);
    assert_eq!(
        narrowed_values,
        vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    );
    assert_eq!(narrow_storage, slice_storage, "storage kind must match");
    assert_eq!(narrow_is_view, slice_is_view, "view flag must match");
    assert_eq!(
        narrow_shares, slice_shares,
        "storage sharing must match slice_tensor's"
    );
    assert!(
        narrow_shares,
        "a device-resident narrow must alias the device allocation"
    );
}
