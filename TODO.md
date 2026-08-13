# ToRSh Development Roadmap

**Status**: v0.2.1 (In Development)

---

## 🛡️ 0.2.0 Production-Hardening Campaign (2026-08-12, IN PROGRESS)

A 13-agent exhaustive audit (stubs / bugs / missing features / perf / dependency purity)
produced **318 verified-by-reading findings (80 critical / 103 high / 110 medium / 25 low)**.
Implementation runs in parallel subagent waves; every bug fix follows a **verify-first
protocol** (write a failing reproduction test before touching production code; findings whose
repro passes are recorded INVALID and left untouched).

**Baseline at campaign start (all green)**: `cargo check --workspace --all-targets`,
`cargo clippy --all-targets -- -D warnings`, `cargo nextest run --workspace` (9,980 pass / 74 skip).
Two pre-existing breakages already fixed inline: oxiarc 0.4.0/0.4.1 mixed-lock E0599 (cargo
update of oxiarc-{lzhuf,brotli,bzip2,lzma,snappy}); torsh-jit E0275 inference overflow
(explicit HashMap type annotations in `analysis.rs::analyze_dependencies`).

### Headline findings being fixed
- **torsh-tensor**: dim-ignoring cumsum/sort/argmin/argmax/sum_dim/var/std; mul/div drop the
  autograd graph; matmul/conv backward holes; naive ijk matmul (no BLAS); hard-coded RNG
  seed 42; f16/bf16 randn transmute garbage; in-place-on-view stride corruption; SimdOptimized
  storage rejecting mutation (breaks set/set_slice/in-place ≥10KB); dangling-Weak view design;
  13k lines of dead `src/ops/**`; buffer pool never recycles.
- **torsh-nn**: Parameter never sets requires_grad; Dropout is identity; softmax ignores dim;
  BatchNorm wrong reshape + running stats never updated; conv_transpose returns zeros;
  cross-attention ignores k/v; LSTM/GRU ignore num_layers.
- **Security**: tar/zip-slip in torsh-hub + torsh-package extraction; hub downloads never
  integrity-checked; fake signature verification (string compares); no-op ModelSandbox.
- **Fabricated data**: CLI train/quantize fabricate losses/accuracy; profiler/bottleneck/
  benchmark report hardcoded metrics; eig/svd return fabricated results beyond 5×5;
  distributed collectives no-op with fake success (DDP silently scales grads by 1/N).
- **Python bindings**: `import rstorch` fails (uint16 import, functional module, sys.modules
  registration); root pyproject.toml cannot build.
- **Deps/purity**: default build compiles BoringSSL (reqwest), Oniguruma/esaxx C++ (tokenizers),
  blake3 asm (scirs2-datasets); banned rustfft via unused imageproc; optirs/tonic/prost/sprs +
  ~20 deps declared-but-unused; no deny.toml. **NEW POLICY: minimize non-COOLJAPAN deps.**

### Wave plan
- [x] **Wave 0**: build blockers (oxiarc lock mix, torsh-jit E0275) — done inline.
- [x] **Wave 1 (done 2026-08-12)**: 10/10 agents green. torsh-core honesty (fabricated GPU specs
  → real sysctl/honest Err, 21 phantom cfgs removed), torsh-tensor foundation (dim-aware
  cumsum/sort/argmin/argmax/sum_dim/var/std; BLAS matmul via oxiblas; mul/div + broadcast
  Add/Sub + ND-matmul backward implemented with finite-difference proofs; conv backward now
  honest error; visited-set toposort backward; entropy-seeded RNG + manual_seed; f16/bf16
  randn fixed; mmap per-tensor temp files; SimdOptimized CoW mutation; strong-Arc view design;
  stride-aware element access), torsh-data (samplers/shuffle/set_epoch/worker ordering; real
  VideoFolder/IMDB loaders), metrics/cluster formulas, hub/package tar-zip-slip + integrity
  wiring, profiler/utils fabrication removal (real measurements). Workspace: check+clippy
  green; 10,242 tests with 5 expected reds folded into Wave 2 (no_grad wiring, 2 test
  tolerances, 2 latent log-instability bugs exposed by real RNG).
- [ ] **Wave 2 (running)**: fixup(tensor crosscut)→autograd(no_grad via torsh-core grad-mode)
  chain, torsh-nn (parameter/dropout/norm/attention/RNN), torsh-optim (amsgrad, F040
  param-replacement design), torsh-autograd (clip_grad_norm, no_grad wiring, checkpointing),
  torsh-functional (losses/attention), torsh-linalg (real eig/svd via scirs2/oxiblas),
  torsh-signal (real filter design), torsh-text (BPE loop, tokenizers purity), torsh-vision,
  torsh-quantization (scale math), torsh-fx/jit (graph rewrites), torsh-graph (Result forwards).
- [x] **Wave 3 (done 2026-08-12)**: 7/7 agents green. torsh-tensor dead-code deletion (**36,476
  lines**: src/ops/**, .bak/.bak2-6, lib_new/ops_legacy/lazy_ops; normal_/multinomial ported
  live) + autograd completion (cat/stack/narrow/select/slice/log_softmax/unary exp·ln·tanh·
  sigmoid·relu now record backward; from_vec length validation; CoW-safe copy_from/set_data).
  torsh-distributed (real TCP process-group backend; every fake-success collective → honest
  Err; MockBackend confined to cfg(test)). torsh-backend Phase 4 (**73,329 lines** of legacy
  CUDA FFI deleted, build.rs cuda_available removed → host-independent API, real GPU path is
  torsh-tensor oxicuda; ConvolutionOps trait signature changed — BREAKING). torsh-python
  import fixes (uint16/functional/sys.modules/submodules; maturin+pytest green; 27 orphan
  files deleted; torsh-ffi split DEFERRED — needs root [workspace] surgery). torsh-cli real
  training loop + real .pt reader (replaced RNG-fabricated losses/quantize accuracy).
  torsh-models/hub (Ed25519 signing, real checksums, model-zoo orphan wiring, NaN-safe sorts).
  Post-wave: 10 cross-crate regressions fixed inline — copy_from view write-through restored
  (PyTorch in-place-on-view semantics; MPNN scatter) + 2 latent test-data bugs exposed by the
  new from_vec validation. Workspace check+clippy+nextest green.
- [x] **Wave 4 (done 2026-08-12)**: 3/3 agents green. **deny.toml created** → `cargo deny check
  bans` OK; dead deps removed (cust/cuda-sys/cudnn-sys workspace decls, optirs/optirs-core,
  protobuf, sprs, fs2, dead rand); lzma-rs → oxiarc-lzma (roundtrip test); **rustfft/ring/
  onig-sys/openblas all removed from the default graph** (imageproc default-features=false,
  tokenizers already pure, text/vision pretrained opt-in); MSRV 1.77 → 1.87; ~35 crate-local
  version pins hoisted to [workspace.dependencies] (zero tree drift). Lock-poison: new
  `torsh_core::sync` recovery helpers + **843 poison-expect sites** converted across
  core/tensor/autograd/backend. Docs: README purity/crate-list/test-count/roadmap truth pass,
  CHANGELOG 0.2.0 section, docs/ version bump, npm-publish.yml added, per-crate README fixes.
- [x] **Wave 5 (done 2026-08-12)**: **aws-lc-sys / aws-lc-rs / ring removed from the default
  build** — reqwest switched to `rustls-no-provider` + pure-Rust `oxitls-rustcrypto-provider`
  + `webpki-roots` (real cert verification, no weakening); 15 client-build sites rewired in
  cli/hub/utils via new `tls.rs` helpers; verified by a genuine HTTPS handshake to
  huggingface.co. 3 more fabrication sites (mock_hf_model / fake mirror health / dummy model
  bytes) → honest Err. torsh-models `download` feature given the same pure-Rust TLS wiring so
  its opt-in client cannot panic on the absent default provider.
- [x] **Gatekeeper review (done)**: independent Opus verification of the critical-fix set —
  RNG entropy, matmul (ndarray/matrixmultiply SIMD GEMM), autograd backward (finite-diff),
  archive-slip sanitizer *proven called* + real Ed25519, distributed honest-Err + real TCP —
  all VERIFIED. One miss caught & fixed post-review: torsh-profiler `ml_analysis.rs`
  parallel-analysis returned fabricated efficiency constants + seed(42) → now computed from the
  real event timeline (interval-union CPU util, per-thread load balance, `Option` memory-eff
  when no bytes data) with deterministic stratified sampling. Also deleted 7 dead
  `*_original_*lines.rs`/`.bak`/`.backup` files and de-mock'd isend/irecv comments.
- [x] **Final verification**: workspace check --all-targets/--all-features + clippy -D warnings
  + nextest + `cargo deny check bans` green; default-graph free of rustfft/ring/onig/openblas/
  aws-lc-sys.

### Final verification snapshot (campaign complete, 2026-08-12)
`cargo check --workspace --all-targets` ✓ · `cargo check --workspace --all-features --all-targets` ✓ ·
`cargo clippy --workspace --all-targets -- -D warnings` ✓ · `cargo nextest run --workspace`
**10,638 passed / 98 skipped / 0 failed** ✓ · `cargo test --workspace --doc` **all pass** ✓ ·
feature-gated suites (quant-experimental 299 / sparse-matlab 277 / data-privacy+audio 452 /
distributed-nccl 377 / tensor-hdf5 748 / models-download 268) all ✓ ·
`cargo deny check bans` **ok** ✓ · default-graph `cargo tree -i {aws-lc-sys, aws-lc-rs, ring,
rustfft, onig-sys, openblas, cust}` all **absent** ✓ · 0 files ≥ 2000 lines ✓ · 0 production
`todo!()`/`unimplemented!()` ✓. Not committed/pushed (awaiting explicit user request).

### Documented follow-ups (post-campaign, not release blockers)
- torsh-ffi split into torsh-capi/torsh-node/torsh-wasm (needs root [workspace] surgery; imports fixed, split deferred).
- Lock-poison remaining crates (~700 sites: distributed 317, fx 62, jit 59, vision 46, …) — helper exists, mechanical.
- Example relocation blocked on an example-rewrite pass (root examples/ have 0.1.x-era API drift); webgpu feature-name fixed, honest status banner added.
- torsh-core cudnn-sys optional C-FFI leak (F203); blake3 asm transitive via scirs2-datasets; hdf5 → oxih5 when ready.
- ~~Deeper perf: Storage::Device residency, sum_dim non-keepdim autograd, gelu/leaky_relu backward records, RNN cell narrow-gradient~~ **ALL DONE — see the 2026-08-13 follow-up campaign below.**

Full finding digest: session scratchpad `discovery.json` / `digest.md` (318 items, F000–F317).

---

## ✅ Perf/Autograd Follow-up Campaign (2026-08-13, COMPLETE — 6 waves, all green)

The four "Deeper perf" follow-ups turned out (4-agent investigation, runtime-probed) to be mostly
SILENT-CORRECTNESS bugs; fixing them cascaded into a full autograd-completeness campaign.

### Landed (uncommitted on branch 0.2.0)
- **Operation::SumDim** + deleted the silently-TRANSPOSING `Operation::Mean` producer (softmax grad
  was 0.54 max-abs wrong; nll backward hard-errored; ~106 call sites affected). mean/var/std/
  softmax/nll/cross_entropy now differentiate correctly by composition (FD-verified).
- **UnaryKind::Gelu + Operation::LeakyRelu{slope: T}** + 14 more unary records (tan/asin/acos/atan/
  sinh/cosh/log10/log2/rsqrt/reciprocal/square→Power, abs) + **AddScalar** + recording
  **maximum/minimum/clamp** (PyTorch tie rules: ties 0.5/0.5, |x|′(0)=0, clamp inclusive-bound).
- **Tensor::map made forward-only** — it was a silent gradient sink (requires_grad=true Leaf) that
  produced measurably wrong GRU/GRUCell gradients. **Exact GELU on all paths** (removed scirs2
  clamped-Pade SIMD kernel: forward was discontinuous at numel=1000, 515% rel err at x=−2.4).
- **TensorStorage::Device residency** (gpu feature): injectable backend + CountingBackend proof —
  chained 3 ops went 3×H2D/3×D2H → **1 upload + 1 download**; immutable device buffers, mutation
  demotes, leak-checked error paths, default build stays oxicuda-free. GPU suite 872 tests.
- **RNN/LSTM/GRU end-to-end BPTT**: stack/cat/dropout severs fixed; `RNN::forward` real
  implementation (was a zeros stub; bidirectional params now registered); `slab_along_dim` hoist
  (LSTM backward O(T²)→O(T)); **ViewKind::Narrow** geometric record (no index_map allocation);
  **narrow() is now an aliasing view** (PyTorch parity + no slab copy per RNN gate split).
- **torsh-series**: hand-written BPTT deleted; `LSTMForecaster::fit` trains via real backward
  (head-to-head vs old BPTT from identical weights before switching; old BPTT mismodeled
  num_layers≥2).
- **torsh-nn functional**: softmax/log_softmax delegate (recording); swish (21.6% wrong grad) and
  mish (SIGN-FLIPPED grad) fixed; relu/sigmoid/tanh/gelu/leaky_relu/elu/selu reconnected
  (compile_time MLP now trainable); **14 losses reconnected** (l1/huber/smooth_l1/focal/dice/
  tversky/wing/center/infonce/bce/triplet_margin/contrastive/multi_margin/cosine_embedding — the
  last had a total batched-call failure, fixed to the PyTorch shape contract).
- **Module norm layers** (LayerNorm/GroupNorm/BatchNorm/InstanceNorm): statistics now on the graph
  (were frozen constants → incomplete gradients); running stats stay detached buffers.
- **Buffer-aware state_dict** (PyTorch parity): running stats survive save/load (were silently
  dropped — reloaded models evaluated differently); Sequential/ModuleList/ModuleDict
  named_buffers/named_children; Box<dyn Module> forwards every trait method (macro).
- **CoW value semantics**: clone-then-set_item no longer writes through (TODO item from
  2026-06-21 closed); views keep PyTorch write-through; `fill_` stride/offset-blindness fixed
  (view fill was clobbering the base prefix). t()/gather/index_select now record (duplicate-index
  gradient accumulation verified).
- **torsh-functional**: similarity losses (cosine_embedding/contrastive/triplet_margin/
  hinge_embedding/margin_ranking) reconnected; stale `row_sum` matmul workaround → `sum_dim`.
- **torsh-text**: padding embedding row is now actually zeroed (fill_ wrote to a dropped local).
- Test hygiene: grad-mode process-global race serialized (GRAD_MODE_GUARD pattern); memory_pool
  and torsh-graph flakes fixed/seeded; wave-introduced convergence test seeded (30/30 gates).

### Final verification (2026-08-13)
`cargo nextest run --workspace` **10,964 passed / 0 failed / 95 skipped** (campaign start: 10,638) ·
gpu suite 872 ✓ · doctests 614 ✓ · check/clippy `-D warnings` default+all-features+gpu all clean ✓ ·
fmt clean ✓ · `cargo deny check bans` ok ✓ · default graph free of oxicuda-backend/aws-lc-sys/
rustfft/onig-sys ✓ · 0 files ≥ 2000 lines ✓ · flake gates 90/90 ✓ · HEAD still `7ec403eb MOS`,
nothing committed.

### Follow-ups from this campaign (not blockers)
- extremum (max_dim/min_dim/amin/amax), cumsum/cumprod, sort backward records (need argmax-scatter /
  reverse-scan / permutation-inverse designs — genuinely different backward rules).
- grad-mode is a process-global AtomicBool (PyTorch's is thread-local); production use in
  torsh-autograd checkpoint.rs. Design decision needed before changing.
- GPU phase 2 on the A4000 Linux box: real-CUDA validation of Device residency (synchronize-before-
  copy semantics), gemm/conv/attention residency (column-major f64 marshalling), device-side
  backward, device memory pool.
- SIMD gelu could be re-added with an exact-tanh vector kernel (upstream scirs2 issue: its
  simd_gelu_f32 is a clamped-Pade approximation; relu/sigmoid SIMD verified exact and kept).
- torsh-nn `Tensor::where_tensor`/`eq` still don't record (worked around via constant-mask
  composition); recording variants would simplify future loss implementations.
- nll_loss masked-sum formulation: a −inf anywhere in a row poisons that row (documented contract;
  intrinsic to every differentiable masked-gather formulation).

---

## 🚀 GPU Backend Migration: scirs2-core → oxicuda 0.3 (IN PROGRESS — 2026-06-25)

### Implementation status (2026-06-25)
Environment NOTE: this dev box HAS a CUDA GPU (NVIDIA RTX A4000, sm_86) + CUDA Toolkit 12.0
+ cuDNN, so GPU work is verifiable here. (An earlier "CPU-only host" assumption was wrong.)

**DONE — verified: `clippy -D warnings` GREEN (default/gpu/cuda) for torsh-{tensor,autograd,
metrics,core,profiler}; gpu_dispatch tests (CpuBackend + REAL A4000) pass; torsh-nn compiles:**
- [x] **Phase 1**: `oxicuda-backend = "0.3"` wired. New `torsh-tensor/src/gpu_dispatch.rs` —
  `GpuDispatch` over `dyn ComputeBackend`, unary/binary f32 marshalling, CpuBackend tests.
- [x] **Phase 2 (live activations)**: relu/sigmoid (`math_ops.rs`) + tanh (`math_ops_trig.rs`)
  GPU-dispatch via `gpu_dispatch::try_unary_f32`. Removed scirs2 GPU from torsh-autograd
  (ReLU backward→CPU), torsh-metrics (dead `GpuBackend` field), torsh-profiler (dead plumbing).
- [x] **Phase 3**: dead scirs2 GPU code removed (torsh-core `gpu` module + `scirs2_gpu_available`
  cfg within it; `backend_integration.rs` scirs2 comment blocks).
- [x] **Phase 5**: `"gpu"` removed from scirs2-core workspace features; per-crate features rewired.
- [x] **Real CUDA backend (case-1: ToRSh-owned, leaf crates)**: new `torsh-tensor/src/
  cuda_backend/{mod,ptx_ops}.rs` implements `oxicuda_backend::ComputeBackend` over the lean
  leaf crates `oxicuda-{driver,launch,ptx}` 0.3 — **NO umbrella facade**. Real GPU `unary` /
  `binary` / `reduce` (PTX elementwise + reduction kernels), device `alloc/free/copy_htod/
  copy_dtoh`, `synchronize`. `active_backend()` (cuda feature) builds it once and adopts it
  when a device is present. **VERIFIED on the A4000**: relu/sigmoid/add run on-GPU and match
  CPU; parallel + single-threaded, no flakiness.
  - Fixed a latent bug copied from oxicuda's umbrella `backend/ptx_ops.rs` `launch_with`: it
    creates+drops a CUDA **context** per launch → "invalid context" on the 2nd op. ToRSh's
    version keeps ONE persistent context and makes only a cheap stream per launch.
    **NB(oxicuda): the umbrella `CudaBackend` likely has this same latent bug — fix upstream.**

**DEFERRED / follow-up:**
- [ ] **gemm / conv2d / attention on GPU**: ToRSh's thin CudaBackend returns `Unsupported`
  for these (`TODO(torsh-cuda-gemm)` / `(torsh-cuda-dnn)`). gemm needs `oxicuda-blas` (f64) +
  ToRSh f32-matmul wiring; conv/attention need `oxicuda-dnn`. Wire when GPU matmul is needed.
- [ ] **Phase 6 (AFTER oxicuda 0.3.0 release procedure, per user)**: add `UnaryOp::Gelu` /
  `Silu` / parameterized `LeakyRelu` + backward ops UPSTREAM in oxicuda, then wire
  gelu/leaky_relu/backward (`TODO(oxicuda-unaryop-gelu)`, `(oxicuda-unaryop-leakyrelu)`,
  `(oxicuda-backward-ops)`).
- [ ] **case-2 (long-term, oxicuda-side)**: factor `CudaBackend` out of the big `oxicuda`
  umbrella into a lean crate so consumers needn't pull the facade; ToRSh could then depend on
  that instead of vendoring `cuda_backend/`. (User chose case-1 for now.)
- [ ] **Phase 4 (deferred by choice, NOT blocked)**: torsh-backend cust-based CUDA
  consolidation. IS compile-verifiable on this box (CUDA SDK present → `cuda_available`); user
  chose "finish the dispatch path first". Tackle next here.
- [ ] **Residual (inert; not compiled / no scirs2 dep)**: disabled
  `ops/{activation,arithmetic,matrix,reduction}.rs` old scirs2 GPU code (`// pub mod ops;`
  disabled); `scirs2_gpu_available` cfg scaffolding in `backend_detection.rs` /
  `device/capabilities.rs` / `perf_monitor.rs` (phantom cfg, CPU branch, warning-free);
  `tensor_cores.rs` error strings. Clean when `ops/` is re-enabled/deleted and in Phase 4.

**Pre-existing, UNRELATED to this work** (reproduced at HEAD with my changes `git stash`ed):
- `torsh-data/src/core_framework.rs:314` — `Tensor::cat(&Vec<Tensor>, isize)` vs expected
  `cat(&[&Tensor], i32)`. Blocks `cargo check --workspace`. Independent of the GPU migration.
  - **RE-VERIFIED 2026-07-06 (during a trustformers 0.2.0 dependency review): no longer reproduces.**
    `core_framework.rs:315` now calls `Tensor::cat(&channel_refs, channel_dim as i32)` with
    `channel_refs: Vec<&Tensor<T>>`, matching the real signature at
    `torsh-tensor/src/advanced_ops.rs:773` (`pub fn cat(tensors: &[&Self], dim: i32) -> Result<Self>`).
    Confirmed with a clean `cargo check -p torsh-data` and a full `cargo check --workspace`
    (both green, no errors) on this HEAD. This was flagged externally as a candidate 0.2.0
    release blocker for publishing — it is not one; closing this line item as resolved.

---

### Original plan (for reference)

**Goal**: Move all GPU functionality off `scirs2-core` onto **oxicuda 0.3**'s unified
`ComputeBackend` trait, and simultaneously **consolidate** ToRSh's orphaned cust-based
native CUDA backend (`torsh-backend/src/cuda/`) onto the same trait. End state: one
pure-Rust GPU dispatch path, no build-time CUDA SDK, `scirs2-core` used only for non-GPU.

**Dependency premise**: `oxicuda-* = "0.3"` from crates.io (publish in progress; most done).

### Investigation findings (why this is low-risk) — 2026-06-25
- [x] **scirs2-core "GPU" consumed by ToRSh is ~0% real GPU.** High-level ops
  (gelu/relu/gemm/reductions) ignore the backend and call `*_cpu_fallback`;
  `execute_kernel` is a no-op stub (`eprintln!` + `Ok(())`); `GpuBackend::preferred()`
  force-returns `Cpu`. ToRSh calls `GpuContext::new(GpuBackend::Cuda).ok()` → `Err`
  (cuda feature off) → `None` → falls back to ToRSh's own CPU/SIMD. **Migrating loses
  zero working GPU functionality.** Refs concentrated in 5 files: `torsh-tensor/src/ops/
  {activation,arithmetic,matrix,reduction}.rs`, `torsh-autograd/src/context/gradient_functions.rs`.
- [x] **torsh-backend/src/cuda (80,860 SLoC / 215 files) is mostly inert.** Orphaned
  (no crate imports `torsh_backend::cuda`), double-gated (`feature cuda` + build.rs
  `cuda_available`, which needs the CUDA SDK on the build host), compute kernels are
  no-ops (no PTX, no cuBLAS), and `memory/optimization/` (41,510 SLoC) is a parked "ML
  memory optimizer" with **zero call sites**. Real substance ≈ device init + memory
  alloc (~6–8k) + cuDNN FFI (~5k). **→ mostly DELETE, small REPLACE.**
- [x] **oxicuda provides a clean unified `ComputeBackend`** (object-safe, zero-dep
  `oxicuda-backend`), implemented by real `CpuBackend` AND real `CudaBackend` (umbrella
  `oxicuda`, wires driver+blas+dnn) + Metal/Vulkan/WebGPU/ROCm/LevelZero. No build-time
  CUDA SDK (libloading at runtime) → **builds on CPU-only CI even with the CUDA path
  compiled in.** `CudaBackend::init()` degrades gracefully without a driver.
  NOTE: do **not** use `oxicuda::tensor_backend::GpuTensor` — it is CPU-simulated despite
  the name. The real path is the `backend` feature (`oxicuda::backend::CudaBackend`).

### Target architecture
```
torsh-tensor / torsh-autograd   (only when tensor device == Cuda/Metal/...)
        │
        ▼
torsh GpuDispatch   (NEW thin adapter; holds Box<dyn oxicuda_backend::ComputeBackend>)
        ├─ CpuBackend      (oxicuda-backend; dev/reference)
        ├─ CudaBackend     (oxicuda backend+blas+dnn; real NVIDIA)
        └─ {Metal,Vulkan,WebGpu,Rocm,LevelZero}Backend   (optional, per feature)
CPU-device tensors KEEP ToRSh's existing native CPU/SIMD path (NOT routed through trait).
```
Design points:
- `ComputeBackend` is a **flat, untyped op layer** (`u64` device pointers; GEMM f64, NN
  ops f32). ToRSh keeps owning tensor/dtype/shape/stride/autograd semantics.
- **MVP dispatch** mirrors today's per-op behavior: `alloc → copy_htod → op → copy_dtoh →
  free` (same as scirs2's per-op `create_buffer`/`read_buffer`). Correctness first.
- **Later optimization**: device-resident storage — add `Storage::Device { ptr: u64,
  backend, len }` so GPU tensors stay on device across ops (kills per-op H2D/D2H).
- Runtime selection: try `CudaBackend::init()`; if driver+device present use it, else the
  GPU path is simply not taken (tensor stays on CPU) — same semantics as today.

### Op mapping (ToRSh current → ComputeBackend)
| ToRSh op | ComputeBackend | Notes |
|---|---|---|
| relu / sigmoid / tanh | `unary(Relu/Sigmoid/Tanh,…)` | ✓ direct (Cpu+Cuda) |
| elementwise add/mul (+ sub/div) | `binary(Add/Mul/Sub/Div,…)` | ✓ direct |
| matmul / gemv | `gemm(…)` / `batched_gemm` | gemv = gemm with n=1 |
| sum reduction (+ mean/max/min) | `reduce(Sum/Mean/Max/Min,…)` | ✓ direct |
| **gelu** | — (gap) | `UnaryOp` has no Gelu → upstream add Gelu (oxicuda-blas already has `elementwise::unary::gelu`), or interim CUDA-only blas-direct |
| **leaky_relu(slope)** | — (gap) | unary has no param → upstream add parameterized variant, or compute via `binary`/mask |
| **relu_backward / *_backward** | — (gap) | not in trait → keep ToRSh CPU backward for MVP; GPU backward deferred (today only ReLU backward is wired, and it's optional) |

### Phase 1 — Dependency wiring + adapter skeleton
- [x] Workspace `Cargo.toml [workspace.dependencies]`: add `oxicuda-backend = "0.3"` and
  `oxicuda = { version = "0.3", features = ["backend","blas","dnn"] }`.
- [x] NEW `torsh-tensor/src/gpu_dispatch.rs`: `GpuDispatch` holding `Box<dyn ComputeBackend>`,
  `OnceLock` singleton, runtime selection (`CudaBackend::init()` → else none), helpers
  `unary_f32 / binary_f32 / gemm_f32 / reduce_f32` (alloc→copy_htod→op→copy_dtoh→free).
- [x] Map `torsh_core::DeviceType::Cuda(_)` (and future Metal/etc.) → backend selection.

### Phase 2 — Replace scirs2-core GPU in torsh-tensor / autograd / metrics
- [x] `torsh-tensor/src/ops/activation.rs`: replace `try_gpu_unary_f32` (relu/sigmoid/tanh/
  gelu) + `try_gpu_kernel_unary_f32` (leaky_relu) with `GpuDispatch::unary_f32`; handle
  gelu/leaky_relu per table. Drop `LeakyReluKernel` + scirs2 gpu imports.
- [x] `torsh-tensor/src/ops/arithmetic.rs`: `add_op`/`mul_op` GPU path → `binary_f32`. Drop
  `ElementwiseAddKernel`/`ElementwiseMulKernel`.
- [x] `torsh-tensor/src/ops/matrix.rs`: `matmul_2d_1d` (GEMV) → `gemm_f32`; **DELETE** stale
  `gpu_matmul_2d` (uses non-existent scirs2 API). Drop `GemvKernel`/`GpuElement`.
- [x] `torsh-tensor/src/ops/reduction.rs`: **DELETE** stale `gpu_sum_all` or reimplement via
  `reduce_f32`. Drop `GpuElement`.
- [x] `torsh-autograd/src/context/gradient_functions.rs`: drop `try_gpu_backward_binary_f32`
  + `ReLUGradient::backward` GPU path (keep CPU backward) until backward ops land upstream.
- [x] `torsh-metrics/src/gpu.rs`: remove `scirs2_core::gpu::GpuBackend` field (compute already CPU).
- [x] `torsh-profiler/src/scirs2_integration.rs`: drop scirs2 `GpuContext` usage or repoint to oxicuda device info.
- [x] Per-crate `Cargo.toml` features: `torsh-tensor:79`, `torsh-autograd:72`, `torsh-metrics:39`
  — change `gpu = [...,"scirs2-core/gpu"]` → `gpu = ["dep:oxicuda-backend"]`; add
  `cuda = ["gpu","dep:oxicuda"]` (umbrella backend/blas/dnn features).

### Phase 3 — Delete dead scirs2-GPU code
- [x] `torsh-core/src/lib.rs:438`: DELETE dead `#[cfg(scirs2_gpu_available)] pub use
  scirs2_core::gpu::*` + fallback (cfg is never set by any build.rs).
- [x] `torsh-core/src/backend_detection.rs`: remove doc-only scirs2 gpu API references (~419–605).
- [x] `torsh-tensor/src/backend_integration.rs`: remove placeholder `GpuContext`; repoint to `GpuDispatch`.

### Phase 4 — Consolidate / retire torsh-backend native CUDA
- [ ] **DELETE** `torsh-backend/src/cuda/memory/optimization/` (41,510 SLoC, 0 call sites).
- [ ] **DELETE** parked scaffolding: `intelligent_task_scheduler.rs`, `intelligent_scheduler.rs`,
  `performance_optimization_coordinator.rs`, `kernel_fusion_optimizer.rs`,
  `high_performance_kernels.rs`, `advanced_gpu_optimizer.rs`, `multi_stream_orchestrator.rs`,
  `multi_stream_usage_examples.rs`, `graph_execution.rs`/`graph_stub.rs`,
  `cooperative_groups.rs`, `occupancy.rs`; trim `memory/statistics/` boilerplate.
- [ ] **REPLACE** the real core (device init/alloc/streams/cuDNN; backend.rs no-op ops) by
  delegating to oxicuda driver+memory+CudaBackend — or remove entirely if `GpuDispatch`
  makes it redundant.
- [ ] `torsh-backend/build.rs`: drop `cuda_available`/`cuda_runtime_available` detection
  (oxicuda handles availability at runtime → no build-time CUDA SDK).
- [ ] `torsh-backend/Cargo.toml`: drop `cust`/`cuda-sys`/`cudnn-sys`; repoint
  `cuda`/`metal`/`webgpu`/`rocm` features to oxicuda backends.

### Phase 5 — Remove GPU from scirs2-core
- [x] Workspace `Cargo.toml:133`: remove `"gpu"` from the `scirs2-core` features list.
- [x] Verify `grep -r 'scirs2_core::gpu\|scirs2_core::cuda' crates/` == 0.

### Phase 6 — Upstream oxicuda additions (small; we own oxicuda)
- [ ] Add `Gelu` (+ `Silu`, parameterized `LeakyRelu`) to `oxicuda-backend::UnaryOp`;
  implement in `CpuBackend`; wire `CudaBackend` to `oxicuda-blas` gelu/silu — so
  activations work uniformly through the flat trait.
- [ ] (Optional) Add backward unary ops (`relu_backward`, …) to enable GPU autograd backward.

### Phase 7 — Verification (No-warnings + Refactoring policy)
- [ ] `cargo build --workspace` (CPU, no CUDA SDK) green.
- [ ] `cargo build --workspace --features cuda` green on CPU-only host (compiles; no device at runtime).
- [ ] `cargo nextest run --workspace --all-features` green.
- [ ] `cargo clippy --workspace --all-features --all-targets -- -D warnings` clean.
- [ ] (If NVIDIA hardware) numerical parity gelu/relu/gemm/sum vs CPU; smoke perf.
- [ ] 0 files ≥ 2000 lines after the torsh-backend cuda deletions.

### Risks / open decisions
- oxicuda on-silicon numeric/perf validation is a v0.4 item → verify on real GPU before relying on it.
- `CudaBackend` lives in umbrella `oxicuda` — confirm 0.3.0 umbrella is published; else depend
  on leaf `oxicuda-{driver,memory,blas,dnn}` and implement a thin `CudaBackend` in ToRSh.
- Flat u64-pointer trait: MVP per-op H2D/D2H (matches today); device-resident `Storage::Device`
  is the real perf win (schedule after correctness).
- f16/bf16/fp8 breadth lives in GPU BLAS/DNN crates, not the flat trait — bridge separately if needed.

---

## ✅ SIMD Performance Optimization: ALL 7 PHASES COMPLETE

**Status**: ✅ **COMPLETED** (January 1, 2026)
**Summary**: All SIMD performance issues have been resolved through a comprehensive 7-phase optimization plan.

### Final Benchmark Results (50K f32 elements, Apple Silicon)

| Benchmark | Time | vs Scalar | Status |
|-----------|------|-----------|--------|
| pure_scalar | 4.5 µs | 1.0x | baseline |
| raw_simd_plus_fast_result | **5.5 µs** | **1.2x** | ✅ **Optimal** |
| tensor_simd_with_locks | 18.5 µs | 4.1x | full tensor path |

### Completed Optimization Phases

- [x] **Phase 1**: scirs2-core Zero-Allocation API (`simd_add_into`, `simd_mul_into`)
- [x] **Phase 2**: Uninit Buffer Allocation (saves ~8µs for 50K elements)
- [x] **Phase 3**: Streamlined SIMD Integration (`add_op_simd_phase3`, `mul_op_simd_phase3`)
- [x] **Phase 4**: Adaptive Size-Based Dispatch (scalar <512, SIMD 512-65K, parallel >65K)
- [x] **Phase 5**: Lock-Free SimdOptimized Storage with Copy-on-Write semantics
- [x] **Phase 6**: AlignedVec API Completion
- [x] **Phase 7**: Direct Slice Access + Fast Result (`from_data_fast`, `try_as_slice_direct`)

**Key Insight**: On Apple Silicon, LLVM auto-vectorizes scalar loops, so raw SIMD ≈ scalar performance. The optimization focused on eliminating abstraction overhead.

**Details**: See `$HOME/.claude/plans/recursive-whistling-pancake.md`

---

### ~~Previous Performance Issues~~ (RESOLVED)

**Benchmark Results (macOS Apple Silicon M-series)** - December 31, 2025 after simplification:
- Element-wise Add (1K): 91.4ns (305x faster than broken hybrid) ✅
- Element-wise Add (50K): 4.45μs (46x faster than broken hybrid) ✅
- Element-wise Add (1M): 277.2μs (7-11x faster than broken hybrid) ✅
- Element-wise Mul (50K): 90.8μs (**334-490% faster** than broken hybrid) ✅
- **Status**: Simple scalar operations outperform complex broken SIMD by 300x

**What We Learned** (Dec 31, 2025):
1. ✅ **VERIFIED**: Real SIMD implementation attempted - 21-570% SLOWER due to memory copies
2. ✅ **ROOT CAUSE**: `Arc<RwLock<Vec<T>>>` architecture requires 4 memory copies for SIMD operations
3. ✅ **SOLUTION**: Removed broken complex logic, simplified to scalar operations (300x improvement)
4. ✅ **INSIGHT**: Memory copy overhead (10-100μs) >> SIMD computation savings (0.1μs)
5. ⏸️ **BLOCKED**: Real SIMD needs TensorView (CRITICAL #1) for zero-copy operations

**Key Insight**: Simple scalar operations outperform complex poorly-architected SIMD by 300x. TensorView must come first.

**Status**: Performance improved 300x by removing broken optimizations. Further improvements blocked on architecture fixes.

### Priority 0: Emergency Fixes (MUST FIX BEFORE RELEASE)

#### CRITICAL #1: Fix Tensor Creation Overhead 🔥 ✅ **PHASES 1 & 2 COMPLETE**
> (Tracked in crates/torsh-tensor/TODO.md — planned 2026-04-19 v0.1.2 slice: blocks A–G dispatched)
- [x] **Phase 1 COMPLETE**: Implement zero-copy scoped access (Dec 31, 2025)
  - [x] `TensorView<'a, T>` and `TensorViewMut<'a, T>` types implemented
  - [x] `with_data_slice()` and `with_data_slice_mut()` methods added
  - [x] 20 comprehensive tests passing (12 unit + 8 integration)
  - [x] Zero memory copies for InMemory/Aligned storage
  - [x] Enables real SIMD operations (unblocks CRITICAL #2)
- [x] **Phase 2 IMPLEMENTED (but failed benchmarks)**: SIMD operations with zero-copy inputs (Dec 31, 2025)
  - [x] `add_op_simd_f32_zero_copy()` and `mul_op_simd_f32_zero_copy()` implemented
  - [x] Updated `add_op()` and `mul_op()` to use zero-copy SIMD (later reverted)
  - [x] Created comprehensive benchmark suite (`zero_copy_simd_benchmark.rs`)
  - [x] All 486 tests passing, zero warnings
  - [x] Benchmarked and discovered: SIMD still 2-5x slower due to output allocations
  - [x] **REVERTED SIMD** to scalar operations (scalar is faster)
  - [x] ⚠️ **CRITICAL #2 STILL BLOCKED** - need Phase 2.5 to fix output allocations
- [x] **Phase 2.5 DONE**: Buffer-writing SIMD implemented in `ops/simd/f32_ops.rs`
  - [x] `add_op_simd_f32_buffer()` / `mul_op_simd_f32_buffer()` — 1 allocation (down from 4)
  - [x] Phase 7 direct SIMD (`add_direct_simd`) for SimdOptimized storage — zero closure overhead
  - [x] Adaptive dispatch: scalar (<512 elems), direct SIMD (512-65K), parallel SIMD (>65K)
- [ ] **Phase 3 PENDING**: Add in-place operation variants (`add_!`, `mul_!`, etc.)
- **Files**: `crates/torsh-tensor/src/{tensor_view.rs, storage.rs, core_ops.rs, ops/arithmetic.rs, ops/simd/f32_ops.rs}`
- **Status**: ⚠️ Phase 1 SUCCESS (zero-copy inputs), Phase 2 FAILED (output allocations)
- **Details**: See `/tmp/simd_benchmark_results_20251231.md` for failure analysis

#### CRITICAL #2: Implement Real SIMD Operations 🔥 ⚠️ **STILL BLOCKED - OUTPUT ALLOCATIONS**
- [x] **Investigation Phase** (Dec 31, 2025 morning):
  - [x] Attempted real SIMD with memory copies: 21-570% SLOWER
  - [x] Root cause identified: Memory copy overhead (20-200μs) >> SIMD benefit (0.1μs)
  - [x] Simplified to scalar, placed SIMD ON HOLD pending architecture fix
- [x] **Architecture Fix Attempt** (Dec 31, 2025 afternoon):
  - [x] CRITICAL #1 Phase 1: Implemented zero-copy scoped access ✅
  - [x] CRITICAL #1 Phase 2: Implemented SIMD with zero-copy inputs ✅
  - [x] Created `add_op_simd_f32_zero_copy()` and `mul_op_simd_f32_zero_copy()`
  - [x] Successfully eliminated input copies (20-200μs saved)
- [x] **Verification & Failure Discovery** (Dec 31, 2025):
  - [x] Ran benchmarks: `cargo bench --bench zero_copy_simd_benchmark --features simd`
  - [x] **CRITICAL FINDING**: SIMD still 2-5x SLOWER than scalar
  - [x] **Root cause**: Output allocations dominate (4 allocations vs 2 for scalar)
  - [x] **Reverted SIMD** to scalar operations (Dec 31, 2025)
- **Files**: `crates/torsh-tensor/src/ops/{arithmetic.rs, simd/f32_ops.rs}`
- **Status**: ✅ Phase 2.5 COMPLETE — buffer-writing SIMD with adaptive dispatch active
- **Details**: See `/tmp/simd_benchmark_results_20251231.md` for original failure analysis

#### CRITICAL #3: Fix Benchmark Methodology 🔥 ✅ **COMPLETED**
- [x] Separate tensor creation from measurement (DONE - Dec 31, 2025)
- [x] Rewrite `simd_performance.rs` benchmarks (DONE - All 7 functions fixed)
- [x] Corrected benchmarks reveal true performance issues (SIMD 5-11x slower)
- [ ] Add memory allocation tracking (TODO)
- **Files**: `crates/torsh-tensor/benches/simd_performance.rs`
- **Status**: Benchmarks now measure actual operation performance correctly

#### CRITICAL #4: Reduce Memory Allocations 🔥 ✅ **BUFFER POOLING COMPLETE (2026-05-18)**
> (Tracked in crates/torsh-tensor/TODO.md — planned 2026-04-19 v0.1.2 slice: blocks A–G dispatched)
- [x] Implement buffer pooling (`scirs2_core::memory::BufferPool`) — wired into 9 hot-path sites via `global_acquire_uninit` + `ReusedBuffer<T>::into_vec(len)` (2026-05-18)
  - `math_ops.rs`: 6 sites (add/sub/mul/div SIMD paths + broadcast_add + broadcast_binary_op)
  - `storage.rs`: get_slice output buffer
  - `shape_ops.rs`: transpose_2d output
  - `ops/arithmetic.rs`: broadcast_binary_op output
  - `ops/matrix.rs`: diagonal extraction output
- [x] **Round 3**: Aligned buffer pool extension — `acquire_uninit_aligned<T>(count, align)` + `global_acquire_uninit_aligned` free function. 3-tuple key `(TypeId, SizeClass, Alignment)`. 4 new tests. (2026-05-18)
- [x] **Round 2/3**: Add in-place operations for element-wise ops — `add_/sub_/mul_/div_` enhanced with broadcast support, 8 new tests
- [ ] Use views instead of clones
- [ ] `shape_ops.rs` `expand` (recursive helper needs refactor for pool integration)
- [ ] `storage.rs:210,247` AlignedVec sites — needs scirs2-core change (`From<ReusedBuffer<T>>` impl)
- **Files**: `crates/torsh-tensor/src/{storage.rs, math_ops.rs, shape_ops.rs, ops/arithmetic.rs, ops/matrix.rs, memory_pool.rs}`
- **Target**: 90% reduction in allocations (hot paths done)

#### Phase 2 GPU Kernel Integration ✅ **5 KERNELS WIRED (2026-05-18)**
- [x] GELU (forward): `GpuContext::gelu()` real CPU fallback, routes to CUDA when available
- [x] ReLU/Sigmoid/Tanh (forward): same pattern via shared `try_gpu_unary_f32<T, F>` helper
- [x] LeakyRelu (forward): via `execute_kernel` architectural pattern (Round 3)
- [x] ElementwiseAdd: via `execute_kernel` for ≥ 65536 elements
- [x] ElementwiseMul: via `execute_kernel` for ≥ 65536 elements (Round 3)
- [x] GemV: via `execute_kernel` for ≥ 65536 elements
- [x] ReLU backward: `GpuContext::relu_backward(grad, input)` (Round 3)
- [ ] Sigmoid/Tanh backward: BLOCKED — `*Gradient` structs save `output_values` but scirs2-core's backward methods need raw input. Requires upstream forward-pass refactor.
- [ ] GELU backward: no `GELUGradient` struct in torsh-autograd yet
- [ ] SwishKernel/ElementwiseSub/Div/BatchGemv: low priority, easy next round
- **Files**: `crates/torsh-tensor/src/ops/{activation.rs, arithmetic.rs, matrix.rs}`, `crates/torsh-autograd/src/context/gradient_functions.rs`

### Priority 1: PyTorch Comparison (REQUIRED)

**Detailed Performance Analysis**: See `/tmp/performance_fixes_todo.md` and `/tmp/corrected_benchmark_analysis.md`

- [ ] Run `cargo run --example pytorch_performance_suite --features pytorch`
- [ ] Document actual performance gap vs PyTorch 2.7
- [ ] Create comparison tables for README
- [ ] Set realistic performance targets:
  - v0.1.0: Within 5x of PyTorch CPU (currently: 10-50x slower)
  - v0.1.0: Match PyTorch CPU
  - v1.0.0: Beat PyTorch by 1.5-2x

### Priority 2: Update Documentation with Honest Claims

- [x] **README.md**: Replaced "2-3x faster than PyTorch" claim with accurate description (2026-05-11)
- [x] **TODO.md**: Phase 2.5 marked done, CUDA support marked ✅ (2026-05-11)
- [ ] **CHANGELOG.md**: Document CUDA enhancements and Python bindings additions
- [ ] Add "Known Issues" section to all docs

### Release Blockers

v0.1.0 Release Status:
1. ✅ Correctness (tests passing) - DONE
2. ✅ API coverage (95%+) - DONE
3. ✅ **SIMD Performance Optimized** - DONE (7 phases complete, Jan 1, 2026)
4. ✅ **Comprehensive benchmarks** - DONE (`zero_copy_simd_benchmark.rs`)
5. ⏳ **Documentation updates** - IN PROGRESS

**Status**: Ready for release after documentation updates

---

## 🎯 Our Vision (Long-term Goals)

Build a **PyTorch-compatible deep learning framework in pure Rust** that combines:
- **Performance**: Competitive with PyTorch (SIMD optimizations complete, ~1.2x scalar baseline)
- **Safety**: Rust's compile-time guarantees eliminate entire classes of bugs
- **Completeness**: Full scientific computing platform through SciRS2 integration
- **Deployment**: Single binary, no Python runtime, edge-to-cloud ready

## ✨ What We Have Now (v0.1.2)

### 🚀 v0.1.2 Status: Production-Ready Core ✅

✅ **Performance issues resolved** (January 1, 2026): All 7 phases of SIMD optimization complete. See completed section above for benchmark results.

### Core Capabilities ✅
- **Tensor Operations**: ~458 PyTorch-compatible operations (96%+ coverage)
- **Automatic Differentiation**: Complete reverse-mode AD with gradient computation
- **Neural Network Layers**: All essential layers (Linear, Conv, BatchNorm, RNN, LSTM, Transformer)
- **Optimizers**: 70+ optimizers including SGD, Adam, AdamW, and advanced variants
- **Data Loading**: Parallel data processing with multi-worker support
- **CPU Backend**: SIMD-optimized operations with excellent performance

### Scientific Computing ✅
- **19 SciRS2 Crates Integrated**: Complete scientific computing ecosystem (0.3.3 **stable**)
- **OxiBLAS 0.1.2**: Optimized BLAS/LAPACK operations with performance improvements
- **scipy.linalg Compatibility**: 35 new linear algebra functions (svd, eig, qr, lu, cholesky, etc.)
- **Graph Neural Networks**: GCN, GAT, GraphSAGE
- **Time Series Analysis**: STL, SSA, Kalman filters
- **Computer Vision**: Spatial operations, feature matching
- **Sparse Tensors**: COO, CSR formats
- **Special Functions**: Gamma, Bessel, error functions

### Quality Metrics ✅
- **9,600+ Unit Tests Passing**: 100% pass rate
- **Zero Compilation Errors**: All workspace packages compile cleanly
- **Zero Warnings**: 100% compliance with no-warnings policy
- **35/35 Packages**: 100% compilation success (torsh-distributed tests excluded)
- **Stable Dependencies**: Built on SciRS2 0.3.3 stable (no RC versions)

### v0.1.0 Milestone
- **🎓 API Stabilization**: Core APIs are stable
- **🎯 100% Pure Rust (Default Features)**: Zero C/Fortran dependencies in default build
  - Removed `libc` → Pure Rust `sysinfo`
  - Removed `ndarray-linalg`/`lapack`/`blas` → OxiBLAS 0.1.2
  - No system BLAS/LAPACK required
  - No C/Fortran compiler needed
- **SciRS2 0.3.3 Stable**: Latest ecosystem release
- **OxiBLAS 0.1.2 Stable**: Performance improvements and bug fixes
- **OptiRS**: Upgraded to latest version
- **✅ SciRS2 POLICY 100% Compliance**:
  - Completed rayon → scirs2_core::parallel_ops migration
  - All parallel operations use scirs2_core exclusively
- **numrs2 Removed**: All functionality migrated to scirs2-core (improved SciRS2 POLICY compliance)
- **torsh-cli Refactored**: Now uses main torsh meta-crate with unified imports
- **Zero Warnings Policy**: Achieved 100% clean build (fixed 60+ warnings)
- **Dependency Upgrades**: Polars 0.52, Tempfile 3.24, Cranelift 0.127
- **Published Dependencies**: No local patches, all from crates.io

### Release Commitments
- **API Stability**: Core APIs (torsh, torsh-nn, torsh-tensor, torsh-autograd) stabilized
- **Production-Ready Core**: All core crates ready for production use
- **Semver Compliance**: Breaking changes minimized and well-documented
- **Quality Guarantee**: 99.99% test pass rate, zero warnings

### PyTorch API Compatibility Checklist

#### Core Tensor Operations ✅ (Nearly Complete - 95%+)
- [x] Basic arithmetic (add, sub, mul, div, pow)
- [x] Matrix operations (matmul, transpose, mm, bmm, tril, triu, diagonal)
- [x] Reduction operations (sum, mean, max, min)
- [x] **Advanced reductions** ✅ (argmax, argmin, prod, cumsum, cumprod)
- [x] **Statistical operations** ✅ NEW (median, median_dim, mode, mode_dim)
- [x] Activation functions (relu, sigmoid, tanh, gelu)
- [x] Shape manipulation (reshape, view, squeeze, unsqueeze, unflatten)
- [x] **Dimension manipulation** ✅ NEW (movedim, moveaxis, swapaxes, swapdims)
- [x] **Tensor manipulation** ✅ (cat, stack, split, chunk, flip, roll, rot90, tile, repeat, repeat_interleave)
- [x] **Advanced indexing** ✅ NEW (gather, scatter, index_select, take_along_dim)
- [x] Creation ops (zeros, ones, randn, arange, linspace)
- [x] Indexing and slicing
- [x] Broadcasting support (expand, expand_as, broadcast_to)
- [x] Comparison operations (eq, ne, lt, gt, le, ge)
- [x] Logical operations (logical_and, logical_or, logical_not)
- [x] **NaN/Inf detection** ✅ (isnan, isinf, isfinite, allclose, isclose)
- [x] **Masked operations** ✅ (masked_fill, masked_fill_, nonzero)
- [x] Trigonometric functions (complete set)
- [x] Complex number support
- [x] FFT operations
- [x] Sorting and searching (sort, argsort, topk)
- [x] Unique and bincount
- [x] Histograms
- [x] Random sampling operations (multinomial, normal_, etc.)
- [x] **In-place operation variants** ✅ (add_, mul_, sub_, div_, relu_, sigmoid_, etc.)

#### Functional API (torch.functional.*) 🚧
- [x] broadcast_tensors
- [x] einsum (basic)
- [x] norm
- [x] cartesian_prod
- [x] cdist
- [x] chain_matmul
- [x] istft/stft
- [x] meshgrid
- [x] tensordot
- [x] unique/unique_consecutive
- [x] block_diag
- [x] atleast_1d/2d/3d
- [x] lu decomposition
- [x] split (advanced variants)

#### Neural Network Modules (torch.nn.*) ✅ (Mostly Complete)
##### Core Layers
- [x] Linear
- [x] Conv1d, Conv2d, Conv3d
- [x] ConvTranspose1d, ConvTranspose2d, ConvTranspose3d
- [x] BatchNorm1d, BatchNorm2d, BatchNorm3d
- [x] LayerNorm
- [x] GroupNorm
- [x] InstanceNorm1d, InstanceNorm2d, InstanceNorm3d
- [x] Dropout, Dropout2d, Dropout3d
- [x] RNN, LSTM, GRU
- [x] Embedding
- [x] EmbeddingBag
- [x] MultiheadAttention
- [x] TransformerEncoder, TransformerDecoder

##### Activation Functions
- [x] ReLU, ReLU6, LeakyReLU, PReLU, ELU, SELU
- [x] Sigmoid, Tanh, Softmax, LogSoftmax
- [x] GELU, SiLU (Swish), Mish
- [x] Hardshrink, Softshrink
- [x] Hardtanh, Softplus, Softsign
- [x] Threshold, Hardsigmoid, Hardswish

##### Pooling Layers
- [x] MaxPool1d, MaxPool2d, MaxPool3d
- [x] AvgPool1d, AvgPool2d, AvgPool3d
- [x] AdaptiveMaxPool1d, AdaptiveMaxPool2d, AdaptiveMaxPool3d
- [x] AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveAvgPool3d
- [x] LPPool1d, LPPool2d
- [x] FractionalMaxPool2d, FractionalMaxPool3d

##### Loss Functions
- [x] MSELoss
- [x] CrossEntropyLoss
- [x] BCELoss, BCEWithLogitsLoss
- [x] NLLLoss
- [x] L1Loss, SmoothL1Loss, HuberLoss
- [x] KLDivLoss
- [x] MarginRankingLoss
- [x] TripletMarginLoss, TripletMarginWithDistanceLoss
- [x] CosineEmbeddingLoss
- [x] CTCLoss
- [x] PoissonNLLLoss, GaussianNLLLoss
- [x] MultiMarginLoss

##### Container Modules
- [x] Sequential
- [x] ModuleList
- [x] ModuleDict
- [x] ParameterList
- [x] ParameterDict

#### Optimizers (torch.optim.*) ✅ (Complete)
- [x] SGD (with momentum and Nesterov)
- [x] Adam
- [x] AdamW
- [x] Adagrad
- [x] RMSprop
- [x] Adadelta
- [x] Adamax
- [x] NAdam
- [x] ASGD (Averaged SGD)
- [x] LBFGS
- [x] RAdam
- [x] Rprop
- [x] SparseAdam

##### Learning Rate Schedulers ✅ (Complete)
- [x] StepLR
- [x] MultiStepLR
- [x] ExponentialLR
- [x] CosineAnnealingLR
- [x] ReduceLROnPlateau
- [x] CyclicLR
- [x] OneCycleLR
- [x] CosineAnnealingWarmRestarts
- [x] PolynomialLR
- [x] LinearLR
- [x] ConstantLR

#### Autograd (torch.autograd.*) ✅ (Core Complete)
- [x] Basic automatic differentiation
- [x] Gradient computation and accumulation
- [x] backward() API
- [x] grad() function
- [x] no_grad() context
- [x] enable_grad() context
- [x] GradientTape functionality
- [x] Higher-order derivatives
- [x] Gradient checkpointing
- [x] Custom autograd functions
- [x] Gradient clipping utilities
- [x] Anomaly detection mode
- [x] Profiler integration

#### Data Loading (torch.utils.data.*) ✅ (Mostly Complete)
- [x] Dataset abstract class
- [x] DataLoader with multiprocessing
- [x] TensorDataset
- [x] ConcatDataset
- [x] Subset
- [x] random_split
- [x] Sampler classes (Random, Sequential, etc.)
- [x] Collate functions
- [x] Worker management
- [x] IterableDataset
- [x] ChainDataset
- [x] DistributedSampler
- [x] WeightedRandomSampler
- [x] BatchSampler improvements

#### Distributed Training (torch.distributed.*) ✅ (Mostly Complete)
- [x] init_process_group
- [x] DistributedDataParallel (DDP)
- [x] FullyShardedDataParallel (FSDP)
- [x] all_reduce, all_gather, broadcast
- [x] RPC framework
- [x] Pipeline parallelism
- [ ] Model parallel support
- [x] Collective communication ops
- [ ] Rendezvous mechanisms
- [ ] Elastic training support

#### CUDA Support (torch.cuda.*) ✅ (Mostly Complete)
- [x] Basic CUDA tensor operations
- [x] Device management
- [x] Memory management (real cudaMalloc/Free/MallocManaged/HostAlloc + real fragmentation analysis)
- [x] cuDNN integration
- [x] cuBLAS integration
- [x] CUDA graphs
- [x] Multi-GPU support (ring all-reduce: Sum/Product/Min/Max/Average, type-safe dispatch)
- [x] Stream management (CudaStream, StreamPool, priority, callbacks, metrics)
- [x] Event synchronization (EventPool, CrossStreamBarrier, AsyncEventWaiter)
- [x] Memory pooling (UnifiedMemoryPoolManager wired to real CUDA allocators)
- [x] Unified memory support (cudaMallocManaged, cudaMemAdvise, cudaMemPrefetchAsync)
- [x] High-performance kernel manager (re-enabled: TensorCore, auto-tuning, kernel cache)
- [x] Kernel fusion optimizer (re-enabled: dependency analysis, code generation)
- [x] Intelligent task scheduler (re-enabled: dynamic priority, ring all-reduce integration)
- [x] Performance optimization coordinator (re-enabled: full 4-component integration)
- [ ] NCCL backend (mock impl; real NCCL requires cudarc/nccl feature — tracked for follow-up)

#### JIT Compilation (torch.jit.*) ✅ (Basic Complete)
- [x] Graph representation
- [x] Basic tracing
- [x] Kernel fusion
- [x] Optimization passes
- [x] Script mode
- [x] TorchScript export/import
- [x] Custom operators
- [x] Mobile optimization (in torsh-utils)
- [ ] Quantization support

#### Utilities (torch.utils.*) 🚧
- [x] checkpoint (gradient checkpointing)
- [x] clip_grad_norm_
- [x] Model serialization helpers
- [x] tensorboard integration
- [x] bottleneck profiler
- [x] collect_env (environment info)
- [x] cpp_extension utilities
- [x] model_zoo functionality
- [x] benchmark utilities
- [x] mobile_optimizer

#### Advanced Features 📋
- [x] torch.fx (graph transformation framework)
- [x] torch.ao.quantization (quantization toolkit)
- [x] torch.sparse (sparse tensor operations)
- [x] torch.linalg (linear algebra module)
- [x] torch.fft (FFT operations)
- [x] torch.special (special functions)
- [x] torch.signal (signal processing)
- [x] torch.profiler (advanced profiling)
- [x] torch.package (model packaging)
- [x] torch.hub (model hub integration)

### Missing Critical Components for v0.1.0

#### High Priority
1. **Attention Mechanisms** (torch.nn.attention.*)
   - [x] FlexAttention
   - [x] Scaled dot-product attention
   - [x] Memory-efficient attention
   - [x] Flash attention integration

2. **Graph Transformation** (torch.fx)
   - [x] Graph capture
   - [x] Graph manipulation
   - [x] Pass manager
   - [x] Subgraph rewriting

3. **Quantization** (torch.ao.quantization)
   - [x] INT8 quantization
   - [x] Quantization-aware training
   - [x] Post-training quantization
   - [x] Quantized operators

4. **Profiling Tools** (torch.profiler)
   - [x] CPU profiler
   - [x] CUDA profiler
   - [x] Memory profiler
   - [x] Chrome trace export

5. **Model Hub** (torch.hub)
   - [x] Model loading from hub
   - [x] Model publishing
   - [x] Dependency resolution
   - [x] Version management

#### Medium Priority
1. **Sparse Operations** (torch.sparse)
   - [x] COO sparse tensors
   - [x] CSR sparse tensors
   - [x] Sparse operations
   - [x] Sparse gradients

2. **Advanced Math** (torch.special, torch.linalg)
   - [x] Special functions (bessel, gamma, etc.)
   - [x] Advanced linear algebra (svd, qr, etc.)
   - [x] Eigenvalue decomposition
   - [x] Matrix functions

3. **Signal Processing** (torch.signal)
   - [x] Windows functions
   - [x] Spectral operations
   - [x] Filtering operations

## 🚀 What's Next: Post-v0.1.0 Roadmap

### Post-Release Goals (Q1 2026)

#### 1. API Stabilization 🔧
- **Goal**: Lock down public APIs for backward compatibility
- **What**: Review all public interfaces based on user feedback
- **Why**: Users need confidence that their code won't break
- **Status**: Collecting feedback from users

#### 2. GPU Acceleration Complete 🎮
- **Goal**: Production-ready CUDA and Metal backends
- **What**:
  - Complete cuDNN integration for all neural network ops
  - Metal Performance Shaders (MPS) optimization
  - Multi-GPU support with efficient data transfer
- **Why**: GPU acceleration is essential for deep learning
- **Status**: CUDA backend 70% complete, Metal 50% complete

#### 3. Distributed Training Enhancement 🌐
- **Goal**: Scale to multi-node training
- **What**:
  - Fully functional DistributedDataParallel (DDP)
  - Pipeline parallelism for large models
  - Gradient compression and communication optimization
- **Why**: Modern models require distributed training
- **Status**: Basic DDP working, needs production hardening

#### 4. Performance Optimization ⚡
- **Goal**: Achieve 2-3x speedup vs PyTorch consistently
- **What**:
  - Kernel fusion for common operation patterns
  - Memory pool optimization
  - SIMD auto-vectorization improvements
  - Profiling-guided optimizations
- **Why**: Performance is a key differentiator
- **Status**: Already 1.5-2x faster, targeting 2-3x

#### 5. Documentation & Examples 📚
- **Goal**: Comprehensive guides for all use cases
- **What**:
  - Complete API documentation
  - Tutorial series (beginner to advanced)
  - Real-world example projects
  - Migration guide from PyTorch
- **Why**: Great docs enable adoption
- **Status**: Basic docs exist, needs expansion

---

## 🎯 v1.0 Vision (Q3 2026)

### Production-Ready Framework

**Core Goals**:
- ✨ **100% PyTorch API compatibility** for common workflows (currently ~80%)
- ⚡ **Consistent 2-3x performance advantage** over PyTorch
- 🛡️ **Enterprise-grade stability** with comprehensive error handling
- 📦 **Pre-trained model zoo** with major architectures
- 🌍 **Industry adoption** by major companies

### What v1.0 Enables

#### For Researchers
- Drop-in replacement for PyTorch with minimal code changes
- Faster iteration cycles due to better performance
- Safer experimentation with Rust's type system
- Access to cutting-edge scientific computing via SciRS2

#### For Production Teams
- Single binary deployment (no Python runtime needed)
- Predictable performance and memory usage
- No GIL issues for concurrent inference
- Edge deployment (mobile, IoT, WASM) out of the box

#### For the Ecosystem
- Foundation for pure-Rust ML applications
- Integration with Rust web frameworks
- Native performance without FFI overhead
- Growing library of Rust-native models

## 🤝 How You Can Help

### As a User

**Try it and give feedback!**
1. **Test with your models** - Try porting PyTorch code and report what breaks
2. **Report bugs** - [Open issues](https://github.com/cool-japan/torsh/issues) with reproduction steps
3. **Suggest API improvements** - What's confusing? What's missing?
4. **Share benchmarks** - How does performance compare for your use case?

### As a Contributor

**We need help in many areas:**
- 🔧 **Core Development**: GPU backends, optimization, distributed training
- 📚 **Documentation**: Tutorials, examples, API docs
- 🧪 **Testing**: More test coverage, edge case discovery
- 🎨 **Tooling**: Better debugging, profiling, and visualization
- 🌐 **Ecosystem**: Integrations with other Rust crates

See [CONTRIBUTING.md](./CONTRIBUTING.md) for details.

---

## 📊 Detailed Implementation Status

Below are detailed checklists of what's implemented. These are primarily for maintainers tracking completeness.

---

## 🚀 **MAJOR INTEGRATION PLAN: SciRS2-Core Performance Features** (2025-09-28)

### 📋 **Integration Context**
Following comprehensive requirements submitted to SciRS2 team for SIMD operations, parallel processing, and GPU acceleration, **SciRS2 team has confirmed ALL requirements are met and exceeded** in the stable release. This integration plan implements the 4-phase rollout recommended by SciRS2 team.

### **✅ SciRS2 Response Confirmation**
- **SIMD Operations**: AVX2/SSE4.1/NEON support with 2-4x speedup guarantee
- **Parallel Operations**: Intelligent chunking with 2-4x speedup, 15-50% improvement over naive parallelism
- **GPU Acceleration**: Multi-backend support (CUDA/Metal/WebGPU/ROCm/OpenCL) with 10-100x speedup for large tensors
- **Ready-to-Deploy**: Production-ready APIs with stability guarantees

---

### **Phase 1: Parallel Operations Integration** ✅ **COMPLETED - December 30, 2025**
**Target Performance**: 2-4x speedup on multi-core tensor operations

#### **Implementation Tasks** ✅
- [x] **Update Cargo.toml dependencies** to use SciRS2 0.1.1 stable
  - Already using `scirs2-core = { version = "0.3.0", features = ["parallel", ...] }`
- [x] **Replace rayon usage** with SciRS2 parallel operations:
  - ✅ **torsh-tensor/src/math_ops.rs**: Replaced 2 `use rayon::prelude::*` with `scirs2_core::parallel_ops::*`
  - ✅ **torsh-backend/src/cpu/scirs2_integration.rs**: Replaced 11 inline rayon imports
  - ✅ **torsh-backend/src/cpu/optimized_kernels.rs**: Migrated to scirs2_core::parallel_ops
  - ✅ **torsh-backend/src/cpu/advanced_rayon_optimizer.rs**: Migrated to scirs2_core::parallel_ops
  - ✅ **torsh-backend/src/cpu/scirs2_parallel.rs**: Fully migrated wrapper module
  - ✅ **torsh-backend/src/sparse_ops.rs**: Migrated sparse operations
  - ✅ **torsh-functional/src/parallel.rs**: Removed ThreadPoolBuildError dependency
- [x] **Parallel operations already use scirs2_core**: No feature flags needed, direct usage
- [x] **Validate with comprehensive tests**:
  - ✅ torsh-functional: 422/422 tests passing
  - ✅ torsh-tensor: 385/385 tests passing
  - ✅ torsh-backend: 727/727 tests passing
  - ✅ Full workspace: 9059/9062 tests passing (99.97%)
- [x] **Zero compilation errors**: All 29 workspace packages compile cleanly
- [ ] **Benchmark comparison** between old rayon and new SciRS2 parallel performance (pending)
- [ ] **Documentation updates** for new parallel API usage patterns (in progress)

#### **Achieved Benefits** ✅
- ✅ **100% SciRS2 POLICY compliance** for parallel operations
- ✅ **Zero integration risk** - all tests passing
- ✅ **Backward compatibility** maintained - existing code works without changes
- ✅ **Clean migration path** - no direct rayon imports in core modules
- 🔄 **Performance validation** - benchmarking pending

---

### **Phase 2: GPU Kernel Integration** 🟡 **HIGH PRIORITY - Next Sprint**
**Target Performance**: 10-100x speedup for large tensors (>50K elements)

#### **Implementation Tasks**
- [ ] **Integrate GPU backends** in `backend_integration.rs`:
  - Replace CUDA/Metal placeholders with SciRS2 GPU kernels
  - Add support for neural network operations:
    ```rust
    use scirs2_core::gpu::kernels::ml::{GeluKernel, LeakyReluKernel, SwishKernel};
    ```
  - Implement element-wise operations:
    ```rust
    use scirs2_core::gpu::kernels::elementwise::{ElementwiseAddKernel, ScalarMulKernel};
    ```
  - Add linear algebra support:
    ```rust
    use scirs2_core::gpu::kernels::blas::{GemvKernel, BatchGemvKernel};
    ```
- [ ] **Update tensor device management** to use multi-backend GPU support
- [ ] **Modernize activation functions** with GPU-accelerated kernels:
  - **math_ops.rs**: Replace CPU-only implementations with GPU-capable kernels
  - **Add advanced activations**: GELU, LeakyReLU, Swish (SiLU) with GPU support
- [ ] **Add comprehensive GPU tests** for all supported backends
- [ ] **Performance benchmarking** to validate 10-100x speedup claims

#### **Expected Benefits (Short-term)**
- 10-100x speedup for GPU-accelerated neural networks
- Multi-backend GPU support (CUDA/Metal/WebGPU/ROCm/OpenCL)
- Production-ready GPU kernel library

---

### **Phase 3: Memory-Aligned SIMD** 🟢 **IN PROGRESS** (Started 2025-12-30)
**Target Performance**: 2-4x speedup over scalar operations with proper memory alignment

#### **Implementation Tasks**
- [x] **SIMD Infrastructure Setup** ✅ (2025-12-30)
  - [x] Made `adaptive_simd` module public for cross-module usage
  - [x] Fixed `element_wise_op_simd_f32` implementation in ops/simd/f32_ops.rs
  - [x] Integrated adaptive SIMD selection (14.17x peak speedup)
  - [x] Added `AlignedVec` support in storage.rs (already implemented)

- [x] **Adaptive SIMD Functions Available** ✅:
  - [x] `adaptive_simd_add_f32` - Hyperoptimized addition
  - [x] `adaptive_simd_mul_f32` - TLB-optimized multiplication (14.17x speedup)
  - [x] `adaptive_simd_div_f32` - Division with SIMD
  - [x] `adaptive_simd_dot_f32` - Dot product optimization

- [x] **SIMD Activation Functions** ✅ (2025-12-31):
  - [x] Uncommented SIMD implementations in activation functions
  - [x] Integrated scirs2-core SIMD functions (relu, sigmoid, gelu)
  - [x] Added SIMD-accelerated relu, gelu, sigmoid for f32 tensors > 1000 elements
  - [x] All 420 torsh-tensor tests passing (100% success rate)

- [x] **Tensor Storage with AlignedVec** ✅:
  - [x] TensorStorage::Aligned variant implemented
  - [x] Automatic selection for arrays > 1KB
  - [x] SIMD_ALIGNMENT support

- [ ] **Performance validation** (Next Sprint):
  - [ ] Benchmark adaptive SIMD vs scalar (target: 2-4x)
  - [ ] Validate 14.17x speedup on medium arrays
  - [x] Cross-platform testing (x86_64 AVX2, ARM64 NEON) — benchmark harness added: `crates/torsh-tensor/benches/cross_platform_simd.rs` (2026-05-18)

#### **Expected Benefits (Medium-term)**
- Memory-aligned SIMD for controlled performance optimization
- Cross-platform consistency across different hardware (x86_64, ARM64)
- Up to 4x improvement over unaligned operations

---

### **Phase 4: Advanced Optimization** 🔵 **OPTIMIZATION - Final Phase**
**Target Performance**: 15-30% automatic performance improvement

#### **Implementation Tasks**
- [x] **Integrated intelligent chunking** system (2026-05-11):
  - `optimized_kernels.rs`: `ChunkingUtils::matrix_blocks(m,n,k,4)` used in `optimized_matmul` for cache-optimal block sizes
  - New `chunked_elementwise`, `chunked_sum`, `chunked_mean` functions using `WorkloadType::{Elementwise,Reduction}`
  - 9 new tests covering all chunked operations
- [x] **Wire chunked dispatch** into `scirs2_integration.rs` simple AND parallel paths (2026-05-13):
  - Simple paths: `add_elementwise_simple`, `mul_elementwise_simple`, `add_scalar_simple`, `mul_scalar_simple`, `sum_simple` use `WorkloadType::Elementwise/Reduction`
  - Parallel paths: 6 chunk_size derivations replaced — matmul (`Matrix` via `matrix_blocks`), 4 SIMD elementwise/scalar paths (`Elementwise`, rounded to multiple of 4 for SIMD lanes), 1 reduction path (`Reduction`)
  - 16/16 `scirs2_integration` tests passing
- [ ] **Add performance profiling** integration for continuous optimization
- [ ] **Comprehensive benchmarking** to validate 15-30% automatic improvements

#### **Expected Benefits (Long-term)**
- Automatic performance optimization through intelligent chunking
- Future-proof architecture supporting new hardware capabilities
- Ecosystem integration with other SciRS2 projects

---

### **Quality Assurance & Risk Mitigation**

#### **Testing Strategy**
- [ ] **Update all 243 existing tests** to work with new SciRS2 APIs
- [ ] **Add performance regression tests** to ensure promised speedups
- [ ] **Cross-platform validation** on x86_64, ARM64, and other architectures
- [ ] **Memory safety validation** for aligned operations
- [ ] **Integration testing** across all ToRSh modules

#### **Risk Management**
- [ ] **Gradual rollout** with feature flags to enable/disable new functionality
- [ ] **Fallback mechanisms** to scalar operations if SciRS2 features unavailable
- [ ] **Comprehensive error handling** for GPU backend failures
- [ ] **Performance monitoring** to detect any regressions
- [ ] **Backward compatibility** maintained throughout integration

#### **Success Metrics**
- [ ] **Achieve SciRS2's performance targets**: 2-4x parallel, 2-4x SIMD, 10-100x GPU speedups
- [ ] **Maintain 100% test pass rate** (currently 243/243 tests passing)
- [ ] **Zero compilation warnings** across all platforms
- [ ] **Successful migration** from rayon to SciRS2 parallel framework

---

### **Integration Timeline**
- **Phase 1 (Parallel)**: 1 week - Immediate deployment for 2-4x speedup
- **Phase 2 (GPU)**: 2 weeks - Major performance gains for neural networks
- **Phase 3 (SIMD)**: 1 week - Memory-aligned optimization
- **Phase 4 (Advanced)**: 1 week - Final optimization and tuning

### **Expected Cumulative Impact**
- **Immediate**: 2-4x speedup on multi-core operations
- **Short-term**: 10-100x speedup for GPU-accelerated workloads
- **Medium-term**: Additional 2-4x SIMD improvements
- **Long-term**: 15-30% automatic optimization + future-proof architecture

**Status**: ✅ **READY FOR INTEGRATION** - SciRS2 team confirms all requirements met and exceeded

---

## Current Status (v0.1.0 Release) ✅

### Infrastructure Complete with Outstanding Test Results
- [x] Core tensor system with PyTorch-compatible API
- [x] Automatic differentiation with computation graphs
- [x] Neural network modules with parameter management
- [x] Optimization algorithms with state management (70+ optimizers)
- [x] Data loading with parallel processing
- [x] Backend abstraction (CPU, CUDA, Metal)
- [x] JIT compilation with kernel fusion
- [x] Functional transformations system
- [x] Tensor operations with advanced features
- [x] Benchmarking infrastructure
- [x] **9,600+ tests passing (100% pass rate)**
- [x] **Zero compilation warnings**

---

## Phase 1: Core Compatibility (v0.1.0 Status) ✅

### Essential for PyTorch Parity
1. **Complete Tensor Operations**
   - [ ] Remaining 20% of core ops
   - [x] Complex number support (Enhanced with real/imag extraction, polar conversion, complex tensor creation)
   - [x] Advanced indexing operations
   - [x] **In-place operation variants** ✅ **COMPLETED (2025-12-30)**
     - [x] Basic operations: add_, mul_, sub_, div_
     - [x] Scalar operations: add_scalar_, mul_scalar_, div_scalar_
     - [x] Activation functions: relu_, sigmoid_, tanh_, gelu_, leaky_relu_
     - [x] Utility functions: clamp_
     - [x] Comprehensive tests (17 tests added)
     - [x] PyTorch-compatible API (requires_grad checking)

2. **Neural Network Completeness**
   - [x] Enhanced activation functions (Added LogSigmoid, Tanhshrink)
   - [x] Advanced loss functions (Added HuberLoss, FocalLoss, TripletMarginLoss, CosineEmbeddingLoss)
   - [x] Parameter containers
   - [x] Lazy modules

3. **Distributed Training**
   - [x] Basic DDP implementation
   - [x] Process group management
   - [x] Collective operations
   - [x] Gradient synchronization with bucketing

4. **Python Bindings** ✅
   - [x] PyO3 integration with complete tensor and neural network bindings
   - [x] Python-compatible API with PyTorch drop-in replacement capability
   - [x] NumPy interoperability with zero-copy operations
   - [x] Complete package structure with proper error handling

## Phase 2: Advanced Features (v0.1.0) 📋

### Performance & Optimization
1. **Advanced Compilation**
   - [ ] TorchScript compatibility
   - [ ] Graph optimizations
   - [ ] Custom operator fusion
   - [ ] AOT compilation

2. **Quantization Support**
   - [ ] INT8 operations
   - [ ] Quantization schemes
   - [ ] Model compression
   - [ ] Deployment optimization

3. **Advanced Backends**
   - [x] WebGPU support
   - [x] ROCm/HIP support (basic implementation)
   - [ ] Intel GPU support
   - [ ] TPU integration

### Ecosystem Integration
1. **Model Hub**
   - [ ] PyTorch model import
   - [ ] ONNX compatibility
   - [ ] Model versioning
   - [ ] Automated testing

2. **Tool Integration**
   - [ ] TensorBoard support
   - [ ] Weights & Biases
   - [ ] MLflow integration
   - [ ] Experiment tracking

## Phase 3: Production Ready (v1.0.0) 📋

### Enterprise Features
1. **Deployment**
   - [ ] Model serving
   - [ ] Edge deployment
   - [ ] Mobile support
   - [ ] WASM compilation

2. **Monitoring**
   - [ ] Performance metrics
   - [ ] Model monitoring
   - [ ] A/B testing
   - [ ] Drift detection

3. **Security**
   - [ ] Model encryption
   - [ ] Secure computation
   - [ ] Privacy-preserving ML
   - [ ] Audit logging

## Compatibility Testing Strategy

### API Compatibility
- [ ] PyTorch API test suite port
- [ ] Behavior compatibility tests
- [ ] Performance regression tests
- [ ] Model migration validators

### Integration Testing
- [ ] Popular model architectures
- [ ] Common training recipes
- [ ] Ecosystem tool compatibility
- [ ] Cross-framework validation

### Migration Tools
- [ ] Automated code converter
- [ ] Model weight converter
- [ ] API compatibility layer
- [ ] Migration guide generator

## Success Metrics

### API Coverage (v0.1.0 targets)
- Core Operations: 80% (400+ ops)
- NN Modules: 90% (all common layers)
- Functional API: 95%
- Optimizers: 100% (all major algorithms)
- Data Loading: 80%
- Autograd: 100% (core functionality)

### Performance (vs PyTorch)
- Training: 1.5-2x faster
- Inference: 2-3x faster
- Memory: 50% reduction
- Compilation: 10x faster

### Adoption
- 1,000+ GitHub stars
- 100+ contributors
- 10+ production deployments
- 50+ ecosystem packages

## Development Principles

1. **PyTorch Compatibility First**: Ensure drop-in replacement capability
2. **Leverage scirs2**: Use existing implementations, don't reinvent
3. **Rust Advantages**: Memory safety, performance, deployment
4. **Test Coverage**: Maintain >90% test coverage
5. **Documentation**: API docs for every public function
6. **Performance**: Benchmark every feature against PyTorch

## Notes

- Priority on PyTorch API compatibility for easy migration
- Focus on most-used features first (80/20 rule)
- Maintain high code quality throughout
- Regular community feedback integration
- Coordinate with scirs2 team for backend features

## Pure Rust Migration (COOLJAPAN Policy)

Goal: keep the default build 100% Pure Rust (no C/C++/asm/Fortran). Items below are ordered by severity.

- [ ] **(HIGH — true C/asm violation) Replace `ring` 0.17 with `oxicrypto` (or RustCrypto).**
  - Declaration: workspace `Cargo.toml` line 278 (`ring = "0.17"`), under the `# Security features for package signing and encryption` comment (line 277). SINGLE consumer: `torsh-package` (`crates/torsh-package/Cargo.toml` line 45, `ring = { workspace = true }` — unconditional `[dependencies]`, not optional, not feature-gated).
  - ALL usage lives in ONE file: `crates/torsh-package/src/security.rs` (verified via `grep -rEn '\bring::'` — only 5 hits, no false positives from `clustering::`/`rendering::`/etc.). Surfaces in use:
    - **AEAD** — `ring::aead` (`UnboundKey` / `LessSafeKey` / `Nonce::try_assume_unique_for_key` / `Aad::empty` / `seal_in_place_append_tag` / `open_in_place`): AES-256-GCM at lines 364-398 and ChaCha20-Poly1305 at lines 400-437 (`use ring::aead;` at line 13).
    - **PBKDF2-HMAC-SHA256** — `ring::pbkdf2` (`pbkdf2::derive`, `PBKDF2_HMAC_SHA256`, 100_000 iterations, 32-byte key): lines 441-453.
    - **CSPRNG** — `ring::rand::{SecureRandom, SystemRandom}` (`rng.fill`): three sites — Ed25519 key seeding at lines 102-106, salt generation at lines 460-465, nonce generation at lines 470-475.
  - IMPORTANT: package **signing is ALREADY pure-Rust** via `ed25519-dalek` (`security.rs` line 12). The `SignatureAlgorithm` enum (line 37) declares `Ed25519` plus `Rsa`/`Ecdsa`, but the latter two are explicitly `(future support)` placeholders and every signing path hardcodes `SignatureAlgorithm::Ed25519` (lines 113/122/131). `ring` therefore provides NO signing in practice — it is ONLY the encryption (AEAD) + KDF + CSPRNG layer, so the word "signing" in the workspace comment (line 277) is stale.
  - Replacement: `oxicrypto` AEAD (AES-256-GCM + ChaCha20-Poly1305) + a PBKDF2-HMAC-SHA256 KDF + a `getrandom`-based CSPRNG; OR RustCrypto (`aes-gcm` + `chacha20poly1305` + `pbkdf2` + `getrandom`). NOTE: ring's `LessSafeKey` / `seal_in_place_append_tag` (append-tag-in-place) shape differs from these crates' `AeadInPlace`/`Aead` traits, so this is a genuine (small, single-file) rewrite of `security.rs`, not a namespace swap.
  - Acceptance: `ring` removed from `crates/torsh-package/Cargo.toml` and workspace `Cargo.toml`; `cargo tree -i ring` is empty (also verify any rustls/reqwest deps, if present, do not re-pull `ring` transitively); `cargo test -p torsh-package` green (encrypt/decrypt round-trip + key-derivation tests pass); no C/asm in the default build.

- [ ] **(consistency-only) Replace `lzma-rs` 0.3 with `oxiarc-lzma`.**
  - Declaration: workspace `Cargo.toml` line 275 (`lzma-rs = "0.3"`), sitting directly beside the existing `oxiarc-deflate = "0.3.2"` (line 273) and `oxiarc-zstd = "0.3.1"` (line 274) under the `# Advanced compression for package management (COOLJAPAN Pure Rust Policy)` comment. SINGLE consumer: `torsh-package` (`crates/torsh-package/Cargo.toml`, `lzma-rs = { workspace = true }`). `lzma-rs` is already pure-Rust, so this is COOLJAPAN OxiARC consistency, NOT a C-dependency violation. Map LZMA encode/decode to `oxiarc-lzma`.
  - Acceptance: `lzma-rs` removed from workspace `Cargo.toml` and `crates/torsh-package/Cargo.toml`, replaced by `oxiarc-lzma`; `cargo tree -i lzma-rs` empty; torsh-package compression round-trip tests green.

## Stubs to implement (added 2026-06-12 by /cooljapan-stub-check)

- [x] `torsh-functional`: `src/attention.rs:518,532,533` — Flash-attention block loop uses full tensor clone instead of proper row/column slicing for q/k/v blocks; implement actual slice extraction per block index.
  - Priority: P2 | Scope: medium | Hint: none

- [x] `torsh-functional`: `src/dropout.rs:368` — `fractional_max_pool2d` is a no-op placeholder returning input unchanged; implement fractional max pooling with random kernel positions. **DONE** (by 2026-06-20 strict-check; verified real 2026-06-21): real Ben Graham (2015) fractional pooling via `fractional_pool_sequence` (`starts[i]=floor(alpha*(i+u))`, PyTorch-compatible), honest errors for invalid sizes. (Curated list predates the strict-check fix.)
  - Priority: P2 | Scope: medium | Hint: none

- [x] `torsh-functional`: `src/linalg/basic.rs:88` — Matrix chain multiplication uses naive left-to-right order; implement optimal-parenthesization via dynamic programming (Hu-Shing or standard DP). **DONE 2026-06-21**: real CLRS O(n³) matrix-chain-order DP + `matrix_chain_optimal_cost`; tests assert textbook costs (7500, 15125) and product vs reference; clippy clean, 25 linalg tests pass.
  - Priority: P2 | Scope: small | Hint: none

- [x] `torsh-functional`: `src/linalg/mod.rs:362` — Eigenvalue decomposition silently fails on degenerate (rank-deficient, repeated-eigenvalue) matrices; add deflation / Wielandt-style handling. **DONE 2026-06-21** (real file `linalg/decompositions.rs`): old power-iteration+Hotelling broke on zero eigenvalues & padded FAKE basis vectors; replaced with scirs2-linalg `eigh`/`eig` + anti-fabrication residual gate (‖Av−λv‖/(|λ|+1)≤1e-3 per pair, else honest Err). Mutation-proven tests ({5,2,2}, {3,0,0}); nextest 516/516.
  - Priority: P2 | Scope: medium | Hint: none

- [x] `torsh-tensor`: `src/shape_ops.rs:728` — Tensor expand copies data element-by-element; implement strided-view expansion (zero-copy broadcast metadata, no data duplication). **DONE 2026-06-21**: confirmed storage model supports zero-copy views (strides/storage_offset, `compute_flat_index` honors stride-0, `Arc` storage share); real strided-view expand (−67 lines of copy), kept `Operation::Leaf` (no `Operation::Expand` so avoids silent requires-grad-no-flow); test proves NO duplication (1 elem→1M, memory unchanged); torsh-tensor 561/561.
  - Priority: P2 | Scope: medium | Hint: none

- [x] `torsh-tensor`: `src/conv.rs:128` — Bias addition in conv uses element-wise add without broadcasting; implement efficient channel-wise bias broadcast. **DONE 2026-06-21**: `add_channel_bias` helper (cache-friendly `chunks_mut` over [N,C,*] blocks) routed through all 5 conv variants, replacing ~90 lines of duplicated index math; mutation-tested correctness tests; clippy clean, 27/27 conv tests. (Premise note: old code WAS broadcasting, just inefficiently/duplicated — no behavior fabrication.)
  - Priority: P2 | Scope: small | Hint: none

- [x] `torsh-tensor`: `src/lazy_loading.rs:447` — `_header_str` is parsed but metadata struct is never populated; deserialize the header string into the actual `TensorMetadata` fields. **DONE 2026-06-21**: real JSON header deserialization populating all `LazyTensorMetadata` fields (was hard-returning shape [100,100]/10000); 4 self-contained parser helpers w/ honest errors on malformed/missing; element_size from canonical `DType::size()`; total_elements validated vs shape; 5 temp-dir tests incl. roundtrip load.
  - Priority: P2 | Scope: small | Hint: none

- [x] `torsh-tensor`: `src/advanced_ops.rs:365` — Autograd Sum operation has no backward pass implementation; add a proper backward that propagates gradients via broadcast. **DONE 2026-06-21**: added `Operation::Sum` variant + broadcast backward; also implemented `Operation::MatMul` backward (grad@rhsᵀ / lhsᵀ@grad). Verified by `test_sum_backward` + `test_matmul_backward` (analytical gradients); torsh-tensor clippy clean, no regression (13 pre-existing pool-lock-poison failures unchanged).
  - Priority: P2 | Scope: small | Hint: none

- [ ] `torsh-tensor`: `src/serialize/data_science.rs:39,89` — Arrow and Parquet serialization return placeholder errors; implement using `arrow-rs` and `parquet` crates.
  - Priority: P2 | Scope: large | Hint: none
  - Locations: `to_arrow()` at :39, `to_parquet()` at :89

- [ ] `torsh-tensor`: `src/serialize/ml_formats.rs:38` — ONNX serialization is a stub; implement using `onnx-rs` or protobuf encoding.
  - Priority: P2 | Scope: large | Hint: none

- [ ] `torsh-jit`: `src/codegen.rs:414` — `generate_kernel` returns an empty placeholder `CompiledKernel`; implement actual Cranelift IR emission for each supported `NodeId` operation type.
  - Priority: P2 | Scope: large | Hint: none

- [x] `torsh-autograd`: `src/meta_gradient.rs:64,119` — `compute_first_order_gradients` returns mock ones-tensors; `compute_second_order_gradients` similarly stubbed; implement real backward/Hessian-vector-product when `AutogradTensor` trait is wired. **DONE** (fabrication removed by 2026-06-20 strict-check; verified 2026-06-21): `compute_first_order_gradients` does a REAL reverse-mode backward (honest error, never mock ones/zeros). Second-order Hessian-vector-product returns an HONEST error pending a non-single-pass tape refactor — genuinely deferred, NOT fabricated.
  - Priority: P2 | Scope: large | Hint: none
  - Locations: first-order :64, second-order :119

- [x] `torsh-autograd`: `src/interactive_debugger.rs:118,122,126,557` — Gradient-norm checking, custom expression evaluation, and step-over/step-out logic all empty; implement each debug command. **DONE 2026-06-21**: real L2 gradient-norm from event/context data (None when absent, no invented value); recursive-descent custom-expression parser w/ honest errors; real step-over/step-out over the recorded event tree. 13 value-asserting tests; clippy clean, debugger 19/19, crate 1109/1110 (only pre-existing adjoint fails).
  - Priority: P2 | Scope: medium | Hint: none

- [x] `torsh-autograd`: `src/flamegraph.rs:527` — `compare_flamegraphs` has no diff logic; implement frame-level comparison (delta time, appeared/disappeared frames). **DONE 2026-06-21**: real `compare()` (frame-path→(self,total) maps, signed deltas, appeared/disappeared sets); test asserts known deltas + exact sets; clippy clean, 7 flamegraph tests pass.
  - Priority: P2 | Scope: small | Hint: none

- [x] `torsh-series`: `src/state_space/particle.rs:465` — Particle smoother backward pass is not implemented; add backward Kalman / two-filter smoother for particle state estimates. **DONE 2026-06-21**: real FFBS backward reweighting (Godsill 2004) w/ Gaussian transition density + log-sum-exp; forward filter now stores particles/weights/transition-means; known-answer test vs analytic Kalman+RTS smoother (matches ~0.01–0.03, verified smoother≠filter); nextest 285/285. (Also fixed a `slice_tensor`/`get_item_flat` offset bug — see new item below.)
  - Priority: P2 | Scope: large | Hint: none

- [x] `torsh-series`: `src/changepoint/mod.rs:81` — PELT changepoint detection uses a simplified O(n²) scan; implement full PELT with optimal partitioning and pruning for O(n log n) complexity. **DONE 2026-06-21**: real optimal partitioning (Jackson 2005) + PELT pruning (Killick 2012), O(1) prefix-sum Gaussian-mean/variance + Laplace costs, BIC penalty; removed a FABRICATION (`CostFunction::KolmogorovSmirnov` secretly computed SSE labeled "KS"); known-answer test detects exactly {50,100}, pruned==un-pruned optimal; nextest 285/285.
  - Priority: P2 | Scope: large | Hint: none

- [x] `torsh-series`: `src/forecast/var.rs:648` — Granger causality F-statistic calculation is a placeholder (returns 0.0); implement proper F-test on restricted vs unrestricted VAR residuals. **DONE 2026-06-21**: real F-test on y-equation RSS (restricted AR vs unrestricted VAR), fixed an all-equations-RSS bug + n≤k underflow guard; tests: caused F=261 vs independent F=0.004; clippy clean, 280 series tests pass.
  - Priority: P2 | Scope: small | Hint: none

- [x] `torsh-vision`: `src/streaming.rs:285,313` — Video frame downscaling in streaming pipeline returns input unchanged; implement nearest/bilinear downscale to the target resolution. **DONE 2026-06-21**: real bilinear downscale (half-pixel-center, edge-clamped), honest error for unsupported ranks; test asserts hand-derived values [2.5,4.5,10.5,12.5] (distinguishes from nearest); clippy clean, 16 streaming tests pass.
  - Priority: P2 | Scope: small | Hint: none
  - Locations: decode path :285, encode path :313

- [ ] `torsh-vision`: `src/feature_detection_advanced.rs:81,92,148` — SuperPoint and Learned-SIFT detectors have no actual neural-network inference; wire to `torsh-nn` forward pass.
  - Priority: P2 | Scope: large | Hint: none
  - Locations: SuperPoint detect :92, Learned-SIFT detect :148

- [x] `torsh-nn`: `src/core/module_ext.rs:200,223,345` — `freeze_parameters` / `unfreeze_parameters` do nothing; `device()` returns None always; implement when `Parameter` exposes requires-grad and device metadata. **DONE 2026-06-21**: root cause was `Parameter.requires_grad: bool` mutated on throwaway clones from `all_named_parameters()` — changed to `Arc<AtomicBool>` (shared like the tensor storage) + `set_requires_grad`; real `device()` from the parameter tensor. Cross-crate verified: workspace build+clippy `--all-features` GREEN, torsh-nn 822 tests pass, 4 new correctness tests.
  - Priority: P2 | Scope: small | Hint: none

- [x] `torsh-sparse`: `src/linalg.rs:1007` — GMRES solver test is `#[ignore]`d due to numerical instability; fix convergence (restart strategy, preconditioning) and re-enable. **DONE 2026-06-21**: root cause was `Tensor::clone()` shares storage (Arc) + `set()` has NO copy-on-write → GMRES overwrote caller's `b`; also Arnoldi breakdown threshold too small for f32. Rewrote GMRES(m) in f64 (MGS+DGKS, incremental Givens, happy-breakdown), never mutating caller tensors; un-ignored + SPD/restart tests w/ exact solutions; nextest 256/256. ⚠️ SAME `clone()+set()` aliasing latent in `conjugate_gradient`:226 & `bicgstab`:318 (see new backlog item).
  - Priority: P2 | Scope: medium | Hint: none

- [x] `torsh-distributed`: `src/communication_scheduler.rs:987` — Tensor serialization for inter-node messaging is a placeholder (`vec![]`); implement proper byte serialization (consider `oxicode` per COOLJAPAN policy). **DONE 2026-06-21**: was `vec![0u8; numel*4]` (silent data loss); now real self-describing LE serializer (magic+version+dtype+shape+raw bytes) + deserializer rejecting empty/truncated/corrupted input; bit-exact round-trip test; clippy clean (default+simd), 334 tests pass.
  - Priority: P2 | Scope: small | Hint: oxicode

- [x] `torsh-backend`: `src/memory_pool.rs:373,381` — `MemoryMappedArray::new()` call has wrong argument count; fix call signature and wire `as_slice()` when memory-mapped path is active. **DONE 2026-06-21** (CRATE CORRECTION: real file is `torsh-tensor/src/memory_pool.rs`, not torsh-backend): real 4-arg signature `new(data, path, mode, offset)`; genuine disk-backed mmap round-trip via `as_slice()`, honest `IoError` on failure; wired `memory_efficient` feature; 3 temp-dir tests; clippy clean (default+feature), nextest 554 pass.
  - Priority: P2 | Scope: small | Hint: none

- [x] `torsh-graph`: `tests/comprehensive_gnn_tests.rs:978` — `memory_efficient` utilities module referenced in tests but never created; implement the module with the expected API. **DONE 2026-06-21**: created `src/utils/memory_efficient.rs` (~610 lines) — real COO `SparseGraph` (from_dense/footprint/density), graph Laplacian (combinatorial + sym-normalized), union-find `adaptive_coarsening`, chunked O(E) neighbor aggregation; fixed 2 latent bugs (footprint=0 on empty, i64-vs-f32). 14 tests cross-checked vs dense reference; clippy clean, nextest 273 pass.
  - Priority: P2 | Scope: medium | Hint: none

- [x] `torsh-sparse`: `src/linalg.rs:226,318` — `conjugate_gradient` and `bicgstab` use the same `let r = b.clone(); r.set(...)` pattern that ALIASES & overwrites the caller's `b` (found 2026-06-21 while fixing GMRES): `Tensor::clone()` shares storage via `Arc` and `Tensor::set()` has NO copy-on-write. **DONE 2026-06-21**: rewrote both with local `Vec<f64>` buffers (caller tensors read-only); also fixed a 2nd latent bug (BiCGSTAB `r_hat=r.clone()` aliasing → rho=0 breakdown). Tests w/ `b`-unchanged canary PROVEN to fail on old code; nextest 257/257, GMRES preserved. Deeper root cause remains: `Tensor::set()` should `make_unique()` (COW) like in-place arithmetic — left as a torsh-tensor follow-up.
  - Priority: P2 | Scope: small | Hint: mirror the GMRES(m) f64 rewrite

- [x] `torsh-tensor`: `Tensor::get_item_flat` calls `storage.get(index)` IGNORING `storage_offset` — so an element read from a `slice_tensor(..)` view returns the BASE tensor's element instead of the slice's (found 2026-06-21 in the particle smoother; worked around there by reading contiguous flat indices). Latent in `torsh-series ParticleFilter::filter`. Fix `get_item_flat` to honor `storage_offset` (and strides) like `compute_flat_index`/`to_vec` do, then drop the workarounds. **DONE 2026-06-23**: honors `storage_offset` + strides (multi-dim → storage_idx translation); `is_contiguous()` also fixed to real row-major stride check; tests verify narrow-view element access.
  - Priority: P2 | Scope: small | Hint: mirror compute_flat_index offset handling

- **NOTE (flagged inconsistency, do NOT auto-edit):** `TODO.md` line 212 (inside `### v0.1.0 Milestone`) currently claims "**🎯 100% Pure Rust (Default Features)**: Zero C/Fortran dependencies in default build". This claim is **currently FALSE**: `ring` (which compiles C and per-arch assembly) is an unconditional default dependency of `torsh-package`, so it is in the default build. Resolving item 1 makes the claim true; until then line 212 should be corrected (e.g. scoped to "except `ring` in torsh-package, pending migration"). Recorded here only — line 212 is intentionally left unedited.

## Stubs to implement (added 2026-06-22 by /cooljapan-stub-check)

This section was produced by a fresh ripgrep sweep (305 raw `TODO|FIXME|HACK|XXX` hits) over `~/work/torsh` (excluding `target/`, `_generated/`, `*.pb.rs`). Noise (license/doc-prose, codegen-emitted Python `# TODO` template strings, `example`/test "re-enable when X exported" comments, `///`/`//!` doc lines, repetitive FFI scipy/pandas/numpy `Fix … compatibility` stubs, and `#[ignore = "… See: TODO.md"]` descriptions) was dropped. Items blocked purely on unbuilt upstream `scirs2-core` GPU/profiling/benchmarking/observability/tensor-core APIs are listed at the bottom as non-actionable. Seed items confirmed by direct inspection are folded in. Note: several of these also appear in the earlier hand-curated list above; they are repeated here in the standard task format for the stub-check pass.

### torsh-functional

- [x] **torsh** `torsh-functional`: `src/attention.rs:518` — `TODO`: `let q_block = query.clone(); // TODO: proper slicing`
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Replace the full-tensor clone with `query.narrow(seq_dim, start_i, end_i - start_i)` (or `index_select`) so each block sees only rows `[start_i..end_i]`.
  - **Risk:** Currently numerically WRONG (every block attends over the whole sequence); fix changes outputs — guard with a reference-vs-naive softmax test.

- [x] **torsh** `torsh-functional`: `src/attention.rs:532` — `TODO`: `let k_block = key.clone(); // TODO: proper slicing`
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Slice key to `[start_j..end_j]` via `narrow`/`index_select`; the causal-mask math below already assumes block-local key ranges.
  - **Risk:** Same correctness bug as :518; verify mask_size alignment after slicing.

- [x] **torsh** `torsh-functional`: `src/attention.rs:533` — `TODO`: `let v_block = value.clone(); // TODO: proper slicing`
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Slice value to `[start_j..end_j]` to match the sliced key block before the weighted-value matmul.
  - **Risk:** Must stay shape-consistent with k_block; off-by-one on end_j corrupts the last block.

- [x] **torsh** `torsh-functional`: `src/linalg/basic.rs:88` — `TODO`: `Use dynamic programming for optimal parenthesization` **DONE 2026-06-21** (verified 2026-06-23): real CLRS O(n³) matrix-chain DP, tests 7500/15125.
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** `chain_matmul` is naive left-to-right; add the classic matrix-chain DP (cost table over dimension list) and multiply in the optimal order.
  - **Risk:** Pure perf/FLOP reduction, result must be numerically identical; cover with an associativity test on non-uniform shapes.

- [x] **torsh** `torsh-functional`: `src/linalg/mod.rs:362` — `TODO`: `Improve eigenvalue decomposition to handle degenerate cases` **DONE 2026-06-21** (verified 2026-06-23): scirs2-linalg eigh/eig + residual gate, degenerate tests pass.
  - **Priority:** P2  **Scope:** medium  **Cross-project:** oxiblas
  - **Approach:** Add a shifted/QR (or Wilkinson-shift) path so repeated/clustered eigenvalues converge instead of stalling on the current iteration.
  - **Risk:** Convergence/ordering changes; test against known degenerate spectra and symmetric matrices.

- [ ] **torsh** `torsh-functional`: `src/utils.rs:348` — `TODO`: `Implement proper in-place operations when tensor mutation is available`
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Blocked on a mutable-tensor API; once `Tensor` exposes in-place mutation, replace the copy-based path with true in-place writes.
  - **Risk:** Blocked-on-API — record only; premature impl would alias shared storage.

- [ ] **torsh** `torsh-functional`: `src/dropout.rs:39` — `TODO`: `Implement inplace operations when available`
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Blocked on in-place tensor mutation; currently returns a fresh tensor. Wire `dropout_` to mutate-in-place once the API exists.
  - **Risk:** Blocked-on-API — record only.

### torsh-tensor

- [x] **torsh** `torsh-tensor`: `src/shape_ops.rs:728` — `TODO`: `Implement efficient expansion with strided views` **DONE 2026-06-21** (verified 2026-06-23): zero-copy stride-0 view expand, no-duplication test.
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** `expand()` materializes a full copy; implement a zero-copy broadcast view using stride-0 on the expanded axes.
  - **Risk:** Stride-0 views interact with contiguity assumptions elsewhere; ensure `contiguous()` / writers materialize before mutation.

- [x] **torsh** `torsh-tensor`: `src/advanced_ops.rs:365` — `TODO`: `Add proper Sum operation for autograd backward pass` **DONE 2026-06-21** (verified 2026-06-23): Operation::Sum backward + MatMul backward.
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** `sum()` sets `requires_grad` but registers no grad fn; register a Sum op whose backward broadcasts the upstream gradient back to input shape.
  - **Risk:** Missing node silently breaks gradients for any graph through `sum()`; add a gradcheck.

- [x] **torsh** `torsh-tensor`: `src/lazy_loading.rs:447` — `TODO`: `Deserialize _header_str into actual metadata` **DONE 2026-06-21** (verified 2026-06-23): real JSON header deserialization, 5 roundtrip tests.
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Parse the safetensors JSON `_header_str` (shape/dtype/data_offsets) instead of returning the hardcoded 100x100/f32 placeholder metadata.
  - **Risk:** Wrong shape/offset corrupts every lazily loaded tensor; test against a real safetensors header.

- [x] **torsh** `torsh-tensor`: `src/convenience.rs:142` — `TODO`: `Add actual stride checking when stride information is available` **DONE 2026-06-23**: real row-major stride check; scalar tensors always contiguous.
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** `is_contiguous()` unconditionally returns `true`; compute expected row-major strides and compare to the tensor's actual strides.
  - **Risk:** Correctness — false `true` makes `contiguous()` skip needed copies for non-contiguous tensors.

- [x] **torsh** `torsh-tensor`: `src/conv.rs:128` — `TODO`: `implement efficient broadcasting` **DONE 2026-06-21** (verified 2026-06-23): add_channel_bias via chunks_mut, 27/27 tests.
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Conv1d bias add is a manual triple loop over to_vec'd data; replace with a broadcasting add over the channel dim.
  - **Risk:** Functional today; change is perf/cleanliness — keep numeric parity.

### torsh-autograd

- [x] **torsh** `torsh-autograd`: `src/stochastic_graphs.rs:237` — `TODO`: `Replace with proper tensor comparison when available` **DONE 2026-06-23**: `uniform.lt(probs)` → Tensor<bool> → where_tensor → {0.0,1.0}; binary/boundary/mean tests pass.
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Bernoulli `sample()` returns the raw uniform tensor instead of `(uniform < probs)`; implement an element-wise less-than to produce the 0/1 mask.
  - **Risk:** Numerically WRONG — current output is not a Bernoulli draw; downstream samplers/log_prob are inconsistent.

- [x] **torsh** `torsh-autograd`: `src/flamegraph.rs:527` — `TODO`: `Implement detailed comparison logic` **DONE 2026-06-21** (verified 2026-06-23): real frame-level diff with signed deltas, appeared/disappeared sets.
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** `compare_flamegraphs` has no diff; compute per-frame deltas (added/removed/changed self+total time) between two captures.
  - **Risk:** Tooling only; low blast radius.

- [x] **torsh** `torsh-autograd`: `src/interactive_debugger.rs:118` — `TODO`: `Implement gradient norm checking` **DONE 2026-06-21** (verified 2026-06-23): real L2 gradient-norm from event/context metadata.
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Wire the debugger's grad-norm command (and the duplicate at :122) to actually compute and report the L2 norm of the inspected gradient.
  - **Risk:** Debug-only; ensure it handles missing/None grads gracefully.

- [x] **torsh** `torsh-autograd`: `src/interactive_debugger.rs:126` — `TODO`: `Implement custom expression evaluation` **DONE 2026-06-21** (verified 2026-06-23): recursive-descent expression parser with full comparison operators.
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Add a small expression evaluator over named tensors/grads for the debugger's `eval` command.
  - **Risk:** Parser/eval surface — sandbox to read-only tensor access.

- [x] **torsh** `torsh-autograd`: `src/interactive_debugger.rs:557` — `TODO`: `Implement step over/out logic` **DONE 2026-06-21** (verified 2026-06-23): real step-over/step-out via event tree traversal.
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Implement step-over / step-out traversal over the backward graph frames (currently single-step only).
  - **Risk:** Debug-only; guard against cycles in the graph walk.

### torsh-series

- [x] **torsh** `torsh-series`: `src/state_space/particle.rs:465` — `TODO`: `Implement backward pass for particle smoothing` **DONE 2026-06-21** (verified 2026-06-23): FFBSi backward smoother via ffbs_backward().
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Smoother currently returns the forward filter unchanged; add an FFBSi (forward-filter backward-simulation) backward sweep over stored particles/weights.
  - **Risk:** Numerically WRONG smoothed estimates today; validate against a linear-Gaussian model where RTS smoother gives ground truth.

- [x] **torsh** `torsh-series`: `src/forecast/var.rs:648` — `TODO`: `Proper F-statistic calculation when VAR implementation is complete` **DONE 2026-06-21** (verified 2026-06-23): real Granger F-stat formula (RSS_r-RSS_u)/q / (RSS_u/df).
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Granger test falls back to `f_stat = 1.0` in the degenerate branch; compute F from restricted/unrestricted RSS with correct dof for all cases (handle near-zero RSS explicitly rather than substituting 1.0).
  - **Risk:** Placeholder F=1.0 yields meaningless p-values; cover with a known causal/non-causal pair.

- [x] **torsh** `torsh-series`: `src/changepoint/mod.rs:81` — `TODO`: `Implement full PELT with optimal partitioning when scirs2-series available` **DONE 2026-06-21** (verified 2026-06-23): PELT pruning via Killick inequality candidates.retain.
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** A working DP already exists but without PELT pruning; add the inequality-based pruning of the candidate set `r` to reach the expected near-linear cost.
  - **Risk:** Pruning bugs drop valid changepoints; test detected set/scores equal the unpruned DP on synthetic step series.

- [ ] **torsh** `torsh-series`: `src/forecast/deep.rs:124` — `TODO`: `Implement training loop when full autograd system is available`
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Deep forecaster `fit` is a no-op; implement the train loop (forward/loss/backward/step) once autograd backward over the model is wired.
  - **Risk:** Partially blocked on autograd backward integration; verify loss decreases on a toy series.

### torsh-cluster

- [x] **torsh** `torsh-cluster`: `src/utils/parallel.rs:215` — `TODO`: `Optimize inertia computation with proper parallel_map_reduce` **DONE 2026-06-23**: parallel_map inertia reduction; matches serial reference within 1e-4.
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Inertia is summed in a serial `for` loop; replace with a parallel map-reduce (scirs2-core `simd_*`/par iterators are already in use just above) accumulating per-sample squared distances.
  - **Risk:** Float reduction order changes inertia in the last ULPs; use a tolerant assert in tests.

### torsh-jit

- [ ] **torsh** `torsh-jit`: `src/codegen.rs:414` — `TODO`: `Implement actual Cranelift code generation`
  - **Priority:** P2  **Scope:** large  **Cross-project:** none
  - **Approach:** `generate_kernel` returns a `CompiledKernel` with empty `code: vec![]`; build real Cranelift IR from the node list (lower ops, emit a callable function, populate inputs/outputs/metadata).
  - **Risk:** JIT is currently non-functional (placeholder); large effort — gate behind tests that execute a generated kernel and compare to interpreter results.

### torsh-vision

- [x] **torsh** `torsh-vision`: `src/streaming.rs:285` — `TODO`: `Implement actual downscaling` **DONE 2026-06-21** (verified 2026-06-23): downscale_frame_bilinear shared helper.
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Adaptive-degradation downscale (decode path) returns the original frame; implement a bilinear resize to the target resolution so load actually drops.
  - **Risk:** Correctness/perf — without it adaptive degradation is a no-op; verify output dims match target.

- [x] **torsh** `torsh-vision`: `src/streaming.rs:313` — `TODO`: `Implement downscaling` **DONE 2026-06-21** (verified 2026-06-23): same bilinear helper reused on encode path.
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Same no-op downscale on the encode/secondary path; share the bilinear resize helper added for :285.
  - **Risk:** Same as :285; keep the two paths consistent.

- [ ] **torsh** `torsh-vision`: `src/feature_detection_advanced.rs:92` — `TODO`: `Implement SuperPoint detection using torsh-nn`
  - **Priority:** P2  **Scope:** large  **Cross-project:** none
  - **Approach:** SuperPoint detector (and the integration point at :81) has no NN inference; wire a `torsh-nn` forward pass producing keypoint heatmap + descriptors.
  - **Risk:** Large; needs a model definition + weights path. Returns empty/placeholder keypoints today.

- [ ] **torsh** `torsh-vision`: `src/feature_detection_advanced.rs:148` — `TODO`: `Implement Learned SIFT detection`
  - **Priority:** P2  **Scope:** large  **Cross-project:** none
  - **Approach:** Learned-SIFT path is a stub; implement the learned descriptor/detector forward via torsh-nn.
  - **Risk:** Large; shares model-loading infra with SuperPoint.

- [ ] **torsh** `torsh-vision`: `src/feature_detection_advanced.rs:263` — `TODO`: `Implement full transformer-style attention with learned parameters`
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Attention-based matcher uses a simplified path; implement multi-head attention with learned projections (depends on the attention slicing fix in torsh-functional).
  - **Risk:** Couples to the attention.rs correctness fixes; validate matching quality on a known pair.

### torsh-nn

- [x] **torsh** `torsh-nn`: `src/core/module_ext.rs:200` — `TODO`: `Implement actual freezing when Parameter supports it` **DONE 2026-06-21** (verified 2026-06-23): freeze/unfreeze via set_requires_grad on Arc<AtomicBool>.
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** `freeze_parameters` (and `unfreeze` at :223) is a no-op; flip each `Parameter`'s requires-grad once the API exposes mutation.
  - **Risk:** Partially blocked on Parameter API; silently fails to freeze today — add a test asserting grad is None after freeze.

- [x] **torsh** `torsh-nn`: `src/core/module_ext.rs:345` — `TODO`: `Implement when Parameter exposes device information` **DONE 2026-06-21** (verified 2026-06-23): device() reads from first parameter's tensor device.
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** `device()` always returns None; return the parameter's device once `Parameter` exposes it.
  - **Risk:** Blocked on Parameter metadata; low risk.

### torsh-sparse

- [x] **torsh** `torsh-sparse`: `src/linalg.rs:1007` — `FIXME`: `#[ignore] GMRES implementation needs numerical refinement` **DONE 2026-06-21** (verified 2026-06-23): BiCGSTAB/CG rewritten with Vec<f64> buffers, ignore removed.
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** GMRES test is `#[ignore]`d for instability; add restart (GMRES(m)) and/or preconditioning, then re-enable the test.
  - **Risk:** Correctness/convergence; validate residual norm decreases monotonically on an SPD system.

### torsh-distributed

- [x] **torsh** `torsh-distributed`: `src/communication_scheduler.rs:987` — `TODO`: `Implement proper tensor serialization` **DONE 2026-06-21** (verified 2026-06-23): real self-describing wire encoder via serialize_tensor_le (magic+dtype+shape+LE data).
  - **Priority:** P2  **Scope:** small  **Cross-project:** oxicode
  - **Approach:** Inter-node tensor payload is a placeholder; serialize tensor bytes/metadata using `oxicode` (COOLJAPAN policy — never bincode).
  - **Risk:** Wire-format must round-trip shape/dtype; cover with a serialize/deserialize equality test.

### torsh-tensor (storage / memory-map)

- [x] **torsh** `torsh-tensor`: `src/memory_pool.rs:373` — `TODO`: `Fix MemoryMappedArray::new() call - requires 4 arguments` **DONE 2026-06-21** (verified 2026-06-23): MemoryMappedArray::new() arity fixed, as_slice() wired.
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** The mmap allocation call has the wrong arity and the result isn't used; fix the `MemoryMappedArray::new(...)` signature and wire `_mmap_array.as_slice()` (:381) into the pool.
  - **Risk:** Compile/feature-gated path; ensure it only activates under the mmap feature.

### torsh-graph

- [x] **torsh** `torsh-graph`: `tests/comprehensive_gnn_tests.rs:978` — `TODO`: `Implement memory_efficient utilities module` **DONE 2026-06-21** (verified 2026-06-23): memory_efficient.rs 720 lines, SparseGraph+Laplacian+coarsening, 14 tests.
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Tests reference a `memory_efficient` utilities module that does not exist; create the module exposing the API the tests expect (e.g. chunked/streaming message passing).
  - **Risk:** Defines new public surface; keep the API minimal and test-driven.

- [ ] **torsh** `torsh-graph`: `src/data.rs:67` — `TODO`: `Implement when scirs2_graph API is stable`
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Graph data conversion stub awaiting scirs2_graph; implement the conversion using whatever stable subset exists, else keep but track.
  - **Risk:** Partially upstream-dependent; verify against a small graph.

### torsh-core (platform detection)

- [ ] **torsh** `torsh-core`: `src/storage/numa.rs:262` — `TODO`: `Implement Windows NUMA detection using GetNumaNodeProcessorMask`
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Windows NUMA node detection is unimplemented; call `GetNumaHighestNodeNumber`/`GetNumaNodeProcessorMask` via the Windows API (feature/cfg-gated to `windows`).
  - **Risk:** Platform-specific, hard to CI on non-Windows; gate and unit-test the parsing logic.

- [ ] **torsh** `torsh-core`: `src/storage/numa.rs:281` — `TODO`: `Implement CPU affinity detection`
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** CPU affinity detection returns a default; query the OS affinity mask per platform.
  - **Risk:** Platform-specific; provide a safe fallback when unavailable.

### Known external/upstream-blocked placeholders (not actionable)

These are gated on unbuilt/unstable upstream APIs (chiefly `scirs2-core` GPU / profiling / benchmarking / observability / tensor-core modules, unstable Rust intrinsics, or absent `mpi`/`cust`/`NCCL` bindings). Recorded for visibility; do NOT pick up as tasks until the upstream surface exists.

- crates/torsh-core/src/simd_arm.rs:248 — `vdotq_s32` intrinsic not yet stable in Rust.
- crates/torsh-core/src/backend_detection.rs:567,574 — scirs2-core gpu opencl/vulkan integration not available.
- crates/torsh-core/src/dtype/traits.rs:262,329 — f16/bf16 FloatElement need scirs2_core::Float impl for half types.
- crates/torsh-core/src/cpu/numa_enhanced.rs:402 ; crates/torsh-core/src/cpu/memory.rs:527 — blocked on unstable std feature (rust issue #117217).
- crates/torsh-autograd/src/grad_mode.rs:650,668,678 — gradient clipping disabled pending tensor/scirs2 integration.
- crates/torsh-autograd/src/blas_integration.rs:704 — register other BLAS providers when available.
- crates/torsh-autograd/src/scirs2_integration.rs:142 — re-enable when SciRS2 API stabilizes.
- crates/torsh-autograd/src/hyperparameter_optimization.rs:250,273,284 — gradient/second-order computation pending autograd API (`backward_single`).
- crates/torsh-series/src/frequency/mod.rs:122,141,434 — scirs2-signal FFT/IFFT/cross-spectral not available (OxiFFT candidate once exposed).
- crates/torsh-signal/src/performance.rs:14 ; wavelets.rs:288,323,339,1070 — scirs2-signal parallel ops / WPT / lifting scheme APIs not stable.
- crates/torsh-backend/src/lib.rs:455 ; memory_defrag.rs:1036 ; zero_copy.rs:757,1070,1109,1122 — scirs2 ROCm / scirs2_cuda memory ops not available.
- crates/torsh-backend/src/cuda/tensor_cores.rs:14,408,520 ; cuda/kernels/mod.rs:438 ; cuda/kernels/tensor_ops.rs:9 ; cuda/buffer.rs:295 — scirs2_core::gpu / cust Module/Function support absent.
- crates/torsh-backend/src/webgpu/{kernels.rs:24,buffer.rs:380,device.rs:1044,backend.rs:413,437,460,896,997} — backend/RNN/Quantization traits not defined yet.
- crates/torsh-backend/src/metal/{buffer.rs:4,313,device.rs:5,156} — BackendStorage/BackendDevice traits absent in current API.
- crates/torsh-backend/src/memory_profiler/mod.rs:155 — types must come from a scirs2-* sub-crate.
- crates/torsh-distributed/src/tensor_parallel.rs:23,467,482,557,598,623,633 — scirs2_core features (AdaptiveChunking, GlobalBufferPool, mmap tensor) not available.
- crates/torsh-distributed/src/metrics.rs:19,928,934,960,993,996,1002,1010,1013,1016,1023 — scirs2_core profiling/benchmarking/observability modules not available.
- crates/torsh-distributed/src/backend.rs:718,937,947,955 — mpi barrier / NCCL communicator bindings not available.
- crates/torsh-tensor/src/math_ops.rs:45,60,64,1207,1217,1227,1237 ; advanced_ops.rs:922,932,949,968,978,990 — scirs2_core gpu/profiling and "actual SciRS2 backend" integration pending.
- crates/torsh-tensor/src/scirs2_stats_integration.rs:93,180,225,274,357,437,559 — scirs2-stats descriptive/correlation/t-test/regression/distribution APIs not stable.
- crates/torsh-tensor/src/backend_integration.rs:14,18,23,715,718,755,762,792,810,814,821,984,990 — scirs2_core GPU backends / GpuDataType / tensor_cores not available.
- crates/torsh-tensor/src/advanced_simd_ops.rs:207,387 — chunk_config args for parallel_map_collect/reduce not yet supported upstream.
- crates/torsh-tensor/src/hardware_accelerators.rs:1189,1221,1253 ; hardware_accelerators_specialized.rs:63,239 — vendor CPU/GPU accelerator + detection-result APIs not expanded.
- crates/torsh-tensor/src/core_ops/types.rs:587 ; lib.rs:229,266 ; lib_new.rs:192 — backend types / CUDA device / AutogradTensor not yet available.
- crates/torsh-tensor/src/serialize/{data_science.rs:39,89,scientific.rs:94,210,232,236,ml_formats.rs:38} — Arrow/Parquet/ONNX + HDF5 string-metadata APIs pending (note: arrow/parquet/onnx must route through COOLJAPAN-approved crates, not arrow-rs/parquet-rs directly).
- crates/torsh-nn/src/hardware_opts.rs:354,372,376,406,410,440,444 — AVX-512/AVX2/NEON tiled matmul via scirs2_core::simd_ops not exposed.
- crates/torsh-functional/src/profiling/{core.rs:203,regression.rs:149} — CPU-utilization / memory detection need scirs2 profiling.
- crates/torsh-python/src/tensor/core.rs:998 — full norm_lp blocked on ops module exposure (p/dim/keepdim currently ignored).

## Stubs to implement (added 2026-07-03 by /stub-check)

- [ ] crates/torsh/src/lib.rs: crates/torsh/src/lib.rs:176 — TODO: Implement ShapeBuilder when available
  - **Approach:** torsh_core::shape::ShapeBuilder does NOT exist anywhere in torsh-core/src (confirmed via direct grep and git log -S search across history — no hits). It genuinely needs to be implemented, not just re-exported. BONUS finding: crates/torsh-core/fuzz/fuzz_targets/fuzz_shape_creation.rs ALREADY references torsh_core::shape::{Shape, ShapeBuilder} and calls ShapeBuilder::new() — this fuzz target would fail `cargo fuzz build` today. Implementing ShapeBuilder in torsh-core fixes both this TODO and that fuzz target simultaneously.
  - **Scope:** medium
  - **Prerequisites:** none
  - **Risk:** Low — currently just an inert comment; risk is the fuzz target silently bit-rotting since fuzz crates are typically excluded from normal workspace builds/CI.

- [ ] crates/torsh/src/lib.rs: crates/torsh/src/lib.rs:552 — TODO: Re-enable when tensor ops module is available
  - **Approach:** NEEDS_CLARIFICATION — stated reason is stale (torsh_tensor::ops confirmed to exist now: real module directory ops/manipulation/{dim_ops,core_ops}.rs, ops/simd/f32_ops.rs, plus an ops.rs.backup showing it was refactored from a single file). The LIKELY REAL blocker (unstated in the comment) is glob-import ambiguity: the very next block re-exports crate::nn::functional::* under #[cfg(feature="nn")], and an adjacent comment explicitly says explicit PascalCase aliases were added "to avoid ambiguous glob imports" with nn::functional's lowercase names (relu, sigmoid, etc.) — re-enabling `pub use crate::tensor::ops::*;` as a second unscoped glob would likely reintroduce that collision. Needs a maintainer decision: re-export tensor::ops with an explicit/aliased list (not a glob) to dodge the collision, or update the comment to state the real (collision) reason and leave permanently disabled by design.
  - **Scope:** medium
  - **Prerequisites:** none
  - **Risk:** Low — inert comment today; main risk is that "when tensor ops module is available" reads as an open blocker to future contributors when the module has in fact been available for a while.

- [ ] examples/distributed_gradient_sync.rs: examples/distributed_gradient_sync.rs:187 — TODO: Backward pass would go here when autograd is fully integrated
  - **Approach:** LIKELY STALE — Tensor::backward() is confirmed fully implemented and documented (torsh-tensor core_ops/types.rs:1448). DDP-specific autograd wiring (whether ddp.forward()'s output stays on the tracked graph through to loss.mean()) was NOT independently re-verified — needs a quick re-test rather than being assumed still-blocked. Example currently fabricates fake gradients via randn() instead of calling the commented-out loss.backward()?. Try re-enabling now that Tensor::backward() exists; if it works, delete the fake-gradient workaround.
  - **Scope:** small
  - **Prerequisites:** none (verify DDP autograd wiring first)
  - **Risk:** Low — example-only code, no library impact. Misleading to readers if left stale.

- [ ] examples/gradient_checkpointing.rs: examples/gradient_checkpointing.rs:6 — TODO: Re-enable when checkpoint functions are properly exported
  - **Approach:** STALE reasoning, but not simply an "export" fix — confirmed NONE of the referenced free functions (auto_checkpoint_sequence, checkpoint, checkpoint_sequential, configure_checkpointing_with_strategy, get_checkpoint_memory_stats) exist anywhere in torsh-autograd under those names. The module was redesigned: torsh-autograd/src/checkpoint_scheduler.rs now provides a struct-based API (CheckpointScheduler, IntegratedCheckpointScheduler, CheckpointConfig, and a CheckpointStrategy enum that does still exist) with methods like record_operation/force_checkpoint/get_stats/process_operation instead of free functions. This isn't a missing pub-use — the whole example needs PORTING to the new API shape.
  - **Scope:** medium
  - **Prerequisites:** none
  - **Risk:** Low (example-only), but misleading to a future maintainer who might just try uncommenting the use block and find it doesn't compile.

- [ ] examples/gradient_checkpointing.rs: examples/gradient_checkpointing.rs:22 — TODO: Re-enable when checkpoint functions are properly exported
  - **Approach:** Duplicate marker for the same large commented-out block as the item above (examples 1-4 in that file, lines ~24-39) — not a separate piece of work, resolve together.
  - **Scope:** medium
  - **Prerequisites:** examples/gradient_checkpointing.rs:6 (same commented-out block)
  - **Risk:** Low — example-only.

- [ ] examples/gradient_checkpointing.rs: examples/gradient_checkpointing.rs:170 — // let stats = get_checkpoint_memory_stats();
  - **Approach:** Same root cause as examples/gradient_checkpointing.rs:6 — get_checkpoint_memory_stats() doesn't exist; nearest equivalent is CheckpointScheduler::get_stats()/IntegratedCheckpointScheduler::stats(). memory_statistics_example() currently hardcodes all-zero output (0,0,0,0.0) instead of calling any real stats API. Part of the same example-porting task.
  - **Scope:** medium
  - **Prerequisites:** examples/gradient_checkpointing.rs:6 (same commented-out block)
  - **Risk:** Low — example-only; prints fake zeros which could mislead a reader into thinking checkpointing has no measurable effect.

## Policy Check Findings (added 2026-07-04 by /policy-check)

### Real Tier-A policy violations (need a decision, not a blind fix)

- [ ] **`aws-lc-sys` real FFI crypto code compiling into the workspace.** `cargo tree --workspace -e normal,build` shows `aws-lc-sys v0.42.0` (via `aws-lc-rs v1.17.1` ← `rustls v0.23.41`) actually compiling in — this is real C/assembly cryptography (AWS-LC, a BoringSSL derivative), not a hypothetical. Root cause: `reqwest 0.13`'s rustls-tls feature selects `aws-lc-rs` as its default `CryptoProvider`; none of the 6 dependent crates (`torsh-cli`, `torsh-hub`, `torsh-text`, `torsh-utils`, `torsh-vision`, and `torsh-data` via `ureq`) override it. Per COOLJAPAN Pure Rust Policy, this should be `oxicrypto-*`/`oxitls-*` instead.
  - **Approach:** investigate whether `reqwest`/`rustls` can be configured to use a pure-Rust `CryptoProvider` (e.g. `rustls`'s own `ring`-free / process-default-provider APIs, or migrating the TLS stack to `oxitls-core` + `oxitls-adapter-rustls-rustcrypto` per the noffi ecosystem — noting `~/work/.pure-rust-governance.md`'s own caveat that this adapter is currently alpha/"DO NOT USE IN PRODUCTION" upstream, so may need `oxitls-adapter-aws-lc` as an explicit, intentionally-quarantined dependency instead if production-readiness matters more than purity right now).
  - **Scope:** medium — touches 6 crates' TLS-transitive dependency configuration, needs testing against real HTTP calls (hub download/upload, dataset download, pretrained model fetch).
  - **Risk:** getting this wrong could break real network functionality in torsh-cli/torsh-hub; needs careful testing, not a blind swap.

- [ ] **`cudnn-sys` feature-gated on `torsh-core`, violating the quarantine model.** `crates/torsh-core/Cargo.toml:118` has `cudnn-sys = { version = "0.0.3", optional = true }` under a `cfg(target_arch = "x86_64", target_os = "linux"/"windows")` target-specific dependency, gated behind a `cudnn` feature at `Cargo.toml:72`. This doesn't show up in `cargo tree` on macOS (this dev machine), but is confirmed real via `Cargo.lock`. Per `~/work/.pure-rust-governance.md` v2 §5, the "quarantine model" explicitly forbids a pure/Role-A crate (torsh-core is a math-kernel crate, never Role B) from having a feature that pulls in `-sys`/FFI when enabled — irreducible FFI must live in a separate, suffix-named quarantine crate instead, excluded from the pure-set.
  - **Approach:** split cuDNN bindings out of `torsh-core` into a separate crate (e.g. `torsh-core-cudnn` or similar suffix-named quarantine crate) that depends on `torsh-core` rather than the reverse, following the governance doc's prescribed pattern.
  - **Scope:** medium — needs care since `torsh-core` is foundational; must not break existing CUDA-feature consumers.
  - **Risk:** low if done as a pure refactor (moving code, not changing behavior), but needs Linux/Windows CI or hardware to properly verify (can't be verified on this macOS dev machine).

- [ ] **`optirs`/`optirs-core` (a sibling COOLJAPAN crate) needs a version bump + publish.** Root `Cargo.toml:154-155` pins `optirs`/`optirs-core = "0.3.1"`. This published version still pins `scirs2-optimize ^0.4`, dragging a whole parallel 0.4.4-generation of `scirs2-core`/`scirs2-linalg`/`scirs2-metrics`/`scirs2-neural`/`scirs2-optimize`/`scirs2-sparse`/`scirs2-stats` (each duplicated against torsh's own direct 0.6.0 deps) plus old `nalgebra`/`simba`/`oxiarc-core`/`oxiarc-lz4`/`oxiarc-zstd` versions into the dependency graph — 9 of the 12 total version-duplicate clusters found workspace-wide trace back to this single cause. **Confirmed the fix already exists**: `~/work/optirs`'s local `Cargo.toml` already declares `scirs2-core = "0.6.0"` etc. (git log shows "bump scirs2" commits), but the crate's own `version` field is still `"0.3.1"` — the bump hasn't been published to crates.io yet.
  - **Approach:** bump the version and `cargo publish` (real publish, not dry-run) from `~/work/optirs`, then update torsh's root `Cargo.toml:154-155` pin to the new version. **This requires explicit user approval before publishing anything** (per the standing "never cargo publish without explicit permission" and "ask User before updating another PJ's codebase" policies) — do not action without asking first.
  - **Scope:** small once approved (a version bump + publish + one Cargo.toml pin update in torsh).
  - **Risk:** low — collapses duplicate dependency chains, no expected behavior change.

### Confirmed non-issues (closing the loop, no action needed)

- [x] **`model compress --algorithm gzip` is NOT a compression-policy violation — confirmed to be an unimplemented CLI stub.** `crates/torsh-cli/src/commands/model/conversion.rs`'s `compress_model()` never calls any codec at all: it sleeps 1s, writes the literal string `"compressed model data"` to the output path, and reports a hardcoded fake `compression_ratio: 0.75`. `torsh-cli/Cargo.toml` has zero compression dependencies. This is part of the already-known, already-deferred torsh-cli mock-surface (see this file's existing CLI-related follow-up entries and `crates/torsh-cli/TODO.md`'s own "Proposed follow-ups") — no new action needed here, just noting it's confirmed non-violating rather than an open question.

### Housekeeping (low-risk cleanup, needs a human look before deleting)

- [ ] **Dead/stale files possibly safe to delete** (not confirmed 100% unreferenced — verify with a repo-wide reference search, e.g. `include!()`/build.rs, before removing):
  - `crates/torsh-cli/src/commands/model.rs.backup` and `crates/torsh-cli/src/commands/model/mod.rs.old` — leftover backup files next to the real `crates/torsh-cli/src/commands/model/mod.rs`.
  - `crates/torsh-benches/src/metrics_original_1727_lines.rs`, `crates/torsh-backend/src/cpu/platform_optimization_original_1706_lines.rs`, `crates/torsh-distributed/src/store_original_1506_lines.rs` — apparent orphaned leftovers from a prior `splitrs` run (no `mod` declaration referencing any of these three found in their respective crates).

### Workspace hygiene (mechanical, well-scoped, good candidate for a dedicated future pass)

- [ ] **20 of 32 crates have hardcoded dependency versions instead of `dep.workspace = true`.** Full detail (which crate, which dep, which line) was captured by the policy-check subagent and should be re-derived fresh when this is picked up (`grep -E '^\w[\w-]* = "[0-9]'` across every `crates/*/Cargo.toml`) rather than copied stale into this TODO — but headline examples worth naming as high-value quick wins: 3 exact duplicates of an existing root `workspace.dependencies` entry (`tokio-test` in torsh-hub, `hdf5` in torsh-sparse, `cudnn-sys` in torsh-core — trivial, just add `.workspace = true`), plus consolidation candidates repeated across many crates without a root entry yet (`num_cpus` in 8 crates, `dirs` in 5 crates with a "6.0" vs "6.0.0" string inconsistency, `walkdir`/`fastrand`/`indicatif`/`pyo3-build-config` each in 3 crates).
  - **Approach:** promote each repeated dependency to root `[workspace.dependencies]` once at its current best version, then switch every consuming crate to `dep.workspace = true`.
  - **Scope:** large (touches ~20 Cargo.toml files) but very low risk (no logic changes, just dependency declaration style) — good candidate for a dedicated, mechanical future pass.
  - **Risk:** minimal; verify with a full workspace build after.

- [x] **Version-drift clusters not fixable from torsh's own Cargo.toml** (informational only — third-party crates own these, no action possible within this repo): `approx` (0.3.2/0.5.1, via `sprs`→`alga`), `base64` (0.13.1/0.22.1, via `tokenizers`→`spm_precompiled`), `block-buffer` (0.10.4/0.12.1, via `ed25519-dalek` lagging `aes-gcm`'s RustCrypto generation). Also: transitive `flate2`/`miniz_oxide`/`rustfft` leak in via `image`/`imageproc`/`ureq`/`backtrace` (used by torsh-data/torsh-vision/torsh-profiler) — not directly fixable without upstream changes to those crates, a soft self-containment gap rather than an actionable violation.

## Release Check Findings (added 2026-07-04 by /release-check)

### Unused dependencies (needs per-dependency human triage, not a blind removal pass)

- [ ] **`cargo +nightly udeps --all-targets --all-features` flagged 25 of 32 crates with at least one unused dependency.** The tool's own caveat applies: it cannot detect usage in doc-tests, and several flagged items are plausibly platform/feature-gated code the tool's current feature combination doesn't compile (e.g. `cudarc` in torsh-sparse, `objc2-metal-performance-shaders`/`raw-cpuid` in torsh-backend) rather than genuinely dead — each needs individual verification before removal, not a bulk delete. Cross-cutting patterns worth attention: `scirs2-autograd` flagged unused in 9 crates, `anyhow` in 6, `criterion` (dev-dep) in 5, `torsh-autograd` in 4, `tracing` in 3. Full per-crate list (paste verbatim, this is the complete finding, re-derive fresh if picked up rather than trusting it's still accurate by then):
  - torsh-autograd: anyhow
  - torsh-backend: anyhow, futures, objc2-metal-performance-shaders, raw-cpuid, sha2, spin
  - torsh-cli: scirs2-metrics (dep); assert_cmd, predicates (dev-dep)
  - torsh-cluster: scirs2-cluster, scirs2-linalg, scirs2-metrics, scirs2-stats
  - torsh-core: anyhow, num-derive, tracing (dep); libfuzzer-sys, proptest (dev-dep)
  - torsh-data: futures, js-sys, wasm-bindgen, web-sys
  - torsh-functional: scirs2-autograd, torsh-backend (dep); criterion (dev-dep)
  - torsh-fx: criterion (dev-dep only)
  - torsh-graph: scirs2-autograd, scirs2-graph, scirs2-spatial, torsh-autograd, torsh-nn
  - torsh-hub: indicatif, scirs2-autograd (dep); tokio-test (dev-dep)
  - torsh-jit: torsh-backend, torsh-tensor, tracing, tracing-subscriber (dep); criterion, proptest (dev-dep)
  - torsh-linalg: scirs2-autograd
  - torsh-metrics: scirs2-metrics (dep); scirs2-autograd (dev-dep)
  - torsh-models: anyhow, torsh-optim
  - torsh-nn: anyhow, oxiarc-deflate, scirs2-cluster, scirs2-neural, torsh-autograd (dep); criterion (dev-dep)
  - **torsh-optim: anyhow, optirs, optirs-core, scirs2-optimize, torsh-autograd** — note `optirs`/`optirs-core` themselves flagged unused here; cross-reference against the "optirs/optirs-core version bump" item in this file's `## Policy Check Findings` section, since if torsh-optim's own code genuinely doesn't use them, that changes the urgency/framing of that pending version-bump decision (though verify it's not false-positive macro/re-export usage first)
  - torsh-package: criterion (dev-dep only)
  - torsh-profiler: torsh-tensor, tracing
  - torsh-series: scirs2-signal, scirs2-stats, torsh-autograd
  - torsh-signal: scirs2-fft, scirs2-signal
  - torsh-sparse: cudarc, sprs
  - torsh-tensor: anyhow, oxiarc-deflate, scirs2-autograd, scirs2-stats, torsh-backend (dep); proptest (dev-dep)
  - torsh-text: scirs2-autograd, torsh-data, torsh-linalg
  - torsh-utils: fs2
  - torsh-vision: dirs, imageproc, oxiarc-archive, scirs2-autograd, scirs2-vision, torsh-data
  - Not flagged (clean): torsh, torsh-benches, torsh-distributed, torsh-ffi, torsh-python, torsh-quantization, torsh-special.
  - **Approach:** for each flagged dependency, verify actual usage (check for feature-gated code paths, macro-only usage, re-exports) before removing; remove genuinely-dead ones from Cargo.toml.
  - **Scope:** large (25 crates) but mechanical once triaged — recommend batching by crate, one subagent per crate or small group.
  - **Risk:** low if properly verified per-dependency first; blind removal risks breaking platform-specific or feature-gated builds this pass's default feature combination didn't exercise.

### Duplicate dependency-version census expanded beyond the previously-documented 12

- [ ] **A fresh `cargo tree --workspace --duplicates` run shows 45 distinct duplicated packages workspace-wide, not the previously-documented 12.** The previously-documented 12-package optirs/oxiarc cluster is unchanged, but the other 33 are ordinary third-party ecosystem generation-churn (e.g. `rand` 0.8/0.9/0.10 three-way, `syn` 1.x/2.x, `thiserror`/`thiserror-impl` 1.x/2.x, `hashbrown` four-way 0.14-0.17, `getrandom`/`rand_core`/`rand_chacha`/`rand_distr` three-way splits, `itertools` three-way, RustCrypto `digest`/`sha2`/`crypto-common` family, `nom`, `gimli`, `darling`/`darling_core`/`darling_macro`, `object`, `core-foundation`, `cpufeatures`, `foldhash`, `num-complex`, `quick-error`, `rustc-hash`, `sysinfo`, `wide`) not directly actionable from torsh's own Cargo.toml. One genuine anomaly worth a maintainer look, not root-caused in this pass: `serde`, `serde_core`, and `winnow` each show up as "duplicates" at identical version strings (1.0.228 / 1.0.228 / 1.0.3 respectively) rather than a real semver split — suggests two non-unified resolution paths (different SourceId/registry path) rather than an actual version conflict.
  - **Approach:** re-run `cargo tree --workspace --duplicates` fresh when picked up (don't trust this snapshot to still be accurate) and root-cause the serde/serde_core/winnow identical-version anomaly specifically; the other 33 packages are third-party-owned churn, informational only.
  - **Scope:** small (just the anomaly) to investigate, informational for the rest.
  - **Risk:** low.

### README.md Roadmap section is stale relative to the actual CHANGELOG

- [ ] **`README.md` lines ~339-367 Roadmap section still marks v0.1.3 as "(Current)" and its v0.2.0 bullet list doesn't reflect what actually shipped.** The `**v0.1.3 (Current)**` marker should no longer say "(Current)" now that 0.2.0 is shipping, and the `**v0.2.0** - *Performance & Polish*` bullet list (generic items like "cuDNN integration", "enhanced distributed training") doesn't reflect what actually shipped in 0.2.0 per the real `## [0.2.0]` CHANGELOG.md entry (pyo3 0.29 migration + Tensor operator overloads, real WebGPU buffer transfers, real autograd hyperparameter gradients, real wavelet transforms, real distributed all_gather/MPI fixes, tensor alignment/mutex-poisoning fixes, CLI unit-conversion/stdout-leak fixes).
  - **Approach:** rewrite the Roadmap's v0.1.3/v0.2.0 entries to match the real CHANGELOG content; this is a content-authoring task, not a mechanical version-string substitution (a mechanical substitution pass already fixed the plain stale `0.1.3` version pins in the install-instructions section separately).
  - **Scope:** small, single-file content edit.
  - **Risk:** low, purely descriptive text.

### 1 real doctest bug found and fixed this pass (informational — already resolved, no action needed)

- [x] **`crates/torsh-ffi/src/c_api/types.rs` (lines 504, 529, 555) had 3 doctest examples using bare `DType`/`TorshDType` with no `use` statement.** Doctests don't inherit the enclosing module's imports — fixed during this release-check pass by adding the correct `use` statements. Also fixed: 8 broken/private rustdoc intra-doc links across `torsh-tensor`, `torsh-signal`, `torsh-nn`, `torsh-sparse`, `torsh-autograd`, `torsh-models`, and `torsh-graph` (links pointing to private items, delinked to plain code spans; one `//!` module-doc scoping quirk in torsh-graph fixed with explicit qualified paths). Marked closed/informational — just recording that it happened, in case a future TODO sweep wonders why these files have small diffs unrelated to the feature work.

### Missing documentation (informational only, out of scope to fix in one pass)

- [ ] **`cargo clippy --all-features -- -W missing_docs` reports 22,186 missing-doc warnings workspace-wide.** Breakdown: 12,977 struct fields, 4,719 enum variants, 1,610 methods, 1,227 associated functions, 972 structs, 233 enums, 170 constants, 105 functions, 67 modules, 57 type aliases, 25 associated types, 13 macros, 8 traits, 2 associated constants, 1 crate. Far too large for a routine pass; if documentation completeness becomes a priority, this would need its own dedicated multi-session effort, likely crate-by-crate.

## Documentation & Code Findings from /readme Pass (added 2026-07-04)

- [ ] **`torsh-autograd`'s `HyperparameterOptimizer` only computes real gradients when a caller explicitly opts out of the default config.** `HyperparameterConfig::default()` sets `second_order: true`, and that code path still calls `compute_second_order_gradient`, an explicit zero-returning placeholder — so by default, the optimizer remains a silent no-op; the real first-order gradient fix from earlier today only activates when a caller explicitly sets `second_order: false`.
  - **Approach:** either implement real second-order (Hessian-vector product) gradients, or change the default to `second_order: false` with a clear doc note about the tradeoff, or make the function return an explicit error/warning when second-order is requested but unimplemented rather than silently returning zero.
  - **Scope:** small (change a default) to large (implement real HVP), depending on which fix is chosen.
  - **Risk:** changing the default could alter behavior for any existing caller relying on `second_order: true`'s current (broken) silent-zero behavior — low likelihood but worth a compatibility note.

- [ ] **`torsh-autograd`'s own internal `src/examples.rs` doesn't exercise real APIs.** Its 13 tests (which `TODO.md` describes as "all examples verified working") compute results by hand (e.g. `basic_gradient_example()` computes `x*x`/`2.0*x` manually rather than calling any real tensor/autograd function; `gradient_clipping_example()` manually norms/scales a `Vec<f32>` instead of calling the crate's real `clip_grad_norm`) with tautological assertions (`assert!(result.is_ok())` on functions that structurally cannot fail).
  - **Approach:** rewrite `examples.rs` to actually call the real autograd API surface, so its tests exercise genuine behavior.
  - **Scope:** medium — one file, but needs care to get the real API calls right.
  - **Risk:** low, this is additive correctness work with no behavior change to production code.

- [ ] **`torsh-python`'s Tensor operator overloads don't support scalar operands.** `tensor + 5` (or any Tensor-scalar arithmetic) still raises `TypeError` — the newly-added `__add__`/`__sub__`/`__mul__`/`__truediv__` only accept another `Tensor`, with no `__radd__`/scalar-promotion path, since the underlying `.add()`/etc. methods themselves don't accept scalars. Also: `sum`/`min`/`flatten`/`argmax`/`argmin` accept `dim`/`keepdim` parameters but silently ignore them (always full-reduce), and tensor-creation functions' `dtype`/`device` parameters are accepted but silently ignored (only `requires_grad` is honored).
  - **Approach:** add scalar-accepting overloads to the underlying Rust `add`/`sub`/`mul`/`div` methods (or a scalar-specific path) and wire `__radd__` etc.; implement real `dim`/`keepdim` reduction logic where currently ignored; implement real `dtype`/`device` handling in tensor creation.
  - **Scope:** medium-large, spans multiple methods across `tensor/core.rs` and `tensor/creation.rs`.
  - **Risk:** low for scalar ops (additive); medium for dim/keepdim and dtype/device since silently-ignored-to-honored is a behavior change existing callers might be relying on (unlikely but possible).

- [ ] **`torsh-python`'s distributed collectives (`all_reduce`/`broadcast`/`scatter`/`gather`) are pure no-op `Ok(())` stubs.** Calling any of them currently succeeds without doing anything.
  - **Approach:** wire to the real `torsh-distributed` primitives (noting the separate, more fundamental finding below about `torsh-distributed`'s own `ProcessGroup` API).
  - **Scope:** medium, blocked on the `torsh-distributed` `ProcessGroup` finding below being resolved first.
  - **Risk:** none currently (already documented as non-functional), but silent no-op success is worse than a loud error — consider making these return an explicit "not implemented" error in the interim rather than a false success.

- [ ] **`torsh-distributed`'s high-level `init_process_group`/`ProcessGroup` API always constructs `MockBackend`, regardless of requested backend type.** This means the real `MpiBackend` fixes made earlier today (real `barrier()`, real `parallel_all_gather` concatenation, the `Universe` lifetime fix) are correct and tested, but unreachable through the standard high-level API most users would call — `create_backend` in `process_group.rs` resolves `Nccl`/`Mpi`/`Gloo` all to mock. `Gloo` additionally has zero real implementation anywhere (no TCP/InfiniBand, silently aliases to mock). The top-level `collectives::all_reduce/broadcast/reduce/send/recv` free functions also have explicitly-commented mock bodies that never touch the tensor.
  - **Approach:** wire `create_backend`'s `Mpi` case to construct the real `MpiBackend` (already correct and tested); decide whether `Gloo` needs a real implementation or should error clearly instead of silently mocking.
  - **Scope:** medium — this is the natural, valuable follow-up to today's low-level MPI fix, making it actually reachable.
  - **Risk:** low if done carefully (the low-level backend is already tested); the main risk is scope creep into implementing Gloo for real, which is a much bigger undertaking and should be split out separately if attempted.

- [ ] **`torsh-functional` has two different `InterpolationMode` enums, and the crate-root re-export resolves to the wrong one for `grid_sample`/`interp2d`.** One enum lives in the `image` module (`Bilinear` variant), another in the `interpolation` module (`Linear` variant) — discovered while verifying README doc examples; worked around in the documentation's import path, but the underlying source-level re-export ambiguity was not fixed and may still confuse real callers who import from the crate root expecting one enum and getting the other.
  - **Approach:** rename one of the two enums to disambiguate (e.g. `ImageInterpolationMode` vs `InterpolationMode`), or make the crate-root re-export explicit/unambiguous with a clear doc comment about which one it is and why two exist.
  - **Scope:** small, but touches a public API name (potentially breaking for any existing caller depending on the current — possibly wrong — re-export target).
  - **Risk:** medium, since it's a public API surface change; needs a decision on which name each enum should keep.

- [ ] **`torsh-hub`'s security/signing feature is placeholder cryptography presented without a clear "not real" caveat until today's README fix.** `security::SecurityManager`'s RSA/Ed25519/ECDSA model-signing are hardcoded placeholder strings, not real cryptographic operations — anyone relying on it to actually verify downloaded-model integrity would have a false sense of security. Also found: `security/sandbox.rs`, `security/signing.rs`, `security/validation.rs` are orphaned dead code (not referenced by any `mod` declaration, logic duplicated in the real `security.rs`). Pretrained-weight factories (ResNet/EfficientNet/ViT/BERT/GPT-2 etc.) silently ignore `pretrained=true` and return random weights with just a log warning — no actual weight download/loading exists. `load_torsh_model()`'s HuggingFace format conversion only covers bert/gpt2/bart/t5, and all 4 conversion directions are literal `Err(NotImplemented)`.
  - **Approach:** for signing — either implement real cryptographic signing (a real, non-trivial security feature) or very clearly gate/label it as non-functional in all user-facing surfaces, not just the README (e.g. runtime warning on use); for pretrained weights — implement real weight downloading, or clearly document the current no-op behavior everywhere it's surfaced (CLI help text, error messages), not just in one README.
  - **Scope:** large for real implementations of either; small for clearer non-functional labeling.
  - **Risk:** the security-signing gap is the more serious one — a security feature that silently does nothing is worse than not having the feature at all, since it creates false confidence. Recommend prioritizing either real implementation or very loud non-functional labeling over leaving it as quiet placeholder code.

- [ ] **`torsh-models` README's code examples use Python-style keyword arguments throughout, which isn't valid Rust and doesn't match real function signatures** (e.g. `resnet18(pretrained=true, num_classes=1000)` when the real signature is `resnet18(num_classes: usize)` with no `pretrained` parameter at all). This is pervasive across most Usage/Tutorial code blocks in that README — fixing it fully means rewriting most of the file's code examples, which was out of scope for a verification-only pass.
  - **Approach:** systematic rewrite of `torsh-models/README.md`'s code examples against real signatures, likely crate-wide since the same drift pattern was found in several other crates (torsh-vision, torsh-sparse, torsh-graph, torsh-fx also had similar issues, already partially fixed today).
  - **Scope:** large (one big doc-rewrite file, but similar work may be needed elsewhere too).
  - **Risk:** none — pure documentation correctness work.

- [ ] **`torsh-profiler`'s README still has 3 of 5 major sections describing a fictional API** (TensorBoard Integration, Advanced Analysis, Multi-GPU Profiling all reference nonexistent types) — only "Basic Profiling" and "Custom Profiling Regions" were fixed today against the real, much lower-level imperative `Profiler` API (`new()`/`start()`/`stop()`/`add_event()`/`get_stats()`, real macros `profile_block!`/`profile_current_function!`).
  - **Approach:** research what real functionality exists (if any) for TensorBoard export, cross-op analysis, and multi-GPU profiling, and either document it accurately or mark these as not-yet-implemented rather than describing fictional APIs.
  - **Scope:** medium, needs per-section investigation.
  - **Risk:** none, pure documentation work, but needs real research to avoid just moving the fabrication elsewhere.

- [ ] **Workspace-policy dependency-declaration violations found during doc verification** (small, mechanical, consistent with the larger "20/32 crates have hardcoded versions" finding already in this file's "Policy Check Findings" section): `torsh-utils/Cargo.toml` pins `base64`, `fs2`, `num_cpus` directly instead of `.workspace = true`.
  - **Approach:** fold into the existing planned workspace-hygiene pass (the "20 of 32 crates have hardcoded dependency versions" item under "Policy Check Findings" → "Workspace hygiene").
  - **Scope:** trivial.
  - **Risk:** none.

## Downstream Dependency Review Findings (added 2026-07-06, from an external trustformers 0.2.0 dependency review)

- [ ] **Publishing 0.2.0 to crates.io is now a concrete unblock for at least one downstream consumer, not just an internal milestone.** The last crates.io release is still `torsh = "0.1.3"` (pinning `scirs2-core 0.5.1`), while this workspace's `Cargo.toml:134` has moved to `scirs2-core = "0.6.0"`. The `trustformers` project (a separate COOLJAPAN ecosystem member) reports it cannot currently adopt torsh as a dependency without pulling in a second, type-incompatible `scirs2` 0.5.x stack alongside its own 0.6.x one — it has deferred a planned `torsh-interop` feature to its own 0.3.x specifically pending a torsh 0.2.0 crates.io release on scirs2 0.6. This doesn't change anything about *how* 0.2.0 should be finished, but it does add external urgency to the existing "no unwrap / zero warnings / all green" release gate already tracked elsewhere in this file — worth factoring into release sequencing/priority.
  - **Approach:** no code change; just release-planning context. When 0.2.0 is otherwise ready and the user explicitly authorizes it, publishing unblocks this downstream consumer.
  - **Scope:** none (informational).
  - **Risk:** none.

- [ ] **No real pure-Rust PyTorch pickle (`.pt`) deserializer exists anywhere in the workspace — three separate partial/placeholder implementations, none of which actually parses the pickle format.** Verified against current HEAD:
  - `torsh-hub/src/lib.rs:1416` — the model-loading path has a bare comment, `// This would require implementing PyTorch pickle format parsing`, with no implementation behind it.
  - `torsh-models/src/utils.rs` — `load`/`save` helpers around lines 163–191, 554, 601–620 are all explicitly self-described as "simplified"/"placeholder": comments read "a full implementation would parse the pickle format properly", "real implementation would use PyTorch's pickle deserialization", "this is a placeholder — real PyTorch loading would parse the pickle format". Its own tests (lines ~729–785) assert the *opposite* of a placeholder file — they check that no literal `"placeholder safetensors file"` / `"placeholder safetensors file with tensor data"` byte strings get written — confirming the placeholder nature is a known, tracked gap rather than an oversight.
  - `torsh-cli/src/commands/model/pytorch_parser.rs` — this crate's `.pt` "parser" is metadata-heuristics only (file/tensor-shape inference), not a real pickle deserializer.
  - This has a concrete, newly-created downstream customer: `trustformers` just removed its `tch` (libtorch FFI) dependency, so a genuine pure-Rust pickle parser in torsh would become the ecosystem's primary PyTorch-interop path rather than an internal nice-to-have.
  - **Approach:** implement a real pure-Rust pickle protocol 0–5 opcode interpreter (`torsh-hub` or a new small `torsh-pickle`-style module) sufficient to reconstruct the `OrderedDict[str, Tensor]` state-dict shape PyTorch `.pt`/`.pth` checkpoints use (storage/tensor `REDUCE` opcodes, `BINPERSID` for storage refs, zip-container `.pt` unwrapping via the existing OxiARC pure-Rust archive stack — not `zip`/`flate2`). This is a substantial, well-scoped IMPLEMENT-POLICY item, not a quick fix.
  - **Scope:** large — a real pickle VM is nontrivial but well-precedented (many pure implementations exist to reference for the opcode set); should be its own dedicated multi-session effort.
  - **Risk:** low to add (purely additive new capability); the risk is scope-creep into full PyTorch tensor-storage/dtype fidelity (fp16/bf16, quantized dtypes, sparse layouts) which should be staged rather than attempted in one pass.

- [ ] **`torsh-distributed`'s default-members exclusion and `torsh-python`/`torsh-ffi`'s PyO3 surface both remain informal maturity gaps, not yet centrally tracked.** `Cargo.toml:59` excludes `crates/torsh-distributed` from `default-members` with the inline comment "Tests/examples need additional API updates (scheduled for alpha.3)" (it is still a full workspace member per `Cargo.toml:20`/`:118`, just not built by default). This is a distinct, more basic maturity signal than the already-tracked `torsh-distributed` `ProcessGroup`/`MockBackend` functional gap further up in this file (the "`init_process_group`/`ProcessGroup` API always constructs `MockBackend`" entry) — that entry is about *what the code does*, this one is about *whether it's even built/tested by default*; worth cross-referencing together when torsh-distributed maturity is next picked up. Separately, `torsh-python`/`torsh-ffi` compile clean at HEAD (`cargo check --workspace` green, including both crates) and are not currently flagged with open PyO3-version-compatibility build errors in this file — the scalar-operand and no-op-stub gaps already tracked above (`torsh-python`'s Tensor operator overloads / distributed collectives entries) are the real, current maturity gaps for that pair of crates; no additional PyO3 compatibility issue was found to add here.
  - **Approach:** when torsh-distributed's alpha.3 default-members re-inclusion is scheduled, resolve it alongside the existing `ProcessGroup`/`MockBackend` wiring fix so both the "not built by default" and "mocked when built" gaps close together.
  - **Scope:** small to re-enable default-members (mechanical); the real work is the already-tracked ProcessGroup wiring (medium, see that entry).
  - **Risk:** low; re-adding to default-members just changes CI/default build surface, no runtime behavior change.
