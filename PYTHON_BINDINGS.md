# ToRSh Python Bindings

PyO3-based, PyTorch-compatible Python bindings for ToRSh. The compiled
extension is exposed as `rstorch._C` and wrapped by the pure-Python package in
`python/rstorch/`.

## Build & install

The single canonical maturin manifest is the workspace-root `pyproject.toml`
(distribution name `torsh`). It points `manifest-path` at
`crates/torsh-python/Cargo.toml`, whose `#[pymodule(name = "_C")]` produces the
`PyInit__C` init symbol that `module-name = "rstorch._C"` expects.

```bash
# From the repo root, inside a virtualenv:
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy
maturin develop            # builds rstorch._C and installs the rstorch package
python -c "import rstorch; print(rstorch.__version__)"
```

`cargo test` does NOT exercise these bindings (the import-time defects are
invisible to Rust). The regression suite is `python/tests/test_import_hardening.py`
and must be run against a maturin-built wheel.

## Source layout (real files)

| Area | Files |
|------|-------|
| Tensor | `crates/torsh-python/src/tensor/{core,creation,...}.rs` |
| Neural nets | `crates/torsh-python/src/nn/{module,linear,conv,normalization,dropout,pooling,container,activation,loss}.rs` |
| Optimizers | `crates/torsh-python/src/optim/{sgd,adam,adagrad,rmsprop,base,lr_scheduler}.rs` |
| Functional | `crates/torsh-python/src/functional.rs` |
| Device / dtype | `crates/torsh-python/src/{device,dtype}.rs` |
| Autograd | `crates/torsh-python/src/autograd.rs` |
| Distributed | `crates/torsh-python/src/distributed.rs` |
| Data | `crates/torsh-python/src/data.rs` |
| Errors | `crates/torsh-python/src/error.rs` |
| Python package | `python/rstorch/{__init__,nn,optim,autograd,distributed,functional}.py` |

There is no `tensor_simple.rs` or `nn_simple.rs`; earlier revisions of this
document referenced files and a `python/torsh/` package that never existed.

## Status matrix

| Feature | Status | Notes |
|---------|--------|-------|
| `import rstorch` and submodules | Working | Submodules registered in `sys.modules` as `rstorch._C.<sub>`. |
| Tensor creation (`zeros`, `ones`, `full`, `randn`, `*_like`, ...) | Working | Provided by the extension; `full` fills correctly. |
| `nn` layers (`Linear`, `Conv1/2d`, `BatchNorm`, `LayerNorm`, `Dropout*`, pooling, `Sequential`, `ModuleList`) | Working | `Module.__call__` dispatches dynamically to the subclass `forward`. |
| Activations (`relu`/`elu`/`selu`/`gelu`/`mish`/`softplus`/`softsign`/`leaky_relu`/`silu`) | Working | Delegated to `torsh_functional`; no relu/tanh placeholders. |
| Loss (`mse_loss`, `l1_loss`, `cross_entropy`, `binary_cross_entropy`) | Working | Delegated to `torsh_functional`. |
| `batch_norm` / `layer_norm` / `dropout` (functional) | Working | Delegated to `torsh_functional`. |
| Optimizers (`SGD`, `Adam`, `AdamW`, `Adagrad`, `RMSprop`) + `lr_scheduler` | Working | Wrap real `torsh_optim`. |
| `autograd.no_grad` / `set_grad_enabled` | Working | Bridged to `torsh_autograd` / `torsh_core::grad_mode` global flag. |
| `distributed` collectives | Not implemented | Raise `NotImplementedError`; `is_available()` returns `False`. No real multi-process backend is wired yet. |
| `uint16` dtype | Not supported | `torsh_core::DType` has no `U16` variant. |

## Known gaps / follow-ups

- Add `DType::U16` to `torsh-core` to restore a `uint16` dtype.
- Wire `rstorch.distributed` to a real (TCP) `torsh-distributed` backend; today
  only a mock backend exists, so the Python collectives raise instead of
  silently corrupting gradients.
- `numpy_compatibility` / `pandas` / `scipy` interop live in `torsh-ffi`, not in
  `torsh-python`.
- Splitting the combined binding cdylib (PyO3 `torsh-python` + N-API `torsh-ffi`)
  into separate `torsh-capi` / `torsh-node` packages is **deferred** to a later
  0.2.x. The current single-crate layout builds, imports, and passes
  maturin + pytest today; the split is a packaging cleanup, not a bugfix.
