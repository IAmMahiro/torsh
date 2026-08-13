"""Regression tests for the ToRSh Python bindings hardening pass.

These cover the `import rstorch` chain and the previously silently-wrong
Python-level helpers. They are the end-to-end acceptance test for Mission A
(F042-F046, F137, F138, F247) and must be run against a maturin-built wheel
(``maturin develop`` / ``maturin build``), not ``cargo test`` — the defects are
invisible to Rust unit tests.
"""

import math

import rstorch
import rstorch.nn as nn
import rstorch.optim as optim
import rstorch.functional as F
import rstorch.autograd as autograd
import rstorch.distributed as dist


def test_import_and_submodules():
    # F042/F043/F044/F045/F137: the package and every submodule import.
    assert rstorch.__version__
    assert hasattr(rstorch, "nn") and hasattr(rstorch, "optim")
    assert hasattr(nn, "Linear") and hasattr(optim, "SGD")
    # F045: submodules are real modules registered in sys.modules.
    import importlib
    assert importlib.import_module("rstorch._C.autograd") is not None
    assert importlib.import_module("rstorch._C.functional") is not None
    assert importlib.import_module("rstorch._C.optim.lr_scheduler") is not None


def test_uint16_absent_but_import_ok():
    # F043: uint16 is intentionally not exported; uint8/uint32/uint64 are.
    assert hasattr(rstorch, "uint8")
    assert hasattr(rstorch, "uint32")
    assert not hasattr(rstorch, "uint16")


def test_full_fills_value():
    # F247: full() previously returned zeros.
    t = rstorch.full([2, 2], 7.0)
    assert t.tolist() == [[7.0, 7.0], [7.0, 7.0]]


def test_full_like_fills_value():
    base = rstorch.zeros([3])
    t = rstorch.full_like(base, 5.0)
    assert t.tolist() == [5.0, 5.0, 5.0]


def test_add_respects_alpha():
    # F247: alpha was silently ignored.
    a = rstorch.ones([2])
    b = rstorch.ones([2])
    out = rstorch.add(a, b, alpha=3.0)  # 1 + 3*1 = 4
    assert out.tolist() == [4.0, 4.0]


def test_sub_respects_alpha():
    a = rstorch.full([2], 10.0)
    b = rstorch.ones([2])
    out = rstorch.sub(a, b, alpha=2.0)  # 10 - 2*1 = 8
    assert out.tolist() == [8.0, 8.0]


def test_linear_is_callable():
    # F138: Module.__call__ used to raise NotImplementedError.
    model = nn.Linear(3, 4)
    x = rstorch.randn([2, 3])
    y = model(x)
    assert y.shape == [2, 4]


def test_python_subclass_forward_dispatch():
    # F138: __call__ must dispatch to a Python subclass override.
    class Net(nn.Module):
        def forward(self, x):
            return F.relu(x)

    net = Net()
    x = rstorch.tensor([-1.0, 2.0, -3.0])
    y = net(x)
    assert y.tolist() == [0.0, 2.0, 0.0]


def test_activations_are_not_relu_or_tanh():
    # F044: elu/selu/gelu/mish/softplus/softsign were relu/tanh stubs.
    x = rstorch.tensor([-1.0, 1.0])

    # softplus(0) == ln(2); relu(0) == 0 -> distinguishes the stub.
    sp = F.softplus(rstorch.tensor([0.0])).tolist()[0]
    assert abs(sp - math.log(2.0)) < 1e-4

    # ELU(-1) == exp(-1) - 1 ~= -0.632 (relu would give 0).
    elu_neg = F.elu(x).tolist()[0]
    assert abs(elu_neg - (math.e ** -1 - 1.0)) < 1e-3

    # softsign(1) == 0.5 (tanh(1) ~= 0.7616).
    ss = F.softsign(rstorch.tensor([1.0])).tolist()[0]
    assert abs(ss - 0.5) < 1e-4


def test_functional_keyword_args():
    # Regression: optional args must keep clean Python keyword names.
    x = rstorch.tensor([-1.0, 2.0])
    assert F.relu(x, inplace=False).tolist() == [0.0, 2.0]
    F.softmax(x, 0, dtype=None)
    # gelu(approximate="tanh") must refuse rather than silently return exact.
    raised = False
    try:
        F.gelu(x, approximate="tanh")
    except NotImplementedError:
        raised = True
    assert raised


def test_distributed_is_honest():
    # F046: collectives must not silently succeed; is_available truthful.
    assert dist.is_available() is False
    t = rstorch.ones([2])
    raised = False
    try:
        dist.all_reduce(t)
    except NotImplementedError:
        raised = True
    assert raised, "all_reduce must raise, not silently return"
