"""Every differentiable op, checked against central finite differences in float64."""

import numpy as np
import pytest

import tensile as ts
import tensile.functional as F
from tensile.gradcheck import gradcheck

rng = np.random.default_rng(42)


def r(*shape, positive=False):
    x = rng.normal(size=shape)
    return np.abs(x) + 0.5 if positive else x


UNARY = {
    "neg": lambda x: -x,
    "exp": lambda x: x.exp(),
    "log": lambda x: (x.abs() + 0.5).log(),
    "sqrt": lambda x: (x * x + 1).sqrt(),
    "abs": lambda x: x.abs(),
    "tanh": lambda x: x.tanh(),
    "sigmoid": lambda x: x.sigmoid(),
    "relu": lambda x: x.relu(),
    "leaky_relu": lambda x: F.leaky_relu(x, 0.1),
    "gelu": F.gelu,
    "pow3": lambda x: x**3,
    "pow_neg": lambda x: (x * x + 1) ** -1.5,
    "clip": lambda x: x.clip(-0.5, 0.7),
    "softmax": lambda x: x.softmax(axis=-1),
    "softmax0": lambda x: x.softmax(axis=0),
    "log_softmax": lambda x: x.log_softmax(axis=-1),
    "sum_all": lambda x: x.sum(),
    "sum_axis": lambda x: x.sum(axis=1),
    "sum_keep": lambda x: x.sum(axis=(0, 2), keepdims=True),
    "mean": lambda x: x.mean(axis=-1),
    "max_all": lambda x: x.max(),
    "max_axis": lambda x: x.max(axis=1),
    "min_axis": lambda x: x.min(axis=0, keepdims=True),
    "var": lambda x: x.var(axis=-1),
    "std_ddof": lambda x: x.std(axis=0, ddof=1),
    "reshape": lambda x: x.reshape(4, -1) * ts.tensor(np.arange(24.0).reshape(4, 6)),
    "transpose": lambda x: x.transpose(2, 0, 1) * ts.tensor(np.arange(24.0).reshape(4, 2, 3)),
    "T": lambda x: x.reshape(6, 4).T.exp(),
    "getitem_slice": lambda x: x[:, 1:, ::2],
    "getitem_int": lambda x: x[1, 0],
    "getitem_fancy_dupes": lambda x: x[np.array([0, 1, 1, 0])],
    "getitem_mask": lambda x: x[x.data > 0],
    "unsqueeze": lambda x: x.unsqueeze(1).sum(axis=1),
    "broadcast_to": lambda x: x[:, :1, :].broadcast_to((2, 5, 4)),
    "flatten": lambda x: x.flatten(1),
    "pad": lambda x: F.pad(x, ((0, 0), (1, 2), (0, 1))),
    "masked_fill": lambda x: x.masked_fill(x.data > 0.5, 0.0),
    "layer_norm_plain": lambda x: F.layer_norm(x, 4),
    "batch_norm_train": lambda x: F.batch_norm(x, None, None, training=True),
}


@pytest.mark.parametrize("name", sorted(UNARY))
def test_unary(name):
    gradcheck(UNARY[name], r(2, 3, 4))


BINARY = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / (b * b + 1),
    "pow": lambda a, b: (a * a + 0.5) ** b,
    "maximum": ts.maximum,
    "minimum": ts.minimum,
    "where": lambda a, b: ts.where(a.data > 0, a, b),
    "cat0": lambda a, b: ts.cat([a, b], axis=0),
    "cat2": lambda a, b: ts.cat([a, b * 2], axis=2),
    "stack": lambda a, b: ts.stack([a, b], axis=1),
}


@pytest.mark.parametrize("name", sorted(BINARY))
def test_binary_same_shape(name):
    gradcheck(BINARY[name], r(2, 3, 4), r(2, 3, 4))


@pytest.mark.parametrize("name", ["add", "sub", "mul", "div", "pow", "maximum", "minimum"])
@pytest.mark.parametrize("shapes", [((2, 3, 4), (4,)), ((2, 1, 4), (1, 3, 1)), ((3, 4), (2, 3, 4)), ((2, 3, 4), ())])
def test_binary_broadcasting(name, shapes):
    gradcheck(BINARY[name], r(*shapes[0]), r(*shapes[1]))


@pytest.mark.parametrize(
    "sa,sb",
    [
        ((3, 4), (4, 5)),
        ((4,), (4, 5)),
        ((3, 4), (4,)),
        ((4,), (4,)),
        ((2, 3, 4), (4, 5)),
        ((2, 3, 4), (2, 4, 5)),
        ((5, 1, 3, 4), (2, 4, 6)),
    ],
)
def test_matmul(sa, sb):
    gradcheck(lambda a, b: a @ b, r(*sa), r(*sb))


def test_scalar_operands_on_both_sides():
    gradcheck(lambda x: 2 * x - 1 / (x * x + 1) + x / 3 - 4 ** (x * 0.1), r(3, 3))


def test_linear():
    gradcheck(F.linear, r(5, 3), r(4, 3), r(4))


@pytest.mark.parametrize("stride,padding", [(1, 0), (1, 1), (2, 1), (2, 0)])
def test_conv2d(stride, padding):
    gradcheck(lambda x, w, b: F.conv2d(x, w, b, stride=stride, padding=padding), r(2, 3, 6, 5), r(4, 3, 3, 2), r(4))


def test_conv2d_no_bias():
    gradcheck(lambda x, w: F.conv2d(x, w, padding=1), r(1, 2, 4, 4), r(3, 2, 3, 3))


@pytest.mark.parametrize("kernel,stride", [(2, None), (3, 1), (2, 1), (3, 2)])
def test_max_pool2d(kernel, stride):
    gradcheck(lambda x: F.max_pool2d(x, kernel, stride), r(2, 3, 6, 6))


@pytest.mark.parametrize("kernel,stride", [(2, None), (3, 1)])
def test_avg_pool2d(kernel, stride):
    gradcheck(lambda x: F.avg_pool2d(x, kernel, stride), r(2, 2, 5, 5))


def test_layer_norm_affine():
    gradcheck(lambda x, w, b: F.layer_norm(x, (3, 4), w, b), r(2, 3, 4), r(3, 4), r(3, 4))


def test_batch_norm_affine_4d():
    gradcheck(lambda x, w, b: F.batch_norm(x, None, None, w, b, training=True), r(4, 3, 2, 2), r(3), r(3))


@pytest.mark.parametrize("smoothing", [0.0, 0.1])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_cross_entropy(reduction, smoothing):
    target = np.array([0, 3, 1, 3, 2])
    gradcheck(lambda x: F.cross_entropy(x, target, reduction=reduction, label_smoothing=smoothing), r(5, 4))


def test_cross_entropy_ignore_index_and_3d():
    target = np.array([[0, -100, 2], [1, 1, -100]])
    gradcheck(lambda x: F.cross_entropy(x, target, ignore_index=-100), r(2, 3, 4))


def test_losses():
    y = rng.random((4, 3))
    gradcheck(lambda p: F.mse_loss(p, y), r(4, 3))
    gradcheck(lambda p: F.l1_loss(p, y), r(4, 3))
    gradcheck(lambda z: F.binary_cross_entropy_with_logits(z, (y > 0.5).astype(float)), r(4, 3))
    gradcheck(lambda z: F.binary_cross_entropy(z.sigmoid(), y), r(4, 3))
    gradcheck(lambda z: F.nll_loss(z.log_softmax(), [0, 2, 1, 1]), r(4, 3))


def test_embedding_with_repeated_ids():
    ids = np.array([[1, 3, 1], [0, 1, 4]])
    gradcheck(lambda w: F.embedding(ids, w), r(5, 3))


@pytest.mark.parametrize("causal", [False, True])
def test_attention(causal):
    gradcheck(
        lambda q, k, v: F.scaled_dot_product_attention(q, k, v, causal=causal),
        r(2, 4, 3),
        r(2, 4, 3),
        r(2, 4, 3),
    )


def test_deep_composite_expression():
    def f(a, b, c):
        h = (a @ b).tanh()
        h = ts.cat([h, h.sigmoid()], axis=1)
        return (h * c).log_softmax(axis=1).max(axis=1) + h.mean()

    gradcheck(f, r(3, 4), r(4, 2), r(3, 4))


def test_gradcheck_catches_a_wrong_backward():
    class BadSquare(ts.Function):
        def forward(self, x):
            self.x = x
            return x * x

        def backward(self, g):
            return g * self.x  # should be 2 * g * x

    with pytest.raises(AssertionError, match="gradient mismatch"):
        gradcheck(lambda x: BadSquare.apply(x), r(3))
