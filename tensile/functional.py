"""Stateless neural-network functions (the `F` in `import tensile.functional as F`)."""

from __future__ import annotations

import numpy as np

from . import ops
from .random import default_rng
from .tensor import Tensor


def _t(x) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)


# -- activations -------------------------------------------------------------


def relu(x: Tensor) -> Tensor:
    return ops.ReLU.apply(x)


def leaky_relu(x: Tensor, negative_slope: float = 0.01) -> Tensor:
    return ops.LeakyReLU.apply(x, slope=negative_slope)


def gelu(x: Tensor) -> Tensor:
    return ops.GELU.apply(x)


def sigmoid(x: Tensor) -> Tensor:
    return ops.Sigmoid.apply(x)


def tanh(x: Tensor) -> Tensor:
    return ops.Tanh.apply(x)


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    return ops.Softmax.apply(x, axis=axis)


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    return ops.LogSoftmax.apply(x, axis=axis)


# -- layers ------------------------------------------------------------------


def linear(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """y = x W^T + b, with W of shape (out_features, in_features)."""
    y = x @ weight.T
    return y + bias if bias is not None else y


def conv2d(x: Tensor, weight: Tensor, bias: Tensor | None = None, stride=1, padding=0) -> Tensor:
    if bias is None:
        return ops.Conv2d.apply(x, weight, stride=stride, padding=padding)
    return ops.Conv2d.apply(x, weight, bias, stride=stride, padding=padding)


def max_pool2d(x: Tensor, kernel_size, stride=None) -> Tensor:
    return ops.MaxPool2d.apply(x, kernel_size=kernel_size, stride=stride)


def avg_pool2d(x: Tensor, kernel_size, stride=None) -> Tensor:
    return ops.AvgPool2d.apply(x, kernel_size=kernel_size, stride=stride)


def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """Inverted dropout: zero each element with probability p, scale the rest by 1/(1-p)."""
    if not training or p == 0:
        return x
    if not 0 <= p < 1:
        raise ValueError(f"dropout probability must be in [0, 1), got {p}")
    keep = default_rng().random(x.shape) >= p
    return x * Tensor((keep / (1 - p)).astype(x.dtype))


def layer_norm(
    x: Tensor, normalized_shape, weight: Tensor | None = None, bias: Tensor | None = None, eps: float = 1e-5
):
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    normalized_shape = tuple(normalized_shape)
    if x.shape[-len(normalized_shape) :] != normalized_shape:
        raise ValueError(f"layer_norm: trailing shape {x.shape} doesn't match {normalized_shape}")
    if weight is None:
        weight = Tensor(np.ones(normalized_shape, dtype=x.dtype))
    if bias is None:
        bias = Tensor(np.zeros(normalized_shape, dtype=x.dtype))
    return ops.LayerNorm.apply(x, weight, bias, n_axes=len(normalized_shape), eps=eps)


def batch_norm(
    x: Tensor,
    running_mean: np.ndarray | None,
    running_var: np.ndarray | None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    """Batch norm over every axis except 1 (channels). Updates running stats in place when training."""
    axes = (0,) + tuple(range(2, x.ndim))
    shape = [1] * x.ndim
    shape[1] = x.shape[1]
    if training:
        mean = x.mean(axis=axes, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=axes, keepdims=True)
        if running_mean is not None:
            n = x.size // x.shape[1]
            running_mean *= 1 - momentum
            running_mean += momentum * mean.data.reshape(-1)
            running_var *= 1 - momentum
            running_var += momentum * var.data.reshape(-1) * n / max(n - 1, 1)
        xhat = (x - mean) / (var + eps) ** 0.5
    else:
        mean = running_mean.reshape(shape)
        std = np.sqrt(running_var.reshape(shape) + eps)
        xhat = (x - Tensor(mean.astype(x.dtype))) / Tensor(std.astype(x.dtype))
    if weight is not None:
        xhat = xhat * weight.reshape(shape)
    if bias is not None:
        xhat = xhat + bias.reshape(shape)
    return xhat


def embedding(indices, weight: Tensor) -> Tensor:
    idx = indices.data if isinstance(indices, Tensor) else np.asarray(indices)
    return weight[idx.astype(np.int64)]


def pad(x: Tensor, pad_width, value: float = 0.0) -> Tensor:
    """Constant padding; `pad_width` follows np.pad: ((before, after), ...) per axis."""
    if len(pad_width) != x.ndim:
        raise ValueError("pad_width needs one (before, after) pair per axis")
    return ops.Pad.apply(x, pad_width=tuple(tuple(p) for p in pad_width), value=value)


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, causal: bool = False, mask=None) -> Tensor:
    """softmax(q k^T / sqrt(d)) v over the last two axes."""
    d = q.shape[-1]
    scores = (q @ k.swapaxes(-1, -2)) * (1.0 / np.sqrt(d))
    t_q, t_k = q.shape[-2], k.shape[-2]
    if causal:
        future = np.triu(np.ones((t_q, t_k), dtype=bool), k=1 + t_k - t_q)
        scores = scores.masked_fill(future, -np.inf)
    if mask is not None:
        scores = scores.masked_fill(~np.asarray(mask, dtype=bool), -np.inf)
    return softmax(scores, axis=-1) @ v


# -- losses ------------------------------------------------------------------


def cross_entropy(logits: Tensor, target, reduction: str = "mean", label_smoothing: float = 0.0, ignore_index=None):
    """Softmax cross-entropy. `logits` is (N, C) or (N, ..., C)-flattened; `target` holds class ids."""
    target = target if isinstance(target, Tensor) else Tensor(np.asarray(target))
    if logits.ndim > 2:
        c = logits.shape[-1]
        logits = logits.reshape(-1, c)
        target = target.reshape(-1)
    return ops.CrossEntropy.apply(
        logits, target, reduction=reduction, label_smoothing=label_smoothing, ignore_index=ignore_index
    )


def nll_loss(log_probs: Tensor, target, reduction: str = "mean") -> Tensor:
    t = target.data if isinstance(target, Tensor) else np.asarray(target)
    picked = -log_probs[np.arange(len(t)), t.astype(np.int64)]
    return _reduce(picked, reduction)


def mse_loss(pred: Tensor, target, reduction: str = "mean") -> Tensor:
    return _reduce((pred - _t(target)) ** 2, reduction)


def l1_loss(pred: Tensor, target, reduction: str = "mean") -> Tensor:
    return _reduce((pred - _t(target)).abs(), reduction)


def binary_cross_entropy_with_logits(logits: Tensor, target, reduction: str = "mean") -> Tensor:
    """Stable BCE: max(z, 0) - z*y + log(1 + exp(-|z|))."""
    y = _t(target)
    loss = logits.relu() - logits * y + (1 + (-logits.abs()).exp()).log()
    return _reduce(loss, reduction)


def binary_cross_entropy(probs: Tensor, target, reduction: str = "mean", eps: float = 1e-7) -> Tensor:
    y = _t(target)
    p = probs.clip(eps, 1 - eps)
    return _reduce(-(y * p.log() + (1 - y) * (1 - p).log()), reduction)


def _reduce(x: Tensor, reduction: str) -> Tensor:
    if reduction == "mean":
        return x.mean()
    if reduction == "sum":
        return x.sum()
    if reduction == "none":
        return x
    raise ValueError(f"unknown reduction {reduction!r}")


__all__ = [
    "avg_pool2d",
    "batch_norm",
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
    "conv2d",
    "cross_entropy",
    "dropout",
    "embedding",
    "gelu",
    "l1_loss",
    "layer_norm",
    "leaky_relu",
    "linear",
    "log_softmax",
    "max_pool2d",
    "mse_loss",
    "nll_loss",
    "pad",
    "relu",
    "scaled_dot_product_attention",
    "sigmoid",
    "softmax",
    "tanh",
]
