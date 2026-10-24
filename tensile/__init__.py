"""tensile: a small deep-learning framework built from scratch on NumPy.

import tensile as ts
x = ts.randn(4, 3, requires_grad=True)
loss = (x @ x.T).relu().sum()
loss.backward()
x.grad
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import data, datasets, functional, nn, optim
from .gradcheck import gradcheck
from .random import default_rng, manual_seed
from .tensor import DEFAULT_DTYPE, Function, Tensor, enable_grad, is_grad_enabled, no_grad
from .viz import to_dot, to_mermaid

__version__ = "0.1.0"


def tensor(data, requires_grad: bool = False, dtype=None) -> Tensor:
    return Tensor(data, requires_grad=requires_grad, dtype=dtype)


def _shape(shape) -> tuple[int, ...]:
    return tuple(shape[0]) if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else tuple(shape)


def zeros(*shape, requires_grad: bool = False, dtype=DEFAULT_DTYPE) -> Tensor:
    return Tensor(np.zeros(_shape(shape), dtype=dtype), requires_grad=requires_grad)


def ones(*shape, requires_grad: bool = False, dtype=DEFAULT_DTYPE) -> Tensor:
    return Tensor(np.ones(_shape(shape), dtype=dtype), requires_grad=requires_grad)


def full(shape, value, requires_grad: bool = False, dtype=DEFAULT_DTYPE) -> Tensor:
    return Tensor(np.full(shape, value, dtype=dtype), requires_grad=requires_grad)


def zeros_like(t: Tensor, requires_grad: bool = False) -> Tensor:
    return Tensor(np.zeros_like(t.data), requires_grad=requires_grad)


def ones_like(t: Tensor, requires_grad: bool = False) -> Tensor:
    return Tensor(np.ones_like(t.data), requires_grad=requires_grad)


def eye(n: int, dtype=DEFAULT_DTYPE) -> Tensor:
    return Tensor(np.eye(n, dtype=dtype))


def arange(*args, dtype=None) -> Tensor:
    return Tensor(np.arange(*args, dtype=dtype))


def randn(*shape, requires_grad: bool = False, dtype=DEFAULT_DTYPE) -> Tensor:
    return Tensor(default_rng().standard_normal(_shape(shape)).astype(dtype), requires_grad=requires_grad)


def rand(*shape, requires_grad: bool = False, dtype=DEFAULT_DTYPE) -> Tensor:
    return Tensor(default_rng().random(_shape(shape)).astype(dtype), requires_grad=requires_grad)


def randint(low: int, high: int, shape) -> Tensor:
    return Tensor(default_rng().integers(low, high, shape))


def cat(tensors, axis: int = 0) -> Tensor:
    from .ops import Concat

    return Concat.apply(*tensors, axis=axis)


def stack(tensors, axis: int = 0) -> Tensor:
    from .ops import Stack

    return Stack.apply(*tensors, axis=axis)


def where(cond, a, b) -> Tensor:
    from .ops import Where

    cond = cond.data if isinstance(cond, Tensor) else np.asarray(cond)
    a = a if isinstance(a, Tensor) else Tensor(a)
    b = b if isinstance(b, Tensor) else Tensor(b)
    return Where.apply(Tensor(cond.astype(bool)), a, b)


def maximum(a: Tensor, b) -> Tensor:
    from .ops import Maximum

    return Maximum.apply(a, a._coerce(b))


def minimum(a: Tensor, b) -> Tensor:
    return -maximum(-a, -a._coerce(b))


def save(state: dict, path: str | Path) -> None:
    """Save a state_dict (name -> array) as an .npz file."""
    np.savez(path, **{k: np.asarray(v) for k, v in state.items()})


def load(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


__all__ = [
    "DEFAULT_DTYPE",
    "Function",
    "Tensor",
    "arange",
    "cat",
    "data",
    "datasets",
    "default_rng",
    "enable_grad",
    "eye",
    "full",
    "functional",
    "gradcheck",
    "is_grad_enabled",
    "load",
    "manual_seed",
    "maximum",
    "minimum",
    "nn",
    "no_grad",
    "ones",
    "ones_like",
    "optim",
    "rand",
    "randint",
    "randn",
    "save",
    "stack",
    "tensor",
    "to_dot",
    "to_mermaid",
    "where",
    "zeros",
    "zeros_like",
]
