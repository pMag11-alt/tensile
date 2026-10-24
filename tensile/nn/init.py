"""Weight initialisers. Each fills a Parameter in place and returns it."""

from __future__ import annotations

import math

import numpy as np

from ..random import default_rng
from ..tensor import Tensor


def _fans(t: Tensor) -> tuple[int, int]:
    if t.ndim < 2:
        raise ValueError("fan in/out needs a tensor with at least 2 dims")
    receptive = int(np.prod(t.shape[2:])) if t.ndim > 2 else 1
    return t.shape[1] * receptive, t.shape[0] * receptive


def uniform_(t: Tensor, low: float = 0.0, high: float = 1.0) -> Tensor:
    t.data = default_rng().uniform(low, high, t.shape).astype(t.dtype)
    return t


def normal_(t: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    t.data = default_rng().normal(mean, std, t.shape).astype(t.dtype)
    return t


def zeros_(t: Tensor) -> Tensor:
    t.data = np.zeros(t.shape, dtype=t.dtype)
    return t


def ones_(t: Tensor) -> Tensor:
    t.data = np.ones(t.shape, dtype=t.dtype)
    return t


def xavier_uniform_(t: Tensor, gain: float = 1.0) -> Tensor:
    fan_in, fan_out = _fans(t)
    bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return uniform_(t, -bound, bound)


def kaiming_uniform_(t: Tensor, a: float = math.sqrt(5)) -> Tensor:
    """He init with the leaky-ReLU gain; a=sqrt(5) reproduces PyTorch's default Linear/Conv init."""
    fan_in, _ = _fans(t)
    gain = math.sqrt(2.0 / (1 + a * a))
    bound = gain * math.sqrt(3.0 / fan_in)
    return uniform_(t, -bound, bound)


def kaiming_normal_(t: Tensor) -> Tensor:
    fan_in, _ = _fans(t)
    return normal_(t, 0.0, math.sqrt(2.0 / fan_in))
