"""Training helpers that operate on parameter gradients."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from ..tensor import Tensor


def clip_grad_norm_(parameters: Iterable[Tensor], max_norm: float) -> float:
    """Scale gradients so their global L2 norm is at most `max_norm`. Returns the norm before clipping."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    total = math.sqrt(sum(float(np.sum(g.data.astype(np.float64) ** 2)) for g in grads))
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        for g in grads:
            g.data = g.data * scale
    return total


def clip_grad_value_(parameters: Iterable[Tensor], clip_value: float) -> None:
    for p in parameters:
        if p.grad is not None:
            p.grad.data = np.clip(p.grad.data, -clip_value, clip_value)
