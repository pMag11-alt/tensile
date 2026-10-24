"""Finite-difference gradient checking."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .tensor import Tensor


def numerical_grad(fn: Callable[..., Tensor], inputs: Sequence[np.ndarray], wrt: int, eps: float = 1e-6) -> np.ndarray:
    """Central-difference estimate of d(sum(fn(*inputs)))/d(inputs[wrt])."""
    x = inputs[wrt]
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        orig = x[i]
        x[i] = orig + eps
        plus = float(fn(*[Tensor(a) for a in inputs]).data.sum())
        x[i] = orig - eps
        minus = float(fn(*[Tensor(a) for a in inputs]).data.sum())
        x[i] = orig
        grad[i] = (plus - minus) / (2 * eps)
    return grad


def gradcheck(
    fn: Callable[..., Tensor],
    *inputs,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    seed_grad: bool = True,
) -> bool:
    """Compare autograd gradients of `fn` against central differences, in float64.

    The output is reduced with a random weighting (when `seed_grad`) rather than a
    plain sum, so errors that cancel out under summation still show up.
    Raises AssertionError with the worst mismatch on failure.
    """
    arrays = [np.array(x, dtype=np.float64) for x in inputs]
    weights = None
    if seed_grad:
        probe = fn(*[Tensor(a) for a in arrays])
        weights = np.random.default_rng(0).normal(size=probe.shape)

    def scalar_fn(*ts):
        out = fn(*ts)
        return out * Tensor(weights) if weights is not None else out

    tensors = [Tensor(a.copy(), requires_grad=True) for a in arrays]
    scalar_fn(*tensors).sum().backward()
    for k, t in enumerate(tensors):
        expected = numerical_grad(scalar_fn, arrays, k, eps)
        actual = np.zeros_like(arrays[k]) if t.grad is None else t.grad.data
        if not np.allclose(actual, expected, atol=atol, rtol=rtol):
            err = np.abs(actual - expected)
            worst = np.unravel_index(np.argmax(err), err.shape)
            raise AssertionError(
                f"gradient mismatch for input {k} at {worst}: autograd={actual[worst]:.6g} "
                f"numerical={expected[worst]:.6g} (max abs err {err.max():.3g})"
            )
    return True
