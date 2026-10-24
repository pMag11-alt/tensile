"""Differentiable primitives.

Each class is one node type in the computation graph. `forward` receives raw
NumPy arrays and returns an array; `backward` receives dL/d(output) and
returns dL/d(input) for every tensor input, in order. Inputs whose gradient
isn't needed (see `needs_input_grad`) may get None.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .tensor import Function, unbroadcast

# ---------------------------------------------------------------------------
# Elementwise arithmetic
# ---------------------------------------------------------------------------


class Add(Function):
    def forward(self, a, b):
        self.shapes = a.shape, b.shape
        return a + b

    def backward(self, g):
        sa, sb = self.shapes
        return unbroadcast(g, sa), unbroadcast(g, sb)


class Sub(Function):
    def forward(self, a, b):
        self.shapes = a.shape, b.shape
        return a - b

    def backward(self, g):
        sa, sb = self.shapes
        return unbroadcast(g, sa), unbroadcast(-g, sb)


class Mul(Function):
    def forward(self, a, b):
        self.a, self.b = a, b
        return a * b

    def backward(self, g):
        a, b = self.a, self.b
        ga = unbroadcast(g * b, a.shape) if self.needs_input_grad[0] else None
        gb = unbroadcast(g * a, b.shape) if self.needs_input_grad[1] else None
        return ga, gb


class Div(Function):
    def forward(self, a, b):
        self.a, self.b = a, b
        return a / b

    def backward(self, g):
        a, b = self.a, self.b
        ga = unbroadcast(g / b, a.shape) if self.needs_input_grad[0] else None
        gb = unbroadcast(-g * a / (b * b), b.shape) if self.needs_input_grad[1] else None
        return ga, gb


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, g):
        return -g


class PowScalar(Function):
    def forward(self, x, exponent):
        self.x, self.p = x, exponent
        return x**exponent

    def backward(self, g):
        p = self.p
        return g * p * self.x ** (p - 1)


class Pow(Function):
    def forward(self, a, b):
        self.a, self.b = a, b
        self.out = a**b
        return self.out

    def backward(self, g):
        a, b, out = self.a, self.b, self.out
        ga = unbroadcast(g * b * a ** (b - 1), a.shape) if self.needs_input_grad[0] else None
        gb = None
        if self.needs_input_grad[1]:
            with np.errstate(divide="ignore", invalid="ignore"):
                log_a = np.where(a > 0, np.log(np.where(a > 0, a, 1)), 0)
            gb = unbroadcast(g * out * log_a, b.shape)
        return ga, gb


class MatMul(Function):
    """`a @ b` with NumPy semantics, including 1-D operands and batch broadcasting."""

    def forward(self, a, b):
        self.a, self.b = a, b
        return a @ b

    def backward(self, g):
        a, b = self.a, self.b
        if a.ndim == 1 and b.ndim == 1:
            return g * b, g * a
        a2 = a[None, :] if a.ndim == 1 else a
        b2 = b[:, None] if b.ndim == 1 else b
        g2 = g
        if a.ndim == 1:
            g2 = np.expand_dims(g2, -2)
        if b.ndim == 1:
            g2 = np.expand_dims(g2, -1)
        ga = gb = None
        if self.needs_input_grad[0]:
            ga = g2 @ np.swapaxes(b2, -1, -2)
            if a.ndim == 1:
                ga = ga.squeeze(-2)
            ga = unbroadcast(ga, a.shape)
        if self.needs_input_grad[1]:
            gb = np.swapaxes(a2, -1, -2) @ g2
            if b.ndim == 1:
                gb = gb.squeeze(-1)
            gb = unbroadcast(gb, b.shape)
        return ga, gb


class Cast(Function):
    def forward(self, x, dtype):
        self.src = x.dtype
        return x.astype(dtype)

    def backward(self, g):
        return g.astype(self.src)


# ---------------------------------------------------------------------------
# Unary math and activations
# ---------------------------------------------------------------------------


class Exp(Function):
    def forward(self, x):
        self.out = np.exp(x)
        return self.out

    def backward(self, g):
        return g * self.out


class Log(Function):
    def forward(self, x):
        self.x = x
        return np.log(x)

    def backward(self, g):
        return g / self.x


class Abs(Function):
    def forward(self, x):
        self.sign = np.sign(x)
        return np.abs(x)

    def backward(self, g):
        return g * self.sign


class Tanh(Function):
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, g):
        return g * (1 - self.out * self.out)


class Sigmoid(Function):
    def forward(self, x):
        # Split by sign so exp never overflows.
        out = np.empty_like(x)
        pos = x >= 0
        out[pos] = 1 / (1 + np.exp(-x[pos]))
        ex = np.exp(x[~pos])
        out[~pos] = ex / (1 + ex)
        self.out = out
        return out

    def backward(self, g):
        return g * self.out * (1 - self.out)


class ReLU(Function):
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, g):
        return g * self.mask


class LeakyReLU(Function):
    def forward(self, x, slope):
        self.scale = np.where(x > 0, 1, slope).astype(x.dtype)
        return x * self.scale

    def backward(self, g):
        return g * self.scale


_SQRT_2_OVER_PI = float(np.sqrt(2 / np.pi))


class GELU(Function):
    """GELU, tanh approximation (as used by GPT-2)."""

    def forward(self, x):
        self.x = x
        self.t = np.tanh(_SQRT_2_OVER_PI * (x + 0.044715 * x**3))
        return 0.5 * x * (1 + self.t)

    def backward(self, g):
        x, t = self.x, self.t
        dinner = _SQRT_2_OVER_PI * (1 + 3 * 0.044715 * x * x)
        return g * (0.5 * (1 + t) + 0.5 * x * (1 - t * t) * dinner)


class Clip(Function):
    def forward(self, x, lo=None, hi=None):
        self.mask = np.ones(x.shape, dtype=bool)
        if lo is not None:
            self.mask &= x >= lo
        if hi is not None:
            self.mask &= x <= hi
        return np.clip(x, lo, hi)

    def backward(self, g):
        return g * self.mask


class Maximum(Function):
    """Elementwise max. Ties send the gradient to the first argument."""

    def forward(self, a, b):
        self.shapes = a.shape, b.shape
        self.a_wins = a >= b
        return np.maximum(a, b)

    def backward(self, g):
        sa, sb = self.shapes
        return unbroadcast(g * self.a_wins, sa), unbroadcast(g * ~self.a_wins, sb)


class Where(Function):
    """`cond ? a : b`. The condition tensor is never differentiated."""

    def forward(self, cond, a, b):
        self.cond = cond.astype(bool)
        self.shapes = a.shape, b.shape
        return np.where(self.cond, a, b)

    def backward(self, g):
        sa, sb = self.shapes
        zero = np.zeros_like(g)
        return None, unbroadcast(np.where(self.cond, g, zero), sa), unbroadcast(np.where(self.cond, zero, g), sb)


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


def _norm_axes(axis, ndim):
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, int):
        axis = (axis,)
    return tuple(sorted(a % ndim for a in axis))


def _expand_reduced(g, shape, axis, keepdims):
    """Reshape a reduced gradient so it broadcasts back over `shape`."""
    if not keepdims:
        g = np.expand_dims(g, _norm_axes(axis, len(shape)))
    return g


class Sum(Function):
    def forward(self, x, axis=None, keepdims=False):
        self.shape, self.axis, self.keepdims = x.shape, axis, keepdims
        return np.sum(x, axis=axis, keepdims=keepdims)

    def backward(self, g):
        g = _expand_reduced(g, self.shape, self.axis, self.keepdims)
        return np.broadcast_to(g, self.shape)


class Mean(Function):
    def forward(self, x, axis=None, keepdims=False):
        self.shape, self.axis, self.keepdims = x.shape, axis, keepdims
        self.count = int(np.prod([x.shape[a] for a in _norm_axes(axis, x.ndim)]))
        return np.mean(x, axis=axis, keepdims=keepdims)

    def backward(self, g):
        g = _expand_reduced(g, self.shape, self.axis, self.keepdims)
        return np.broadcast_to(g / self.count, self.shape)


class Max(Function):
    """Max reduction. Tied maxima share the gradient equally."""

    def forward(self, x, axis=None, keepdims=False):
        self.axis, self.keepdims = axis, keepdims
        m = np.max(x, axis=axis, keepdims=True)
        self.mask = x == m
        self.mask_count = self.mask.sum(axis=axis, keepdims=True)
        return m if keepdims else np.squeeze(m, axis=_norm_axes(axis, x.ndim))

    def backward(self, g):
        g = _expand_reduced(g, self.mask.shape, self.axis, self.keepdims)
        return g * self.mask / self.mask_count


# ---------------------------------------------------------------------------
# Shape manipulation
# ---------------------------------------------------------------------------


class Reshape(Function):
    def forward(self, x, shape):
        self.shape = x.shape
        return x.reshape(shape)

    def backward(self, g):
        return g.reshape(self.shape)


class Transpose(Function):
    def forward(self, x, axes=None):
        self.axes = tuple(reversed(range(x.ndim))) if axes is None else tuple(a % x.ndim for a in axes)
        return np.transpose(x, self.axes)

    def backward(self, g):
        return np.transpose(g, np.argsort(self.axes))


class BroadcastTo(Function):
    def forward(self, x, shape):
        self.shape = x.shape
        return np.broadcast_to(x, shape)

    def backward(self, g):
        return unbroadcast(g, self.shape)


def _is_basic_index(index) -> bool:
    items = index if isinstance(index, tuple) else (index,)
    return all(i is None or i is Ellipsis or isinstance(i, (int, np.integer, slice)) for i in items)


class GetItem(Function):
    def forward(self, x, index):
        self.shape, self.dtype, self.index = x.shape, x.dtype, index
        return x[index]

    def backward(self, g):
        out = np.zeros(self.shape, dtype=g.dtype)
        if _is_basic_index(self.index):
            out[self.index] += g
        else:
            # Advanced indexing may repeat positions; add.at accumulates them.
            np.add.at(out, self.index, g)
        return out


class Concat(Function):
    def forward(self, *xs, axis=0):
        self.axis = axis
        self.splits = np.cumsum([x.shape[axis] for x in xs])[:-1]
        return np.concatenate(xs, axis=axis)

    def backward(self, g):
        return tuple(np.split(g, self.splits, axis=self.axis))


class Stack(Function):
    def forward(self, *xs, axis=0):
        self.axis = axis
        return np.stack(xs, axis=axis)

    def backward(self, g):
        return tuple(np.moveaxis(g, self.axis, 0))


class Pad(Function):
    def forward(self, x, pad_width, value=0.0):
        self.slices = tuple(slice(lo, lo + n) for (lo, _), n in zip(pad_width, x.shape))
        return np.pad(x, pad_width, constant_values=value)

    def backward(self, g):
        return g[self.slices]


# ---------------------------------------------------------------------------
# Softmax family and losses
# ---------------------------------------------------------------------------


class Softmax(Function):
    def forward(self, x, axis=-1):
        self.axis = axis
        e = np.exp(x - x.max(axis=axis, keepdims=True))
        self.out = e / e.sum(axis=axis, keepdims=True)
        return self.out

    def backward(self, g):
        y = self.out
        return y * (g - (g * y).sum(axis=self.axis, keepdims=True))


class LogSoftmax(Function):
    def forward(self, x, axis=-1):
        self.axis = axis
        shifted = x - x.max(axis=axis, keepdims=True)
        self.out = shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
        return self.out

    def backward(self, g):
        return g - np.exp(self.out) * g.sum(axis=self.axis, keepdims=True)


class CrossEntropy(Function):
    """Fused log-softmax + negative log-likelihood over class axis 1 of (N, C) logits."""

    def forward(self, logits, target, reduction="mean", label_smoothing=0.0, ignore_index=None):
        n, c = logits.shape
        target = target.astype(np.int64)
        shifted = logits - logits.max(axis=1, keepdims=True)
        logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        valid = np.ones(n, dtype=bool) if ignore_index is None else target != ignore_index
        safe_target = np.where(valid, target, 0)
        nll = -logp[np.arange(n), safe_target]
        if label_smoothing:
            nll = (1 - label_smoothing) * nll - label_smoothing * logp.mean(axis=1)
        nll = nll * valid
        self.logp, self.target, self.valid = logp, safe_target, valid
        self.reduction, self.eps = reduction, label_smoothing
        self.count = max(int(valid.sum()), 1)
        if reduction == "mean":
            return nll.sum() / self.count
        if reduction == "sum":
            return nll.sum()
        return nll

    def backward(self, g):
        n, c = self.logp.shape
        grad = np.exp(self.logp)
        grad[np.arange(n), self.target] -= 1 - self.eps
        if self.eps:
            grad -= self.eps / c
        grad *= self.valid[:, None]
        if self.reduction == "mean":
            grad *= g / self.count
        elif self.reduction == "sum":
            grad *= g
        else:
            grad *= g[:, None]
        return grad.astype(self.logp.dtype, copy=False), None


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


class LayerNorm(Function):
    """Fused layer norm over the trailing `n_axes` dimensions."""

    def forward(self, x, weight, bias, n_axes=1, eps=1e-5):
        axes = tuple(range(x.ndim - n_axes, x.ndim))
        mu = x.mean(axis=axes, keepdims=True)
        var = x.var(axis=axes, keepdims=True)
        self.rstd = 1 / np.sqrt(var + eps)
        self.xhat = (x - mu) * self.rstd
        self.weight, self.axes = weight, axes
        return self.xhat * weight + bias

    def backward(self, g):
        xhat, rstd, axes = self.xhat, self.rstd, self.axes
        batch_axes = tuple(range(g.ndim - len(axes)))
        gw = (g * xhat).sum(axis=batch_axes)
        gb = g.sum(axis=batch_axes)
        dxhat = g * self.weight
        gx = rstd * (
            dxhat - dxhat.mean(axis=axes, keepdims=True) - xhat * (dxhat * xhat).mean(axis=axes, keepdims=True)
        )
        return gx, gw, gb


# ---------------------------------------------------------------------------
# Convolution and pooling (NCHW)
# ---------------------------------------------------------------------------


def _pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)


class Conv2d(Function):
    """2-D cross-correlation via im2col + one big matrix multiply."""

    def forward(self, x, w, b=None, stride=1, padding=0):
        sh, sw = _pair(stride)
        ph, pw = _pair(padding)
        n, c, h, wd = x.shape
        f, c2, kh, kw = w.shape
        if c != c2:
            raise ValueError(f"conv2d: input has {c} channels but weight expects {c2}")
        xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw))) if ph or pw else x
        ho = (h + 2 * ph - kh) // sh + 1
        wo = (wd + 2 * pw - kw) // sw + 1
        win = sliding_window_view(xp, (kh, kw), axis=(2, 3))[:, :, ::sh, ::sw]  # N,C,Ho,Wo,kh,kw
        cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(n * ho * wo, c * kh * kw)
        out = cols @ w.reshape(f, -1).T  # (N*Ho*Wo, F)
        if b is not None:
            out += b
        self.cols, self.w = cols, w
        self.meta = (x.shape, xp.shape, (sh, sw), (ph, pw), (ho, wo))
        return out.reshape(n, ho, wo, f).transpose(0, 3, 1, 2)

    def backward(self, g):
        (n, c, h, wd), (_, _, hp, wp), (sh, sw), (ph, pw), (ho, wo) = self.meta
        f, _, kh, kw = self.w.shape
        g2 = g.transpose(0, 2, 3, 1).reshape(-1, f)
        gw = (g2.T @ self.cols).reshape(self.w.shape)
        gb = g.sum(axis=(0, 2, 3)) if len(self.needs_input_grad) > 2 else None
        gx = None
        if self.needs_input_grad[0]:
            gcols = (g2 @ self.w.reshape(f, -1)).reshape(n, ho, wo, c, kh, kw)
            gxp = np.zeros((n, c, hp, wp), dtype=g.dtype)
            for i in range(kh):
                for j in range(kw):
                    gxp[:, :, i : i + sh * ho : sh, j : j + sw * wo : sw] += gcols[:, :, :, :, i, j].transpose(
                        0, 3, 1, 2
                    )
            gx = gxp[:, :, ph : ph + h, pw : pw + wd]
        return (gx, gw, gb) if gb is not None else (gx, gw)


class MaxPool2d(Function):
    def forward(self, x, kernel_size, stride=None):
        kh, kw = _pair(kernel_size)
        sh, sw = _pair(stride or kernel_size)
        n, c, h, w = x.shape
        win = sliding_window_view(x, (kh, kw), axis=(2, 3))[:, :, ::sh, ::sw]  # N,C,Ho,Wo,kh,kw
        ho, wo = win.shape[2:4]
        flat = win.reshape(n, c, ho, wo, kh * kw)
        arg = flat.argmax(axis=-1)
        self.shape, self.arg, self.k, self.s = x.shape, arg, (kh, kw), (sh, sw)
        return np.take_along_axis(flat, arg[..., None], axis=-1)[..., 0]

    def backward(self, g):
        n, c, h, w = self.shape
        (kh, kw), (sh, sw) = self.k, self.s
        _, _, ho, wo = g.shape
        rows = np.arange(ho)[:, None] * sh + self.arg // kw
        cols = np.arange(wo)[None, :] * sw + self.arg % kw
        ni = np.arange(n)[:, None, None, None]
        ci = np.arange(c)[None, :, None, None]
        gx = np.zeros(self.shape, dtype=g.dtype)
        if sh >= kh and sw >= kw:
            gx[ni, ci, rows, cols] = g  # windows don't overlap: no duplicate targets
        else:
            np.add.at(gx, (ni, ci, rows, cols), g)
        return gx


class AvgPool2d(Function):
    def forward(self, x, kernel_size, stride=None):
        kh, kw = _pair(kernel_size)
        sh, sw = _pair(stride or kernel_size)
        win = sliding_window_view(x, (kh, kw), axis=(2, 3))[:, :, ::sh, ::sw]
        self.shape, self.k, self.s = x.shape, (kh, kw), (sh, sw)
        return win.mean(axis=(-2, -1))

    def backward(self, g):
        (kh, kw), (sh, sw) = self.k, self.s
        _, _, ho, wo = g.shape
        gx = np.zeros(self.shape, dtype=g.dtype)
        share = g / (kh * kw)
        for i in range(kh):
            for j in range(kw):
                gx[:, :, i : i + sh * ho : sh, j : j + sw * wo : sw] += share
        return gx
