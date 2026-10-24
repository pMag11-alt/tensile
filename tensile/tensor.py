"""The Tensor type and the reverse-mode autodiff engine.

A Tensor wraps a NumPy array. Every differentiable operation is a `Function`
subclass with a `forward` that works on raw arrays and a `backward` that maps
the gradient of the output to gradients of the inputs. When gradients are
enabled, applying a Function records it on the output tensor (`_ctx`), which
builds the computation graph that `Tensor.backward()` walks in reverse
topological order.
"""

from __future__ import annotations

import numpy as np

DEFAULT_DTYPE = np.float32

# ---------------------------------------------------------------------------
# Gradient mode
# ---------------------------------------------------------------------------

_grad_enabled = True


def is_grad_enabled() -> bool:
    return _grad_enabled


class _GradMode:
    """Context manager / decorator that sets gradient tracking on or off."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._stack: list[bool] = []

    def __enter__(self):
        global _grad_enabled
        self._stack.append(_grad_enabled)
        _grad_enabled = self.enabled
        return self

    def __exit__(self, *exc):
        global _grad_enabled
        _grad_enabled = self._stack.pop()
        return False

    def __call__(self, fn):
        import functools

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with _GradMode(self.enabled):
                return fn(*args, **kwargs)

        return wrapper


def no_grad() -> _GradMode:
    """Disable graph recording: `with no_grad(): ...` or `@no_grad()`."""
    return _GradMode(False)


def enable_grad() -> _GradMode:
    return _GradMode(True)


# ---------------------------------------------------------------------------
# Function: one node of the computation graph
# ---------------------------------------------------------------------------


class Function:
    """Base class for differentiable operations.

    Subclasses implement `forward(*arrays, **params) -> array` and
    `backward(grad) -> tuple of arrays (or None)`, one entry per tensor input.
    Anything needed by backward is stashed on `self` during forward.
    """

    parents: tuple[Tensor, ...] = ()
    needs_input_grad: tuple[bool, ...] = ()

    def forward(self, *args, **kwargs):  # pragma: no cover - abstract
        raise NotImplementedError

    def backward(self, grad):  # pragma: no cover - abstract
        raise NotImplementedError

    @classmethod
    def apply(cls, *inputs, **params) -> Tensor:
        tensors = tuple(x if isinstance(x, Tensor) else Tensor(x) for x in inputs)
        ctx = cls()
        ctx.needs_input_grad = tuple(t.requires_grad for t in tensors)
        out = ctx.forward(*(t.data for t in tensors), **params)
        requires_grad = _grad_enabled and any(ctx.needs_input_grad)
        result = Tensor._wrap(out, requires_grad)
        if requires_grad:
            ctx.parents = tensors
            result._ctx = ctx
        return result

    @property
    def name(self) -> str:
        return type(self).__name__


def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum `grad` down to `shape`, undoing NumPy broadcasting."""
    if grad.shape == shape:
        return grad
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    axes = tuple(i for i, n in enumerate(shape) if n == 1 and grad.shape[i] != 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)
    return grad.reshape(shape)


# ---------------------------------------------------------------------------
# Tensor
# ---------------------------------------------------------------------------


def _as_array(data, dtype=None) -> np.ndarray:
    if isinstance(data, Tensor):
        data = data.data
    if dtype is not None:
        return np.asarray(data, dtype=dtype)
    if isinstance(data, (np.ndarray, np.generic)):
        return np.asarray(data)
    arr = np.asarray(data)
    # Python floats default to float32, like torch.
    if arr.dtype == np.float64:
        arr = arr.astype(DEFAULT_DTYPE)
    return arr


class Tensor:
    """An n-dimensional array that can record operations for autodiff."""

    # Make NumPy defer to Tensor's reflected operators (np.float32(2) * t).
    __array_ufunc__ = None

    def __init__(self, data, requires_grad: bool = False, dtype=None):
        arr = _as_array(data, dtype)
        if requires_grad and arr.dtype.kind != "f":
            raise TypeError(f"only floating point tensors can require gradients, got {arr.dtype}")
        self.data: np.ndarray = arr
        self.grad: Tensor | None = None
        self.requires_grad = requires_grad
        self._ctx: Function | None = None
        self._retain_grad = False

    @classmethod
    def _wrap(cls, arr, requires_grad: bool = False) -> Tensor:
        t = cls.__new__(cls)
        t.data = np.asarray(arr)
        t.grad = None
        t.requires_grad = requires_grad
        t._ctx = None
        t._retain_grad = False
        return t

    # -- basic properties -------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def is_leaf(self) -> bool:
        return self._ctx is None

    @property
    def grad_fn(self) -> Function | None:
        return self._ctx

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        body = np.array2string(self.data, separator=", ", precision=4, prefix="tensor(")
        extras = []
        if self.data.dtype not in (DEFAULT_DTYPE, np.int64):
            extras.append(f"dtype={self.data.dtype}")
        if self._ctx is not None:
            extras.append(f"grad_fn=<{self._ctx.name}Backward>")
        elif self.requires_grad:
            extras.append("requires_grad=True")
        return f"tensor({body}{''.join(', ' + e for e in extras)})"

    def numpy(self) -> np.ndarray:
        return self.data

    def item(self):
        return self.data.item()

    def tolist(self):
        return self.data.tolist()

    def __array__(self, dtype=None, copy=None):
        return self.data if dtype is None else self.data.astype(dtype)

    def detach(self) -> Tensor:
        return Tensor._wrap(self.data)

    def requires_grad_(self, flag: bool = True) -> Tensor:
        if flag and self.data.dtype.kind != "f":
            raise TypeError("only floating point tensors can require gradients")
        self.requires_grad = flag
        return self

    def retain_grad(self) -> Tensor:
        """Keep `.grad` on a non-leaf tensor after backward()."""
        self._retain_grad = True
        return self

    def astype(self, dtype) -> Tensor:
        return ops.Cast.apply(self, dtype=np.dtype(dtype))

    def float(self) -> Tensor:
        return self.astype(np.float32)

    def double(self) -> Tensor:
        return self.astype(np.float64)

    def zero_(self) -> Tensor:
        self.data[...] = 0
        return self

    # -- autodiff -----------------------------------------------------------

    def backward(self, grad=None) -> None:
        """Accumulate d(self)/d(leaf) into `.grad` of every leaf that requires it."""
        if not self.requires_grad:
            raise RuntimeError("backward() called on a tensor that does not require grad")
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be given for non-scalar outputs")
            grad = np.ones_like(self.data)
        else:
            grad = _as_array(grad, self.data.dtype)
            if grad.shape != self.shape:
                raise ValueError(f"grad shape {grad.shape} != tensor shape {self.shape}")

        # Iterative post-order DFS so deep graphs don't hit the recursion limit.
        order: list[Tensor] = []
        seen: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            if node._ctx is not None:
                for parent in node._ctx.parents:
                    if parent.requires_grad and id(parent) not in seen:
                        stack.append((parent, False))

        grads: dict[int, np.ndarray] = {id(self): grad}
        for node in reversed(order):
            g = grads.pop(id(node), None)
            if g is None:
                continue
            if node._ctx is None or node._retain_grad:
                node._accumulate(g)
            if node._ctx is None:
                continue
            ctx = node._ctx
            parent_grads = ctx.backward(g)
            if not isinstance(parent_grads, tuple):
                parent_grads = (parent_grads,)
            for parent, pg in zip(ctx.parents, parent_grads):
                if pg is None or not parent.requires_grad:
                    continue
                if pg.shape != parent.shape:
                    raise RuntimeError(
                        f"{ctx.name}.backward produced grad of shape {pg.shape} for input of shape {parent.shape}"
                    )
                prev = grads.get(id(parent))
                grads[id(parent)] = pg if prev is None else prev + pg

    def _accumulate(self, g: np.ndarray) -> None:
        g = g.astype(self.data.dtype, copy=False)
        if self.grad is None:
            self.grad = Tensor._wrap(np.array(g, copy=True))
        else:
            self.grad.data = self.grad.data + g

    # -- operators ------------------------------------------------------------

    def _coerce(self, other) -> Tensor:
        if isinstance(other, Tensor):
            return other
        if isinstance(other, (int, float, bool)) and self.data.dtype.kind == "f":
            return Tensor._wrap(np.asarray(other, dtype=self.data.dtype))
        return Tensor(other)

    def __add__(self, other):
        return ops.Add.apply(self, self._coerce(other))

    def __radd__(self, other):
        return ops.Add.apply(self._coerce(other), self)

    def __sub__(self, other):
        return ops.Sub.apply(self, self._coerce(other))

    def __rsub__(self, other):
        return ops.Sub.apply(self._coerce(other), self)

    def __mul__(self, other):
        return ops.Mul.apply(self, self._coerce(other))

    def __rmul__(self, other):
        return ops.Mul.apply(self._coerce(other), self)

    def __truediv__(self, other):
        return ops.Div.apply(self, self._coerce(other))

    def __rtruediv__(self, other):
        return ops.Div.apply(self._coerce(other), self)

    def __neg__(self):
        return ops.Neg.apply(self)

    def __pow__(self, exponent):
        if isinstance(exponent, Tensor):
            return ops.Pow.apply(self, exponent)
        return ops.PowScalar.apply(self, exponent=exponent)

    def __rpow__(self, base):
        return ops.Pow.apply(self._coerce(base), self)

    def __matmul__(self, other):
        return ops.MatMul.apply(self, self._coerce(other))

    def __rmatmul__(self, other):
        return ops.MatMul.apply(self._coerce(other), self)

    def __getitem__(self, index):
        return ops.GetItem.apply(self, index=_unwrap_index(index))

    # Comparisons produce boolean tensors and are never differentiable.
    def __lt__(self, other):
        return Tensor._wrap(self.data < _as_array(other))

    def __le__(self, other):
        return Tensor._wrap(self.data <= _as_array(other))

    def __gt__(self, other):
        return Tensor._wrap(self.data > _as_array(other))

    def __ge__(self, other):
        return Tensor._wrap(self.data >= _as_array(other))

    def __eq__(self, other):  # type: ignore[override]
        return Tensor._wrap(self.data == _as_array(other))

    def __ne__(self, other):  # type: ignore[override]
        return Tensor._wrap(self.data != _as_array(other))

    __hash__ = object.__hash__

    def __bool__(self):
        return bool(self.data)

    # -- reductions -----------------------------------------------------------

    def sum(self, axis=None, keepdims: bool = False) -> Tensor:
        return ops.Sum.apply(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims: bool = False) -> Tensor:
        return ops.Mean.apply(self, axis=axis, keepdims=keepdims)

    def max(self, axis=None, keepdims: bool = False) -> Tensor:
        return ops.Max.apply(self, axis=axis, keepdims=keepdims)

    def min(self, axis=None, keepdims: bool = False) -> Tensor:
        return -ops.Max.apply(-self, axis=axis, keepdims=keepdims)

    def var(self, axis=None, keepdims: bool = False, ddof: int = 0) -> Tensor:
        """Variance with NumPy semantics (`ddof=0` is the biased estimator)."""
        mu = self.mean(axis=axis, keepdims=True)
        sq = (self - mu) ** 2
        n = self.data.size if axis is None else int(np.prod([self.shape[a] for a in _axes(axis, self.ndim)]))
        return sq.sum(axis=axis, keepdims=keepdims) / (n - ddof)

    def std(self, axis=None, keepdims: bool = False, ddof: int = 0) -> Tensor:
        return self.var(axis=axis, keepdims=keepdims, ddof=ddof) ** 0.5

    def argmax(self, axis=None) -> Tensor:
        return Tensor._wrap(np.argmax(self.data, axis=axis))

    def argmin(self, axis=None) -> Tensor:
        return Tensor._wrap(np.argmin(self.data, axis=axis))

    # -- elementwise -----------------------------------------------------------

    def exp(self):
        return ops.Exp.apply(self)

    def log(self):
        return ops.Log.apply(self)

    def sqrt(self):
        return ops.PowScalar.apply(self, exponent=0.5)

    def abs(self):
        return ops.Abs.apply(self)

    def tanh(self):
        return ops.Tanh.apply(self)

    def sigmoid(self):
        return ops.Sigmoid.apply(self)

    def relu(self):
        return ops.ReLU.apply(self)

    def clip(self, min=None, max=None):
        return ops.Clip.apply(self, lo=min, hi=max)

    def softmax(self, axis: int = -1):
        return ops.Softmax.apply(self, axis=axis)

    def log_softmax(self, axis: int = -1):
        return ops.LogSoftmax.apply(self, axis=axis)

    def masked_fill(self, mask, value: float) -> Tensor:
        mask = _as_array(mask).astype(bool)
        return ops.Where.apply(Tensor._wrap(mask), Tensor._wrap(np.asarray(value, dtype=self.dtype)), self)

    # -- shape ---------------------------------------------------------------

    def reshape(self, *shape) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return ops.Reshape.apply(self, shape=shape)

    view = reshape

    def flatten(self, start_axis: int = 0) -> Tensor:
        start_axis %= max(self.ndim, 1)
        return self.reshape(*self.shape[:start_axis], -1)

    def transpose(self, *axes) -> Tensor:
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        return ops.Transpose.apply(self, axes=axes or None)

    permute = transpose

    @property
    def T(self) -> Tensor:
        return self.transpose()

    def swapaxes(self, a: int, b: int) -> Tensor:
        axes = list(range(self.ndim))
        axes[a], axes[b] = axes[b], axes[a]
        return self.transpose(*axes)

    def unsqueeze(self, axis: int) -> Tensor:
        return self.reshape(np.expand_dims(self.data, axis).shape)

    def squeeze(self, axis=None) -> Tensor:
        return self.reshape(np.squeeze(self.data, axis).shape)

    def broadcast_to(self, shape) -> Tensor:
        return ops.BroadcastTo.apply(self, shape=tuple(shape))

    expand = broadcast_to


def _axes(axis, ndim: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, int):
        axis = (axis,)
    return tuple(a % ndim for a in axis)


def _unwrap_index(index):
    if isinstance(index, Tensor):
        return index.data
    if isinstance(index, tuple):
        return tuple(i.data if isinstance(i, Tensor) else i for i in index)
    if isinstance(index, list):
        return np.asarray(index)
    return index


from . import ops  # noqa: E402  (ops needs Tensor and Function defined first)
