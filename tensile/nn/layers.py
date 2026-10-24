"""Standard layers built on tensile.functional."""

from __future__ import annotations

import math

import numpy as np

from .. import functional as F
from ..tensor import DEFAULT_DTYPE, Tensor
from . import init
from .module import Module, ModuleList, Parameter


def _empty(*shape) -> Parameter:
    return Parameter(np.zeros(shape, dtype=DEFAULT_DTYPE))


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = init.kaiming_uniform_(_empty(out_features, in_features))
        self.bias = (
            init.uniform_(_empty(out_features), -1 / math.sqrt(in_features), 1 / math.sqrt(in_features))
            if bias
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, bias: bool = True):
        super().__init__()
        kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding = (kh, kw), stride, padding
        self.weight = init.kaiming_uniform_(_empty(out_channels, in_channels, kh, kw))
        bound = 1 / math.sqrt(in_channels * kh * kw)
        self.bias = init.uniform_(_empty(out_channels), -bound, bound) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}"
        )


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None) -> None:
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride

    def forward(self, x: Tensor) -> Tensor:
        return F.max_pool2d(x, self.kernel_size, self.stride)

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride or self.kernel_size}"


class AvgPool2d(MaxPool2d):
    def forward(self, x: Tensor) -> Tensor:
        return F.avg_pool2d(x, self.kernel_size, self.stride)


class Flatten(Module):
    def __init__(self, start_axis: int = 1) -> None:
        super().__init__()
        self.start_axis = start_axis

    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(self.start_axis)


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return F.dropout(x, self.p, self.training)

    def extra_repr(self) -> str:
        return f"p={self.p}"


class _Activation(Module):
    fn = staticmethod(lambda x: x)

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x)


class ReLU(_Activation):
    fn = staticmethod(F.relu)


class GELU(_Activation):
    fn = staticmethod(F.gelu)


class Tanh(_Activation):
    fn = staticmethod(F.tanh)


class Sigmoid(_Activation):
    fn = staticmethod(F.sigmoid)


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        return F.leaky_relu(x, self.negative_slope)


class Identity(_Activation):
    pass


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps: float = 1e-5) -> None:
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps
        self.weight = Parameter(np.ones(self.normalized_shape, dtype=DEFAULT_DTYPE))
        self.bias = Parameter(np.zeros(self.normalized_shape, dtype=DEFAULT_DTYPE))

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}"


class _BatchNorm(Module):
    expected_ndim: tuple[int, ...] = ()

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.num_features, self.eps, self.momentum = num_features, eps, momentum
        self.weight = Parameter(np.ones(num_features, dtype=DEFAULT_DTYPE))
        self.bias = Parameter(np.zeros(num_features, dtype=DEFAULT_DTYPE))
        self.register_buffer("running_mean", np.zeros(num_features, dtype=DEFAULT_DTYPE))
        self.register_buffer("running_var", np.ones(num_features, dtype=DEFAULT_DTYPE))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim not in self.expected_ndim:
            raise ValueError(f"{type(self).__name__} expects {self.expected_ndim}-D input, got shape {x.shape}")
        return F.batch_norm(
            x, self.running_mean, self.running_var, self.weight, self.bias, self.training, self.momentum, self.eps
        )

    def extra_repr(self) -> str:
        return f"{self.num_features}, eps={self.eps}, momentum={self.momentum}"


class BatchNorm1d(_BatchNorm):
    expected_ndim = (2, 3)


class BatchNorm2d(_BatchNorm):
    expected_ndim = (4,)


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.weight = init.normal_(_empty(num_embeddings, embedding_dim))

    def forward(self, indices) -> Tensor:
        return F.embedding(indices, self.weight)

    def extra_repr(self) -> str:
        return f"{self.num_embeddings}, {self.embedding_dim}"


class MultiheadAttention(Module):
    """Multi-head self-attention over (batch, time, embed) inputs."""

    def __init__(self, embed_dim: int, num_heads: int, causal: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim, self.num_heads, self.causal = embed_dim, num_heads, causal
        self.qkv = Linear(embed_dim, 3 * embed_dim)
        self.proj = Linear(embed_dim, embed_dim)
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        b, t, c = x.shape
        h = self.num_heads
        qkv = self.qkv(x).reshape(b, t, 3, h, c // h).transpose(2, 0, 3, 1, 4)  # 3, B, H, T, D
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = F.scaled_dot_product_attention(q, k, v, causal=self.causal)  # B, H, T, D
        y = y.transpose(0, 2, 1, 3).reshape(b, t, c)
        return self.dropout(self.proj(y))

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, causal={self.causal}"


class TransformerBlock(Module):
    """Pre-norm transformer block: x + attn(ln(x)), then x + mlp(ln(x))."""

    def __init__(self, embed_dim: int, num_heads: int, causal: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiheadAttention(embed_dim, num_heads, causal=causal, dropout=dropout)
        self.ln2 = LayerNorm(embed_dim)
        self.mlp = _MLP(embed_dim, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class _MLP(Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.fc = Linear(dim, 4 * dim)
        self.act = GELU()
        self.proj = Linear(4 * dim, dim)
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.proj(self.act(self.fc(x))))


__all__ = [
    "AvgPool2d",
    "BatchNorm1d",
    "BatchNorm2d",
    "Conv2d",
    "Dropout",
    "Embedding",
    "Flatten",
    "GELU",
    "Identity",
    "LayerNorm",
    "LeakyReLU",
    "Linear",
    "MaxPool2d",
    "ModuleList",
    "MultiheadAttention",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "TransformerBlock",
]
