"""Optimisers and learning-rate schedules.

Optimisers keep PyTorch's update rules exactly (the parity tests check this),
including parameter groups so e.g. biases can skip weight decay.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np


class Optimizer:
    defaults: dict = {}

    def __init__(self, params, **hyper) -> None:
        params = list(params)
        if not params:
            raise ValueError("optimizer got an empty parameter list")
        groups = params if isinstance(params[0], dict) else [{"params": params}]
        self.param_groups: list[dict] = []
        for group in groups:
            g = {**self.defaults, **hyper, **group}
            g["params"] = list(g["params"])
            self.param_groups.append(g)
        self.state: dict[int, dict] = {}

    def zero_grad(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                p.grad = None

    def step(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    st = self.state.setdefault(id(p), {})
                    p.data = self._update(p.data, p.grad.data.astype(p.dtype, copy=False), st, group).astype(
                        p.dtype, copy=False
                    )

    def _update(self, w: np.ndarray, g: np.ndarray, state: dict, group: dict) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    @property
    def lr(self) -> float:
        return self.param_groups[0]["lr"]


class SGD(Optimizer):
    defaults = {"lr": 1e-2, "momentum": 0.0, "nesterov": False, "weight_decay": 0.0, "dampening": 0.0}

    def _update(self, w, g, state, group):
        if group["weight_decay"]:
            g = g + group["weight_decay"] * w
        mu = group["momentum"]
        if mu:
            buf = state.get("momentum")
            buf = g.copy() if buf is None else mu * buf + (1 - group["dampening"]) * g
            state["momentum"] = buf
            g = g + mu * buf if group["nesterov"] else buf
        return w - group["lr"] * g


class Adam(Optimizer):
    defaults = {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0}
    decoupled = False

    def _update(self, w, g, state, group):
        b1, b2 = group["betas"]
        lr, wd = group["lr"], group["weight_decay"]
        if wd and self.decoupled:
            w = w * (1 - lr * wd)
        elif wd:
            g = g + wd * w
        t = state["step"] = state.get("step", 0) + 1
        m = state["m"] = b1 * state.get("m", 0.0) + (1 - b1) * g
        v = state["v"] = b2 * state.get("v", 0.0) + (1 - b2) * g * g
        m_hat = m / (1 - b1**t)
        v_hat = v / (1 - b2**t)
        return w - lr * m_hat / (np.sqrt(v_hat) + group["eps"])


class AdamW(Adam):
    """Adam with decoupled weight decay (Loshchilov & Hutter)."""

    defaults = {**Adam.defaults, "weight_decay": 1e-2}
    decoupled = True


class RMSprop(Optimizer):
    defaults = {"lr": 1e-2, "alpha": 0.99, "eps": 1e-8, "weight_decay": 0.0, "momentum": 0.0}

    def _update(self, w, g, state, group):
        if group["weight_decay"]:
            g = g + group["weight_decay"] * w
        sq = state["square_avg"] = group["alpha"] * state.get("square_avg", 0.0) + (1 - group["alpha"]) * g * g
        step = g / (np.sqrt(sq) + group["eps"])
        if group["momentum"]:
            step = state["buf"] = group["momentum"] * state.get("buf", 0.0) + step
        return w - group["lr"] * step


# ---------------------------------------------------------------------------
# Learning-rate schedules
# ---------------------------------------------------------------------------


class LRScheduler:
    """Call `step()` once per epoch (or per iteration, if that's how you count)."""

    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.last_epoch = 0

    def get_lr(self, base_lr: float, epoch: int) -> float:  # pragma: no cover
        raise NotImplementedError

    def step(self) -> None:
        self.last_epoch += 1
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = self.get_lr(base, self.last_epoch)


class StepLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1) -> None:
        super().__init__(optimizer)
        self.step_size, self.gamma = step_size, gamma

    def get_lr(self, base_lr, epoch):
        return base_lr * self.gamma ** (epoch // self.step_size)


class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0) -> None:
        super().__init__(optimizer)
        self.T_max, self.eta_min = T_max, eta_min

    def get_lr(self, base_lr, epoch):
        t = min(epoch, self.T_max)
        return self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * t / self.T_max)) / 2


class LambdaLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, lr_lambda: Callable[[int], float]) -> None:
        super().__init__(optimizer)
        self.lr_lambda = lr_lambda
        for group, base in zip(optimizer.param_groups, self.base_lrs):
            group["lr"] = base * lr_lambda(0)

    def get_lr(self, base_lr, epoch):
        return base_lr * self.lr_lambda(epoch)


def warmup_cosine(warmup: int, total: int, floor: float = 0.1) -> Callable[[int], float]:
    """Linear warm-up then cosine decay to `floor` x the base rate; for LambdaLR."""

    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min((step - warmup) / max(total - warmup, 1), 1.0)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return f


__all__ = [
    "SGD",
    "Adam",
    "AdamW",
    "RMSprop",
    "Optimizer",
    "LRScheduler",
    "StepLR",
    "CosineAnnealingLR",
    "LambdaLR",
    "warmup_cosine",
]
