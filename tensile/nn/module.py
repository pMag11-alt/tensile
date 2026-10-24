"""Module: the base class for layers and models."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator

import numpy as np

from ..tensor import Tensor


class Parameter(Tensor):
    """A Tensor that a Module registers as trainable."""

    def __init__(self, data, requires_grad: bool = True):
        super().__init__(data, requires_grad=requires_grad)

    def __repr__(self) -> str:
        return "Parameter containing:\n" + super().__repr__()


class Module:
    """Base class: assign Parameters, sub-Modules or buffers as attributes and they're tracked."""

    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "_buffers", OrderedDict())
        object.__setattr__(self, "training", True)

    # -- registration ----------------------------------------------------------

    def __setattr__(self, name: str, value) -> None:
        if "_parameters" not in self.__dict__:
            raise AttributeError(f"call super().__init__() before assigning {name!r} in {type(self).__name__}")
        for registry in (self._parameters, self._modules, self._buffers):
            registry.pop(name, None)
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str):
        d = self.__dict__
        for registry in ("_parameters", "_modules", "_buffers"):
            if registry in d and name in d[registry]:
                return d[registry][name]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __delattr__(self, name: str) -> None:
        for registry in (self._parameters, self._modules, self._buffers):
            if name in registry:
                del registry[name]
                return
        object.__delattr__(self, name)

    def register_buffer(self, name: str, value: np.ndarray) -> None:
        """Non-trainable state (e.g. BatchNorm running stats) that is saved in state_dict()."""
        self._buffers[name] = np.asarray(value)

    def add_module(self, name: str, module: Module) -> None:
        self._modules[name] = module

    # -- traversal --------------------------------------------------------------

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        seen: set[int] = set()
        for name, module in self.named_modules(prefix):
            for pname, p in module._parameters.items():
                if id(p) not in seen:  # shared (tied) weights are yielded once
                    seen.add(id(p))
                    yield (f"{name}.{pname}" if name else pname), p

    def parameters(self) -> Iterator[Parameter]:
        for _, p in self.named_parameters():
            yield p

    def named_buffers(self, prefix: str = "") -> Iterator[tuple[str, np.ndarray]]:
        for name, module in self.named_modules(prefix):
            for bname, b in module._buffers.items():
                yield (f"{name}.{bname}" if name else bname), b

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, Module]]:
        yield prefix, self
        for name, child in self._modules.items():
            yield from child.named_modules(f"{prefix}.{name}" if prefix else name)

    def modules(self) -> Iterator[Module]:
        for _, m in self.named_modules():
            yield m

    def children(self) -> Iterator[Module]:
        yield from self._modules.values()

    def num_parameters(self) -> int:
        return sum(p.size for p in self.parameters())

    # -- modes --------------------------------------------------------------

    def train(self, mode: bool = True) -> Module:
        for m in self.modules():
            object.__setattr__(m, "training", mode)
        return self

    def eval(self) -> Module:
        return self.train(False)

    def to(self, dtype) -> Module:
        """Cast every floating-point parameter and buffer to `dtype`, in place."""
        dtype = np.dtype(dtype)
        for p in self.parameters():
            p.data = p.data.astype(dtype)
            p.grad = None
        for m in self.modules():
            for name, b in m._buffers.items():
                if b.dtype.kind == "f":
                    m._buffers[name] = b.astype(dtype)
        return self

    def double(self) -> Module:
        return self.to(np.float64)

    def float(self) -> Module:
        return self.to(np.float32)

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def apply(self, fn) -> Module:
        for m in self.modules():
            fn(m)
        return self

    # -- state ------------------------------------------------------------

    def state_dict(self) -> OrderedDict[str, np.ndarray]:
        state: OrderedDict[str, np.ndarray] = OrderedDict()
        for name, p in self.named_parameters():
            state[name] = p.data.copy()
        for name, b in self.named_buffers():
            state[name] = b.copy()
        return state

    def load_state_dict(self, state, strict: bool = True) -> None:
        params = dict(self.named_parameters())
        owners = {}
        for mname, module in self.named_modules():
            for bname in module._buffers:
                owners[f"{mname}.{bname}" if mname else bname] = (module, bname)
        expected = set(params) | set(owners)
        missing, unexpected = expected - set(state), set(state) - expected
        if strict and (missing or unexpected):
            raise KeyError(f"state_dict mismatch: missing={sorted(missing)} unexpected={sorted(unexpected)}")
        for key, value in state.items():
            value = np.asarray(value)
            if key in params:
                p = params[key]
                if p.shape != value.shape:
                    raise ValueError(f"{key}: expected shape {p.shape}, got {value.shape}")
                p.data = value.astype(p.dtype, copy=True)
            elif key in owners:
                module, bname = owners[key]
                module._buffers[bname] = value.copy()

    # -- calling ----------------------------------------------------------

    def forward(self, *args, **kwargs):  # pragma: no cover - abstract
        raise NotImplementedError(f"{type(self).__name__} must implement forward()")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}({self.extra_repr()}"]
        if self._modules:
            for name, child in self._modules.items():
                child_repr = repr(child).replace("\n", "\n  ")
                lines.append(f"  ({name}): {child_repr}")
            return "\n".join(lines) + "\n)"
        return lines[0] + ")"


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        super().__init__()
        for i, layer in enumerate(layers):
            self.add_module(str(i), layer)

    def forward(self, x):
        for layer in self._modules.values():
            x = layer(x)
        return x

    def __getitem__(self, i: int) -> Module:
        return list(self._modules.values())[i]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())


class ModuleList(Module):
    def __init__(self, modules=()) -> None:
        super().__init__()
        for m in modules:
            self.append(m)

    def append(self, module: Module) -> None:
        self.add_module(str(len(self._modules)), module)

    def __getitem__(self, i: int) -> Module:
        return list(self._modules.values())[i]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())
