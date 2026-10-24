"""Datasets and mini-batch loading."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence

import numpy as np

from .random import default_rng
from .tensor import Tensor


class Dataset:
    def __len__(self) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    def __getitem__(self, i):  # pragma: no cover - abstract
        raise NotImplementedError


class TensorDataset(Dataset):
    """Wraps arrays that share a first dimension. Supports fast batched indexing."""

    def __init__(self, *arrays) -> None:
        self.arrays = [a.data if isinstance(a, Tensor) else np.asarray(a) for a in arrays]
        n = len(self.arrays[0])
        if any(len(a) != n for a in self.arrays):
            raise ValueError("all arrays must have the same length")

    def __len__(self) -> int:
        return len(self.arrays[0])

    def __getitem__(self, i):
        return tuple(a[i] for a in self.arrays)

    def get_batch(self, indices: np.ndarray):
        return tuple(a[indices] for a in self.arrays)


def default_collate(samples: Sequence) -> tuple:
    return tuple(np.stack(field) for field in zip(*samples))


class DataLoader:
    """Iterates a Dataset in mini-batches of Tensors.

    If the dataset has `get_batch(indices)` (TensorDataset does), whole batches
    are gathered with one fancy-index per array instead of per-sample Python calls.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        collate_fn: Callable | None = None,
        transform: Callable | None = None,
    ) -> None:
        self.dataset, self.batch_size = dataset, batch_size
        self.shuffle, self.drop_last = shuffle, drop_last
        self.collate_fn = collate_fn or default_collate
        self.transform = transform

    def __len__(self) -> int:
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)

    def __iter__(self) -> Iterator[tuple[Tensor, ...]]:
        n = len(self.dataset)
        order = default_rng().permutation(n) if self.shuffle else np.arange(n)
        for start in range(0, n, self.batch_size):
            idx = order[start : start + self.batch_size]
            if self.drop_last and len(idx) < self.batch_size:
                break
            if hasattr(self.dataset, "get_batch"):
                batch = self.dataset.get_batch(idx)
            else:
                batch = self.collate_fn([self.dataset[int(i)] for i in idx])
            if self.transform is not None:
                batch = self.transform(*batch)
            yield tuple(b if isinstance(b, Tensor) else Tensor(b) for b in batch)


def random_split(dataset: TensorDataset, lengths: Sequence[int]) -> list[TensorDataset]:
    if sum(lengths) != len(dataset):
        raise ValueError("lengths must sum to the dataset size")
    order = default_rng().permutation(len(dataset))
    parts, start = [], 0
    for n in lengths:
        parts.append(TensorDataset(*dataset.get_batch(order[start : start + n])))
        start += n
    return parts
