"""The global random generator used by initialisers, dropout and data shuffling."""

from __future__ import annotations

import numpy as np

_rng = np.random.default_rng()


def manual_seed(seed: int) -> None:
    """Make every random choice in tensile reproducible."""
    global _rng
    _rng = np.random.default_rng(seed)


def default_rng() -> np.random.Generator:
    return _rng
