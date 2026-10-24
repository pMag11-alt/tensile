"""Downloaders for the datasets used by the examples. Files are cached in ~/.cache/tensile."""

from __future__ import annotations

import gzip
import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np

CACHE = Path(os.environ.get("TENSILE_CACHE", Path.home() / ".cache" / "tensile"))

_MNIST_MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]
_MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def _download(urls: list[str], dest: Path, md5: str | None = None) -> Path:
    if dest.exists() and (md5 is None or _md5(dest) == md5):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in urls:
        try:
            print(f"downloading {url}")
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                f.write(r.read())
            if md5 is not None and _md5(tmp) != md5:
                raise ValueError("checksum mismatch")
            tmp.replace(dest)
            return dest
        except Exception as e:  # try the next mirror
            errors.append(f"{url}: {e}")
    raise RuntimeError("download failed:\n  " + "\n  ".join(errors))


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    ndim = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def mnist(root: Path | str | None = None):
    """Return (x_train, y_train, x_test, y_test): uint8 images (N, 28, 28) and int64 labels."""
    root = Path(root) if root else CACHE / "mnist"
    arrays = []
    for name, md5 in _MNIST_FILES.items():
        path = _download([m + name for m in _MNIST_MIRRORS], root / name, md5)
        arrays.append(_read_idx(path))
    x_train, y_train, x_test, y_test = arrays
    return x_train, y_train.astype(np.int64), x_test, y_test.astype(np.int64)


def tiny_shakespeare(root: Path | str | None = None) -> str:
    """~1 MB of Shakespeare plays, as used by char-rnn."""
    root = Path(root) if root else CACHE
    path = _download([_SHAKESPEARE_URL], root / "tinyshakespeare.txt")
    return path.read_text(encoding="utf-8")


def spiral(n_per_class: int = 100, classes: int = 3, noise: float = 0.2, seed: int = 0):
    """The classic CS231n spiral: (X of shape (N, 2), y of shape (N,))."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c in range(classes):
        r = np.linspace(0.0, 1.0, n_per_class)
        theta = np.linspace(c * 4.0, (c + 1) * 4.0, n_per_class) + rng.normal(0, noise, n_per_class)
        xs.append(np.stack([r * np.sin(theta), r * np.cos(theta)], axis=1))
        ys.append(np.full(n_per_class, c))
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.int64)
