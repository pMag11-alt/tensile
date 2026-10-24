"""Fit a 3-class spiral with a small MLP and plot the decision boundary.

python examples/spiral.py --out docs/spiral.png
"""

from __future__ import annotations

import argparse

import numpy as np

import tensile as ts
import tensile.functional as F
import tensile.nn as nn
import tensile.optim as optim
from tensile.data import DataLoader, TensorDataset
from tensile.datasets import spiral


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--out", default="spiral.png")
    args = ap.parse_args()

    ts.manual_seed(0)
    X, y = spiral(200, 3, noise=0.25)
    model = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))
    opt = optim.Adam(model.parameters(), lr=0.01)
    sched = optim.CosineAnnealingLR(opt, T_max=args.epochs)
    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
    for epoch in range(1, args.epochs + 1):
        for xb, yb in loader:
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
        sched.step()
        if epoch % 50 == 0:
            with ts.no_grad():
                logits = model(ts.tensor(X))
                acc = (logits.argmax(axis=1).data == y).mean()
                print(f"epoch {epoch}: loss {F.cross_entropy(logits, y).item():.4f} acc {acc:.1%}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    xx, yy = np.meshgrid(np.linspace(-1.1, 1.1, 400), np.linspace(-1.1, 1.1, 400))
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    with ts.no_grad():
        zz = model(ts.tensor(grid)).argmax(axis=1).data.reshape(xx.shape)
    colors = ["#2a6fdb", "#e0593b", "#2fa36b"]
    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5, 2.5], cmap=ListedColormap(colors), alpha=0.25)
    ax.scatter(X[:, 0], X[:, 1], c=[colors[c] for c in y], s=12, edgecolors="white", linewidths=0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("2-64-64-3 MLP on a 3-class spiral", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
