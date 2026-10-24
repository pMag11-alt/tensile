"""Train an MLP or a small CNN on MNIST.

python examples/mnist.py --model cnn --epochs 3
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import tensile as ts
import tensile.functional as F
import tensile.nn as nn
import tensile.optim as optim
from tensile.data import DataLoader, TensorDataset
from tensile.datasets import mnist


def make_model(kind: str) -> nn.Module:
    if kind == "mlp":
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),  # 16 x 14 x 14
        nn.Conv2d(16, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),  # 32 x 7 x 7
        nn.Flatten(),
        nn.Dropout(0.25),
        nn.Linear(32 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


@ts.no_grad()
def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, batch: int = 1000) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(x), batch):
        correct += int((model(ts.tensor(x[i : i + batch])).argmax(axis=1).data == y[i : i + batch]).sum())
    model.train()
    return correct / len(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mlp", "cnn"], default="cnn")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", help="write the trained weights to this .npz file")
    ap.add_argument("--min-acc", type=float, help="exit with an error if final test accuracy is lower (for CI)")
    args = ap.parse_args()

    ts.manual_seed(args.seed)
    x_train, y_train, x_test, y_test = mnist()
    norm = lambda x: ((x.astype(np.float32) / 255.0 - 0.1307) / 0.3081)[:, None]  # noqa: E731
    x_train, x_test = norm(x_train), norm(x_test)

    model = make_model(args.model)
    print(model)
    print(f"{model.num_parameters():,} parameters\n")
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    sched = optim.LambdaLR(opt, optim.warmup_cosine(warmup=100, total=args.epochs * len(loader), floor=0.05))

    for epoch in range(1, args.epochs + 1):
        start, running, seen = time.perf_counter(), 0.0, 0
        for step, (xb, yb) in enumerate(loader, 1):
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item() * len(yb)
            seen += len(yb)
            if step % 100 == 0:
                print(f"\repoch {epoch} step {step}/{len(loader)} loss {running / seen:.4f}", end="", flush=True)
        acc = evaluate(model, x_test, y_test)
        print(
            f"\repoch {epoch}: train loss {running / seen:.4f} | test acc {acc:.2%} "
            f"| {time.perf_counter() - start:.0f}s"
        )

    if args.min_acc is not None and acc < args.min_acc:
        raise SystemExit(f"test accuracy {acc:.2%} is below the required {args.min_acc:.2%}")
    if args.save:
        ts.save(model.state_dict(), args.save)
        print(f"saved weights to {args.save}")


if __name__ == "__main__":
    main()
