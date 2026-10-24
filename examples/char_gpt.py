"""A character-level GPT trained on Shakespeare, in pure NumPy via tensile.

    python examples/char_gpt.py --steps 3000          # train (~5 min on a laptop CPU)
    python examples/char_gpt.py --sample --tokens 500  # generate from the checkpoint
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import tensile as ts
import tensile.functional as F
import tensile.nn as nn
import tensile.optim as optim
from tensile.datasets import tiny_shakespeare

CHECKPOINT = Path("checkpoints/char_gpt.npz")


class GPT(nn.Module):
    def __init__(self, vocab: int, block: int, dim: int, heads: int, layers: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = block
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(block, dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(nn.TransformerBlock(dim, heads, causal=True, dropout=dropout) for _ in range(layers))
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        for name, p in self.named_parameters():  # GPT-2 style init
            if name.endswith("weight") and p.ndim == 2:
                nn.init.normal_(p, 0.0, 0.02)
            if name.endswith("proj.weight"):
                nn.init.normal_(p, 0.0, 0.02 / np.sqrt(2 * layers))

    def forward(self, idx: np.ndarray) -> ts.Tensor:
        t = idx.shape[1]
        h = self.drop(self.tok(idx) + self.pos(np.arange(t)))
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln(h))

    @ts.no_grad()
    def generate(self, idx: np.ndarray, n: int, temperature: float = 0.8, top_k: int | None = 20) -> np.ndarray:
        rng = ts.default_rng()
        for _ in range(n):
            logits = self(idx[:, -self.block :]).data[:, -1, :] / temperature
            if top_k:
                kth = np.sort(logits, axis=-1)[:, -top_k][:, None]
                logits = np.where(logits < kth, -np.inf, logits)
            p = np.exp(logits - logits.max(axis=-1, keepdims=True))
            p /= p.sum(axis=-1, keepdims=True)
            nxt = np.array([[rng.choice(p.shape[-1], p=row)] for row in p])
            idx = np.concatenate([idx, nxt], axis=1)
        return idx


def batches(data: np.ndarray, block: int, size: int):
    rng = ts.default_rng()
    while True:
        starts = rng.integers(0, len(data) - block - 1, size)
        x = np.stack([data[s : s + block] for s in starts])
        y = np.stack([data[s + 1 : s + block + 1] for s in starts])
        yield x, y


@ts.no_grad()
def estimate_loss(model: GPT, data: np.ndarray, block: int, iters: int = 20) -> float:
    model.eval()
    gen = batches(data, block, 32)
    losses = [F.cross_entropy(model(x), y).item() for x, y in (next(gen) for _ in range(iters))]
    model.train()
    return float(np.mean(losses))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample", action="store_true", help="skip training, sample from the checkpoint")
    ap.add_argument("--tokens", type=int, default=600)
    ap.add_argument("--prompt", default="ROMEO:\n")
    ap.add_argument("--log", help="write the loss curve to this JSON file")
    args = ap.parse_args()

    ts.manual_seed(args.seed)
    text = tiny_shakespeare()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    split = int(0.9 * len(data))
    train, val = data[:split], data[split:]

    model = GPT(len(chars), args.block, args.dim, args.heads, args.layers)
    if args.sample:
        model.load_state_dict(ts.load(CHECKPOINT))
    else:
        print(f"{model.num_parameters():,} parameters, vocab {len(chars)}, {len(train):,} training chars")
        decay = [p for n, p in model.named_parameters() if p.ndim == 2]
        no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
        opt = optim.AdamW(
            [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
            lr=args.lr,
            betas=(0.9, 0.99),
        )
        sched = optim.LambdaLR(opt, optim.warmup_cosine(warmup=100, total=args.steps, floor=0.1))
        gen = batches(train, args.block, args.batch_size)
        history, start = [], time.perf_counter()
        for step in range(1, args.steps + 1):
            x, y = next(gen)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % 250 == 0 or step == 1:
                val_loss = estimate_loss(model, val, args.block)
                elapsed = time.perf_counter() - start
                history.append({"step": step, "train": loss.item(), "val": val_loss, "seconds": elapsed})
                print(f"step {step:5d} | train {loss.item():.3f} | val {val_loss:.3f} | {elapsed:.0f}s")
        CHECKPOINT.parent.mkdir(exist_ok=True)
        ts.save(model.state_dict(), CHECKPOINT)
        if args.log:
            Path(args.log).write_text(json.dumps(history, indent=1))

    model.eval()
    prompt = np.array([[stoi[c] for c in args.prompt]])
    out = model.generate(prompt, args.tokens)[0]
    print("\n" + "".join(chars[i] for i in out))


if __name__ == "__main__":
    main()
