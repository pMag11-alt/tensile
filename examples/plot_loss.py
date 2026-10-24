"""Plot the loss curve written by `char_gpt.py --log`.

    python examples/plot_loss.py docs/char_gpt_loss.json docs/char_gpt_loss.png
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main(src: str, dst: str) -> None:
    history = json.load(open(src))
    steps = [h["step"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 3.4), dpi=120)
    ax.plot(steps, [h["train"] for h in history], label="train", color="#2a6fdb")
    ax.plot(steps, [h["val"] for h in history], label="validation", color="#e0593b")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy (nats/char)")
    ax.set_title("char-level GPT on Tiny Shakespeare (0.8M params, NumPy CPU)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(dst)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
