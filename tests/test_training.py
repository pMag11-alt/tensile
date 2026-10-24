"""End-to-end: optimisers, schedulers, data loading and real (small) learning problems."""

import math

import numpy as np
import pytest

import tensile as ts
import tensile.functional as F
import tensile.nn as nn
import tensile.optim as optim
from tensile.data import DataLoader, Dataset, TensorDataset, random_split
from tensile.datasets import spiral


@pytest.mark.parametrize(
    "make",
    [
        lambda p: optim.SGD(p, lr=0.1),
        lambda p: optim.SGD(p, lr=0.05, momentum=0.9),
        lambda p: optim.Adam(p, lr=0.1),
        lambda p: optim.AdamW(p, lr=0.1, weight_decay=0.0),
        lambda p: optim.RMSprop(p, lr=0.05),
    ],
)
def test_optimizers_minimize_a_quadratic(make):
    target = np.array([1.0, -2.0, 3.0])
    w = ts.zeros(3, requires_grad=True)
    opt = make([w])
    for _ in range(300):
        opt.zero_grad()
        ((w - ts.tensor(target)) ** 2).sum().backward()
        opt.step()
    np.testing.assert_allclose(w.data, target, atol=1e-2)


def test_param_groups_with_different_lr():
    a = ts.ones(1, requires_grad=True)
    b = ts.ones(1, requires_grad=True)
    opt = optim.SGD([{"params": [a]}, {"params": [b], "lr": 0.0}], lr=0.5)
    (a + b).sum().backward()
    opt.step()
    assert a.item() == 0.5 and b.item() == 1.0


def test_schedulers():
    w = ts.ones(1, requires_grad=True)
    opt = optim.SGD([w], lr=1.0)
    sched = optim.StepLR(opt, step_size=2, gamma=0.5)
    lrs = []
    for _ in range(5):
        sched.step()
        lrs.append(opt.lr)
    assert lrs == [1.0, 0.5, 0.5, 0.25, 0.25]

    opt = optim.SGD([w], lr=1.0)
    cos = optim.CosineAnnealingLR(opt, T_max=10)
    for _ in range(5):
        cos.step()
    assert opt.lr == pytest.approx(0.5)

    opt = optim.SGD([w], lr=1.0)
    warm = optim.LambdaLR(opt, optim.warmup_cosine(warmup=4, total=20, floor=0.1))
    assert opt.lr == pytest.approx(0.25)
    for _ in range(3):
        warm.step()
    assert opt.lr == pytest.approx(1.0)
    for _ in range(30):
        warm.step()
    assert opt.lr == pytest.approx(0.1)


def test_dataloader_batching():
    ds = TensorDataset(np.arange(10), np.arange(10) * 2)
    batches = list(DataLoader(ds, batch_size=4))
    assert [len(b[0]) for b in batches] == [4, 4, 2]
    assert batches[0][1].tolist() == [0, 2, 4, 6]
    assert len(DataLoader(ds, batch_size=4)) == 3
    assert len(DataLoader(ds, batch_size=4, drop_last=True)) == 2
    assert [len(b[0]) for b in DataLoader(ds, batch_size=4, drop_last=True)] == [4, 4]


def test_dataloader_shuffle_is_a_permutation_and_seeded():
    ds = TensorDataset(np.arange(50))
    ts.manual_seed(0)
    first = np.concatenate([b[0].data for b in DataLoader(ds, batch_size=7, shuffle=True)])
    ts.manual_seed(0)
    again = np.concatenate([b[0].data for b in DataLoader(ds, batch_size=7, shuffle=True)])
    assert sorted(first.tolist()) == list(range(50))
    assert first.tolist() != list(range(50))
    np.testing.assert_array_equal(first, again)


def test_dataloader_with_map_style_dataset_and_transform():
    class Squares(Dataset):
        def __len__(self):
            return 5

        def __getitem__(self, i):
            return np.float32(i), np.int64(i * i)

    loader = DataLoader(Squares(), batch_size=2, transform=lambda x, y: (x * 10, y))
    xs, ys = zip(*[(x.tolist(), y.tolist()) for x, y in loader])
    assert xs == ([0.0, 10.0], [20.0, 30.0], [40.0])
    assert ys == ([0, 1], [4, 9], [16])


def test_random_split():
    ds = TensorDataset(np.arange(10))
    a, b = random_split(ds, [7, 3])
    assert len(a) == 7 and len(b) == 3
    assert sorted(np.concatenate([a.arrays[0], b.arrays[0]]).tolist()) == list(range(10))


def test_learns_xor():
    ts.manual_seed(0)
    x = ts.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])
    model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2))
    opt = optim.Adam(model.parameters(), lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()
    assert model(x).argmax(axis=1).tolist() == [0, 1, 1, 0]


def test_learns_the_spiral():
    ts.manual_seed(0)
    X, y = spiral(100, 3)
    model = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))
    opt = optim.Adam(model.parameters(), lr=0.01)
    loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
    for _ in range(60):
        for xb, yb in loader:
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
    acc = (model(ts.tensor(X)).argmax(axis=1).data == y).mean()
    assert acc > 0.97


def test_tiny_cnn_learns_to_detect_stripe_orientation():
    """Horizontal vs vertical stripes: needs the conv layer to learn an oriented edge filter."""
    ts.manual_seed(0)
    rng = np.random.default_rng(0)

    def make(n):
        imgs = rng.normal(0, 0.3, size=(n, 1, 8, 8)).astype(np.float32)
        labels = rng.integers(0, 2, n)
        for img, lab in zip(imgs, labels):
            k = rng.integers(0, 8)
            if lab:
                img[0, :, k] += 2
            else:
                img[0, k, :] += 2
        return imgs, labels

    X, y = make(256)
    model = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(64, 2))
    opt = optim.Adam(model.parameters(), lr=0.01)
    for _ in range(15):
        for xb, yb in DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True):
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
    Xt, yt = make(200)
    model.eval()
    assert (model(ts.tensor(Xt)).argmax(axis=1).data == yt).mean() > 0.95


def test_tiny_transformer_memorizes_a_sequence():
    """A 1-layer causal transformer should learn to continue a fixed cyclic pattern."""
    ts.manual_seed(0)
    vocab, t = 5, 8
    seq = np.array([0, 3, 1, 4, 2] * 4)

    class LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(vocab, 16)
            self.pos = nn.Embedding(t, 16)
            self.block = nn.TransformerBlock(16, 2)
            self.head = nn.Linear(16, vocab)

        def forward(self, idx):
            h = self.tok(idx) + self.pos(np.arange(idx.shape[1]))
            return self.head(self.block(h))

    model = LM()
    opt = optim.AdamW(model.parameters(), lr=0.01)
    starts = np.arange(len(seq) - t)
    x = np.stack([seq[s : s + t] for s in starts])
    y = np.stack([seq[s + 1 : s + t + 1] for s in starts])
    for _ in range(150):
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
    assert math.isfinite(loss.item())
