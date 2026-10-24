"""Forward values and gradients match PyTorch. Skipped when torch isn't installed."""

import numpy as np
import pytest

import tensile as ts
import tensile.functional as F
import tensile.nn as nn
import tensile.optim as optim

torch = pytest.importorskip("torch")
TF = torch.nn.functional

rng = np.random.default_rng(7)


def both(*shapes):
    arrays = [rng.normal(size=s) for s in shapes]
    return (
        [ts.tensor(a, requires_grad=True) for a in arrays],
        [torch.tensor(a, requires_grad=True) for a in arrays],
    )


def assert_parity(ours, theirs, ours_in, theirs_in, atol=1e-8):
    np.testing.assert_allclose(ours.data, theirs.detach().numpy(), atol=atol, rtol=1e-6)
    weight = rng.normal(size=ours.shape)
    (ours * ts.tensor(weight)).sum().backward()
    (theirs * torch.tensor(weight)).sum().backward()
    for a, b in zip(ours_in, theirs_in):
        np.testing.assert_allclose(a.grad.data, b.grad.numpy(), atol=atol, rtol=1e-6)


CASES = {
    "gelu": ((3, 5), lambda x: F.gelu(x), lambda x: TF.gelu(x, approximate="tanh")),
    "softmax": ((3, 5), lambda x: F.softmax(x, -1), lambda x: TF.softmax(x, -1)),
    "log_softmax": ((3, 5), lambda x: F.log_softmax(x, 0), lambda x: TF.log_softmax(x, 0)),
    "sigmoid": ((3, 5), F.sigmoid, torch.sigmoid),
    "leaky_relu": ((3, 5), lambda x: F.leaky_relu(x, 0.2), lambda x: TF.leaky_relu(x, 0.2)),
    "layer_norm": ((2, 3, 6), lambda x: F.layer_norm(x, 6), lambda x: TF.layer_norm(x, (6,))),
    "max_pool": ((2, 3, 8, 8), lambda x: F.max_pool2d(x, 2), lambda x: TF.max_pool2d(x, 2)),
    "max_pool_overlap": ((2, 3, 7, 7), lambda x: F.max_pool2d(x, 3, 2), lambda x: TF.max_pool2d(x, 3, 2)),
    "avg_pool": ((2, 3, 8, 8), lambda x: F.avg_pool2d(x, 2), lambda x: TF.avg_pool2d(x, 2)),
    "var_std": ((4, 4), lambda x: x.var(axis=0, ddof=1) + x.std(axis=1), lambda x: x.var(0) + x.std(1, correction=0)),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_unary_parity(name):
    shape, ours_fn, theirs_fn = CASES[name]
    (a,), (b,) = both(shape)
    assert_parity(ours_fn(a), theirs_fn(b), [a], [b])


@pytest.mark.parametrize("stride,padding", [(1, 0), (1, 1), (2, 1)])
def test_conv2d_parity(stride, padding):
    ours, theirs = both((2, 3, 9, 7), (5, 3, 3, 3), (5,))
    assert_parity(
        F.conv2d(*ours, stride=stride, padding=padding),
        TF.conv2d(*theirs, stride=stride, padding=padding),
        ours,
        theirs,
    )


def test_cross_entropy_parity():
    (x,), (xt,) = both((6, 4))
    target = np.array([0, 1, 3, 3, 2, -1])
    assert_parity(
        F.cross_entropy(x, target, label_smoothing=0.1, ignore_index=-1),
        TF.cross_entropy(xt, torch.tensor(target), label_smoothing=0.1, ignore_index=-1),
        [x],
        [xt],
    )


def test_bce_with_logits_parity():
    (x,), (xt,) = both((6, 4))
    y = (rng.random((6, 4)) > 0.5).astype(float)
    assert_parity(
        F.binary_cross_entropy_with_logits(x, y), TF.binary_cross_entropy_with_logits(xt, torch.tensor(y)), [x], [xt]
    )


def test_causal_attention_parity():
    ours, theirs = both((2, 3, 5, 4), (2, 3, 5, 4), (2, 3, 5, 4))
    assert_parity(
        F.scaled_dot_product_attention(*ours, causal=True),
        TF.scaled_dot_product_attention(*theirs, is_causal=True),
        ours,
        theirs,
    )


def copy_weights(ours: nn.Module, theirs: "torch.nn.Module"):
    for (name, p), (tname, tp) in zip(ours.named_parameters(), theirs.named_parameters()):
        assert p.shape == tuple(tp.shape), (name, tname)
        with torch.no_grad():
            tp.copy_(torch.tensor(p.data))


def test_cnn_training_steps_match_torch():
    """Same init + same batches + Adam => identical weights after several steps."""
    ts.manual_seed(0)
    ours = nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(4 * 4 * 4, 10),
    )
    theirs = torch.nn.Sequential(
        torch.nn.Conv2d(1, 4, 3, padding=1),
        torch.nn.BatchNorm2d(4),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Flatten(),
        torch.nn.Linear(4 * 4 * 4, 10),
    )
    ours.double()
    theirs.double()
    copy_weights(ours, theirs)
    opt = optim.Adam(ours.parameters(), lr=1e-2, weight_decay=1e-3)
    topt = torch.optim.Adam(theirs.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(5):
        x = rng.normal(size=(8, 1, 8, 8))
        y = rng.integers(0, 10, 8)
        opt.zero_grad()
        loss = F.cross_entropy(ours(ts.tensor(x)), y)
        loss.backward()
        opt.step()
        topt.zero_grad()
        tloss = TF.cross_entropy(theirs(torch.tensor(x)), torch.tensor(y))
        tloss.backward()
        topt.step()
        assert abs(loss.item() - tloss.item()) < 1e-9
    for p, tp in zip(ours.parameters(), theirs.parameters()):
        np.testing.assert_allclose(p.data, tp.detach().numpy(), atol=1e-9)
    # running statistics tracked identically, and eval mode agrees
    np.testing.assert_allclose(ours[1].running_mean, theirs[1].running_mean.numpy(), atol=1e-9)
    np.testing.assert_allclose(ours[1].running_var, theirs[1].running_var.numpy(), atol=1e-9)
    x = rng.normal(size=(3, 1, 8, 8))
    ours.eval()
    theirs.eval()
    np.testing.assert_allclose(ours(ts.tensor(x)).data, theirs(torch.tensor(x)).detach().numpy(), atol=1e-9)


@pytest.mark.parametrize(
    "make",
    [
        lambda ps: (optim.SGD(ps, lr=0.1), lambda tp: torch.optim.SGD(tp, lr=0.1)),
        lambda ps: (
            optim.SGD(ps, lr=0.1, momentum=0.9, nesterov=True, weight_decay=0.01),
            lambda tp: torch.optim.SGD(tp, lr=0.1, momentum=0.9, nesterov=True, weight_decay=0.01),
        ),
        lambda ps: (optim.Adam(ps, lr=0.05), lambda tp: torch.optim.Adam(tp, lr=0.05)),
        lambda ps: (optim.AdamW(ps, lr=0.05), lambda tp: torch.optim.AdamW(tp, lr=0.05)),
        lambda ps: (
            optim.RMSprop(ps, lr=0.01, momentum=0.5),
            lambda tp: torch.optim.RMSprop(tp, lr=0.01, momentum=0.5),
        ),
    ],
    ids=["sgd", "sgd-nesterov-wd", "adam", "adamw", "rmsprop"],
)
def test_optimizer_updates_match_torch(make):
    w0 = rng.normal(size=(4, 3))
    w = ts.tensor(w0.copy(), requires_grad=True)
    tw = torch.tensor(w0.copy(), requires_grad=True)
    opt, tmake = make([w])
    topt = tmake([tw])
    for _ in range(6):
        target = rng.normal(size=(4, 3))
        opt.zero_grad()
        ((w - ts.tensor(target)) ** 2).sum().backward()
        opt.step()
        topt.zero_grad()
        ((tw - torch.tensor(target)) ** 2).sum().backward()
        topt.step()
    np.testing.assert_allclose(w.data, tw.detach().numpy(), atol=1e-10)


def test_transformer_block_matches_handwritten_torch():
    ts.manual_seed(3)
    block = nn.TransformerBlock(8, 2, causal=True).double()
    params = {k: torch.tensor(v.data, requires_grad=True) for k, v in block.named_parameters()}
    x0 = rng.normal(size=(2, 5, 8))
    x = ts.tensor(x0, requires_grad=True)
    xt = torch.tensor(x0, requires_grad=True)

    def lin(h, name):
        return h @ params[f"{name}.weight"].T + params[f"{name}.bias"]

    def ref(h):
        b, t, c = h.shape
        a = TF.layer_norm(h, (c,), params["ln1.weight"], params["ln1.bias"])
        q, k, v = lin(a, "attn.qkv").split(c, dim=-1)
        q, k, v = (z.view(b, t, 2, c // 2).transpose(1, 2) for z in (q, k, v))
        y = TF.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, t, c)
        h = h + lin(y, "attn.proj")
        m = TF.layer_norm(h, (c,), params["ln2.weight"], params["ln2.bias"])
        return h + lin(TF.gelu(lin(m, "mlp.fc"), approximate="tanh"), "mlp.proj")

    ours, theirs = block(x), ref(xt)
    assert_parity(ours, theirs, [x], [xt])
    for name, p in block.named_parameters():
        np.testing.assert_allclose(p.grad.data, params[name].grad.numpy(), atol=1e-8, err_msg=name)
