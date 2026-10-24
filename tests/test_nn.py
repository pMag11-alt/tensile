import numpy as np
import pytest

import tensile as ts
import tensile.functional as F
import tensile.nn as nn
import tensile.optim as optim


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(0.5), nn.Linear(8, 2))
        self.norm = nn.BatchNorm1d(2)
        self.scale = nn.Parameter(np.ones(1, dtype=np.float32))
        self.not_a_param = np.zeros(3)

    def forward(self, x):
        return self.norm(self.body(x)) * self.scale


def test_parameter_registration_and_names():
    m = Tiny()
    names = [n for n, _ in m.named_parameters()]
    assert names == [
        "scale",
        "body.0.weight",
        "body.0.bias",
        "body.3.weight",
        "body.3.bias",
        "norm.weight",
        "norm.bias",
    ]
    assert m.num_parameters() == 1 + 4 * 8 + 8 + 8 * 2 + 2 + 2 + 2
    assert [n for n, _ in m.named_buffers()] == ["norm.running_mean", "norm.running_var"]


def test_super_init_is_required():
    class Broken(nn.Module):
        def __init__(self):
            self.w = nn.Parameter(np.zeros(1))

    with pytest.raises(AttributeError, match="super"):
        Broken()


def test_reassigning_attribute_replaces_registration():
    m = Tiny()
    m.scale = 3.0
    assert "scale" not in dict(m.named_parameters())
    del m.body
    assert not any(n.startswith("body") for n, _ in m.named_parameters())


def test_tied_weights_are_yielded_once():
    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(10, 4)
            self.head = nn.Linear(4, 10, bias=False)
            self.head.weight = self.emb.weight

    assert Tied().num_parameters() == 40


def test_train_eval_propagates():
    m = Tiny()
    m.eval()
    assert all(not mod.training for mod in m.modules())
    m.train()
    assert all(mod.training for mod in m.modules())


def test_state_dict_roundtrip_and_strictness(tmp_path):
    ts.manual_seed(0)
    a = Tiny()
    a(ts.randn(5, 4))  # update BN running stats
    ts.save(a.state_dict(), tmp_path / "w.npz")
    b = Tiny()
    b.load_state_dict(ts.load(tmp_path / "w.npz"))
    a.eval()
    b.eval()
    x = ts.randn(3, 4)
    np.testing.assert_allclose(a(x).data, b(x).data)
    with pytest.raises(KeyError):
        b.load_state_dict({"scale": np.ones(1)})
    with pytest.raises(ValueError):
        b.load_state_dict({**a.state_dict(), "scale": np.ones(2)})


def test_to_double_casts_params_and_buffers():
    m = Tiny().double()
    assert all(p.dtype == np.float64 for p in m.parameters())
    assert m.norm.running_mean.dtype == np.float64


def test_repr_tree():
    text = repr(Tiny())
    assert "(body): Sequential(" in text and "(0): Linear(in_features=4, out_features=8, bias=True)" in text


def test_dropout_train_vs_eval():
    ts.manual_seed(1)
    d = nn.Dropout(0.25)
    x = ts.ones(10_000)
    y = d(x)
    kept = (y.data != 0).mean()
    assert 0.72 < kept < 0.78
    assert y.data.max() == pytest.approx(1 / 0.75)
    assert abs(y.data.mean() - 1) < 0.05
    d.eval()
    assert d(x) is x


def test_batchnorm_normalizes_in_train_and_uses_running_stats_in_eval():
    bn = nn.BatchNorm2d(3)
    x = ts.tensor(np.random.default_rng(0).normal(5, 3, size=(16, 3, 4, 4)).astype(np.float32))
    y = bn(x)
    np.testing.assert_allclose(y.data.mean(axis=(0, 2, 3)), 0, atol=1e-5)
    np.testing.assert_allclose(y.data.std(axis=(0, 2, 3)), 1, atol=1e-3)
    assert np.all(bn.running_mean > 0.4)  # moved 10% toward ~5
    bn.eval()
    y_eval = bn(x)
    assert not np.allclose(y_eval.data, y.data)
    with pytest.raises(ValueError):
        bn(ts.randn(2, 3))


def test_layer_shapes():
    x = ts.randn(2, 3, 16, 16)
    assert nn.Conv2d(3, 8, 3)(x).shape == (2, 8, 14, 14)
    assert nn.Conv2d(3, 8, 3, stride=2, padding=1)(x).shape == (2, 8, 8, 8)
    assert nn.Conv2d(3, 8, (3, 5), padding=(1, 2))(x).shape == (2, 8, 16, 16)
    assert nn.MaxPool2d(2)(x).shape == (2, 3, 8, 8)
    assert nn.AvgPool2d(4)(x).shape == (2, 3, 4, 4)
    assert nn.Flatten()(x).shape == (2, 768)
    assert nn.Embedding(10, 5)(np.array([[1, 2, 3]])).shape == (1, 3, 5)
    assert nn.LayerNorm(16)(x).shape == x.shape
    assert nn.MultiheadAttention(16, 4)(ts.randn(2, 7, 16)).shape == (2, 7, 16)


def test_conv_rejects_channel_mismatch():
    with pytest.raises(ValueError, match="channels"):
        nn.Conv2d(4, 2, 3)(ts.randn(1, 3, 5, 5))


def test_causal_attention_ignores_the_future():
    ts.manual_seed(0)
    attn = nn.MultiheadAttention(8, 2, causal=True)
    x = ts.randn(1, 6, 8)
    y = attn(x).data
    x2 = x.data.copy()
    x2[0, 4:] += 10.0  # perturb only the last two positions
    y2 = attn(ts.tensor(x2)).data
    np.testing.assert_allclose(y[0, :4], y2[0, :4], atol=1e-5)
    assert not np.allclose(y[0, 4:], y2[0, 4:])


def test_causal_attention_gradient_does_not_flow_backwards_in_time():
    ts.manual_seed(0)
    attn = nn.MultiheadAttention(8, 2, causal=True)
    x = ts.randn(1, 5, 8, requires_grad=True)
    attn(x)[0, 1].sum().backward()  # output at t=1 depends only on t<=1
    assert np.abs(x.grad.data[0, 2:]).max() == 0
    assert np.abs(x.grad.data[0, :2]).max() > 0


def test_clip_grad_norm():
    w = ts.tensor(np.zeros(4), requires_grad=True)
    (w * ts.tensor([3.0, 4.0, 0.0, 0.0])).sum().backward()
    norm = nn.utils.clip_grad_norm_([w], max_norm=1.0)
    assert norm == pytest.approx(5.0)
    assert np.linalg.norm(w.grad.data) == pytest.approx(1.0, rel=1e-4)


def test_module_zero_grad_and_optimizer_zero_grad():
    m = nn.Linear(3, 2)
    F.mse_loss(m(ts.randn(4, 3)), np.zeros((4, 2))).backward()
    assert m.weight.grad is not None
    m.zero_grad()
    assert m.weight.grad is None
    opt = optim.SGD(m.parameters())
    F.mse_loss(m(ts.randn(4, 3)), np.zeros((4, 2))).backward()
    opt.zero_grad()
    assert all(p.grad is None for p in m.parameters())


def test_kaiming_default_init_has_expected_scale():
    ts.manual_seed(0)
    w = nn.Linear(1000, 500).weight.data
    bound = 1 / np.sqrt(1000)  # PyTorch's default Linear bound
    assert w.max() <= bound and w.min() >= -bound
    assert abs(w.std() - bound / np.sqrt(3)) < 1e-3
