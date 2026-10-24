import numpy as np
import pytest

import tensile as ts


def test_python_floats_default_to_float32_but_arrays_keep_dtype():
    assert ts.tensor([1.0, 2.0]).dtype == np.float32
    assert ts.tensor(np.array([1.0])).dtype == np.float64
    assert ts.tensor([1, 2]).dtype.kind == "i"


def test_scalars_do_not_upcast_float32():
    x = ts.ones(3)
    assert (x * 2.5 + 1).dtype == np.float32
    assert (np.float32(2) * x).dtype == np.float32
    assert (2**x).dtype == np.float32


def test_only_float_tensors_can_require_grad():
    with pytest.raises(TypeError):
        ts.tensor([1, 2], requires_grad=True)


def test_backward_requires_scalar_or_explicit_grad():
    x = ts.ones(2, 2, requires_grad=True)
    y = x * 3
    with pytest.raises(RuntimeError, match="non-scalar"):
        y.backward()
    y.backward(np.ones((2, 2)))
    np.testing.assert_array_equal(x.grad.data, np.full((2, 2), 3))


def test_backward_on_constant_raises():
    with pytest.raises(RuntimeError):
        ts.ones(1).backward()


def test_gradients_accumulate_across_backward_calls():
    x = ts.tensor([1.0, 2.0], requires_grad=True)
    (x * x).sum().backward()
    (x * x).sum().backward()
    np.testing.assert_allclose(x.grad.data, [4.0, 8.0])


def test_diamond_graph_sums_both_paths():
    x = ts.tensor(3.0, requires_grad=True)
    a = x * 2
    b = x * x
    (a * b).backward()  # d/dx 2x^3 = 6x^2
    assert x.grad.item() == pytest.approx(54.0)


def test_same_tensor_used_twice_in_one_op():
    x = ts.tensor([1.5, -2.0], requires_grad=True)
    (x * x).sum().backward()
    np.testing.assert_allclose(x.grad.data, [3.0, -4.0])


def test_deep_graph_does_not_hit_recursion_limit():
    x = ts.tensor(1.0, requires_grad=True)
    y = x
    for _ in range(20_000):
        y = y * 1.0001
    y.backward()
    assert x.grad.item() == pytest.approx(1.0001**20_000, rel=1e-3)


def test_no_grad_context_and_decorator():
    x = ts.ones(2, requires_grad=True)
    with ts.no_grad():
        y = x * 2
        assert not ts.is_grad_enabled()
        with ts.enable_grad():
            assert (x * 2).requires_grad
    assert ts.is_grad_enabled()
    assert not y.requires_grad and y.grad_fn is None

    @ts.no_grad()
    def f(t):
        return t * 2

    assert not f(x).requires_grad


def test_detach_cuts_the_graph():
    x = ts.tensor(2.0, requires_grad=True)
    y = (x * x).detach() * x
    y.backward()
    assert x.grad.item() == pytest.approx(4.0)


def test_retain_grad_on_intermediate():
    x = ts.tensor([1.0, 2.0], requires_grad=True)
    h = (x * 3).retain_grad()
    (h * h).sum().backward()
    np.testing.assert_allclose(h.grad.data, [6.0, 12.0])
    np.testing.assert_allclose(x.grad.data, [18.0, 36.0])


def test_non_leaf_grads_not_kept_by_default():
    x = ts.tensor([1.0], requires_grad=True)
    h = x * 2
    h.sum().backward()
    assert h.grad is None and x.grad is not None


def test_constants_in_graph_get_no_grad():
    x = ts.tensor([1.0, 2.0], requires_grad=True)
    c = ts.tensor([5.0, 6.0])
    (x * c).sum().backward()
    assert c.grad is None


def test_comparisons_return_bool_tensors():
    x = ts.tensor([1.0, 2.0, 3.0])
    assert (x > 1.5).tolist() == [False, True, True]
    assert (x == 2).tolist() == [False, True, False]
    assert len({x, x}) == 1  # still hashable by identity


def test_repr_shows_grad_fn():
    x = ts.tensor([1.0], requires_grad=True)
    assert "requires_grad=True" in repr(x)
    assert "grad_fn=<MulBackward>" in repr(x * 2)


def test_shape_helpers():
    x = ts.arange(24).reshape(2, 3, 4)
    assert x.shape == (2, 3, 4)
    assert x.T.shape == (4, 3, 2)
    assert x.swapaxes(0, 2).shape == (4, 3, 2)
    assert x.unsqueeze(0).shape == (1, 2, 3, 4)
    assert x.unsqueeze(0).squeeze(0).shape == (2, 3, 4)
    assert x.flatten().shape == (24,)
    assert x.flatten(1).shape == (2, 12)
    assert x.reshape((6, 4)).shape == (6, 4)
    assert x.permute(1, 0, 2).shape == (3, 2, 4)


def test_argmax_and_item():
    x = ts.tensor([[1.0, 5.0], [7.0, 2.0]])
    assert x.argmax(axis=1).tolist() == [1, 0]
    assert x.max().item() == 7.0
    assert x.min().item() == 1.0


def test_max_splits_gradient_between_ties():
    x = ts.tensor([3.0, 1.0, 3.0], requires_grad=True)
    x.max().backward()
    np.testing.assert_allclose(x.grad.data, [0.5, 0.0, 0.5])


def test_sigmoid_is_stable_for_large_inputs():
    x = ts.tensor([-1000.0, 0.0, 1000.0], requires_grad=True)
    y = x.sigmoid()
    np.testing.assert_allclose(y.data, [0.0, 0.5, 1.0])
    y.sum().backward()
    assert np.all(np.isfinite(x.grad.data))


def test_log_softmax_is_stable():
    x = ts.tensor([[1000.0, 0.0, -1000.0]])
    out = x.log_softmax()
    assert np.all(np.isfinite(out.data))
    assert out.data[0, 0] == pytest.approx(0.0)


def test_cast_backward_restores_dtype():
    x = ts.tensor(np.ones(3, dtype=np.float32), requires_grad=True)
    x.double().sum().backward()
    assert x.grad.dtype == np.float32


def test_factories_and_seed():
    ts.manual_seed(5)
    a = ts.randn(3, 3)
    ts.manual_seed(5)
    b = ts.randn((3, 3))
    np.testing.assert_array_equal(a.data, b.data)
    assert ts.zeros(2, 3).shape == (2, 3)
    assert ts.ones_like(a).data.sum() == 9
    assert ts.full((2,), 7).tolist() == [7, 7]
    assert ts.eye(3).data.trace() == 3
    assert ts.randint(0, 5, (100,)).data.max() < 5


def test_where_maximum_minimum():
    a = ts.tensor([1.0, 5.0, 3.0])
    b = ts.tensor([4.0, 2.0, 3.0])
    assert ts.where(a > b, a, b).tolist() == [4.0, 5.0, 3.0]
    assert ts.maximum(a, b).tolist() == [4.0, 5.0, 3.0]
    assert ts.minimum(a, 2.0).tolist() == [1.0, 2.0, 2.0]


def test_custom_function():
    class Cube(ts.Function):
        def forward(self, x):
            self.x = x
            return x**3

        def backward(self, g):
            return 3 * g * self.x**2

    x = ts.tensor([2.0], requires_grad=True)
    Cube.apply(x).sum().backward()
    assert x.grad.item() == pytest.approx(12.0)


def test_save_load_roundtrip(tmp_path):
    state = {"w": np.arange(6.0).reshape(2, 3), "b": np.ones(2)}
    ts.save(state, tmp_path / "m.npz")
    back = ts.load(tmp_path / "m.npz")
    np.testing.assert_array_equal(back["w"], state["w"])


def test_mermaid_and_dot():
    w = ts.randn(3, 2, requires_grad=True)
    x = ts.randn(4, 3)
    loss = (x @ w).relu().mean()
    md = ts.to_mermaid(loss, params={"w": w})
    assert md.startswith("flowchart LR")
    assert '"w<br/>3×2"' in md and "MatMul" in md and "ReLU" in md and "Mean" in md
    dot = ts.to_dot(loss)
    assert dot.startswith("digraph") and "->" in dot
