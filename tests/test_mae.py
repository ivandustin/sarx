from jax.numpy import array
from jax import grad
from sarx import mae


def test_mae():
    x = array([0.0, 0.0])
    y = array([1.0, 1.0])
    assert mae(x, y) == 1.0


def test_grad():
    g = grad(lambda x: mae(0.0, 0.0 + x))
    assert g(0.0) == 0.0
    assert g(10.0) == 1.0
    assert g(-10.0) == -1.0
