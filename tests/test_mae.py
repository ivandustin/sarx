from jax.numpy import array
from jax import grad
from sarx import mae


def test_mae():
    x = array([0.0, 0.0])
    y = array([1.0, 1.0])
    assert mae(x, y) == 1.0


def test_grad():
    function = grad(lambda x: mae(0.0, 0.0 + x))
    assert function(0.0) == 0.0
    assert function(10.0) == 1.0
    assert function(-10.0) == -1.0
