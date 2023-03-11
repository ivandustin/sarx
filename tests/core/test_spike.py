from jax.numpy import array, array_equal
from jax import grad
from sarx.core import spike


def test_spike():
    input = array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    expected = array([0.0, 0.0, 0.0, 0.0, 1.0, 1.5, 2.0, 2.0, 2.0])
    actual = spike(input)
    assert array_equal(actual, expected)


def test_grad():
    assert grad(spike)(0.0) == 1.0
