from jax.numpy import array, array_equal
from sarx import spike


def test():
    input = array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    expected = array([0.0, 0.0, 0.0, 0.0, 1.0, 1.5, 2.0, 2.0, 2.0])
    actual = spike(input)
    assert array_equal(actual, expected)
