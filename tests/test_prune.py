from jax.numpy import array, inf, nan, array_equal
from sarx import prune


def test_prune():
    input = array([nan, -inf, -1.0, 0.0, 1.0, inf, nan])
    expected = array([0.0, -inf, -1.0, 0.0, 1.0, inf, 0.0])
    actual = prune(input)
    assert array_equal(actual, expected)
