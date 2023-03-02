from jax.numpy import arange, array, array_equal
from sarx.first import first


def test():
    matrix = arange(1, 10).reshape(3, 3)
    expected = array([
        [1],
        [4],
        [7]
    ])
    assert array_equal(first(matrix), expected)
