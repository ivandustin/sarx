from jax.numpy import arange, array, array_equal
from sarx.core.neurogenesis import left


def test():
    matrix = arange(1, 10).reshape(3, 3)
    expected = array([
        [1, 2],
        [4, 5],
        [7, 8]
    ])
    assert array_equal(left(matrix), expected)
