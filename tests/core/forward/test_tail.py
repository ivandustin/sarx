from jax.numpy import arange, array, array_equal
from sarx.core.forward import tail


def test():
    matrix = arange(1, 10).reshape(3, 3)
    expected = array([
        [2, 3],
        [5, 6],
        [8, 9]
    ])
    assert array_equal(tail(matrix), expected)
