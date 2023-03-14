from jax.numpy import array, array_equal
from sarx.core.forward import tail


def test():
    matrix = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    expected = array([
        [2, 3],
        [5, 6],
        [8, 9]
    ])
    assert array_equal(tail(matrix), expected)
