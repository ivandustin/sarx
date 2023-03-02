from jax.numpy import arange, array, array_equal
from sarx.right import right


def test():
    matrix = arange(1, 10).reshape(3, 3)
    expected = array([
        [3],
        [6],
        [9]
    ])
    assert array_equal(right(matrix), expected)
