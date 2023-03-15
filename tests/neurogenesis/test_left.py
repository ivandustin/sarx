from jax.numpy import array, array_equal
from sarx.neurogenesis import left


def test():
    matrix = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    expected = array([
        [1, 2],
        [4, 5],
        [7, 8]
    ])
    assert array_equal(left(matrix), expected)
