from jax.numpy import array, array_equal
from sarx.insert import insert


def test():
    a = array([
        [0],
        [0],
        [0]
    ])
    b = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    expected = array([
        [1, 2, 0, 3],
        [4, 5, 0, 6],
        [7, 8, 0, 9]
    ])
    assert array_equal(insert(a, b), expected)
