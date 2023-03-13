from jax.numpy import array, inf, array_equal
from sarx.core import gc


def test():
    x = array([-inf, -9, -8, 1, 0, 1, 8, 9, inf])
    y = array([-8, -8, -8, 1, 0, 1, 8, 8, 8])
    assert array_equal(gc(x), y)
