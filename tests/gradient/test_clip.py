from jax.numpy import array, inf, array_equal
from sarx.gradient import clip


def test():
    x = array([-inf, -8.5, -8.0, -1.5, 0.0, 1.5, 8.0, 8.5, inf])
    y = array([-8.0, -8.0, -8.0, -1.5, 0.0, 1.5, 8.0, 8.0, 8.0])
    assert array_equal(clip(x), y)
