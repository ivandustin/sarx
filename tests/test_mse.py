from jax.numpy import array, mean, square, array_equal
from sarx import mse


def test():
    y = array([1, 2, 3])
    yhat = array([0, 1, 2])
    expected = mean(square(y - yhat))
    actual = mse(y, yhat)
    assert array_equal(actual, expected)
