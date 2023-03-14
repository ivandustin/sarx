from jax.numpy import square, inf
from jax import grad
from sarx.gradient.clipping import identity


def test():
    assert identity(1) == 1


def test_gradient():
    assert grad(function)(-inf) == -4.0
    assert grad(function)(1.0) == -2.0
    assert grad(function)(2.0) == 0.0
    assert grad(function)(3.0) == 2.0
    assert grad(function)(inf) == 4.0


def function(x):
    return square(2.0 - identity(1.0 * x))
