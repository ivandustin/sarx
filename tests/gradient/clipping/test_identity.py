from jax.numpy import square, inf
from jax import grad
from sarx.gradient.clipping import identity


def test():
    function = identity(None, None)
    assert function(1) == 1


def test_gradient():
    function = grad(equation(identity(4.0)))
    assert function(-inf) == -4.0
    assert function(1.0) == -2.0
    assert function(2.0) == 0.0
    assert function(3.0) == 2.0
    assert function(inf) == 4.0


def equation(identity):
    def function(x):
        return square(2.0 - identity(1.0 * x))
    return function
