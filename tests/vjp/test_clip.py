from jax.numpy import square, inf
from jax import grad
from sarx import identity
from sarx.vjp import clip


def test():
    function = grad(equation(clip(identity, 4.0)))
    assert function(-inf) == -4.0
    assert function(1.0) == -2.0
    assert function(2.0) == 0.0
    assert function(3.0) == 2.0
    assert function(inf) == 4.0


def equation(identity):
    def function(x):
        return square(2.0 - identity(1.0 * x))
    return function
