from jax.numpy import array, array_equal, inf, ones_like, sum, square
from jax import grad
from pytest import fixture
from sarx.core import spike as spike_function
from sarx import spike


@fixture
def input():
    return array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])


def test(input):
    assert array_equal(spike(input), spike_function(input))


def test_gradient(input):
    assert array_equal(grad(lambda x: sum(spike(x)))(input), ones_like(input))


def test_maximum_gradient():
    function = grad(equation(spike))
    assert function(inf, 2.0) == 0.0
    assert function(inf, 1.0) == 2.0
    assert function(1.0, 2.0) == -2.0
    assert function(0.0, inf) == -4.0
    assert function(0.0, -inf) == 4.0


def equation(f):
    def function(x, y):
        return square(y - f(1.0 * x))
    return function
