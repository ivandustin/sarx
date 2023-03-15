from jax.numpy import array, array_equal, inf, ones_like, sum, square
from jax import grad
from sarx import activate, spike
from pytest import fixture


@fixture
def input():
    return array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])


def test(input):
    assert array_equal(activate(input), spike(input))


def test_gradient(input):
    assert array_equal(grad(lambda x: sum(activate(x)))(input), ones_like(input))


def test_maximum_gradient():
    function = grad(equation(activate))
    assert function(inf, 2.0) == 0.0
    assert function(inf, 1.0) == 2.0
    assert function(1.0, 2.0) == -2.0
    assert function(0.0, inf) == -4.0
    assert function(0.0, -inf) == 4.0


def equation(activate):
    def function(x, y):
        return square(y - activate(1.0 * x))
    return function
