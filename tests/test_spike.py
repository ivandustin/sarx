from jax.numpy import array, array_equal, inf, nan, ones_like, sum
from jax import grad
from pytest import fixture
from sarx.core import spike as spike_function
from sarx import spike


@fixture
def input():
    return array([nan, -inf, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, inf])


def test(input):
    assert array_equal(spike(input), spike_function(input))


def test_gradient(input):
    assert array_equal(grad(lambda x: sum(spike(x)))(input), ones_like(input))
