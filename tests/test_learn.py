from jax.numpy import array, all, isclose
from jax.random import PRNGKey
from jax.lax import fori_loop
from jax import grad
from sarx import network, update, loss


def test():
    key = PRNGKey(0)
    model = network(key, 1)
    x = array([
        [0.1]
    ])
    y = array([
        [1.7]
    ])
    assert all(model(x) == 0)
    model = train(model, x, y)
    assert isclose(model(x), y)


def train(network, x, y):
    def body(_, network):
        gradient = grad(loss)(network, x, y)
        return update(network, gradient, 10.0)
    return fori_loop(0, 50, body, network)
