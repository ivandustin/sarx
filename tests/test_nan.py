from jax.numpy import array, isclose, any, isnan
from jax.random import PRNGKey, split
from jax.lax import fori_loop
from jax import grad
from sarx import network, update, loss, neurogenesis, apply


def test():
    key = PRNGKey(0)
    model = network(key, 1)
    for key in split(key, 20):
        model = neurogenesis(key, model)
    x = array([
        [1.0]
    ])
    y = array([
        [1.7]
    ])
    model = train(model, x, y)
    assert isclose(apply(model, x), y)
    for synapse in model:
        assert not any(isnan(synapse))


def train(network, x, y):
    def body(_, network):
        gradient = grad(loss)(network, x, y)
        return update(network, gradient, 0.1)
    return fori_loop(0, 50, body, network)
