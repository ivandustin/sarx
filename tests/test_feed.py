from jax.numpy import array, isclose
from jax.random import PRNGKey
from jax.lax import fori_loop
from sarx import network, feed, infer, spike, mae, gd


def test():
    key = PRNGKey(0)
    model = network(key, 1)
    x = array([[1.0]])
    y = array([[1.7]])
    before = predict(model, x)
    model = train(model, x, y)
    after = predict(model, x)
    assert not isclose(before, y, atol=0.05)
    assert isclose(after, y, atol=0.05)


def train(network, x, y):
    def body(_, network):
        return feed(loss, update)(network, x, y)
    return fori_loop(0, 2, body, network)


def predict(network, x):
    return infer(spike)(network, x)[-1]


def loss(network, x, y):
    return mae(y, predict(network, x))


def update(synapse, gradient):
    return gd(synapse, 0.1, gradient)
