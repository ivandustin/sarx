from jax.numpy import array, isclose
from jax.tree_util import tree_map
from jax.random import PRNGKey
from jax.lax import fori_loop
from sarx import network, feed, infer, spike, mae, gd


def test():
    key = PRNGKey(0)
    model = network(key, 1)
    x = array([[1.0]])
    y = array([[1.7]])
    before = predict(model, x)
    model = train(model, x, y, 0.1)
    model = train(model, x, y, 0.001)
    model = train(model, x, y, 0.00001)
    after = predict(model, x)
    assert not isclose(before, y)
    assert isclose(after, y)


def train(network, x, y, learning_rate):
    def body(_, network):
        return feed(loss, update(learning_rate))(network, x, y)
    return fori_loop(0, 60, body, network)


def predict(network, x):
    return infer(spike)(network, x)[1][-1]


def loss(network, x, y):
    return mae(y, predict(network, x))


def update(learning_rate):
    def function(network, gradient):
        return tree_map(gd(learning_rate), network, gradient)
    return function
