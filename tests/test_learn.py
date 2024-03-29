from jax.numpy import array, isclose, clip
from jax.random import PRNGKey, split
from jax.tree_util import tree_map
from jax.lax import fori_loop
from jax import grad
from optax import apply_updates, sgd
from sarx import apply, loss, network, neurogenesis
import pytest


@pytest.mark.parametrize("n", [0, 20])
def test(n):
    key = PRNGKey(0)
    model = network(key, 1)
    for key in split(key, n):
        model = neurogenesis(key, model)
    x = array([
        [1.0]
    ])
    y = array([
        [1.7]
    ])
    model = train(model, x, y)
    assert isclose(apply(model, x), y)


def train(network, x, y):
    optimizer = sgd(0.1)
    state = optimizer.init(network)

    def body(_, args):
        network, state = args
        gradient = grad(loss)(network, x, y)
        updates, state = optimizer.update(gradient, state)
        network = apply_updates(network, updates)
        network = tree_map(lambda x: clip(x, -2.0, 2.0), network)
        return network, state

    return fori_loop(0, 50, body, (network, state))[0]
