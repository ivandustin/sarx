from jax.numpy import array, array_equal, clip
from jax.random import PRNGKey, split
from jax.tree_util import tree_map
from jax.lax import fori_loop
from jax import grad
from optax import apply_updates, sgd
from sarx import apply, loss, network, neurogenesis
import pytest


@pytest.mark.parametrize("n", [1, 20])
@pytest.mark.parametrize("x,y", [
    (
        array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ]),
        array([
            [0.0],
            [1.0],
            [1.0],
            [0.0]
        ])
    ),
    (
        array([
            [0.0, 0.0],
            [0.0, 2.0],
            [2.0, 0.0],
            [2.0, 2.0]
        ]),
        array([
            [0.0],
            [2.0],
            [2.0],
            [0.0]
        ])
    )
])
def test(n, x, y):
    key = PRNGKey(0)
    model = network(key, 2)
    for key in split(key, n):
        model = neurogenesis(key, model)
    model = train(model, x, y)
    assert array_equal(apply(model, x).clip(0, 1), y.clip(0, 1))


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

    return fori_loop(0, 30, body, (network, state))[0]
