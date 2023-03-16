from jax.numpy import array, array_equal
from jax.random import PRNGKey, split
from jax.lax import fori_loop
from jax import grad
from optax import apply_updates, sgd
from sarx import apply, loss, network, neurogenesis


def test():
    key = PRNGKey(0)
    key_a, key_b = split(key)
    model = network(key_a, 2)
    model = neurogenesis(key_b, model)
    x = array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ])
    y = array([
        [0.0],
        [1.0],
        [1.0],
        [0.0]
    ])
    model = train(model, x, y)
    assert array_equal(apply(model, x).clip(0, 1), y)


def train(network, x, y):
    optimizer = sgd(0.1)
    state = optimizer.init(network)

    def body(_, args):
        network, state = args
        gradient = grad(loss)(network, x, y)
        updates, state = optimizer.update(gradient, state)
        return apply_updates(network, updates), state

    return fori_loop(0, 30, body, (network, state))[0]
