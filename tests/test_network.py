from jax.random import PRNGKey
from sarx import network


def test():
    key = PRNGKey(0)
    net = network(key, 1000)
    assert len(net) == 1
    assert net[0].shape == (1000, 1)
