from jax.random import PRNGKey
from sarx.core import network


def test():
    key = PRNGKey(0)
    model = network(key, 1000)
    assert len(model) == 1
    assert model[0].shape == (1000, 1)
