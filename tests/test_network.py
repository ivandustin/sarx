from jax.random import PRNGKey
from sarx import network, Network


def test():
    key = PRNGKey(0)
    model = network(key, 1000)
    assert len(model) == 1
    assert model[0].shape == (1000, 1)
    assert type(model) == Network
