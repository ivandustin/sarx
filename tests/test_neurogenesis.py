from jax.random import PRNGKey, split
from sarx import network, neurogenesis, Network


def test():
    key = PRNGKey(0)
    keys = split(key, 6)
    model = network(key, 100)
    for key in keys:
        model = neurogenesis(key, model)
    assert len(model) == 7
    assert model[0].shape == (100, 7)
    assert model[1].shape == (1, 6)
    assert model[2].shape == (1, 5)
    assert model[3].shape == (1, 4)
    assert model[4].shape == (1, 3)
    assert model[5].shape == (1, 2)
    assert model[6].shape == (1, 1)
    assert type(model) == Network
