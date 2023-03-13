from jax.random import PRNGKey
from sarx.core.network import network


def test_default_layers():
    key = PRNGKey(0)
    model = network(key, 1000)
    assert len(model) == 1
    assert model[0].shape == (1000, 1)


def test_two_layers():
    key = PRNGKey(0)
    model = network(key, 1000, 2)
    assert len(model) == 2
    assert model[0].shape == (1000, 2)
    assert model[1].shape == (1, 1)


def test_three_layers():
    key = PRNGKey(0)
    model = network(key, 1000, 3)
    assert len(model) == 3
    assert model[0].shape == (1000, 3)
    assert model[1].shape == (1, 2)
    assert model[2].shape == (1, 1)
