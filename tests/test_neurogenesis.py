from jax.random import PRNGKey, split
from sarx import network, neurogenesis


def test():
    key = PRNGKey(0)
    keys = split(key, 6)
    net = network(key, 100)
    for key in keys:
        net = neurogenesis(key, net)
    assert len(net) == 7
    assert net[0].shape == (100, 7)
    assert net[1].shape == (1, 6)
    assert net[2].shape == (1, 5)
    assert net[3].shape == (1, 4)
    assert net[4].shape == (1, 3)
    assert net[5].shape == (1, 2)
    assert net[6].shape == (1, 1)
