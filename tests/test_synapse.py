from jax.random import PRNGKey
from jax.numpy import isclose
from sarx.synapse import synapse


def test():
    key = PRNGKey(0)
    weights = synapse(key, shape=(1000,))
    assert isclose(weights.mean(), 1.5, atol=0.01)
    assert isclose(weights.std(), 0.1, atol=0.01)
