from jax.numpy import float32, isclose
from jax.random import PRNGKey
from sarx.synapse import synapse


def test():
    key = PRNGKey(0)
    weights = synapse(key, shape=(1000,))
    assert weights.dtype == float32
    assert not weights.weak_type
    assert isclose(weights.mean(), 1.5, atol=0.01)
    assert isclose(weights.std(), 0.1, atol=0.01)
