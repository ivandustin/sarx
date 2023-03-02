from jax.random import PRNGKey
from jax.numpy import isclose
from sarx import synapse

def test():
    key = PRNGKey(0)
    syn = synapse(key, shape=(1000,))
    assert isclose(syn.mean(), 1.5, atol=0.01)
    assert isclose(syn.std(), 0.1, atol=0.01)
