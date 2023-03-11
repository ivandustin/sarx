from jax.random import split
from .classes import Network
from .synapse import synapse
from .insert import insert


def neurogenesis(key, network) -> Network:
    keys = split(key, len(network))
    old = [insert(synapse(key, shape=(weights.shape[0], 1)), weights)
           for key, weights in zip(keys, network)]
    new = [synapse(key, shape=(1, 1))]
    return Network(old + new)
