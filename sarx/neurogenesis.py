from jax.random import split
from .synapse import synapse
from .insert import insert


def neurogenesis(key, net):
    keys = split(key, len(net))
    a = [insert(synapse(key, shape=(syn.shape[0], 1)), syn)
         for key, syn in zip(keys, net)]
    b = [synapse(key, shape=(1, 1))]
    return a + b
