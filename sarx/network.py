from .classes import Network
from .synapse import synapse


def network(key, n) -> Network:
    return Network([synapse(key, shape=(n, 1))])
