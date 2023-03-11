from .synapse import synapse


def network(key, n):
    return [synapse(key, shape=(n, 1))]
