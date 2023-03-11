from .core import network as f
from .classes import Network


def network(key, n):
    return Network(f(key, n))
