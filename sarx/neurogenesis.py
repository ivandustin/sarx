from .core import neurogenesis as f
from .classes import Network


def neurogenesis(key, network):
    return Network(f(key, network))
