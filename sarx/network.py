from .core import network as network_function
from .classes import Network


def network(*args, **kwargs):
    return Network(network_function(*args, **kwargs))
