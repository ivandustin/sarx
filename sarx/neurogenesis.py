from .core import neurogenesis as neurogenesis_function
from .classes import Network


def neurogenesis(*args, **kwargs):
    return Network(neurogenesis_function(*args, **kwargs))
