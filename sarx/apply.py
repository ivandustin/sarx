from .forward import forward
from jax import jit


@jit
def apply(network, x):
    return forward(network, x)[1][-1]
