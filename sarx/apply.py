from jax import jit
from .forward import forward


@jit
def apply(network, x):
    return forward(network, x)[1][-1]
