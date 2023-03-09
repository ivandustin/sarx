from multimethod import multimethod
from jax.tree_util import tree_map
from jax.typing import ArrayLike
from jax.numpy import clip
from .gd import gd


@multimethod
def update(network: ArrayLike, gradient: ArrayLike, learning_rate):
    return gd(network, clip(gradient, -8, 8), learning_rate)


@multimethod
def update(network, gradient, learning_rate):
    return tree_map(
        lambda network, gradient: update(network, gradient, learning_rate),
        network,
        gradient,
    )
