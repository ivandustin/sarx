from functools import partial
from jax.tree_util import tree_map
from .core import update as update_function


def update(network, gradient, learning_rate):
    return tree_map(
        partial(update_function, learning_rate=learning_rate),
        network,
        gradient,
    )
