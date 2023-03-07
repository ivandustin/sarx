from jax.tree_util import tree_map
from .prune import prune
from .gd import gd


def update(network, gradient, learning_rate):
    return tree_map(
        lambda network, gradient: prune(gd(network, gradient, learning_rate)),
        network,
        gradient,
    )
