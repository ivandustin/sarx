from jax.tree_util import tree_map
from jax import grad


def feed(loss, update):
    def function(network, input, expected):
        gradient = grad(loss)(network, input, expected)
        return tree_map(update, network, gradient)
    return function
