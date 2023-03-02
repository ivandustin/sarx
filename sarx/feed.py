from jax.tree_util import tree_map
from jax import grad


def feed(loss, update):
    def function(network, x, y):
        gradient = grad(loss)(network, x, y)
        return tree_map(update, network, gradient)
    return function
