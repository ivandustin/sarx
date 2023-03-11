from jax.tree_util import register_pytree_node
from sarx.tree_unflatten import tree_unflatten
from sarx.tree_flatten import tree_flatten
from sarx.apply import apply


class Network(list):
    __call__ = apply


register_pytree_node(Network, tree_flatten, tree_unflatten)
