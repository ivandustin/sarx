from jax.tree_util import register_pytree_node_class
from ..pytrees import List
from ..apply import apply


@register_pytree_node_class
class Network(List):
    __call__ = apply
