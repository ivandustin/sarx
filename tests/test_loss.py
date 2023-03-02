from jax.numpy import add
from sarx import loss


def test():
    function = loss(add, add)
    assert function(1, 2, 3) == 6
    assert function(1, 2, lambda x: x) == 6
