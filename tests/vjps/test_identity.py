from sarx.vjps import identity as vjp_identity
from sarx import identity
from jax import grad


def test():
    assert grad(vjp_identity(identity))(0.0) == 1.0
