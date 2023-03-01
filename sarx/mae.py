from jax.numpy import abs, mean, where
from jax import custom_jvp


@custom_jvp
def mae(x, y):
    return mean(abs(y - x))


@mae.defjvp
def mae_jvp(primals, tangents):
    x, y = primals
    dx, dy = tangents
    return mae(x, y), where(y > x, 1.0, where(y < x, -1.0, 0.0)) * dy
