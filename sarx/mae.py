from jax.numpy import abs, mean, sign
from jax import custom_jvp


@custom_jvp
def mae(x, y):
    return mean(abs(y - x))


@mae.defjvp
def mae_jvp(primals, tangents):
    x, y = primals
    _, dy = tangents
    return mae(x, y), sign(y - x) * dy
