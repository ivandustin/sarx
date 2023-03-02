from jax.numpy import where, isnan


def prune(x):
    return where(isnan(x), 0, x)
