from jax.numpy import clip


def gc(gradient):
    return clip(gradient, -8, 8)
