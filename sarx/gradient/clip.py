from jax.numpy import clip as clip_function


def clip(gradient):
    return clip_function(gradient, -8, 8)
