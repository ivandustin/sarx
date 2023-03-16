from jax import custom_vjp


def vjp(f, g):
    wrapper = custom_vjp(f)

    def forward(*args, **kwargs):
        return f(*args, **kwargs), None

    def backward(_, gradient):
        return (g(gradient),)

    wrapper.defvjp(forward, backward)
    return wrapper
