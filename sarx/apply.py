from .forward import forward


def apply(network, x):
    return forward(network, x)[1][-1]
