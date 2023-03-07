from .forward import forward


def predict(network, x):
    return forward(network, x)[1][-1]
