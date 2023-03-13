from sarx.gradient import descent, clip


def update(network, gradient, learning_rate):
    return descent(network, clip(gradient), learning_rate)
