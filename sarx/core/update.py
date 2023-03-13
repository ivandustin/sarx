from .gc import gc
from .gd import gd


def update(network, gradient, learning_rate):
    return gd(network, gc(gradient), learning_rate)
