from typing import Callable
from multimethod import multimethod
from .apply import apply
from .mse import mse


@multimethod
def loss(network, x, y):
    return mse(y, apply(network, x))


@multimethod
def loss(f: Callable):
    def function(network, x):
        yhat = apply(network, x)
        return mse(f(yhat), yhat)
    return function
