from typing import Callable
from multimethod import multimethod
from .predict import predict
from .mse import mse


@multimethod
def loss(network, x, y):
    return mse(y, predict(network, x))


@multimethod
def loss(f: Callable):
    def function(network, x):
        yhat = predict(network, x)
        return mse(f(yhat), yhat)
    return function
