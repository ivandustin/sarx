from .first import first
from .tail import tail


def infer(activate):
    def function(S, x):
        n = 0
        X = []
        for s in S:
            n = (x @ s) + n
            x = activate(first(n))
            n = tail(n)
            X = X + [x]
        return X
    return function
