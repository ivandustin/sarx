from .first import first
from .tail import tail


def infer(activate):
    def function(S, x):
        n = 0
        A = []
        B = []
        for s in S:
            n = (x @ s) + n
            a = first(n)
            x = activate(a)
            n = tail(n)
            A = A + [a]
            B = B + [x]
        return A, B
    return function
