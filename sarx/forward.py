from typing import Callable
from multimethod import multimethod
from .first import first
from .spike import spike
from .tail import tail


@multimethod
def forward(f: Callable):
    def function(S, x):
        n = 0
        A = []
        B = []
        for s in S:
            n = (x @ s) + n
            a = first(n)
            x = f(a)
            n = tail(n)
            A = A + [a]
            B = B + [x]
        return A, B
    return function


@multimethod
def forward(S, x):
    return forward(spike)(S, x)
