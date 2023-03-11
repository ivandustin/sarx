from .core import forward as f
from .core import spike


def forward(S, x):
    return f(spike)(S, x)
