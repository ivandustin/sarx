from .core.forward import forward as forward_function
from .spike import spike


def forward(*args, **kwargs):
    return forward_function(spike)(*args, **kwargs)
