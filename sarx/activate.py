from .vjps import identity as vjp_identity
from .vjps import clip as vjp_clip
from .identity import identity
from .compose import compose
from .spike import spike


activate = compose(vjp_clip(identity, 4.0), vjp_identity(spike))