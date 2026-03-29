from .multihead_readout import MultiHeadReadout
from .per_head_convnetlayer import PerHeadConvNetLayer

__all__ = ["MultiHeadReadout", "PerHeadConvNetLayer"]

try:
    from .multihead_edge_readout import MultiHeadEdgeReadout

    __all__.append("MultiHeadEdgeReadout")
except ImportError:
    pass  # allegro not installed
