from .multihead_models import MultiHeadNequIPGNNModel
from .extract_head import extract_head, extract_summed_heads

__all__ = ["MultiHeadNequIPGNNModel", "extract_head", "extract_summed_heads"]

try:
    from .multihead_allegro_models import MultiHeadAllegroModel

    __all__.append("MultiHeadAllegroModel")
except ImportError:
    pass  # allegro not installed
