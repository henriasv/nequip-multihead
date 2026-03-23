from . import _keys
from ._version import __version__

# Register HEAD_KEY as a graph/long field in NequIP's key registry
# so that batched_from_list and frame_from_batched handle it correctly.
from nequip.data._key_registry import register_fields

register_fields(
    graph_fields=[_keys.HEAD_KEY],
    long_fields=[_keys.HEAD_KEY],
)

__all__ = ["__version__"]
