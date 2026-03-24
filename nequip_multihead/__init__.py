from . import _keys
from ._version import __version__

# Register HEAD_KEY as a graph/long field in NequIP's key registry
# so that batched_from_list and frame_from_batched handle it correctly.
from nequip.data._key_registry import register_fields

register_fields(
    graph_fields=[_keys.HEAD_KEY],
    long_fields=[_keys.HEAD_KEY],
)

# Register libraries that need to be marked as external during packaging.
# sympy is a transitive dependency that torch.package doesn't handle automatically.
try:
    from nequip.scripts._package_utils import register_libraries_as_external_for_packaging

    register_libraries_as_external_for_packaging(extern_modules=["sympy"])
except ImportError:
    pass

__all__ = ["__version__"]
