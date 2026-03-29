"""Utility to extract a single head (or sum of heads) from a trained multi-head model.

Provides model modifiers for use with ``nequip-compile --modifiers``.
Supports both node-based (NequIP) and edge-based (Allegro) multi-head readouts.
"""
import copy

import torch

from nequip.data import AtomicDataDict
from nequip.nn import (
    GraphModel,
    SequentialGraphNetwork,
    AtomwiseReduce,
)
from nequip.nn._graph_mixin import GraphModuleMixin
from nequip.nn.model_modifier_utils import model_modifier

from nequip_multihead.nn.multihead_readout import MultiHeadReadout
from nequip_multihead.nn.per_head_convnetlayer import PerHeadConvNetLayer

try:
    from nequip_multihead.nn.multihead_edge_readout import MultiHeadEdgeReadout

    _HAS_EDGE_READOUT = True
except ImportError:
    _HAS_EDGE_READOUT = False

# Tuple of classes to search for when finding multi-head readout modules
_MULTIHEAD_READOUT_CLASSES = (MultiHeadReadout,)
if _HAS_EDGE_READOUT:
    _MULTIHEAD_READOUT_CLASSES = (MultiHeadReadout, MultiHeadEdgeReadout)


def _is_instance_by_name(obj, cls):
    """Check isinstance, falling back to class name for torch.package compatibility."""
    if isinstance(obj, cls):
        return True
    return type(obj).__name__ == cls.__name__ and hasattr(obj, "__module__")


class SingleHeadConv(GraphModuleMixin, torch.nn.Module):
    """Wraps a single head's components from a PerHeadConvNetLayer for deployment.

    Runs the shared preprocessing (linear_1, normalization) and the specified
    head's TP + scatter + linear_2 + activation, writing the result to
    ``NODE_FEATURES_KEY``.
    """

    def __init__(self, per_head_conv: PerHeadConvNetLayer, head_name: str):
        super().__init__()
        self.head_name = head_name
        # Shared components
        self.linear_1 = per_head_conv.linear_1
        self.avg_num_neighbors_norm = per_head_conv.avg_num_neighbors_norm
        self.ghost_exchange = per_head_conv.ghost_exchange
        self.edge_mlp = per_head_conv.edge_mlp
        self.is_first_layer = per_head_conv.is_first_layer
        self._activation = per_head_conv._activation
        # Per-head components
        self.head_modules = per_head_conv.heads[head_name]
        self._head_weight_numel = per_head_conv._head_weight_numels[head_name]
        self._init_irreps(
            irreps_in=per_head_conv.irreps_in,
            irreps_out=per_head_conv.irreps_out,
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if AtomicDataDict.LMP_MLIAP_DATA_KEY in data:
            num_local_nodes = data[AtomicDataDict.LMP_MLIAP_DATA_KEY].nlocal
        else:
            num_local_nodes = AtomicDataDict.num_nodes(data)

        x = data[AtomicDataDict.NODE_FEATURES_KEY]
        if not self.is_first_layer:
            x = x[:num_local_nodes]

        # Self-connection
        sc = None
        if "sc" in self.head_modules:
            node_attrs = data[AtomicDataDict.NODE_ATTRS_KEY]
            if not self.is_first_layer:
                node_attrs = node_attrs[:num_local_nodes]
            sc = self.head_modules["sc"](x, node_attrs)

        x = self.linear_1(x)

        data_copy = data.copy()
        data_copy[AtomicDataDict.NODE_FEATURES_KEY] = x
        data_copy = self.avg_num_neighbors_norm(data_copy)
        x = data_copy[AtomicDataDict.NODE_FEATURES_KEY]

        if not self.is_first_layer:
            data_copy[AtomicDataDict.NODE_FEATURES_KEY] = x
            data_copy = self.ghost_exchange(data_copy, ghost_included=False)
            x = data_copy[AtomicDataDict.NODE_FEATURES_KEY]

        edge_weights = self.edge_mlp(data[AtomicDataDict.EDGE_EMBEDDING_KEY])
        head_weights = edge_weights[:, :self._head_weight_numel]

        x = self.head_modules["tp_scatter"](
            x=x,
            edge_attr=data[AtomicDataDict.EDGE_ATTRS_KEY],
            edge_weight=head_weights,
            edge_dst=data[AtomicDataDict.EDGE_INDEX_KEY][0],
            edge_src=data[AtomicDataDict.EDGE_INDEX_KEY][1],
        )[:num_local_nodes]

        x = self.head_modules["linear_2"](x)
        x = self._activation(x)

        if sc is not None:
            x = x + sc

        data[AtomicDataDict.NODE_FEATURES_KEY] = x
        return data


def _is_multihead_readout(obj):
    """Check if obj is any multi-head readout type (node or edge based)."""
    for cls in _MULTIHEAD_READOUT_CLASSES:
        if _is_instance_by_name(obj, cls):
            return True
    return False


def _is_edge_based_readout(obj):
    """Check if obj is an edge-based multi-head readout (Allegro)."""
    if not _HAS_EDGE_READOUT:
        return False
    return _is_instance_by_name(obj, MultiHeadEdgeReadout)


def _find_multihead_readout(model):
    """Walk the model tree to find any multi-head readout module.

    Returns:
        The multi-head readout module, or None if not found.
    """
    mhr = None

    def _walk(module):
        nonlocal mhr
        if _is_multihead_readout(module):
            mhr = module
            return
        for name, child in module.named_children():
            _walk(child)

    _walk(model)
    return mhr


def _find_seq_containing_readout(model):
    """Find the SequentialGraphNetwork containing the multi-head readout.

    Returns:
        (seq_net, key_name) tuple, or None if not found.
    """
    def _walk(module):
        if _is_instance_by_name(module, SequentialGraphNetwork):
            for name, child in module.named_children():
                if _is_multihead_readout(child):
                    return module, name
        for name, child in module.named_children():
            result = _walk(child)
            if result is not None:
                return result
        return None

    return _walk(model)


def _replace_seq_in_model(model, old_seq, new_seq):
    """Replace a SequentialGraphNetwork in the model hierarchy."""
    for name, child in model.named_children():
        if child is old_seq:
            setattr(model, name, new_seq)
            return True
        if _replace_seq_in_model(child, old_seq, new_seq):
            return True
    return False


def extract_head(model: GraphModel, head_name: str) -> GraphModel:
    """Extract a single head from a multi-head model into a standalone single-head model.

    The returned model does not require ``HEAD_KEY`` in input data and produces
    identical outputs to the multi-head model for the specified head.

    Supports both node-based (:class:`MultiHeadReadout`, NequIP) and
    edge-based (:class:`MultiHeadEdgeReadout`, Allegro) readout modules.

    Args:
        model: A :class:`~nequip.nn.GraphModel` containing a multi-head readout.
        head_name: Name of the head to extract.

    Returns:
        A new :class:`~nequip.nn.GraphModel` with the multi-head readout
        replaced by the extracted head's pipeline + ``AtomwiseReduce``.
    """
    model = copy.deepcopy(model)

    mhr = _find_multihead_readout(model)
    if mhr is None:
        raise ValueError(
            "No MultiHeadReadout or MultiHeadEdgeReadout found in model. "
            "Is this a multi-head model?"
        )

    if head_name not in mhr.head_names:
        raise ValueError(
            f"Head '{head_name}' not found. "
            f"Available heads: {mhr.head_names}"
        )

    result = _find_seq_containing_readout(model)
    if result is None:
        raise ValueError(
            "Could not find SequentialGraphNetwork containing multi-head readout"
        )
    seq_net, multihead_key = result

    # Extract head modules
    head_modules = mhr.heads[head_name]
    readout = head_modules["readout"]
    scale_shift = head_modules["scale_shift"]
    is_edge = _is_edge_based_readout(mhr)

    # Create AtomwiseReduce matching the original
    reduce = AtomwiseReduce(
        irreps_in=scale_shift.irreps_out,
        reduce="sum",
        field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
    )

    # Replace: remove multihead_readout (and per_head_conv if present),
    # insert individual modules
    new_modules = {}
    for name, child in seq_net.named_children():
        if _is_instance_by_name(child, PerHeadConvNetLayer):
            # Replace PerHeadConvNetLayer with single-head version
            new_modules["final_conv"] = SingleHeadConv(child, head_name)
        elif name == multihead_key:
            # Replace with individual head modules
            new_modules["edge_energy_readout" if is_edge else "per_atom_energy_readout"] = readout
            if is_edge:
                new_modules["edge_eng_sum"] = head_modules["edge_reduce"]
            new_modules["per_type_energy_scale_shift"] = scale_shift
            new_modules["total_energy_sum"] = reduce
        else:
            new_modules[name] = child

    # Build new SequentialGraphNetwork
    new_seq = SequentialGraphNetwork(new_modules)

    _replace_seq_in_model(model, seq_net, new_seq)

    # Update model's irreps
    model._init_irreps(
        irreps_in=model.irreps_in,
        irreps_out=new_seq.irreps_out if hasattr(model, 'model') and hasattr(model.model, 'irreps_out') else model.irreps_out,
    )

    return model


class SummedHeadsConvReadout(GraphModuleMixin, torch.nn.Module):
    """Runs multiple heads' full pipelines (conv → readout → scale_shift) and sums.

    Used when extracting summed heads from a model with ``PerHeadConvNetLayer``.
    Each head has its own ``SingleHeadConv`` + ``ScalarMLP`` + ``PerTypeScaleShift``.
    """

    def __init__(self, head_pipelines: list):
        """Args:
            head_pipelines: list of ``(single_head_conv, readout, scale_shift)`` tuples.
        """
        super().__init__()
        self.head_convs = torch.nn.ModuleList([c for c, _, _ in head_pipelines])
        self.head_readouts = torch.nn.ModuleList([r for _, r, _ in head_pipelines])
        self.head_scale_shifts = torch.nn.ModuleList(
            [s for _, _, s in head_pipelines]
        )
        # Output irreps: PER_ATOM_ENERGY_KEY from scale_shift
        # We explicitly set irreps_out to avoid inheriting NODE_FEATURES_KEY
        # from the conv's irreps_in (which has full backbone irreps)
        self._init_irreps(
            irreps_in=head_pipelines[0][0].irreps_in,
            irreps_out=head_pipelines[0][2].irreps_out,
        )
        # Override NODE_FEATURES_KEY in irreps_out to match the conv output (scalars)
        from e3nn.o3 import Irreps

        self.irreps_out[AtomicDataDict.NODE_FEATURES_KEY] = (
            head_pipelines[0][0].irreps_out[AtomicDataDict.NODE_FEATURES_KEY]
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        total = None
        for conv, readout, scale_shift in zip(
            self.head_convs, self.head_readouts, self.head_scale_shifts
        ):
            head_data = {k: v for k, v in data.items()}
            head_data = conv(head_data)
            head_data = readout(head_data)
            head_data = scale_shift(head_data)
            e = head_data[AtomicDataDict.PER_ATOM_ENERGY_KEY]
            total = e if total is None else total + e

        data[AtomicDataDict.PER_ATOM_ENERGY_KEY] = total
        # Update NODE_FEATURES_KEY to match the conv output dimensions
        # (the last SingleHeadConv already set it, but we need it consistent)
        data[AtomicDataDict.NODE_FEATURES_KEY] = head_data[
            AtomicDataDict.NODE_FEATURES_KEY
        ]
        return data


class SummedHeadsReadout(GraphModuleMixin, torch.nn.Module):
    """Runs multiple heads' readout+scale_shift pipelines and sums per-atom energies.

    Each head's pipeline (readout MLP -> PerTypeScaleShift) is run independently,
    producing a per-atom energy tensor. All per-atom energies are then summed
    element-wise. This enables deploying a single model that computes e.g.
    ``E_base + E_delta`` from a multi-head training run.

    Args:
        head_pipelines: list of ``(readout, scale_shift)`` tuples, where each
            readout is a :class:`~nequip.nn.ScalarMLP` (or fused/wrapper) and
            each scale_shift is a :class:`~nequip.nn.PerTypeScaleShift`.
    """

    def __init__(self, head_pipelines: list):
        super().__init__()
        self.head_readouts = torch.nn.ModuleList(
            [r for r, _ in head_pipelines]
        )
        self.head_scale_shifts = torch.nn.ModuleList(
            [s for _, s in head_pipelines]
        )
        # Use first pipeline's irreps for the module
        first_readout = head_pipelines[0][0]
        last_scale_shift = head_pipelines[0][1]
        self._init_irreps(
            irreps_in=first_readout.irreps_in,
            irreps_out=last_scale_shift.irreps_out,
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # We must clone tensors that readout/scale_shift will overwrite,
        # so that AOT tracing doesn't see aliased writes through shared
        # dict references. The readout writes PER_ATOM_ENERGY_KEY and
        # scale_shift reads/writes it, so we need a fresh dict per head.
        total = None
        for readout, scale_shift in zip(
            self.head_readouts, self.head_scale_shifts
        ):
            head_data = {k: v for k, v in data.items()}
            head_data = readout(head_data)
            head_data = scale_shift(head_data)
            e = head_data[AtomicDataDict.PER_ATOM_ENERGY_KEY]
            total = e if total is None else total + e

        data[AtomicDataDict.PER_ATOM_ENERGY_KEY] = total
        return data


class SummedHeadsEdgeReadout(GraphModuleMixin, torch.nn.Module):
    """Runs multiple heads' edge readout pipelines and sums per-atom energies.

    Each head's pipeline (readout MLP -> EdgewiseReduce -> PerTypeScaleShift)
    is run independently. All per-atom energies are then summed element-wise.
    This is the Allegro equivalent of :class:`SummedHeadsReadout`.

    Args:
        head_pipelines: list of ``(readout, edge_reduce, scale_shift)`` tuples.
    """

    def __init__(self, head_pipelines: list):
        super().__init__()
        self.head_readouts = torch.nn.ModuleList(
            [r for r, _, _ in head_pipelines]
        )
        self.head_edge_reduces = torch.nn.ModuleList(
            [e for _, e, _ in head_pipelines]
        )
        self.head_scale_shifts = torch.nn.ModuleList(
            [s for _, _, s in head_pipelines]
        )
        first_readout = head_pipelines[0][0]
        last_scale_shift = head_pipelines[0][2]
        self._init_irreps(
            irreps_in=first_readout.irreps_in,
            irreps_out=last_scale_shift.irreps_out,
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        total = None
        for readout, edge_reduce, scale_shift in zip(
            self.head_readouts, self.head_edge_reduces, self.head_scale_shifts
        ):
            # data.copy() is critical because EdgewiseReduce writes to
            # NODE_FEATURES_KEY as a side effect.
            head_data = {k: v for k, v in data.items()}
            head_data = readout(head_data)
            head_data = edge_reduce(head_data)
            head_data = scale_shift(head_data)
            e = head_data[AtomicDataDict.PER_ATOM_ENERGY_KEY]
            total = e if total is None else total + e

        data[AtomicDataDict.PER_ATOM_ENERGY_KEY] = total
        return data


def extract_summed_heads(model: GraphModel, head_names: list) -> GraphModel:
    """Extract multiple heads from a multi-head model and sum their outputs.

    The returned model runs each requested head's readout pipeline and sums
    the resulting per-atom energies. This is useful for delta-learning
    workflows where the deployed prediction is e.g.
    ``E_base + E_delta = E_target``.

    Supports both node-based (:class:`MultiHeadReadout`, NequIP) and
    edge-based (:class:`MultiHeadEdgeReadout`, Allegro) readout modules.

    The returned model does not require ``HEAD_KEY`` in input data.

    Args:
        model: A :class:`~nequip.nn.GraphModel` containing a multi-head readout.
        head_names: List of head names to sum (must all be present in the model).

    Returns:
        A new :class:`~nequip.nn.GraphModel` with the multi-head readout
        replaced by a summed readout + ``AtomwiseReduce``.
    """
    model = copy.deepcopy(model)

    mhr = _find_multihead_readout(model)
    if mhr is None:
        raise ValueError(
            "No MultiHeadReadout or MultiHeadEdgeReadout found in model. "
            "Is this a multi-head model?"
        )

    for hn in head_names:
        if hn not in mhr.head_names:
            raise ValueError(
                f"Head '{hn}' not found. "
                f"Available heads: {mhr.head_names}"
            )

    result = _find_seq_containing_readout(model)
    if result is None:
        raise ValueError(
            "Could not find SequentialGraphNetwork containing multi-head readout"
        )
    seq_net, multihead_key = result

    is_edge = _is_edge_based_readout(mhr)

    # Build per-head pipelines and summed readout
    if is_edge:
        head_pipelines = []
        for hn in head_names:
            hm = mhr.heads[hn]
            head_pipelines.append((hm["readout"], hm["edge_reduce"], hm["scale_shift"]))
        summed = SummedHeadsEdgeReadout(head_pipelines)
        reduce_irreps_in = head_pipelines[0][2].irreps_out
    else:
        head_pipelines_2 = []
        for hn in head_names:
            hm = mhr.heads[hn]
            head_pipelines_2.append((hm["readout"], hm["scale_shift"]))
        summed = SummedHeadsReadout(head_pipelines_2)
        reduce_irreps_in = head_pipelines_2[0][1].irreps_out

    # Create AtomwiseReduce
    reduce = AtomwiseReduce(
        irreps_in=reduce_irreps_in,
        reduce="sum",
        field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
    )

    # Check for PerHeadConvNetLayer (NequIP only) — needs special handling
    per_head_conv = None
    per_head_conv_key = None
    if not is_edge:
        for name, child in seq_net.named_children():
            if _is_instance_by_name(child, PerHeadConvNetLayer):
                per_head_conv = child
                per_head_conv_key = name
                break

    # Replace multi-head readout (and PerHeadConvNetLayer if present)
    new_modules = {}
    if per_head_conv is not None:
        # Build combined conv+readout pipeline per head, then sum
        conv_pipelines = []
        for hn in head_names:
            conv = SingleHeadConv(per_head_conv, hn)
            readout = mhr.heads[hn]["readout"]
            scale_shift = mhr.heads[hn]["scale_shift"]
            conv_pipelines.append((conv, readout, scale_shift))

        summed_conv_readout = SummedHeadsConvReadout(conv_pipelines)

        for name, child in seq_net.named_children():
            if name == per_head_conv_key:
                continue  # skip PerHeadConvNetLayer
            elif name == multihead_key:
                new_modules["summed_heads_readout"] = summed_conv_readout
                new_modules["total_energy_sum"] = reduce
            else:
                new_modules[name] = child
    else:
        for name, child in seq_net.named_children():
            if name == multihead_key:
                new_modules["summed_heads_readout"] = summed
                new_modules["total_energy_sum"] = reduce
            else:
                new_modules[name] = child

    new_seq = SequentialGraphNetwork(new_modules)

    _replace_seq_in_model(model, seq_net, new_seq)

    model._init_irreps(
        irreps_in=model.irreps_in,
        irreps_out=new_seq.irreps_out if hasattr(model, 'model') and hasattr(model.model, 'irreps_out') else model.irreps_out,
    )

    return model
