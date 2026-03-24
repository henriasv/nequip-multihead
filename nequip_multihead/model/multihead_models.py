"""Multi-head NequIP model builders.

These use NequIP's standard backbone components (ConvNetLayer, embeddings)
and append multi-head readout layers on top.
"""
import math
import warnings
from e3nn import o3

from nequip.data import AtomicDataDict
from nequip.nn import (
    GraphModel,
    SequentialGraphNetwork,
    ScalarMLP,
    PerTypeScaleShift,
    ConvNetLayer,
    ForceStressOutput,
    ApplyFactor,
)
from nequip.nn.embedding import (
    NodeTypeEmbed,
    PolynomialCutoff,
    EdgeLengthNormalizer,
    BesselEdgeLengthEncoding,
    SphericalHarmonicEdgeAttrs,
)
from nequip.model.utils import model_builder
from nequip.model.energy_modules import _append_energy_modules

from nequip_multihead.nn import MultiHeadReadout, PerHeadConvNetLayer
from nequip_multihead._keys import HEAD_KEY

from typing import Sequence, Optional, List, Dict, Union, Callable


@model_builder
def MultiHeadNequIPGNNModel(
    num_layers: int = 4,
    l_max: int = 1,
    parity: bool = True,
    num_features: Union[int, List[int]] = 32,
    type_embed_num_features: Optional[int] = None,
    radial_mlp_depth: int = 1,
    radial_mlp_width: int = 128,
    # multi-head params
    head_names: List[str] = None,
    per_head_l_max: Optional[Dict[str, int]] = None,
    **kwargs,
) -> GraphModel:
    """Multi-head NequIP GNN model with shared backbone and per-head readouts.

    Args:
        head_names: list of head name strings (required)
        per_head_l_max: dict mapping head name to l_max for the final layer.
            Heads with lower l_max use fewer TP paths, constraining the
            per-atom energy decomposition. Default: all heads use backbone l_max.
        (other args same as NequIPGNNModel)
    """
    assert head_names is not None and len(head_names) >= 1, (
        "head_names is required for MultiHeadNequIPGNNModel"
    )
    assert num_layers > 0

    if per_head_l_max is not None:
        for hn, hl in per_head_l_max.items():
            assert hl <= l_max, (
                f"per_head_l_max['{hn}'] = {hl} exceeds backbone l_max = {l_max}"
            )

    # === spherical harmonics ===
    irreps_edge_sh = repr(o3.Irreps.spherical_harmonics(lmax=l_max))

    # === handle num_features ===
    if isinstance(num_features, int):
        num_features = [num_features] * (l_max + 1)
    assert len(num_features) == l_max + 1

    type_embed_num_features = (
        type_embed_num_features
        if type_embed_num_features is not None
        else num_features[0]
    )

    # === convnet hidden irreps ===
    feature_irreps_hidden = repr(
        o3.Irreps(
            [
                (num_features[l], (l, p))
                for l in range(l_max + 1)
                for p in (
                    (1, -1) if parity else ((1,) if l % 2 == 0 else (-1,))
                )
            ]
        )
    )

    if per_head_l_max is not None:
        # With per-head l_max, shared layers keep full irreps;
        # PerHeadConvNetLayer replaces the last ConvNetLayer
        feature_irreps_hidden_list = [feature_irreps_hidden] * (num_layers - 1)
        radial_mlp_depth_list = [radial_mlp_depth] * (num_layers - 1)
        radial_mlp_width_list = [radial_mlp_width] * (num_layers - 1)
    else:
        feature_irreps_hidden_list = [feature_irreps_hidden] * (num_layers - 1)
        radial_mlp_depth_list = [radial_mlp_depth] * num_layers
        radial_mlp_width_list = [radial_mlp_width] * num_layers
        # Last layer outputs scalars only
        feature_irreps_hidden_list += [repr(o3.Irreps([(num_features[0], (0, 1))]))]

    # Build the full model
    model = _FullMultiHeadNequIPGNNModel(
        irreps_edge_sh=irreps_edge_sh,
        type_embed_num_features=type_embed_num_features,
        feature_irreps_hidden=feature_irreps_hidden_list,
        radial_mlp_depth=radial_mlp_depth_list,
        radial_mlp_width=radial_mlp_width_list,
        head_names=head_names,
        per_head_l_max=per_head_l_max,
        per_head_conv_radial_mlp_depth=radial_mlp_depth,
        per_head_conv_radial_mlp_width=radial_mlp_width,
        **kwargs,
    )
    return model


@model_builder
def _FullMultiHeadNequIPGNNModel(
    r_max: float,
    type_names: Sequence[str],
    # convnet params
    radial_mlp_depth: Sequence[int],
    radial_mlp_width: Sequence[int],
    feature_irreps_hidden: Sequence[Union[str, o3.Irreps]],
    # irreps and dims
    irreps_edge_sh: Union[int, str, o3.Irreps],
    type_embed_num_features: int,
    categorical_graph_field_embed: Optional[List[Dict[str, int]]] = None,
    # readout
    readout_mlp_hidden_layers_depth: int = 0,
    readout_mlp_hidden_layers_width: Optional[int] = None,
    readout_mlp_nonlinearity: Optional[str] = "silu",
    # edge length encoding
    per_edge_type_cutoff: Optional[Dict[str, Union[float, Dict[str, float]]]] = None,
    num_bessels: int = 8,
    bessel_trainable: bool = False,
    polynomial_cutoff_p: int = 6,
    # edge sum normalization
    avg_num_neighbors: Optional[Union[float, Dict[str, float]]] = None,
    # per atom energy params
    per_type_energy_scales=None,
    per_type_energy_shifts=None,
    per_type_energy_scales_trainable: Optional[bool] = False,
    per_type_energy_shifts_trainable: Optional[bool] = False,
    pair_potential: Optional[Dict] = None,
    # multi-head params
    head_names: List[str] = None,
    per_head_l_max: Optional[Dict[str, int]] = None,
    per_head_conv_radial_mlp_depth: Optional[int] = None,
    per_head_conv_radial_mlp_width: Optional[int] = None,
    # derivatives
    do_derivatives: bool = True,
    # convnet options
    convnet_sc: bool = True,
    learnable_shift: bool = False,
    convnet_resnet: bool = False,
    convnet_nonlinearity_type: str = "gate",
    convnet_nonlinearity_scalars: Dict[int, Callable] = {"e": "silu", "o": "tanh"},
    convnet_nonlinearity_gates: Dict[int, Callable] = {"e": "silu", "o": "tanh"},
) -> GraphModel:
    """Full multi-head NequIP model with explicit parameters."""
    assert all(tn.isalnum() for tn in type_names)
    assert not learnable_shift or (convnet_sc or convnet_resnet)

    num_layers = len(radial_mlp_depth)
    assert len(radial_mlp_depth) == len(radial_mlp_width) == len(feature_irreps_hidden)

    if per_head_l_max is None:
        assert all(l == 0 for l in o3.Irreps(feature_irreps_hidden[-1]).ls)

    if per_type_energy_scales is None:
        warnings.warn(
            "Found `per_type_energy_scales=None` -- it is recommended to set "
            "`per_type_energy_scales` for better numerics during training."
        )
    if per_type_energy_shifts is None:
        warnings.warn(
            "Found `per_type_energy_shifts=None` -- it is HIGHLY recommended to set "
            "`per_type_energy_shifts`."
        )

    # === encode and embed features ===
    type_embed = NodeTypeEmbed(
        type_names=type_names,
        num_features=type_embed_num_features,
        categorical_graph_field_embed=categorical_graph_field_embed,
    )
    spharm = SphericalHarmonicEdgeAttrs(
        irreps_edge_sh=irreps_edge_sh,
        irreps_in=type_embed.irreps_out,
    )
    edge_norm = EdgeLengthNormalizer(
        r_max=r_max,
        type_names=type_names,
        per_edge_type_cutoff=per_edge_type_cutoff,
        irreps_in=spharm.irreps_out,
    )
    bessel_encode = BesselEdgeLengthEncoding(
        num_bessels=num_bessels,
        trainable=bessel_trainable,
        cutoff=PolynomialCutoff(polynomial_cutoff_p),
        edge_invariant_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
        irreps_in=edge_norm.irreps_out,
    )
    factor = ApplyFactor(
        in_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
        factor=(2 * math.pi) / (r_max * r_max),
        irreps_in=bessel_encode.irreps_out,
    )

    modules = {
        "type_embed": type_embed,
        "spharm": spharm,
        "edge_norm": edge_norm,
        "bessel_encode": bessel_encode,
        "factor": factor,
    }
    prev_irreps_out = factor.irreps_out

    # === convnet layers ===
    for layer_i in range(num_layers):
        current_convnet = ConvNetLayer(
            irreps_in=prev_irreps_out,
            feature_irreps_hidden=feature_irreps_hidden[layer_i],
            convolution_kwargs={
                "radial_mlp_depth": radial_mlp_depth[layer_i],
                "radial_mlp_width": radial_mlp_width[layer_i],
                "use_sc": convnet_sc
                if learnable_shift
                else (layer_i != 0) and convnet_sc,
                "is_first_layer": layer_i == 0,
                "avg_num_neighbors": avg_num_neighbors,
                "type_names": type_names,
            },
            resnet=convnet_resnet
            if learnable_shift
            else (layer_i != 0) and convnet_resnet,
            nonlinearity_type=convnet_nonlinearity_type,
            nonlinearity_scalars=convnet_nonlinearity_scalars,
            nonlinearity_gates=convnet_nonlinearity_gates,
        )
        prev_irreps_out = current_convnet.irreps_out
        modules.update({f"layer{layer_i}_convnet": current_convnet})

    # === readout ===
    if readout_mlp_hidden_layers_width is None:
        readout_mlp_hidden_layers_width = o3.Irreps(feature_irreps_hidden[-1]).dim

    # === multi-head readout ===
    # Handle per_type_energy_scales/shifts as dict-of-dicts for multi-head
    per_head_scales = per_type_energy_scales
    per_head_shifts = per_type_energy_shifts

    if isinstance(per_head_scales, dict) and "all" in per_head_scales:
        per_head_scales = {name: per_head_scales["all"] for name in head_names}
    if isinstance(per_head_shifts, dict) and "all" in per_head_shifts:
        per_head_shifts = {name: per_head_shifts["all"] for name in head_names}

    # === per-head l_max: insert PerHeadConvNetLayer before readout ===
    if per_head_l_max is not None:
        full_per_head_l_max = {}
        backbone_l_max = max(
            ir.l
            for _, ir in o3.Irreps(
                prev_irreps_out[AtomicDataDict.NODE_FEATURES_KEY]
            )
        )
        for hn in head_names:
            full_per_head_l_max[hn] = per_head_l_max.get(hn, backbone_l_max)

        per_head_conv = PerHeadConvNetLayer(
            irreps_in=prev_irreps_out,
            head_names=head_names,
            per_head_l_max=full_per_head_l_max,
            radial_mlp_depth=per_head_conv_radial_mlp_depth or 1,
            radial_mlp_width=per_head_conv_radial_mlp_width or 128,
            use_sc=convnet_sc,
            is_first_layer=False,
            avg_num_neighbors=avg_num_neighbors,
            type_names=type_names,
            nonlinearity_scalars=convnet_nonlinearity_scalars,
        )
        modules.update({"per_head_conv": per_head_conv})
        prev_irreps_out = per_head_conv.irreps_out

    multihead_readout = MultiHeadReadout(
        head_names=head_names,
        type_names=type_names,
        readout_mlp_hidden_layers_depth=readout_mlp_hidden_layers_depth,
        readout_mlp_hidden_layers_width=readout_mlp_hidden_layers_width,
        readout_mlp_nonlinearity=readout_mlp_nonlinearity,
        per_head_energy_scales=per_head_scales,
        per_head_energy_shifts=per_head_shifts,
        per_type_energy_scales_trainable=per_type_energy_scales_trainable,
        per_type_energy_shifts_trainable=per_type_energy_shifts_trainable,
        irreps_in=prev_irreps_out,
    )
    modules.update({"multihead_readout": multihead_readout})

    # === finalize ===
    energy_model = SequentialGraphNetwork(modules)
    energy_model = _append_energy_modules(
        model=energy_model,
        type_names=type_names,
        pair_potential=pair_potential,
    )
    fso = ForceStressOutput(energy_model, do_derivatives)

    # Add HEAD_KEY to the model's irreps_in so that GraphModel passes it
    # through to the inner model (where MultiHeadReadout uses it).
    fso.irreps_in[HEAD_KEY] = None
    return fso
