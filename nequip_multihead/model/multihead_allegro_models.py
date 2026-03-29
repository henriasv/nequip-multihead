"""Multi-head Allegro model builders.

These use Allegro's standard backbone components (TwoBodySphericalHarmonicTensorEmbed,
Allegro_Module, EdgewiseReduce) and append multi-head edge readout layers on top.
"""
import warnings
from e3nn import o3

from nequip.data import AtomicDataDict
from nequip.nn import (
    SequentialGraphNetwork,
    ScalarMLP,
    AtomwiseReduce,
    ForceStressOutput,
)
from nequip.nn.embedding import (
    EdgeLengthNormalizer,
    AddRadialCutoffToData,
    PolynomialCutoff,
)
from nequip.model.utils import model_builder

from allegro.nn import (
    TwoBodySphericalHarmonicTensorEmbed,
    EdgewiseReduce,
    Allegro_Module,
)

from nequip_multihead.nn.multihead_edge_readout import MultiHeadEdgeReadout
from nequip_multihead._keys import HEAD_KEY

from hydra.utils import instantiate
from typing import Sequence, Union, Optional, Dict, List


@model_builder
def MultiHeadAllegroModel(
    l_max: int,
    parity: bool = True,
    head_names: List[str] = None,
    **kwargs,
):
    """Multi-head Allegro model with shared backbone and per-head edge readouts.

    Args:
        l_max: maximum spherical harmonics order
        parity: whether to include features with odd mirror parity
        head_names: list of head name strings (required)
        (other args same as AllegroModel)
    """
    assert head_names is not None and len(head_names) >= 1, (
        "head_names is required for MultiHeadAllegroModel"
    )

    irreps_edge_sh = repr(o3.Irreps.spherical_harmonics(l_max, p=-1))
    # set tensor_track_allowed_irreps
    if parity:
        tensor_track_allowed_irreps = o3.Irreps(
            [(1, (this_l, p)) for this_l in range(l_max + 1) for p in (1, -1)]
        )
    else:
        tensor_track_allowed_irreps = irreps_edge_sh

    return _FullMultiHeadAllegroModel(
        irreps_edge_sh=irreps_edge_sh,
        tensor_track_allowed_irreps=tensor_track_allowed_irreps,
        head_names=head_names,
        **kwargs,
    )


@model_builder
def _FullMultiHeadAllegroModel(
    r_max: float,
    type_names: Sequence[str],
    # irreps
    irreps_edge_sh: Union[int, str, o3.Irreps],
    tensor_track_allowed_irreps: Union[str, o3.Irreps],
    # scalar embed
    radial_chemical_embed: Dict,
    radial_chemical_embed_dim: Optional[int] = None,
    per_edge_type_cutoff: Optional[Dict[str, Union[float, Dict[str, float]]]] = None,
    # scalar embed MLP
    scalar_embed_mlp_hidden_layers_depth: int = 1,
    scalar_embed_mlp_hidden_layers_width: int = 64,
    scalar_embed_mlp_nonlinearity: str = "silu",
    # allegro layers
    num_layers: int = 2,
    num_scalar_features: int = 64,
    num_tensor_features: int = 16,
    allegro_mlp_hidden_layers_depth: int = 1,
    allegro_mlp_hidden_layers_width: int = 64,
    allegro_mlp_nonlinearity: Optional[str] = "silu",
    tp_path_channel_coupling: bool = True,
    # readout
    readout_mlp_hidden_layers_depth: int = 1,
    readout_mlp_hidden_layers_width: int = 32,
    readout_mlp_nonlinearity: Optional[str] = "silu",
    # edge sum normalization
    avg_num_neighbors: Union[float, Dict[str, float]] = None,
    # allegro layers defaults
    weight_individual_irreps: bool = True,
    # per atom energy params
    per_type_energy_scales=None,
    per_type_energy_shifts=None,
    per_type_energy_scales_trainable: Optional[bool] = False,
    per_type_energy_shifts_trainable: Optional[bool] = False,
    pair_potential: Optional[Dict] = None,
    # derivatives
    do_derivatives: bool = True,
    # weight initialization and normalization
    forward_normalize: bool = True,
    # multi-head params
    head_names: List[str] = None,
):
    """Full multi-head Allegro model with explicit parameters."""
    assert all(tn.isalnum() for tn in type_names)

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

    # === two-body scalar embedding (identical to FullAllegroModel) ===
    edge_norm = EdgeLengthNormalizer(
        r_max=r_max,
        type_names=type_names,
        per_edge_type_cutoff=per_edge_type_cutoff,
    )
    radial_chemical_embed_module = instantiate(
        radial_chemical_embed,
        type_names=type_names,
        module_output_dim=(
            num_scalar_features
            if radial_chemical_embed_dim is None
            else radial_chemical_embed_dim
        ),
        forward_weight_init=forward_normalize,
        scalar_embed_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
        irreps_in=edge_norm.irreps_out,
    )

    scalar_embed_mlp = ScalarMLP(
        output_dim=num_scalar_features,
        hidden_layers_depth=scalar_embed_mlp_hidden_layers_depth,
        hidden_layers_width=scalar_embed_mlp_hidden_layers_width,
        nonlinearity=scalar_embed_mlp_nonlinearity,
        bias=False,
        forward_weight_init=forward_normalize,
        field=AtomicDataDict.EDGE_EMBEDDING_KEY,
        out_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
        irreps_in=radial_chemical_embed_module.irreps_out,
    )

    # === two-body tensor embedding ===
    tensor_embed = TwoBodySphericalHarmonicTensorEmbed(
        irreps_edge_sh=irreps_edge_sh,
        num_tensor_features=num_tensor_features,
        forward_weight_init=forward_normalize,
        scalar_embedding_in_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
        tensor_basis_out_field=AtomicDataDict.EDGE_ATTRS_KEY,
        tensor_embedding_out_field=AtomicDataDict.EDGE_FEATURES_KEY,
        irreps_in=scalar_embed_mlp.irreps_out,
    )

    # === allegro module ===
    allegro = Allegro_Module(
        num_layers=num_layers,
        num_scalar_features=num_scalar_features,
        num_tensor_features=num_tensor_features,
        tensor_track_allowed_irreps=tensor_track_allowed_irreps,
        avg_num_neighbors=avg_num_neighbors,
        type_names=type_names,
        latent_kwargs={
            "hidden_layers_depth": allegro_mlp_hidden_layers_depth,
            "hidden_layers_width": allegro_mlp_hidden_layers_width,
            "nonlinearity": allegro_mlp_nonlinearity,
            "bias": False,
            "forward_weight_init": forward_normalize,
        },
        tp_path_channel_coupling=tp_path_channel_coupling,
        weight_individual_irreps=weight_individual_irreps,
        tensor_basis_in_field=AtomicDataDict.EDGE_ATTRS_KEY,
        tensor_features_in_field=AtomicDataDict.EDGE_FEATURES_KEY,
        scalar_in_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
        scalar_out_field=AtomicDataDict.EDGE_FEATURES_KEY,
        irreps_in=tensor_embed.irreps_out,
    )

    modules = {
        "edge_norm": edge_norm,
        "radial_chemical_embed": radial_chemical_embed_module,
        "scalar_embed_mlp": scalar_embed_mlp,
        "tensor_embed": tensor_embed,
        "allegro": allegro,
    }

    # === multi-head edge readout ===
    # Handle per_type_energy_scales/shifts as dict-of-dicts for multi-head
    per_head_scales = per_type_energy_scales
    per_head_shifts = per_type_energy_shifts

    if isinstance(per_head_scales, dict) and "all" in per_head_scales:
        per_head_scales = {name: per_head_scales["all"] for name in head_names}
    if isinstance(per_head_shifts, dict) and "all" in per_head_shifts:
        per_head_shifts = {name: per_head_shifts["all"] for name in head_names}

    multihead_readout = MultiHeadEdgeReadout(
        head_names=head_names,
        type_names=type_names,
        readout_mlp_hidden_layers_depth=readout_mlp_hidden_layers_depth,
        readout_mlp_hidden_layers_width=readout_mlp_hidden_layers_width,
        readout_mlp_nonlinearity=readout_mlp_nonlinearity,
        per_head_energy_scales=per_head_scales,
        per_head_energy_shifts=per_head_shifts,
        per_type_energy_scales_trainable=per_type_energy_scales_trainable,
        per_type_energy_shifts_trainable=per_type_energy_shifts_trainable,
        avg_num_neighbors=avg_num_neighbors,
        forward_normalize=forward_normalize,
        irreps_in=allegro.irreps_out,
    )
    modules["multihead_readout"] = multihead_readout

    # === pair potentials ===
    prev_irreps_out = multihead_readout.irreps_out
    if pair_potential is not None:
        if AtomicDataDict.EDGE_CUTOFF_KEY not in prev_irreps_out:
            cutoff = AddRadialCutoffToData(
                cutoff=PolynomialCutoff(6),
                irreps_in=prev_irreps_out,
            )
            prev_irreps_out = cutoff.irreps_out
            modules["cutoff"] = cutoff

        pair_potential = instantiate(
            pair_potential,
            type_names=type_names,
            irreps_in=prev_irreps_out,
        )
        prev_irreps_out = pair_potential.irreps_out
        modules["pair_potential"] = pair_potential

    # === finalize model ===
    energy_model = SequentialGraphNetwork(modules)
    fso = ForceStressOutput(energy_model, do_derivatives)

    # Add HEAD_KEY to the model's irreps_in so that GraphModel passes it
    # through to the inner model (where MultiHeadEdgeReadout uses it).
    fso.irreps_in[HEAD_KEY] = None
    return fso
