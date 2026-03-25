"""Per-head ConvNetLayer with shared edge MLP and per-head tensor product paths.

Each head can use a different l_max for the input features while sharing the
same edge MLP weights. Heads with lower l_max use a strict subset of the
tensor product paths (and weights) of heads with higher l_max.
"""
import torch
from math import prod

from e3nn.o3._irreps import Irreps
from e3nn.o3._linear import Linear
from e3nn.o3._tensor_product._sub import FullyConnectedTensorProduct

from nequip.data import AtomicDataDict
from nequip.nn._graph_mixin import GraphModuleMixin
from nequip.nn.mlp import ScalarMLPFunction
from nequip.nn._tp_scatter_base import TensorProductScatter
from nequip.nn.norm import AvgNumNeighborsNorm
from nequip.nn._ghost_exchange_base import NoOpGhostExchangeModule

from typing import Dict, List, Optional, Sequence, Union, Callable, Any


class PerHeadConvNetLayer(GraphModuleMixin, torch.nn.Module):
    """ConvNetLayer with per-head tensor product paths and shared edge MLP.

    Replaces the last ConvNetLayer in a multi-head model. Each head can
    use a different ``l_max`` for input features. The edge MLP is shared
    across all heads; each head's TensorProduct uses a subset of the
    full instruction set (filtered by input irrep ``l <= head_l_max``).

    Args:
        irreps_in: input irreps dict (from previous ConvNetLayer)
        head_names: list of head name strings
        per_head_l_max: dict mapping head name to l_max for that head
        radial_mlp_depth: depth of shared edge MLP
        radial_mlp_width: width of shared edge MLP
        use_sc: whether to use self-connection
        is_first_layer: whether this is the first layer
        avg_num_neighbors: for normalization
        type_names: atom type names
        nonlinearity_scalars: activation functions for scalars
    """

    def __init__(
        self,
        irreps_in,
        head_names: List[str],
        per_head_l_max: Dict[str, int],
        radial_mlp_depth: int = 1,
        radial_mlp_width: int = 8,
        use_sc: bool = True,
        is_first_layer: bool = False,
        type_names: Optional[Sequence[str]] = None,
        avg_num_neighbors: Optional[Union[float, Dict[str, float]]] = None,
        nonlinearity_scalars: Dict[str, str] = {"e": "silu", "o": "tanh"},
    ):
        super().__init__()

        self.head_names = head_names
        self.per_head_l_max = per_head_l_max
        self.is_first_layer = is_first_layer

        acts = {
            "abs": torch.abs,
            "tanh": torch.tanh,
            "silu": torch.nn.functional.silu,
        }
        nonlinearity_scalars = {
            1: nonlinearity_scalars["e"],
            -1: nonlinearity_scalars["o"],
        }

        # Output is scalar-only for each head (readout expects scalars)
        # But each head may have a different number of scalar features
        # depending on how many TP paths it uses.
        # We'll compute per-head output irreps and set the "max" as module output.

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[
                AtomicDataDict.EDGE_EMBEDDING_KEY,
                AtomicDataDict.EDGE_ATTRS_KEY,
                AtomicDataDict.NODE_FEATURES_KEY,
                AtomicDataDict.NODE_ATTRS_KEY,
            ],
            my_irreps_in={
                AtomicDataDict.EDGE_EMBEDDING_KEY: Irreps(
                    [
                        (
                            irreps_in[AtomicDataDict.EDGE_EMBEDDING_KEY].num_irreps,
                            (0, 1),
                        )
                    ]
                )
            },
        )

        feature_irreps_in = self.irreps_in[AtomicDataDict.NODE_FEATURES_KEY]
        irreps_edge_attr = self.irreps_in[AtomicDataDict.EDGE_ATTRS_KEY]

        # === Shared components ===
        self.avg_num_neighbors_norm = AvgNumNeighborsNorm(
            avg_num_neighbors=avg_num_neighbors, type_names=type_names
        )

        self.linear_1 = Linear(
            irreps_in=feature_irreps_in,
            irreps_out=feature_irreps_in,
            internal_weights=True,
            shared_weights=True,
        )

        self.ghost_exchange = NoOpGhostExchangeModule(
            field=AtomicDataDict.NODE_FEATURES_KEY, irreps_in=self.irreps_in
        )

        # === Build full instruction set (max l_max across heads) ===
        max_l_max = max(per_head_l_max.values())
        # Target: scalar-only output
        feature_irreps_out = Irreps(
            [
                (mul, ir)
                for mul, ir in feature_irreps_in
                if ir.l == 0
            ]
        )

        full_instructions = []
        full_irreps_mid = []
        for i, (mul, ir_in) in enumerate(feature_irreps_in):
            for j, (_, ir_edge) in enumerate(irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in feature_irreps_out:
                        k = len(full_irreps_mid)
                        full_irreps_mid.append((mul, ir_out))
                        full_instructions.append((i, j, k, "uvu", True))

        full_irreps_mid = Irreps(full_irreps_mid)
        full_irreps_mid_sorted, full_perm, _ = full_irreps_mid.sort()
        full_instructions_sorted = [
            (i_in1, i_in2, full_perm[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in full_instructions
        ]

        # Build the full TP to get the canonical weight layout
        full_tp_scatter = TensorProductScatter(
            feature_irreps_in,
            irreps_edge_attr,
            full_irreps_mid_sorted,
            full_instructions_sorted,
        )
        full_weight_numel = full_tp_scatter.tp.weight_numel

        # === Shared edge MLP (outputs weights for ALL paths) ===
        self.edge_mlp = ScalarMLPFunction(
            input_dim=self.irreps_in[AtomicDataDict.EDGE_EMBEDDING_KEY].num_irreps,
            output_dim=full_weight_numel,
            hidden_layers_depth=radial_mlp_depth,
            hidden_layers_width=radial_mlp_width,
            nonlinearity="silu",
            bias=False,
            forward_weight_init=True,
        )

        # === Per-head components ===
        # For each head, determine which instructions to use based on l_max,
        # build a TP, and compute weight index mapping from full → head weights.

        heads = {}
        per_head_scalar_dims = {}
        per_head_weight_numels = {}

        for head_name in head_names:
            head_l_max = per_head_l_max[head_name]

            # Filter instructions: keep only those where input feature l <= head_l_max
            head_instructions = []
            head_irreps_mid = []
            # Track which full instructions this head uses
            head_full_instruction_indices = []

            for idx, (i_in1, i_in2, i_out_full, mode, train) in enumerate(
                full_instructions
            ):
                _, ir_in = feature_irreps_in[i_in1]
                if ir_in.l <= head_l_max:
                    k = len(head_irreps_mid)
                    mul = feature_irreps_in[i_in1].mul
                    ir_out = full_irreps_mid[i_out_full] if i_out_full < len(full_irreps_mid) else full_irreps_mid[full_perm.tolist().index(i_out_full)]
                    head_irreps_mid.append((mul, ir_out[1] if isinstance(ir_out, tuple) else ir_out.ir))
                    head_instructions.append((i_in1, i_in2, k, mode, train))
                    head_full_instruction_indices.append(idx)

            head_irreps_mid = Irreps(head_irreps_mid)
            head_irreps_mid_sorted, head_perm, _ = head_irreps_mid.sort()
            head_instructions_sorted = [
                (i_in1, i_in2, head_perm[i_out], mode, train)
                for i_in1, i_in2, i_out, mode, train in head_instructions
            ]

            head_tp_scatter = TensorProductScatter(
                feature_irreps_in,
                irreps_edge_attr,
                head_irreps_mid_sorted,
                head_instructions_sorted,
            )

            # The head's TP uses a contiguous prefix of the full TP's
            # weight vector because instructions are sorted by l and
            # lower-l paths come first. Verify this property holds.

            # linear_2: maps head's irreps_mid → scalar output
            head_linear_2 = Linear(
                irreps_in=head_irreps_mid_sorted.simplify(),
                irreps_out=feature_irreps_out,
                internal_weights=True,
                shared_weights=True,
            )

            # Scalar activation
            head_activation = torch.nn.SiLU()

            # Self-connection (optional)
            head_sc = None
            if use_sc:
                head_sc = FullyConnectedTensorProduct(
                    feature_irreps_in,
                    self.irreps_in[AtomicDataDict.NODE_ATTRS_KEY],
                    feature_irreps_out,
                )

            head_modules = torch.nn.ModuleDict({
                "tp_scatter": head_tp_scatter,
                "linear_2": head_linear_2,
            })
            if head_sc is not None:
                head_modules["sc"] = head_sc

            heads[head_name] = head_modules

            head_weight_numel = head_tp_scatter.tp.weight_numel
            assert head_weight_numel <= full_weight_numel, (
                f"Head '{head_name}' weight_numel ({head_weight_numel}) exceeds "
                f"full weight_numel ({full_weight_numel})"
            )

            # Verify contiguous prefix property by checking that matching
            # instructions appear in the same order in both TPs
            self._verify_contiguous_weights(
                full_tp_scatter.tp, head_tp_scatter.tp, head_name
            )

            # Track per-head weight count for contiguous slicing
            per_head_weight_numels[head_name] = head_weight_numel

            # Track output dimensions
            per_head_scalar_dims[head_name] = feature_irreps_out.dim

        self.heads = torch.nn.ModuleDict(heads)
        self.per_head_scalar_dims = per_head_scalar_dims
        self._head_weight_numels = per_head_weight_numels
        self._activation = torch.nn.SiLU()

        # Set output irreps: scalar features for NODE_FEATURES_KEY,
        # pass through all other keys from input (edge attrs, etc.)
        self.irreps_out.update(self.irreps_in)
        self.irreps_out[AtomicDataDict.NODE_FEATURES_KEY] = feature_irreps_out

    @staticmethod
    def _verify_contiguous_weights(full_tp, head_tp, head_name: str):
        """Verify that the head's TP weights are a contiguous prefix of the full TP's.

        This property holds because TP instructions are sorted by output irreps
        (all scalar l=0), and input irreps are sorted by l. Lower-l input paths
        appear first in both the full and head TPs, so the head's weight block
        is always a prefix of the full weight block.
        """
        # Match each head instruction to the corresponding full instruction
        full_offset = 0
        head_offset = 0
        h_idx = 0
        head_instructions = [ins for ins in head_tp.instructions if ins.has_weight]
        full_instructions = [ins for ins in full_tp.instructions if ins.has_weight]

        for f_ins in full_instructions:
            f_size = prod(f_ins.path_shape)
            if h_idx < len(head_instructions):
                h_ins = head_instructions[h_idx]
                if (
                    f_ins.i_in1 == h_ins.i_in1
                    and f_ins.i_in2 == h_ins.i_in2
                    and f_ins.path_shape == h_ins.path_shape
                ):
                    # This full instruction matches the next head instruction
                    assert full_offset == head_offset, (
                        f"Head '{head_name}' weight offset mismatch: "
                        f"full offset {full_offset} != head offset {head_offset} "
                        f"at instruction {h_idx}. Weights are not contiguous."
                    )
                    head_offset += prod(h_ins.path_shape)
                    h_idx += 1
            full_offset += f_size

        assert h_idx == len(head_instructions), (
            f"Head '{head_name}': only matched {h_idx}/{len(head_instructions)} "
            f"instructions to full TP"
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        """Run per-head convolution.

        Stores per-head scalar features in ``data`` under keys
        ``_per_head_features_{head_name}`` for downstream use by
        ``MultiHeadReadout``. Returns the modified data dict.
        """
        if AtomicDataDict.LMP_MLIAP_DATA_KEY in data:
            num_local_nodes = data[AtomicDataDict.LMP_MLIAP_DATA_KEY].nlocal
        else:
            num_local_nodes = AtomicDataDict.num_nodes(data)

        x = data[AtomicDataDict.NODE_FEATURES_KEY]

        # Truncate if not first layer
        if not self.is_first_layer:
            x = x[:num_local_nodes]

        # Shared self-connection input (before linear_1)
        sc_inputs = {}
        for head_name in self.head_names:
            if "sc" in self.heads[head_name]:
                node_attrs = data[AtomicDataDict.NODE_ATTRS_KEY]
                if not self.is_first_layer:
                    node_attrs = node_attrs[:num_local_nodes]
                sc_inputs[head_name] = self.heads[head_name]["sc"](x, node_attrs)

        # Shared linear_1
        x = self.linear_1(x)

        # Shared normalization
        data_copy = data.copy()
        data_copy[AtomicDataDict.NODE_FEATURES_KEY] = x
        data_copy = self.avg_num_neighbors_norm(data_copy)
        x = data_copy[AtomicDataDict.NODE_FEATURES_KEY]

        # Ghost exchange (if not first layer)
        if not self.is_first_layer:
            data_copy[AtomicDataDict.NODE_FEATURES_KEY] = x
            data_copy = self.ghost_exchange(data_copy, ghost_included=False)
            x = data_copy[AtomicDataDict.NODE_FEATURES_KEY]

        # Shared edge MLP → full weight vector
        edge_weights = self.edge_mlp(data[AtomicDataDict.EDGE_EMBEDDING_KEY])

        # Per-head TP + scatter + linear_2 + activation
        head_outputs = {}
        edge_attr = data[AtomicDataDict.EDGE_ATTRS_KEY]
        edge_dst = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_src = data[AtomicDataDict.EDGE_INDEX_KEY][1]

        for head_name in self.head_names:
            # Slice weights for this head (contiguous from 0 since TP
            # instructions are sorted by l, lower-l paths come first)
            head_weight_numel = self._head_weight_numels[head_name]
            head_weights = edge_weights[:, :head_weight_numel]

            # TP + scatter
            head_x = self.heads[head_name]["tp_scatter"](
                x=x,
                edge_attr=edge_attr,
                edge_weight=head_weights,
                edge_dst=edge_dst,
                edge_src=edge_src,
            )[:num_local_nodes]

            # linear_2
            head_x = self.heads[head_name]["linear_2"](head_x)

            # Scalar activation
            head_x = self._activation(head_x)

            # Self-connection
            if head_name in sc_inputs:
                head_x = head_x + sc_inputs[head_name]

            data[f"_per_head_features_{head_name}"] = head_x

        # Set NODE_FEATURES_KEY to the first head's output so that
        # irreps_out validation passes in SequentialGraphNetwork.
        # MultiHeadReadout will override this per-head anyway.
        data[AtomicDataDict.NODE_FEATURES_KEY] = data[
            f"_per_head_features_{self.head_names[0]}"
        ]

        return data
