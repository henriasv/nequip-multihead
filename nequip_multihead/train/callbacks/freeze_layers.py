"""Callback to freeze early interaction layers during finetuning."""
import logging

import lightning
import torch

from nequip.nn import ConvNetLayer, SequentialGraphNetwork

logger = logging.getLogger(__name__)


def _is_instance_by_name(obj, cls):
    """Check isinstance, falling back to class name for torch.package compatibility.

    When models are loaded from .nequip.zip packages via torch.package, classes
    may exist in a different namespace. The standard isinstance check fails, but
    the class name still matches.
    """
    if isinstance(obj, cls):
        return True
    return type(obj).__name__ == cls.__name__


class FreezeLayersCallback(lightning.Callback):
    """Freeze the first N interaction layers and all embedding layers.

    Everything before the last ``(total - num_frozen_interaction_layers)``
    interaction layers is frozen. The readout, PerTypeScaleShift, and
    unfrozen interaction layers remain trainable.

    Interaction layers are identified by type (:class:`~nequip.nn.ConvNetLayer`),
    making this robust to different model naming conventions (e.g. models loaded
    from ``.nequip.zip`` packages vs. built from config).

    Compatible with ``EMALightningModule``, ``StochasticWeightAveraging``,
    and ``torch.compile`` — see project docs for details.

    Args:
        num_frozen_interaction_layers: Number of interaction layers to freeze,
            counting from the input end of the network. Must be less than the
            total number of interaction layers.
    """

    def __init__(self, num_frozen_interaction_layers: int):
        super().__init__()
        if num_frozen_interaction_layers < 0:
            raise ValueError(
                f"num_frozen_interaction_layers must be >= 0, got {num_frozen_interaction_layers}"
            )
        self.num_frozen_interaction_layers = num_frozen_interaction_layers

    def on_fit_start(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule):
        model = pl_module.model

        # Unwrap to find SequentialGraphNetwork
        seq_net = self._find_sequential(model)
        if seq_net is None:
            raise RuntimeError(
                "Could not find SequentialGraphNetwork in model. "
                "FreezeLayersCallback requires a NequIP-style model."
            )

        # Identify ConvNetLayer modules and their position in the sequence
        conv_indices = []
        children_list = list(seq_net.named_children())
        for i, (name, child) in enumerate(children_list):
            if _is_instance_by_name(child, ConvNetLayer):
                conv_indices.append(i)

        total_conv = len(conv_indices)
        if total_conv == 0:
            raise RuntimeError(
                "No ConvNetLayer modules found in SequentialGraphNetwork. "
                "FreezeLayersCallback only supports NequIP models with ConvNetLayer."
            )
        if self.num_frozen_interaction_layers >= total_conv:
            raise ValueError(
                f"num_frozen_interaction_layers ({self.num_frozen_interaction_layers}) "
                f"must be less than total interaction layers ({total_conv}). "
                f"At least one layer must remain trainable."
            )

        if self.num_frozen_interaction_layers == 0:
            logger.info("FreezeLayersCallback: num_frozen_interaction_layers=0, nothing to freeze.")
            return

        # Freeze everything up to and including the last frozen ConvNetLayer
        last_frozen_idx = conv_indices[self.num_frozen_interaction_layers - 1]

        frozen_count = 0
        trainable_count = 0
        log_lines = []

        for i, (name, child) in enumerate(children_list):
            n_params = sum(p.numel() for p in child.parameters())
            if i <= last_frozen_idx:
                for p in child.parameters():
                    p.requires_grad_(False)
                frozen_count += n_params
                log_lines.append(f"  FROZEN    {name:30s} {type(child).__name__:30s} {n_params:>8,} params")
            else:
                trainable_count += n_params
                log_lines.append(f"  TRAINABLE {name:30s} {type(child).__name__:30s} {n_params:>8,} params")

        # Also count ForceStressOutput params (usually 0, but be explicit)
        fso_params = sum(
            p.numel() for n, p in model.named_parameters()
            if not any(n.startswith(cn) for cn, _ in children_list)
        )
        if fso_params > 0:
            trainable_count += fso_params

        header = (
            f"FreezeLayersCallback: froze {self.num_frozen_interaction_layers}/{total_conv} "
            f"interaction layers ({frozen_count:,} frozen, {trainable_count:,} trainable params)"
        )
        logger.info(header)
        for line in log_lines:
            logger.info(line)

    def _find_sequential(self, module):
        """Find SequentialGraphNetwork by walking the module tree.

        Uses name-based matching as fallback for torch.package compatibility.
        """
        if _is_instance_by_name(module, SequentialGraphNetwork):
            return module
        for child in module.children():
            result = self._find_sequential(child)
            if result is not None:
                return result
        return None
