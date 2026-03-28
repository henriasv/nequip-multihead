"""Stochastic Weight Averaging (SWA) callback for late-training stabilization.

Maintains an equal-weight running average of model parameters during the SWA
phase, producing smoother energy surfaces and more robust models.  Inspired
by MACE's "Stage Two" training.

Compatible with ``EMALightningModule``: EMA smooths step-level noise while
SWA averages epoch-level snapshots.  They operate on different timescales
and are complementary (this is the same pattern MACE uses).

Typical two-phase training schedule::

    Phase 1 (epochs 0–299): CosineAnnealingLR  lr 0.01 → 0.001
    Phase 2 (epochs 300–399): SWA at constant lr 0.001, weights averaged

Example config::

    trainer:
      max_epochs: 400
      callbacks:
        - _target_: nequip_multihead.train.callbacks.StochasticWeightAveraging
          swa_start_epoch: 300
          swa_lr: 0.001

    training_module:
      _target_: nequip.train.EMALightningModule
      optimizer:
        _target_: torch.optim.Adam
        lr: 0.01
      lr_scheduler:
        scheduler:
          _target_: torch.optim.lr_scheduler.CosineAnnealingLR
          T_max: 300
          eta_min: 0.001
        interval: epoch
"""

import logging
from typing import Dict, List, Optional

import torch
import lightning
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import LRSchedulerConfig
from torch.optim.swa_utils import SWALR

logger = logging.getLogger(__name__)


class StochasticWeightAveraging(Callback):
    """Average model weights during late training for improved stability.

    At ``swa_start_epoch``, the callback:

    1. Takes a snapshot of the current model weights.
    2. Replaces the LR scheduler with
       :class:`~torch.optim.swa_utils.SWALR` at a constant ``swa_lr``.
    3. Optionally overrides loss coefficients (MACE-style).

    Each subsequent epoch, the model weights are folded into an
    equal-weight running average.  When training ends, the SWA-averaged
    weights are compiled into the model via ``nequip-compile``.

    Works with both ``NequIPLightningModule`` and ``EMALightningModule``.
    When EMA is active, SWA averages the *training* weights (not the EMA
    weights) at epoch boundaries — the same pattern MACE uses.

    The SWA model is accessed at deployment time via::

        nequip-compile last.ckpt model.pt2 --mode aotinductor \\
            --device cuda --target ase \\
            --modifiers load_swa_weights

    Args:
        swa_start_epoch: Epoch at which SWA begins (0-based).
        swa_lr: Constant learning rate during SWA phase.
        swa_loss_coeffs: Optional dict of loss coefficient overrides
            applied at ``swa_start_epoch`` via
            ``pl_module.loss.set_coeffs()``.
        annealing_epochs: Number of epochs to anneal LR from current
            value to ``swa_lr``.  Default ``1`` means immediate switch.

    Example with MACE-style loss reweighting::

        callbacks:
          - _target_: nequip_multihead.train.callbacks.StochasticWeightAveraging
            swa_start_epoch: 300
            swa_lr: 0.001
            swa_loss_coeffs:
              per_atom_energy_mse: 1000.0
              forces_mse: 100.0
              stress_mse: 10.0
    """

    def __init__(
        self,
        swa_start_epoch: int,
        swa_lr: float,
        swa_loss_coeffs: Optional[Dict[str, float]] = None,
        annealing_epochs: int = 1,
    ):
        assert swa_start_epoch >= 1, "swa_start_epoch must be >= 1"
        assert swa_lr > 0, "swa_lr must be positive"
        assert annealing_epochs >= 1, "annealing_epochs must be >= 1"

        self.swa_start_epoch = swa_start_epoch
        self.swa_lr = swa_lr
        self.swa_loss_coeffs = swa_loss_coeffs
        self.annealing_epochs = annealing_epochs

        # Internal state (persisted via state_dict)
        self._swa_started: bool = False
        self._n_averaged: int = 0
        self._averaged_params: Optional[List[torch.Tensor]] = None

    def on_fit_start(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
    ):
        if len(trainer.optimizers) != 1:
            raise RuntimeError(
                "StochasticWeightAveraging requires exactly one optimizer."
            )
        if (
            trainer.max_epochs is not None
            and trainer.max_epochs != -1
            and self.swa_start_epoch >= trainer.max_epochs
        ):
            raise RuntimeError(
                f"swa_start_epoch ({self.swa_start_epoch}) must be < "
                f"max_epochs ({trainer.max_epochs})"
            )

        # If restarting with SWA already active, the original LR scheduler
        # was replaced in the previous run.  Clear Lightning's scheduler
        # configs so it doesn't try to restore state into the wrong type.
        if self._swa_started and trainer.lr_scheduler_configs:
            trainer.lr_scheduler_configs.clear()

    def on_train_epoch_start(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
    ):
        if trainer.current_epoch < self.swa_start_epoch:
            return

        if not self._swa_started:
            # ---- First SWA epoch: initialize ----
            self._swa_started = True
            self._averaged_params = [
                p.detach().clone() for p in pl_module.model.parameters()
            ]
            self._n_averaged = 1

            # Replace LR scheduler with SWALR
            optimizer = trainer.optimizers[0]
            swa_scheduler = SWALR(
                optimizer,
                swa_lr=self.swa_lr,
                anneal_epochs=self.annealing_epochs,
                anneal_strategy="linear",
            )
            new_cfg = LRSchedulerConfig(swa_scheduler)
            if trainer.lr_scheduler_configs:
                trainer.lr_scheduler_configs[0] = new_cfg
            else:
                trainer.lr_scheduler_configs.append(new_cfg)

            # Optional loss coefficient override
            if self.swa_loss_coeffs is not None:
                pl_module.loss.set_coeffs(self.swa_loss_coeffs)

            logger.info(
                f"SWA phase started at epoch {trainer.current_epoch} "
                f"(lr={self.swa_lr}, annealing_epochs={self.annealing_epochs})"
            )
        else:
            # ---- Subsequent SWA epochs: update running average ----
            with torch.no_grad():
                for avg_p, model_p in zip(
                    self._averaged_params, pl_module.model.parameters()
                ):
                    avg_p.add_(
                        (model_p.detach() - avg_p) / (self._n_averaged + 1)
                    )
            self._n_averaged += 1

    def on_train_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
    ):
        if self._averaged_params is None or self._n_averaged == 0:
            return

        # Save a separate checkpoint with SWA weights swapped into the
        # model, alongside the normal last.ckpt (which keeps training/EMA
        # weights).  This mirrors MACE's approach of saving both
        # model.pt and model_stagetwo.pt.
        swa_path = _swa_checkpoint_path(trainer)
        logger.info(
            f"SWA: saving averaged weights ({self._n_averaged} snapshots) "
            f"to {swa_path}"
        )

        # Temporarily swap SWA weights into the model
        original_params = [
            p.detach().clone() for p in pl_module.model.parameters()
        ]
        with torch.no_grad():
            for avg_p, model_p in zip(
                self._averaged_params, pl_module.model.parameters()
            ):
                model_p.copy_(avg_p.to(model_p.device))

        trainer.save_checkpoint(str(swa_path))

        # Restore original weights
        with torch.no_grad():
            for orig_p, model_p in zip(
                original_params, pl_module.model.parameters()
            ):
                model_p.copy_(orig_p)

        # Free memory
        self._averaged_params = None

    def state_dict(self):
        return {
            "swa_start_epoch": self.swa_start_epoch,
            "swa_lr": self.swa_lr,
            "swa_loss_coeffs": self.swa_loss_coeffs,
            "annealing_epochs": self.annealing_epochs,
            "swa_started": self._swa_started,
            "n_averaged": self._n_averaged,
            "averaged_params": (
                [p.cpu() for p in self._averaged_params]
                if self._averaged_params is not None
                else None
            ),
        }

    def load_state_dict(self, state_dict):
        self._swa_started = state_dict["swa_started"]
        self._n_averaged = state_dict["n_averaged"]
        self._averaged_params = state_dict.get("averaged_params")


def _swa_checkpoint_path(trainer: lightning.Trainer) -> str:
    """Derive the SWA checkpoint path from the trainer's output directory."""
    import pathlib

    # Use the same directory as last.ckpt
    for cb in trainer.checkpoint_callbacks:
        if hasattr(cb, "dirpath") and cb.dirpath:
            return str(pathlib.Path(cb.dirpath) / "swa_last.ckpt")
    # Fallback: trainer's default_root_dir
    return str(pathlib.Path(trainer.default_root_dir) / "swa_last.ckpt")
