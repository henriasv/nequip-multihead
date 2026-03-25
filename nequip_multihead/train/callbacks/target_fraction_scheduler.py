# Target fraction loss coefficient schedulers for NequIP.
# Adjusts loss coefficients to maintain target loss/gradient fractions throughout training.
import torch
import lightning
from lightning.pytorch.callbacks import Callback
from nequip.train import NequIPLightningModule
from typing import Dict, Optional


class TargetFractionLossScheduler(Callback):
    """Adjust loss coefficients to maintain target loss fractions.

    Instead of setting fixed coefficients, this scheduler sets **target fractions**
    of the total weighted loss that each component should contribute. At each update,
    it reads the current raw (unweighted) loss values and computes coefficients such
    that::

        coeff_i * raw_loss_i / sum(coeff_j * raw_loss_j) ≈ target_fraction_i

    Example::

        callbacks:
          - _target_: nequip_multihead.train.callbacks.TargetFractionLossScheduler
            target_fractions:
              forces_mse: 0.90
              per_atom_energy_mse: 0.09
              stress_mse: 0.01
            interval: epoch
            frequency: 5
            ema_decay: 0.9

    Args:
        target_fractions: dict mapping loss metric names to target fractions (must sum to ~1)
        interval: ``"batch"`` or ``"epoch"``
        frequency: number of intervals between coefficient updates
        ema_decay: EMA decay for smoothing (0 = no smoothing, 0.99 = very smooth)
        eps: small value to avoid division by zero
    """

    def __init__(
        self,
        target_fractions: Dict[str, float],
        interval: str = "epoch",
        frequency: int = 5,
        ema_decay: float = 0.9,
        eps: float = 1e-12,
    ):
        assert interval in ["batch", "epoch"]
        assert frequency >= 1

        frac_sum = sum(target_fractions.values())
        assert abs(frac_sum - 1.0) < 0.05, (
            f"target_fractions must sum to ~1.0, got {frac_sum}"
        )
        target_fractions = {k: v / frac_sum for k, v in target_fractions.items()}
        assert all(v > 0 for v in target_fractions.values()), (
            "all target_fractions must be positive"
        )

        self.target_fractions = target_fractions
        self.interval = interval
        self.frequency = frequency
        self.ema_decay = ema_decay
        self.eps = eps

        self.smoothed_losses: Optional[Dict[str, float]] = None

    def _update_coeffs(
        self,
        raw_losses: Dict[str, float],
        trainer: lightning.Trainer,
        pl_module: NequIPLightningModule,
    ):
        if self.interval == "epoch":
            step = trainer.current_epoch
        else:
            step = trainer.global_step

        if step % self.frequency != 0:
            return

        current = {
            k: raw_losses[k]
            for k in self.target_fractions
            if raw_losses.get(k) is not None
        }
        if len(current) != len(self.target_fractions):
            return

        if self.smoothed_losses is None:
            self.smoothed_losses = dict(current)
        else:
            for k in current:
                self.smoothed_losses[k] = (
                    self.ema_decay * self.smoothed_losses[k]
                    + (1 - self.ema_decay) * current[k]
                )

        new_coeffs = {}
        for k in self.target_fractions:
            raw = max(self.smoothed_losses[k], self.eps)
            new_coeffs[k] = self.target_fractions[k] / raw

        total = sum(new_coeffs.values())
        new_coeffs = {k: v / total for k, v in new_coeffs.items()}

        pl_module.loss.set_coeffs(new_coeffs)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if trainer.global_step == 0:
            return
        if self.interval == "batch":
            self._update_coeffs(pl_module.loss.metrics_values_step, trainer, pl_module)

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == 0:
            return
        if self.interval == "epoch":
            self._update_coeffs(pl_module.loss.metrics_values_epoch, trainer, pl_module)

    def state_dict(self):
        return {
            "target_fractions": self.target_fractions,
            "interval": self.interval,
            "frequency": self.frequency,
            "ema_decay": self.ema_decay,
            "eps": self.eps,
            "smoothed_losses": self.smoothed_losses,
        }

    def load_state_dict(self, state_dict):
        self.target_fractions = state_dict["target_fractions"]
        self.interval = state_dict["interval"]
        self.frequency = state_dict["frequency"]
        self.ema_decay = state_dict["ema_decay"]
        self.eps = state_dict["eps"]
        self.smoothed_losses = state_dict["smoothed_losses"]


class GradientNormFractionScheduler(Callback):
    """Adjust loss coefficients to maintain target gradient-norm fractions.

    Every ``frequency`` steps, computes the gradient norm that each loss component
    contributes to the total parameter update. Uses ``torch.autograd.grad`` with
    ``retain_graph=True`` on individual loss components before the main backward pass.

    The computation happens in ``on_before_backward`` when the computation graph
    is still alive, allowing separate gradient computation for each loss component.

    **Cost:** N calls to ``torch.autograd.grad`` every ``frequency`` steps.
    Each call traverses the backward graph but does NOT accumulate gradients
    (uses ``create_graph=False``). For ``frequency=100`` and N=3, overhead is ~3%.

    Example::

        callbacks:
          - _target_: nequip_multihead.train.callbacks.GradientNormFractionScheduler
            target_fractions:
              forces_mse: 0.90
              per_atom_energy_mse: 0.09
              stress_mse: 0.01
            frequency: 100
            ema_decay: 0.95

    Args:
        target_fractions: dict mapping loss metric names to target gradient fractions (final values)
        initial_target_fractions: optional dict of starting fractions (default: same as final).
            When set, fractions linearly interpolate from initial to final between
            ``ramp_start_epoch`` and ``ramp_end_epoch``.
        ramp_start_epoch: epoch to start ramping (default: 0)
        ramp_end_epoch: epoch to reach final targets (default: 100)
        frequency: number of training steps between measurements
        ema_decay: EMA decay for smoothing gradient norm estimates
        eps: small value to avoid division by zero

    Example with ramping::

        callbacks:
          - _target_: nequip_multihead.train.callbacks.GradientNormFractionScheduler
            initial_target_fractions:
              forces_mse: 0.80
              per_atom_energy_mse: 0.10
              stress_mse: 0.10
            target_fractions:
              forces_mse: 0.40
              per_atom_energy_mse: 0.40
              stress_mse: 0.20
            ramp_start_epoch: 50
            ramp_end_epoch: 200
            frequency: 50
    """

    def __init__(
        self,
        target_fractions: Dict[str, float],
        initial_target_fractions: Optional[Dict[str, float]] = None,
        ramp_start_epoch: int = 0,
        ramp_end_epoch: int = 100,
        frequency: int = 100,
        ema_decay: float = 0.95,
        eps: float = 1e-12,
    ):
        assert frequency >= 1

        def _normalize(fracs):
            frac_sum = sum(fracs.values())
            assert abs(frac_sum - 1.0) < 0.05, (
                f"target_fractions must sum to ~1.0, got {frac_sum}"
            )
            fracs = {k: v / frac_sum for k, v in fracs.items()}
            assert all(v > 0 for v in fracs.values()), (
                "all target_fractions must be positive"
            )
            return fracs

        self.target_fractions = _normalize(target_fractions)

        if initial_target_fractions is not None:
            assert set(initial_target_fractions.keys()) == set(target_fractions.keys()), (
                "initial_target_fractions must have same keys as target_fractions"
            )
            self.initial_target_fractions = _normalize(initial_target_fractions)
        else:
            self.initial_target_fractions = None

        self.ramp_start_epoch = ramp_start_epoch
        self.ramp_end_epoch = ramp_end_epoch
        assert ramp_end_epoch > ramp_start_epoch, (
            "ramp_end_epoch must be > ramp_start_epoch"
        )

        self.frequency = frequency
        self.ema_decay = ema_decay
        self.eps = eps

        self.smoothed_gnorms: Optional[Dict[str, float]] = None

    def _get_current_targets(self, epoch: int) -> Dict[str, float]:
        """Get interpolated target fractions for the current epoch."""
        if self.initial_target_fractions is None:
            return self.target_fractions

        if epoch <= self.ramp_start_epoch:
            return self.initial_target_fractions
        elif epoch >= self.ramp_end_epoch:
            return self.target_fractions
        else:
            # Linear interpolation
            t = (epoch - self.ramp_start_epoch) / (
                self.ramp_end_epoch - self.ramp_start_epoch
            )
            return {
                k: (1 - t) * self.initial_target_fractions[k]
                + t * self.target_fractions[k]
                for k in self.target_fractions
            }

    def on_train_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: NequIPLightningModule,
        outputs,
        batch,
        batch_idx: int,
    ):
        """Measure per-component gradient norms after the training step.

        Runs an **eager** forward pass (bypassing torch.compile) on the
        current batch, computes individual loss components, and measures
        gradient norms via ``torch.autograd.grad``. This avoids conflicts
        with torch.compile's inplace tensor optimizations.

        Cost: one extra eager forward + N backward passes every
        ``frequency`` steps. The compiled path handles 99% of steps.
        """
        if trainer.global_step < 1:
            return
        if trainer.global_step % self.frequency != 0:
            return

        params = [p for p in pl_module.parameters() if p.requires_grad]
        if not params:
            return

        loss_manager = pl_module.loss
        assert hasattr(loss_manager, "metrics_tensors_step"), (
            "GradientNormFractionScheduler requires nequip with metrics_tensors_step "
            "support in MetricsManager. Update your nequip installation."
        )

        # Run an eager forward pass (no compile, no inplace ops) so that
        # autograd.grad with retain_graph=True works correctly.
        # Import sub-modules before any use of torch.Tensor to avoid
        # shadowing issues in some Python/PyTorch versions.
        import torch._functorch.config as functorch_config
        import torch._dynamo as _dynamo

        @_dynamo.disable
        def _eager_forward(model, data):
            return model(data)

        @_dynamo.disable
        def _eager_loss(loss_mgr, out, tgt):
            return loss_mgr(out, tgt, prefix="_gnorm_")

        old_donated = functorch_config.donated_buffer
        functorch_config.donated_buffer = False
        try:
            # Copy batch to avoid issues from compiled training step
            # having mutated the data dict
            batch_copy = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            target = pl_module.process_target(batch_copy, batch_idx)
            output = _eager_forward(pl_module, batch_copy)
            _eager_loss(loss_manager, output, target)

            gnorms = {}
            for metric_name in self.target_fractions:
                tensor = loss_manager.metrics_tensors_step.get(metric_name)
                if tensor is None or not isinstance(tensor, torch.Tensor):
                    continue
                if not tensor.requires_grad:
                    continue

                grads = torch.autograd.grad(
                    tensor,
                    params,
                    retain_graph=True,
                    allow_unused=True,
                )
                gnorm_sq = 0.0
                for g in grads:
                    if g is not None:
                        gnorm_sq += g.detach().norm().item() ** 2
                gnorms[metric_name] = gnorm_sq ** 0.5
        finally:
            functorch_config.donated_buffer = old_donated

        self._update_from_gnorms(gnorms, trainer, pl_module)

    def _update_from_gnorms(
        self,
        gnorms: Dict[str, float],
        trainer: "lightning.Trainer",
        pl_module: NequIPLightningModule,
    ):
        """Update coefficients based on measured gradient norms.

        Handles partial measurements: only metrics that produced valid
        gradients on this batch are updated. The EMA for unmeasured
        metrics carries forward from previous measurements. Coefficient
        updates only happen once all metrics have been initialized.

        Uses time-varying target fractions if ``initial_target_fractions``
        was set — linearly interpolates between initial and final targets
        based on the current epoch.
        """
        if not gnorms:
            return

        # EMA smooth — initialize or update
        if self.smoothed_gnorms is None:
            # First measurement: need all metrics present to initialize
            if len(gnorms) == len(self.target_fractions):
                self.smoothed_gnorms = dict(gnorms)
            else:
                return  # wait until we've seen all metrics
        else:
            # Update only the metrics that were measured this step
            for k in gnorms:
                self.smoothed_gnorms[k] = (
                    self.ema_decay * self.smoothed_gnorms[k]
                    + (1 - self.ema_decay) * gnorms[k]
                )

        # Only update coefficients if all metrics have smoothed values
        if len(self.smoothed_gnorms) != len(self.target_fractions):
            return

        # Get current targets (possibly interpolated)
        current_targets = self._get_current_targets(trainer.current_epoch)

        # Compute coefficients: coeff_i = target_i / smoothed_gnorm_i
        new_coeffs = {}
        for k in current_targets:
            g = max(self.smoothed_gnorms[k], self.eps)
            new_coeffs[k] = current_targets[k] / g

        # Normalize to sum to 1
        total = sum(new_coeffs.values())
        new_coeffs = {k: v / total for k, v in new_coeffs.items()}

        pl_module.loss.set_coeffs(new_coeffs)

    def state_dict(self):
        return {
            "target_fractions": self.target_fractions,
            "initial_target_fractions": self.initial_target_fractions,
            "ramp_start_epoch": self.ramp_start_epoch,
            "ramp_end_epoch": self.ramp_end_epoch,
            "frequency": self.frequency,
            "ema_decay": self.ema_decay,
            "eps": self.eps,
            "smoothed_gnorms": self.smoothed_gnorms,
        }

    def load_state_dict(self, state_dict):
        self.target_fractions = state_dict["target_fractions"]
        self.initial_target_fractions = state_dict.get("initial_target_fractions")
        self.ramp_start_epoch = state_dict.get("ramp_start_epoch", 0)
        self.ramp_end_epoch = state_dict.get("ramp_end_epoch", 100)
        self.frequency = state_dict["frequency"]
        self.ema_decay = state_dict["ema_decay"]
        self.eps = state_dict["eps"]
        self.smoothed_gnorms = state_dict["smoothed_gnorms"]
