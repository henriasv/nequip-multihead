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
        target_fractions: dict mapping loss metric names to target gradient fractions
        frequency: number of training steps between measurements
        ema_decay: EMA decay for smoothing gradient norm estimates
        eps: small value to avoid division by zero
    """

    def __init__(
        self,
        target_fractions: Dict[str, float],
        frequency: int = 100,
        ema_decay: float = 0.95,
        eps: float = 1e-12,
    ):
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
        self.frequency = frequency
        self.ema_decay = ema_decay
        self.eps = eps

        self.smoothed_gnorms: Optional[Dict[str, float]] = None

    def on_before_backward(
        self,
        trainer: lightning.Trainer,
        pl_module: NequIPLightningModule,
        loss: torch.Tensor,
    ):
        """Measure per-component gradient norms before the main backward pass.

        Uses ``torch.autograd.grad`` with ``retain_graph=True`` on each cached
        loss tensor from ``MetricsManager.metrics_tensors_step``. This gives
        the true gradient norm contribution from each loss component.
        """
        if trainer.global_step < 1:
            return
        if trainer.global_step % self.frequency != 0:
            return

        # Get trainable parameters
        params = [p for p in pl_module.parameters() if p.requires_grad]
        if not params:
            return

        # Access cached live loss tensors from the MetricsManager
        loss_manager = pl_module.loss
        assert hasattr(loss_manager, "metrics_tensors_step"), (
            "GradientNormFractionScheduler requires nequip with metrics_tensors_step "
            "support in MetricsManager. Update your nequip installation."
        )

        gnorms = {}
        for metric_name in self.target_fractions:
            tensor = loss_manager.metrics_tensors_step.get(metric_name)
            assert tensor is not None and isinstance(tensor, torch.Tensor), (
                f"Loss component '{metric_name}' not found in metrics_tensors_step. "
                f"Available: {list(loss_manager.metrics_tensors_step.keys())}"
            )
            assert tensor.requires_grad, (
                f"Loss tensor for '{metric_name}' does not require grad. "
                f"Cannot compute gradient norms."
            )

            # Compute gradient norm for this loss component.
            # Temporarily disable donated_buffer optimization which is
            # incompatible with retain_graph=True under torch.compile.
            import torch._functorch.config as functorch_config

            old_donated = functorch_config.donated_buffer
            functorch_config.donated_buffer = False
            try:
                grads = torch.autograd.grad(
                    tensor,
                    params,
                    retain_graph=True,
                    allow_unused=True,
                )
            finally:
                functorch_config.donated_buffer = old_donated
            # Compute total gradient norm across all parameters
            gnorm_sq = 0.0
            for g in grads:
                if g is not None:
                    gnorm_sq += g.detach().norm().item() ** 2
            gnorms[metric_name] = gnorm_sq ** 0.5

        self._update_from_gnorms(gnorms, pl_module)

    def _update_from_gnorms(
        self,
        gnorms: Dict[str, float],
        pl_module: NequIPLightningModule,
    ):
        """Update coefficients based on measured gradient norms."""
        if len(gnorms) != len(self.target_fractions):
            return

        # EMA smooth
        if self.smoothed_gnorms is None:
            self.smoothed_gnorms = dict(gnorms)
        else:
            for k in gnorms:
                self.smoothed_gnorms[k] = (
                    self.ema_decay * self.smoothed_gnorms[k]
                    + (1 - self.ema_decay) * gnorms[k]
                )

        # Compute coefficients: coeff_i = target_i / smoothed_gnorm_i
        new_coeffs = {}
        for k in self.target_fractions:
            g = max(self.smoothed_gnorms[k], self.eps)
            new_coeffs[k] = self.target_fractions[k] / g

        # Normalize to sum to 1
        total = sum(new_coeffs.values())
        new_coeffs = {k: v / total for k, v in new_coeffs.items()}

        pl_module.loss.set_coeffs(new_coeffs)

    def state_dict(self):
        return {
            "target_fractions": self.target_fractions,
            "frequency": self.frequency,
            "ema_decay": self.ema_decay,
            "eps": self.eps,
            "smoothed_gnorms": self.smoothed_gnorms,
        }

    def load_state_dict(self, state_dict):
        self.target_fractions = state_dict["target_fractions"]
        self.frequency = state_dict["frequency"]
        self.ema_decay = state_dict["ema_decay"]
        self.eps = state_dict["eps"]
        self.smoothed_gnorms = state_dict["smoothed_gnorms"]
