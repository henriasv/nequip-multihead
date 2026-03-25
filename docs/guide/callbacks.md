# Loss Coefficient Scheduling

Multi-head training benefits from dynamic loss coefficient scheduling — adjusting the relative weight of energy, force, and stress loss components during training. `nequip-multihead` provides two callback schedulers.

## GradientNormFractionScheduler

Adjusts loss coefficients so that each loss component contributes a target fraction of the total gradient norm. This is the recommended scheduler for multi-head training.

### How it works

Every `frequency` steps, the callback measures the gradient norm that each loss component would contribute to the parameter update (via `torch.autograd.grad` with `retain_graph=True`). It then adjusts coefficients so that the measured gradient-norm fractions match the targets.

For example, with `target_fractions: {forces_mse: 0.9, per_atom_energy_mse: 0.09, stress_mse: 0.01}`, the scheduler ensures that ~90% of the parameter update magnitude comes from the force loss, regardless of the absolute scale of each component.

### Basic usage

```yaml
trainer:
  callbacks:
    - _target_: nequip_multihead.train.callbacks.GradientNormFractionScheduler
      target_fractions:
        forces_mse: 0.90
        per_atom_energy_mse: 0.09
        stress_mse: 0.01
      frequency: 50        # measure every 50 steps
      ema_decay: 0.95       # smooth gradient norm estimates
```

The metric names (`forces_mse`, `per_atom_energy_mse`, `stress_mse`) must match the names in your loss configuration.

### Time-varying targets (ramping)

Start force-heavy and ramp toward balanced targets:

```yaml
trainer:
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
      ema_decay: 0.95
```

This linearly interpolates from `initial_target_fractions` to `target_fractions` between epochs 50 and 200. Before epoch 50, the initial targets are used. After epoch 200, the final targets are used.

### NaN-safe behavior

With `ConcatDataset` mixing force-supervised and energy-only data, some batches may produce NaN for certain loss components (e.g. `stress_mse` on energy-only frames). The scheduler gracefully skips metrics without valid gradients on those measurement steps. The EMA-smoothed estimate from previous valid measurements carries forward.

### Compatibility with `compile_mode: compile`

The scheduler is compatible with `torch.compile`. It temporarily disables the `donated_buffer` optimization during gradient norm measurement (which requires `retain_graph=True`). This has no performance impact since measurement only happens every `frequency` steps.

## TargetFractionLossScheduler

A simpler alternative that adjusts coefficients based on the loss values themselves (not gradient norms). Less accurate but cheaper — no extra backward passes.

```yaml
trainer:
  callbacks:
    - _target_: nequip_multihead.train.callbacks.TargetFractionLossScheduler
      target_fractions:
        forces_mse: 0.90
        per_atom_energy_mse: 0.09
        stress_mse: 0.01
      frequency: 50
      interval: batch       # or "epoch"
      ema_decay: 0.95
```

### When to use which

| Scheduler | Accuracy | Cost | Recommendation |
|-----------|----------|------|----------------|
| GradientNormFractionScheduler | High — measures actual gradient contributions | N extra backward passes every `frequency` steps | Use for production training |
| TargetFractionLossScheduler | Approximate — uses loss values as proxy | No extra backward passes | Use for quick experiments |

```{note}
Both schedulers require the ``metrics_tensors_step`` feature in NequIP's ``MetricsManager``, which is included in our NequIP fork (``henriasv/nequip@feature/parameterized-modifiers``).
```
