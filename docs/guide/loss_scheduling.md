# Dynamic Loss Scheduling

Adjust the relative weight of force, energy, and stress losses during training based on their actual gradient contributions.

## Why schedule loss coefficients?

Static loss coefficients (e.g. `forces: 10.0, energy: 1.0`) don't account for how much each component actually influences the model parameters. A force coefficient of 10.0 might produce 99% of the gradient update in early training but only 50% later as forces converge. Dynamic scheduling maintains target gradient fractions throughout training.

## GradientNormFractionScheduler

The recommended scheduler. Every `frequency` steps, it measures the gradient norm each loss component contributes, then adjusts coefficients to match target fractions.

```yaml
trainer:
  callbacks:
    - _target_: nequip_multihead.train.callbacks.GradientNormFractionScheduler
      target_fractions:
        forces_mse: 0.90
        per_atom_energy_mse: 0.09
        stress_mse: 0.01
      frequency: 50
      ema_decay: 0.95
```

This ensures ~90% of the parameter update magnitude comes from forces, regardless of absolute loss scales.

### Ramping: start force-heavy, end balanced

For delta-learning, starting with high force weight and ramping toward balanced targets often works well — forces need to converge first for the energy landscape to be meaningful:

```yaml
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
```

Before epoch 50: forces get 80%. Between 50–200: linear ramp. After 200: balanced 40/40/20.

### Cost

One extra eager forward + N backward passes every `frequency` steps (~3% overhead for `frequency=50`, N=3 components). The compiled training path handles 97% of steps at full speed.

## TargetFractionLossScheduler

A simpler alternative that uses loss values (not gradient norms) as a proxy. No extra backward passes — just arithmetic on already-computed losses.

```yaml
    - _target_: nequip_multihead.train.callbacks.TargetFractionLossScheduler
      target_fractions:
        forces_mse: 0.90
        per_atom_energy_mse: 0.09
        stress_mse: 0.01
      frequency: 50
      interval: batch
      ema_decay: 0.95
```

Less accurate than gradient-norm scheduling but cheaper. Good for quick experiments.

## When to use which

- **GradientNormFractionScheduler**: production training, multi-head with diverse supervision
- **TargetFractionLossScheduler**: quick experiments, single-head training
- **No scheduler**: if static coefficients work well for your system

```{warning}
The gradient norm measurement does not account for `accumulate_grad_batches`. The relative fractions are still correct, but absolute norms reflect a single batch.
```
