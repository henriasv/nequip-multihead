# Training Schedule and SWA

Tools for controlling loss coefficients and weight averaging during training.

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

## Stochastic Weight Averaging (SWA)

SWA stabilizes weights in late training by averaging model parameters over multiple epochs at a low learning rate. This produces smoother energy surfaces and more robust MD.  Inspired by MACE's "Stage Two" training.

### Typical two-phase schedule

1. **Phase 1**: Cosine annealing (e.g. 300 epochs, lr 0.01 → 0.001)
2. **Phase 2**: SWA at constant lr (e.g. 100 epochs at 0.001), averaging weights

```yaml
trainer:
  max_epochs: 400
  callbacks:
    - _target_: lightning.pytorch.callbacks.ModelCheckpoint
      dirpath: ${hydra:runtime.output_dir}
      filename: best
      save_last: true
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
      eta_min: 0.001    # match swa_lr for smooth transition
    interval: epoch
```

At epoch 300, the callback:
- Replaces the LR scheduler with a constant `swa_lr`
- Takes the first weight snapshot
- On each subsequent epoch, folds the new weights into a running average

At training end, the SWA-averaged model is saved as **`swa_last.ckpt`** alongside the normal `last.ckpt`.

### EMA + SWA

SWA is compatible with `EMALightningModule`. They're complementary:

- **EMA** smooths step-level noise (updates every optimizer step)
- **SWA** averages epoch-level snapshots (updates every epoch during SWA phase)

This is the same pattern MACE uses. After training you get two checkpoints:

| Checkpoint | Contains |
|------------|----------|
| `last.ckpt` | EMA weights (smoothed training model) |
| `swa_last.ckpt` | SWA-averaged weights (averaged over SWA phase) |

### Deploying the SWA model

Compile from `swa_last.ckpt` just like from `last.ckpt`:

```bash
# Single head
nequip-compile swa_last.ckpt model_swa.nequip.pt2 \
  --mode aotinductor --device cuda --target ase \
  --modifiers extract_head head_name=baseline

# Summed heads (delta-learning)
nequip-compile swa_last.ckpt model_swa.nequip.pt2 \
  --mode aotinductor --device cuda --target ase \
  --modifiers extract_summed_heads head_names=baseline+delta
```

### Optional: MACE-style loss reweighting

MACE bumps energy weight during Stage Two.  The `swa_loss_coeffs` parameter applies new loss coefficients when the SWA phase starts:

```yaml
    - _target_: nequip_multihead.train.callbacks.StochasticWeightAveraging
      swa_start_epoch: 300
      swa_lr: 0.001
      swa_loss_coeffs:
        per_atom_energy_mse: 1000.0
        forces_mse: 100.0
        stress_mse: 10.0
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `swa_start_epoch` | *(required)* | Epoch to begin SWA (0-based) |
| `swa_lr` | *(required)* | Constant LR during SWA phase |
| `swa_loss_coeffs` | `null` | Optional loss coefficient overrides |
| `annealing_epochs` | `1` | Epochs to anneal LR to `swa_lr` |
| `save_snapshots` | `false` | Save per-epoch weight snapshots for debugging |

## Dataloader shuffling

```{warning}
Always set `shuffle: true` in the dataloader config. PyTorch's `DataLoader` defaults to sequential ordering, which means every epoch sees frames in the same order. This biases SWA averages and can slow convergence in general.
```

```yaml
dataloader:
  _target_: torch.utils.data.DataLoader
  batch_size: 5
  shuffle: true
```
