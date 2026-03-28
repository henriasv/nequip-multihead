# Troubleshooting

## NaN loss from the first step

**Problem**: All losses are NaN immediately.

**Cause**: `per_type_energy_scales` is NaN because `${training_data_stats:forces_rms}` included NaN forces from energy-only data.

**Fix**: Use `MultiHeadDataStatisticsManager`:
```yaml
stats_manager:
  _target_: nequip_multihead.data.MultiHeadDataStatisticsManager
  type_names: ${model_type_names}
```
Or specify energy scales explicitly: `per_type_energy_scales: {baseline: 10.0, delta: 1.0}`.

## "EnergyForceLoss does not support ignore_nan"

**Problem**: Training crashes because the loss function can't handle NaN force labels.

**Fix**: Use `EnergyForceStressLoss` with `stress: 0.0`:
```yaml
loss:
  _target_: nequip.train.EnergyForceStressLoss
  coeffs: {total_energy: 1.0, forces: 10.0, stress: 0.0}
  ignore_nan: {forces: true, stress: true}
```

## Unstable MD from energy-only head

**Problem**: The delta head produces finite energies but unstable molecular dynamics.

**Causes** (check in order):
1. Energy scale too large — don't use `forces_rms` for energy-only heads. Use `1.0`.
2. Missing `per_head_l_max` — try `per_head_l_max: {delta: 0}` to constrain the per-atom decomposition.
3. Not enough training — energy-only heads converge slower. Check that validation energy MAE is still decreasing.

**Important**: Good validation metrics don't guarantee MD stability. Always validate with short NPT before production.

## "extract_head is not a registered model modifier"

**Problem**: `nequip-compile --modifiers extract_head` fails because the modifier isn't found.

**Causes**:
1. **Wrong checkpoint**: If trained with the old fork (`_target_: nequip.model.NequIPGNNModel`), use `--head` instead of `--modifiers`. Only checkpoints from the extension (`_target_: nequip_multihead.model.MultiHeadNequIPGNNModel`) have the modifier.
2. **Extension not installed**: `nequip-multihead` must be installed in the same environment as `nequip-compile`.

## Non-periodic clusters crash with inf

**Problem**: `ForceStressOutput` produces inf stress on non-periodic clusters.

**Fix**: Add a large dummy cell during data preparation:
```python
atoms.cell = 100.0 * np.eye(3)  # dummy cell, pbc stays False
```

## SWA checkpoint gives same result as last.ckpt

**Problem**: Compiling from `swa_last.ckpt` gives the same model as `last.ckpt`.

**Cause**: The SWA phase never started. Check that `swa_start_epoch < max_epochs` and that training reached `swa_start_epoch`.

**Diagnosis**: Look for `SWA phase started` in the training log. If missing, the training ended before reaching `swa_start_epoch`.

## EMA checkpoint assertion error on best.ckpt

**Problem**: `nequip-package` or `nequip-compile` on `best.ckpt` fails with "EMA module loaded in a state where it does not contain EMA weights."

**Cause**: Lightning saves `best.ckpt` during validation while EMA weights are swapped. Our NequIP fork fixes this.

**Fix**: Use `last.ckpt`, or ensure you have `henriasv/nequip@feature/parameterized-modifiers` installed.
