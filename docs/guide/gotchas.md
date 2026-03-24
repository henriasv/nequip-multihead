# Known Issues and Gotchas

## Energy scales for energy-only heads

Using `forces_rms` from a force-supervised head as the energy scale for an energy-only head causes the readout to operate at very small values, degrading autograd force quality and producing unstable molecular dynamics. Use `1.0` (no scaling) for energy-only heads.

```yaml
per_type_energy_scales:
  baseline: 10.0   # from forces_rms of baseline data
  delta: 1.0       # NOT forces_rms — critical for force quality
```

## NaN forces_rms from ConcatDataset

When `ConcatDataset` mixes force-supervised and energy-only data, the default `CommonDataStatisticsManager` computes `forces_rms: nan` (because NaN forces are included).

**Fix**: Use `nequip_multihead.data.MultiHeadDataStatisticsManager` instead — it uses `ignore_nan=True` for force statistics. See [Configuration: NaN-safe statistics](configuration.md#nan-safe-statistics).

Alternatively, specify energy scales explicitly (e.g. `per_type_energy_scales: {baseline: 10.0, delta: 1.0}`).

## EnergyForceLoss vs EnergyForceStressLoss

`EnergyForceLoss` does not support `ignore_nan`. For multi-head training with energy-only heads, use `EnergyForceStressLoss` with `stress: 0.0` and `ignore_nan: {forces: true, stress: true}`.

## Non-periodic clusters need a dummy cell

`ForceStressOutput` computes stress via `virial / volume`. Non-periodic clusters with a zero cell cause `inf`. Add a large dummy cell (e.g. `100 * eye(3)`) with `pbc=False` during data preparation.

## Validation metrics don't predict MD stability

Standard energy/force MAE metrics are not sufficient to assess whether an energy-only head will produce physically meaningful forces for molecular dynamics. A model with better validation metrics can produce catastrophically wrong MD trajectories. Always validate delta-learning models with short NPT simulations before production use.

## torch.package limitations

`nequip-package` may not correctly handle extension package modules. Compile directly from checkpoints rather than packages.
