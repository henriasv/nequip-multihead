# Configuration

## Model

Use `nequip_multihead.model.MultiHeadNequIPGNNModel` as the model builder. It accepts all standard `NequIPGNNModel` parameters plus multi-head specific ones:

```yaml
training_module:
  model:
    _target_: nequip_multihead.model.MultiHeadNequIPGNNModel
    seed: ${seed}
    model_dtype: float64
    type_names: ${model_type_names}
    r_max: ${cutoff_radius}
    l_max: 2
    parity: true
    num_layers: 4
    num_features: 64

    # Multi-head parameters
    head_names: [baseline, delta]

    # Per-head energy shifts (isolated atom energies per level of theory)
    per_type_energy_shifts:
      baseline:
        H: -13.587
        O: -431.267
      delta:
        H: -0.37
        O: -9.73

    # Per-head energy scales
    per_type_energy_scales:
      baseline: 10.0    # e.g. forces_rms from baseline data
      delta: 1.0         # energy-only head: do NOT use forces_rms
```

### Multi-head parameters

- **`head_names`** (required): List of head name strings. Each head gets its own readout MLP and per-type scale/shift.

- **`per_type_energy_shifts`**: Dict mapping head names to per-type energy shifts. Use isolated atom energies from each level of theory. Different methods can have very different absolute energies.

- **`per_type_energy_scales`**: Dict mapping head names to per-type energy scales. Use the special key `all` to broadcast the same value to every head:

  ```yaml
  per_type_energy_scales:
    all: 10.0
  ```

```{warning}
**Do not use `forces_rms` for energy-only heads.** This causes the readout to operate at very small values, degrading the numerical quality of autograd forces and producing unstable molecular dynamics. For energy-only heads, use `1.0` (no scaling).
```

## Data

Use `ConcatDataset` to combine datasets from different levels of theory into a single dataloader. Each dataset is stamped with a head index via `HeadStamper`:

```yaml
data:
  _target_: nequip.data.datamodule.NequIPDataModule
  seed: ${seed}

  train_dataset:
    _target_: torch.utils.data.ConcatDataset
    datasets:
      # Head 0: DFT data with energies + forces
      - _target_: nequip.data.dataset.ASEDataset
        file_path: dft_train.xyz
        transforms:
          - _target_: nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper
            model_type_names: ${model_type_names}
          - _target_: nequip.data.transforms.NeighborListTransform
            r_max: ${cutoff_radius}
          - _target_: nequip_multihead.transforms.HeadStamper
            head_index: 0

      # Head 1: higher-level data with energies only
      - _target_: nequip.data.dataset.ASEDataset
        file_path: rpa_train.xyz
        transforms:
          - _target_: nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper
            model_type_names: ${model_type_names}
          - _target_: nequip.data.transforms.NeighborListTransform
            r_max: ${cutoff_radius}
          - _target_: nequip_multihead.transforms.HeadStamper
            head_index: 1

  val_dataset:
    - _target_: nequip.data.dataset.ASEDataset
      file_path: dft_val.xyz
      transforms:
        - _target_: nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper
          model_type_names: ${model_type_names}
        - _target_: nequip.data.transforms.NeighborListTransform
          r_max: ${cutoff_radius}
        - _target_: nequip_multihead.transforms.HeadStamper
          head_index: 0
```

```{important}
The `head_index` in `HeadStamper` must match the position in the `head_names` list. For `head_names: [baseline, delta]`, the baseline dataset uses `head_index: 0` and the delta dataset uses `head_index: 1`.
```

## Loss

Use `EnergyForceStressLoss` with `ignore_nan` for energy-only heads that have NaN force labels:

```yaml
training_module:
  loss:
    _target_: nequip.train.EnergyForceStressLoss
    per_atom_energy: true
    coeffs:
      total_energy: 1.0
      forces: 100.0
      stress: 0.0
    ignore_nan:
      forces: true
      stress: true
```

## Data preparation for energy-only heads

When a head has no force labels, the force and stress fields must be present in the data but populated with NaN values. This allows `ignore_nan` to skip these entries during loss computation.

Non-periodic structures (e.g. gas-phase clusters) require a finite dummy cell because `ForceStressOutput` computes stress via `virial / volume`. Add a large dummy cell (e.g. `100 * eye(3)`) with `pbc=False`.

```{important}
`EnergyForceLoss` does **not** support `ignore_nan`. Use `EnergyForceStressLoss` even if you don't need stress predictions. Set the stress coefficient to `0.0`.
```
