# Training a Delta-Learning Model

Train a single model that combines a cheap baseline (e.g. DFT with forces) and an expensive correction (e.g. CCSD(T) energy only). At deployment, both heads are summed to predict the target-level energy surface.

## Prepare your data

You need two datasets:

1. **Baseline** — periodic structures with energy + forces (+ optionally stress)
2. **Delta correction** — clusters or periodic structures with energy only

The delta dataset should contain the *difference* between target and baseline energies, not the absolute target energies.

### Energy-only data requirements

For the delta dataset where forces are not available:

- **Forces must be NaN**, not missing. Set `forces = np.full((n_atoms, 3), np.nan)` in your extxyz files.
- **Stress must be NaN** if not available.
- **Non-periodic clusters need a dummy cell** — e.g. `100 * np.eye(3)` with `pbc=False`. Without this, `ForceStressOutput` divides virial by zero volume.

```python
# Example: preparing energy-only delta data
from ase.io import write
import numpy as np

for atoms in delta_frames:
    atoms.arrays["forces"] = np.full((len(atoms), 3), np.nan)
    atoms.info["stress"] = np.full(6, np.nan)
    if not any(atoms.pbc):
        atoms.cell = 100.0 * np.eye(3)
    write("delta_train.xyz", atoms, append=True)
```

## Configure the model

The full config is in `configs/delta_learning.yaml`. Here are the key sections:

### Model with per_head_l_max

```yaml
model:
  _target_: nequip_multihead.model.MultiHeadNequIPGNNModel
  head_names: [baseline, delta]
  l_max: 2
  num_layers: 4

  per_head_l_max:
    baseline: 2    # full angular momentum (force-supervised)
    delta: 0       # scalar paths only (energy-only)
```

`per_head_l_max` constrains the delta head to use only l=0 tensor product paths in the final interaction layer. This reduces the degrees of freedom in the per-atom energy decomposition, producing better autograd forces for the energy-only head. The baseline head uses the full l_max since it has force supervision.

Both heads share the same edge MLP weights — training the baseline head's l=0 paths directly improves the delta head's representation.

### Energy scales

```yaml
  per_type_energy_scales:
    baseline: ${training_data_stats:forces_rms}
    delta: 1.0
```

```{warning}
Do **not** use `forces_rms` for the delta head. The delta energies are typically much smaller than baseline energies. Using `forces_rms` causes the readout to operate at very small values, degrading autograd force quality and producing unstable MD.
```

### Data with ConcatDataset

```yaml
data:
  train_dataset:
    _target_: torch.utils.data.ConcatDataset
    datasets:
      - _target_: nequip.data.dataset.ASEDataset
        file_path: baseline_train.xyz
        transforms:
          # ... standard transforms ...
          - _target_: nequip_multihead.transforms.HeadStamper
            head_index: 0
      - _target_: nequip.data.dataset.ASEDataset
        file_path: delta_train.xyz
        transforms:
          - _target_: nequip_multihead.transforms.HeadStamper
            head_index: 1

  stats_manager:
    _target_: nequip_multihead.data.MultiHeadDataStatisticsManager
    type_names: ${model_type_names}
```

Use `MultiHeadDataStatisticsManager` instead of the default — it excludes NaN forces when computing `forces_rms`.

### Loss with ignore_nan

```yaml
loss:
  _target_: nequip.train.EnergyForceStressLoss
  coeffs:
    total_energy: 1.0
    forces: 10.0
    stress: 0.0
  ignore_nan:
    forces: true
    stress: true
```

```{note}
You must use `EnergyForceStressLoss`, not `EnergyForceLoss` — only the former supports `ignore_nan`.
```

## Train

```bash
nequip-train --config-dir=. --config-name=your_config
```

With `compile_mode: compile` in the model config, the first epoch is slow (compilation) but subsequent epochs are 3-4x faster.

### Sampling behavior

With `ConcatDataset`, each frame is seen exactly once per epoch. If the baseline has 1000 frames and delta has 100, the model sees baseline data ~10x more often. There is no recycling of the smaller dataset — this is intentional and generally appropriate.

## Deploy

### Single head (baseline only)

```bash
nequip-compile last.ckpt baseline.nequip.pt2 \
  --mode aotinductor --device cuda --target ase \
  --modifiers extract_head head_name=baseline
```

### Summed heads (baseline + delta = target level)

```bash
nequip-compile last.ckpt target.nequip.pt2 \
  --mode aotinductor --device cuda --target ase \
  --modifiers extract_summed_heads head_names=baseline+delta
```

The summed model runs both heads' readouts and sums the per-atom energies. Forces come from autograd of the summed energy — correct by construction.

### Use with ASE

```python
from nequip.integrations.ase import NequIPCalculator

calc = NequIPCalculator.from_compiled_model(
    "target.nequip.pt2", device="cuda",
    chemical_species_to_atom_type_map=True,
)
atoms.calc = calc
```

```{important}
Always validate delta-learning models with short NPT simulations before production use. Good validation metrics do not guarantee stable MD — a model with better energy MAE can produce wrong densities.
```
