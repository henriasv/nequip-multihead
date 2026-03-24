# nequip-multihead

Multi-head training extension for [NequIP](https://github.com/mir-group/nequip).

Train a single model on multiple datasets with different levels of theory (e.g. DFT with forces + CCSD(T) energy only) using a shared equivariant backbone and per-head readout networks.

## Features

- **Multi-head readout** with per-head `ScalarMLP` + `PerTypeScaleShift`
- **Per-head angular momentum** (`per_head_l_max`): constrain energy-only heads to lower-l tensor product paths for better autograd force quality
- **Head extraction** via `nequip-compile --modifiers extract_head head_name=dft`
- **Summed heads** for delta-learning: `--modifiers extract_summed_heads head_names=dft+rpa`
- Works with standard `nequip-train`, `ConcatDataset`, and `EnergyForceStressLoss` with `ignore_nan`

## Installation

```bash
pip install git+https://github.com/henriasv/nequip.git@feature/parameterized-modifiers
pip install git+https://github.com/henriasv/nequip-multihead.git
```

## Quick start

```yaml
# In your nequip training config:
training_module:
  model:
    _target_: nequip_multihead.model.MultiHeadNequIPGNNModel
    head_names: [baseline, delta]
    per_type_energy_shifts:
      baseline: {H: -13.58, O: -431.27}
      delta: {H: -0.37, O: -9.73}
    per_type_energy_scales:
      baseline: 10.0
      delta: 1.0
    # ... standard NequIP params ...
```

See the [documentation](https://henriasv.github.io/nequip-multihead/) for full configuration guide.

## License

MIT
