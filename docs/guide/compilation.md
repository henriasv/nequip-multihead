# Compilation and Deployment

## Head extraction

Multi-head models must have a specific head extracted before deployment. The `extract_head` model modifier strips the `MultiHeadReadout` and replaces it with the selected head's readout, producing a standard single-head NequIP model.

### Compile a single head

```bash
nequip-compile checkpoint.ckpt dft_head.nequip.pt2 \
  --mode aotinductor --device cuda --target ase \
  --modifiers extract_head head_name=dft
```

### Compile summed heads (delta-learning)

For delta-learning deployment where the prediction is `E_base + E_delta`:

```bash
nequip-compile checkpoint.ckpt target.nequip.pt2 \
  --mode aotinductor --device cuda --target ase \
  --modifiers extract_summed_heads head_names=baseline+delta
```

The `+` separator indicates which heads to sum. The compiled model runs each head's readout independently and sums the per-atom energies. Forces are computed via autograd of the summed energy.

### Using compiled models

Compiled models are standard NequIP models — no multi-head awareness needed downstream:

```python
from nequip.integrations.ase import NequIPCalculator

calc = NequIPCalculator.from_compiled_model(
    "dft_head.nequip.pt2",
    device="cuda",
    chemical_species_to_atom_type_map=True,
)

atoms.calc = calc
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

## Packaging

```{warning}
`nequip-package` has known issues with extension package modules. Compile directly from checkpoints (`.ckpt`) rather than packages (`.nequip.zip`).
```

## Checkpoint notes

- Use `last.ckpt` for compilation. The `best.ckpt` is saved during validation when EMA weights are swapped — our NequIP fork includes a fix for this.
- The checkpoint stores the model's `_target_` (e.g. `nequip_multihead.model.MultiHeadNequIPGNNModel`), so the extension package must be installed when loading.
