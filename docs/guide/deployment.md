# Deploying Multi-Head Models

After training, extract individual heads or sum multiple heads for deployment. The compiled model is a standard NequIP model — all downstream integrations (ASE, LAMMPS, TorchSim) work without multi-head awareness.

## Extract a single head

```bash
nequip-compile last.ckpt baseline.nequip.pt2 \
  --mode aotinductor --device cuda --target ase \
  --modifiers extract_head head_name=baseline
```

This strips the `MultiHeadReadout`, replacing it with the specified head's readout + scale/shift. The result is a standard single-head NequIP model.

## Extract summed heads (delta-learning)

For delta-learning where the deployed prediction is E_base + E_delta:

```bash
nequip-compile last.ckpt target.nequip.pt2 \
  --mode aotinductor --device cuda --target ase \
  --modifiers extract_summed_heads head_names=baseline+delta
```

Both `+` and `,` separators work: `head_names=baseline+delta` or `head_names=baseline,delta`.

The compiled model sums both heads' per-atom energies. Forces and stress come from autograd of the summed energy.

## Use with ASE

```python
from nequip.integrations.ase import NequIPCalculator

calc = NequIPCalculator.from_compiled_model(
    "target.nequip.pt2",
    device="cuda",
    chemical_species_to_atom_type_map=True,
)
atoms.calc = calc
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

## Use with LAMMPS

Compile with the `pair_nequip` target:

```bash
nequip-compile last.ckpt target.nequip.pt2 \
  --mode aotinductor --device cuda --target pair_nequip \
  --modifiers extract_head head_name=baseline
```

Then in your LAMMPS input:

```
pair_style nequip
pair_coeff * * target.nequip.pt2 Cu Al
```

## Which checkpoint to use

Use **`last.ckpt`** for compilation. With EMA training, `best.ckpt` may have swapped weight state from being saved during validation. Our NequIP fork fixes this, but `last.ckpt` is safer.

## Compilation requirements

- **CUDA**: AOT compilation works on CUDA. CPU compilation has a PyTorch 2.10 codegen bug for float64 models.
- **Same environment**: The `nequip-multihead` package must be installed when compiling from a checkpoint, since the checkpoint references the extension's model builder.
