# Integration Test Status (2026-03-24)

## What works

| Step | Status | Notes |
|------|--------|-------|
| `pip install` from GitHub | ✅ | Both nequip fork and extension |
| Import all modules | ✅ | |
| Build model (with/without per_head_l_max) | ✅ | |
| Forward pass (different heads → different energies) | ✅ | Required HEAD_KEY in model irreps_in fix |
| Training loop (manual, 5 steps) | ✅ | |
| ConcatDataset with mixed-head batches | ✅ | |
| Checkpoint save/load | ✅ | |
| `nequip-train` with YAML config | ✅ | Requires explicit energy scales (see gotcha #2) |
| `extract_head` (eager mode) | ✅ | Produces correct output matching multi-head model |
| `extract_summed_heads` (eager mode) | ✅ | Sum matches individual heads |
| `modify()` with extract_head modifier | ✅ | Available via `--modifiers extract_head head_name=dft` |
| `nequip-package build` | ⚠️ | Creates file, but can't read it back (torch.package extern_modules issue) |
| `nequip-compile` (AOT Inductor) | ❌ | FX tracing fails: "fx'ed models for different input shapes do not agree" |

## Known gotchas

### 1. EMA checkpoint bug (best.ckpt)

`nequip-package` and `nequip-compile` on `best.ckpt` fail with:
```
AssertionError: EMA module loaded in a state where it does not contain EMA weights
```

**Cause**: Lightning's `ModelCheckpoint.on_validation_end` fires before `LightningModule.on_validation_end`, saving the checkpoint while EMA weights are swapped.

**Fix**: Included in `henriasv/nequip@feature/parameterized-modifiers` — `on_save_checkpoint` corrects the state.

**Workaround**: Use `last.ckpt` instead of `best.ckpt`.

### 2. NaN forces_rms from ConcatDataset with energy-only heads

When `ConcatDataset` concatenates a force-supervised dataset and an energy-only dataset (NaN forces), `CommonDataStatisticsManager` computes `forces_rms: nan`. Using `${training_data_stats:forces_rms}` as `per_type_energy_scales` then produces NaN scales → NaN model output.

**Fix**: Specify energy scales explicitly instead of using `${training_data_stats:forces_rms}`.

```yaml
per_type_energy_scales:
  pbe_d3: 10.0  # explicit, NOT ${training_data_stats:forces_rms}
  rpa: 1.0
```

### 3. torch.package can't load extension modules

`nequip-package info` and `nequip-compile` from `.nequip.zip` fail with:
```
RuntimeError: PytorchStreamReader failed locating file .data/extern_modules
```

**Cause**: `torch.package` needs `nequip_multihead` declared as an external module. The packaging doesn't register it.

**Status**: Not fixed. Compile directly from checkpoints, not packages.

### 4. AOT Inductor FX tracing fails on extracted models

`nequip-compile` with `--modifiers extract_head` fails with:
```
RuntimeError: the fx'ed models for different input shapes do not agree
```

**Cause**: The `SequentialGraphNetwork` rebuilt by `extract_head` has shape-specialized operations (e.g. `PerTypeScaleShift.select` expanding to 0 vs N atoms). NequIP's `nequip_make_fx` traces with two different shapes and expects identical FX graphs.

**Status**: Not fixed. The fork's version works because the model structure differs. Need to investigate what the extension's extracted model does differently.

**Workaround**: Use eager mode inference (no compilation) for extracted models, or use the fork.
