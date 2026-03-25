# Running the Integration Tests

nequip-multihead ships with GPU integration tests that verify the full
train → compile → deploy pipeline. These live in `tests/` and require a
CUDA GPU.

## Quick start

```bash
cd tests/
sbatch run_full_pipeline_test.sh        # full pipeline
sbatch run_scheduler_compile_test.sh    # scheduler + compile
```

Both scripts assume a SLURM cluster with a `normal` partition and
`gpu:1` GRES.  Adjust the `#SBATCH` headers if your cluster differs.
They activate the `nequip_multihead_ext` conda environment — change this
to match your setup.

## What the tests cover

### Full pipeline test (`test_full_pipeline.py`)

Exercises the complete lifecycle in a single run:

1. **Train** a 2-head model (Cu + Al, EMT data, `compile_mode: compile`, 5 epochs).
2. **Compile** three variants from the same checkpoint:
   - `extract_head head_name=Cu_head`
   - `extract_head head_name=Al_head`
   - `extract_summed_heads head_names=Cu_head+Al_head`
3. **Verify** via eager inference with modifiers applied to the checkpoint:

| Test | What it checks |
|------|----------------|
| 1 | Cu\_head and Al\_head produce different energies |
| 2 | All energies are finite |
| 3 | Summed energy ≈ Cu\_head + Al\_head |
| 4 | Per-atom energies differ between heads |
| 5 | Per-atom sums match sum of individuals |
| 6 | Compiled `.pt2` files exist and are non-empty |
| 7 | Applying the same modifier twice gives the same result |

This test catches regressions in `modify()` return-value handling,
`extract_head`, `extract_summed_heads`, and the compilation pipeline.

**Run directly** (on a GPU node):

```bash
python tests/test_full_pipeline.py --work-dir /tmp/my_test
```

### Scheduler + compile test (`run_scheduler_compile_test.sh`)

Verifies that `GradientNormFractionScheduler` works with
`compile_mode: compile`.  This is the combination that requires the
eager `GraphModel.forward()` bypass to avoid `retain_graph` conflicts
with torch.compile's inplace optimizations.

Two configs are tested:

| Config | Heads | Data |
|--------|-------|------|
| `test_scheduler_compile.yaml` | 2 (Cu + Al) | Full supervision |
| `test_5head_scheduler_compile.yaml` | 5 | Mixed: energy-only, force-only, stress-only via `nan_keys` |

The 5-head test is the harder case — it exercises `ignore_nan`, partial
gradient norms, and the scheduler's EMA carry-forward on batches where
some loss components have no gradient.

## Test configs

All configs use EMT test data (self-contained, no external files) and
tiny architectures (`l_max=1`, `num_features=16`, `num_layers=2`) so
they complete in under a minute on GPU.

| Config | Model | Epochs | Batches/epoch | Scheduler fires |
|--------|-------|--------|---------------|-----------------|
| `test_scheduler_compile.yaml` | float32, compile | 5 | 8 | ~8× |
| `test_5head_scheduler_compile.yaml` | float32, compile | 5 | 10 | ~10× |

## Adding new tests

New test configs go in `tests/`.  Follow the pattern:

- Use `EMTTestDataset` for self-contained data
- Use `float32` + `compile_mode: compile` to test the compiled path
- Keep architectures minimal for fast turnaround
- Add the config name to the appropriate SLURM runner script
