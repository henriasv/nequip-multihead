# How Multi-Head Training Works

## The problem multi-head solves

In computational chemistry, you often have a cheap method (DFT) that can produce forces and a more accurate method (CCSD(T), RPA) that's too expensive for forces. Multi-head training lets a single model learn from both: a shared backbone captures the physics, while separate readout heads specialize to each energy surface.

The shared backbone benefits from force supervision (the cheap head), while the expensive head gets better representations "for free" — it inherits the structural understanding without needing its own force labels.

## Architecture

```
                    ┌─ Head 0: ScalarMLP → ScaleShift ─┐
Input → Backbone → ─┤                                   ├→ Select by HEAD_KEY → Energy → Forces
                    └─ Head 1: ScalarMLP → ScaleShift ─┘
```

- **Backbone**: Standard NequIP layers (embedding + ConvNetLayers). Shared across all heads. This is where ~95% of the parameters and compute live.
- **MultiHeadReadout**: Each head has its own `ScalarMLP` (readout) and `PerTypeScaleShift` (per-element energy normalization). Tiny compared to the backbone.
- **HEAD_KEY selection**: Each training frame is stamped with a head index. `MultiHeadReadout` computes all heads' energies but selects the appropriate one per frame.
- **Forces**: Computed by `ForceStressOutput` via autograd of the selected head's energy w.r.t. positions.

## The per-atom decomposition problem

An energy-only head learns accurate *total* energies but can produce poor *forces*. Why?

The total energy is a sum of per-atom contributions: E = Σ eᵢ. The training loss constrains only the sum. With N atoms, there are N-1 unconstrained degrees of freedom in how energy is distributed among atoms. The readout MLP can redistribute energy freely without changing the total — but this redistribution corrupts the position gradients (autograd forces).

Higher-l equivariant features make this worse: they give the readout more directionally-sensitive basis functions to redistribute energy, increasing the degrees of freedom.

## Per-head angular momentum (per_head_l_max)

`per_head_l_max` constrains energy-only heads to use fewer angular momentum channels in the final interaction layer, reducing the decomposition degrees of freedom.

```
Shared backbone (full l_max)
    ↓
PerHeadConvNetLayer:
  ├─ Head 0 (l_max=2): all TP paths      ← force-supervised, needs full expressiveness
  └─ Head 1 (l_max=0): l=0 paths only    ← energy-only, constrained decomposition
```

### Weight sharing

The `PerHeadConvNetLayer` shares a single edge MLP across all heads. Each head's `TensorProduct` uses a contiguous prefix of the full weight vector (lower-l paths come first in the sorted instruction list). Training the force-supervised head's l=0 paths directly improves the representation used by the energy-only head.

This means the energy-only head isn't learning in isolation — it benefits from force supervision through the shared weights.

## Design decisions

### ConcatDataset vs CombinedLoader

We use `ConcatDataset` (single dataloader, mixed-head batches) rather than `CombinedLoader` (one dataloader per head). Advantages:

- Simpler: works with standard NequIP training — no custom training_step
- Natural sampling: each frame seen once per epoch, weighted by dataset size
- No data recycling: avoids artificial oversampling of smaller datasets
- Compatible with `compile_mode: compile` without modifications

### Extension package vs NequIP fork

Multi-head is an extension package (`nequip-multihead`) rather than part of NequIP core. This keeps NequIP's infrastructure simpler and allows independent versioning. The extension uses NequIP's standard extension points: `@model_builder`, `GraphModuleMixin`, `@model_modifier`, and `ConcatDataset`.

The NequIP fork (`feature/parameterized-modifiers`) adds only minimal infrastructure: parameterized compile modifiers, EMA checkpoint fix, small-molecule FX fix, and MetricsManager tensor caching.
