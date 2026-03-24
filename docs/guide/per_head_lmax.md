# Per-Head Angular Momentum (`per_head_l_max`)

## The problem

Energy-only heads suffer from per-atom energy decomposition degeneracy. The total energy provides only one constraint per frame, but N atoms have N-1 unconstrained degrees of freedom in the per-atom decomposition. Higher-l equivariant features give the readout more ways to redistribute energy among atoms without affecting the total, worsening autograd force quality.

## The solution

`per_head_l_max` restricts which angular momentum features each head can use in the final interaction layer. The last `ConvNetLayer` is replaced by a `PerHeadConvNetLayer` where each head uses only tensor product paths with input angular momentum l <= its configured `per_head_l_max`.

```yaml
model:
  _target_: nequip_multihead.model.MultiHeadNequIPGNNModel
  l_max: 2
  num_layers: 4
  head_names: [baseline, delta]
  per_head_l_max:
    baseline: 2    # full l_max — force-supervised
    delta: 0       # scalar paths only — constrains decomposition
```

## How it works

The `PerHeadConvNetLayer` shares a single edge MLP across all heads. Each head's `TensorProduct` uses a contiguous subset of the full weight vector — paths are sorted by l, so lower-l paths come first.

```
Shared backbone layers 0..N-2  (full l_max irreps)
    ↓
PerHeadConvNetLayer (SHARED edge MLP, per-head TP paths):
  ├─ Head 0 (l_max=2):  TP with all paths       ← full weight vector
  ├─ Head 1 (l_max=0):  TP with l=0 paths only  ← subset of same weights
    ↓
MultiHeadReadout (per-head ScalarMLP + ScaleShift)
```

Training the force-supervised head's l=0 tensor product paths directly improves the representation used by the energy-only head, since they share the same weights.

## Configuration

Heads without an entry in `per_head_l_max` default to the backbone's `l_max`. If `per_head_l_max` is omitted entirely, all heads use the full `l_max` (backward compatible).

```{tip}
Use the default linear readout (`readout_mlp_hidden_layers_depth: 0`) for multi-head models where some heads are energy-only. Adding hidden layers degraded force quality for energy-only heads in our testing.
```

```{important}
`per_head_l_max` requires `num_layers >= 2` since the last ConvNetLayer becomes per-head while the preceding layers remain shared.
```
