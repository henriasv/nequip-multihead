# Introduction

## What is nequip-multihead?

`nequip-multihead` is an extension package for [NequIP](https://github.com/mir-group/nequip) that adds multi-head training support. It allows a single model to learn from multiple datasets with different levels of theory — for example, DFT with forces and CCSD(T) with energies only — using a shared equivariant backbone and separate per-head readout networks.

## When to use multi-head training

Multi-head training is useful when:

- You have a large, inexpensive dataset (e.g. PBE-D3 with energies, forces, and stresses) and a smaller, expensive dataset (e.g. RPA or CCSD(T) with energies only)
- Both datasets describe the same chemical system and atom types
- You want the expensive-level head to benefit from the structural information learned by the cheaper-level head

A common application is **delta-learning**, where one head learns the baseline energy surface (with forces) and another learns the correction to a higher level of theory (energy only). At deployment, both heads are summed to predict the target-level energy surface.

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.8+
- NequIP 0.17+ (with parameterized modifier support)

### Install from GitHub

```bash
# Install NequIP with required extensions
pip install git+https://github.com/henriasv/nequip.git@feature/parameterized-modifiers

# Install nequip-multihead
pip install git+https://github.com/henriasv/nequip-multihead.git
```

### Verify installation

```python
from nequip_multihead.model import MultiHeadNequIPGNNModel
from nequip_multihead.nn import MultiHeadReadout, PerHeadConvNetLayer
from nequip_multihead.transforms import HeadStamper
print("OK")
```

## Architecture

The model architecture extends NequIP's standard GNN:

```
Shared backbone (embedding + ConvNetLayers 0..N-2)
    ↓
[Optional: PerHeadConvNetLayer — per-head l_max control]
    ↓
MultiHeadReadout:
  ├─ Head 0: ScalarMLP → PerTypeScaleShift → (selected by HEAD_KEY)
  ├─ Head 1: ScalarMLP → PerTypeScaleShift → (selected by HEAD_KEY)
  └─ ...
    ↓
AtomwiseReduce → total energy
    ↓
ForceStressOutput → forces, stress (via autograd)
```

Each frame in the training data is stamped with a `HEAD_KEY` integer indicating which head it belongs to. The `MultiHeadReadout` computes all heads' energies but selects the appropriate one per frame based on `HEAD_KEY`. Forces are computed via autograd of the selected head's energy.

## Citation

If you use this package, please cite NequIP:

```bibtex
@article{batzner2022,
  title={E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials},
  author={Batzner, Simon and Musaelian, Albert and Sun, Lixin and Geiger, Mario and Mailoa, Jonathan P and Kornbluth, Mordechai and Molinari, Nicola and Smidt, Tess E and Kozinsky, Boris},
  journal={Nature communications},
  volume={13},
  number={1},
  pages={2453},
  year={2022}
}
```
