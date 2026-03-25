# Installation and Prerequisites

## Prerequisites

- Python 3.10+
- PyTorch 2.8+
- CUDA GPU (for compilation and fast training)
- NequIP 0.17+ with parameterized modifier support

## Install

```bash
pip install git+https://github.com/henriasv/nequip.git@feature/parameterized-modifiers
pip install git+https://github.com/henriasv/nequip-multihead.git
```

## Verify

```python
from nequip_multihead.model import MultiHeadNequIPGNNModel
from nequip_multihead.nn import MultiHeadReadout, PerHeadConvNetLayer
print("OK")
```

## Citation

If you use this package, please cite NequIP:

```bibtex
@article{batzner2022,
  title={E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials},
  author={Batzner, Simon and Musaelian, Albert and Sun, Lixin and Geiger, Mario and Mailoa, Jonathan P and Kornbluth, Mordechai and Molinari, Nicola and Smidt, Tess E and Kozinsky, Boris},
  journal={Nature communications},
  volume={13},
  pages={2453},
  year={2022}
}
```
