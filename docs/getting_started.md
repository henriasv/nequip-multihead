# Getting Started

Train your first multi-head NequIP model in 5 minutes.

## Install

```bash
pip install git+https://github.com/henriasv/nequip.git@feature/parameterized-modifiers
pip install git+https://github.com/henriasv/nequip-multihead.git
```

Verify:

```bash
python -c "from nequip_multihead.model import MultiHeadNequIPGNNModel; print('OK')"
```

## Train a 2-head model

This example trains a model with two heads on EMT test data. Head 0 learns Cu energies/forces, head 1 learns Al energies/forces — both through a shared backbone.

Clone the repo and run:

```bash
git clone https://github.com/henriasv/nequip-multihead.git
cd nequip-multihead
nequip-train --config-dir=configs --config-name=minimal_multihead
```

Training completes in under a minute on GPU. You'll see loss decreasing for both heads.

### What's in the config

The key multi-head sections in `configs/minimal_multihead.yaml`:

**Model** — uses the extension's builder with `head_names`:

```yaml
model:
  _target_: nequip_multihead.model.MultiHeadNequIPGNNModel
  head_names: [Cu_head, Al_head]
  per_type_energy_scales:
    Cu_head: ${training_data_stats:forces_rms}
    Al_head: ${training_data_stats:forces_rms}
  per_type_energy_shifts:
    Cu_head: {Cu: 0.0, Al: 0.0}
    Al_head: {Cu: 0.0, Al: 0.0}
  # ... standard NequIP params (l_max, num_layers, etc.)
```

**Dataloader** — always use `shuffle: true` so frames are reordered each epoch:

```yaml
dataloader:
  _target_: torch.utils.data.DataLoader
  batch_size: 5
  shuffle: true
```

**Data** — `ConcatDataset` with `HeadStamper` labeling each dataset:

```yaml
train_dataset:
  _target_: torch.utils.data.ConcatDataset
  datasets:
    - _target_: nequip.data.dataset.EMTTestDataset
      transforms:
        # ... standard transforms ...
        - _target_: nequip_multihead.transforms.HeadStamper
          head_index: 0    # matches position in head_names
      element: Cu
    - _target_: nequip.data.dataset.EMTTestDataset
      transforms:
        - _target_: nequip_multihead.transforms.HeadStamper
          head_index: 1
      element: Al
```

**Stats manager** — NaN-safe version for mixed datasets:

```yaml
stats_manager:
  _target_: nequip_multihead.data.MultiHeadDataStatisticsManager
  type_names: ${model_type_names}
```

## Verify the trained model

```python
import torch
from nequip.utils.global_state import set_global_state
set_global_state(allow_tf32=False)

from nequip.model.saved_models.load_utils import load_saved_model
from nequip.model.utils import _EAGER_MODEL_KEY
from nequip.train.lightning import _SOLE_MODEL_KEY
from nequip_multihead._keys import HEAD_KEY
from nequip.data import AtomicDataDict

# Load checkpoint
model = load_saved_model("outputs/.../last.ckpt", _EAGER_MODEL_KEY, _SOLE_MODEL_KEY)

# Check: different heads produce different energies
# (use your actual test data here)
```

## Extract and compile a head

Extract a single head for deployment:

```bash
nequip-compile last.ckpt cu_head.nequip.pt2 \
  --mode aotinductor --device cuda --target ase \
  --modifiers extract_head head_name=Cu_head
```

Use with ASE:

```python
from nequip.integrations.ase import NequIPCalculator

calc = NequIPCalculator.from_compiled_model(
    "cu_head.nequip.pt2", device="cuda",
    chemical_species_to_atom_type_map=True,
)
atoms.calc = calc
energy = atoms.get_potential_energy()
```

## Next steps

- [Delta-learning workflow](guide/delta_learning.md) — baseline DFT + CCSD(T) correction with `per_head_l_max`
- [Loss scheduling](guide/loss_scheduling.md) — dynamic force/energy balance during training
- [Architecture](explanation/architecture.md) — why multi-head training works and design decisions
