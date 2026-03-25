nequip-multihead
================

Multi-head training extension for `NequIP <https://github.com/mir-group/nequip>`_.
Train a single model on multiple datasets with different levels of theory
(e.g. DFT with forces + CCSD(T) energy only) using shared backbone and
per-head readout networks.

Key features:

- **Multi-head readout**: Per-head ``ScalarMLP`` + ``PerTypeScaleShift``, selected by ``HEAD_KEY``
- **Per-head angular momentum** (``per_head_l_max``): Constrain energy-only heads to lower-l tensor product paths to improve autograd force quality
- **Head extraction**: ``extract_head`` and ``extract_summed_heads`` model modifiers for deployment via ``nequip-compile``
- **ConcatDataset integration**: Single dataloader with mixed-head batches, compatible with standard NequIP training

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   intro/intro
   guide/guide
   api/api
