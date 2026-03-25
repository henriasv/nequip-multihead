nequip-multihead
================

Multi-head training extension for `NequIP <https://github.com/mir-group/nequip>`_.
Train a single model on multiple datasets with different levels of theory using
a shared backbone and per-head readout networks.

.. code-block:: text

   Data (DFT + CCSD(T)) → Train → Extract head → Compile → Deploy (ASE/LAMMPS)

.. toctree::
   :maxdepth: 2
   :caption: Tutorial

   getting_started

.. toctree::
   :maxdepth: 2
   :caption: How-To Guides

   guide/delta_learning
   guide/loss_scheduling
   guide/deployment
   guide/troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: Explanation

   explanation/architecture

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/api
   intro/intro
