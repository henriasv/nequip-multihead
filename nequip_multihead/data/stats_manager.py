"""NaN-safe data statistics manager for multi-head training.

When using ConcatDataset with mixed force-supervised and energy-only heads,
the standard CommonDataStatisticsManager computes forces_rms including NaN
entries, producing NaN. This manager uses ignore_nan=True for force statistics
so that only finite force values contribute to the RMS.
"""
from nequip.data import AtomicDataDict
from nequip.data.stats_manager import (
    DataStatisticsManager,
    RootMeanSquare,
    Mean,
    NumNeighbors,
    PerAtomModifier,
)

from typing import Dict, List, Optional, Any


def MultiHeadDataStatisticsManager(
    dataloader_kwargs: Optional[Dict[str, Any]] = None,
    type_names: Optional[List[str]] = None,
):
    """Data statistics manager that handles NaN forces from energy-only heads.

    Drop-in replacement for ``CommonDataStatisticsManager`` when using
    ``ConcatDataset`` with mixed force-supervised and energy-only data.
    Forces statistics (``forces_rms``, ``per_type_forces_rms``) ignore NaN
    entries so that ``${training_data_stats:forces_rms}`` evaluates to the
    correct RMS of only the finite force values.

    Usage in config:

    .. code-block:: yaml

        data:
          stats_manager:
            _target_: nequip_multihead.data.MultiHeadDataStatisticsManager
            type_names: ${model_type_names}

    Then ``${training_data_stats:forces_rms}`` works correctly even with
    energy-only heads in the ConcatDataset.
    """
    metrics = [
        {
            "name": "num_neighbors_mean",
            "field": NumNeighbors(),
            "metric": Mean(),
        },
        {
            "name": "per_type_num_neighbors_mean",
            "field": NumNeighbors(),
            "metric": Mean(),
            "per_type": True,
        },
        {
            "name": "per_atom_energy_mean",
            "field": PerAtomModifier(AtomicDataDict.TOTAL_ENERGY_KEY),
            "metric": Mean(),
        },
        {
            "name": "forces_rms",
            "field": AtomicDataDict.FORCE_KEY,
            "metric": RootMeanSquare(),
            "ignore_nan": True,
        },
        {
            "name": "per_type_forces_rms",
            "field": AtomicDataDict.FORCE_KEY,
            "metric": RootMeanSquare(),
            "per_type": True,
            "ignore_nan": True,
        },
    ]
    return DataStatisticsManager(metrics, dataloader_kwargs, type_names)
