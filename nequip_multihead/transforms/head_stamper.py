import math
from typing import List, Optional

import torch

from nequip_multihead._keys import HEAD_KEY


class HeadStamper(torch.nn.Module):
    """Transform that stamps each data sample with a head index.

    Used with ``ConcatDataset`` to label which head each frame belongs to.
    The head index is stored in ``HEAD_KEY`` as a per-frame integer tensor.

    Optionally sets specified keys to NaN, useful for heads that lack
    certain labels (e.g. energy-only heads with no force labels).

    Args:
        head_index: integer head index to stamp on each sample.
        nan_keys: optional list of data keys to fill with NaN.
    """

    def __init__(self, head_index: int, nan_keys: Optional[List[str]] = None):
        super().__init__()
        self.head_index = head_index
        self.nan_keys = nan_keys or []

    def forward(self, data):
        data[HEAD_KEY] = torch.tensor([self.head_index], dtype=torch.long)
        for key in self.nan_keys:
            if key in data and isinstance(data[key], torch.Tensor):
                data[key] = torch.full_like(data[key], math.nan)
        return data
