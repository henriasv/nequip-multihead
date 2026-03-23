import torch

from nequip_multihead._keys import HEAD_KEY


class HeadStamper(torch.nn.Module):
    """Transform that stamps each data sample with a head index.

    Used with ``ConcatDataset`` to label which head each frame belongs to.
    The head index is stored in ``HEAD_KEY`` as a per-frame integer tensor.

    Args:
        head_index: integer head index to stamp on each sample.
    """

    def __init__(self, head_index: int):
        super().__init__()
        self.head_index = head_index

    def forward(self, data):
        data[HEAD_KEY] = torch.tensor([self.head_index], dtype=torch.long)
        return data
