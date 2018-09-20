import torch
import torch.nn as nn


from deeptagger import constants


class CRFLayer(nn.Module):
    """From  ."""

    def __init__(self):
        super().__init__()
        # layers
        pass

    def build(self, options):
        self.is_built = True

    def init_weights(self):
        pass

    def forward(self, batch):
        assert self.is_built
        h = None
        return h

    def _decode(self, input):
        pass
