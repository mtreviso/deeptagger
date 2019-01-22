import torch.nn as nn


class CRFLayer(nn.Module):
    """From  ."""

    def __init__(self):
        super().__init__()
        self.is_built = False
        self.dropout = None
        # layers
        pass

    def build(self, options):
        self.is_built = True
        self.dropout = options.dropout

    def init_weights(self):
        pass

    def forward(self, batch):
        assert self.is_built
        h = None
        return h

    def _decode(self, input):
        pass
