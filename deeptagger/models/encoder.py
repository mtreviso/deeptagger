from torch import nn

from deeptagger.models.utils import clones
from deeptagger.modules.layer_norm import LayerNorm


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, nb_layers=1):
        super().__init__()
        self.layers = clones(layer, nb_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
