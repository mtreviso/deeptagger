from torch import nn

from deeptagger.models.utils import clones
from deeptagger.modules.layer_norm import LayerNorm


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, nb_layers=1):
        super(Decoder, self).__init__()
        self.layers = clones(layer, nb_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
