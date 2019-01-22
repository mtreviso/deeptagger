from torch import nn

from deeptagger.modules.sublayer import SublayerConnection


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout=0.0):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.input_sublayer = SublayerConnection(size, dropout=dropout)
        self.output_sublayer = SublayerConnection(size, dropout=dropout)

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        apply_attn = lambda x: self.self_attn(x, x, x, mask)[0]  # get only the outputs - ignore the probs
        x = self.input_sublayer(x, apply_attn)
        return self.output_sublayer(x, lambda x: self.feed_forward(x))


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout=0.0):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.input_sublayer = SublayerConnection(size, dropout=dropout)
        self.middle_sublayer = SublayerConnection(size, dropout=dropout)
        self.output_sublayer = SublayerConnection(size, dropout=dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.input_sublayer(x, lambda x: self.self_attn(x, x, x, tgt_mask)[0])
        x = self.middle_sublayer(x, lambda x: self.src_attn(x, m, m, src_mask)[0])
        return self.output_sublayer(x, lambda x: self.feed_forward(x))
