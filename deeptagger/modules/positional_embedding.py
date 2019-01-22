from torch import nn

from deeptagger.modules.positional_encoding import PositionalEncoding


class PositionalEmbedding(nn.Module):
    """An embedding layer followed by positional encoding."""

    def __init__(self, vocab_size, embedding_size, max_seq_len=1000, dropout=0.0, scale=True):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_size)
        self.pe = PositionalEncoding(max_seq_len, embedding_size, dropout=dropout, scale=scale)

    def forward(self, x):
        x = self.emb(x)
        x = self.pe(x)
        return x
