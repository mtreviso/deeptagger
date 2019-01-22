from torch import nn

from deeptagger.models.decoder import Decoder
from deeptagger.models.encoder import Encoder
from deeptagger.models.encoder_decoder import EncoderDecoder
from deeptagger.models.generator import Generator
from deeptagger.modules.multi_headed_attention import MultiHeadedAttention
from deeptagger.modules.pointwise_ffn import PositionwiseFeedForward
from deeptagger.modules.positional_embedding import PositionalEmbedding
from deeptagger.modules.scorer import DotProductScorer
from deeptagger.modules.transformer_layer import EncoderLayer, DecoderLayer


class Transformer(nn.Module):
    """Make a transformer model."""

    def __init__(self,
                 source_vocab_size,
                 target_vocab_size,
                 nb_layers=6,
                 hidden_size=512,
                 ff_hidden_size=2048,
                 nb_heads=8,
                 max_seq_len=5000,
                 dropout_encoder=0.1,
                 dropout_decoder=0.1,
                 dropout_emb=0.1):
        super().__init__()
        query_size = key_size = value_size = hidden_size

        encoder_emb = PositionalEmbedding(source_vocab_size, hidden_size, max_seq_len=max_seq_len, dropout=dropout_emb)
        encoder_scorer = DotProductScorer()
        encoder_attn = MultiHeadedAttention(encoder_scorer, nb_heads, query_size, key_size, value_size, hidden_size,
                                            dropout=dropout_encoder)
        encoder_ff = PositionwiseFeedForward(hidden_size, ff_hidden_size, dropout=dropout_encoder)
        encoder_layer = EncoderLayer(hidden_size, encoder_attn, encoder_ff, dropout=dropout_encoder)
        encoder = Encoder(encoder_layer, nb_layers=nb_layers)

        decoder_emb = PositionalEmbedding(target_vocab_size, hidden_size, max_seq_len=max_seq_len, dropout=dropout_emb)
        decoder_scorer = DotProductScorer()
        decoder_attn1 = MultiHeadedAttention(decoder_scorer, nb_heads, query_size, key_size, value_size, hidden_size,
                                             dropout=dropout_decoder)
        decoder_attn2 = MultiHeadedAttention(decoder_scorer, nb_heads, query_size, key_size, value_size, hidden_size,
                                             dropout=dropout_decoder)
        decoder_ff = PositionwiseFeedForward(hidden_size, ff_hidden_size, dropout=dropout_decoder)
        decoder_layer = DecoderLayer(hidden_size, decoder_attn1, decoder_attn2, decoder_ff, dropout=dropout_decoder)
        decoder = Decoder(decoder_layer, nb_layers=nb_layers)

        generator = Generator(hidden_size, target_vocab_size)

        self.encdec = EncoderDecoder(encoder, decoder, encoder_emb, decoder_emb, generator)
        self._init_params()

    def _init_params(self):
        for p in self.encdec.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.encdec(src, tgt, src_mask, tgt_mask)

    def encode(self, src, src_mask):
        return self.encdec.encode(src, src_mask)

    def decode(self, memory, tgt, src_mask, tgt_mask):
        return self.encdec.decode(memory, tgt, src_mask, tgt_mask)


if __name__ == '__main__':
    batch_size = 8
    source_len = 7
    target_len = 3
    source_vocab_size = 100
    target_vocab_size = 90

    tf = Transformer(source_vocab_size, target_vocab_size)
