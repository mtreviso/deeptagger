import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from deeptagger import constants
from deeptagger.models.model import Model


class SimpleLSTM(Model):
    """Just a regular LSTM network

    TODO: add references.

    """

    def __init__(
            self,
            words_field,
            tags_field,
            prefixes_field=None,
            suffixes_field=None,
            caps_field=None,
            **kwargs):
        """."""
        super().__init__(**kwargs)

        # Default fields and embeddings
        self.words_field = words_field
        self.tags_field = tags_field
        self.word_embeddings = words_field.vocab.vectors

        # Extra features
        self.prefixes_field = prefixes_field
        self.suffixes_field = suffixes_field
        self.caps_field = caps_field

        # loss
        self._loss = nn.NLLLoss(weight=self._loss_weights,
                                ignore_index=self._loss_ignore_index)

    def build(
        self,
        word_embeddings_size=100,
        prefix_embeddings_size=20,
        suffix_embeddings_size=20,
        caps_embeddings_size=5,
        hidden_size=100,
        dropout=0.5,
        emb_dropout=0.4,
        bidirectional=True,
        sum_bidir=False,
        freeze_embeddings=False,
    ):
        """."""
        if self.word_embeddings is not None:
            word_embeddings_size = self.word_embeddings.size(1)

        self.word_emb = nn.Embedding(
            num_embeddings=self.word_embeddings.size(0),
            embedding_dim=word_embeddings_size,
            padding_idx=self.words_padding_idx,
            _weight=self.word_embeddings,
        )

        if freeze_embeddings:
            self.word_emb.weight.requires_grad = False
            self.word_emb.bias.requires_grad = False

        self.is_bidir = bidirectional
        self.sum_bidir = sum_bidir
        self.gru = nn.LSTM(word_embeddings_size,
                           hidden_size,
                           bidirectional=bidirectional,
                           batch_first=True)
        self.hidden = None

        n = 2 if self.is_bidir else 1
        n = 1 if self.sum_bidir else n
        self.linear_out = nn.Linear(n * hidden_size, self.nb_classes)

        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.Relu()
        self.dropout_emb = nn.Dropout(emb_dropout)
        self.dropout_gru = nn.Dropout(dropout)

        self.init_weights()

        # # Set model to a specific gpu device
        if self.device is not None:
            torch.cuda.set_device(self.device)
            self.cuda()

        self.is_built = True

    def init_weights(self):
        pass

    def init_hidden(self, batch_size, hidden_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        num_layers = 2 if self.is_bidir else 1
        return (torch.zeros(num_layers, batch_size, hidden_size),
                torch.zeros(num_layers, batch_size, hidden_size))

    def forward(self, batch):
        assert self.is_built

        # (ts, bs) -> (bs, ts)
        h = batch.words
        mask = h != constants.PAD_ID
        lengths = mask.int().sum(dim=-1)

        # initialize GRU hidden state
        self.hidden = self.init_hidden(batch.words.shape[0],
                                       self.gru.hidden_size)

        # (bs, ts) -> (bs, ts, emb_dim)
        h = self.word_emb(h)
        h = self.dropout_emb(h)

        # (bs, ts, pool_size) -> (bs, ts, hidden_size)
        h = pack(h, lengths, batch_first=True)
        h, self.hidden = self.gru(h, self.hidden)
        h, _ = unpack(h, batch_first=True)

        # if you'd like to sum instead of concatenate:
        if self.sum_bidir:
            h = (h[:, :, :self.gru.hidden_size] +
                 h[:, :, self.gru.hidden_size:])

        h = self.dropout_gru(h)

        # (bs, ts, hidden_size) -> (bs, ts, nb_classes)
        h = F.log_softmax(self.linear_out(h), dim=-1)

        # remove <bos> and <eos> tokens
        # (bs, ts, nb_classes) -> (bs, ts-2, nb_classes)
        h = h[:, 1:-1, :]

        return h
