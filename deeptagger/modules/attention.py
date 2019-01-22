import torch
from torch import nn
from torch.nn import functional as F

from deeptagger.models.utils import unsqueeze_as
from deeptagger.modules.scorer import (DotProductScorer, GeneralScorer,
                                       OperationScorer, MultiLayerScorer)


class Attention(nn.Module):
    """Generic Attention Implementation.
           Use query and keys to compute scores (energies), then apply
           softmax to scores and combine the returned probs with `values`
           by calculating a dot product between them
        """

    def __init__(self, scorer, dropout=0.0):
        super().__init__()
        self.scorer = scorer
        self.dropout = nn.Dropout(p=dropout)
        self.NEG_INF = -1e9

    def forward(self, query, keys, values, mask=None):
        # get scores (aka energies)
        scores = self.scorer(query, keys)

        # mask out scores to infinity before softmax
        if mask is not None:
            # expand dims at the end
            mask = unsqueeze_as(mask, scores, dim=-1)
            scores = scores.masked_fill(mask == 0, self.NEG_INF)

        # apply softmax to get probs
        p_attn = F.softmax(scores, dim=-1)

        # apply dropout if it is provided (used in Transformer)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)

        # dot product between p_attn and values
        # o_attn = torch.matmul(p_attn, values)
        o_attn = torch.einsum('b...ts,b...sm->b...tm', [p_attn, values])
        return o_attn, p_attn


if __name__ == '__main__':
    from deeptagger.models.utils import sequence_mask

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    batch_size = 4
    source_len = 7
    target_len = 3
    query_size = 10
    keys_size = 20
    attn_size = 15

    # query vectors
    q = torch.randn(batch_size, query_size)

    # set of query vectors
    qs = torch.randn(batch_size, target_len, query_size)

    # keys vectors (a key vector for each encoder word)
    ks = torch.randn(batch_size, source_len, keys_size)

    # keys vectors with same size as query vectors
    kq = torch.randn(batch_size, source_len, query_size)

    # values vectors (same shape as keys)
    vs = torch.randn(batch_size, source_len, keys_size)

    # values vectors with same size as query vectors
    vq = torch.randn(batch_size, source_len, query_size)

    # self attention on target (decoder)
    attn = Attention(DotProductScorer(), dropout=0.1)
    out, probs = attn(qs, qs, qs)
    assert (list(out.shape) == list(qs.shape))
    assert (list(probs.shape) == [batch_size, qs.shape[1], qs.shape[1]])

    # self attention on source (encoder)
    attn = Attention(DotProductScorer(), dropout=0.1)
    out, probs = attn(ks, ks, ks)
    assert (list(out.shape) == list(ks.shape))
    assert (list(probs.shape) == [batch_size, ks.shape[1], ks.shape[1]])

    # masked self attention on target (decoder)
    mask = sequence_mask(torch.LongTensor([2, 1, 2, 3]))
    attn = Attention(DotProductScorer(), dropout=0.1)
    out, probs = attn(qs, qs, qs, mask=mask)
    assert (list(out.shape) == list(qs.shape))
    assert (list(probs.shape) == [batch_size, qs.shape[1], qs.shape[1]])

    # masked self attention on source (encoder)
    mask = sequence_mask(torch.LongTensor([5, 3, 7, 4]))
    attn = Attention(DotProductScorer(), dropout=0.1)
    out, probs = attn(ks, ks, ks, mask=mask)
    assert (list(out.shape) == list(ks.shape))
    assert (list(probs.shape) == [batch_size, ks.shape[1], ks.shape[1]])

    # decoder attend to encoder - multiplicative attention
    attn = Attention(GeneralScorer(query_size, keys_size), dropout=0.1)
    out, probs = attn(qs, ks, ks)
    assert (list(out.shape) == [batch_size, qs.shape[1], ks.shape[-1]])
    assert (list(probs.shape) == [batch_size, qs.shape[1], ks.shape[1]])

    # decoder attend to encoder - concat attention
    attn = Attention(OperationScorer(query_size, keys_size, attn_size,
                                     op='concat'), dropout=0.1)
    out, probs = attn(qs, ks, ks)
    assert (list(out.shape) == [batch_size, qs.shape[1], ks.shape[-1]])
    assert (list(probs.shape) == [batch_size, qs.shape[1], ks.shape[1]])

    # decoder attend to encoder using a mlp with two hidden layers of
    # 5 neurons each
    attn = Attention(MultiLayerScorer(query_size, keys_size,
                                      layer_sizes=[5, 5]), dropout=0.1)
    out, probs = attn(qs, ks, ks)
    assert (list(out.shape) == [batch_size, qs.shape[1], ks.shape[-1]])
    assert (list(probs.shape) == [batch_size, qs.shape[1], ks.shape[1]])
