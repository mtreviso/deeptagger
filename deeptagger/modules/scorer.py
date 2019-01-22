import numpy as np
import torch
from torch import nn

from deeptagger.models.utils import make_mergeable_tensors


class Scorer(nn.Module):
    """Score function for Attention module."""

    def __init__(self, scaled=True):
        super().__init__()
        self.scaled = scaled

    def scale(self, hidden_size):
        return np.sqrt(hidden_size) if self.scaled else 1

    def forward(self, query, keys):
        """Computes Scores for each key of size n given the query of size m.
         query:  bs x ... x target_len x m
         keys:   bs x ... x source_len x n
         return: bs x ... x target_len x source_len
        """
        raise NotImplementedError


class DotProductScorer(Scorer):
    """Implement DotProduct function for attention.
       Query and keys should have the same size.
    """

    def forward(self, query, keys):
        assert (keys.shape[-1] == query.shape[-1])
        scale = self.scale(keys.shape[-1])
        # score = torch.matmul(query, keys.transpose(-1, -2))
        score = torch.einsum('b...tx,b...sx->b...ts', [query, keys])
        return score / scale


class GeneralScorer(Scorer):
    """Implement GeneralScorer (aka Multiplicative) for attention"""

    def __init__(self, query_size, key_size, **kwargs):
        super().__init__(**kwargs)
        self.W = nn.Parameter(torch.randn(query_size, key_size))

    def forward(self, query, keys):
        scale = self.scale(max(self.W.shape))
        # score = torch.matmul(torch.matmul(query, self.W), keys.transpose(-1, -2))
        score = torch.einsum('b...tm,mn,b...sn->b...ts', [query, self.W, keys])
        return score / scale


class OperationScorer(Scorer):
    """Base class for ConcatScorer and AdditiveScorer"""

    def __init__(self, query_size, key_size, attn_hidden_size,
                 op='concat', activation=nn.Tanh, **kwargs):
        super().__init__(**kwargs)
        assert op in ['concat', 'add', 'mul']
        self.op = op
        self.activation = activation()
        self.W1 = nn.Parameter(torch.randn(key_size, attn_hidden_size))
        self.W2 = nn.Parameter(torch.randn(query_size, attn_hidden_size))
        if self.op == 'concat':
            self.v = nn.Parameter(torch.randn(2 * attn_hidden_size))
        else:
            self.v = nn.Parameter(torch.randn(attn_hidden_size))

    def f(self, x1, x2):
        """Perform an operation on x1 and x2"""
        if self.op == 'add':
            x = x1 + x2
        elif self.op == 'mul':
            x = x1 * x2
        else:
            x = torch.cat((x1, x2), dim=-1)
        return self.activation(x)

    def forward(self, query, keys):
        scale = self.scale(max(*self.W1.shape, *self.W2.shape))
        # x1 = torch.matmul(keys, self.W1)
        # x2 = torch.matmul(query, self.W2)
        x1 = torch.einsum('b...sn,nh->b...sh', [keys, self.W1])
        x2 = torch.einsum('b...tm,mh->b...th', [query, self.W2])
        x1, x2 = make_mergeable_tensors(x1, x2)
        # score = torch.matmul(self.f(x1, x2), self.v)
        score = torch.einsum('b...tsh,h->b...ts', [self.f(x1, x2), self.v])
        return score / scale


class MultiLayerScorer(Scorer):
    """MultiLayerPerceptron Scorer with variable nb o layers and neurons."""

    def __init__(self, query_size, key_size,
                 layer_sizes=None, activation=nn.Tanh, **kwargs):
        super().__init__(**kwargs)
        if layer_sizes is None:
            layer_sizes = [min(query_size, key_size) // 2]
        input_size = query_size + key_size
        output_size = 1
        layer_sizes = [input_size] + layer_sizes + [output_size]
        sizes = zip(layer_sizes[:-1], layer_sizes[1:])
        layers = []
        for n_in, n_out in sizes:
            layers.append(nn.Linear(n_in, n_out))
            layers.append(activation())
        self.mlp = nn.ModuleList(layers)

    def forward(self, query, keys):
        x1, x2 = make_mergeable_tensors(keys, query)
        x = torch.cat((x1, x2), dim=-1)
        for layer in self.mlp:
            x = layer(x)
        return x.squeeze(-1)  # remove last dimension


if __name__ == '__main__':
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    batch_size = 4
    source_len = 7
    target_len = 3
    query_size = 10
    keys_size = 20
    attn_size = 15

    # query vectors
    q = torch.randn(batch_size, target_len, query_size)

    # keys vectors (a key vector for each encoder word)
    ks = torch.randn(batch_size, source_len, keys_size)

    # keys vectors with same size as query vectors
    kq = torch.randn(batch_size, source_len, query_size)

    out = DotProductScorer()(q, kq)
    assert (list(out.shape) == [batch_size, q.shape[1], kq.shape[1]])

    out = GeneralScorer(query_size, keys_size)(q, ks)
    assert (list(out.shape) == [batch_size, q.shape[1], ks.shape[1]])

    out = OperationScorer(query_size, keys_size, attn_size, op='add')(q, ks)
    assert (list(out.shape) == [batch_size, q.shape[1], ks.shape[1]])

    out = OperationScorer(query_size, keys_size, attn_size, op='mul')(q, ks)
    assert (list(out.shape) == [batch_size, q.shape[1], ks.shape[1]])

    out = OperationScorer(query_size, keys_size, attn_size, op='concat')(q, ks)
    assert (list(out.shape) == [batch_size, q.shape[1], ks.shape[1]])

    out = OperationScorer(query_size, query_size, attn_size, op='add')(q, q)
    assert (list(out.shape) == [batch_size, q.shape[1], q.shape[1]])

    out = MultiLayerScorer(query_size, keys_size, layer_sizes=[10, 5, 5], activation=nn.Sigmoid)(q, ks)
    assert (list(out.shape) == [batch_size, q.shape[1], ks.shape[1]])

    out = MultiLayerScorer(query_size, keys_size, layer_sizes=[10, 5, 5])(q, ks)
    assert (list(out.shape) == [batch_size, q.shape[1], ks.shape[1]])
