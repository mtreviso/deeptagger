import torch
from torch import nn


class LayerNorm(nn.Module):
    """Construct a layer normalization module.
    https://arxiv.org/abs/1607.06450

    It normalizes the outputs of neurons for a given layer:
    out = (gamma * (x - x.mean(-1)) / (x.std(-1) + eps)) + beta

    Args:
        hidden_size (int): number of neurons in the layer x
        eps (float): factor to prevent division by zero
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
