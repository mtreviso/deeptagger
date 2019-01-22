from torch import nn
import torch.nn.functional as F

"""
Code based on The Annotated Transformer by Sasha Rush
"""


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
