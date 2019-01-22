from torch import nn


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation with a variable activation, residual and
    layer norm."""

    def __init__(self, size, hidden_size, dropout=0.1, activation=nn.ReLU()):
        super().__init__()
        self.w_1 = nn.Linear(size, hidden_size)
        self.w_2 = nn.Linear(hidden_size, size)
        self.layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.dropout_in = nn.Dropout(dropout)
        self.dropout_out = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x):
        z = self.dropout_in(self.activation(self.w_1(self.layer_norm(x))))
        z = self.dropout_out(self.w_2(z))
        z = z + x
        return z


if __name__ == '__main__':
    import torch

    x = torch.randn(2, 3, 4)
    pff = PositionwiseFeedForward(4, 100, dropout=0.5, activation=nn.SELU())
    print(pff(x).summary())
