import torch
import torch.nn as nn


class GRUGate(nn.Module):
    def __init__(self, input_dim: int, bg: float = 0.0):
        super(GRUGate, self).__init__()
        """
        Overview:
            Init GRU
        Arguments:
            - input_dim: (`int`): input dimension.
            - bg: (`float`): bias

        """

        self.Wr = nn.Linear(input_dim, input_dim, bias=False)
        self.Ur = nn.Linear(input_dim, input_dim, bias=False)
        self.Wz = nn.Linear(input_dim, input_dim, bias=False)
        self.Uz = nn.Linear(input_dim, input_dim, bias=False)
        self.Wg = nn.Linear(input_dim, input_dim, bias=False)
        self.Ug = nn.Linear(input_dim, input_dim, bias=False)
        self.bg = nn.Parameter(torch.full([input_dim], bg))  # bias

        nn.init.xavier_uniform_(self.Wr.weight)
        nn.init.xavier_uniform_(self.Ur.weight)
        nn.init.xavier_uniform_(self.Wz.weight)
        nn.init.xavier_uniform_(self.Uz.weight)
        nn.init.xavier_uniform_(self.Wg.weight)
        nn.init.xavier_uniform_(self.Ug.weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Overview:
            Forward method
        Arguments:
            - x: (`torch.Tensor`): input.
            - y: ('torch.Tensor`): hidden input.
        Return:
        """

        r = nn.Sigmoid()(self.Wr(y) + self.Ur(x))
        z = nn.Sigmoid()(self.Wz(y) + self.Uz(x) - self.bg)
        h = nn.Tanh()(self.Wg(y) + self.Ug(torch.mul(r, x)))
        return torch.mul(1 - z, x) + torch.mul(z, h)