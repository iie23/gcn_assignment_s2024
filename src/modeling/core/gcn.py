import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modeling.core.layers import GCNLayer


class GCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super(GCN, self).__init__()
        # TODO: add L layers of GCN
        L = 2
        self.layers = [GCNLayer(input_dim, hidden_dim)]
        for _ in range(1, L-1):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim))
        self.layers.append(GCNLayer(hidden_dim, output_dim))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.sparse_coo) -> torch.Tensor:
        # given the input node features, and the adjacency matrix, run GCN
        # The order of operations should roughly be:
        # 1. Apply the first GCN layer
        # 2. Apply Relu
        # 3. Apply Dropout
        # 4. Apply the second GCN layer

        relu = nn.ReLU()
        dropout = nn.Dropout(self.dropout)

        hidden = self.layers[0].forward(x, adj)
        hidden = relu(hidden)
        hidden = dropout(hidden)
        for i in range(1, len(self.layers)-1):
            hidden = self.layers[i].forward(hidden, adj)
            hidden = relu(hidden)
            hidden = dropout(hidden)
        output = self.layers[-1].forward(hidden, adj)

        return output
