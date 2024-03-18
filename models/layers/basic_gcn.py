import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


class BasicGCN(nn.Module):
    def __init__(self, edge_index, in_feats, out_feats, activation=None, dropout=0):
        super(BasicGCN, self).__init__()
        self.edge_index = edge_index
        self.dropout = dropout
        self.activation = activation
        self.conv1 = GCNConv(in_feats, out_feats)

    def forward(self, x):
        x = self.conv1(x, self.edge_index)
        if self.activation is not None:
            x = self.activation(x)
        if self.training:  # Apply dropout only during training
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
