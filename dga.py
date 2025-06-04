import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.data import Data
import torch_scatter

class IntraConv(MessagePassing):
    def __init__(self, in_feats, out_feats, aggregator_type='mean', feat_drop=0., add_self=True, norm=None,
                 activation=None, **kwargs):
        super(IntraConv, self).__init__(aggr=aggregator_type, **kwargs)  # Aggregator type: "mean"
        self.fc_self = nn.Linear(in_feats, out_feats)
        self.fc_neigh = nn.Linear(in_feats, out_feats)
        self.add_self = add_self
        self.dropout = nn.Dropout(feat_drop)
        self.norm = norm
        self.activation = activation

    def forward(self, x, edge_index):
        x = self.dropout(x)
        if self.add_self:
            x_self = self.fc_self(x)
        else:
            x_self = 0

        # Perform message passing
        out = self.propagate(edge_index, x=x)
        out = self.fc_neigh(out)
        out = out + x_self

        if self.norm is not None:
            out = self.norm(out)
        if self.activation is not None:
            out = self.activation(out)

        return out

    def message(self, x_j):
        return x_j

    def aggregate(self, inputs, index):
        return torch_scatter.scatter_mean(inputs, index, dim=0)


class DGA(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers=2, dropout=0.3):
        super(DGA, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                IntraConv(in_feats if _ == 0 else hidden_feats, hidden_feats, feat_drop=dropout, activation=F.relu)
            )
        self.fc_out = nn.Linear(hidden_feats, out_feats)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(in_feats, out_feats)

    def forward(self, x, edge_index):
        initial_x = self.linear1(x)
        for layer in self.layers:
            x = layer(x, edge_index)
            x = self.dropout(x)
            if len(x[0]) == 32:
                x32 =x
        return x32, self.fc_out(x)

