import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch.nn import Linear
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    torch_sparse,
)
from torch import Tensor

class LIMLP(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, num_layers, dropout=0.0, tail_activation=False, activation=nn.ReLU(inplace=True)):
        super().__init__()
        layers = []
        if num_layers == 1:
            layers.append(Linear(input_channels, output_channels))
            if tail_activation:
                layers.append(activation)
        else:
            layers.append(Linear(input_channels, hidden_channels))
            for _ in range(num_layers - 2):
                layers.append(activation)
                layers.append(nn.Dropout(dropout))
                layers.append(Linear(hidden_channels, hidden_channels))
            layers.append(activation)
            layers.append(nn.Dropout(dropout))
            layers.append(Linear(hidden_channels, output_channels))
            if tail_activation:
                layers.append(activation)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class LILinear(nn.Module):
    def __init__(self, in_features, out_features, origin_infeat, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.trans_src = Linear(origin_infeat, in_features)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, src):
        t_src = self.trans_src(src)
        trans_input = x * t_src
        out = F.linear(trans_input, self.weight, self.bias)
        return out

class LASAGESConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggregator_type, origin_infeat, num_trans, activation=nn.ReLU()):
        super().__init__(aggr=aggregator_type)  # "mean" aggregation
        self.fc_neigh_benign = LIMLP(in_channels, in_channels, out_channels, num_trans, activation=activation)
        self.fc_neigh_fraud = LIMLP(in_channels, in_channels, out_channels, num_trans, activation=activation)
        self.fc_balance = Linear(in_channels, 1)
        self.balance_w = nn.Sigmoid()
        self.activation = activation
        self.fc_self = Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        h_self = self.fc_self(x)
        return h_self + out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out, x):
        neigh_fr = self.fc_neigh_fraud(aggr_out)
        neigh_be = self.fc_neigh_benign(aggr_out)
        balance = self.balance_w(self.fc_balance(x))
        return self.activation(neigh_fr * balance + neigh_be * (1 - balance))

class LASAGE_S(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, num_trans=1, aggregator_type="mean"):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(LASAGESConv(in_channels, hidden_channels, aggregator_type, origin_infeat=in_channels, num_trans=num_trans))
            elif i == num_layers - 1:
                self.layers.append(LASAGESConv(hidden_channels, out_channels, aggregator_type, origin_infeat=in_channels, num_trans=num_trans))
            else:
                self.layers.append(LASAGESConv(hidden_channels, hidden_channels, aggregator_type, origin_infeat=in_channels, num_trans=num_trans))
        self.dropout = dropout
        self.linear1 = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, batch=None):
        initial_x = self.linear1(x)
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if len(x[0]) == 32:
                x32 =x
        x = self.layers[-1](x, edge_index)
        if batch is not None:
            x = global_mean_pool(x, batch)
        return x32, x
