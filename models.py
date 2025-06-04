import torch
import torch_geometric.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.linear1 = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        initial_x = self.linear1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x=x, edge_index=edge_index).relu()
        x32 = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x=x, edge_index=edge_index)
        return x32, x + initial_x

class DualGNN(torch.nn.Module):
    def __init__(self,  out_channels, gnn):
        super().__init__()
        self.gnn = gnn
        self.attention_weights = torch.nn.Parameter(torch.Tensor(1, out_channels))
        torch.nn.init.xavier_uniform_(self.attention_weights)
        self.linear1 = torch.nn.Linear(384, out_channels)

    def forward(self, x1, x2, edge_index):
        x321, out1 = self.gnn(x1, edge_index)
        x322, out2 = self.gnn(x2, edge_index)
        concat32 = torch.stack([x321, x322], dim=1)
        concat = torch.stack([out1, out2], dim=1)
        attn_scores = F.softmax(self.attention_weights, dim=1)
        out32 = torch.matmul(attn_scores, concat32)
        weighted_out = torch.matmul(attn_scores, concat)
        return out32.squeeze(1), weighted_out.squeeze(1)

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.linear1 = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        initial_x = self.linear1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x=x, edge_index=edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x=x, edge_index=edge_index)
        return x

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, hidden_channels, 8)
        self.conv2 = SAGEConv(8 * hidden_channels, out_channels)
        self.linear1 = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        initial_x = self.linear1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x=x, edge_index=edge_index).relu()
        x32 = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x=x, edge_index=edge_index)
        return x32, x


class CaGCN(torch.nn.Module):
    def __init__(self, base_model, out_channels, hidden_channels):
        super(CaGCN, self).__init__()
        self.base_model = base_model
        self.conv1 = GCNConv(out_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)

        for para in self.base_model.parameters():
            para.requires_grad = False

    def forward(self, x, edge_index, edge_weight=None):
        logist = self.base_model(x, edge_index, edge_weight)
        x = F.dropout(logist, p=0.5, training=self.training)
        x = self.conv1(x=x, edge_index=edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        temperature= self.conv2(x=x, edge_index=edge_index)
        temperature = torch.log(torch.exp(temperature) + torch.tensor(1.1))
        return logist * temperature