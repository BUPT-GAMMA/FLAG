import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.utils import add_self_loops, degree
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.utils import subgraph, index_to_mask, k_hop_subgraph, mask_to_index

class CAREGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CAREGNNLayer, self).__init__(aggr='mean')  # Use mean aggregation for simplicity
        self.lin = Linear(in_channels, out_channels)
        self.mlp = Sequential(
            Linear(2 * in_channels, out_channels),
            ReLU(),
            Linear(out_channels, 1)
        )

    def forward(self, x, edge_index):
        # Add self-loops to the adjacency matrix (if needed)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i: central node, x_j: neighbor node
        # Compute the pairwise similarity between x_i and x_j
        pairwise_features = torch.cat([x_i, x_j], dim=1)
        similarity_score = torch.sigmoid(self.mlp(pairwise_features))
        return similarity_score * x_j  # Weighted by similarity score

    def update(self, aggr_out):
        # Apply linear transformation
        return self.lin(aggr_out)


class CAREGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(CAREGNN, self).__init__()
        self.conv1 = CAREGNNLayer(in_channels, hidden_channels)
        self.conv2 = CAREGNNLayer(hidden_channels, out_channels)
        self.linear1 = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        initial_x = self.linear1(x)
        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x32 = x
        # Second layer
        x = self.conv2(x, edge_index)
        return x32, x