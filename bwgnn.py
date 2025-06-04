import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
import sympy
import scipy

class PolyConv(MessagePassing):
    def __init__(self, in_feats, out_feats, theta, activation=F.leaky_relu, lin=False, bias=False):
        super(PolyConv, self).__init__(aggr='add')  # Use add aggregation for simplicity
        self._theta = theta
        self._k = len(self._theta)
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)
        self.activation = activation
        self.lin = lin

    def forward(self, x, edge_index):
        # Initialize h with the first term of the polynomial
        h = self._theta[0] * x
        for k in range(1, self._k):
            # Apply the polynomial recursion Laplacian
            x = self.propagate(edge_index, x=x)
            h += self._theta[k] * x

        if self.lin:
            h = self.linear(h)
            h = self.activation(h)

        return h

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out


class BWGNN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, d=2):
        super(BWGNN, self).__init__()
        self.thetas = self.calculate_theta2(d=d)
        self.conv = nn.ModuleList([PolyConv(h_feats, h_feats, theta, lin=False) for theta in self.thetas])

        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats * len(self.conv), h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.linear1 = nn.Linear(in_feats, num_classes)

        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        initial_x = self.linear1(x)
        h = self.linear(x)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)

        h_final = []
        for conv in self.conv:
            h_conv = conv(h, edge_index)
            h_final.append(h_conv)

        h_final = torch.cat(h_final, dim=-1)
        h = self.linear3(h_final)
        h = self.act(h)
        x32 = h
        h = self.linear4(h)

        return x32, h + initial_x

    @staticmethod
    def calculate_theta2(d):
        """Calculate the coefficients for the polynomials."""
        thetas = []
        x = sympy.symbols('x')
        for i in range(d + 1):
            f = sympy.poly((x / 2) ** i * (1 - x / 2) ** (d - i) / (scipy.special.beta(i + 1, d + 1 - i)))
            coeff = f.all_coeffs()
            inv_coeff = [float(coeff[d - i]) for i in range(d + 1)]
            thetas.append(inv_coeff)
        return thetas
