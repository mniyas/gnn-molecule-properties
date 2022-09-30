import argparse
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import NNConv, global_add_pool

NUM_NODE_FEAT = 11
NUM_EDGE_FEAT = 4


class MPNN(torch.nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        num_node_features = self.args.get("num_node_features", NUM_NODE_FEAT)
        num_edge_features = self.args.get("num_edge_features", NUM_EDGE_FEAT)
        conv1_net = nn.Sequential(
            nn.Linear(num_edge_features, 32),
            nn.ReLU(),
            nn.Linear(32, num_node_features * 32),
        )
        conv2_net = nn.Sequential(
            nn.Linear(num_edge_features, 32), nn.ReLU(), nn.Linear(32, 32 * 16)
        )
        self.conv1 = NNConv(num_node_features, 32, conv1_net)
        self.conv2 = NNConv(32, 16, conv2_net)
        self.fc_1 = nn.Linear(16, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Parameters
        ----------
        data
            Torch Geometric Data object with attributes: batch, x, edge_index, edge_attr

        Returns
        -------
        torch.Tensor
            of dimensions (B, 1)
        """
        batch, x, edge_index, edge_attr = (
            data.batch,
            data.x,
            data.edge_index,
            data.edge_attr,
        )
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = global_add_pool(x, batch)
        x = F.relu(self.fc_1(x))
        return self.out(x)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--num_node_features", type=int, default=NUM_NODE_FEAT)
        parser.add_argument("--num_edge_features", type=int, default=NUM_EDGE_FEAT)
        return parser
