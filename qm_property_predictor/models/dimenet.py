import argparse
import os
import os.path as osp
from math import pi as PI
from math import sqrt
from typing import Any, Callable, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding, Linear
from torch_geometric.data import Data, Dataset, download_url
from torch_geometric.data.makedirs import makedirs
from torch_geometric.nn import NNConv, global_add_pool, radius_graph
from torch_geometric.nn.acts import swish
from torch_geometric.nn.inits import glorot_orthogonal
from torch_scatter import scatter
from torch_sparse import SparseTensor

from .mm_utils import BesselBasisLayer, SphericalBasisLayer

HIDDEN_CHANNELS = 128
OUT_CHANNELS = 1
NUM_BLOCKS = 6
NUM_BILINEAR = 8
NUM_SPHERICAL = 7
NUM_RADIAL = 6
CUTOFF = 5.0
ENVELOPE_EXPONENT = 5
NUM_BEFORE_SKIP = 1
NUM_AFTER_SKIP = 2
NUM_OUTPUT_LAYERS = 3
MAX_NUM_NEIGHBORS = 32

qm9_target_dict = {
    0: "mu",
    1: "alpha",
    2: "homo",
    3: "lumo",
    5: "r2",
    6: "zpve",
    7: "U0",
    8: "U",
    9: "H",
    10: "G",
    11: "Cv",
}


class EmbeddingBlock(torch.nn.Module):
    def __init__(self, num_radial, hidden_channels, act=swish):
        super().__init__()
        self.act = act

        self.emb = Embedding(95, hidden_channels)
        self.lin_rbf = Linear(num_radial, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, rbf, i, j):
        x = self.emb(x)
        rbf = self.act(self.lin_rbf(rbf))
        return self.act(self.lin(torch.cat([x[i], x[j], rbf], dim=-1)))


class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels, act=swish):
        super().__init__()
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class InteractionBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        num_bilinear,
        num_spherical,
        num_radial,
        num_before_skip,
        num_after_skip,
        act=swish,
    ):
        super().__init__()
        self.act = act

        self.lin_rbf = Linear(num_radial, hidden_channels, bias=False)
        self.lin_sbf = Linear(num_spherical * num_radial, num_bilinear, bias=False)

        # Dense transformations of input messages.
        self.lin_kj = Linear(hidden_channels, hidden_channels)
        self.lin_ji = Linear(hidden_channels, hidden_channels)

        self.W = torch.nn.Parameter(torch.Tensor(hidden_channels, num_bilinear, hidden_channels))

        self.layers_before_skip = torch.nn.ModuleList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)]
        )
        self.lin = Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)]
        )

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)
        self.W.data.normal_(mean=0, std=2 / self.W.size(0))
        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

    def forward(self, x, rbf, sbf, idx_kj, idx_ji):
        rbf = self.lin_rbf(rbf)
        sbf = self.lin_sbf(sbf)

        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))
        x_kj = x_kj * rbf
        x_kj = torch.einsum("wj,wl,ijl->wi", sbf, x_kj[idx_kj], self.W)
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0))

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h


class OutputBlock(torch.nn.Module):
    def __init__(self, num_radial, hidden_channels, out_channels, num_layers, act=swish):
        super().__init__()
        self.act = act

        self.lin_rbf = Linear(num_radial, hidden_channels, bias=False)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.lin = Linear(hidden_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        self.lin.weight.data.fill_(0)

    def forward(self, x, rbf, i, num_nodes=None):
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes)
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)


class DimeNet(torch.nn.Module):
    r"""The directional message passing neural network (DimeNet) from the
    `"Directional Message Passing for Molecular Graphs"
    <https://arxiv.org/abs/2003.03123>`_ paper.
    DimeNet transforms messages based on the angle between them in a
    rotation-equivariant fashion.

    .. note::

        For an example of using a pretrained DimeNet variant, see
        `examples/qm9_pretrained_dimenet.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        qm9_pretrained_dimenet.py>`_.

    Args:
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of building blocks.
        num_bilinear (int): Size of the bilinear layer tensor.
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff: (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        envelope_exponent (int, optional): Shape of the smooth cutoff.
            (default: :obj:`5`)
        num_before_skip: (int, optional): Number of residual layers in the
            interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip: (int, optional): Number of residual layers in the
            interaction blocks after the skip connection. (default: :obj:`2`)
        num_output_layers: (int, optional): Number of linear layers for the
            output blocks. (default: :obj:`3`)
        act: (Callable, optional): The activation funtion.
            (default: :obj:`swish`)
    """

    def __init__(
        self,
        args: argparse.Namespace,
    ) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.max_num_neighbors = self.args.get("max_num_neighbors", MAX_NUM_NEIGHBORS)
        self.cutoff = self.args.get("cutoff", CUTOFF)
        num_blocks = self.args.get("num_blocks", NUM_BLOCKS)
        envelope_exponent = self.args.get("envelope_exponent", ENVELOPE_EXPONENT)
        num_radial = self.args.get("num_radial", NUM_RADIAL)
        num_spherical = self.args.get("num_spherical", NUM_SPHERICAL)
        hidden_channels = self.args.get("hidden_channels", HIDDEN_CHANNELS)
        out_channels = self.args.get("out_channels", OUT_CHANNELS)
        num_output_layers = self.args.get("num_output_layers", NUM_OUTPUT_LAYERS)
        num_bilinear = self.args.get("num_bilinear", NUM_BILINEAR)
        num_before_skip = self.args.get("num_before_skip", NUM_BEFORE_SKIP)
        num_after_skip = self.args.get("num_after_skip", NUM_AFTER_SKIP)
        act = swish
        self.rbf = BesselBasisLayer("DimeNet", num_radial, self.cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(
            "DimeNet", num_spherical, num_radial, self.cutoff, envelope_exponent
        )

        self.emb = EmbeddingBlock(num_radial, hidden_channels, act)

        self.output_blocks = torch.nn.ModuleList(
            [
                OutputBlock(num_radial, hidden_channels, out_channels, num_output_layers, act)
                for _ in range(num_blocks + 1)
            ]
        )

        self.interaction_blocks = torch.nn.ModuleList(
            [
                InteractionBlock(
                    hidden_channels,
                    num_bilinear,
                    num_spherical,
                    num_radial,
                    num_before_skip,
                    num_after_skip,
                    act,
                )
                for _ in range(num_blocks)
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.rbf.reset_parameters()
        self.emb.reset_parameters()
        for out in self.output_blocks:
            out.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()

    def triplets(self, edge_index, num_nodes):
        row, col = edge_index  # j->i

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes))
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Node indices (k->j->i) for triplets.
        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k  # Remove i == k triplets.
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

        # Edge indices (k-j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji = adj_t_row.storage.row()[mask]

        return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji

    def forward(self, data: Data) -> torch.Tensor:
        batch, z, pos = (data.batch, data.z, data.pos)
        edge_index = radius_graph(
            pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors
        )
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(edge_index, num_nodes=z.size(0))

        # Calculate distances.
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        # Calculate angles.
        pos_i = pos[idx_i]
        pos_ji, pos_ki = pos[idx_j] - pos_i, pos[idx_k] - pos_i
        a = (pos_ji * pos_ki).sum(dim=-1)
        b = torch.cross(pos_ji, pos_ki).norm(dim=-1)
        angle = torch.atan2(b, a)
        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)
        x = self.emb(z, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

        # Interaction blocks.
        for interaction_block, output_block in zip(
            self.interaction_blocks, self.output_blocks[1:]
        ):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P += output_block(x, rbf, i)

        return P.sum(dim=0) if batch is None else scatter(P, batch, dim=0)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--hidden_channels", type=int, default=HIDDEN_CHANNELS)
        parser.add_argument("--out_channels", type=int, default=OUT_CHANNELS)
        parser.add_argument("--num_blocks", type=int, default=NUM_BLOCKS)
        parser.add_argument("--num_bilinear", type=int, default=NUM_BILINEAR)
        parser.add_argument("--num_spherical", type=int, default=NUM_SPHERICAL)
        parser.add_argument("--num_radial", type=int, default=NUM_RADIAL)
        parser.add_argument("--cutoff", type=float, default=CUTOFF)
        parser.add_argument("--envelope_exponent", type=int, default=ENVELOPE_EXPONENT)
        parser.add_argument("--num_before_skip", type=int, default=NUM_BEFORE_SKIP)
        parser.add_argument("--num_after_skip", type=int, default=NUM_AFTER_SKIP)
        parser.add_argument("--num_output_layers", type=int, default=NUM_OUTPUT_LAYERS)
        return parser
