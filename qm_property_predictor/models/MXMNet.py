import argparse
import math

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_add_pool, radius_graph
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_scatter import scatter
from torch_sparse import SparseTensor

from .mm_utils import BesselBasisLayer, SphericalBasisLayer

DIM = 128
N_LAYER = 6
CUTOFF = 5.0
NUM_SPHERICAL = 7
NUM_RADIAL = 6
ENVELOPE_EXPONENT = 5


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def __call__(self, model, num_updates=99999):
        decay = min(self.decay, (1.0 + num_updates) / (10.0 + num_updates))
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)


def MLP(channels):
    return nn.Sequential(
        *[
            nn.Sequential(nn.Linear(channels[i - 1], channels[i]), SiLU())
            for i in range(1, len(channels))
        ]
    )


class Residual(nn.Module):
    def __init__(self, dim):
        super(Residual, self).__init__()

        self.mlp = MLP([dim, dim, dim])

    def forward(self, m):
        m1 = self.mlp(m)
        m_out = m1 + m
        return m_out


class Global_MP(MessagePassing):
    def __init__(self, dim):
        super(Global_MP, self).__init__()
        self.flow = "target_to_source"
        self.dim = dim

        self.h_mlp = MLP([self.dim, self.dim])

        self.res1 = Residual(self.dim)
        self.res2 = Residual(self.dim)
        self.res3 = Residual(self.dim)
        self.mlp = MLP([self.dim, self.dim])

        self.x_edge_mlp = MLP([self.dim * 3, self.dim])
        self.linear = nn.Linear(self.dim, self.dim, bias=False)

    def forward(self, h, edge_attr, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=h.size(0))

        res_h = h

        # Integrate the Cross Layer Mapping inside the Global Message Passing
        h = self.h_mlp(h)

        # Message Passing operation
        h = self.propagate(edge_index, x=h, num_nodes=h.size(0), edge_attr=edge_attr)

        # Update function f_u
        h = self.res1(h)
        h = self.mlp(h) + res_h
        h = self.res2(h)
        h = self.res3(h)

        # Message Passing operation
        h = self.propagate(edge_index, x=h, num_nodes=h.size(0), edge_attr=edge_attr)

        return h

    def message(self, x_i, x_j, edge_attr, edge_index, num_nodes):
        num_edge = edge_attr.size()[0]

        x_edge = torch.cat((x_i[:num_edge], x_j[:num_edge], edge_attr), -1)
        x_edge = self.x_edge_mlp(x_edge)

        x_j = torch.cat((self.linear(edge_attr) * x_edge, x_j[num_edge:]), dim=0)

        return x_j

    def update(self, aggr_out):

        return aggr_out


class Local_MP(torch.nn.Module):
    def __init__(self, dim):
        super(Local_MP, self).__init__()
        self.dim = dim

        self.h_mlp = MLP([self.dim, self.dim])

        self.mlp_kj = MLP([3 * self.dim, self.dim])
        self.mlp_ji_1 = MLP([3 * self.dim, self.dim])
        self.mlp_ji_2 = MLP([self.dim, self.dim])
        self.mlp_jj = MLP([self.dim, self.dim])

        self.mlp_sbf1 = MLP([self.dim, self.dim, self.dim])
        self.mlp_sbf2 = MLP([self.dim, self.dim, self.dim])
        self.lin_rbf1 = nn.Linear(self.dim, self.dim, bias=False)
        self.lin_rbf2 = nn.Linear(self.dim, self.dim, bias=False)

        self.res1 = Residual(self.dim)
        self.res2 = Residual(self.dim)
        self.res3 = Residual(self.dim)

        self.lin_rbf_out = nn.Linear(self.dim, self.dim, bias=False)

        self.h_mlp = MLP([self.dim, self.dim])

        self.y_mlp = MLP([self.dim, self.dim, self.dim, self.dim])
        self.y_W = nn.Linear(self.dim, 1)

    def forward(
        self, h, rbf, sbf1, sbf2, idx_kj, idx_ji_1, idx_jj, idx_ji_2, edge_index, num_nodes=None
    ):
        res_h = h

        # Integrate the Cross Layer Mapping inside the Local Message Passing
        h = self.h_mlp(h)

        # Message Passing 1
        j, i = edge_index
        m = torch.cat([h[i], h[j], rbf], dim=-1)

        m_kj = self.mlp_kj(m)
        m_kj = m_kj * self.lin_rbf1(rbf)
        m_kj = m_kj[idx_kj] * self.mlp_sbf1(sbf1)
        m_kj = scatter(m_kj, idx_ji_1, dim=0, dim_size=m.size(0), reduce="add")

        m_ji_1 = self.mlp_ji_1(m)

        m = m_ji_1 + m_kj

        # Message Passing 2       (index jj denotes j'i in the main paper)
        m_jj = self.mlp_jj(m)
        m_jj = m_jj * self.lin_rbf2(rbf)
        m_jj = m_jj[idx_jj] * self.mlp_sbf2(sbf2)
        m_jj = scatter(m_jj, idx_ji_2, dim=0, dim_size=m.size(0), reduce="add")

        m_ji_2 = self.mlp_ji_2(m)

        m = m_ji_2 + m_jj

        # Aggregation
        m = self.lin_rbf_out(rbf) * m
        h = scatter(m, i, dim=0, dim_size=h.size(0), reduce="add")

        # Update function f_u
        h = self.res1(h)
        h = self.h_mlp(h) + res_h
        h = self.res2(h)
        h = self.res3(h)

        # Output Module
        y = self.y_mlp(h)
        y = self.y_W(y)

        return h, y


class MXMNet(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(MXMNet, self).__init__()
        self.args = vars(args) if args is not None else {}
        self.dim = self.args.get("dim", DIM)
        self.n_layer = self.args.get("n_layer", N_LAYER)
        self.cutoff = self.args.get("cutoff", CUTOFF)
        num_spherical = self.args.get("num_spherical", NUM_SPHERICAL)
        num_radial = self.args.get("num_radial", NUM_RADIAL)
        envelope_exponent = self.args.get("envelope_exponent", ENVELOPE_EXPONENT)

        self.embeddings = nn.Parameter(torch.ones((5, self.dim)))

        self.rbf_l = BesselBasisLayer(16, 5, envelope_exponent)
        self.rbf_g = BesselBasisLayer(16, self.cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, 5, envelope_exponent)

        self.rbf_g_mlp = MLP([16, self.dim])
        self.rbf_l_mlp = MLP([16, self.dim])

        self.sbf_1_mlp = MLP([num_spherical * num_radial, self.dim])
        self.sbf_2_mlp = MLP([num_spherical * num_radial, self.dim])

        self.global_layers = torch.nn.ModuleList()
        for layer in range(self.n_layer):
            self.global_layers.append(Global_MP(self.dim))

        self.local_layers = torch.nn.ModuleList()
        for layer in range(self.n_layer):
            self.local_layers.append(Local_MP(self.dim))

        self.init()

    def init(self):
        stdv = math.sqrt(3)
        self.embeddings.data.uniform_(-stdv, stdv)

    def indices(self, edge_index, num_nodes):
        row, col = edge_index

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes))

        # Compute the node indices for two-hop angles
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k
        idx_i_1, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji_1 = adj_t_row.storage.row()[mask]

        # Compute the node indices for one-hop angles
        adj_t_col = adj_t[col]

        num_pairs = adj_t_col.set_value(None).sum(dim=1).to(torch.long)
        idx_i_2 = row.repeat_interleave(num_pairs)
        idx_j1 = col.repeat_interleave(num_pairs)
        idx_j2 = adj_t_col.storage.col()

        idx_ji_2 = adj_t_col.storage.row()
        idx_jj = adj_t_col.storage.value()

        return idx_i_1, idx_j, idx_k, idx_kj, idx_ji_1, idx_i_2, idx_j1, idx_j2, idx_jj, idx_ji_2

    def forward(self, data):
        batch, x, pos, edge_index = (data.batch, data.x, data.pos, data.edge_index)
        x = torch.argmax(x[:, :5], dim=1)
        # x = data.x
        # edge_index = data.edge_index
        # pos = data.pos
        # batch = data.batch
        # Initialize node embeddings
        h = torch.index_select(self.embeddings, 0, x.long())

        # Get the edges and pairwise distances in the local layer
        edge_index_l, _ = remove_self_loops(edge_index)
        j_l, i_l = edge_index_l
        dist_l = (pos[i_l] - pos[j_l]).pow(2).sum(dim=-1).sqrt()

        # Get the edges pairwise distances in the global layer
        row, col = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=500)
        edge_index_g = torch.stack([row, col], dim=0)
        edge_index_g, _ = remove_self_loops(edge_index_g)
        j_g, i_g = edge_index_g
        dist_g = (pos[i_g] - pos[j_g]).pow(2).sum(dim=-1).sqrt()

        # Compute the node indices for defining the angles
        (
            idx_i_1,
            idx_j,
            idx_k,
            idx_kj,
            idx_ji,
            idx_i_2,
            idx_j1,
            idx_j2,
            idx_jj,
            idx_ji_2,
        ) = self.indices(edge_index_l, num_nodes=h.size(0))

        # Compute the two-hop angles
        pos_ji_1, pos_kj = pos[idx_j] - pos[idx_i_1], pos[idx_k] - pos[idx_j]
        a = (pos_ji_1 * pos_kj).sum(dim=-1)
        b = torch.cross(pos_ji_1, pos_kj).norm(dim=-1)
        angle_1 = torch.atan2(b, a)

        # Compute the one-hop angles
        pos_ji_2, pos_jj = pos[idx_j1] - pos[idx_i_2], pos[idx_j2] - pos[idx_j1]
        a = (pos_ji_2 * pos_jj).sum(dim=-1)
        b = torch.cross(pos_ji_2, pos_jj).norm(dim=-1)
        angle_2 = torch.atan2(b, a)

        # Get the RBF and SBF embeddings
        rbf_g = self.rbf_g(dist_g)
        rbf_l = self.rbf_l(dist_l)
        sbf_1 = self.sbf(dist_l, angle_1, idx_kj)
        sbf_2 = self.sbf(dist_l, angle_2, idx_jj)

        rbf_g = self.rbf_g_mlp(rbf_g)
        rbf_l = self.rbf_l_mlp(rbf_l)
        sbf_1 = self.sbf_1_mlp(sbf_1)
        sbf_2 = self.sbf_2_mlp(sbf_2)

        # Perform the message passing schemes
        node_sum = 0

        for layer in range(self.n_layer):
            h = self.global_layers[layer](h, rbf_g, edge_index_g)
            h, t = self.local_layers[layer](
                h, rbf_l, sbf_1, sbf_2, idx_kj, idx_ji, idx_jj, idx_ji_2, edge_index_l
            )
            node_sum += t

        # Readout
        output = global_add_pool(node_sum, batch)
        return output

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--dim", type=int, default=DIM, help="Size of input hidden units.")
        parser.add_argument(
            "--n_layer", type=int, default=N_LAYER, help="Number of hidden layers."
        )
        parser.add_argument(
            "--cutoff", type=float, default=CUTOFF, help="Distance cutoff used in the global layer"
        )
        # TODO: Add help description
        parser.add_argument("--num_spherical", type=int, default=NUM_SPHERICAL)
        parser.add_argument("--num_radial", type=int, default=NUM_RADIAL)
        parser.add_argument("--envelope_exponent", type=int, default=ENVELOPE_EXPONENT)
        return parser
