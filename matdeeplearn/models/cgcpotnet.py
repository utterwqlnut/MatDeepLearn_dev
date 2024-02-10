from typing import Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from torch import Tensor
from torch.nn import BatchNorm1d, Linear, Module, Sequential
from torch_geometric.nn import (
    CGConv,
    Set2Set,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_scatter import scatter, scatter_add, scatter_max, scatter_mean

from matdeeplearn.common.registry import registry
from matdeeplearn.models.base_model import BaseModel, conditional_grad
from matdeeplearn.models.cgcnn import CGCNN
from matdeeplearn.preprocessor.helpers import GaussianSmearing


@registry.register_model("CGCPotnet")
class CGCPotnet(BaseModel):
    def __init__(
        self,
        node_dim,
        edge_dim,
        output_dim,
        dim1=64,
        dim2=64,
        pre_fc_count=1,
        post_fc_count=1,
        pool="global_mean_pool",
        pool_order="early",
        batch_norm=True,
        batch_track_stats=True,
        act="relu",
        dropout_rate=0.0,
        inf_edge_features: int = 64,
        rbfmin: float = -4.0,
        rbfmax: float = 4.0,
        gc_count=4,
        **kwargs
    ) -> None:
        super(CGCPotnet, self).__init__(**kwargs)

        self.batch_track_stats = batch_track_stats
        self.batch_norm = batch_norm
        self.pool = pool
        self.act = act
        self.pool_order = pool_order
        self.dropout_rate = dropout_rate
        self.pre_fc_count = pre_fc_count
        self.dim1 = dim1
        self.dim2 = dim2
        self.post_fc_count = post_fc_count
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.rbfmin = rbfmin
        self.rbfmax = rbfmax
        self.gc_count = gc_count
        self.inf_edge_features = inf_edge_features

        self.inf_embedding = RBFExpansion(rbfmin, rbfmax, inf_edge_features)

        # assert short_gc_count > 0, "Need at least 1 Short GC Layer"
        # assert long_gc_count > 0, "Need at least 1 Long GC Layer"
        if pre_fc_count == 0:
            self.gc_dim, self.post_fc_dim = self.node_dim, self.node_dim
        else:
            self.gc_dim, self.post_fc_dim = dim1, dim1

        self.pre_lin_list = self._setup_pre_gnn_layers()
        self.conv_list, self.bn_list = self._setup_gnn_layers()
        self.post_lin_list, self.lin_out = self._setup_post_gnn_layers()

        if self.pool_order == "early" and self.pool == "set2set":
            self.set2set = Set2Set(self.post_fc_dim * 2, processing_steps=3)
        elif self.pool_order == "late" and self.pool == "set2set":
            self.set2set = Set2Set(self.output_dim, processing_steps=3, num_layers=1)
            # workaround for doubled dimension by set2set; if late pooling not recommended to use set2set
            self.lin_out_2 = torch.nn.Linear(self.output_dim * 2, self.output_dim)

    @property
    def target_attr(self):
        return "y"

    def _setup_pre_gnn_layers(self):
        """Sets up pre-GNN dense layers (NOTE: in v0.1 this is always set to 1 layer)."""
        pre_lin_list = torch.nn.ModuleList()
        if self.pre_fc_count > 0:
            pre_lin_list = torch.nn.ModuleList()
            for i in range(self.pre_fc_count):
                if i == 0:
                    lin = torch.nn.Linear(self.node_dim, self.dim1)
                else:
                    lin = torch.nn.Linear(self.dim1, self.dim1)
                pre_lin_list.append(lin)

        return pre_lin_list

    def _setup_gnn_layers(self):
        """Sets up GNN layers."""
        conv_list = torch.nn.ModuleList()
        bn_list = torch.nn.ModuleList()
        for i in range(self.gc_count):

            conv = CGConv(
                self.gc_dim,
                self.edge_dim + self.inf_edge_features,
                aggr="mean",
                batch_norm=False,
            )
            conv_list.append(conv)
            # Track running stats set to false can prevent some instabilities; this causes other issues with different val/test performance from loader size?
            if self.batch_norm:
                bn = BatchNorm1d(
                    self.gc_dim, track_running_stats=self.batch_track_stats
                )
                bn_list.append(bn)

        return conv_list, bn_list

    def _setup_post_gnn_layers(self):
        """Sets up post-GNN dense layers (NOTE: in v0.1 there was a minimum of 2 dense layers, and fc_count(now post_fc_count) added to this number. In the current version, the minimum is zero)."""
        post_lin_list = torch.nn.ModuleList()

        if self.post_fc_count > 0:
            for i in range(self.post_fc_count):
                if i == 0:
                    # Set2set pooling has doubled dimension
                    if self.pool_order == "early" and self.pool == "set2set":
                        lin = torch.nn.Linear(self.post_fc_dim * 2, self.dim2)
                    else:
                        lin = torch.nn.Linear(self.post_fc_dim, self.dim2)
                else:
                    lin = torch.nn.Linear(self.dim2, self.dim2)
                post_lin_list.append(lin)
            lin_out = torch.nn.Linear(self.dim2, self.output_dim)
            # Set up set2set pooling (if used)

        # else post_fc_count is 0
        else:
            if self.pool_order == "early" and self.pool == "set2set":
                lin_out = torch.nn.Linear(self.post_fc_dim * 2, self.output_dim)
            else:
                lin_out = torch.nn.Linear(self.post_fc_dim, self.output_dim)

        return post_lin_list, lin_out

    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        inf_attr = self.inf_embedding(data.inf_edge_attr)

        for i in range(0, len(self.pre_lin_list)):
            if i == 0:
                out = self.pre_lin_list[i](data.x)
                out = getattr(F, self.act)(out)
            else:
                out = self.pre_lin_list[i](out)
                out = getattr(F, self.act)(out)

        # GNN layers
        for i in range(0, self.gc_count):
            index = data.edge_index
            attr = torch.cat((data.edge_attr, inf_attr), dim=1)

            if len(self.pre_lin_list) == 0 and i == 0:
                if self.batch_norm:
                    out = self.conv_list[i](data.x, index, attr)
                    out = self.bn_list[i](out)
                else:
                    out = self.conv_list[i](data.x, index, attr)
            else:
                if self.batch_norm:
                    out = self.conv_list[i](out, index, attr)
                    out = self.bn_list[i](out)
                else:
                    out = self.conv_list[i](out, index, attr)
                    # out = getattr(F, self.act)(out)
            out = F.dropout(out, p=self.dropout_rate, training=self.training)

        if self.prediction_level == "graph":
            if self.pool_order == "early":
                if self.pool == "set2set":
                    out = self.set2set(out, data.batch)
                else:
                    out = getattr(torch_geometric.nn, self.pool)(out, data.batch)
                for i in range(0, len(self.post_lin_list)):
                    out = self.post_lin_list[i](out)
                    out = getattr(F, self.act)(out)
                out = self.lin_out(out)

            elif self.pool_order == "late":
                for i in range(0, len(self.post_lin_list)):
                    out = self.post_lin_list[i](out)
                    out = getattr(F, self.act)(out)
                out = self.lin_out(out)
                if self.pool == "set2set":
                    out = self.set2set(out, data.batch)
                    out = self.lin_out_2(out)
                else:
                    out = getattr(torch_geometric.nn, self.pool)(out, data.batch)

        elif self.prediction_level == "node":
            for i in range(0, len(self.post_lin_list)):
                out = self.post_lin_list[i](out)
                out = getattr(F, self.act)(out)
            out = self.lin_out(out)

        return out

    def forward(self, data):
        output = {}
        out = self._forward(data)
        output["output"] = out

        if self.gradient == True and out.requires_grad == True:
            volume = torch.einsum(
                "zi,zi->z",
                data.cell[:, 0, :],
                torch.cross(data.cell[:, 1, :], data.cell[:, 2, :], dim=1),
            ).unsqueeze(-1)
            grad = torch.autograd.grad(
                out,
                [data.pos, data.displacement],
                grad_outputs=torch.ones_like(out),
                create_graph=self.training,
            )
            forces = -1 * grad[0]
            stress = grad[1]
            stress = stress / volume.view(-1, 1, 1)

            output["pos_grad"] = forces
            output["cell_grad"] = stress
        else:
            output["pos_grad"] = None
            output["cell_grad"] = None

        return output


class RBFExpansion(Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale: Optional[float] = None,
        type: str = "gaussian",
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer("centers", torch.linspace(vmin, vmax, bins))
        self.type = type

        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale**2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        base = self.gamma * (distance - self.centers)
        if self.type == "gaussian":
            return (-(base**2)).exp()
        elif self.type == "quadratic":
            return base**2
        elif self.type == "linear":
            return base
        elif self.type == "inverse_quadratic":
            return 1.0 / (1.0 + base**2)
        elif self.type == "multiquadric":
            return (1.0 + base**2).sqrt()
        elif self.type == "inverse_multiquadric":
            return 1.0 / (1.0 + base**2).sqrt()
        elif self.type == "spline":
            return base**2 * (base + 1.0).log()
        elif self.type == "poisson_one":
            return (base - 1.0) * (-base).exp()
        elif self.type == "poisson_two":
            return (base - 2.0) / 2.0 * base * (-base).exp()
        elif self.type == "matern32":
            return (1.0 + 3**0.5 * base) * (-(3**0.5) * base).exp()
        elif self.type == "matern52":
            return (1.0 + 5**0.5 * base + 5 / 3 * base**2) * (
                -(5**0.5) * base
            ).exp()
        else:
            raise Exception("No Implemented Radial Basis Method")