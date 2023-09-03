import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from torch import Tensor
from torch.nn import BatchNorm1d, Linear, Sequential
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


@registry.register_model("DiGNN")
class DiGNN(BaseModel):
    def __init__(
        self,
        node_dim,
        edge_dim,
        output_dim,
        dim1=64,
        dim2=64,
        pre_fc_count=1,
        gc_count=3,
        post_fc_count=1,
        pool="global_mean_pool",
        pool_order="early",
        batch_norm=True,
        batch_track_stats=True,
        act="relu",
        dropout_rate=0.0,
        **kwargs
    ) -> None:
        super(DiGNN, self).__init__(**kwargs)

        self.batch_track_stats = batch_track_stats
        self.batch_norm = batch_norm
        self.pool = pool
        self.act = act
        self.pool_order = pool_order
        self.dropout_rate = dropout_rate
        self.pre_fc_count = pre_fc_count
        self.dim1 = dim1
        self.dim2 = dim2
        self.gc_count = gc_count
        self.post_fc_count = post_fc_count
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        self.distance_expansion = GaussianSmearing(
            0.0, self.cutoff_radius, self.edge_steps
        )

        self.cgcnn_short = LongShortCGCNN(
            node_dim,
            edge_dim,
            output_dim,
            long=True,
            dim1=self.dim1,
            dim2=self.dim2,
            pre_fc_count=self.pre_fc_count,
            gc_count=self.gc_count,
            post_fc_count=self.post_fc_count,
            pool=self.pool,
            pool_order=self.pool_order,
            batch_norm=self.batch_norm,
            batch_track_stats=self.batch_track_stats,
            act=self.act,
            dropout_rate=self.dropout_rate,
            **kwargs,
        )
        self.cgcnn_long = LongShortCGCNN(
            node_dim,
            edge_dim,
            output_dim,
            long=False,
            dim1=self.dim1,
            dim2=self.dim2,
            pre_fc_count=self.pre_fc_count,
            gc_count=self.gc_count,
            post_fc_count=self.post_fc_count,
            pool=self.pool,
            pool_order=self.pool_order,
            batch_norm=self.batch_norm,
            batch_track_stats=self.batch_track_stats,
            act=self.act,
            dropout_rate=self.dropout_rate,
            **kwargs,
        )

        assert gc_count > 0, "Need at least 1 GC layer"
        if pre_fc_count == 0:
            self.gc_dim, self.post_fc_dim = self.node_dim, self.node_dim
        else:
            self.gc_dim, self.post_fc_dim = dim1, dim1

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
        out1 = self.cgcnn_short(data)["output"]
        out2 = self.cgcnn_long(data)["output"]
        out = out2  # torch.cat((out1, out2))
        # data.batch = data.batch.repeat_interleave(2)
        # print(out)
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


class LongShortCGCNN(CGCNN):
    def __init__(
        self,
        node_dim,
        edge_dim,
        output_dim,
        long,
        dim1=64,
        dim2=64,
        pre_fc_count=1,
        gc_count=3,
        post_fc_count=1,
        pool="global_mean_pool",
        pool_order="early",
        batch_norm=True,
        batch_track_stats=True,
        act="relu",
        dropout_rate=0.0,
        **kwargs
    ):
        super(LongShortCGCNN, self).__init__(
            node_dim,
            edge_dim,
            output_dim,
            dim1,
            dim2,
            pre_fc_count,
            gc_count,
            post_fc_count,
            pool,
            pool_order,
            batch_norm,
            batch_track_stats,
            act,
            dropout_rate,
            **kwargs,
        )
        self.long = long
        self.prediction_level = "n/a"

    def forward(self, data):
        if self.long:
            data.edge_attr = data.long_edge_attr
            data.edge_index = data.long_edge_index
        else:
            data.edge_attr = data.short_edge_attr
            data.edge_index = data.short_edge_index

        return super(LongShortCGCNN, self).forward(data)
