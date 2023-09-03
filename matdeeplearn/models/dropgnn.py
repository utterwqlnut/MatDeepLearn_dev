import random

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


@registry.register_model("DropGNN")
class DropGNN(CGCNN):
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
    ):
        super(DropGNN, self).__init__(
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
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, data):
        # CGCNN but drop some short dist edges
        data.mask1 = self.dropout(data.mask1).to(torch.bool)
        mask = (data.mask1) | (data.mask2)

        zero = data.edge_index[0]
        one = data.edge_index[1]

        data.edge_attr = data.edge_attr[mask]
        zero = zero[mask]
        one = one[mask]

        data.edge_index = torch.stack((zero, one))

        return super(DropGNN, self).forward(data)
