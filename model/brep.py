import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from .cluster_pooling import DEC
from typing import Literal, Tuple


class brain_rep(nn.Module):
    def __init__(self, args, node_sz, time_series_sz, corr_pearson_sz, nlayers, dropout=0.,
                 cluster_num=4, pooling=True, orthogonal=True, freeze_center=True, project_assignment=True):
        super().__init__()
        forward_dim = corr_pearson_sz

        self.activation = {'gelu': nn.GELU, 'leaky_relu': nn.LeakyReLU, 'elu': nn.ELU}[args.activation]()
        self.dropout = nn.Dropout(dropout)

        # orthogonal clustering readout
        self.pooling = pooling
        if pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(forward_dim * node_sz, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, forward_dim * node_sz)
            )
            self.dec = DEC(cluster_number=cluster_num, hidden_dimension=forward_dim, encoder=self.encoder,
                           orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment)

        self.dim_reduction = nn.Sequential(
            nn.Linear(forward_dim, 8),
            nn.LeakyReLU()
        )

        if pooling:
            self.fc = nn.Sequential(
                nn.Linear(8 * cluster_num, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 32),
                nn.LeakyReLU(),
                nn.Linear(32, 2)
            )

        hid = 256

        self.mlp_act = "relu"
        layers = []
        if nlayers == 1:
            layers.append(nn.Linear(forward_dim, forward_dim))
        else:
            layers.append(nn.Linear(forward_dim, forward_dim))
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(forward_dim, forward_dim))
            layers.append(nn.Linear(forward_dim, forward_dim))
        self.layers = nn.ModuleList(layers)

    def internal_forward(self, h: torch.Tensor) -> torch.Tensor:
        for li, layer in enumerate(self.layers):
            residual = h
            h = layer(h)

            if li != (len(self.layers) - 1):
                h = h + residual
                if self.mlp_act == "relu":
                    h = F.relu(h)
                elif self.mlp_act == "tanh":
                    h = torch.tanh(h)
        return h


    def forward(self,
                corr: torch.tensor,
                node_features: torch.tensor):

        bz, node_sz, corr_sz = corr.shape

        topo = corr.clone()
        topo = self.internal_forward(topo)


        graph_level_topo, assignment = self.dec(topo)
        graph_level_topo = self.dim_reduction(graph_level_topo)
        graph_level_topo = graph_level_topo.reshape(bz, -1)
        result = self.fc(graph_level_topo)

        return result