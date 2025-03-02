from dataclasses import dataclass
from typing import Literal

import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv

from .base_graph_embedder import BaseGraphEmbedder

ReadoutType = Literal["sum", "max", "mean"]


class GraphConvNet(BaseGraphEmbedder):
    """Graph Convolutional Network"""

    @dataclass
    class Params(BaseGraphEmbedder.Params):
        n_layers: int
        """Number of GCN layers."""
        hidden_dim: int
        """Size of the hidden layers outputs."""
        readout: ReadoutType
        """Readout type: how the node embeddings are aggregated."""

    def __init__(self, net_params: Params) -> None:
        super().__init__(net_params=net_params)
        self.readout = net_params.readout
        self.convs = nn.ModuleList(
            [GraphConv(self.input_size, net_params.hidden_dim)]
            + [
                GraphConv(net_params.hidden_dim, net_params.hidden_dim)
                for _ in range(net_params.n_layers - 2)
            ]
            + [GraphConv(net_params.hidden_dim, net_params.output_size)]
        )

    def _embed(self, g: dgl.DGLGraph, h: th.Tensor) -> th.Tensor:
        g = dgl.batch([dgl.add_self_loop(gg) for gg in dgl.unbatch(g)])
        for conv in self.convs:
            h = conv(g, h)
            h = F.leaky_relu(h)
        g.ndata["h"] = h

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, "h")
        elif self.readout == "max":
            hg = dgl.max_nodes(g, "h")
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, "h")
        else:
            raise NotImplementedError
        return hg

    def forward(self, g: dgl.DGLGraph) -> th.Tensor:  # type: ignore[override]
        h = self.concatenate_op_features(g)
        return self.normalize_embedding(self._embed(g, h))
