import typing

import dgl
import torch as th
from udao_spark.model.embedders import (
    GAT_Transformer,
    GraphTransformer,
    GraphTransformerHeightEncodingSuperNode,
    TreeCNN,
)
from udao_spark.model.embedders.graph_conv_net import GraphConvNet


class MaskProbability:
    MASK_PROBABILITY: float = 0.3


def set_probability(proba: float) -> None:
    MaskProbability.MASK_PROBABILITY = proba


@typing.runtime_checkable
class MaskedModel(typing.Protocol):
    bernoulli: th.distributions.bernoulli.Bernoulli
    training: bool
    op_type: bool
    op_cbo: bool
    op_enc: bool
    op_hist: bool
    op_bitmap: bool
    op_embedder: typing.Optional[th.nn.Module]
    op_hist_embedder: typing.Optional[th.nn.Module]
    op_bitmap_embedder: typing.Optional[th.nn.Module]

    def _get_masked_tensor(self, features: th.Tensor) -> th.Tensor: ...
    def _concatenate_op_features(self, g: dgl.DGLGraph) -> th.Tensor: ...


def _get_masked_tensor(self: MaskedModel, features: th.Tensor) -> th.Tensor:
    masked_rows = self.bernoulli.sample([features.shape[0]])
    masked_rows = masked_rows.type(th.bool)
    return features.masked_fill(masked_rows, 0)


def _concatenate_op_features(self: MaskedModel, g: dgl.DGLGraph) -> th.Tensor:
    op_list = []
    if self.op_type:
        op_list.append(self.op_embedder(g.ndata["op_gid"]))
    if self.op_cbo:
        op_list.append(g.ndata["cbo"])
    if self.op_enc:
        op_enc = g.ndata["op_enc"]
        if self.training:
            op_enc = self._get_masked_tensor(op_enc)
        op_list.append(op_enc)
    if self.op_hist:
        op_list.append(self.op_hist_embedder(g.ndata["hist"]))
    if self.op_bitmap:
        op_list.append(self.op_bitmap_embedder(g.ndata["bitmap"]))
    op_tensor = th.cat(op_list, dim=1) if len(op_list) > 1 else op_list[0]
    return op_tensor


class MaskedGraphConv(GraphConvNet):
    def __init__(self, net_params: GraphConvNet.Params) -> None:
        super().__init__(net_params=net_params)
        self.bernoulli = th.distributions.bernoulli.Bernoulli(
            th.Tensor([MaskProbability.MASK_PROBABILITY])
        )

    def _get_masked_tensor(self: MaskedModel, features: th.Tensor) -> th.Tensor:
        return _get_masked_tensor(self, features)

    def concatenate_op_features(self: MaskedModel, g: dgl.DGLGraph) -> th.Tensor:
        """Apply masked training for word2vec features."""
        return _concatenate_op_features(self, g)


class MaskedGraphTransformer(GraphTransformerHeightEncodingSuperNode):
    def __init__(self, net_params: GraphTransformerHeightEncodingSuperNode.Params) -> None:
        super().__init__(net_params=net_params)
        self.bernoulli = th.distributions.bernoulli.Bernoulli(
            th.Tensor([MaskProbability.MASK_PROBABILITY])
        )

    def _get_masked_tensor(self: MaskedModel, features: th.Tensor) -> th.Tensor:
        return _get_masked_tensor(self, features)

    def concatenate_op_features(self: MaskedModel, g: dgl.DGLGraph) -> th.Tensor:
        """Apply masked training for word2vec features."""
        return _concatenate_op_features(self, g)


class MaskedGraphTransformerExtra(GraphTransformerHeightEncodingSuperNode):
    def __init__(
        self, net_params: GraphTransformerHeightEncodingSuperNode.Params
    ) -> None:
        super().__init__(net_params=net_params)
        self.bernoulli = th.distributions.bernoulli.Bernoulli(
            th.Tensor([MaskProbability.MASK_PROBABILITY])
        )

    def _get_masked_tensor(self: MaskedModel, features: th.Tensor) -> th.Tensor:
        return _get_masked_tensor(self, features)

    def concatenate_op_features(self: MaskedModel, g: dgl.DGLGraph) -> th.Tensor:
        """Apply masked training for word2vec features."""
        return _concatenate_op_features(self, g)


class MaskedTreeCNN(TreeCNN):
    def __init__(self, net_params: TreeCNN.Params) -> None:
        super().__init__(net_params=net_params)
        self.bernoulli = th.distributions.bernoulli.Bernoulli(
            th.Tensor([MaskProbability.MASK_PROBABILITY])
        )

    def _get_masked_tensor(self: MaskedModel, features: th.Tensor) -> th.Tensor:
        return _get_masked_tensor(self, features)

    def concatenate_op_features(self: MaskedModel, g: dgl.DGLGraph) -> th.Tensor:
        """Apply masked training for word2vec features."""
        return _concatenate_op_features(self, g)


class MaskedGATTransformer(GAT_Transformer):
    def __init__(self, net_params: GAT_Transformer.Params) -> None:
        super().__init__(net_params=net_params)
        self.bernoulli = th.distributions.bernoulli.Bernoulli(
            th.Tensor([MaskProbability.MASK_PROBABILITY])
        )

    def _get_masked_tensor(self: MaskedModel, features: th.Tensor) -> th.Tensor:
        return _get_masked_tensor(self, features)

    def concatenate_op_features(self: MaskedModel, g: dgl.DGLGraph) -> th.Tensor:
        """Apply masked training for word2vec features."""
        return _concatenate_op_features(self, g)
