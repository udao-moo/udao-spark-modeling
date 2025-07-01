from .models import (
    MaskedGATTransformer,
    MaskedGraphConv,
    MaskedGraphTransformer,
    MaskedGraphTransformerExtra,
    MaskedTreeCNN, MaskProbability,
)
from udao.model import UdaoModel
from udao_spark.model.regressors.sk_mlp import SkipConnectionMLP
from udao_spark.model.utils import (
    GraphConvNetSKMLPParams,
    GraphTransformerSKMLPParams,
    TreeCNNSKParams,
)

class GraphConvMaskedNetSKMLPParams(GraphConvNetSKMLPParams):
    def hash(self) -> str:
        super_hash = super().hash()
        super_hash = super_hash.replace("graph_gcn", "graph_gcn_masked")
        return super_hash

class TreeCNNMAskedSKParams(TreeCNNSKParams):
    def hash(self) -> str:
        super_hash = super().hash()
        super_hash = super_hash.replace("tree_cnn", "tree_cnn_masked")
        return super_hash

class  GATMaskedParams(GraphTransformerSKMLPParams):
    def hash(self) -> str:
        super_hash = super().hash()
        super_hash = super_hash.replace("masked", f"masked_{MaskProbability.MASK_PROBABILITY}")
        return super_hash


def get_graph_gcn_masked_sk_mlp(params: GraphConvMaskedNetSKMLPParams) -> UdaoModel:
    model = UdaoModel.from_config(
        embedder_cls=MaskedGraphConv,
        regressor_cls=SkipConnectionMLP,
        iterator_shape=params.iterator_shape,
        embedder_params={
            "output_size": params.output_size,  # 128
            "n_layers": params.gcn_n_layers,  # 2
            "hidden_dim": params.output_size,  # same as out_size
            "readout": params.readout,  # "mean"
            "op_groups": params.op_groups,  # all types
            "type_embedding_dim": params.type_embedding_dim,  # 8
            "hist_embedding_dim": params.hist_embedding_dim,  # 32
            "bitmap_embedding_dim": params.bitmap_embedding_dim,  # 32
            "embedding_normalizer": params.embedding_normalizer,  # None
        },
        regressor_params={
            "n_layers": params.n_layers,  # 3
            "hidden_dim": params.hidden_dim,  # 512
            "dropout": params.dropout,  # 0.1
            "use_batchnorm": params.use_batchnorm,  # True
            "activation": params.activate,  # "relu"
        },
    )
    return model


def get_graph_transformer_masked_sk_mlp(
    params: GraphTransformerSKMLPParams,
) -> UdaoModel:
    model = UdaoModel.from_config(
        embedder_cls=MaskedGraphTransformer,
        regressor_cls=SkipConnectionMLP,
        iterator_shape=params.iterator_shape,
        embedder_params={
            "output_size": params.output_size,  # 128
            "pos_encoding_dim": params.pos_encoding_dim,  # 8
            "n_layers": params.gtn_n_layers,  # 2
            "n_heads": params.gtn_n_heads,  # 2
            "hidden_dim": params.output_size,  # same as out_size
            "readout": params.readout,  # "mean"
            "op_groups": params.op_groups,  # all types
            "type_embedding_dim": params.type_embedding_dim,  # 8
            "hist_embedding_dim": params.hist_embedding_dim,  # 32
            "bitmap_embedding_dim": params.bitmap_embedding_dim,  # 32
            "embedding_normalizer": params.embedding_normalizer,  # None
            "attention_layer_name": params.attention_layer_name,  # "GTN"
            "dropout": params.gtn_dropout,
            "max_dist": params.max_dist,  # None
            "max_height": params.max_height,  # None
            "non_siblings_map": params.non_siblings_map,  # None
        },
        regressor_params={
            "n_layers": params.n_layers,  # 3
            "hidden_dim": params.hidden_dim,  # 512
            "dropout": params.dropout,  # 0.1
            "use_batchnorm": params.use_batchnorm,  # True
            "activation": params.activate,  # "relu"
        },
    )
    return model


def get_graph_transformer_height_encoding_super_node_masked_sk_mlp(
    params: GraphTransformerSKMLPParams,
) -> UdaoModel:
    model = UdaoModel.from_config(
        embedder_cls=MaskedGraphTransformerExtra,
        regressor_cls=SkipConnectionMLP,
        iterator_shape=params.iterator_shape,
        embedder_params={
            "output_size": params.output_size,  # 128
            "pos_encoding_dim": params.pos_encoding_dim,  # 8
            "n_layers": params.gtn_n_layers,  # 2
            "n_heads": params.gtn_n_heads,  # 2
            "hidden_dim": params.output_size,  # same as out_size
            "readout": params.readout,  # "mean"
            "op_groups": params.op_groups,  # all types
            "type_embedding_dim": params.type_embedding_dim,  # 8
            "hist_embedding_dim": params.hist_embedding_dim,  # 32
            "bitmap_embedding_dim": params.bitmap_embedding_dim,  # 32
            "embedding_normalizer": params.embedding_normalizer,  # None
            "attention_layer_name": params.attention_layer_name,  # "GTN"
            "dropout": params.gtn_dropout,
            "max_dist": params.max_dist,  # None
            "max_height": params.max_height,  # None
            "non_siblings_map": params.non_siblings_map,  # None
        },
        regressor_params={
            "n_layers": params.n_layers,  # 3
            "hidden_dim": params.hidden_dim,  # 512
            "dropout": params.dropout,  # 0.1
            "use_batchnorm": params.use_batchnorm,  # True
            "activation": params.activate,  # "relu"
        },
    )
    return model


def get_gat_masked_sk_mlp(params: GATMaskedParams) -> UdaoModel:
    model = UdaoModel.from_config(
        embedder_cls=MaskedGATTransformer,
        regressor_cls=SkipConnectionMLP,
        iterator_shape=params.iterator_shape,
        embedder_params={
            "output_size": params.output_size,  # 128
            "pos_encoding_dim": params.pos_encoding_dim,  # 8
            "n_layers": params.gtn_n_layers,  # 2
            "n_heads": params.gtn_n_heads,  # 2
            "hidden_dim": params.output_size,  # same as out_size
            "readout": params.readout,  # "mean"
            "op_groups": params.op_groups,  # all types
            "type_embedding_dim": params.type_embedding_dim,  # 8
            "hist_embedding_dim": params.hist_embedding_dim,  # 32
            "bitmap_embedding_dim": params.bitmap_embedding_dim,  # 32
            "embedding_normalizer": params.embedding_normalizer,  # None
            "attention_layer_name": params.attention_layer_name,  # "GTN"
            "dropout": params.gtn_dropout,
            "max_dist": params.max_dist,  # None
            "max_height": params.max_height,  # None
            "non_siblings_map": params.non_siblings_map,  # None
        },
        regressor_params={
            "n_layers": params.n_layers,  # 3
            "hidden_dim": params.hidden_dim,  # 512
            "dropout": params.dropout,  # 0.1
            "use_batchnorm": params.use_batchnorm,  # True
            "activation": params.activate,  # "relu"
        },
    )
    return model


def get_tree_cnn_masked_sk_mlp(params: TreeCNNMAskedSKParams) -> UdaoModel:
    if (
        params.readout != "max"
        or params.output_size != 64
        or params.tcnn_hidden_dim != 256
    ):
        raise ValueError("does not respect the original paper")
    model = UdaoModel.from_config(
        embedder_cls=MaskedTreeCNN,
        regressor_cls=SkipConnectionMLP,
        iterator_shape=params.iterator_shape,
        embedder_params={
            "output_size": params.output_size,  # 64
            "hidden_dim": params.tcnn_hidden_dim,  # 256
            "readout": params.readout,  # "mean"
            "op_groups": params.op_groups,  # ["type", "cbo", "op_enc"]
            "type_embedding_dim": params.type_embedding_dim,  # 8
            "hist_embedding_dim": params.hist_embedding_dim,  # 32
            "bitmap_embedding_dim": params.bitmap_embedding_dim,  # 32
            "embedding_normalizer": params.embedding_normalizer,  # None
        },
        regressor_params={
            "n_layers": params.n_layers,  # 3
            "hidden_dim": params.hidden_dim,  # 512
            "dropout": params.dropout,  # 0.1
            "use_batchnorm": params.use_batchnorm,  # True
            "activation": params.activate,  # "relu"
        },
    )
    return model
