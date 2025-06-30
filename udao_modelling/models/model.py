from enum import Enum
from typing import Callable, Protocol

import attrs

import udao_modelling.models as models_params
import udao_modelling.params as params
import udao_spark.model.utils as udao_utils
import udao_modelling.mask.utils


class UdaoModelParams(Protocol):
    @staticmethod
    def from_dict(dict): ...


@attrs.define
class ModelAttributes:
    model_name: str
    cli_params: attrs.field()
    model_params: UdaoModelParams
    structured_params: attrs.field()
    model_creator: Callable


class Model(ModelAttributes, Enum):
    GTN = (
        "gtn",
        params.GraphTransformerParams,
        udao_utils.GraphTransformerSKMLPParams,
        models_params.GraphTransformerStructuredParams,
        udao_utils.get_graph_transformer_sk_mlp,
    )
    GTN_NO_PE = (
        "gtn_no_pe",
        params.GraphTransformerParams,
        udao_utils.GraphTransformerSKMLPParams,
        models_params.GraphTransformerStructuredParams,
        udao_utils.get_graph_transformer_height_encoding_super_node_sk_mlp
    )
    GAT = (
        "gat",
        params.GraphTransformerParams,
        udao_utils.GraphTransformerSKMLPParams,
        models_params.GraphTransformerStructuredParams,
        udao_utils.get_gat_sk_mlp,
    )
    GAT_NO_PE = (
        "gat_no_pe",
        params.GraphTransformerParams,
        udao_utils.GraphTransformerSKMLPParams,
        models_params.GraphTransformerStructuredParams,
        udao_utils.get_gat_sk_mlp,
    )
    GAT_NO_PE_MASKED = (
        "gat_no_pe_masked",
        params.GraphTransformerParams,
        udao_utils.GraphTransformerSKMLPParams,
        models_params.GraphTransformerStructuredParams,
        udao_modelling.mask.utils.get_gat_masked_sk_mlp
    )
    GTN_NO_PE_MASKED = (
        "gtn_no_pe_masked",
        params.GraphTransformerParams,
        udao_utils.GraphTransformerSKMLPParams,
        models_params.GraphTransformerStructuredParams,
        udao_modelling.mask.utils.get_graph_transformer_masked_sk_mlp,
    )
    GTN_MASKED = (
        "gtn_masked",
        params.GraphTransformerParams,
        udao_utils.GraphTransformerSKMLPParams,
        models_params.GraphTransformerStructuredParams,
        udao_modelling.mask.utils.get_graph_transformer_masked_sk_mlp,
    )
    GAT_MASKED = (
        "gat_masked",
        params.GraphTransformerParams,
        udao_modelling.mask.utils.GATMaskedParams,
        models_params.GraphTransformerStructuredParams,
        udao_modelling.mask.utils.get_gat_masked_sk_mlp,
    )
    GCN = (
        "gcn",
        params.GraphConvParams,
        udao_utils.GraphConvNetSKMLPParams,
        models_params.GraphConvStructuredParams,
        udao_utils.get_graph_gcn_sk_mlp
    )
    GCN_MASKED = (
        "gcn_masked",
        params.GraphConvParams,
        udao_modelling.mask.utils.GraphConvMaskedNetSKMLPParams,
        models_params.GraphConvStructuredParams,
        udao_modelling.mask.utils.get_graph_gcn_masked_sk_mlp
    )
    TREE_CNN = (
        "tree_cnn",
        params.TreeCNNParams,
        udao_utils.TreeCNNSKParams,
        models_params.TreeCNNStructuredParams,
        udao_utils.get_tree_cnn_sk_mlp
    )
    TREE_CNN_MASKED = (
        "tree_cnn_masked",
        params.TreeCNNParams,
        udao_modelling.mask.utils.TreeCNNMAskedSKParams,
        models_params.TreeCNNStructuredParams,
        udao_modelling.mask.utils.get_tree_cnn_masked_sk_mlp
    )
    QF = (
        "query_former",
        params.QueryFormerParams,
        udao_utils.GraphTransformerSKMLPParams,
        models_params.QueryformerStructuredParams,
        udao_utils.get_graph_transformer_sk_mlp
    )
    QF_MASKED = (
        "query_former_masked",
        params.QueryFormerParams,
        udao_utils.GraphTransformerSKMLPParams,
        models_params.QueryformerStructuredParams,
        udao_modelling.mask.utils.get_graph_transformer_masked_sk_mlp
    )


