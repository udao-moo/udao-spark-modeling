from pathlib import Path
from typing import cast

import torch as th
from udao.data import QueryPlanIterator
from udao.data.handler.data_processor import DataProcessor
from udao.data.utils.query_plan import random_flip_positional_encoding
from udao.model.utils.utils import set_deterministic_torch
from udao.utils.logging import logger

from udao_spark.data.utils import get_split_iterators
from udao_spark.model.utils import (
    GraphTransformerSKMLPParams,
    MyLearningParams,
    get_graph_transformer_sk_mlp,
    get_non_siblings_map,
    param_init,
    train_and_dump,
)
from udao_spark.utils.params import get_graph_transformer_params, wrap_sk_mlp_params
from udao_trace.utils import PickleHandler

logger.setLevel("INFO")
if __name__ == "__main__":
    params = wrap_sk_mlp_params(get_graph_transformer_params()).parse_args()
    set_deterministic_torch(params.seed)
    if params.benchmark == "tpcds":
        th.set_float32_matmul_precision("medium")  # type: ignore
    print(params)
    device = "gpu" if th.cuda.is_available() else "cpu"
    tensor_dtypes = th.float32
    th.set_default_dtype(tensor_dtypes)  # type: ignore

    # Data definition
    ta, pw = param_init(Path(__file__).parent, params)
    split_iterators = get_split_iterators(pw=pw, ta=ta, tensor_dtypes=tensor_dtypes)
    train_iterator = cast(QueryPlanIterator, split_iterators["train"])
    split_iterators["train"].set_augmentations(
        [train_iterator.make_graph_augmentation(random_flip_positional_encoding)]
    )
    dp = PickleHandler.load(pw.cc_extract_prefix, "data_processor.pkl")
    if not isinstance(dp, DataProcessor):
        raise TypeError(f"Expected DataProcessor, got {type(dp)}")
    template_plans = dp.feature_extractors["query_structure"].template_plans
    non_siblings_map = get_non_siblings_map(
        {k: query.graph for k, query in template_plans.items()}
    )

    # Model definition and training
    model_params = GraphTransformerSKMLPParams.from_dict(
        {
            "iterator_shape": split_iterators["train"].shape,
            "op_groups": params.op_groups,
            "output_size": params.output_size,
            "pos_encoding_dim": params.pos_encoding_dim,
            "gtn_n_layers": params.gtn_n_layers,
            "gtn_n_heads": params.gtn_n_heads,
            "readout": params.readout,
            "type_embedding_dim": params.type_embedding_dim,
            "hist_embedding_dim": params.hist_embedding_dim,
            "bitmap_embedding_dim": params.bitmap_embedding_dim,
            "embedding_normalizer": params.embedding_normalizer,
            "gtn_dropout": params.gtn_dropout,
            "n_layers": params.n_layers,
            "hidden_dim": params.hidden_dim,
            "dropout": params.dropout,
            "use_batchnorm": params.use_batchnorm,
            "activate": params.activate,
            "attention_layer_name": "RAAL",
            "non_siblings_map": non_siblings_map,
        }
    )

    if params.loss_weights is not None:
        if len(params.loss_weights) != len(ta.get_objectives()):
            raise ValueError(
                f"loss_weights must have the same length as objectives, "
                f"got {len(params.loss_weights)} and {len(ta.get_objectives())}"
            )

    learning_params = MyLearningParams.from_dict(
        {
            "epochs": params.epochs,
            "batch_size": params.batch_size,
            "init_lr": params.init_lr,
            "min_lr": params.min_lr,
            "weight_decay": params.weight_decay,
            "loss_weights": params.loss_weights,
        }
    )

    model = get_graph_transformer_sk_mlp(model_params)

    train_and_dump(
        ta=ta,
        pw=pw,
        model=model,
        split_iterators=split_iterators,
        extract_params=pw.extract_params,
        model_params=model_params,
        learning_params=learning_params,
        params=params,
        device=device,
    )
