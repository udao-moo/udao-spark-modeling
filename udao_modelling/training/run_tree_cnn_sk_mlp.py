from pathlib import Path
from typing import Annotated, cast

import torch as th
import typer
from udao_modelling.params import TreeCNNParams, get_parameters
from udao.data import QueryPlanIterator
from udao.data.handler.data_processor import DataProcessor
from udao.model.utils.utils import set_deterministic_torch
from udao.utils.logging import logger
from udao_spark.data.utils import get_split_iterators
from udao_spark.model.utils import (
    MyLearningParams,
    TreeCNNSKParams,
    get_tree_cnn_sk_mlp,
    param_init,
    train_and_dump,
)
from udao_trace.utils import PickleHandler

logger.setLevel("INFO")

app = typer.Typer()


def main(
    config: Annotated[str, typer.Argument(help="Location of config file")],
) -> None:
    params = get_parameters(config, TreeCNNParams)
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
    dp = PickleHandler.load(pw.cc_extract_prefix, "data_processor.pkl")
    if not isinstance(dp, DataProcessor):
        raise TypeError(f"Expected DataProcessor, got {type(dp)}")
    op_node2id = dp.feature_extractors["query_structure"].operation_types
    # Model definition and training
    model_params = TreeCNNSKParams.from_dict(
        {
            "iterator_shape": split_iterators["train"].shape,
            "op_groups": params.op_groups,
            "output_size": params.output_size,
            "tcnn_hidden_dim": params.tcnn_hidden_dim,
            "type_embedding_dim": params.type_embedding_dim,
            "hist_embedding_dim": params.hist_embedding_dim,
            "bitmap_embedding_dim": params.bitmap_embedding_dim,
            "n_layers": params.n_layers,
            "hidden_dim": params.hidden_dim,
            "dropout": params.dropout,
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

    model = get_tree_cnn_sk_mlp(model_params)

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


if __name__ == "__main__":
    app()
