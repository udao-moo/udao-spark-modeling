from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Tuple, cast

import pandas as pd
import torch as th
from udao.data import BaseIterator, QueryPlanIterator, TabularContainer
from udao.data.utils.utils import DatasetType
from udao.model.utils.utils import set_deterministic_torch
from udao.utils.logging import logger

from udao_spark.data.iterators.tabular_iterator import XFerTabularIterator
from udao_spark.model.utils import (
    MyLearningParams,
    XFerSKMLPParams,
    get_xfer_sk_mlp,
    train_and_dump_base,
)
from udao_spark.utils.collaborators import TypeAdvisor
from udao_spark.utils.params import get_base_learning, wrap_sk_mlp_params
from udao_trace.utils import PickleHandler


def get_xfer_params() -> ArgumentParser:
    # fmt: off
    parser = get_base_learning()
    parser.add_argument("--data_header", type=str,
                        help="the header of the cached data")
    parser.add_argument("--embedding_path", type=str,
                        help="the path to the embedding files under the data_header")
    # Regressor parameters
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Number of layers in the regressor")
    parser.add_argument("--hidden_dim", type=int, default=32,
                        help="Hidden dimension of the regressor")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    # fmt: on
    return parser


logger.setLevel("INFO")
if __name__ == "__main__":
    params = wrap_sk_mlp_params(get_xfer_params()).parse_args()
    set_deterministic_torch(params.seed)
    if params.benchmark == "tpcds":
        th.set_float32_matmul_precision("medium")  # type: ignore
    print(params)
    device = "gpu" if th.cuda.is_available() else "cpu"
    tensor_dtypes = th.float32
    th.set_default_dtype(tensor_dtypes)  # type: ignore
    ALL_SPLITS: Tuple[DatasetType, ...] = ("train", "val", "test")

    try:
        data_header = str(Path(__file__).parent / params.data_header)
        loaded_obj = PickleHandler.load(data_header, "split_iterators.pkl")
        split_iterators = cast(Dict[DatasetType, QueryPlanIterator], loaded_obj)
        embedding_path = str(
            Path(__file__).parent / params.data_header / params.embedding_path
        )
        embedding_dict: Dict[DatasetType, pd.DataFrame] = {
            split: cast(
                pd.DataFrame,
                PickleHandler.load(embedding_path, f"query_embedding_{split}_cpu.pkl"),
            )
            for split in ALL_SPLITS
        }
        input_embedding_dim = embedding_dict["train"].shape[1]
    except FileNotFoundError as e:
        print(f"Data not found: {e}")
        exit(1)
    # mash trained embedding with the split iterators
    split_iterators_embedding: Dict[DatasetType, BaseIterator] = {}

    for split in ALL_SPLITS:
        iterator = split_iterators[split]
        new_iterator = XFerTabularIterator(
            iterator.keys,
            TabularContainer(embedding_dict[split]),
            iterator.tabular_features,
            iterator.objectives,
        )
        split_iterators_embedding[split] = new_iterator

    # Model definition and training
    model_params = XFerSKMLPParams.from_dict(
        {
            "input_embedding_dim": input_embedding_dim,
            "feature_names": split_iterators["train"].shape.feature_names,
            "output_names": split_iterators["train"].shape.output_names,
            "n_layers": params.n_layers,
            "hidden_dim": params.hidden_dim,
            "dropout": params.dropout,
            "use_batchnorm": params.use_batchnorm,
            "activate": params.activate,
        }
    )
    model = get_xfer_sk_mlp(model_params)

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

    train_and_dump_base(
        bm=params.benchmark,
        ta=TypeAdvisor(q_type=params.q_type),
        model=model,
        ckp_header=embedding_path + "/" + model_params.hash(),
        split_iterators=split_iterators_embedding,
        learning_params=learning_params,
        device=device,
        num_workers=0 if params.debug else params.num_workers,
        hp_params={
            "model_params": model_params.to_dict(),
            "learning_params": learning_params.__dict__,
        },
        dump_query_embedding=False,
        base_dir=str(Path(__file__).parent),
        fold=params.fold,
    )
