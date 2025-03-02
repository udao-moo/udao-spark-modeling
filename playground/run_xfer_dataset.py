from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple, cast

import torch as th
from udao.data.utils.utils import DatasetType
from udao.model.utils.utils import set_deterministic_torch

from udao_spark.model.utils import (
    MyLearningParams,
    XFerSKMLPParams,
    get_xfer_sk_mlp,
    train_and_dump_base,
)
from udao_spark.utils.collaborators import TypeAdvisor
from udao_spark.utils.logging import logger
from udao_spark.utils.params import get_base_learning, wrap_sk_mlp_params
from udao_spark.xter import finetune_setup, get_embedding_path, get_xfer_splits


def get_xfer_params() -> ArgumentParser:
    # fmt: off
    parser = get_base_learning()
    parser.add_argument("--data_header", type=str,
                        help="the header of the cached data")
    parser.add_argument("--ckp_header", type=str,
                        help="the path to the header of ckpt for the trained model")
    parser.add_argument("--finetune", action="store_true",
                        help="Enable fine tuning mode")
    parser.add_argument("--finetune_layers", type=int, default=None,
                        help="Number of layers to fine tune, None means all")
    parser.add_argument("--augmented", action="store_true",
                        help="Add augmented data for training")
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

    bm = params.benchmark
    data_header = str(Path(__file__).parent / params.data_header)
    ckp_header = str(Path(__file__).parent / params.ckp_header)
    num_workers = 0 if params.debug else params.num_workers
    augmented = params.augmented

    split_iterators_embedding, meta = get_xfer_splits(
        bm, data_header, ckp_header, num_workers, device, augmented
    )
    # Model definition and training
    model_params = XFerSKMLPParams.from_dict(
        {
            "input_embedding_dim": cast(int, meta["input_embedding_dim"]),
            "feature_names": cast(List[str], meta["feature_names"]),
            "output_names": cast(List[str], meta["output_names"]),
            "n_layers": params.n_layers,
            "hidden_dim": params.hidden_dim,
            "dropout": params.dropout,
            "use_batchnorm": params.use_batchnorm,
            "activate": params.activate,
        }
    )
    model = get_xfer_sk_mlp(model_params)
    if params.finetune:
        finetune_setup(ckp_header, model, params.finetune_layers)
    xfer_settings = []
    if params.augmented:
        xfer_settings.append("augmented")
    if params.finetune:
        xfer_settings.append("finetune")
        if params.finetune_layers is not None:
            xfer_settings.append(f"last_{params.finetune_layers}")

    xfer_ckp_header = get_embedding_path(data_header, ckp_header, bm)
    if xfer_settings:
        xfer_ckp_header += "/" + "_".join(xfer_settings) + "/" + model_params.hash()
    else:
        xfer_ckp_header += "/" + model_params.hash()

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
        ckp_header=xfer_ckp_header,
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
