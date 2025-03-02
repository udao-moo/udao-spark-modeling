import glob
import os
from typing import Any, Dict, Optional, Tuple, cast

import pandas as pd
import torch as th
from udao.data import BaseIterator, QueryPlanIterator, TabularContainer
from udao.data.utils.utils import DatasetType
from udao.model import UdaoModel

from udao_spark.data.iterators.tabular_iterator import XFerTabularIterator
from udao_spark.model.model_server import ModelServer
from udao_spark.model.utils import save_query_embeddings
from udao_spark.utils.logging import logger
from udao_trace.utils import PickleHandler


def load_ms(ckp_header: str) -> ModelServer:
    model_params_path = os.path.dirname(ckp_header) + "/model_struct_params.json"
    ckp_weight_path = glob.glob(f"{ckp_header}/*.ckpt")[0]
    model_sign = ckp_header.split("/learning")[0].split("/")[-1][:-12]
    return ModelServer.from_ckp_path(
        model_sign=model_sign,
        model_params_path=model_params_path,
        weights_path=ckp_weight_path,
        verbose=False,
    )


def get_embedding_path(data_header: str, ckp_header: str, benchmark: str) -> str:
    data_sign = data_header.split("/")[-1]
    return f"{ckp_header}/{benchmark}/{data_sign}"


def get_query_embeddings(
    benchmark: str,
    data_header: str,
    ckp_header: str,
    num_workers: int,
    device: str,
    augmented: bool,
) -> Tuple[Dict[DatasetType, pd.DataFrame], Dict[DatasetType, QueryPlanIterator], str]:
    all_splits: Tuple[DatasetType, ...] = ("train", "val", "test")

    loaded_obj = PickleHandler.load(data_header, "split_iterators.pkl")
    split_iterators = cast(Dict[DatasetType, QueryPlanIterator], loaded_obj)

    # to generate the query embedding for the target benchmark bm.
    embedding_path = get_embedding_path(data_header, ckp_header, benchmark)
    if len(glob.glob(f"{embedding_path}/query_embedding_*_cpu.pkl")) != 3:
        print("no embedding found, start generating...")
        ms = load_ms(ckp_header)
        save_query_embeddings(
            module=ms.module,
            split_iterators=split_iterators,
            num_workers=num_workers,
            ckp_learning_header=embedding_path,
            device=device,
        )
    try:
        embedding_dict: Dict[DatasetType, pd.DataFrame] = {
            split: cast(
                pd.DataFrame,
                PickleHandler.load(embedding_path, f"query_embedding_{split}_cpu.pkl"),
            )
            for split in all_splits
        }
    except Exception as e:
        print(f"Loading embedding error: {e}")
        exit(1)

    if augmented:
        logger.info("Augmenting data")
        aug_embedding_dict: Dict[DatasetType, pd.DataFrame] = {
            split: cast(
                pd.DataFrame,
                PickleHandler.load(ckp_header, f"query_embedding_{split}_cpu.pkl"),
            )
            for split in all_splits
        }
        aug_data_header = "/".join(ckp_header.split("/")[:-2])
        aug_loaded_obj = PickleHandler.load(aug_data_header, "split_iterators.pkl")
        aug_split_iterators = cast(Dict[DatasetType, QueryPlanIterator], aug_loaded_obj)
        for split in all_splits[:2]:
            split_iterators[split].keys += aug_split_iterators[split].keys
            split_iterators[split].tabular_features.data = pd.concat(
                [
                    split_iterators[split].tabular_features.data,
                    aug_split_iterators[split].tabular_features.data,
                ]
            )
            split_iterators[split].objectives.data = pd.concat(
                [
                    split_iterators[split].objectives.data,
                    aug_split_iterators[split].objectives.data,
                ]
            )
            embedding_dict[split] = pd.concat(
                [embedding_dict[split], aug_embedding_dict[split]]
            )

    return embedding_dict, split_iterators, embedding_path


def get_xfer_splits(
    bm: str,
    data_header: str,
    ckp_header: str,
    num_workers: int,
    device: str,
    augmented: bool,
) -> Tuple[Dict[DatasetType, BaseIterator], Dict[str, Any]]:
    embedding_dict, split_iterators, embedding_path = get_query_embeddings(
        bm, data_header, ckp_header, num_workers, device, augmented
    )

    # mash trained embedding with the split iterators
    split_iterators_embedding: Dict[DatasetType, BaseIterator] = {}
    all_splits: Tuple[DatasetType, ...] = ("train", "val", "test")
    for split in all_splits:
        iterator = split_iterators[split]
        new_iterator = XFerTabularIterator(
            iterator.keys,
            TabularContainer(embedding_dict[split]),
            iterator.tabular_features,
            iterator.objectives,
        )
        split_iterators_embedding[split] = new_iterator

    input_embedding_dim = embedding_dict[all_splits[0]].shape[1]

    meta = {
        "input_embedding_dim": input_embedding_dim,
        "feature_names": split_iterators["train"].shape.feature_names,
        "output_names": split_iterators["train"].shape.output_names,
    }

    return split_iterators_embedding, meta


def finetune_setup(
    ckp_header: str, model: UdaoModel, finetune_layers: Optional[int] = None
) -> None:
    ms = load_ms(ckp_header)
    model.regressor.load_state_dict(ms.module.model.regressor.state_dict())

    if finetune_layers is not None:
        lc_cnt = sum(isinstance(m, th.nn.Linear) for m in model.regressor.layers)
        freeze_limit = lc_cnt - finetune_layers
        found_linear = 0
        for module in model.regressor.layers:
            if isinstance(module, th.nn.Linear):
                if found_linear < freeze_limit:
                    for param in module.parameters():
                        param.requires_grad = False
                found_linear += 1
        for param_name, param_value in model.named_parameters():
            logger.info(f"{param_name}: {param_value.requires_grad}")
