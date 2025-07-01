from typing import Dict, cast

import torch as th
from udao.data import BaseIterator, QueryPlanIterator
from udao.data.utils.query_plan import random_flip_positional_encoding
from udao.data.utils.utils import DatasetType
from udao.model.utils.utils import set_deterministic_torch
from udao_spark.data.utils import get_split_iterators
from udao_spark.utils.collaborators import PathWatcher, TypeAdvisor


def setup(seed: int, benchmark: str, tensor_dtypes: th.dtype) -> None:
    set_deterministic_torch(seed)
    if benchmark == "tpcds":
        th.set_float32_matmul_precision("medium")  # type: ignore
    th.set_default_dtype(tensor_dtypes)  # type: ignore



def setup_and_get_split_iterators(
    seed: int, benchmark: str, type_advisor: TypeAdvisor, path_watcher: PathWatcher
) -> Dict[DatasetType, BaseIterator]:
    """Set appropriate seeds for reproducibility and return split_iterators used for training"""
    set_deterministic_torch(seed)
    if benchmark == "tpcds":
        th.set_float32_matmul_precision("medium")  # type: ignore
    tensor_dtypes = th.float32
    th.set_default_dtype(tensor_dtypes)  # type: ignore
    split_iterators = get_split_iterators(
        pw=path_watcher, ta=type_advisor, tensor_dtypes=tensor_dtypes
    )
    train_iterator = cast(QueryPlanIterator, split_iterators["train"])
    split_iterators["train"].set_augmentations(
        [train_iterator.make_graph_augmentation(random_flip_positional_encoding)]
    )
    return split_iterators
