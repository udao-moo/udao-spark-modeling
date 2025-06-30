from pathlib import Path
from typing import List, Optional, TypeVar

import cattrs
import tomlkit
from attrs import define


@define
class BaseParams:
    benchmark: str = "tpch"
    scale_factor: int = 100
    q_type: str = "q_compile"
    debug: bool = False
    seed: int = 42
    fold: Optional[int] = None
    fold_peek_percentage: int = 0
    data_percentage: Optional[int] = None
    benchmark_ext: Optional[str] = None
    ext_data_amount: Optional[int] = None
    ext_up_to_n_joins: Optional[int] = None
    data_percentage2: Optional[int] = None
    benchmark_ext2: Optional[str] = None
    ext_data_amount2: Optional[int] = None
    ext_up_to_n_joins2: Optional[int] = None


@define
class BaseLearningParams(BaseParams):
    init_lr: float = 1e-1
    min_lr: float = 1e-5
    weight_decay: float = 1e-2
    epochs: int = 2
    batch_size: int = 512
    loss_weights: Optional[List[int]] = None
    num_workers: int = 15


@define
class GraphBaseParser(BaseLearningParams):
    lpe_size: int = 8
    vec_size: int = 16
    op_groups: List[str] = ["type", "cbo", "op_enc", "hist", "bitmap"]


@define
class GraphSkipMLPParams(GraphBaseParser):
    activate: str = "relu"
    use_batchnorm: bool = False


T = TypeVar("T")


def get_parameters(config_filename: str, object_type: type[T]) -> T:
    config = tomlkit.loads(Path(config_filename).read_text())
    return cattrs.structure(config, object_type)
