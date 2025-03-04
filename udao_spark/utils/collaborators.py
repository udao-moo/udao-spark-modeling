import os.path
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from udao_trace.utils import JsonHandler

from .constants import (
    ALPHA,
    ALPHA_COMPILE,
    ALPHA_QS_PLUS,
    BETA,
    EPS,
    GAMMA,
    THETA_C,
    THETA_P,
    THETA_S,
)
from .exceptions import NoBenchmarkError, NoQTypeError
from .logging import logger
from .params import ExtractParams, QType


def get_data_sign(bm: str, debug: bool) -> str:
    if bm == "tpch":
        return f"22x{10 if debug else 2273}"
    if bm == "tpcds":
        return f"102x{10 if debug else 490}"
    if bm == "job":
        return "100000x1"
    if bm == "tpcds+job":
        return f"102x{10 if debug else 490}+100000x1"
    if bm == "tpch+job":
        return f"22x{10 if debug else 2273}+100000x1"
    if bm == "tpcds+tpch":
        return f"102x{10 if debug else 490}+22x{10 if debug else 2273}"
    if bm == "tpcds+tpch+job":
        return f"102x{10 if debug else 490}+22x{10 if debug else 2273}+100000x1"
    raise NoBenchmarkError(bm)


class PathWatcher:
    def __init__(
        self,
        base_dir: Path,
        benchmark: str,
        debug: bool,
        extract_params: ExtractParams,
        fold: Optional[int],
        data_percentage: Optional[int] = None,
        benchmark_ext: Optional[str] = None,
        ext_data_amount: Optional[int] = None,
        ext_up_to_n_joins: Optional[int] = None,
        data_percentage2: Optional[int] = None,
        benchmark_ext2: Optional[str] = None,
        ext_data_amount2: Optional[int] = None,
        ext_up_to_n_joins2: Optional[int] = None,
        fold_peek_percentage: int = 0,
    ):
        self.base_dir = base_dir
        self.benchmark = benchmark
        self.debug = debug
        self.fold = fold
        self.fold_peek_percentage = fold_peek_percentage
        self.data_percentage = data_percentage
        self.data_percentage2 = data_percentage2
        self.benchmark_ext = benchmark_ext
        self.benchmark_ext2 = benchmark_ext2
        self.ext_up_to_n_joins = ext_up_to_n_joins
        self.ext_up_to_n_joins2 = ext_up_to_n_joins2
        self.extract_params = extract_params

        if fold is None and fold_peek_percentage > 0:
            raise ValueError("fold_peek_percentage is only supported when fold is set")

        if benchmark_ext and not ext_data_amount:
            raise ValueError(
                "ext_data_amount must be specified when benchmark_ext is specified"
            )
        if benchmark_ext2 and not ext_data_amount2:
            raise ValueError(
                "ext_data_amount2 must be specified when benchmark_ext2 is specified"
            )
        if benchmark_ext2 and not benchmark_ext:
            raise ValueError(
                "benchmark_ext must be specified when benchmark_ext2 is specified"
            )
        self.ext_data_amount = ext_data_amount
        self.ext_data_amount2 = ext_data_amount2

        if benchmark == "job" and fold is not None:
            raise ValueError("fold is not supported for job benchmark")

        data_prefix = f"{str(self.base_dir)}/data/{self.benchmark}"

        cc_prefix = self.get_cc_prefix(with_ext=True)
        cc_extract_prefix = f"{cc_prefix}/{extract_params.hash()}"
        if fold is not None:
            if fold not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                raise ValueError(f"fold must be in [1, 2, ..., 10], got {fold}")
            cc_extract_prefix += f"-{fold}"
            if self.fold_peek_percentage > 0:
                if self.benchmark == "job":
                    raise ValueError(
                        "peek percentage is not supported for job benchmark"
                    )
                if self.benchmark_ext is not None or self.benchmark_ext2 is not None:
                    raise ValueError(
                        "have not sorted out the naming when mixing benchmarks."
                    )
                cc_extract_prefix += f"_peek_{self.fold_peek_percentage}_percents"
        if data_percentage is not None:
            if data_percentage not in list(range(101)):
                raise ValueError("data_percentage must be in [0, 1, ..., 100]")
            cc_extract_prefix += f"_{data_percentage}"
        if data_percentage2 is not None:
            if data_percentage2 not in list(range(101)):
                raise ValueError("data_percentage2 must be in [0, 1, ..., 100]")
            cc_extract_prefix += f"_{data_percentage2}"
        if self.benchmark_ext:
            if self.ext_data_amount:
                if benchmark == "job" and self.ext_data_amount > 27371:
                    raise ValueError("ext_data_amount must be less than 27371")
                cc_extract_prefix += f"_ext_{self.ext_data_amount}"
            else:
                raise Exception(
                    "when benchmark_ext is specified, " "explicitly specific the amount"
                )
            if self.ext_up_to_n_joins:
                cc_extract_prefix += f"_upto{self.ext_up_to_n_joins}joins"
            if self.benchmark_ext2:
                if self.ext_data_amount2:
                    cc_extract_prefix += f"_ext2_{self.ext_data_amount2}"
                else:
                    raise Exception(
                        "when benchmark_ext2 is specified, "
                        "explicitly specific the amount"
                    )
                if self.ext_up_to_n_joins2:
                    cc_extract_prefix += f"_upto{self.ext_up_to_n_joins2}joins"
        elif self.benchmark_ext2:
            raise ValueError(
                "benchmark_ext2 must be specified after benchmark_ext is set"
            )

        self.data_prefix = data_prefix
        self.cc_prefix = cc_prefix
        self.cc_extract_prefix = cc_extract_prefix
        self._checkpoint_split()

    def get_cc_prefix(self, with_ext: bool) -> str:
        data_sign = self.get_data_sign(with_ext)
        cc_prefix = f"{str(self.base_dir)}/cache_and_ckp/{self.benchmark}_{data_sign}"
        return cc_prefix

    def _checkpoint_split(self) -> None:
        json_name = "extract_param.json"
        if not Path(f"{self.cc_extract_prefix}/{json_name}").exists():
            JsonHandler.dump_to_file(
                self.extract_params.__dict__,
                f"{self.cc_extract_prefix}/{json_name}",
                indent=2,
            )
            logger.info(f"saved split params to {self.cc_extract_prefix}/{json_name}")
        else:
            logger.info(f"found {self.cc_extract_prefix}/{json_name}")

    def get_data_sign(self, with_ext: bool = True) -> str:
        benchmark = self.benchmark
        data_sign = get_data_sign(benchmark, self.debug)
        benchmark_ext = self.benchmark_ext
        if benchmark_ext and with_ext:
            if benchmark == "job" and benchmark_ext == "job-ext":
                data_sign += "+ext_27371x1"
            elif benchmark == "tpcds" and benchmark_ext == "tpcds-ext-selected":
                data_sign += "+ext_selected_174x50"
            elif benchmark == "tpcds" and benchmark_ext == "tpcds-ext-star-joins":
                data_sign += "+ext_star_joins_5518x10"
            elif benchmark in ["tpcds+job", "tpcds+tpch+job"]:
                data_sign += "+ext_27371x1"
                if self.benchmark_ext2:
                    if self.benchmark_ext2 == "tpcds-ext-selected":
                        data_sign += "+ext2_selected_174x50"
                    elif self.benchmark_ext2 == "tpcds-ext-star-joins":
                        data_sign += "+ext2_star_joins_5518x10"
                    else:
                        raise ValueError(f"invalid {benchmark_ext} as ext2")
            else:
                raise ValueError(f"invalid {benchmark_ext}")
        return data_sign

    def get_ori_data_header(self, q_type: str) -> str:
        data_sign = self.get_data_sign(with_ext=False)
        return f"{self.data_prefix}/{q_type}_{data_sign}.csv"

    def get_ext_data_header(self, q_type: str) -> str:
        if self.benchmark_ext is None:
            raise ValueError("benchmark_ext must be specified")
        data_prefix = os.path.dirname(self.data_prefix) + "/" + self.benchmark_ext
        if self.benchmark_ext == "tpcds-ext-selected":
            data_sign = "174x50"
        elif self.benchmark_ext == "tpcds-ext-star-joins":
            data_sign = "5518x10"
        else:
            raise ValueError(f"invalid {self.benchmark_ext}")
        return f"{data_prefix}/{q_type}_{data_sign}.csv"


class TypeAdvisor:
    def __init__(self, q_type: QType):
        self.q_type = q_type

    def get_graph_column(self) -> str:
        if self.q_type in ["q_compile", "q_all"]:
            return "lqp"
        if self.q_type in ["qs_lqp_compile", "qs_lqp_runtime"]:
            return "qs_lqp"
        if self.q_type in ["qs_pqp_runtime"]:
            return "qs_pqp"
        raise NoQTypeError(self.q_type)

    def get_tabular_columns(self) -> List[str]:
        if self.q_type in ["q_compile"]:
            return ALPHA + THETA_C + THETA_P + THETA_S
        if self.q_type in ["qs_lqp_compile"]:
            return ALPHA_COMPILE + THETA_C + THETA_P + THETA_S
        if self.q_type in ["q_all", "qs_lqp_runtime"]:
            return ALPHA + BETA + GAMMA + THETA_C + THETA_P + THETA_S
        if self.q_type in "qs_pqp_runtime":
            return ALPHA + ALPHA_QS_PLUS + BETA + GAMMA + THETA_C + THETA_S
        raise NoQTypeError(self.q_type)

    def get_decision_columns(self) -> List[str]:
        if self.q_type in ["q_compile", "qs_lqp_compile", "qs_lqp_runtime"]:
            return THETA_C + THETA_P + THETA_S
        if self.q_type in ["q_all"]:
            return THETA_P + THETA_S
        if self.q_type in ["qs_pqp_runtime"]:
            return THETA_S
        raise NoQTypeError(self.q_type)

    def get_tabular_non_decision_columns(self) -> List[str]:
        if self.q_type in ["q_compile"]:
            return ALPHA
        if self.q_type in ["qs_lqp_compile"]:
            return ALPHA_COMPILE
        if self.q_type in ["q_all"]:
            return ALPHA + BETA + GAMMA + THETA_C
        if self.q_type in ["qs_lqp_runtime"]:
            return ALPHA + BETA + GAMMA
        if self.q_type in ["qs_pqp_runtime"]:
            return ALPHA + ALPHA_QS_PLUS + BETA + GAMMA + THETA_C
        raise NoQTypeError(self.q_type)

    def get_objectives(self) -> List[str]:
        if self.q_type in ["q_compile", "q_all"]:
            return ["latency_s", "io_mb"]
        if self.q_type in ["qs_lqp_compile", "qs_lqp_runtime", "qs_pqp_runtime"]:
            return ["latency_s", "io_mb", "ana_latency_s"]
        raise NoQTypeError(self.q_type)

    def get_ag_objectives(self) -> List[str]:
        if self.q_type in ["q_compile", "q_all"]:
            return ["latency_s", "io_mb"]
        if self.q_type in ["qs_lqp_compile", "qs_lqp_runtime", "qs_pqp_runtime"]:
            return ["io_mb", "ana_latency_s"]
        raise NoQTypeError(self.q_type)

    def get_q_type_for_cache(self) -> str:
        if self.q_type in ["q_compile", "q_all"]:
            return self.q_type
        if self.q_type in ["qs_lqp_compile", "qs_lqp_runtime", "qs_pqp_runtime"]:
            return "qs"
        raise NoQTypeError(self.q_type)

    def size_mb_in_log(self, operator: Dict) -> float:
        if self.q_type in ["qs_lqp_compile"]:
            x = operator["stats"]["compileTime"]["sizeInBytes"] / 1024.0 / 1024.0
        elif self.q_type in ["qs_lqp_runtime"]:
            x = operator["stats"]["runtime"]["sizeInBytes"] / 1024.0 / 1024.0
        elif self.q_type in ["q_compile", "q_all", "qs_pqp_runtime"]:
            x = operator["sizeInBytes"] / 1024.0 / 1024.0
        else:
            raise NoQTypeError(self.q_type)
        return np.log(np.clip(x, a_min=EPS, a_max=None))

    def rows_count_in_log(self, operator: Dict) -> float:
        if self.q_type in ["qs_lqp_compile"]:
            x = operator["stats"]["compileTime"]["rowCount"] * 1.0
        elif self.q_type in ["qs_lqp_runtime"]:
            x = operator["stats"]["runtime"]["rowCount"] * 1.0
        elif self.q_type in ["q_compile", "q_all", "qs_pqp_runtime"]:
            x = operator["rowCount"] * 1.0
        else:
            raise NoQTypeError(self.q_type)
        return np.log(np.clip(x, a_min=EPS, a_max=None))

    def get_op_name(self, operator: Dict) -> str:
        if self.q_type.startswith("qs_pqp"):
            # drop the suffix "Exec"
            return operator["className"].split(".")[-1][:-4]
        return operator["className"].split(".")[-1]
