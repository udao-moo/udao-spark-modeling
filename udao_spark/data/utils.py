import itertools
import os.path
from collections import Counter
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import torch as th
from sklearn.model_selection import train_test_split
from udao.data import BaseIterator
from udao.data.handler.data_handler import DataHandler
from udao.data.utils.utils import DatasetType, train_test_val_split_on_column

from udao_spark.utils.constants import (
    ALPHA_LQP_RAW,
    ALPHA_QS_RAW,
    BETA,
    BETA_RAW,
    EPS,
    GAMMA,
    THETA_C,
    THETA_P,
    THETA_RAW,
    THETA_S,
)
from udao_trace.configuration import SparkConf
from udao_trace.utils import BenchmarkType, JsonHandler, ParquetHandler, PickleHandler
from udao_trace.workload import Benchmark

from ..utils.collaborators import PathWatcher, TypeAdvisor
from ..utils.logging import logger
from ..utils.params import UdaoParams
from .handlers.data_processor import create_udao_data_processor


# Data Processing
def _im_process(df: pd.DataFrame) -> pd.DataFrame:
    df["IM-sizeInMB"] = df["IM-inputSizeInBytes"] / 1024 / 1024
    df["IM-sizeInMB-log"] = np.log(df["IM-sizeInMB"].to_numpy().clip(min=EPS))
    df["IM-rowCount"] = df["IM-inputRowCount"]
    df["IM-rowCount-log"] = np.log(df["IM-rowCount"].to_numpy().clip(min=EPS))
    for c in ALPHA_LQP_RAW:
        del df[c]
    return df


def extract_compile_time_im(graph_json_str: str) -> Tuple[float, float]:
    graph = JsonHandler.load_json_from_str(graph_json_str)
    operators, links = graph["operators"], graph["links"]
    outgoing_ids_set = set(link["toId"] for link in links)
    input_ids_set = set(range(len(operators))) - outgoing_ids_set
    im_size = sum(
        [
            operators[str(i)]["stats"]["compileTime"]["sizeInBytes"] / 1024.0 / 1024.0
            for i in input_ids_set
        ]
    )
    im_rows_count = sum(
        [
            operators[str(i)]["stats"]["compileTime"]["rowCount"] * 1.0
            for i in input_ids_set
        ]
    )
    return im_size, im_rows_count


def _im_process_compile(df: pd.DataFrame) -> pd.DataFrame:
    """a post-computation for compile-time input meta of each query stage"""
    df[["IM-sizeInMB-compile", "IM-rowCount-compile"]] = np.array(
        np.vectorize(extract_compile_time_im)(df["qs_lqp"])
    ).T
    df["IM-sizeInMB-compile-log"] = np.log(
        df["IM-sizeInMB-compile"].to_numpy().clip(min=EPS)
    )
    df["IM-rowCount-compile-log"] = np.log(
        df["IM-rowCount-compile"].to_numpy().clip(min=EPS)
    )
    return df


def extract_partition_distribution(pd_raw: str) -> Tuple[float, float, float]:
    pd = np.array(
        list(
            chain.from_iterable(
                JsonHandler.load_json_from_str(pd_raw.replace("'", '"')).values()
            )
        )
    )
    if pd.size == 0:
        return 0.0, 0.0, 0.0
    mu, std, max_val, min_val = np.mean(pd), np.std(pd), np.max(pd), np.min(pd)
    ratio1 = std / mu
    ratio2 = (max_val - mu) / mu
    ratio3 = (max_val - min_val) / mu
    return ratio1, ratio2, ratio3


def prepare_data(
    df: pd.DataFrame, sc: SparkConf, benchmark: str, q_type: str, ext: Optional[str]
) -> pd.DataFrame:
    bm = Benchmark(benchmark_type=BenchmarkType(benchmark), ext=ext)
    df.rename(columns={p: kid for p, kid in zip(THETA_RAW, sc.knob_ids)}, inplace=True)
    df["tid"] = df["template"].apply(lambda x: bm.get_template_id(str(x)))
    variable_names = sc.knob_ids
    theta = THETA_C + THETA_P + THETA_S
    if variable_names != theta:
        raise ValueError(f"variable_names != theta: {variable_names} != {theta}")
    df[variable_names] = sc.deconstruct_configuration(
        df[variable_names].astype(str).values
    )

    # extract alpha
    if q_type == "q":
        df[ALPHA_LQP_RAW] = df[ALPHA_LQP_RAW].astype(float)
        df = _im_process(df)
    elif q_type == "qs":
        df[ALPHA_QS_RAW] = df[ALPHA_QS_RAW].astype(float)
        df = _im_process(df)
        df = _im_process_compile(df)
        df["IM-init-part-num"] = df["InitialPartitionNum"].astype(float)
        df["IM-init-part-num-log"] = np.log(
            df["IM-init-part-num"].to_numpy().clip(min=EPS)
        )
        df.rename(columns={"total_task_duration_s": "ana_latency_s"}, inplace=True)
        df["ana_latency_s"] = df["ana_latency_s"] / df["k1"] / df["k3"]
        del df["InitialPartitionNum"]
    else:
        raise ValueError

    # extract beta
    df[BETA] = [
        extract_partition_distribution(pd_raw)
        for pd_raw in df[BETA_RAW].values.squeeze()
    ]
    for c in BETA_RAW:
        del df[c]

    # extract gamma:
    df[GAMMA] = df[GAMMA].astype(float)

    return df


def define_index_with_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    if "id" in df.columns:
        raise Exception("id column already exists!")
    df["id"] = df[columns].astype(str).apply("-".join, axis=1).to_list()
    df.set_index("id", inplace=True)
    return df


def save_and_log_index(index_splits: Dict, pw: PathWatcher, name: str) -> None:
    try:
        PickleHandler.save(index_splits, pw.cc_prefix, name)
    except FileExistsError as e:
        if not pw.debug:
            raise e
        logger.warning(f"skip saving {name}")
    lengths = [str(len(index_splits[split])) for split in ["train", "val", "test"]]
    logger.info(f"got index in {name}, tr/val/te={'/'.join(lengths)}")


def save_and_log_with_path(
    df: pd.DataFrame,
    index_columns: List[str],
    header: str,
    name: str,
    debug: bool = False,
) -> None:
    df = define_index_with_columns(df, columns=index_columns)
    try:
        ParquetHandler.save(df, header, f"{name}.parquet")
    except FileExistsError as e:
        if not debug:
            raise e
        logger.warning(f"skip saving {name}.parquet")
    logger.info(f"prepared {name} shape: {df.shape}")


def save_and_log_df(
    df: pd.DataFrame,
    index_columns: List[str],
    pw: PathWatcher,
    name: str,
) -> None:
    save_and_log_with_path(df, index_columns, pw.cc_prefix, name, pw.debug)


def checkpoint_model_structure(pw: PathWatcher, model_params: UdaoParams) -> str:
    model_struct_hash = model_params.hash()
    ckp_header = f"{pw.cc_extract_prefix}/{model_struct_hash}"
    json_name = "model_struct_params.json"
    if not Path(f"{ckp_header}/{json_name}").exists():
        JsonHandler.dump_to_file(
            model_params.to_dict(),
            f"{ckp_header}/{json_name}",
            indent=2,
        )
        logger.info(f"saved model structure params to {ckp_header}")
    else:
        logger.info(f"found {ckp_header}/{json_name}")
    return ckp_header


def train_test_val_split_on_column_leave_out_fold(
    df: pd.DataFrame,
    groupby_col: str,
    fold: int,
    n_folds: int,
    random_state: Optional[int] = None,
) -> Dict[DatasetType, pd.DataFrame]:
    if n_folds != 10:
        raise ValueError(f"n_folds must be 10, got {n_folds}")
    if fold not in range(1, 11):
        raise ValueError(f"fold must be in [1, 2, ..., 10], got {fold}")

    n_tids_per_fold = len(df[groupby_col].unique()) // n_folds
    unique_tids = df[groupby_col].unique()
    np.random.seed(random_state)
    np.random.shuffle(unique_tids)
    if 0 < fold < 10:
        test_tids = unique_tids[(fold - 1) * n_tids_per_fold : fold * n_tids_per_fold]
    elif fold == 10:
        test_tids = unique_tids[(fold - 1) * n_tids_per_fold :]
    else:
        raise ValueError(f"fold must be in [1, 2, ..., 10], got {fold}")

    trainval_tids = unique_tids[~np.isin(unique_tids, test_tids)]
    train_tids, val_tids = train_test_split(
        trainval_tids, test_size=n_tids_per_fold, random_state=random_state
    )

    train_df = df[df[groupby_col].isin(train_tids)]
    val_df = df[df[groupby_col].isin(val_tids)]
    test_df = df[df[groupby_col].isin(test_tids)]

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }


def magic_setup(pw: PathWatcher, seed: int) -> None:
    """magic set to make sure
    1. data has been properly processed and effectively saved.
    2. data split to make sure q_compile/q/qs share the same appid for tr/val/te.
    """
    benchmark = pw.benchmark
    debug = pw.debug

    # Prepare data
    try:
        df_q_compile = ParquetHandler.load(pw.cc_prefix, "df_q_compile.parquet")
        df_q = ParquetHandler.load(pw.cc_prefix, "df_q_all.parquet")
        df_qs = ParquetHandler.load(pw.cc_prefix, "df_qs.parquet")
        logger.info("Loaded df_q_compile, df_q, df_qs from cache")
    except Exception as e:
        logger.warning(f"Failed to load df_q_compile, df_q, df_qs from cache: {e}")
        df_q_raw = pd.read_csv(pw.get_ori_data_header("q"), low_memory=pw.debug)
        df_qs_raw = pd.read_csv(pw.get_ori_data_header("qs"), low_memory=pw.debug)
        logger.info(f"raw df_q shape: {df_q_raw.shape}")
        logger.info(f"raw df_qs shape: {df_qs_raw.shape}")
        sc = SparkConf(str(pw.base_dir / "assets/spark_configuration_aqe_on.json"))
        ext = pw.benchmark_ext
        df_q = prepare_data(df_q_raw, sc, benchmark, q_type="q", ext=ext)
        df_qs = prepare_data(df_qs_raw, sc, benchmark, q_type="qs", ext=ext)
        df_q_compile = df_q[df_q["lqp_id"] == 0].copy()  # for compile-time df
        df_rare = df_q_compile.groupby("tid").filter(lambda x: len(x) < 5)
        if df_rare.shape[0] > 0:
            logger.warning(f"Drop rare templates: {df_rare['tid'].unique()}")
            df_q_compile = df_q_compile.groupby("tid").filter(lambda x: len(x) >= 5)
        else:
            logger.info("No rare templates")
        # Compute the index for df_q_compile, df_q and df_qs
        save_and_log_df(df_q_compile, ["appid"], pw, "df_q_compile")
        save_and_log_df(df_q, ["appid", "lqp_id"], pw, "df_q_all")
        save_and_log_df(df_qs, ["appid", "qs_id"], pw, "df_qs")
        logger.info("Saved and loaded df_q_compile, df_q, df_qs from cache")

    # Split data for df_q_compile
    if pw.fold is None:
        df_splits_q_compile = train_test_val_split_on_column(
            df=df_q_compile,
            groupby_col="tid",
            val_frac=0.2 if debug else 0.1,
            test_frac=0.2 if debug else 0.1,
            random_state=seed,
        )
    else:
        df_splits_q_compile = train_test_val_split_on_column_leave_out_fold(
            df=df_q_compile,
            groupby_col="tid",
            fold=pw.fold,
            n_folds=10,
            random_state=seed,
        )
        if pw.fold_peek_percentage > 0:
            df_train = df_splits_q_compile["train"]
            df_val = df_splits_q_compile["val"]
            df_test = df_splits_q_compile["test"]
            # peek queries incrementally upto ID (90%)
            # so need to make sure the remaining 10% of ID is always in the test set
            loaded = PickleHandler.load(pw.cc_prefix, "index_splits_q_compile.pkl")
            index_dict_in_dist = cast(Dict[str, List[str]], loaded)
            split_to_peek_aids: Dict[str, List[str]] = {"val": [], "test": []}
            for split, df_ in zip(["val", "test"], [df_val, df_test]):
                tid_to_aids = df_.groupby("tid")["appid"].apply(list).to_dict()
                for tid, aids in tid_to_aids.items():
                    # total numbers to peek for tid
                    n_peek = int(len(aids) * pw.fold_peek_percentage / 100)
                    remaining_aids = sorted(
                        list(set(aids) - set(index_dict_in_dist[split]))
                    )
                    if n_peek > len(remaining_aids):
                        raise Exception(f"not enough aids to peek for {split}")
                    np.random.seed(seed)
                    np.random.shuffle(remaining_aids)
                    split_to_peek_aids[split] += remaining_aids[:n_peek]
            df_splits_q_compile = {
                "train": pd.concat(
                    [
                        df_train,
                        df_val.loc[split_to_peek_aids["val"]],
                        df_test.loc[split_to_peek_aids["test"]],
                    ]
                ),
                "val": df_val[~df_val["appid"].isin(split_to_peek_aids["val"])],
                "test": df_test[~df_test["appid"].isin(split_to_peek_aids["test"])],
            }

    index_splits_q_compile = {
        split: df.index.to_list() for split, df in df_splits_q_compile.items()
    }
    index_splits_qs = {
        split: df_qs[df_qs.appid.isin(appid_list)].index.to_list()
        for split, appid_list in index_splits_q_compile.items()
    }
    index_splits_q = {
        split: df_q[df_q.appid.isin(appid_list)].index.to_list()
        for split, appid_list in index_splits_q_compile.items()
    }
    # Save the index_splits
    suffix = ""
    if pw.fold:
        suffix = f"-{pw.fold}"
        if pw.fold_peek_percentage > 0:
            suffix += f"_peek_{pw.fold_peek_percentage}_percents"
    save_and_log_index(
        index_splits_q_compile, pw, f"index_splits_q_compile{suffix}.pkl"
    )
    save_and_log_index(index_splits_q, pw, f"index_splits_q_all{suffix}.pkl")
    save_and_log_index(index_splits_qs, pw, f"index_splits_qs{suffix}.pkl")


def tpc_setup_compile_only(pw: PathWatcher, seed: int) -> None:
    def get_ext_df_q_compile() -> pd.DataFrame:
        sc = SparkConf(str(pw.base_dir / "assets/spark_configuration_aqe_on.json"))
        df_raw = pd.read_csv(pw.get_ext_data_header("q"), low_memory=False)
        if pw.benchmark_ext is None:
            raise ValueError("benchmark_ext must be specified for EXT data")
        df_q = prepare_data(df_raw, sc, pw.benchmark_ext, "q", ext=None)
        df_q_compile_k_ = df_q[df_q["lqp_id"] == 0].copy()
        df_q_compile_k_["template"] = df_q_compile_k_["template"].astype(str)
        return df_q_compile_k_

    # prepare data
    try:
        df_q_compile = ParquetHandler.load(pw.cc_prefix, "df_q_compile.parquet")
    except Exception as e:
        logger.warning(f"Failed to load df_q_compile from cache: {e}")
        df_dict = {}
        for k in ["ORI", "EXT"]:
            if k == "ORI":
                df_q_compile_k = ParquetHandler.load(
                    pw.get_cc_prefix(False), "df_q_compile.parquet"
                )
            else:
                df_q_compile_k = get_ext_df_q_compile()
            df_dict[k] = df_q_compile_k
        df_q_compile = pd.concat(df_dict.values())
        save_and_log_df(df_q_compile, ["appid"], pw, "df_q_compile")
        logger.info("Saved and loaded df_q_compile_* from cache")

    suffix = "" if pw.fold is None else f"-{pw.fold}"
    index_splits_name = f"index_splits_q_compile{suffix}.pkl"
    index_splits_q_compile = PickleHandler.load(
        pw.get_cc_prefix(False), index_splits_name
    )
    if not isinstance(index_splits_q_compile, Dict):
        raise TypeError(
            f"index_splits_q_compile is not a dict: {index_splits_q_compile}"
        )

    df_q_compile_ext = get_ext_df_q_compile()
    logger.info(f"EXT data for compile #: {len(df_q_compile_ext)}")
    Counter(df_q_compile_ext["tid"])

    if pw.benchmark_ext == BenchmarkType.TPCDS_EXT_SELECTED.value:
        df_train_ext, df_val_ext = train_test_split(
            df_q_compile_ext,
            test_size=0.1,
            stratify=df_q_compile_ext["tid"],
            random_state=seed,
        )
    elif pw.benchmark_ext == BenchmarkType.TPCDS_EXT_STAR_JOINS.value:
        df_train_ext, df_val_ext = train_test_split(
            df_q_compile_ext,
            test_size=0.1,
            stratify=None,
            random_state=seed,
        )
    else:
        raise ValueError(f"invalid benchmark_ext: {pw.benchmark_ext}")
    index_splits_q_compile["train_ext"] = df_train_ext.appid.to_list()
    index_splits_q_compile["val_ext"] = df_val_ext.appid.to_list()
    n_ext_tr = len(df_train_ext)
    n_ext_val = len(df_val_ext)
    logger.info(f"Split data for EXT: train/val = {n_ext_tr}/{n_ext_val}")
    save_and_log_index(index_splits_q_compile, pw, index_splits_name)


def get_ext_df(pw: PathWatcher) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sc = SparkConf(str(pw.base_dir / "assets/spark_configuration_aqe_on.json"))
    df_q_raw = pd.read_csv(pw.get_ext_data_header("q"), low_memory=False)
    df_qs_raw = pd.read_csv(pw.get_ext_data_header("qs"), low_memory=False)
    print(f"raw df_q shape: {df_q_raw.shape}")
    print(f"raw df_qs shape: {df_qs_raw.shape}")
    if pw.benchmark_ext is None:
        raise ValueError("benchmark_ext must be specified for EXT data")

    df_q_ = prepare_data(df_q_raw, sc, pw.benchmark_ext, "q", ext=None)
    df_qs_ = prepare_data(df_qs_raw, sc, pw.benchmark_ext, q_type="qs", ext=None)
    df_q_compile_ = df_q_[df_q_["lqp_id"] == 0].copy()  # for compile-time df
    df_q_compile_["template"] = df_q_compile_["template"].astype(str)

    loc = f"{pw.cc_prefix}/{pw.benchmark_ext}"
    save_and_log_with_path(df_q_compile_, ["appid"], loc, "df_q_compile")
    save_and_log_with_path(df_q_, ["appid", "lqp_id"], loc, "df_q_all")
    save_and_log_with_path(df_qs_, ["appid", "qs_id"], loc, "df_qs")
    return df_q_compile_, df_q_, df_qs_


def job_setup(pw: PathWatcher, seed: int) -> None:
    header = pw.data_prefix
    df_raw_dict = {
        "TRAIN": pd.read_csv(f"{header}/train_q_100000x1.csv", low_memory=pw.debug),
        "SYNTHETIC": pd.read_csv(
            f"{header}/synthetic_q_5000x1.csv", low_memory=pw.debug
        ),
        "LIGHT": pd.read_csv(f"{header}/light_q_70x1.csv", low_memory=pw.debug),
    }
    # prepare data
    try:
        df_q_compile = ParquetHandler.load(pw.cc_prefix, "df_q_compile.parquet")
    except Exception as e:
        logger.warning(f"Failed to load df_q_compile_* from cache: {e}")
        df_dict = {}
        sc = SparkConf(str(pw.base_dir / "assets/spark_configuration_aqe_on.json"))
        if pw.benchmark_ext == BenchmarkType.JOB_EXT.value:
            df_raw_dict["EXT"] = pd.read_csv(
                f"{header}/ext_q_27371x1.csv", low_memory=pw.debug
            )
            logger.info(f"Loaded EXT data #: {len(df_raw_dict['EXT'])}")
        for k, df_raw in df_raw_dict.items():
            if not isinstance(df_raw["template"].iloc[0], str):
                df_raw["template"] = k + df_raw["template"].astype(int).astype(str)
            df_q = prepare_data(df_raw, sc, pw.benchmark, "q", ext=None)
            df_q_compile_k = df_q[df_q["lqp_id"] == 0].copy()  # for compile-time df
            df_dict[k] = df_q_compile_k
        df_q_compile = pd.concat(df_dict.values())
        save_and_log_df(df_q_compile, ["appid"], pw, "df_q_compile")
        logger.info("Saved and loaded df_q_compile_* from cache")
        df_train, df_val = train_test_split(
            df_dict["TRAIN"],
            test_size=0.1,
            stratify=None,
            random_state=seed,
        )
        index_splits_q_compile = {
            "train": df_train.appid.to_list(),
            "val": df_val.appid.to_list(),
            "test": df_dict["SYNTHETIC"].appid.tolist()
            + df_dict["LIGHT"].appid.tolist(),
        }
        if pw.benchmark_ext:
            logger.info(f"EXT data for compile #: {len(df_dict['EXT'])}")
            df_train_ext, df_val_ext = train_test_split(
                df_dict["EXT"], test_size=0.1, stratify=None, random_state=seed
            )
            index_splits_q_compile["train_ext"] = df_train_ext.appid.to_list()
            index_splits_q_compile["val_ext"] = df_val_ext.appid.to_list()
            n_ext_tr = len(df_train_ext)
            n_ext_val = len(df_val_ext)
            logger.info(f"Split data for EXT: train/val = {n_ext_tr}/{n_ext_val}")
        save_and_log_index(index_splits_q_compile, pw, "index_splits_q_compile.pkl")


# Data Split Index
def extract_index_splits(
    pw: PathWatcher, seed: int, q_type: str
) -> Tuple[pd.DataFrame, Dict[DatasetType, List[str]]]:
    if pw.benchmark == "job" and q_type != "q_compile":
        raise NotImplementedError("job benchmark only supports q_compile")

    suffix = ""
    if pw.fold:
        suffix = f"-{pw.fold}"
        if pw.fold_peek_percentage > 0:
            suffix += f"_peek_{pw.fold_peek_percentage}_percents"
    index_splits_name = f"index_splits_{q_type}{suffix}.pkl"
    if (
        not Path(f"{pw.cc_prefix}/{index_splits_name}").exists()
        or not Path(f"{pw.cc_prefix}/df_{q_type}.parquet").exists()
    ):
        logger.info(
            f"not found {index_splits_name} or df_{q_type}.parquet "
            f"under {pw.cc_prefix}, start magic setup..."
        )
        if pw.benchmark != "job":
            if pw.benchmark_ext:
                tpc_setup_compile_only(pw, seed)
            else:
                magic_setup(pw, seed)
        else:
            job_setup(pw, seed)
    else:
        logger.info(f"found {pw.cc_prefix}/df_{q_type}.pkl, loading...")
        logger.info(f"found {pw.cc_prefix}/{index_splits_name}, loading...")

    if not Path(f"{pw.cc_prefix}/{index_splits_name}").exists():
        raise FileNotFoundError(f"{pw.cc_prefix}/{index_splits_name} not found")
    if not Path(f"{pw.cc_prefix}/df_{q_type}.parquet").exists():
        raise FileNotFoundError(f"{pw.cc_prefix}/df_{q_type}.parquet not found")

    df = ParquetHandler.load(pw.cc_prefix, f"df_{q_type}.parquet")
    index_splits = PickleHandler.load(pw.cc_prefix, index_splits_name)
    if not isinstance(index_splits, Dict):
        raise TypeError(f"index_splits is not a dict: {index_splits}")
    return df, index_splits


def extract_index_splits_wrapper(
    pw: PathWatcher, seed: int, q_type: str
) -> Tuple[pd.DataFrame, Union[Dict[str, Dict], Dict[str, List[str]]]]:
    if "+" in pw.benchmark:
        benchmarks = pw.benchmark.split("+")
        dfs, index_splits_list = zip(
            *[
                extract_index_splits(
                    PathWatcher(
                        pw.base_dir,
                        bm,
                        pw.debug,
                        pw.extract_params,
                        None,
                        pw.data_percentage,
                        pw.benchmark_ext,
                        pw.ext_data_amount,
                    )
                    if bm == "job"
                    else PathWatcher(
                        pw.base_dir,
                        bm,
                        pw.debug,
                        pw.extract_params,
                        pw.fold if bm == "tpcds" else None,
                        pw.data_percentage2 if bm == "tpcds" else None,
                        pw.benchmark_ext2 if bm == "tpcds" else None,
                        pw.ext_data_amount2 if bm == "tpcds" else None,
                    ),
                    seed,
                    q_type,
                )
                for bm in benchmarks
            ]
        )
        df = pd.concat(dfs)
        index_splits = {
            bm: index_splits_
            for bm, index_splits_ in zip(benchmarks, index_splits_list)
        }
        return df, index_splits
    else:
        return extract_index_splits(pw=pw, seed=seed, q_type=q_type)


def count_joins(lqp: str) -> int:
    return len(
        [
            k
            for k, v in JsonHandler.load_json_from_str(lqp)["operators"].items()
            if v["className"].split(".")[-1] == "Join"
        ]
    )


def get_ext_joins(bm_ext: str) -> Dict[str, int]:
    if bm_ext == "tpcds-ext-star-joins":
        ds_ext = ParquetHandler.load(
            "cache_and_ckp/tpcds_102x490+ext_star_joins_5518x10/tpcds-ext-star-joins",
            "df_q_compile.parquet",
        )
        ds_ext_to_njoins = ds_ext["lqp"].apply(lambda x: count_joins(x)).to_dict()
        return ds_ext_to_njoins  # type: ignore
    elif bm_ext == "job-ext":
        job = ParquetHandler.load("cache_and_ckp/job_100000x1", "df_q_compile.parquet")
        job_all = ParquetHandler.load(
            "cache_and_ckp/job_100000x1+ext_27371x1", "df_q_compile.parquet"
        )
        job_ext = job_all[~job_all.index.isin(job.index)]
        job_ext_to_njoins = job_ext["lqp"].apply(lambda x: count_joins(x)).to_dict()
        return job_ext_to_njoins  # type: ignore
    else:
        raise ValueError(f"unsupported benchmark_ext: {bm_ext}")


def aggregate_index_splits(
    index_splits: Dict[str, List[str]],
    data_percentage: Optional[int],
    benchmark_ext: Optional[str],
    ext_data_amount: Optional[int],
    ext_up_to_n_joins: Optional[int],
) -> Dict[str, List[str]]:
    if data_percentage is not None:
        index_splits = {
            k: v[: int(np.ceil(len(v) * data_percentage / 100))]
            if k in ["train", "val"]
            else v
            for k, v in index_splits.items()
        }
    if benchmark_ext is not None:
        if ext_data_amount is None:
            raise ValueError(
                "ext_data_amount must be specified when benchmark_ext is specified"
            )
        else:
            if benchmark_ext == "job-ext" and ext_data_amount > 27371:
                raise ValueError("ext_data_amount must be less than 27371")
            else:
                logger.info(
                    f"ext_data_amount = {ext_data_amount},"
                    f"ttl_ext_tr = {len(index_splits['train_ext'])},"  # type: ignore
                    f"ttl_ext_val = {len(index_splits['val_ext'])}"  # type: ignore
                )
        logger.info(
            f"Before extending data, tr/val = "
            f"{len(index_splits['train'])}/{len(index_splits['val'])}"
        )
        val_amount = int(np.ceil(ext_data_amount * 0.1))
        train_amount = ext_data_amount - val_amount
        logger.info("Target extended data, tr/val = " f"{train_amount}/{val_amount}")
        train_ext_list = index_splits["train_ext"][:train_amount]  # type: ignore
        val_ext_list = index_splits["val_ext"][:val_amount]  # type: ignore
        if ext_up_to_n_joins is not None:
            filtered = set(
                [
                    k
                    for k, v in get_ext_joins(benchmark_ext).items()
                    if v <= ext_up_to_n_joins
                ]
            )
            train_ext_list = [i for i in train_ext_list if i in filtered]
            val_ext_list = [i for i in val_ext_list if i in filtered]
            logger.info(
                "After filtering by n_joins, tr/val = "
                f"{len(train_ext_list)}/{len(val_ext_list)}"
            )

        logger.info(
            f"Get extended data, tr/val = {len(train_ext_list)}/{len(val_ext_list)}"
        )
        index_splits["train"] += train_ext_list
        index_splits["val"] += val_ext_list
        del index_splits["train_ext"]  # type: ignore
        del index_splits["val_ext"]  # type: ignore
        logger.info(
            "After extending data, tr/val = "
            f"{len(index_splits['train'])}/{len(index_splits['val'])}"
        )
    return index_splits


def extract_and_save_iterators(
    pw: PathWatcher,
    ta: TypeAdvisor,
    tensor_dtypes: th.dtype,
    cache_file: str = "split_iterators.pkl",
    hists: Optional[Dict[Tuple[str, str], np.ndarray]] = None,
    table_samples: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[DatasetType, BaseIterator]:
    if "job" in pw.benchmark and ta.q_type != "q_compile":
        raise NotImplementedError("job benchmark only supports q_compile")

    params = pw.extract_params
    if Path(f"{pw.cc_extract_prefix}/{cache_file}").exists():
        raise FileExistsError(f"{pw.cc_extract_prefix}/{cache_file} already exists.")
    logger.info("start extracting split_iterators")
    df, index_splits = extract_index_splits_wrapper(
        pw=pw, seed=params.seed, q_type=ta.get_q_type_for_cache()
    )
    if len(pw.benchmark.split("+")) > 1:
        """
        job has the priority to
        get the data_percentage, benchmark_ext, ext_data_amount
        """
        if pw.benchmark.endswith("+job"):
            index_splits_job = aggregate_index_splits(
                index_splits=index_splits["job"],  # type: ignore
                data_percentage=pw.data_percentage,
                benchmark_ext=pw.benchmark_ext,
                ext_data_amount=pw.ext_data_amount,
                ext_up_to_n_joins=pw.ext_up_to_n_joins,
            )
            index_splits_bm_list = []
            if len(pw.benchmark.split("+")) == 2:  # tpcds+job
                bm_tpc = pw.benchmark.split("+")[0]
                if bm_tpc == "tpch":
                    raise ValueError(f"unsupported benchmark: {pw.benchmark}")
                index_splits_bm_list.append(
                    aggregate_index_splits(
                        index_splits=index_splits[bm_tpc],  # type: ignore
                        data_percentage=pw.data_percentage2,
                        benchmark_ext=pw.benchmark_ext2,
                        ext_data_amount=pw.ext_data_amount2,
                        ext_up_to_n_joins=pw.ext_up_to_n_joins2,
                    )
                )
            else:
                for bm_tpc in pw.benchmark.split("+")[:-1]:  # tpcds+tpch+job
                    index_splits_bm_list.append(
                        aggregate_index_splits(
                            index_splits=index_splits[bm_tpc],  # type: ignore
                            data_percentage=pw.data_percentage2
                            if bm_tpc == "tpcds"
                            else None,
                            benchmark_ext=pw.benchmark_ext2
                            if bm_tpc == "tpcds"
                            else None,
                            ext_data_amount=pw.ext_data_amount2
                            if bm_tpc == "tpcds"
                            else None,
                            ext_up_to_n_joins=pw.ext_up_to_n_joins2
                            if bm_tpc == "tpcds"
                            else None,
                        )
                    )

            index_splits = {
                split: list(
                    itertools.chain.from_iterable(
                        index_splits_bm[split]
                        for index_splits_bm in index_splits_bm_list
                    )
                )
                + index_splits_  # type: ignore
                for split, index_splits_ in index_splits_job.items()
            }
        else:
            if pw.benchmark != "tpcds+tpch":
                raise ValueError(f"unsupported benchmark: {pw.benchmark}")
            index_splits_ds = aggregate_index_splits(
                index_splits=index_splits["tpcds"],  # type: ignore
                data_percentage=pw.data_percentage,
                benchmark_ext=pw.benchmark_ext,
                ext_data_amount=pw.ext_data_amount,
                ext_up_to_n_joins=pw.ext_up_to_n_joins,
            )
            index_splits_h = aggregate_index_splits(
                index_splits=index_splits["tpch"],  # type: ignore
                data_percentage=None,
                benchmark_ext=None,
                ext_data_amount=None,
                ext_up_to_n_joins=None,
            )
            index_splits = {
                split: index_splits_ds[split] + index_splits_h[split]
                for split in ["train", "val", "test"]
            }
    else:
        if not index_splits:
            raise ValueError(f"index_splits is empty: {index_splits}")
        if not isinstance(list(index_splits.values())[0], List):
            raise TypeError(f"index_splits is not a list: {index_splits} for job")
        index_splits = aggregate_index_splits(
            index_splits=index_splits,  # type: ignore
            data_percentage=pw.data_percentage,
            benchmark_ext=pw.benchmark_ext,
            ext_data_amount=pw.ext_data_amount,
            ext_up_to_n_joins=pw.ext_up_to_n_joins,
        )

    logger.info("index_splits:")
    for split, index_list in index_splits.items():
        logger.info(f"{split} #: {len(index_list)}")

    df = df.loc[list(itertools.chain.from_iterable(index_splits.values()))]

    cache_file_dp = "data_processor.pkl"
    if Path(f"{pw.cc_extract_prefix}/{cache_file_dp}").exists():
        raise FileExistsError(f"{pw.cc_extract_prefix}/{cache_file_dp} already exists.")
    logger.info("start creating data_processor")
    data_processor = create_udao_data_processor(
        ta=ta,
        sc=SparkConf(str(pw.base_dir / "assets/spark_configuration_aqe_on.json")),
        lpe_size=params.lpe_size,
        vec_size=params.vec_size,
        tensor_dtypes=tensor_dtypes,
        hists=hists,
        table_samples=table_samples,
    )
    data_handler = DataHandler(
        df.reset_index(),
        DataHandler.Params(
            index_column="id",
            stratify_on="tid",
            val_frac=0.2 if params.debug else 0.1,
            test_frac=0.2 if params.debug else 0.1,
            dryrun=False,
            data_processor=data_processor,
            random_state=params.seed,
        ),
    )
    data_handler.index_splits = index_splits
    logger.info("extracting split_iterators...")
    split_iterators = data_handler.get_iterators()

    PickleHandler.save(data_processor, pw.cc_extract_prefix, cache_file_dp)
    logger.info(f"saved {pw.cc_extract_prefix}/{cache_file_dp} after fitting")
    PickleHandler.save(split_iterators, pw.cc_extract_prefix, cache_file)
    logger.info(f"saved {pw.cc_extract_prefix}/{cache_file}")
    return split_iterators


def get_hist(path: str) -> Dict[Tuple[str, str], np.ndarray]:
    df = pd.read_csv(path)
    hists = {
        (row["table"], row["column"]): np.array(
            list(map(float, row["hists"][1:-1].split(", ")))
        )
        for _, row in df.iterrows()
    }
    return hists


def get_bitmap(path: str) -> Dict[str, np.ndarray]:
    bitmap = PickleHandler.load(
        header=os.path.dirname(path),
        file_name=os.path.basename(path),
    )  # type: ignore
    return bitmap  # type: ignore


def get_split_iterators(
    pw: PathWatcher,
    ta: TypeAdvisor,
    tensor_dtypes: th.dtype,
    hists: Optional[Dict[Tuple[str, str], np.ndarray]] = None,
    table_samples: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[DatasetType, BaseIterator]:
    cache_file = "split_iterators.pkl"
    bm = pw.benchmark

    if "job" in bm and ta.q_type != "q_compile":
        raise NotImplementedError("job benchmark only supports q_compile")

    data_stats_header = f"{pw.base_dir}/assets/data_stats"
    if hists is None:
        if "+" in bm:
            hists = {}
            for bm_ in bm.split("+"):
                hists_ = get_hist(f"{data_stats_header}/regrouped_{bm_}_hist.csv")
                hists_ = {
                    (bm + "." + table, column): hist
                    for (table, column), hist in hists_.items()
                }
                hists.update(hists_)
        else:
            hists = get_hist(f"{data_stats_header}/regrouped_{bm}_hist.csv")

    if table_samples is None:
        if "+" in bm:
            table_samples = {}
            for bm_ in bm.split("+"):
                dbname = "job" if bm_ == "job" else f"{bm_}_100"
                bitmap_name = f"{dbname}_samples.pkl"
                table_samples_ = PickleHandler.load(data_stats_header, bitmap_name)
                if not isinstance(table_samples_, Dict):
                    raise TypeError(f"table_samples_ is not a dict: {table_samples_}")
                table_samples_ = {
                    dbname + "." + table: df for table, df in table_samples_.items()
                }
                table_samples.update(table_samples_)
        else:
            dbname = "job" if bm == "job" else f"{bm}_100"
            bitmap_name = f"{dbname}_samples.pkl"
            table_samples = PickleHandler.load(data_stats_header, bitmap_name)  # type: ignore
            if not isinstance(table_samples, Dict):
                raise TypeError(f"table_samples is not a dict: {table_samples}")
            table_samples = {
                dbname + "." + table: df for table, df in table_samples.items()
            }
    if not Path(f"{pw.cc_extract_prefix}/{cache_file}").exists():
        return extract_and_save_iterators(
            pw=pw,
            ta=ta,
            tensor_dtypes=tensor_dtypes,
            cache_file=cache_file,
            hists=hists,
            table_samples=table_samples,
        )
    split_iterators = PickleHandler.load(pw.cc_extract_prefix, cache_file)
    if not isinstance(split_iterators, Dict):
        raise TypeError("split_iterators not found or not a desired type")
    return split_iterators


def get_lhs_confs(
    spark_conf: SparkConf, n_samples: int, seed: int, normalize: bool
) -> pd.DataFrame:
    lhs_conf_raw = spark_conf.get_lhs_configurations(n_samples, seed=seed)
    lhs_conf = pd.DataFrame(
        data=spark_conf.deconstruct_configuration(lhs_conf_raw.values),
        columns=spark_conf.knob_ids,
    )
    if normalize:
        theta_all_minmax = (
            np.array(spark_conf.knob_min),
            np.array(spark_conf.knob_max),
        )
        lhs_conf_norm = (lhs_conf - theta_all_minmax[0]) / (
            theta_all_minmax[1] - theta_all_minmax[0]
        )
        return lhs_conf_norm
    return lhs_conf


def wrap_to_df(data: Dict[str, Dict[str, List[float]]]) -> pd.DataFrame:
    df_data = []
    for key, values in data.items():
        row = {}
        for metric, stats in values.items():
            row[f"{metric}_mu"] = stats[0]
            row[f"{metric}_std"] = stats[1]
        df_data.append(row)

    # Create the DataFrame
    df = pd.DataFrame(df_data, index=list(data.keys()))
    return df
