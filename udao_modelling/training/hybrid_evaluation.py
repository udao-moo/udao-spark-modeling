"""This module tries to combine TPCDS*v1 and TPCDS*v2 for latency prediction.

TPCDS*v1 is our best dataset; TPCDS*v2 is our best augmented one.
The models trained on the first one tend to underestimate the latency.
The models trained on the second one tend to overestimate the latency.

We want to evaluate the test queries with the first model, tend the highest
predictions and feed them to the other model.
"""
import numpy as np

import pickle
from pathlib import Path
import pandas as pd
from udao_modelling.mask.models import MaskProbability, set_probability
from typing import Annotated, cast

import torch as th
import typer
from udao.utils.logging import logger

from udao.data import QueryPlanIterator

from udao_modelling.models import (
    Model,
    check_loss_objective_consistency,
    get_model_and_learning_parameters,
)
from udao_modelling.params import get_parameters
from udao_modelling.training import setup
from udao_modelling.data import BenchmarkDataset
from udao_spark.data.utils import checkpoint_model_structure, prepare_data, save_and_log_with_path
from udao_spark.model.utils import (
    param_init,
    train_and_dump, get_tuned_trainer,
)
from udao_trace.configuration import SparkConf
from udao_trace.utils import PickleHandler, BenchmarkType

logger.setLevel("INFO")

app = typer.Typer()


# TODO (glachaud): the approach for masking is not ideal. It should be incorporated in the
# parameters of the model directly.
@app.command()
def main(
    model_name: Annotated[str, typer.Argument(help="Name of graph embedding model")],
    benchmark_name: Annotated[str, typer.Argument(help="Name of benchmark")],
    config: Annotated[str, typer.Argument(help="Location of config file")],
    predictions: Annotated[str, typer.Option(help="Link to the original predictions")],
    tpcds: Annotated[str, typer.Option(help="Link to TPCDS csv")],
    fold: Annotated[
        int, typer.Option(help="Fold (for tpcds). 0 is treated as in-distribution.")
    ] = 0,
) -> None:
    # I expect the code structure to look a bit like the following
    # 1. first, I load the test results from TPCDSv1 (technically can skip passing the model for now)
    # 2. I filter the highest predictions
    # 3. I load the TPCDS data
    # 4. I create the test set using the data processor from TPCDSv2
    # 5. I use the TPCDSv2 model to make the predictions on this subset
    # 6. Merge the two dataframes
    # 7. Write to disk
    # I overwrite the results from this subset in the original results files
    # I save the new results in a different place (maybe alongside the TPCDSv2 models)

    if predictions is None or tpcds is None:
        raise KeyError("Missing parameter")

    # 1.
    original_results = pickle.loads(Path(predictions).read_bytes())
    original_results = original_results["obj_df"]

    #2.
    # glachaud (pmqslsyp): I removed the code with PERCENTILE. I compute all the results for that.

    #3.
    try:
        filtered_tpcds = pd.read_parquet(f"/mnt/disk1/guillaume/data/tpcds/df_q_compile_{fold}.parquet")
    except FileNotFoundError:
        logger.warning("File doesn't exist. Creating df_q_compile")
        tpcds_data = pd.read_csv(tpcds, low_memory=False)
        tpcds_data = tpcds_data[tpcds_data["appid"].isin(original_results.index.tolist())]
        sc = SparkConf("/mnt/disk1/guillaume/programming/udao-vldb-2025/playground/assets/spark_configuration_aqe_on.json")
        filtered_tpcds = prepare_data(tpcds_data, sc, BenchmarkType.TPCDS, "q", ext=None)
        save_and_log_with_path(
            filtered_tpcds, ["appid"],"/mnt/disk1/guillaume/data/tpcds", f"df_q_compile_{fold}", False
        )
    filtered_tpcds = pd.read_parquet(f"/mnt/disk1/guillaume/data/tpcds/df_q_compile_{fold}.parquet")

    model_factory = Model[model_name]
    benchmark_factory = BenchmarkDataset[benchmark_name]
    params = get_parameters(config, model_factory.cli_params)
    if fold > 0:
        params.fold = fold
    print(params)  # not really needed since we have the config file now
    tensor_dtypes = th.float32
    device = "gpu" if th.cuda.is_available() else "cpu"
    type_advisor, path_watcher = param_init(Path(__file__).parent, params)
    setup(params.seed, params.benchmark, tensor_dtypes)
    split_iterators = benchmark_factory.get_split_iterators(
        path_watcher, type_advisor, tensor_dtypes
    )
    split_iterators = benchmark_factory.post_processor(split_iterators)
    data_processor = PickleHandler.load(
        path_watcher.cc_extract_prefix, "data_processor.pkl"
    )

    filtered_tpcds = filtered_tpcds.reset_index()
    filtered_tpcds.set_index("id", inplace=True, drop=False)

    logger.warning("Creating iterator")
    #4.
    iterator = data_processor.make_iterator(
        keys=filtered_tpcds.index.to_list(), data=filtered_tpcds, split="test"
    )

    model_params, learning_params = get_model_and_learning_parameters(
        params, model_factory, split_iterators["train"].shape
    )
    check_loss_objective_consistency(params.loss_weights, type_advisor.get_objectives())
    model = model_factory.model_creator(model_params)

    ckp_header = checkpoint_model_structure(pw=path_watcher, model_params=model_params)
    # `get_tuned_trainer` is a dirty function, but it returns early if the model is found.
    trainer, module, ckp_learning_header, found = get_tuned_trainer(
        ckp_header,
        model,
        split_iterators,
        type_advisor.get_objectives(),
        learning_params,
        device,
        num_workers=params.num_workers,
    )

    # 5.
    iterator = cast(QueryPlanIterator, iterator)

    logger.warning("Creating dataloader")
    dataloader = iterator.get_dataloader(
        batch_size=5000,
        num_workers=params.num_workers,
        shuffle=False,
    )

    all_pred = []
    for batch_id, (feature, y) in enumerate(dataloader):
        with th.no_grad():
            y_hat = module.model(feature).detach().cpu()
        all_pred.append(y_hat)
        if (batch_id + 1) % 10 == 0:
            print(f"batch {batch_id + 1}/{len(dataloader)} done")
    pred = th.cat(all_pred, dim=0).numpy()

    obj_df = iterator.objectives.data
    obj_names = obj_df.columns.to_list()
    obj_pred_names = [f"{n}_pred" for n in obj_names]
    obj_df[obj_pred_names] = pred

    # 7.
    obj_df.to_csv(f"/mnt/disk1/guillaume/data/model_2/fold_{fold}/augmented_tpcds_results.csv")



if __name__ == "__main__":
    app()
