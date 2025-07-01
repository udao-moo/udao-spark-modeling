from pathlib import Path
from udao_modelling.mask.models import set_probability
from typing import Annotated, cast

import torch as th
import typer
from udao.utils.logging import logger
from scipy.spatial import distance_matrix
import numpy as np

from udao_modelling.models import (
    Model,
    check_loss_objective_consistency,
    get_model_and_learning_parameters,
)
from udao.data import QueryPlanIterator

import pandas as pd
import polars as pl
import polars.selectors as cs
import pyarrow as pa
import pyarrow.parquet as pq
import pickle

from udao_modelling.params import get_parameters
from udao_modelling.training import setup
from udao_modelling.data import BenchmarkDataset
from udao_spark.data.utils import checkpoint_model_structure
from udao_spark.model.utils import (
    param_init,
    get_tuned_trainer,
)
from udao_trace.utils import PickleHandler
import umap

logger.setLevel("INFO")

app = typer.Typer()


# TODO (glachaud): the approach for masking is not ideal. It should be incorporated in the
# parameters of the model directly.
@app.command()
def main(
    model_name: Annotated[str, typer.Argument(help="Name of graph embedding model")],
    benchmark_name: Annotated[str, typer.Argument(help="Name of benchmark")],
    config: Annotated[str, typer.Argument(help="Location of config file")],
    fold: Annotated[
        int, typer.Option(help="Fold (for tpcds). 0 is treated as in-distribution.")
    ] = 0,
    mask: Annotated[float, typer.Option(help="Probability of masking")] = 0.0,
) -> None:
    if mask > 0:
        set_probability(mask)
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
    model_params, learning_params = get_model_and_learning_parameters(
        params, model_factory, split_iterators["train"].shape
    )
    ckp_header = checkpoint_model_structure(pw=path_watcher, model_params=model_params)

    train_embeddings = pl.read_parquet(f"{ckp_header}/train_embeddings.parquet")
    x = train_embeddings.select(cs.contains("emb")).to_numpy()
    unseen_embeddings = pl.read_parquet(f"{ckp_header}/unseen_templates_embeddings.parquet")
    u = unseen_embeddings.select(cs.contains("emb")).to_numpy()

    for n_components in [2, 3, 4, 5, 6]:
        logger.warn(f"Started saving distance for n_components={n_components}")
        mapper = pickle.loads(Path(f"{ckp_header}/umap_{n_components}.pkl").read_bytes())
        x_emb = mapper.transform(x)
        u_emb = mapper.transform(u)
        dist = distance_matrix(u_emb, x_emb)
        # compute the average distance to the 100 closest points
        NEAREST_NEIGHBOURS = 100
        dist = np.sort(dist, axis=1)[:, :NEAREST_NEIGHBOURS].mean(axis=1)
        unseen_embeddings = unseen_embeddings.with_columns(**{f"umape_{n_components}": dist})
        logger.warn(f"Finished saving distance for n_components={n_components}")
    unseen_embeddings.write_parquet(f"{ckp_header}/unseen_templates_distance.parquet")
    logger.warn(f"Finished computing distance for all UMAP models")

if __name__ == "__main__":
    app()
