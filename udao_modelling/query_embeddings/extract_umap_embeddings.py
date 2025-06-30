from pathlib import Path
from udao_modelling.mask.models import set_probability
from typing import Annotated

import torch as th
import typer
import pickle
from udao.utils.logging import logger

from udao_modelling.models import (
    Model,
    check_loss_objective_consistency,
    get_model_and_learning_parameters,
)

import polars as pl
import polars.selectors as cs
import umap

from udao_modelling.params import get_parameters
from udao_modelling.training import setup
from udao_modelling.data import BenchmarkDataset
from udao_spark.data.utils import checkpoint_model_structure
from udao_spark.model.utils import (
    param_init,
)

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
    type_advisor, path_watcher = param_init(Path(__file__).parent, params)
    setup(params.seed, params.benchmark, tensor_dtypes)
    split_iterators = benchmark_factory.get_split_iterators(
        path_watcher, type_advisor, tensor_dtypes
    )
    split_iterators = benchmark_factory.post_processor(split_iterators)

    model_params, learning_params = get_model_and_learning_parameters(
        params, model_factory, split_iterators["train"].shape
    )
    check_loss_objective_consistency(params.loss_weights, type_advisor.get_objectives())

    ckp_header = checkpoint_model_structure(pw=path_watcher, model_params=model_params)
    # `get_tuned_trainer` is a dirty function, but it returns early if the model is found.

    logger.warning("Load embeddings from parquet")
    train_embeddings = pl.read_parquet(Path(ckp_header) / "train_embeddings.parquet")
    test_embeddings = pl.read_parquet(Path(ckp_header) / "test_embeddings.parquet")

    logger.warning("Extract x and y structures")
    x_train = train_embeddings.select(cs.contains("emb")).to_numpy()
    x_test = test_embeddings.select(cs.contains("emb")).to_numpy()

    mapper = umap.UMAP(n_neighbors=100, n_components=2)

    logger.warning("Fit umap to training data")
    train_umap = mapper.fit_transform(x_train)

    logger.warning("Apply fitted umap to test data")
    test_umap = mapper.transform(x_test)

    logger.warning("Save umap embeddings to disk")
    train_umap_path = Path(ckp_header) / "train_umap.pkl"
    test_umap_path = Path(ckp_header) / "test_umap.pkl"
    Path(train_umap_path).write_bytes(pickle.dumps(train_umap))
    Path(test_umap_path).write_bytes(pickle.dumps(test_umap))


if __name__ == "__main__":
    app()
