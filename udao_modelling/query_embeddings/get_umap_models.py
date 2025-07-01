from pathlib import Path
from udao_modelling.mask.models import set_probability
from typing import Annotated

import torch as th
import typer
from udao.utils.logging import logger

from udao_modelling.models import (
    Model,
    get_model_and_learning_parameters,
)

import polars as pl
import numpy as np
import polars.selectors as cs
from scipy.spatial import distance_matrix
import pickle
from sklearn.decomposition import PCA

from udao_modelling.params import get_parameters
from udao_modelling.training import setup
from udao_modelling.data import BenchmarkDataset
from udao_spark.data.utils import checkpoint_model_structure
from udao_spark.model.utils import (
    param_init,
)
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

    # Load embeddings

    logger.warning("Loading embeddings")
    train_embeddings = pl.read_parquet(f"{ckp_header}/train_embeddings.parquet")
    x = train_embeddings.select(cs.contains("emb")).to_numpy()
    # TODO (glachaud): change name to match current naming convention (not stable)
    unseen_embeddings = pl.read_parquet(
        f"{ckp_header}/unseen_queries_embeddings.parquet"
    )
    u = unseen_embeddings.select(cs.contains("emb")).to_numpy()
    logger.warning("Embeddings loaded")

    # Compute approximate optimal number of components
    logger.warning("Computing PCA to find suitable number of components")
    pca = PCA()
    pca.fit(x)
    best_n_components = (
        np.where(np.cumsum(pca.explained_variance_ratio_) > 0.9)[0][0] + 1
    )
    logger.warning(f"Best number of components is: {best_n_components}")

    # Train UMAP using number of components obtained from PCA
    logger.warning(f"PComputing UMAP projection with {best_n_components}")
    mapper = umap.UMAP(n_neighbors=100, n_components=best_n_components)
    x_emb = mapper.fit_transform(x)
    Path(f"{ckp_header}/umap_best_{best_n_components}.pkl").write_bytes(
        pickle.dumps(mapper)
    )
    logger.warning(f"Saved model with n_components = {best_n_components}")

    # Compute distance between unseen templates and training points
    logger.warning("Projecting unseen queries with UMAP")
    u_emb = mapper.transform(u)
    logger.warning("Computing distance between unseen queries and training queries")
    dist = distance_matrix(u_emb, x_emb)
    logger.warning(
        "Finished computing distance between unseen queries and training queries"
    )
    # compute the average distance to the 100 closest points
    NEAREST_NEIGHBOURS = 100
    dist = np.sort(dist, axis=1)[:, :NEAREST_NEIGHBOURS].mean(axis=1)
    unseen_embeddings = unseen_embeddings.with_columns(
        **{f"umap_{best_n_components}": dist}
    )
    logger.warning(f"Finished saving distance for n_components={best_n_components}")
    unseen_embeddings.write_parquet(f"{ckp_header}/unseen_templates_distance.parquet")
    logger.warning("Finished computing distance for all UMAP models")


if __name__ == "__main__":
    app()
