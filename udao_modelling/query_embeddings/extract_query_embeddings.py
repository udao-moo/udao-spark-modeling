from pathlib import Path
from udao_modelling.mask.models import set_probability
from typing import Annotated, cast

import torch as th
import typer
from udao.utils.logging import logger

from udao_modelling.models import (
    Model,
    check_loss_objective_consistency,
    get_model_and_learning_parameters,
)
from udao.data import QueryPlanIterator

import pyarrow as pa
import pyarrow.parquet as pq

from udao_modelling.params import get_parameters
from udao_modelling.training import setup
from udao_modelling.data import BenchmarkDataset
from udao_spark.data.utils import checkpoint_model_structure
from udao_spark.model.utils import (
    param_init,
    get_tuned_trainer,
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

    for split, iterator in split_iterators.items():
        if split == "train":
            iterator.set_augmentations([])
        iterator = cast(QueryPlanIterator, iterator)
        dataloader = iterator.get_dataloader(
            batch_size=5000,
            num_workers=params.num_workers,
            shuffle=False,
        )
        all_embeddings = []
        for batch_id, (feature, y) in enumerate(dataloader):
            with th.no_grad():
                embedding = (
                    module.model.embedder(feature.embedding_input).detach().cpu()
                )
            all_embeddings.append(embedding)
        embedding = th.cat(all_embeddings, dim=0).numpy()
        indices = iterator.objectives.data.index
        embedding_table = pa.table(
            [indices.tolist(), *embedding.transpose()],
            names=["appid", *[f"emb-{i}" for i in range(embedding.shape[1])]],
        )
        pq.write_table(
            embedding_table,
            f"{ckp_header}/{split}_embeddings.parquet",
            compression=None,
        )


if __name__ == "__main__":
    app()
