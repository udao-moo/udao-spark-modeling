from pathlib import Path
from udao_modelling.mask.models import MaskProbability, set_probability
from typing import Annotated

import torch as th
import typer
from udao.utils.logging import logger

from udao_modelling.models import (
    Model,
    check_loss_objective_consistency,
    get_model_and_learning_parameters,
)
from udao_modelling.params import get_parameters
from udao_modelling.training import setup
from udao_modelling.data import BenchmarkDataset
from udao_spark.model.utils import (
    param_init,
    train_and_dump,
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
    fold: Annotated[int, typer.Option(help="Fold (for tpcds). 0 is treated as in-distribution.")] = 0,
    mask: Annotated[float, typer.Option(help="Probability of masking")] = 0.,
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

    train_and_dump(
        ta=type_advisor,
        pw=path_watcher,
        model=model,
        split_iterators=split_iterators,
        extract_params=path_watcher.extract_params,
        model_params=model_params,
        learning_params=learning_params,
        params=params,
        device=device,
    )


if __name__ == "__main__":
    app()
