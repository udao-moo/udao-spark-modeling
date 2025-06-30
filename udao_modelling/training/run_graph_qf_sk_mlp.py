from pathlib import Path
from typing import Annotated

import torch as th
import typer
from udao.data.handler.data_processor import DataProcessor
from udao.utils.logging import logger
from udao_spark.model.utils import (
    add_dist_to_graph,
    add_height_to_graph,
    add_new_rows_for_df,
    add_new_rows_for_series,
    add_super_node_to_graph,
    param_init,
    train_and_dump,
    update_dgl_graphs,
)
from udao_trace.utils import PickleHandler

from udao_modelling.params import get_parameters
from udao_modelling.training import setup
from udao_modelling.data import BenchmarkDataset
from udao_modelling.models import (
    Model,
    check_loss_objective_consistency,
    get_model_and_learning_parameters,
)

logger.setLevel("INFO")
app = typer.Typer()


@app.command()
def main(
    model_name: Annotated[str, typer.Argument(help="Name of graph embedding model")],
    benchmark_name: Annotated[str, typer.Argument(help="Name of benchmark")],
    config: Annotated[str, typer.Argument(help="Location of config file")],
    fold: Annotated[
        int, typer.Option(help="Fold (for tpcds). 0 is treated as in-distribution.")
    ] = 0,
) -> None:
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

    dp = PickleHandler.load(path_watcher.cc_extract_prefix, "data_processor.pkl")
    if not isinstance(dp, DataProcessor):
        raise TypeError(f"Expected DataProcessor, got {type(dp)}")
    template_plans = dp.feature_extractors["query_structure"].template_plans
    template_plans = update_dgl_graphs(
        template_plans,
        funcs=[add_super_node_to_graph, add_height_to_graph, add_dist_to_graph],
    )

    target_template_plan_ids = (
        list(split_iterators["train"].query_structure_container.template_plans.keys())
        + list(split_iterators["val"].query_structure_container.template_plans.keys())
        + list(split_iterators["test"].query_structure_container.template_plans.keys())
    )
    max_height = max(
        [
            template_plans[t].graph.ndata["height"].max()
            for t in target_template_plan_ids
        ]
    ).item()
    max_dist = max(
        [
            template_plans[t].graph.edata["dist"].max().item()
            for t in target_template_plan_ids
        ]
    )
    supper_gid = len(dp.feature_extractors["query_structure"].operation_types)

    for k, v in split_iterators.items():
        split_iterators[k].query_structure_container.template_plans = template_plans
        operation_types = split_iterators[k].query_structure_container.operation_types
        graph_features = split_iterators[k].query_structure_container.graph_features
        other_graph_features = split_iterators[k].other_graph_features

        operation_types = add_new_rows_for_series(operation_types, supper_gid)
        graph_features = add_new_rows_for_df(
            graph_features, [0] * len(graph_features.columns)
        )
        for exist_keys in ["op_enc", "hist", "bitmap"]:
            other_graph_features[exist_keys].data = add_new_rows_for_df(
                other_graph_features[exist_keys].data,
                [0] * len(other_graph_features[exist_keys].data.columns),
            )

        split_iterators[k].query_structure_container.operation_types = operation_types
        split_iterators[k].query_structure_container.graph_features = graph_features
        split_iterators[k].other_graph_features = other_graph_features

    params.max_height = max_height
    params.max_dist = max_dist
    # Model definition and training
    # model_params = GraphTransformerSKMLPParams.from_dict(
    #     {
    #         "iterator_shape": split_iterators["train"].shape,
    #         "op_groups": params.op_groups,
    #         "output_size": params.output_size,
    #         "pos_encoding_dim": params.pos_encoding_dim,
    #         "gtn_n_layers": params.gtn_n_layers,
    #         "gtn_n_heads": params.gtn_n_heads,
    #         "readout": params.readout,
    #         "type_embedding_dim": params.type_embedding_dim,
    #         "hist_embedding_dim": params.hist_embedding_dim,
    #         "bitmap_embedding_dim": params.bitmap_embedding_dim,
    #         "embedding_normalizer": params.embedding_normalizer,
    #         "gtn_dropout": params.gtn_dropout,
    #         "n_layers": params.n_layers,
    #         "hidden_dim": params.hidden_dim,
    #         "dropout": params.dropout,
    #         "use_batchnorm": params.use_batchnorm,
    #         "activate": params.activate,
    #         "attention_layer_name": "QF",
    #         "max_dist": max_dist,
    #         "max_height": max_height,
    #     }
    # )

    # if params.loss_weights is not None:
    #    if len(params.loss_weights) != len(type_advisor.get_objectives()):
    #        raise ValueError(
    #            f"loss_weights must have the same length as objectives, "
    #            f"got {len(params.loss_weights)} and {len(type_advisor.get_objectives())}"
    #        )

    # learning_params = MyLearningParams.from_dict(
    #    {
    #        "epochs": params.epochs,
    #        "batch_size": params.batch_size,
    #        "init_lr": params.init_lr,
    #        "min_lr": params.min_lr,
    #        "weight_decay": params.weight_decay,
    #        "loss_weights": params.loss_weights,
    #    }
    # )
    model_params, learning_params = get_model_and_learning_parameters(
        params, model_factory, split_iterators["train"].shape
    )
    check_loss_objective_consistency(params.loss_weights, type_advisor.get_objectives())
    model = model_factory.model_creator(model_params)

    # model = get_graph_transformer_sk_mlp(model_params)

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
