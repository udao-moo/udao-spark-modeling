import pathlib
import pickle
from typing import Annotated, Dict, List, Optional

# TODO (glachaud): I can't install polars with the current constraints imposed by `udao`.
import polars as pl
import tomlkit
import attrs
from cattrs import structure
import typer

from ._result import (
    Metrics,
    Result,
    TestPickleFile,
    TestPickleFileWithDropMetrics,
    ValPickleFile,
)


class ExtractionConstants:
    TEST_FILE_GLOB = "**/obj_df_test_tpcds_cpu.pkl"
    VALIDATION_RESULTS_GLOB = "obj_df_val_with_test_latency_s*"

    @classmethod
    def set_file_glob(cls, test_glob: str) -> None:
        cls.TEST_FILE_GLOB = test_glob


app = typer.Typer()


def init_dict() -> Dict[str, List]:
    """Initialize dictionary that contains all results"""
    return {
        "val_wmape": [],
        "wmape": [],
        "p50_wape": [],
        "p90_wape": [],
        "model": [],
        "fold": [],
        "run": [],
        "test_type": [],
    }


def add_to_dict(all_results: Dict, result: Result) -> Dict[str, List]:
    """Add the results of a model run to dict"""
    result_dict = all_results.copy()
    result_dict["val_wmape"].append(result.val_wmape)
    result_dict["wmape"].append(result.wmape)
    result_dict["p50_wape"].append(result.p50_wape)
    result_dict["p90_wape"].append(result.p90_wape)
    result_dict["model"].append(result.model)
    result_dict["fold"].append(result.fold)
    result_dict["run"].append(result.run)
    result_dict["test_type"].append(result.test_type)
    return result_dict


def structure_metric_data(
    base_path: pathlib.Path,
    path: pathlib.Path,
    metrics: Metrics,
    test_type: str,
    val_wmape: float,
) -> Result:
    """Add model info to result metrics"""
    latency_metrics = metrics.latency_s
    result_path = path.relative_to(base_path)
    fold, model, run, _ = result_path.parts
    latency_metrics.fold = fold
    latency_metrics.model = model
    latency_metrics.run = run
    latency_metrics.test_type = test_type
    latency_metrics.val_wmape = val_wmape
    return latency_metrics


def retrieve_val_wmape(path: pathlib.Path) -> float:
    validation_path = next(
        iter(path.parent.glob(ExtractionConstants.VALIDATION_RESULTS_GLOB))
    )
    validation_data = pickle.loads(validation_path.read_bytes())
    validation_structured = structure(validation_data, ValPickleFile)
    return validation_structured.metrics.latency_s.wmape


# TODO (glachaud): the conditional statement is making it complex
# TODO (glachaud): Ideally, I would just have two different `extract_single_run`, based on the dataset
def extract_single_run(
    all_results: dict,
    base_path: pathlib.Path,
    path: pathlib.Path,
    *,
    use_drop_metrics: bool,
):
    result_dict = all_results.copy()
    results = pickle.loads(path.read_bytes())
    val_wmape = retrieve_val_wmape(path)
    if use_drop_metrics:
        structured_results = structure(results, TestPickleFileWithDropMetrics)
        metrics = structure_metric_data(
            base_path, path, structured_results.metrics, "metrics", val_wmape
        )
        drop1_metrics = structure_metric_data(
            base_path,
            path,
            structured_results.drop1_metrics,
            "drop1_metrics",
            val_wmape,
        )
        drop2_metrics = structure_metric_data(
            base_path,
            path,
            structured_results.drop2_metrics,
            "drop2_metrics",
            val_wmape,
        )
        result_dict = add_to_dict(result_dict, metrics)
        result_dict = add_to_dict(result_dict, drop1_metrics)
        result_dict = add_to_dict(result_dict, drop2_metrics)
    else:
        structured_results = structure(results, TestPickleFile)
        metrics = structure_metric_data(
            base_path, path, structured_results.metrics, "metrics", val_wmape
        )
        result_dict = add_to_dict(result_dict, metrics)
    return result_dict


@attrs.define
class Params:
    path: pathlib.Path
    output_path: pathlib.Path
    test_glob: Optional[str]
    use_drop_metrics: bool = False
    base_path: pathlib.Path = pathlib.Path(".")


@app.command()
def extract_results(
    config_file: Annotated[str, typer.Argument(help="Location of config file")],
) -> pl.DataFrame:
    config = tomlkit.loads(pathlib.Path(config_file).read_text())
    params = structure(config, Params)
    result_dict = init_dict()
    if params.test_glob is not None:
        ExtractionConstants.set_file_glob(params.test_glob)

    result_paths = params.path.glob(ExtractionConstants.TEST_FILE_GLOB)
    for path in result_paths:
        result_dict = extract_single_run(
            result_dict,
            params.base_path,
            path,
            use_drop_metrics=params.use_drop_metrics,
        )
    result_df = pl.DataFrame(result_dict)

    print(result_df)
    if params.output_path is not None:
        result_df = result_df.unique()
        result_df.write_parquet(params.output_path)


if __name__ == "__main__":
    app()
