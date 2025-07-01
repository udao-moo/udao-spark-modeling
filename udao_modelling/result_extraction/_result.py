from attrs import define


@define
class Result:
    wmape: float
    p50_wape: float
    p90_wape: float
    val_wmape: float = 0
    model: str = ""
    fold: str = ""
    run: str = ""
    test_type: str = ""


@define
class Metrics:
    latency_s: Result
    io_mb: Result


@define
class ValPickleFile:
    metrics: Metrics

@define
class TestPickleFile:
    metrics: Metrics

@define
class TestPickleFileWithDropMetrics:
    metrics: Metrics
    drop1_metrics: Metrics
    drop2_metrics: Metrics
