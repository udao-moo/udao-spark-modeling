from enum import Enum


class VarTypes(Enum):
    INT = "int"
    BOOL = "bool"
    CATEGORY = "category"
    FLOAT = "float"


class ScaleTypes(Enum):
    LOG = "log"
    LINEAR = "linear"


class BenchmarkType(Enum):
    TPCH = "tpch"
    TPCDS = "tpcds"
    TPCDS_EXT = "tpcds-ext"
    TPCDS_EXT_SELECTED = "tpcds-ext-selected"
    TPCDS_EXT_STAR_JOINS = "tpcds-ext-star-joins"
    TPCXBB = "tpcxbb"
    JOB_LIGHT = "job-light"
    JOB_SYNTHETIC = "job-synthetic"
    JOB_TRAIN = "job-train"
    JOB_EXT = "job-ext"
    JOB = "job"


class ClusterName(Enum):
    HEX1 = "hex1"
    HEX2 = "hex2"
    HEX3 = "hex3"
