from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
from tqdm import tqdm

from query_generator.duckdb_connection.utils import (
  RawDuckDBHistograms,
  RawDuckDBMostCommonValues,
  RawDuckDBTableDescription,
  get_columns,
  get_distinct_count,
  get_equi_height_histogram,
  get_frequent_non_null_values,
  get_histogram_excluding_common_values,
  get_tables,
)
from query_generator.utils.exceptions import InvalidHistogramTypeError


class MostCommonValuesColumns(Enum):
  VALUE = "value"
  COUNT = "count"


class RedundantHistogramsDataType(Enum):
  """
  This class was made for compatibility with old code that
  generated this histogram:
  https://github.com/udao-moo/udao-spark-optimizer-dev/blob/main
  /playground/assets/data_stats/regrouped_job_hist.csv
  """

  INTEGER = "int"
  STRING = "string"


class HistogramColumns(Enum):
  TABLE = "table"
  COLUMN = "column"
  HISTOGRAM = "histogram"
  DISTINCT_COUNT = "distinct_count"
  DTYPE = "dtype"
  MOST_COMMON_VALUES = "most_common_values"
  HISTOGRAM_MCV = "histogram-mcv"  # histogram excluding most common values
  TABLE_SIZE = "table_size"


@dataclass
class HistogramParams:
  con: duckdb.DuckDBPyConnection
  table: str
  column: RawDuckDBTableDescription
  histogram_size: int


class DuckDBHistogramParser:
  """Class to represent a histogram in DuckDB."""

  def __init__(
    self, raw_histogram: list[RawDuckDBHistograms], duckdb_type: str
  ):
    self.bins = [data.bin for data in raw_histogram]
    self.counts = [data.count for data in raw_histogram]
    self._get_lower_upper_bounds()

  def _get_lower_upper_bounds(self) -> None:
    self.lower_bounds: list[str | None] = []
    self.upper_bounds: list[str] = []
    if len(self.bins) == 0:
      return
    # First bin is always special because it has a format
    # of "x <= 6" or "x <= AAAAAAAAAAAA" or "x <= 1998-01-01"
    self.lower_bounds.append(None)
    self.upper_bounds.append(self.bins[0][5:])
    # the rest of them are standard like
    # "AAAAAAAAKBAAAAAA < x <= AAAAAAAAOAAAAAAA"
    # "12 < x <= 18"
    # "2000-01-02 < x <= 2001-01-01"
    for bin in self.bins[1:]:
      if " < x <= " in bin:
        lower_bound, upper_bound = bin.split(" < x <= ")
        self.lower_bounds.append(lower_bound)
        self.upper_bounds.append(upper_bound)

  def get_equiwidth_histogram_array(self) -> list[str]:
    return self.upper_bounds


def get_most_common_values(
  con: duckdb.DuckDBPyConnection,
  table: str,
  column: str,
  common_value_size: int,
) -> list[RawDuckDBMostCommonValues]:
  return get_frequent_non_null_values(con, table, column, common_value_size)


def get_histogram_array(histogram_params: HistogramParams) -> list[str]:
  histogram_raw = get_equi_height_histogram(
    histogram_params.con,
    histogram_params.table,
    histogram_params.column.column_name,
    histogram_params.histogram_size,
  )
  histogram_parser = DuckDBHistogramParser(
    histogram_raw, histogram_params.column.column_type
  )
  return histogram_parser.get_equiwidth_histogram_array()


def get_histogram_array_excluding_common_values(
  histogram_params: HistogramParams,
  common_values_size: int,
  distinct_count: int,
) -> list[str]:
  histogram_array: list[RawDuckDBHistograms] = []
  if distinct_count > common_values_size:
    histogram_array = get_histogram_excluding_common_values(
      histogram_params.con,
      histogram_params.table,
      histogram_params.column.column_name,
      histogram_params.histogram_size,
      common_values_size,
    )
  histogram_parser = DuckDBHistogramParser(
    histogram_array,
    histogram_params.column.column_type,
  )
  return histogram_parser.get_equiwidth_histogram_array()


def query_histograms(
  histogram_size: int,
  common_values_size: int,
  con: duckdb.DuckDBPyConnection,
  *,
  include_mvc: bool,
) -> pl.DataFrame:
  """Creates histograms for the given dataset.
  Args:
    histogram_size (int): Size of the histogram.
    common_values_size (int): Size of the most common values.
    con (duckdb.DuckDBPyConnection): DuckDB connection object.
    include_mvc (bool): Whether to include most common values in the histogram
  """
  rows: list[dict[str, Any]] = []
  tables = get_tables(con)
  for table in tqdm(tables, position=0):
    columns = get_columns(con, table)
    pbar = tqdm(columns, desc="Starting…", position=1, leave=False)

    # Get table size
    table_size = get_size_of_table(con, table)
    for column in pbar:
      pbar.set_description(
        f"Processing table {table} column {column.column_name}"
      )
      histogram_params = HistogramParams(con, table, column, histogram_size)
      # Get Histogram array
      histogram_array = get_histogram_array(histogram_params)

      # Get distinct count
      distinct_count = get_distinct_count(con, table, column.column_name)

      row_dict: dict[str, Any] = {
        HistogramColumns.TABLE.value: table,
        HistogramColumns.COLUMN.value: column.column_name,
        HistogramColumns.HISTOGRAM.value: histogram_array,
        HistogramColumns.DISTINCT_COUNT.value: distinct_count,
        HistogramColumns.DTYPE.value: column.column_type,
        HistogramColumns.TABLE_SIZE.value: table_size,
      }
      if include_mvc:
        # Get most common values
        most_common_values = get_most_common_values(
          con,
          table,
          column.column_name,
          common_values_size,
        )

        # Get histogram array excluding common values
        histogram_array_excluding_mcv = (
          get_histogram_array_excluding_common_values(
            histogram_params,
            common_values_size,
            distinct_count,
          )
        )

        row_dict |= {
          HistogramColumns.MOST_COMMON_VALUES.value: [
            {
              MostCommonValuesColumns.VALUE.value: value.value,
              MostCommonValuesColumns.COUNT.value: value.count,
            }
            for value in most_common_values
          ],
          HistogramColumns.HISTOGRAM_MCV.value: histogram_array_excluding_mcv,
        }

      rows.append(row_dict)
  return pl.DataFrame(rows)


def get_size_of_table(
  con: duckdb.DuckDBPyConnection,
  table: str,
) -> int:
  result = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
  return result[0] if result else 0


def get_basic_element_of_redundant_histogram(
  dtype: str,
) -> str:
  if dtype == RedundantHistogramsDataType.INTEGER.value:
    return "0"
  if dtype == RedundantHistogramsDataType.STRING.value:
    return "A"
  raise InvalidHistogramTypeError(dtype)


def force_histogram_to_lenght(
  original_histogram: list[str],
  desired_length: int,
  dtype: str,
) -> list[str]:
  if len(original_histogram) == desired_length:
    return original_histogram
  if len(original_histogram) == 0:
    return [get_basic_element_of_redundant_histogram(dtype)] * desired_length

  base, extra = divmod(desired_length, len(original_histogram))
  result: list[str] = []
  for i, item in enumerate(original_histogram):
    result.extend([item] * (base + (1 if i < extra else 0)))
  return result


def get_redundant_bins(
  histogram_df: pl.DataFrame, desired_length: int
) -> pl.DataFrame:
  return histogram_df.with_columns(
    pl.struct([HistogramColumns.HISTOGRAM.value, HistogramColumns.DTYPE.value])
    .map_elements(
      lambda row: force_histogram_to_lenght(
        row[HistogramColumns.HISTOGRAM.value],
        desired_length,
        row[HistogramColumns.DTYPE.value],
      ),
      return_dtype=pl.List(pl.Utf8),
    )
    .alias("redundant_histogram")
  )


def get_redundant_histogram_type(histogram_df: pl.DataFrame) -> pl.DataFrame:
  return histogram_df.with_columns(
    pl.when(pl.col(HistogramColumns.DTYPE.value) == "VARCHAR")
    .then(pl.lit(RedundantHistogramsDataType.STRING.value))
    .when(pl.col(HistogramColumns.DTYPE.value) == "INTEGER")
    .then(pl.lit(RedundantHistogramsDataType.INTEGER.value))
    .otherwise(pl.col(HistogramColumns.DTYPE.value))
    .alias(HistogramColumns.DTYPE.value)
  )


def get_redundant_histograms_name_convention(
  histogram_df: pl.DataFrame,
) -> pl.DataFrame:
  """
  We only want to comply with old code. This is bad naming convention
  """
  return histogram_df.rename(
    {"histogram": "bins", "redundant_histogram": "hists"}
  )


def make_redundant_histograms(
  histogram_path: Path, desired_length: int
) -> pl.DataFrame:
  histogram_df = pl.read_parquet(histogram_path)
  modified_dtype_df = get_redundant_histogram_type(histogram_df)
  redundant_histogram = get_redundant_bins(modified_dtype_df, desired_length)
  return get_redundant_histograms_name_convention(redundant_histogram)
