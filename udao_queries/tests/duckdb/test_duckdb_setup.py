import pytest

from query_generator.duckdb_connection.binning import (
  get_result_from_duckdb,
)
from query_generator.duckdb_connection.setup import setup_duckdb
from query_generator.utils.definitions import Dataset


def test_dev_duckdb_setup_tpch():
  """Test the setup of DuckDB."""
  # Setup DuckDB
  con = setup_duckdb(Dataset.TPCH, 0.1)
  assert con is not None, "DuckDB connection should not be None"
  assert con.execute("SELECT 1").fetchall() == [(1,)], "DuckDB should return 1"
  assert con.sql("show tables").fetchall() == [
    ("customer",),
    ("lineitem",),
    ("nation",),
    ("orders",),
    ("part",),
    ("partsupp",),
    ("region",),
    ("supplier",),
  ], "DuckDB should have the TPCH tables"


def test_dev_duckdb_setup_tpcds():
  """Test the setup of DuckDB."""
  # Setup DuckDB
  con = setup_duckdb(Dataset.TPCDS, 0.1)
  assert con is not None, "DuckDB connection should not be None"
  assert con.execute("SELECT 1").fetchall() == [(1,)], "DuckDB should return 1"
  assert con.sql("show tables").fetchall() == [
    ("call_center",),
    ("catalog_page",),
    ("catalog_returns",),
    ("catalog_sales",),
    ("customer",),
    ("customer_address",),
    ("customer_demographics",),
    ("date_dim",),
    ("household_demographics",),
    ("income_band",),
    ("inventory",),
    ("item",),
    ("promotion",),
    ("reason",),
    ("ship_mode",),
    ("store",),
    ("store_returns",),
    ("store_sales",),
    ("time_dim",),
    ("warehouse",),
    ("web_page",),
    ("web_returns",),
    ("web_sales",),
    ("web_site",),
  ], "DuckDB should have the TPCDS tables"


@pytest.mark.parametrize(
  "query, expected_result",
  [
    ("SELECT COUNT(*) FROM customer", 10000),
    ("SELECT 1", 1),
  ],
)
def test_duck_db_execution(query, expected_result):
  """Test the execution of queries in DuckDB."""
  # Setup DuckDB
  con = setup_duckdb(Dataset.TPCDS, 0.1)
  val = get_result_from_duckdb(query, con)
  assert val == expected_result, f"Expected {expected_result}, but got {val}"
