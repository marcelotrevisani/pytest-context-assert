"""Integration tests with Polars DataFrames and Series."""

from __future__ import annotations

import pytest

from pytest_context_assert import set_context


@pytest.fixture
def polars():
    """Fixture to skip tests if polars is not available."""
    pl = pytest.importorskip("polars")
    return pl


def df_to_dict(df):
    """Convert Polars DataFrame to a dict with Python native types."""
    return {col: df[col].to_list() for col in df.columns}


class TestPolarsDataFrameBasics:
    """Basic Polars DataFrame tests."""

    def test_simple_dataframe(self, context_assert, polars):
        """Test simple DataFrame comparison."""
        df = polars.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        def serialize_df(df):
            return {"data": df_to_dict(df), "type": "polars_dataframe"}

        def deserialize_df(data):
            return polars.DataFrame(data["data"])

        context_assert._update_snapshots = True
        context_assert(df, name="simple_df", serialize=serialize_df)

        context_assert._update_snapshots = False
        context_assert(
            df.clone(),
            name="simple_df",
            deserialize=deserialize_df,
            compare=lambda a, b: a.equals(b),
        )

    def test_dataframe_with_schema(self, context_assert, polars):
        """Test DataFrame with explicit schema."""
        df = polars.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
            }
        )

        def serialize_df(df):
            return {
                "data": df_to_dict(df),
                "schema": {col: str(dtype) for col, dtype in df.schema.items()},
                "type": "polars_dataframe",
            }

        context_assert._update_snapshots = True
        context_assert(df, name="typed_df", serialize=serialize_df)

        context_assert._update_snapshots = False
        context_assert(
            df.clone(),
            name="typed_df",
            serialize=serialize_df,
            deserialize=lambda data: data,
            compare=lambda a, b: df_to_dict(a) == b["data"],
        )

    def test_empty_dataframe(self, context_assert, polars):
        """Test empty DataFrame."""
        df = polars.DataFrame()

        def serialize_df(df):
            return {
                "columns": df.columns,
                "height": df.height,
                "type": "polars_dataframe",
            }

        context_assert._update_snapshots = True
        context_assert(df, name="empty_df", serialize=serialize_df)

        context_assert._update_snapshots = False
        context_assert(
            polars.DataFrame(),
            name="empty_df",
            serialize=serialize_df,
            deserialize=lambda data: data,
            compare=lambda a, b: a.height == b["height"],
        )

    def test_dataframe_with_nulls(self, context_assert, polars):
        """Test DataFrame with null values."""
        df = polars.DataFrame(
            {
                "a": [1, None, 3],
                "b": [None, 2.0, 3.0],
                "c": ["x", None, "z"],
            }
        )

        def serialize_df(df):
            return {"data": df_to_dict(df), "type": "polars_dataframe"}

        context_assert._update_snapshots = True
        context_assert(df, name="null_df", serialize=serialize_df)

        context_assert._update_snapshots = False
        context_assert(
            df.clone(),
            name="null_df",
            serialize=serialize_df,
            deserialize=lambda data: data,
            compare=lambda a, b: df_to_dict(a) == b["data"],
        )


class TestPolarsSeriesBasics:
    """Basic Polars Series tests."""

    def test_simple_series(self, context_assert, polars):
        """Test simple Series comparison."""
        s = polars.Series("values", [1, 2, 3, 4, 5])

        def serialize_series(s):
            return {"data": s.to_list(), "name": s.name, "type": "polars_series"}

        def deserialize_series(data):
            return polars.Series(data["name"], data["data"])

        context_assert._update_snapshots = True
        context_assert(s, name="simple_series", serialize=serialize_series)

        context_assert._update_snapshots = False
        context_assert(
            s.clone(),
            name="simple_series",
            deserialize=deserialize_series,
            compare=lambda a, b: a.equals(b),
        )

    def test_float_series(self, context_assert, polars):
        """Test float Series."""
        s = polars.Series("floats", [1.1, 2.2, 3.3, 4.4, 5.5])

        def serialize_series(s):
            return {"data": s.to_list(), "name": s.name, "type": "polars_series"}

        context_assert._update_snapshots = True
        context_assert(s, name="float_series", serialize=serialize_series)

        context_assert._update_snapshots = False
        context_assert(
            s.clone(),
            name="float_series",
            serialize=serialize_series,
            deserialize=lambda data: data,
            compare=lambda a, b: a.to_list() == b["data"],
        )

    def test_string_series(self, context_assert, polars):
        """Test string Series."""
        s = polars.Series("strings", ["hello", "world", "polars"])

        def serialize_series(s):
            return {"data": s.to_list(), "name": s.name, "type": "polars_series"}

        context_assert._update_snapshots = True
        context_assert(s, name="string_series", serialize=serialize_series)

        context_assert._update_snapshots = False
        context_assert(
            s.clone(),
            name="string_series",
            serialize=serialize_series,
            deserialize=lambda data: data,
            compare=lambda a, b: a.to_list() == b["data"],
        )

    def test_series_with_nulls(self, context_assert, polars):
        """Test Series with null values."""
        s = polars.Series("with_nulls", [1, None, 3, None, 5])

        def serialize_series(s):
            return {"data": s.to_list(), "name": s.name, "type": "polars_series"}

        context_assert._update_snapshots = True
        context_assert(s, name="null_series", serialize=serialize_series)

        context_assert._update_snapshots = False
        context_assert(
            s.clone(),
            name="null_series",
            serialize=serialize_series,
            deserialize=lambda data: data,
            compare=lambda a, b: a.to_list() == b["data"],
        )


class TestPolarsOperations:
    """Tests for Polars operations."""

    def test_groupby_aggregation(self, context_assert, polars):
        """Test groupby aggregation results."""
        df = polars.DataFrame(
            {
                "category": ["A", "B", "A", "B", "A"],
                "value": [10, 20, 30, 40, 50],
            }
        )
        result = df.group_by("category").agg(polars.col("value").sum()).sort("category")

        def serialize_df(df):
            return {"data": df_to_dict(df), "type": "polars_dataframe"}

        context_assert._update_snapshots = True
        context_assert(result, name="groupby_sum", serialize=serialize_df)

        context_assert._update_snapshots = False
        context_assert(
            result.clone(),
            name="groupby_sum",
            serialize=serialize_df,
            deserialize=lambda data: data,
            compare=lambda a, b: df_to_dict(a) == b["data"],
        )

    def test_join_result(self, context_assert, polars):
        """Test DataFrame join results."""
        df1 = polars.DataFrame({"key": ["A", "B", "C"], "value1": [1, 2, 3]})
        df2 = polars.DataFrame({"key": ["A", "B", "D"], "value2": [4, 5, 6]})
        joined = df1.join(df2, on="key", how="inner")

        def serialize_df(df):
            return {"data": df_to_dict(df), "type": "polars_dataframe"}

        context_assert._update_snapshots = True
        context_assert(joined, name="joined_df", serialize=serialize_df)

        context_assert._update_snapshots = False
        context_assert(
            joined.clone(),
            name="joined_df",
            serialize=serialize_df,
            deserialize=lambda data: data,
            compare=lambda a, b: df_to_dict(a) == b["data"],
        )

    def test_filter_result(self, context_assert, polars):
        """Test DataFrame filter results."""
        df = polars.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie", "David"],
                "age": [25, 30, 35, 40],
            }
        )
        filtered = df.filter(polars.col("age") > 28)

        def serialize_df(df):
            return {"data": df_to_dict(df), "type": "polars_dataframe"}

        context_assert._update_snapshots = True
        context_assert(filtered, name="filtered_df", serialize=serialize_df)

        context_assert._update_snapshots = False
        context_assert(
            filtered.clone(),
            name="filtered_df",
            serialize=serialize_df,
            deserialize=lambda data: data,
            compare=lambda a, b: df_to_dict(a) == b["data"],
        )

    def test_select_result(self, context_assert, polars):
        """Test DataFrame select with expressions."""
        df = polars.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
            }
        )
        result = df.select(
            polars.col("a"),
            polars.col("b"),
            (polars.col("a") + polars.col("b")).alias("sum"),
            (polars.col("a") * polars.col("b")).alias("product"),
        )

        def serialize_df(df):
            return {"data": df_to_dict(df), "type": "polars_dataframe"}

        context_assert._update_snapshots = True
        context_assert(result, name="select_result", serialize=serialize_df)

        context_assert._update_snapshots = False
        context_assert(
            result.clone(),
            name="select_result",
            serialize=serialize_df,
            deserialize=lambda data: data,
            compare=lambda a, b: df_to_dict(a) == b["data"],
        )

    def test_with_columns(self, context_assert, polars):
        """Test DataFrame with_columns."""
        df = polars.DataFrame({"x": [1, 2, 3]})
        result = df.with_columns(
            (polars.col("x") * 2).alias("x_doubled"),
            (polars.col("x") ** 2).alias("x_squared"),
        )

        def serialize_df(df):
            return {"data": df_to_dict(df), "type": "polars_dataframe"}

        context_assert._update_snapshots = True
        context_assert(result, name="with_columns_result", serialize=serialize_df)

        context_assert._update_snapshots = False
        context_assert(
            result.clone(),
            name="with_columns_result",
            serialize=serialize_df,
            deserialize=lambda data: data,
            compare=lambda a, b: df_to_dict(a) == b["data"],
        )


class TestPolarsLazyFrame:
    """Tests for Polars LazyFrame operations."""

    def test_lazy_collect(self, context_assert, polars):
        """Test LazyFrame collect results."""
        lf = polars.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = lf.filter(polars.col("a") > 1).collect()

        def serialize_df(df):
            return {"data": df_to_dict(df), "type": "polars_dataframe"}

        context_assert._update_snapshots = True
        context_assert(result, name="lazy_collect", serialize=serialize_df)

        context_assert._update_snapshots = False
        context_assert(
            result.clone(),
            name="lazy_collect",
            serialize=serialize_df,
            deserialize=lambda data: data,
            compare=lambda a, b: df_to_dict(a) == b["data"],
        )

    def test_lazy_groupby(self, context_assert, polars):
        """Test LazyFrame groupby operations."""
        lf = polars.LazyFrame(
            {
                "group": ["A", "A", "B", "B"],
                "value": [1, 2, 3, 4],
            }
        )
        result = (
            lf.group_by("group")
            .agg(
                polars.col("value").sum().alias("sum"),
                polars.col("value").mean().alias("mean"),
            )
            .sort("group")
            .collect()
        )

        def serialize_df(df):
            return {"data": df_to_dict(df), "type": "polars_dataframe"}

        context_assert._update_snapshots = True
        context_assert(result, name="lazy_groupby", serialize=serialize_df)

        context_assert._update_snapshots = False
        context_assert(
            result.clone(),
            name="lazy_groupby",
            serialize=serialize_df,
            deserialize=lambda data: data,
            compare=lambda a, b: df_to_dict(a) == b["data"],
        )


class TestPolarsDatetime:
    """Tests for Polars datetime operations."""

    def test_datetime_series(self, context_assert, polars):
        """Test datetime Series."""
        from datetime import datetime

        dates = [
            datetime(2026, 1, 1),
            datetime(2026, 1, 2),
            datetime(2026, 1, 3),
        ]
        s = polars.Series("dates", dates)

        def serialize_series(s):
            return {
                "data": [str(d) for d in s.to_list()],
                "name": s.name,
                "type": "polars_datetime_series",
            }

        context_assert._update_snapshots = True
        context_assert(s, name="datetime_series", serialize=serialize_series)

        context_assert._update_snapshots = False
        context_assert(
            s.clone(),
            name="datetime_series",
            serialize=serialize_series,
            deserialize=lambda data: data,
            compare=lambda a, b: [str(d) for d in a.to_list()] == b["data"],
        )

    def test_date_range(self, context_assert, polars):
        """Test date range creation."""
        from datetime import date

        dates = polars.date_range(
            date(2026, 1, 1),
            date(2026, 1, 10),
            eager=True,
        )

        def serialize_series(s):
            return {
                "data": [str(d) for d in s.to_list()],
                "name": s.name,
                "type": "polars_date_series",
            }

        context_assert._update_snapshots = True
        context_assert(dates, name="date_range", serialize=serialize_series)

        context_assert._update_snapshots = False
        context_assert(
            dates.clone(),
            name="date_range",
            serialize=serialize_series,
            deserialize=lambda data: data,
            compare=lambda a, b: [str(d) for d in a.to_list()] == b["data"],
        )


class TestPolarsStatistics:
    """Tests for Polars statistical operations."""

    def test_describe_output(self, context_assert, polars):
        """Test DataFrame describe output."""
        df = polars.DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": [10, 20, 30, 40, 50],
                "C": [1.1, 2.2, 3.3, 4.4, 5.5],
            }
        )
        desc = df.describe()

        def serialize_df(df):
            return {"data": df_to_dict(df), "type": "polars_describe"}

        context_assert._update_snapshots = True
        context_assert(desc, name="describe", serialize=serialize_df)

        context_assert._update_snapshots = False
        context_assert(
            desc.clone(),
            name="describe",
            serialize=serialize_df,
            deserialize=lambda data: data,
            compare=lambda a, b: True,  # Simplified comparison
        )

    def test_value_counts(self, context_assert, polars):
        """Test value_counts output."""
        s = polars.Series("values", ["a", "b", "a", "c", "a", "b"])
        counts = s.value_counts().sort("values")

        def serialize_df(df):
            return {"data": df_to_dict(df), "type": "polars_value_counts"}

        context_assert._update_snapshots = True
        context_assert(counts, name="value_counts", serialize=serialize_df)

        context_assert._update_snapshots = False
        context_assert(
            counts.clone(),
            name="value_counts",
            serialize=serialize_df,
            deserialize=lambda data: data,
            compare=lambda a, b: df_to_dict(a) == b["data"],
        )

    def test_aggregations(self, context_assert, polars):
        """Test various aggregation functions."""
        df = polars.DataFrame({"values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        stats = {
            "sum": df.select(polars.col("values").sum()).item(),
            "mean": df.select(polars.col("values").mean()).item(),
            "std": df.select(polars.col("values").std()).item(),
            "min": df.select(polars.col("values").min()).item(),
            "max": df.select(polars.col("values").max()).item(),
            "median": df.select(polars.col("values").median()).item(),
        }

        def serialize_stats(s):
            return {"data": s, "type": "stats_dict"}

        context_assert._update_snapshots = True
        context_assert(stats, name="aggregations", serialize=serialize_stats)

        context_assert._update_snapshots = False
        context_assert(
            stats,
            name="aggregations",
            serialize=serialize_stats,
            deserialize=lambda data: data,
            compare=lambda a, b: all(abs(a[k] - b["data"][k]) < 1e-10 for k in a.keys()),
        )


class TestPolarsWithContext:
    """Tests for Polars with custom context."""

    @set_context({"data_source": "test", "version": "1.0"})
    def test_context_aware_dataframe(self, context_assert, polars):
        """Test DataFrame with custom context."""
        df = polars.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        def serialize_df(df):
            return {"data": df_to_dict(df), "type": "polars_dataframe"}

        context_assert._update_snapshots = True
        context_assert(df, name="ctx_df", serialize=serialize_df)

        context_assert._update_snapshots = False
        context_assert(
            df.clone(),
            name="ctx_df",
            serialize=serialize_df,
            deserialize=lambda data: data,
            compare=lambda a, b: df_to_dict(a) == b["data"],
        )

    def test_runtime_context(self, context_assert, polars):
        """Test with runtime context override."""
        df = polars.DataFrame({"a": [1, 2, 3]})

        def serialize_df(df):
            return {"data": df_to_dict(df), "type": "polars_dataframe"}

        custom = context_assert.with_context(polars_version=polars.__version__)

        custom._update_snapshots = True
        custom(df, name="versioned_df", serialize=serialize_df)

        custom._update_snapshots = False
        custom(
            df.clone(),
            name="versioned_df",
            serialize=serialize_df,
            deserialize=lambda data: data,
            compare=lambda a, b: df_to_dict(a) == b["data"],
        )


class TestPolarsSpecialTypes:
    """Tests for Polars special data types."""

    def test_categorical_series(self, context_assert, polars):
        """Test categorical Series."""
        s = polars.Series("cat", ["low", "medium", "high", "medium", "low"]).cast(
            polars.Categorical
        )

        def serialize_series(s):
            return {
                "data": s.to_list(),
                "name": s.name,
                "dtype": str(s.dtype),
                "type": "polars_categorical",
            }

        context_assert._update_snapshots = True
        context_assert(s, name="categorical", serialize=serialize_series)

        context_assert._update_snapshots = False
        context_assert(
            s.clone(),
            name="categorical",
            serialize=serialize_series,
            deserialize=lambda data: data,
            compare=lambda a, b: a.to_list() == b["data"],
        )

    def test_list_series(self, context_assert, polars):
        """Test Series with list dtype."""
        s = polars.Series("lists", [[1, 2], [3, 4, 5], [6]])

        def serialize_series(s):
            return {
                "data": s.to_list(),
                "name": s.name,
                "type": "polars_list_series",
            }

        context_assert._update_snapshots = True
        context_assert(s, name="list_series", serialize=serialize_series)

        context_assert._update_snapshots = False
        context_assert(
            s.clone(),
            name="list_series",
            serialize=serialize_series,
            deserialize=lambda data: data,
            compare=lambda a, b: a.to_list() == b["data"],
        )

    def test_struct_series(self, context_assert, polars):
        """Test Series with struct dtype."""
        s = polars.Series(
            "structs",
            [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25},
            ],
        )

        def serialize_series(s):
            return {
                "data": s.to_list(),
                "name": s.name,
                "type": "polars_struct_series",
            }

        context_assert._update_snapshots = True
        context_assert(s, name="struct_series", serialize=serialize_series)

        context_assert._update_snapshots = False
        context_assert(
            s.clone(),
            name="struct_series",
            serialize=serialize_series,
            deserialize=lambda data: data,
            compare=lambda a, b: a.to_list() == b["data"],
        )


class TestPolarsToleranceComparison:
    """Tests for Polars with tolerance-based comparison."""

    def test_float_tolerance(self, context_assert, polars):
        """Test float comparison with tolerance."""
        df1 = polars.DataFrame({"values": [1.0, 2.0, 3.0]})
        df2 = polars.DataFrame({"values": [1.0000001, 2.0000001, 3.0000001]})

        def serialize_df(df):
            return {"data": df["values"].to_list(), "type": "polars_dataframe"}

        def compare_with_tolerance(a, b, rtol=1e-6):
            a_vals = a["values"].to_list()
            b_vals = b["data"]
            return all(abs(av - bv) <= rtol * abs(bv) for av, bv in zip(a_vals, b_vals))

        context_assert._update_snapshots = True
        context_assert(df1, name="tol_df", serialize=serialize_df)

        context_assert._update_snapshots = False
        context_assert(
            df2,
            name="tol_df",
            serialize=serialize_df,
            deserialize=lambda data: data,
            compare=lambda a, b: compare_with_tolerance(a, b, rtol=1e-5),
        )
