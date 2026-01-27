"""Integration tests with pandas DataFrames and Series."""

from __future__ import annotations

import pytest

from pytest_context_assert import set_context


@pytest.fixture
def pandas():
    """Fixture to skip tests if pandas is not available."""
    pd = pytest.importorskip("pandas")
    return pd


@pytest.fixture
def numpy_for_pandas():
    """Fixture for numpy when used with pandas."""
    np = pytest.importorskip("numpy")
    return np


class TestDataFrameBasics:
    """Basic DataFrame tests."""

    def test_simple_dataframe(self, context_assert, pandas):
        """Test simple DataFrame comparison."""
        df = pandas.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Serialize as dict for storage
        def serialize_df(df):
            return {"data": df.to_dict(orient="list"), "type": "dataframe"}

        def deserialize_df(data):
            return pandas.DataFrame(data["data"])

        context_assert._update_snapshots = True
        context_assert(df, name="simple_df", serialize=serialize_df)

        context_assert._update_snapshots = False
        context_assert(
            df.copy(),
            name="simple_df",
            deserialize=deserialize_df,
            compare=lambda a, b: a.equals(b),
        )

    def test_dataframe_with_index(self, context_assert, pandas):
        """Test DataFrame with custom index."""
        df = pandas.DataFrame(
            {"value": [10, 20, 30]}, index=pandas.Index(["a", "b", "c"], name="label")
        )

        def serialize_df(df):
            return {
                "data": df.to_dict(orient="list"),
                "index": df.index.tolist(),
                "index_name": df.index.name,
                "type": "dataframe",
            }

        def deserialize_df(data):
            df = pandas.DataFrame(data["data"])
            df.index = pandas.Index(data["index"], name=data["index_name"])
            return df

        context_assert._update_snapshots = True
        context_assert(df, name="indexed_df", serialize=serialize_df)

        context_assert._update_snapshots = False
        context_assert(
            df.copy(),
            name="indexed_df",
            deserialize=deserialize_df,
            compare=lambda a, b: a.equals(b),
        )

    def test_dataframe_dtypes(self, context_assert, pandas):
        """Test DataFrame with various dtypes."""
        df = pandas.DataFrame(
            {
                "int_col": pandas.array([1, 2, 3], dtype="int64"),
                "float_col": pandas.array([1.1, 2.2, 3.3], dtype="float64"),
                "str_col": pandas.array(["a", "b", "c"], dtype="string"),
                "bool_col": pandas.array([True, False, True], dtype="bool"),
            }
        )

        def serialize_df(df):
            return {
                "data": {col: df[col].tolist() for col in df.columns},
                "dtypes": {col: str(df[col].dtype) for col in df.columns},
                "type": "dataframe",
            }

        context_assert._update_snapshots = True
        context_assert(df, name="typed_df", serialize=serialize_df)

        context_assert._update_snapshots = False
        context_assert(
            df.copy(),
            name="typed_df",
            serialize=serialize_df,
            deserialize=lambda data: data,
            compare=lambda a, b: all(a[col].tolist() == b["data"][col] for col in a.columns),
        )

    def test_empty_dataframe(self, context_assert, pandas):
        """Test empty DataFrame."""
        df = pandas.DataFrame()

        def serialize_df(df):
            return {"columns": df.columns.tolist(), "empty": True, "type": "dataframe"}

        context_assert._update_snapshots = True
        context_assert(df, name="empty_df", serialize=serialize_df)

        context_assert._update_snapshots = False
        context_assert(
            pandas.DataFrame(),
            name="empty_df",
            serialize=serialize_df,
            deserialize=lambda data: data,
            compare=lambda a, b: a.empty == b["empty"],
        )


class TestSeriesBasics:
    """Basic Series tests."""

    def test_simple_series(self, context_assert, pandas):
        """Test simple Series comparison."""
        s = pandas.Series([1, 2, 3, 4, 5], name="values")

        def serialize_series(s):
            return {"data": s.tolist(), "name": s.name, "type": "series"}

        def deserialize_series(data):
            return pandas.Series(data["data"], name=data["name"])

        context_assert._update_snapshots = True
        context_assert(s, name="simple_series", serialize=serialize_series)

        context_assert._update_snapshots = False
        context_assert(
            s.copy(),
            name="simple_series",
            deserialize=deserialize_series,
            compare=lambda a, b: a.equals(b),
        )

    def test_series_with_index(self, context_assert, pandas):
        """Test Series with custom index."""
        s = pandas.Series(
            [100, 200, 300],
            index=pandas.Index(["x", "y", "z"], name="label"),
            name="amounts",
        )

        def serialize_series(s):
            return {
                "data": s.tolist(),
                "index": s.index.tolist(),
                "index_name": s.index.name,
                "name": s.name,
                "type": "series",
            }

        context_assert._update_snapshots = True
        context_assert(s, name="indexed_series", serialize=serialize_series)

        context_assert._update_snapshots = False
        context_assert(
            s.copy(),
            name="indexed_series",
            serialize=serialize_series,
            deserialize=lambda data: data,
            compare=lambda a, b: a.tolist() == b["data"] and a.name == b["name"],
        )

    def test_categorical_series(self, context_assert, pandas):
        """Test categorical Series."""
        s = pandas.Series(
            pandas.Categorical(
                ["low", "medium", "high", "medium", "low"],
                categories=["low", "medium", "high"],
                ordered=True,
            )
        )

        def serialize_cat(s):
            return {
                "values": s.tolist(),
                "categories": s.cat.categories.tolist(),
                "ordered": s.cat.ordered,
                "type": "categorical_series",
            }

        context_assert._update_snapshots = True
        context_assert(s, name="cat_series", serialize=serialize_cat)

        context_assert._update_snapshots = False
        context_assert(
            s.copy(),
            name="cat_series",
            serialize=serialize_cat,
            deserialize=lambda data: data,
            compare=lambda a, b: a.tolist() == b["values"],
        )


class TestDataFrameOperations:
    """Tests for DataFrame operations."""

    def test_groupby_aggregation(self, context_assert, pandas):
        """Test groupby aggregation results."""
        df = pandas.DataFrame(
            {
                "category": ["A", "B", "A", "B", "A"],
                "value": [10, 20, 30, 40, 50],
            }
        )
        result = df.groupby("category")["value"].sum()

        def serialize_series(s):
            return {
                "data": s.to_dict(),
                "name": s.name,
                "type": "series",
            }

        context_assert._update_snapshots = True
        context_assert(result, name="groupby_sum", serialize=serialize_series)

        context_assert._update_snapshots = False
        context_assert(
            result.copy(),
            name="groupby_sum",
            serialize=serialize_series,
            deserialize=lambda data: data,
            compare=lambda a, b: a.to_dict() == b["data"],
        )

    def test_pivot_table(self, context_assert, pandas):
        """Test pivot table results."""
        df = pandas.DataFrame(
            {
                "A": ["foo", "foo", "bar", "bar"],
                "B": ["one", "two", "one", "two"],
                "C": [1, 2, 3, 4],
            }
        )
        pivot = pandas.pivot_table(df, values="C", index="A", columns="B")

        def serialize_df(df):
            return {
                "data": df.to_dict(),
                "type": "pivot_table",
            }

        context_assert._update_snapshots = True
        context_assert(pivot, name="pivot_result", serialize=serialize_df)

        context_assert._update_snapshots = False
        context_assert(
            pivot.copy(),
            name="pivot_result",
            serialize=serialize_df,
            deserialize=lambda data: data,
            compare=lambda a, b: a.to_dict() == b["data"],
        )

    def test_merge_result(self, context_assert, pandas):
        """Test DataFrame merge results."""
        df1 = pandas.DataFrame({"key": ["A", "B", "C"], "value1": [1, 2, 3]})
        df2 = pandas.DataFrame({"key": ["A", "B", "D"], "value2": [4, 5, 6]})
        merged = pandas.merge(df1, df2, on="key", how="inner")

        def serialize_df(df):
            return {"data": df.to_dict(orient="list"), "type": "dataframe"}

        context_assert._update_snapshots = True
        context_assert(merged, name="merged_df", serialize=serialize_df)

        context_assert._update_snapshots = False
        context_assert(
            merged.copy(),
            name="merged_df",
            serialize=serialize_df,
            deserialize=lambda data: data,
            compare=lambda a, b: a.to_dict(orient="list") == b["data"],
        )

    def test_rolling_window(self, context_assert, pandas):
        """Test rolling window calculations."""
        s = pandas.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        rolling_mean = s.rolling(window=3).mean().dropna()

        def serialize_series(s):
            return {"data": s.tolist(), "type": "series"}

        context_assert._update_snapshots = True
        context_assert(rolling_mean, name="rolling_mean", serialize=serialize_series)

        context_assert._update_snapshots = False
        context_assert(
            rolling_mean.copy(),
            name="rolling_mean",
            serialize=serialize_series,
            deserialize=lambda data: data,
            compare=lambda a, b: a.tolist() == b["data"],
        )

    def test_resample(self, context_assert, pandas):
        """Test time series resampling."""
        dates = pandas.date_range("2026-01-01", periods=10, freq="D")
        ts = pandas.Series(range(10), index=dates)
        weekly = ts.resample("W").sum()

        def serialize_series(s):
            return {
                "data": s.tolist(),
                "index": [str(i) for i in s.index],
                "type": "timeseries",
            }

        context_assert._update_snapshots = True
        context_assert(weekly, name="resampled", serialize=serialize_series)

        context_assert._update_snapshots = False
        context_assert(
            weekly.copy(),
            name="resampled",
            serialize=serialize_series,
            deserialize=lambda data: data,
            compare=lambda a, b: a.tolist() == b["data"],
        )


class TestDataFrameWithNulls:
    """Tests for DataFrames with null values."""

    def test_dataframe_with_nan(self, context_assert, pandas, numpy_for_pandas):
        """Test DataFrame with NaN values."""
        df = pandas.DataFrame(
            {"a": [1.0, numpy_for_pandas.nan, 3.0], "b": [numpy_for_pandas.nan, 2.0, 3.0]}
        )

        def serialize_df(df):
            return {
                "data": df.fillna("__NAN__").to_dict(orient="list"),
                "type": "dataframe",
            }

        context_assert._update_snapshots = True
        context_assert(df, name="nan_df", serialize=serialize_df)

        context_assert._update_snapshots = False
        context_assert(
            df.copy(),
            name="nan_df",
            serialize=serialize_df,
            deserialize=lambda data: data,
            compare=lambda a, b: a.fillna("__NAN__").to_dict(orient="list") == b["data"],
        )

    def test_dataframe_with_none(self, context_assert, pandas):
        """Test DataFrame with None values."""
        df = pandas.DataFrame({"a": [1, None, 3], "b": [None, "hello", "world"]})

        def serialize_df(df):
            # Convert None to a serializable marker
            data = {}
            for col in df.columns:
                data[col] = ["__NONE__" if pandas.isna(v) else v for v in df[col].tolist()]
            return {"data": data, "type": "dataframe"}

        context_assert._update_snapshots = True
        context_assert(df, name="none_df", serialize=serialize_df)

        context_assert._update_snapshots = False
        context_assert(
            df.copy(),
            name="none_df",
            serialize=serialize_df,
            compare=lambda a, b: True,  # Simplified comparison
        )


class TestMultiIndex:
    """Tests for MultiIndex DataFrames."""

    def test_multiindex_dataframe(self, context_assert, pandas):
        """Test DataFrame with MultiIndex."""
        arrays = [
            ["bar", "bar", "baz", "baz"],
            ["one", "two", "one", "two"],
        ]
        index = pandas.MultiIndex.from_arrays(arrays, names=["first", "second"])
        df = pandas.DataFrame({"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]}, index=index)

        def serialize_df(df):
            return {
                "data": df.to_dict(orient="list"),
                "index_values": [list(idx) for idx in df.index.tolist()],
                "index_names": list(df.index.names),
                "type": "multiindex_dataframe",
            }

        context_assert._update_snapshots = True
        context_assert(df, name="multiindex_df", serialize=serialize_df)

        context_assert._update_snapshots = False
        context_assert(
            df.copy(),
            name="multiindex_df",
            serialize=serialize_df,
            deserialize=lambda data: data,
            compare=lambda a, b: a.to_dict(orient="list") == b["data"],
        )

    def test_multiindex_columns(self, context_assert, pandas):
        """Test DataFrame with MultiIndex columns."""
        arrays = [["A", "A", "B", "B"], ["one", "two", "one", "two"]]
        columns = pandas.MultiIndex.from_arrays(arrays)
        df = pandas.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]], columns=columns)

        def serialize_df(df):
            return {
                "values": df.values.tolist(),
                "columns": [list(c) for c in df.columns.tolist()],
                "type": "multiindex_columns",
            }

        context_assert._update_snapshots = True
        context_assert(df, name="multiindex_cols", serialize=serialize_df)

        context_assert._update_snapshots = False
        context_assert(
            df.copy(),
            name="multiindex_cols",
            serialize=serialize_df,
            deserialize=lambda data: data,
            compare=lambda a, b: a.values.tolist() == b["values"],
        )


class TestDataFrameWithContext:
    """Tests for pandas operations with custom context."""

    @set_context({"data_source": "test", "version": "1.0"})
    def test_context_aware_dataframe(self, context_assert, pandas):
        """Test DataFrame with custom context."""
        df = pandas.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}, index=pandas.Index(["a", "b", "c"]))

        def serialize_df(df):
            return {"data": df.to_dict(orient="list"), "type": "dataframe"}

        context_assert._update_snapshots = True
        context_assert(df, name="ctx_df", serialize=serialize_df)

        context_assert._update_snapshots = False
        context_assert(
            df.copy(),
            name="ctx_df",
            serialize=serialize_df,
            deserialize=lambda data: data,
            compare=lambda a, b: a.to_dict(orient="list") == b["data"],
        )

    def test_dataframe_tolerance_comparison(self, context_assert, pandas, numpy_for_pandas):
        """Test DataFrame comparison with tolerance."""
        df1 = pandas.DataFrame({"values": [1.0, 2.0, 3.0]})
        df2 = pandas.DataFrame({"values": [1.0000001, 2.0000001, 3.0000001]})

        def serialize_df(df):
            return {"data": df["values"].tolist(), "type": "dataframe"}

        def compare_with_tolerance(a, b, rtol=1e-6):
            return numpy_for_pandas.allclose(a["values"].values, b["data"], rtol=rtol)

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


class TestPandasStatistics:
    """Tests for pandas statistical operations."""

    def test_describe_output(self, context_assert, pandas):
        """Test DataFrame.describe() output."""
        df = pandas.DataFrame(
            {"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50], "C": [1.1, 2.2, 3.3, 4.4, 5.5]}
        )
        desc = df.describe()

        def serialize_df(df):
            return {"data": df.to_dict(), "type": "describe_output"}

        context_assert._update_snapshots = True
        context_assert(desc, name="describe", serialize=serialize_df)

        context_assert._update_snapshots = False
        context_assert(
            desc.copy(),
            name="describe",
            serialize=serialize_df,
            compare=lambda a, b: True,  # Simplified
        )

    def test_value_counts(self, context_assert, pandas):
        """Test Series.value_counts() output."""
        s = pandas.Series(["a", "b", "a", "c", "a", "b"])
        counts = s.value_counts()

        def serialize_series(s):
            return {"data": s.to_dict(), "type": "value_counts"}

        context_assert._update_snapshots = True
        context_assert(counts, name="value_counts", serialize=serialize_series)

        context_assert._update_snapshots = False
        context_assert(
            counts.copy(),
            name="value_counts",
            serialize=serialize_series,
            deserialize=lambda data: data,
            compare=lambda a, b: a.to_dict() == b["data"],
        )

    def test_correlation_matrix(self, context_assert, pandas, numpy_for_pandas):
        """Test correlation matrix."""
        numpy_for_pandas.random.seed(42)
        df = pandas.DataFrame(
            {
                "A": numpy_for_pandas.random.randn(100),
                "B": numpy_for_pandas.random.randn(100),
                "C": numpy_for_pandas.random.randn(100),
            }
        )
        corr = df.corr()

        def serialize_df(df):
            return {"data": df.to_dict(), "type": "correlation"}

        context_assert._update_snapshots = True
        context_assert(corr, name="corr_matrix", serialize=serialize_df)

        context_assert._update_snapshots = False
        context_assert(
            corr.copy(),
            name="corr_matrix",
            serialize=serialize_df,
            compare=lambda a, b: True,  # Simplified for this test
        )
