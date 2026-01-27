"""Tests for serializers."""

from __future__ import annotations

import pytest

from pytest_context_assert.serializers import (
    DictSerializer,
    GenericSerializer,
    ListSerializer,
    NumpySerializer,
    PandasSerializer,
    PolarsSerializer,
    ScalarSerializer,
    deserialize_value,
    get_serializer,
    serialize_value,
)


class TestScalarSerializer:
    """Tests for ScalarSerializer."""

    def test_can_serialize(self):
        serializer = ScalarSerializer()
        assert serializer.can_serialize(42)
        assert serializer.can_serialize(3.14)
        assert serializer.can_serialize("hello")
        assert serializer.can_serialize(True)
        assert serializer.can_serialize(None)
        assert not serializer.can_serialize([1, 2, 3])

    def test_serialize_int(self):
        serializer = ScalarSerializer()
        result = serializer.serialize(42)
        assert result == {"value": 42, "type": "int"}

    def test_serialize_float(self):
        serializer = ScalarSerializer()
        result = serializer.serialize(3.14)
        assert result == {"value": 3.14, "type": "float"}

    def test_serialize_string(self):
        serializer = ScalarSerializer()
        result = serializer.serialize("hello")
        assert result == {"value": "hello", "type": "str"}

    def test_serialize_bool(self):
        serializer = ScalarSerializer()
        result = serializer.serialize(True)
        assert result == {"value": True, "type": "bool"}

    def test_serialize_none(self):
        serializer = ScalarSerializer()
        result = serializer.serialize(None)
        assert result == {"value": None, "type": "none"}

    def test_deserialize_int(self):
        serializer = ScalarSerializer()
        result = serializer.deserialize({"value": 42, "type": "int"})
        assert result == 42

    def test_deserialize_float(self):
        serializer = ScalarSerializer()
        result = serializer.deserialize({"value": 3.14, "type": "float"})
        assert result == 3.14

    def test_deserialize_none(self):
        serializer = ScalarSerializer()
        result = serializer.deserialize({"value": None, "type": "none"})
        assert result is None

    def test_deserialize_bool_true(self):
        serializer = ScalarSerializer()
        result = serializer.deserialize({"value": True, "type": "bool"})
        assert result is True

    def test_deserialize_bool_false(self):
        serializer = ScalarSerializer()
        result = serializer.deserialize({"value": False, "type": "bool"})
        assert result is False

    def test_deserialize_string(self):
        serializer = ScalarSerializer()
        result = serializer.deserialize({"value": "hello world", "type": "str"})
        assert result == "hello world"

    def test_serialize_positive_infinity(self):
        import math

        serializer = ScalarSerializer()
        result = serializer.serialize(math.inf)
        assert result == {
            "value": "__POSITIVE_INFINITY__",
            "type": "float",
            "_special": True,
        }

    def test_serialize_negative_infinity(self):
        import math

        serializer = ScalarSerializer()
        result = serializer.serialize(-math.inf)
        assert result == {
            "value": "__NEGATIVE_INFINITY__",
            "type": "float",
            "_special": True,
        }

    def test_serialize_nan(self):
        import math

        serializer = ScalarSerializer()
        result = serializer.serialize(math.nan)
        assert result == {
            "value": "__NAN__",
            "type": "float",
            "_special": True,
        }

    def test_deserialize_positive_infinity(self):
        import math

        serializer = ScalarSerializer()
        result = serializer.deserialize(
            {
                "value": "__POSITIVE_INFINITY__",
                "type": "float",
                "_special": True,
            }
        )
        assert math.isinf(result) and result > 0

    def test_deserialize_negative_infinity(self):
        import math

        serializer = ScalarSerializer()
        result = serializer.deserialize(
            {
                "value": "__NEGATIVE_INFINITY__",
                "type": "float",
                "_special": True,
            }
        )
        assert math.isinf(result) and result < 0

    def test_deserialize_nan(self):
        import math

        serializer = ScalarSerializer()
        result = serializer.deserialize(
            {
                "value": "__NAN__",
                "type": "float",
                "_special": True,
            }
        )
        assert math.isnan(result)

    def test_special_float_roundtrip(self):
        import math

        serializer = ScalarSerializer()

        # Test positive infinity roundtrip
        serialized = serializer.serialize(math.inf)
        deserialized = serializer.deserialize(serialized)
        assert math.isinf(deserialized) and deserialized > 0

        # Test negative infinity roundtrip
        serialized = serializer.serialize(-math.inf)
        deserialized = serializer.deserialize(serialized)
        assert math.isinf(deserialized) and deserialized < 0

        # Test NaN roundtrip
        serialized = serializer.serialize(math.nan)
        deserialized = serializer.deserialize(serialized)
        assert math.isnan(deserialized)

    def test_serialize_numpy_infinity(self, numpy):
        serializer = ScalarSerializer()

        # numpy positive infinity
        result = serializer.serialize(numpy.inf)
        assert result["_special"] is True
        assert result["value"] == "__POSITIVE_INFINITY__"

        # numpy negative infinity
        result = serializer.serialize(-numpy.inf)
        assert result["_special"] is True
        assert result["value"] == "__NEGATIVE_INFINITY__"

    def test_serialize_numpy_nan(self, numpy):
        serializer = ScalarSerializer()
        result = serializer.serialize(numpy.nan)
        assert result["_special"] is True
        assert result["value"] == "__NAN__"


class TestListSerializer:
    """Tests for ListSerializer."""

    def test_can_serialize(self):
        serializer = ListSerializer()
        assert serializer.can_serialize([1, 2, 3])
        assert serializer.can_serialize((1, 2, 3))
        assert not serializer.can_serialize("hello")

    def test_serialize_list(self):
        serializer = ListSerializer()
        result = serializer.serialize([1, 2, 3])
        assert result["type"] == "list"
        assert len(result["value"]) == 3

    def test_serialize_tuple(self):
        serializer = ListSerializer()
        result = serializer.serialize((1, 2, 3))
        assert result["type"] == "tuple"

    def test_deserialize_list(self):
        serializer = ListSerializer()
        data = {
            "type": "list",
            "value": [
                {"value": 1, "type": "int"},
                {"value": 2, "type": "int"},
            ],
        }
        result = serializer.deserialize(data)
        assert result == [1, 2]

    def test_deserialize_tuple(self):
        serializer = ListSerializer()
        data = {
            "type": "tuple",
            "value": [
                {"value": 1, "type": "int"},
                {"value": 2, "type": "int"},
            ],
        }
        result = serializer.deserialize(data)
        assert result == (1, 2)

    def test_serialize_empty_list(self):
        serializer = ListSerializer()
        result = serializer.serialize([])
        assert result == {"type": "list", "value": []}

    def test_serialize_nested_list(self):
        serializer = ListSerializer()
        result = serializer.serialize([[1, 2], [3, 4]])
        assert result["type"] == "list"
        assert len(result["value"]) == 2

    def test_serialize_mixed_types(self):
        serializer = ListSerializer()
        result = serializer.serialize([1, "hello", 3.14, True])
        assert result["type"] == "list"
        assert len(result["value"]) == 4


class TestDictSerializer:
    """Tests for DictSerializer."""

    def test_can_serialize(self):
        serializer = DictSerializer()
        assert serializer.can_serialize({"a": 1})
        assert not serializer.can_serialize([1, 2])

    def test_serialize_dict(self):
        serializer = DictSerializer()
        result = serializer.serialize({"a": 1, "b": "hello"})
        assert result["type"] == "dict"
        assert "a" in result["value"]
        assert "b" in result["value"]

    def test_deserialize_dict(self):
        serializer = DictSerializer()
        data = {
            "type": "dict",
            "value": {
                "a": {"value": 1, "type": "int"},
                "b": {"value": "hello", "type": "str"},
            },
        }
        result = serializer.deserialize(data)
        assert result == {"a": 1, "b": "hello"}

    def test_serialize_empty_dict(self):
        serializer = DictSerializer()
        result = serializer.serialize({})
        assert result == {"type": "dict", "value": {}}

    def test_serialize_nested_dict(self):
        serializer = DictSerializer()
        result = serializer.serialize({"outer": {"inner": 42}})
        assert result["type"] == "dict"
        assert "outer" in result["value"]


class TestNumpySerializer:
    """Tests for NumpySerializer."""

    def test_can_serialize(self, numpy):
        serializer = NumpySerializer()
        arr = numpy.array([1, 2, 3])
        assert serializer.can_serialize(arr)
        assert not serializer.can_serialize([1, 2, 3])

    def test_serialize_array(self, numpy):
        serializer = NumpySerializer()
        arr = numpy.array([1.0, 2.0, 3.0])
        result = serializer.serialize(arr)
        assert result["type"] == "ndarray"
        assert result["value"] == [1.0, 2.0, 3.0]
        assert result["shape"] == [3]
        assert "float" in result["dtype"]

    def test_deserialize_array(self, numpy):
        serializer = NumpySerializer()
        data = {
            "type": "ndarray",
            "value": [1.0, 2.0, 3.0],
            "dtype": "float64",
            "shape": [3],
        }
        result = serializer.deserialize(data)
        assert isinstance(result, numpy.ndarray)
        assert numpy.array_equal(result, numpy.array([1.0, 2.0, 3.0]))

    def test_serialize_2d_array(self, numpy):
        serializer = NumpySerializer()
        arr = numpy.array([[1, 2], [3, 4]])
        result = serializer.serialize(arr)
        assert result["shape"] == [2, 2]

    def test_deserialize_2d_array(self, numpy):
        serializer = NumpySerializer()
        data = {
            "type": "ndarray",
            "value": [[1, 2], [3, 4]],
            "dtype": "int64",
            "shape": [2, 2],
        }
        result = serializer.deserialize(data)
        assert result.shape == (2, 2)

    def test_serialize_int_array(self, numpy):
        serializer = NumpySerializer()
        arr = numpy.array([1, 2, 3], dtype=numpy.int32)
        result = serializer.serialize(arr)
        assert "int" in result["dtype"]

    def test_serialize_complex_array(self, numpy):
        serializer = NumpySerializer()
        arr = numpy.array([1 + 2j, 3 + 4j])
        result = serializer.serialize(arr)
        assert "complex" in result["dtype"]


class TestPandasSerializer:
    """Tests for PandasSerializer."""

    @pytest.fixture
    def pandas(self):
        pytest.importorskip("pandas")
        import pandas as pd

        return pd

    def test_can_serialize_dataframe(self, pandas):
        serializer = PandasSerializer()
        df = pandas.DataFrame({"a": [1, 2], "b": [3, 4]})
        assert serializer.can_serialize(df)

    def test_can_serialize_series(self, pandas):
        serializer = PandasSerializer()
        s = pandas.Series([1, 2, 3], name="test")
        assert serializer.can_serialize(s)

    def test_cannot_serialize_non_pandas(self):
        serializer = PandasSerializer()
        assert not serializer.can_serialize([1, 2, 3])
        assert not serializer.can_serialize({"a": 1})
        assert not serializer.can_serialize(42)

    def test_serialize_dataframe(self, pandas):
        serializer = PandasSerializer()
        df = pandas.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        result = serializer.serialize(df)

        assert result["type"] == "pandas.DataFrame"
        assert result["columns"] == ["a", "b"]
        assert result["value"]["a"] == [1, 2]
        assert result["value"]["b"] == [3.0, 4.0]
        assert result["index"] == [0, 1]

    def test_serialize_dataframe_with_custom_index(self, pandas):
        serializer = PandasSerializer()
        df = pandas.DataFrame(
            {"a": [1, 2], "b": [3, 4]}, index=pandas.Index(["x", "y"], name="idx")
        )
        result = serializer.serialize(df)

        assert result["index"] == ["x", "y"]
        assert result["index_name"] == "idx"

    def test_serialize_series(self, pandas):
        serializer = PandasSerializer()
        s = pandas.Series([1, 2, 3], name="test_series")
        result = serializer.serialize(s)

        assert result["type"] == "pandas.Series"
        assert result["value"] == [1, 2, 3]
        assert result["name"] == "test_series"
        assert result["index"] == [0, 1, 2]

    def test_serialize_series_with_custom_index(self, pandas):
        serializer = PandasSerializer()
        s = pandas.Series(
            [1, 2, 3],
            name="test",
            index=pandas.Index(["a", "b", "c"], name="letter"),
        )
        result = serializer.serialize(s)

        assert result["index"] == ["a", "b", "c"]
        assert result["index_name"] == "letter"

    def test_deserialize_dataframe(self, pandas):
        serializer = PandasSerializer()
        data = {
            "type": "pandas.DataFrame",
            "value": {"a": [1, 2], "b": [3.0, 4.0]},
            "columns": ["a", "b"],
            "index": [0, 1],
            "index_name": None,
        }
        result = serializer.deserialize(data)

        assert isinstance(result, pandas.DataFrame)
        assert list(result.columns) == ["a", "b"]
        assert result["a"].tolist() == [1, 2]
        assert result["b"].tolist() == [3.0, 4.0]

    def test_deserialize_dataframe_with_custom_index(self, pandas):
        serializer = PandasSerializer()
        data = {
            "type": "pandas.DataFrame",
            "value": {"a": [1, 2]},
            "columns": ["a"],
            "index": ["x", "y"],
            "index_name": "idx",
        }
        result = serializer.deserialize(data)

        assert result.index.tolist() == ["x", "y"]
        assert result.index.name == "idx"

    def test_deserialize_series(self, pandas):
        serializer = PandasSerializer()
        data = {
            "type": "pandas.Series",
            "value": [1, 2, 3],
            "name": "test_series",
            "index": [0, 1, 2],
            "index_name": None,
        }
        result = serializer.deserialize(data)

        assert isinstance(result, pandas.Series)
        assert result.name == "test_series"
        assert result.tolist() == [1, 2, 3]

    def test_deserialize_series_with_custom_index(self, pandas):
        serializer = PandasSerializer()
        data = {
            "type": "pandas.Series",
            "value": [1, 2, 3],
            "name": "test",
            "index": ["a", "b", "c"],
            "index_name": "letter",
        }
        result = serializer.deserialize(data)

        assert result.index.tolist() == ["a", "b", "c"]
        assert result.index.name == "letter"

    def test_dataframe_roundtrip(self, pandas):
        serializer = PandasSerializer()
        df = pandas.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        serialized = serializer.serialize(df)
        deserialized = serializer.deserialize(serialized)

        assert df.equals(deserialized)

    def test_series_roundtrip(self, pandas):
        serializer = PandasSerializer()
        s = pandas.Series([1, 2, 3], name="test")
        serialized = serializer.serialize(s)
        deserialized = serializer.deserialize(serialized)

        assert s.equals(deserialized)


class TestPolarsSerializer:
    """Tests for PolarsSerializer."""

    @pytest.fixture
    def polars(self):
        pytest.importorskip("polars")
        import polars as pl

        return pl

    def test_can_serialize_dataframe(self, polars):
        serializer = PolarsSerializer()
        df = polars.DataFrame({"a": [1, 2], "b": [3, 4]})
        assert serializer.can_serialize(df)

    def test_can_serialize_series(self, polars):
        serializer = PolarsSerializer()
        s = polars.Series("test", [1, 2, 3])
        assert serializer.can_serialize(s)

    def test_cannot_serialize_non_polars(self):
        serializer = PolarsSerializer()
        assert not serializer.can_serialize([1, 2, 3])
        assert not serializer.can_serialize({"a": 1})
        assert not serializer.can_serialize(42)

    def test_serialize_dataframe(self, polars):
        serializer = PolarsSerializer()
        df = polars.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        result = serializer.serialize(df)

        assert result["type"] == "polars.DataFrame"
        assert result["columns"] == ["a", "b"]
        assert result["value"]["a"] == [1, 2]
        assert result["value"]["b"] == [3.0, 4.0]
        assert "schema" in result

    def test_serialize_dataframe_preserves_schema(self, polars):
        serializer = PolarsSerializer()
        df = polars.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        result = serializer.serialize(df)

        assert "Int64" in result["schema"]["a"]
        assert "String" in result["schema"]["b"] or "Utf8" in result["schema"]["b"]

    def test_serialize_series(self, polars):
        serializer = PolarsSerializer()
        s = polars.Series("test_series", [1, 2, 3])
        result = serializer.serialize(s)

        assert result["type"] == "polars.Series"
        assert result["value"] == [1, 2, 3]
        assert result["name"] == "test_series"
        assert "dtype" in result

    def test_deserialize_dataframe(self, polars):
        serializer = PolarsSerializer()
        data = {
            "type": "polars.DataFrame",
            "value": {"a": [1, 2], "b": [3.0, 4.0]},
            "columns": ["a", "b"],
            "schema": {"a": "Int64", "b": "Float64"},
        }
        result = serializer.deserialize(data)

        assert isinstance(result, polars.DataFrame)
        assert result.columns == ["a", "b"]
        assert result["a"].to_list() == [1, 2]
        assert result["b"].to_list() == [3.0, 4.0]

    def test_deserialize_series(self, polars):
        serializer = PolarsSerializer()
        data = {
            "type": "polars.Series",
            "value": [1, 2, 3],
            "name": "test_series",
            "dtype": "Int64",
        }
        result = serializer.deserialize(data)

        assert isinstance(result, polars.Series)
        assert result.name == "test_series"
        assert result.to_list() == [1, 2, 3]

    def test_dataframe_roundtrip(self, polars):
        serializer = PolarsSerializer()
        df = polars.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        serialized = serializer.serialize(df)
        deserialized = serializer.deserialize(serialized)

        assert df.equals(deserialized)

    def test_series_roundtrip(self, polars):
        serializer = PolarsSerializer()
        s = polars.Series("test", [1, 2, 3])
        serialized = serializer.serialize(s)
        deserialized = serializer.deserialize(serialized)

        assert s.equals(deserialized)


class TestGenericSerializer:
    """Tests for GenericSerializer."""

    def test_can_serialize_anything(self):
        serializer = GenericSerializer()
        assert serializer.can_serialize(42)
        assert serializer.can_serialize("hello")
        assert serializer.can_serialize(object())

    def test_serialize_with_to_dict(self):
        class HasToDict:
            def to_dict(self):
                return {"key": "value"}

        serializer = GenericSerializer()
        result = serializer.serialize(HasToDict())
        assert result["_serialization"] == "to_dict"
        assert result["value"] == {"key": "value"}

    def test_serialize_with_dict_attr(self):
        class HasDict:
            def __init__(self):
                self.x = 1
                self.y = 2

        serializer = GenericSerializer()
        result = serializer.serialize(HasDict())
        assert result["_serialization"] == "__dict__"
        assert result["value"]["x"] == 1
        assert result["value"]["y"] == 2

    def test_serialize_fallback_to_str(self):
        serializer = GenericSerializer()
        result = serializer.serialize(42)
        assert result["_serialization"] == "str"
        assert result["value"] == "42"


class TestGetSerializer:
    """Tests for get_serializer function."""

    def test_get_scalar_serializer(self):
        assert isinstance(get_serializer(42), ScalarSerializer)
        assert isinstance(get_serializer(3.14), ScalarSerializer)
        assert isinstance(get_serializer("hello"), ScalarSerializer)

    def test_get_list_serializer(self):
        assert isinstance(get_serializer([1, 2, 3]), ListSerializer)
        assert isinstance(get_serializer((1, 2, 3)), ListSerializer)

    def test_get_dict_serializer(self):
        assert isinstance(get_serializer({"a": 1}), DictSerializer)

    def test_get_numpy_serializer(self, numpy):
        arr = numpy.array([1, 2, 3])
        assert isinstance(get_serializer(arr), NumpySerializer)

    def test_get_pandas_serializer(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"a": [1, 2]})
        s = pd.Series([1, 2, 3])
        assert isinstance(get_serializer(df), PandasSerializer)
        assert isinstance(get_serializer(s), PandasSerializer)

    def test_get_polars_serializer(self):
        pl = pytest.importorskip("polars")
        df = pl.DataFrame({"a": [1, 2]})
        s = pl.Series("test", [1, 2, 3])
        assert isinstance(get_serializer(df), PolarsSerializer)
        assert isinstance(get_serializer(s), PolarsSerializer)


class TestSerializeDeserializeRoundtrip:
    """Tests for serialization roundtrip."""

    def test_scalar_roundtrip(self):
        for value in [42, 3.14, "hello", True, None]:
            serialized = serialize_value(value)
            deserialized = deserialize_value(serialized)
            assert deserialized == value

    def test_list_roundtrip(self):
        value = [1, 2, "three", 4.0]
        serialized = serialize_value(value)
        deserialized = deserialize_value(serialized)
        assert deserialized == value

    def test_dict_roundtrip(self):
        value = {"a": 1, "b": "hello", "c": [1, 2, 3]}
        serialized = serialize_value(value)
        deserialized = deserialize_value(serialized)
        assert deserialized == value

    def test_numpy_roundtrip(self, numpy):
        value = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        serialized = serialize_value(value)
        deserialized = deserialize_value(serialized)
        assert numpy.array_equal(value, deserialized)

    def test_nested_structure_roundtrip(self):
        value = {
            "numbers": [1, 2, 3],
            "nested": {"a": 1, "b": [4, 5]},
            "scalar": "test",
        }
        serialized = serialize_value(value)
        deserialized = deserialize_value(serialized)
        assert deserialized == value

    def test_special_float_roundtrip(self):
        import math

        # Positive infinity
        serialized = serialize_value(math.inf)
        deserialized = deserialize_value(serialized)
        assert math.isinf(deserialized) and deserialized > 0

        # Negative infinity
        serialized = serialize_value(-math.inf)
        deserialized = deserialize_value(serialized)
        assert math.isinf(deserialized) and deserialized < 0

        # NaN
        serialized = serialize_value(math.nan)
        deserialized = deserialize_value(serialized)
        assert math.isnan(deserialized)

    def test_list_with_special_floats_roundtrip(self):
        import math

        value = [1.0, math.inf, -math.inf, math.nan, 2.0]
        serialized = serialize_value(value)
        deserialized = deserialize_value(serialized)

        assert deserialized[0] == 1.0
        assert math.isinf(deserialized[1]) and deserialized[1] > 0
        assert math.isinf(deserialized[2]) and deserialized[2] < 0
        assert math.isnan(deserialized[3])
        assert deserialized[4] == 2.0

    def test_dict_with_special_floats_roundtrip(self):
        import math

        value = {
            "normal": 1.0,
            "pos_inf": math.inf,
            "neg_inf": -math.inf,
            "nan": math.nan,
        }
        serialized = serialize_value(value)
        deserialized = deserialize_value(serialized)

        assert deserialized["normal"] == 1.0
        assert math.isinf(deserialized["pos_inf"]) and deserialized["pos_inf"] > 0
        assert math.isinf(deserialized["neg_inf"]) and deserialized["neg_inf"] < 0
        assert math.isnan(deserialized["nan"])

    def test_pandas_dataframe_roundtrip(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        serialized = serialize_value(df)
        deserialized = deserialize_value(serialized)
        assert df.equals(deserialized)

    def test_pandas_series_roundtrip(self):
        pd = pytest.importorskip("pandas")
        s = pd.Series([1, 2, 3], name="test")
        serialized = serialize_value(s)
        deserialized = deserialize_value(serialized)
        assert s.equals(deserialized)

    def test_polars_dataframe_roundtrip(self):
        pl = pytest.importorskip("polars")
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        serialized = serialize_value(df)
        deserialized = deserialize_value(serialized)
        assert df.equals(deserialized)

    def test_polars_series_roundtrip(self):
        pl = pytest.importorskip("polars")
        s = pl.Series("test", [1, 2, 3])
        serialized = serialize_value(s)
        deserialized = deserialize_value(serialized)
        assert s.equals(deserialized)
