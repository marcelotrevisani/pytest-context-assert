"""Value serialization for snapshot storage.

This module provides serializers that convert Python values to YAML-compatible
dictionary representations and back. Each serializer handles specific types
and knows how to preserve type information during serialization.

Built-in Serializers:
    ScalarSerializer: Handles int, float, str, bool, None, and numpy scalars.
        Also handles special float values (inf, -inf, nan).
    ListSerializer: Handles list and tuple types with recursive serialization.
    DictSerializer: Handles dict types with recursive value serialization.
    NumpySerializer: Handles numpy.ndarray with shape and dtype preservation.
    PandasSerializer: Handles pandas.DataFrame and pandas.Series with full
        metadata (columns, index, index_name).
    PolarsSerializer: Handles polars.DataFrame and polars.Series with schema
        and dtype information.
    GenericSerializer: Fallback for unknown types using to_dict(), __dict__,
        or str() representation.

Usage:
    >>> from pytest_context_assert.serializers import serialize_value, deserialize_value
    >>> data = serialize_value([1, 2, 3])
    >>> value = deserialize_value(data)

    >>> import pandas as pd
    >>> df = pd.DataFrame({"a": [1, 2, 3]})
    >>> data = serialize_value(df)
    >>> df_restored = deserialize_value(data)

Serializer Priority:
    When get_serializer() is called, serializers are tried in order:
    1. ScalarSerializer
    2. NumpySerializer
    3. PandasSerializer
    4. PolarsSerializer
    5. ListSerializer
    6. DictSerializer
    7. GenericSerializer (always matches)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


class Serializer(ABC):
    """Abstract base class for value serializers."""

    @abstractmethod
    def can_serialize(self, value: Any) -> bool:
        """Check if this serializer can handle the given value."""
        ...

    @abstractmethod
    def serialize(self, value: Any) -> dict[str, Any]:
        """Serialize a value to a dictionary representation.

        Returns:
            Dictionary containing at minimum 'value' and 'type' keys.
        """
        ...

    @abstractmethod
    def deserialize(self, data: dict[str, Any]) -> Any:
        """Deserialize a dictionary back to the original value type."""
        ...


class ScalarSerializer(Serializer):
    """Serializer for scalar types (int, float, str, bool, None).

    Also handles special float values:
    - positive infinity (+inf)
    - negative infinity (-inf)
    - NaN (not a number)
    """

    SCALAR_TYPES = (int, float, str, bool, type(None))
    TYPE_MAP = {
        int: "int",
        float: "float",
        str: "str",
        bool: "bool",
        type(None): "none",
    }
    REVERSE_TYPE_MAP = {v: k for k, v in TYPE_MAP.items()}

    # Special float value markers
    POSITIVE_INF = "__POSITIVE_INFINITY__"
    NEGATIVE_INF = "__NEGATIVE_INFINITY__"
    NAN = "__NAN__"

    def can_serialize(self, value: Any) -> bool:
        # Handle numpy scalar types
        if isinstance(value, self.SCALAR_TYPES):
            return True
        # Check for numpy scalar types
        try:
            import numpy as np

            if isinstance(value, (np.integer, np.floating, np.bool_)):
                return True
        except ImportError:
            pass
        return False

    def _is_special_float(self, value: float) -> str | None:
        """Check if value is a special float and return its marker."""
        import math

        if math.isinf(value):
            return self.POSITIVE_INF if value > 0 else self.NEGATIVE_INF
        if math.isnan(value):
            return self.NAN
        return None

    def _from_special_float(self, marker: str) -> float | None:
        """Convert a special float marker back to the float value."""
        import math

        if marker == self.POSITIVE_INF:
            return math.inf
        elif marker == self.NEGATIVE_INF:
            return -math.inf
        elif marker == self.NAN:
            return math.nan
        return None

    def serialize(self, value: Any) -> dict[str, Any]:
        # Convert numpy types to Python types
        python_value = value
        type_name = self.TYPE_MAP.get(type(value))

        if type_name is None:
            # Handle numpy scalar types
            try:
                import numpy as np

                if isinstance(value, np.integer):
                    python_value = int(value)
                    type_name = "int"
                elif isinstance(value, np.floating):
                    python_value = float(value)
                    type_name = "float"
                elif isinstance(value, np.bool_):
                    python_value = bool(value)
                    type_name = "bool"
            except ImportError:
                pass

        if type_name is None:
            type_name = type(value).__name__

        # Handle special float values (inf, -inf, nan)
        if type_name == "float" and isinstance(python_value, float):
            special_marker = self._is_special_float(python_value)
            if special_marker is not None:
                return {
                    "value": special_marker,
                    "type": "float",
                    "_special": True,
                }

        return {
            "value": python_value,
            "type": type_name,
        }

    def deserialize(self, data: dict[str, Any]) -> Any:
        type_name = data.get("type", "")
        value = data.get("value")
        is_special = data.get("_special", False)

        if type_name == "none":
            return None

        # Handle special float values
        if type_name == "float" and is_special and isinstance(value, str):
            special_value = self._from_special_float(value)
            if special_value is not None:
                return special_value

        if type_name in self.REVERSE_TYPE_MAP:
            expected_type = self.REVERSE_TYPE_MAP[type_name]
            if value is not None and not isinstance(value, expected_type):
                return expected_type(value)
        return value


class ListSerializer(Serializer):
    """Serializer for list types."""

    def can_serialize(self, value: Any) -> bool:
        return isinstance(value, (list, tuple))

    def serialize(self, value: Any) -> dict[str, Any]:
        is_tuple = isinstance(value, tuple)
        serialized_items = []
        for item in value:
            serializer = get_serializer(item)
            serialized_items.append(serializer.serialize(item))

        return {
            "value": serialized_items,
            "type": "tuple" if is_tuple else "list",
        }

    def deserialize(self, data: dict[str, Any]) -> Any:
        type_name = data.get("type", "list")
        items = data.get("value", [])

        deserialized = []
        for item in items:
            if isinstance(item, dict) and "type" in item:
                serializer = _get_serializer_by_type(item["type"])
                deserialized.append(serializer.deserialize(item))
            else:
                deserialized.append(item)

        if type_name == "tuple":
            return tuple(deserialized)
        return deserialized


class DictSerializer(Serializer):
    """Serializer for dictionary types."""

    def can_serialize(self, value: Any) -> bool:
        return isinstance(value, dict)

    def serialize(self, value: Any) -> dict[str, Any]:
        serialized_items = {}
        for k, v in value.items():
            serializer = get_serializer(v)
            serialized_items[k] = serializer.serialize(v)

        return {
            "value": serialized_items,
            "type": "dict",
        }

    def deserialize(self, data: dict[str, Any]) -> Any:
        items = data.get("value", {})

        deserialized = {}
        for k, v in items.items():
            if isinstance(v, dict) and "type" in v:
                serializer = _get_serializer_by_type(v["type"])
                deserialized[k] = serializer.deserialize(v)
            else:
                deserialized[k] = v

        return deserialized


class NumpySerializer(Serializer):
    """Serializer for numpy arrays."""

    def can_serialize(self, value: Any) -> bool:
        try:
            import numpy as np

            return isinstance(value, np.ndarray)
        except ImportError:
            return False

    def _convert_complex_to_serializable(self, value: Any) -> Any:
        """Convert complex numbers to a serializable dict format."""
        if isinstance(value, complex):
            return {"_complex": True, "real": value.real, "imag": value.imag}
        elif isinstance(value, list):
            return [self._convert_complex_to_serializable(v) for v in value]
        return value

    def _convert_serializable_to_complex(self, value: Any) -> Any:
        """Convert serialized complex dicts back to complex numbers."""
        if isinstance(value, dict) and value.get("_complex"):
            return complex(value["real"], value["imag"])
        elif isinstance(value, list):
            return [self._convert_serializable_to_complex(v) for v in value]
        return value

    def serialize(self, value: Any) -> dict[str, Any]:
        import numpy as np

        dtype_str = str(value.dtype)
        serialized_value = value.tolist()

        # Handle complex dtypes specially to avoid !!python/complex tags
        if np.issubdtype(value.dtype, np.complexfloating):
            serialized_value = self._convert_complex_to_serializable(serialized_value)

        return {
            "value": serialized_value,
            "type": "ndarray",
            "dtype": dtype_str,
            "shape": list(value.shape),
        }

    def deserialize(self, data: dict[str, Any]) -> Any:
        import numpy as np

        value = data.get("value")
        dtype = data.get("dtype", "float64")
        shape = data.get("shape")

        # Handle complex arrays
        if "complex" in dtype:
            value = self._convert_serializable_to_complex(value)

        arr = np.array(value, dtype=dtype)
        if shape:
            arr = arr.reshape(shape)
        return arr


class PandasSerializer(Serializer):
    """Serializer for pandas DataFrames and Series."""

    def can_serialize(self, value: Any) -> bool:
        module = type(value).__module__
        type_name = type(value).__name__
        return "pandas" in module and type_name in ("DataFrame", "Series")

    def _is_pandas_dataframe(self, value: Any) -> bool:
        return type(value).__name__ == "DataFrame"

    def _is_pandas_series(self, value: Any) -> bool:
        return type(value).__name__ == "Series"

    def serialize(self, value: Any) -> dict[str, Any]:
        if self._is_pandas_dataframe(value):
            return {
                "value": value.to_dict(orient="list"),
                "type": "pandas.DataFrame",
                "columns": value.columns.tolist(),
                "index": value.index.tolist(),
                "index_name": value.index.name,
            }
        else:  # Series
            return {
                "value": value.tolist(),
                "type": "pandas.Series",
                "name": value.name,
                "index": value.index.tolist(),
                "index_name": value.index.name,
            }

    def deserialize(self, data: dict[str, Any]) -> Any:
        try:
            import pandas as pd
        except ImportError:
            return data.get("value")

        type_name = data.get("type", "")

        if type_name == "pandas.DataFrame":
            df = pd.DataFrame(data["value"])
            if "index" in data and data["index"]:
                df.index = pd.Index(data["index"], name=data.get("index_name"))
            return df
        elif type_name == "pandas.Series":
            s = pd.Series(data["value"], name=data.get("name"))
            if "index" in data and data["index"]:
                s.index = pd.Index(data["index"], name=data.get("index_name"))
            return s

        return data.get("value")


class PolarsSerializer(Serializer):
    """Serializer for polars DataFrames and Series."""

    def can_serialize(self, value: Any) -> bool:
        module = type(value).__module__
        type_name = type(value).__name__
        return "polars" in module and type_name in ("DataFrame", "Series")

    def _is_polars_dataframe(self, value: Any) -> bool:
        return type(value).__name__ == "DataFrame"

    def _is_polars_series(self, value: Any) -> bool:
        return type(value).__name__ == "Series"

    def _df_to_dict(self, df: Any) -> dict:
        """Convert Polars DataFrame to a dict with Python native types."""
        return {col: df[col].to_list() for col in df.columns}

    def serialize(self, value: Any) -> dict[str, Any]:
        if self._is_polars_dataframe(value):
            return {
                "value": self._df_to_dict(value),
                "type": "polars.DataFrame",
                "columns": value.columns,
                "schema": {col: str(dtype) for col, dtype in value.schema.items()},
            }
        else:  # Series
            return {
                "value": value.to_list(),
                "type": "polars.Series",
                "name": value.name,
                "dtype": str(value.dtype),
            }

    def deserialize(self, data: dict[str, Any]) -> Any:
        try:
            import polars as pl
        except ImportError:
            return data.get("value")

        type_name = data.get("type", "")

        if type_name == "polars.DataFrame":
            return pl.DataFrame(data["value"])
        elif type_name == "polars.Series":
            return pl.Series(data.get("name", ""), data["value"])

        return data.get("value")


class GenericSerializer(Serializer):
    """Fallback serializer that attempts to serialize any object."""

    def can_serialize(self, value: Any) -> bool:
        return True

    def serialize(self, value: Any) -> dict[str, Any]:
        type_name = type(value).__name__
        module = type(value).__module__

        if hasattr(value, "to_dict"):
            return {
                "value": value.to_dict(),
                "type": f"{module}.{type_name}",
                "_serialization": "to_dict",
            }
        elif hasattr(value, "__dict__"):
            return {
                "value": value.__dict__.copy(),
                "type": f"{module}.{type_name}",
                "_serialization": "__dict__",
            }
        else:
            return {
                "value": str(value),
                "type": f"{module}.{type_name}",
                "_serialization": "str",
            }

    def deserialize(self, data: dict[str, Any]) -> Any:
        return data.get("value")


_SERIALIZERS: list[Serializer] = [
    ScalarSerializer(),
    NumpySerializer(),
    PandasSerializer(),
    PolarsSerializer(),
    ListSerializer(),
    DictSerializer(),
    GenericSerializer(),
]


def get_serializer(value: Any) -> Serializer:
    """Get the appropriate serializer for a value.

    Args:
        value: The value to serialize.

    Returns:
        A Serializer instance capable of handling the value.
    """
    for serializer in _SERIALIZERS:
        if serializer.can_serialize(value):
            return serializer
    return GenericSerializer()


def _get_serializer_by_type(type_name: str) -> Serializer:
    """Get a serializer based on the type name from stored data."""
    type_map = {
        "int": ScalarSerializer(),
        "float": ScalarSerializer(),
        "str": ScalarSerializer(),
        "bool": ScalarSerializer(),
        "none": ScalarSerializer(),
        "list": ListSerializer(),
        "tuple": ListSerializer(),
        "dict": DictSerializer(),
        "ndarray": NumpySerializer(),
        "pandas.DataFrame": PandasSerializer(),
        "pandas.Series": PandasSerializer(),
        "polars.DataFrame": PolarsSerializer(),
        "polars.Series": PolarsSerializer(),
    }
    return type_map.get(type_name, GenericSerializer())


def serialize_value(value: Any) -> dict[str, Any]:
    """Convenience function to serialize any value.

    Args:
        value: The value to serialize.

    Returns:
        Dictionary representation of the value.
    """
    serializer = get_serializer(value)
    return serializer.serialize(value)


def deserialize_value(data: dict[str, Any]) -> Any:
    """Convenience function to deserialize any stored value.

    Args:
        data: Dictionary representation from storage.

    Returns:
        The deserialized value.
    """
    type_name = data.get("type", "")
    serializer = _get_serializer_by_type(type_name)
    return serializer.deserialize(data)
