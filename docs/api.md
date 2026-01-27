# API Reference

## Fixture

### `context_assert`

The main fixture provided by the plugin. Use it in your test functions to make context-aware assertions.

```python
def test_example(context_assert):
    result = compute()
    context_assert(result)
```

## ContextAssert Class

The `context_assert` fixture returns a `ContextAssert` instance.

### Methods

#### `__call__(value, *, rtol=None, atol=None, name=None, compare=None, serialize=None, deserialize=None)`

Assert that a value matches the stored snapshot for the current context.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `value` | Any | required | The actual value to compare |
| `rtol` | float | `1e-7` | Relative tolerance for float/array comparisons |
| `atol` | float | `0` | Absolute tolerance for float/array comparisons |
| `name` | str | `None` | Name for this assertion (auto-generated as "assertion_1", "assertion_2", etc. based on call order if not provided) |
| `compare` | callable | `None` | Custom comparison function `(actual, expected) -> bool` |
| `serialize` | callable | `None` | Custom serialization function `(value) -> dict` |
| `deserialize` | callable | `None` | Custom deserialization function `(dict) -> value` |

**Returns:** `None`

**Raises:**
- `ContextAssertionError`: If the value doesn't match the expected snapshot
- `MissingSnapshotError`: If no snapshot exists and update mode is not enabled

**Example:**

```python
def test_values(context_assert):
    # Simple assertion
    context_assert(42)

    # With tolerance
    context_assert(3.14159, rtol=1e-5)

    # Named assertion
    context_assert(result, name="my_result")

    # Custom comparison
    context_assert(
        "HELLO",
        name="greeting",
        compare=lambda a, b: a.lower() == b.lower()
    )

    # Custom serialization
    context_assert(
        custom_obj,
        name="custom",
        serialize=lambda obj: {"x": obj.x, "y": obj.y},
        deserialize=lambda d: CustomObj(d["x"], d["y"]),
        compare=lambda a, b: a.x == b.x
    )
```

---

#### `assert_match(value, *, name=None, **kwargs)`

Syrupy-style alias for `__call__`. All parameters are passed through.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `value` | Any | required | The actual value to compare |
| `name` | str | `None` | Name for this assertion |
| `**kwargs` | Any | - | Additional parameters (rtol, atol, compare, serialize, deserialize) |

**Example:**

```python
def test_syrupy_style(context_assert):
    context_assert.assert_match(result, name="my_result")
    context_assert.assert_match(value, name="tolerant", rtol=1e-5)
```

---

#### `with_context(**kwargs) -> ContextAssert`

Create a new `ContextAssert` instance with additional custom context keys.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `**kwargs` | Any | Custom context key-value pairs |

**Returns:** A new `ContextAssert` instance with merged context

**Example:**

```python
def test_gpu(context_assert):
    custom = context_assert.with_context(gpu="nvidia", cuda="11.8")
    # Context key becomes: darwin-arm64-accelerate-cuda_11.8-gpu_nvidia
    custom(result, name="gpu_result")
```

---

#### `update(value, *, name=None, context_key=None)`

Explicitly update a snapshot value without comparing.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `value` | Any | required | The value to store |
| `name` | str | `None` | Name for this assertion |
| `context_key` | str | `None` | Context key to update (defaults to current context) |

**Example:**

```python
def test_update(context_assert):
    # Update snapshot for current context
    context_assert.update(new_value, name="my_value")

    # Update snapshot for specific context
    context_assert.update(value, name="my_value", context_key="linux-x86_64")
```

---

#### `set_default(value, *, name=None)`

Set the default value for this assertion. The default is used when no context-specific value exists.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `value` | Any | required | The default value to store |
| `name` | str | `None` | Name for this assertion |

**Example:**

```python
def test_with_default(context_assert):
    context_assert.set_default(42, name="my_value")
```

---

### Properties

#### `context_key -> str`

Get the current context key string.

```python
def test_context(context_assert):
    print(context_assert.context_key)
    # Output: darwin-arm64-accelerate
```

#### `storage -> SnapshotStorage`

Get the snapshot storage instance for the current test.

```python
def test_storage(context_assert):
    storage = context_assert.storage
    print(storage.snapshot_path)
```

---

## ContextResolver Class

Resolves the current execution context.

### Constructor

```python
ContextResolver(custom_context: dict[str, Any] | None = None)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `custom_context` | dict | `None` | Custom context key-value pairs to merge |

### Methods

#### `get_platform() -> str`

Get the current platform.

**Returns:** `"linux"`, `"darwin"`, or `"windows"`

#### `get_architecture() -> str`

Get the current CPU architecture.

**Returns:** `"x86_64"`, `"arm64"`, `"x86"`, or the raw machine name

#### `get_blas() -> str | None`

Get the BLAS library in use (requires NumPy).

**Returns:** `"openblas"`, `"mkl"`, `"accelerate"`, `"blis"`, or `None`

#### `get_context() -> dict[str, str]`

Get the full context dictionary.

**Returns:** Dictionary with keys like `platform`, `arch`, `blas`, plus any custom keys

#### `get_context_key() -> str`

Get the context key string.

**Returns:** Hyphen-separated string like `"darwin-arm64-accelerate"`

#### `with_custom_context(**kwargs) -> ContextResolver`

Create a new resolver with additional custom context.

**Returns:** New `ContextResolver` instance

---

## SnapshotStorage Class

Manages reading and writing snapshot files.

### Constructor

```python
SnapshotStorage(
    test_file: Path,
    test_name: str,
    snapshot_dir: str = "__snapshots__"
)
```

### Methods

#### `exists() -> bool`

Check if a snapshot file exists for this test.

#### `load() -> dict[str, Any]`

Load the snapshot data from file.

#### `save(data: dict[str, Any])`

Save snapshot data to file.

#### `get_value(context_key: str, name: str | None = None) -> Any | None`

Get the expected value for a context, with fallback to default.

#### `set_value(context_key: str, value: Any, name: str | None = None)`

Store a value for a specific context.

#### `has_value(context_key: str, name: str | None = None) -> bool`

Check if a value exists for the given context.

---

## Comparators

### Base Class: `Comparator`

Abstract base class for all comparators.

```python
class Comparator(ABC):
    def can_compare(self, value: Any) -> bool: ...
    def compare(self, actual: Any, expected: Any, *, rtol=1e-7, atol=0) -> ComparisonResult: ...
```

### `ComparisonResult`

Dataclass returned by comparators.

```python
@dataclass
class ComparisonResult:
    equal: bool
    message: str = ""
    actual: Any = None
    expected: Any = None
    diff: str | None = None
```

### Built-in Comparators

#### `ScalarComparator`

For `int`, `float`, `str`, `bool`, `None`. Float comparison uses:

```python
abs(actual - expected) <= atol + rtol * abs(expected)
```

Special float values are handled correctly:
- `inf == inf` returns `True`
- `-inf == -inf` returns `True`
- `nan == nan` returns `True` (for snapshot testing purposes)
- `inf != -inf` returns `False`

#### `NumpyArrayComparator`

For `numpy.ndarray`. Uses `np.allclose()` for floating-point arrays. Also handles complex arrays.

#### `ListComparator`

For `list` and `tuple`. Compares element-by-element recursively.

#### `DictComparator`

For `dict`. Compares keys and values recursively.

#### `GenericComparator`

Fallback using `==` operator.

#### `FunctionComparator`

Uses a user-provided comparison function.

```python
from pytest_context_assert import FunctionComparator

comparator = FunctionComparator(lambda a, b: a.lower() == b.lower())
result = comparator.compare("Hello", "HELLO")
assert result.equal
```

### `get_comparator(value: Any) -> Comparator`

Get the appropriate comparator for a value type.

---

## Serializers

### Base Class: `Serializer`

Abstract base class for all serializers.

```python
class Serializer(ABC):
    def can_serialize(self, value: Any) -> bool: ...
    def serialize(self, value: Any) -> dict[str, Any]: ...
    def deserialize(self, data: dict[str, Any]) -> Any: ...
```

### Built-in Serializers

#### `ScalarSerializer`

Handles scalar types: `int`, `float`, `str`, `bool`, `None`.

Also handles special float values:
- Positive infinity (`math.inf`) - serialized as `"__POSITIVE_INFINITY__"`
- Negative infinity (`-math.inf`) - serialized as `"__NEGATIVE_INFINITY__"`
- NaN (`math.nan`) - serialized as `"__NAN__"`

Numpy scalar types (`np.int64`, `np.float64`, etc.) are automatically converted to Python types.

```python
from pytest_context_assert.serializers import ScalarSerializer
import math

serializer = ScalarSerializer()

# Regular scalars
serializer.serialize(42)  # {'value': 42, 'type': 'int'}
serializer.serialize(3.14)  # {'value': 3.14, 'type': 'float'}

# Special floats
serializer.serialize(math.inf)  # {'value': '__POSITIVE_INFINITY__', 'type': 'float', '_special': True}
serializer.serialize(math.nan)  # {'value': '__NAN__', 'type': 'float', '_special': True}
```

#### `ListSerializer`

Handles `list` and `tuple` types. Elements are recursively serialized.

```python
serializer.serialize([1, 2, 3])  # {'value': [...], 'type': 'list'}
serializer.serialize((1, 2, 3))  # {'value': [...], 'type': 'tuple'}
```

#### `DictSerializer`

Handles `dict` types. Values are recursively serialized.

```python
serializer.serialize({"a": 1})  # {'value': {...}, 'type': 'dict'}
```

#### `NumpySerializer`

Handles `numpy.ndarray` types. Preserves shape and dtype.

```python
import numpy as np
from pytest_context_assert.serializers import NumpySerializer

serializer = NumpySerializer()
arr = np.array([[1.0, 2.0], [3.0, 4.0]])
serializer.serialize(arr)
# {
#     'value': [[1.0, 2.0], [3.0, 4.0]],
#     'type': 'ndarray',
#     'dtype': 'float64',
#     'shape': [2, 2]
# }
```

Complex arrays are handled specially to ensure YAML compatibility.

#### `PandasSerializer`

Handles `pandas.DataFrame` and `pandas.Series` types. Preserves columns, index, and index name.

```python
import pandas as pd
from pytest_context_assert.serializers import PandasSerializer

serializer = PandasSerializer()

# DataFrame
df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
serializer.serialize(df)
# {
#     'value': {'a': [1, 2], 'b': [3, 4]},
#     'type': 'pandas.DataFrame',
#     'columns': ['a', 'b'],
#     'index': [0, 1],
#     'index_name': None
# }

# Series
s = pd.Series([1, 2, 3], name="values")
serializer.serialize(s)
# {
#     'value': [1, 2, 3],
#     'type': 'pandas.Series',
#     'name': 'values',
#     'index': [0, 1, 2],
#     'index_name': None
# }
```

#### `PolarsSerializer`

Handles `polars.DataFrame` and `polars.Series` types. Preserves columns, schema, and dtypes.

```python
import polars as pl
from pytest_context_assert.serializers import PolarsSerializer

serializer = PolarsSerializer()

# DataFrame
df = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
serializer.serialize(df)
# {
#     'value': {'a': [1, 2], 'b': [3.0, 4.0]},
#     'type': 'polars.DataFrame',
#     'columns': ['a', 'b'],
#     'schema': {'a': 'Int64', 'b': 'Float64'}
# }

# Series
s = pl.Series("values", [1, 2, 3])
serializer.serialize(s)
# {
#     'value': [1, 2, 3],
#     'type': 'polars.Series',
#     'name': 'values',
#     'dtype': 'Int64'
# }
```

#### `GenericSerializer`

Fallback serializer for other types. Attempts serialization in this order:

1. If object has `to_dict()` method, uses it
2. If object has `__dict__` attribute, uses it
3. Falls back to `str()` representation

### Helper Functions

```python
from pytest_context_assert.serializers import serialize_value, deserialize_value, get_serializer

# Get appropriate serializer for a value
serializer = get_serializer([1, 2, 3])  # Returns ListSerializer

# Serialize any value (auto-detects type)
data = serialize_value([1, 2, 3])
# {'value': [...], 'type': 'list'}

# Deserialize back
value = deserialize_value(data)
# [1, 2, 3]

# Works with pandas/polars automatically
import pandas as pd
df = pd.DataFrame({"a": [1, 2, 3]})
data = serialize_value(df)  # Uses PandasSerializer
df_restored = deserialize_value(data)  # Returns DataFrame
```

### Serializer Priority

When `get_serializer()` is called, serializers are tried in this order:

1. `ScalarSerializer` - int, float, str, bool, None, numpy scalars
2. `NumpySerializer` - numpy.ndarray
3. `PandasSerializer` - pandas.DataFrame, pandas.Series
4. `PolarsSerializer` - polars.DataFrame, polars.Series
5. `ListSerializer` - list, tuple
6. `DictSerializer` - dict
7. `GenericSerializer` - fallback for all other types

---

## Exceptions

### `ContextAssertionError`

Raised when an assertion fails.

**Attributes:**
- `actual`: The actual value
- `expected`: The expected value
- `context_key`: The context key that failed
- `diff`: Difference description (if available)

### `MissingSnapshotError`

Raised when no snapshot exists and update mode is not enabled.

**Attributes:**
- `context_key`: The context key that was looked up
- `snapshot_path`: Path where the snapshot would be stored

---

## pytest Hooks

### `pytest_context_assert_get_context() -> dict | None`

Hook to add custom context keys. Implement in `conftest.py`:

```python
def pytest_context_assert_get_context():
    return {"custom_key": "custom_value"}
```

---

## Markers

### `@pytest.mark.context_assert_context(**kwargs)`

Override context for specific tests.

```python
@pytest.mark.context_assert_context(gpu="nvidia")
def test_gpu(context_assert):
    context_assert(result)
```
