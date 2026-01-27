# pytest-context-assert

A pytest plugin for context-aware assertions with snapshot-style storage. Store expected values in YAML files that can vary based on platform, architecture, BLAS library, or any custom context you define.

## Features

- **Context-aware snapshots**: Different expected values for different platforms, architectures, or custom contexts
- **YAML storage**: Human-readable snapshot files stored alongside your tests
- **Tolerance support**: Compare floating-point values with `rtol` and `atol`
- **NumPy integration**: Native support for numpy arrays with shape and dtype preservation
- **Pandas support**: Automatic serialization and comparison of DataFrames and Series
- **Polars support**: Automatic serialization and comparison of Polars DataFrames and Series
- **Special float handling**: Proper serialization of `inf`, `-inf`, and `nan` values
- **Custom serialization**: Define how your custom objects are stored and compared
- **Syrupy-style API**: Familiar `assert_match` interface
- **Dynamic context**: Use environment variables and runtime detection in context specification

## Installation

```bash
pip install pytest-context-assert
```

For NumPy support:

```bash
pip install pytest-context-assert[numpy]
```

For Pandas support:

```bash
pip install pytest-context-assert[pandas]
```

For Polars support:

```bash
pip install pytest-context-assert[polars]
```

For all optional dependencies:

```bash
pip install pytest-context-assert[all]
```

## Quick Start

```python
def test_calculation(context_assert):
    result = my_function()
    context_assert(result)  # Stores/compares snapshot based on current context
```

Run with `--context-assert-update` to create or update snapshots:

```bash
pytest --context-assert-update
```

## Basic Usage

### Simple Assertions

```python
def test_basic(context_assert):
    # Scalar values
    context_assert(42, name="answer")

    # Strings
    context_assert("hello world", name="greeting")

    # Lists and dicts
    context_assert([1, 2, 3], name="numbers")
    context_assert({"key": "value"}, name="config")
```

### Multiple Assertions Per Test

```python
def test_multi_step(context_assert):
    # Auto-named assertions
    context_assert(step1_result)  # assertion_1
    context_assert(step2_result)  # assertion_2
    context_assert(step3_result)  # assertion_3

    # Or use explicit names
    context_assert(step1_result, name="after_step1")
    context_assert(step2_result, name="after_step2")
```

### Floating-Point Tolerance

```python
def test_numerical(context_assert):
    result = compute_pi()
    context_assert(result, rtol=1e-10, atol=1e-15, name="pi_value")
```

### Syrupy-Style API

```python
def test_syrupy_style(context_assert):
    result = my_function()
    context_assert.assert_match(result, name="result")
```

## Context-Aware Testing

### Understanding Context

The plugin automatically detects:
- **Platform**: `linux`, `darwin`, `windows`
- **Architecture**: `x86_64`, `arm64`
- **BLAS library**: `mkl`, `openblas`, `accelerate`

Snapshot files store values per context:

```yaml
_metadata:
  version: 2
  test_name: test_matrix_multiply
assertions:
  result:
  - __context__:
      platform: darwin
      arch: arm64
      blas: accelerate
    value: 1.0000001
    type: float
  - __context__:
      platform: linux
      arch: x86_64
      blas: mkl
    value: 1.0000002
    type: float
  - __context__:
      default: true
    value: 1.0
    type: float
```

### Custom Context with Decorator

```python
from pytest_context_assert import set_context

@set_context({"openblas_target": "haswell", "num_threads": "4"})
def test_with_custom_context(context_assert):
    result = matrix_operation()
    context_assert(result, name="matrix_result")
```

### Dynamic Context from Environment

```python
from pytest_context_assert import (
    set_context,
    env,
    get_platform,
    get_arch,
    get_openblas_coretype,
    build_context,
)

# Using environment variables
@set_context({
    "openblas_coretype": get_openblas_coretype(),  # From OPENBLAS_CORETYPE
    "num_threads": env("OMP_NUM_THREADS", "1"),
})
def test_env_context(context_assert):
    ...

# Using build_context helper
@set_context(build_context(
    include_platform=True,
    include_arch=True,
    include_blas=True,
    env_vars=["OPENBLAS_CORETYPE", "MKL_NUM_THREADS"],
    custom_key="my_value",
))
def test_comprehensive_context(context_assert):
    ...

# Runtime context with with_context
def test_runtime_context(context_assert):
    ctx = context_assert.with_context(
        runtime_var=os.environ.get("MY_VAR", "default")
    )
    ctx(result, name="dynamic_result")
```

### Available Context Helpers

```python
from pytest_context_assert import (
    # Environment variable helpers
    env,                    # Get env var with default
    env_or_skip,           # Get env var or "__SKIP__"
    context_from_env,      # Build dict from multiple env vars

    # Platform/system helpers
    get_platform,          # "linux", "darwin", "windows"
    get_arch,              # "x86_64", "arm64"
    get_python_version,    # "3.10", "3.11"

    # BLAS/NumPy helpers
    get_openblas_coretype, # From OPENBLAS_CORETYPE env var
    get_blas_num_threads,  # From *_NUM_THREADS env vars
    detect_numpy_blas,     # Detect numpy's BLAS library

    # CI/Environment helpers
    get_ci_platform,       # "github", "gitlab", etc.
    get_conda_env,         # Current conda env name

    # Builder helper
    build_context,         # Build comprehensive context dict
)
```

## NumPy Integration

### Basic Array Assertions

```python
def test_numpy_arrays(context_assert):
    import numpy as np

    # 1D array
    arr = np.array([1.0, 2.0, 3.0])
    context_assert(arr, name="1d_array")

    # 2D array
    matrix = np.array([[1, 2], [3, 4]])
    context_assert(matrix, name="matrix")

    # With tolerance for floating-point
    result = np.linalg.eigvals(matrix)
    context_assert(result, rtol=1e-10, name="eigenvalues")
```

### Linear Algebra Results

```python
def test_linear_algebra(context_assert):
    import numpy as np

    A = np.array([[1.0, 2.0], [3.0, 4.0]])

    # Matrix operations
    context_assert(np.linalg.inv(A), rtol=1e-10, name="inverse")
    context_assert(np.linalg.det(A), rtol=1e-10, name="determinant")

    # Decompositions
    U, S, Vh = np.linalg.svd(A)
    context_assert(S, rtol=1e-10, name="singular_values")
```

### Complex Arrays

```python
def test_complex_arrays(context_assert):
    import numpy as np

    arr = np.array([1+2j, 3+4j, 5+6j])
    context_assert(arr, name="complex_array")
```

### Platform-Specific Numerical Results

```python
from pytest_context_assert import set_context, get_openblas_coretype

@set_context({"openblas_target": get_openblas_coretype()})
def test_matrix_multiply(context_assert):
    """Test that might have different results on different CPU architectures."""
    import numpy as np

    np.random.seed(42)
    A = np.random.rand(100, 100)
    B = np.random.rand(100, 100)
    result = A @ B

    context_assert(result, rtol=1e-10, name="matmul_result")
```

## Pandas Integration

Pandas DataFrames and Series are automatically serialized and deserialized. The plugin preserves column names, index values, and index names.

### DataFrame Assertions

```python
def test_dataframe(context_assert):
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    context_assert(df, name="dataframe")
```

The snapshot stores the DataFrame with full metadata:

```yaml
assertions:
  dataframe:
  - __context__:
      platform: darwin
      arch: arm64
    value:
      a: [1, 2, 3]
      b: [4.0, 5.0, 6.0]
    type: pandas.DataFrame
    columns: [a, b]
    index: [0, 1, 2]
    index_name: null
```

### Series Assertions

```python
def test_series(context_assert):
    import pandas as pd

    s = pd.Series([1, 2, 3, 4, 5], name="values")
    context_assert(s, name="series")
```

### DataFrames with Custom Index

```python
def test_dataframe_with_index(context_assert):
    import pandas as pd

    df = pd.DataFrame(
        {"a": [1, 2, 3]},
        index=pd.Index(["x", "y", "z"], name="letter")
    )
    context_assert(df, name="indexed_df")
```

### Aggregation Results

```python
def test_groupby(context_assert):
    import pandas as pd

    df = pd.DataFrame({
        "category": ["A", "B", "A", "B"],
        "value": [10, 20, 30, 40],
    })
    result = df.groupby("category")["value"].sum()
    context_assert(result, name="groupby_sum")
```

### Custom Comparison for DataFrames

```python
def test_dataframe_custom_compare(context_assert):
    import pandas as pd

    df = pd.DataFrame({"a": [1.0001, 2.0001], "b": [3.0001, 4.0001]})

    # Use custom comparison with tolerance
    def compare_df(actual, expected):
        return actual.round(2).equals(expected.round(2))

    context_assert(df, name="df", compare=compare_df)
```

## Polars Integration

Polars DataFrames and Series are automatically serialized and deserialized. The plugin preserves column names, schema information, and data types.

### DataFrame Assertions

```python
def test_polars_dataframe(context_assert):
    import polars as pl

    df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    context_assert(df, name="dataframe")
```

The snapshot stores the DataFrame with schema information:

```yaml
assertions:
  dataframe:
  - __context__:
      platform: darwin
      arch: arm64
    value:
      a: [1, 2, 3]
      b: [4.0, 5.0, 6.0]
    type: polars.DataFrame
    columns: [a, b]
    schema:
      a: Int64
      b: Float64
```

### Series Assertions

```python
def test_polars_series(context_assert):
    import polars as pl

    s = pl.Series("values", [1, 2, 3, 4, 5])
    context_assert(s, name="series")
```

### DataFrame Operations

```python
def test_polars_operations(context_assert):
    import polars as pl

    df = pl.DataFrame({
        "category": ["A", "B", "A", "B"],
        "value": [10, 20, 30, 40],
    })

    # Group by and aggregate
    result = df.group_by("category").agg(pl.col("value").sum())
    context_assert(result.sort("category"), name="groupby_sum")

    # Filter operations
    filtered = df.filter(pl.col("value") > 15)
    context_assert(filtered, name="filtered")
```

### LazyFrame Operations

```python
def test_polars_lazy(context_assert):
    import polars as pl

    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    # LazyFrame operations - collect before asserting
    result = (
        df.lazy()
        .filter(pl.col("a") > 1)
        .select(pl.col("b") * 2)
        .collect()
    )
    context_assert(result, name="lazy_result")
```

## Custom Objects

### Dataclasses

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

def test_dataclass(context_assert):
    point = Point(3.0, 4.0)

    def serialize(p):
        return {"x": p.x, "y": p.y, "type": "Point"}

    def deserialize(data):
        return Point(data["x"], data["y"])

    context_assert(
        point,
        name="point",
        serialize=serialize,
        deserialize=deserialize,
        compare=lambda a, b: a == b,
    )
```

### Custom Comparison

```python
def test_custom_compare(context_assert):
    # Case-insensitive string comparison
    context_assert(
        "HELLO WORLD",
        name="greeting",
        compare=lambda a, b: a.lower() == b.lower(),
    )

    # Partial dict comparison (only check specific keys)
    context_assert(
        {"important": 1, "timestamp": "varies"},
        name="partial",
        compare=lambda a, b: a.get("important") == b.get("important"),
    )
```

### API Response Objects

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class APIResponse:
    status: int
    data: dict
    timestamp: datetime | None = None

def test_api_response(context_assert):
    response = APIResponse(status=200, data={"users": []})

    def serialize(r):
        return {
            "status": r.status,
            "data": r.data,
            "type": "APIResponse",
        }

    context_assert(
        response,
        name="api_response",
        serialize=serialize,
        compare=lambda a, b: a.status == b["status"] and a.data == b["data"],
    )
```

### Datetime Objects

```python
from datetime import datetime, date, timedelta

def test_datetime(context_assert):
    dt = datetime(2026, 1, 20, 15, 30, 45)

    def serialize(d):
        return {"iso": d.isoformat(), "type": "datetime"}

    def deserialize(data):
        return datetime.fromisoformat(data["iso"])

    context_assert(
        dt,
        name="timestamp",
        serialize=serialize,
        deserialize=deserialize,
        compare=lambda a, b: a == b,
    )
```

### Enums

```python
from enum import Enum

class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"

def test_enum(context_assert):
    color = Color.RED

    def serialize(c):
        return {"value": c.value, "name": c.name, "type": "Color"}

    def deserialize(data):
        return Color(data["value"])

    context_assert(
        color,
        name="color",
        serialize=serialize,
        deserialize=deserialize,
        compare=lambda a, b: a == b,
    )
```

## CLI Options

```bash
# Update snapshots for current context
pytest --context-assert-update

# Update all contexts (use with caution)
pytest --context-assert-update-all

# Override context detection
pytest --context-assert-context=linux-x86_64-mkl

# Custom snapshot directory
pytest --context-assert-dir=my_snapshots
```

## Configuration (pyproject.toml)

```toml
[tool.pytest.ini_options]
context_assert_dir = "__snapshots__"
context_assert_default_rtol = "1e-7"
context_assert_default_atol = "0"
```

## Snapshot File Structure

Snapshots are stored in `__snapshots__/<test_file>/<test_name>.yaml`:

```
tests/
  test_math.py
  __snapshots__/
    test_math/
      test_addition.yaml
      test_multiplication.yaml
```

## Advanced: Custom Hooks

Add custom context in `conftest.py`:

```python
def pytest_context_assert_get_context():
    """Return custom context to merge with detected context."""
    return {
        "cuda": os.environ.get("CUDA_VERSION"),
        "gpu": "nvidia" if has_nvidia_gpu() else "none",
    }
```

## API Reference

### `context_assert` Fixture

The main fixture for context-aware assertions.

#### `__call__(value, *, rtol=None, atol=None, name=None, compare=None, serialize=None, deserialize=None)`

Assert that a value matches the stored snapshot.

**Parameters:**
- `value`: The actual value to compare
- `rtol` (float, optional): Relative tolerance for float comparisons
- `atol` (float, optional): Absolute tolerance for float comparisons
- `name` (str, optional): Name for this assertion
- `compare` (callable, optional): Custom comparison function `(actual, expected) -> bool`
- `serialize` (callable, optional): Custom serialization function `(value) -> dict`
- `deserialize` (callable, optional): Custom deserialization function `(dict) -> value`

#### `assert_match(value, *, name=None, **kwargs)`

Syrupy-style alias for `__call__`.

#### `with_context(**kwargs)`

Create a new context_assert with additional custom context.

#### `update(value, *, name=None, context=None)`

Explicitly update a snapshot value.

#### `set_default(value, *, name=None)`

Set the default value for this assertion.

#### `context` (property)

Get the current context dictionary.

#### `context_key` (property)

Get the current context key string.

### `@set_context` Decorator

```python
from pytest_context_assert import set_context

# With dictionary
@set_context({"key": "value"})
def test_func(context_assert):
    ...

# With kwargs
@set_context(key="value")
def test_func(context_assert):
    ...
```

## Examples

### Testing Numerical Libraries

```python
from pytest_context_assert import set_context, get_openblas_coretype

@set_context({"openblas_target": get_openblas_coretype()})
def test_matrix_multiply(context_assert):
    import numpy as np

    np.random.seed(42)
    A = np.random.rand(100, 100)
    B = np.random.rand(100, 100)
    result = A @ B

    context_assert(result, rtol=1e-10, name="matmul_result")
```

### Testing ML Model Outputs

```python
@set_context({"model_version": "1.0", "backend": "tensorflow"})
def test_model_prediction(context_assert):
    model = load_model()
    input_data = prepare_input()
    prediction = model.predict(input_data)

    context_assert(prediction.tolist(), rtol=1e-5, name="prediction")
```

### Testing Data Transformations

```python
def test_etl_pipeline(context_assert):
    import pandas as pd

    # Test each stage of the pipeline
    raw = load_raw_data()
    context_assert(raw.shape[0], name="raw_row_count")

    cleaned = clean_data(raw)
    context_assert(cleaned.isnull().sum().to_dict(), name="null_counts")

    transformed = transform_data(cleaned)

    def serialize_stats(df):
        return {"data": df.describe().to_dict(), "type": "stats"}

    context_assert(transformed, name="transform_stats", serialize=serialize_stats)
```

### CI/CD Integration

```python
from pytest_context_assert import set_context, get_ci_platform, get_platform

@set_context({
    "ci": get_ci_platform() or "local",
    "os": get_platform(),
})
def test_ci_aware(context_assert):
    """Test with CI-aware context for debugging failures."""
    result = compute_something()
    context_assert(result, name="ci_result")
```

### Cross-Platform Testing

```python
from pytest_context_assert import set_context, build_context

@set_context(build_context(
    include_platform=True,
    include_arch=True,
    include_blas=True,
))
def test_cross_platform(context_assert):
    """Results stored separately per platform/arch/blas combination."""
    import numpy as np

    result = np.linalg.eigvals(np.random.rand(10, 10))
    context_assert(result, rtol=1e-10, name="eigenvalues")
```

## Best Practices

1. **Use named assertions** when you have multiple assertions in one test
2. **Set appropriate tolerances** for floating-point comparisons
3. **Use `default` context** for values that don't vary across platforms
4. **Review snapshot diffs** before committing updated snapshots
5. **Keep snapshots in version control** alongside your tests
6. **Use custom serialization** for complex objects to control what's stored
7. **Use `@set_context`** for tests with known platform-specific behavior

## License

MIT
