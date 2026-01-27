# Examples

## Basic Usage

### Simple Value Assertion

```python
def test_simple_calculation(context_assert):
    result = 2 + 2
    context_assert(result)
```

Run with `--context-assert-update` to create the snapshot, then run normally to compare.

### Multiple Assertions in One Test

```python
def test_multiple_values(context_assert):
    # Names are auto-generated based on call order
    context_assert(compute_a())  # "assertion_1"
    context_assert(compute_b())  # "assertion_2"
    context_assert(compute_c())  # "assertion_3"

    # Or use explicit names for clarity
    context_assert(compute_d(), name="result_d")
```

---

## Numeric Comparisons

### Float with Tolerance

```python
def test_float_precision(context_assert):
    # Platform-specific floating point results
    result = math.sin(math.pi / 4)

    # Allow small differences
    context_assert(result, rtol=1e-10, atol=1e-15)
```

### NumPy Arrays

```python
import numpy as np

def test_array_computation(context_assert):
    arr = np.linspace(0, 1, 100)
    result = np.sin(arr)

    # Compare with tolerance
    context_assert(result, rtol=1e-7, name="sin_values")

def test_matrix_operations(context_assert):
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    result = A @ B

    context_assert(result, name="matrix_product")
```

### Special Float Values

```python
import math

def test_special_floats(context_assert):
    # Infinity values are properly serialized and compared
    context_assert(math.inf, name="positive_infinity")
    context_assert(-math.inf, name="negative_infinity")
    context_assert(math.nan, name="nan_value")

def test_list_with_special_floats(context_assert):
    # Works in collections too
    values = [1.0, math.inf, -math.inf, math.nan]
    context_assert(values, name="mixed_floats")
```

---

## Pandas DataFrames

### Basic DataFrame Assertions

```python
import pandas as pd

def test_dataframe(context_assert):
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "score": [85.5, 90.0, 78.5]
    })
    context_assert(df, name="users")

def test_series(context_assert):
    s = pd.Series([1, 2, 3, 4, 5], name="values")
    context_assert(s, name="numbers")
```

### DataFrames with Custom Index

```python
import pandas as pd

def test_custom_index(context_assert):
    df = pd.DataFrame(
        {"value": [100, 200, 300]},
        index=pd.Index(["a", "b", "c"], name="key")
    )
    context_assert(df, name="indexed_df")

def test_datetime_index(context_assert):
    dates = pd.date_range("2024-01-01", periods=3)
    df = pd.DataFrame(
        {"value": [1, 2, 3]},
        index=dates
    )
    # Convert datetime index to string for serialization
    def serialize(df):
        return {
            "value": df.to_dict(orient="list"),
            "type": "dataframe",
            "index": df.index.strftime("%Y-%m-%d").tolist()
        }
    context_assert(df, name="time_series", serialize=serialize)
```

### Aggregation Results

```python
import pandas as pd

def test_groupby(context_assert):
    df = pd.DataFrame({
        "category": ["A", "B", "A", "B", "A"],
        "value": [10, 20, 30, 40, 50]
    })

    # Group by sum
    result = df.groupby("category")["value"].sum()
    context_assert(result, name="sum_by_category")

    # Multiple aggregations
    agg = df.groupby("category").agg({"value": ["mean", "std"]})
    context_assert(agg, name="stats_by_category", rtol=1e-10)
```

---

## Polars DataFrames

### Basic Polars Assertions

```python
import polars as pl

def test_polars_dataframe(context_assert):
    df = pl.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "score": [85.5, 90.0, 78.5]
    })
    context_assert(df, name="users")

def test_polars_series(context_assert):
    s = pl.Series("values", [1, 2, 3, 4, 5])
    context_assert(s, name="numbers")
```

### Polars Operations

```python
import polars as pl

def test_polars_filter(context_assert):
    df = pl.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "score": [85, 70, 92]
    })

    # Filter passing scores
    passing = df.filter(pl.col("score") >= 75)
    context_assert(passing, name="passing_students")

def test_polars_groupby(context_assert):
    df = pl.DataFrame({
        "category": ["A", "B", "A", "B", "A"],
        "value": [10, 20, 30, 40, 50]
    })

    result = df.group_by("category").agg(
        pl.col("value").sum().alias("total"),
        pl.col("value").mean().alias("average")
    ).sort("category")

    context_assert(result, name="category_stats")
```

### LazyFrame Operations

```python
import polars as pl

def test_lazy_execution(context_assert):
    df = pl.DataFrame({
        "a": range(1000),
        "b": range(1000, 2000)
    })

    # Build lazy query
    result = (
        df.lazy()
        .filter(pl.col("a") > 500)
        .select([
            pl.col("a"),
            (pl.col("b") * 2).alias("b_doubled")
        ])
        .collect()
    )

    context_assert(result, name="lazy_result")
```

### Polars with Tolerance

```python
import polars as pl

def test_polars_floating_point(context_assert):
    df = pl.DataFrame({
        "x": [1.0, 2.0, 3.0],
        "y": [0.1, 0.2, 0.3]
    })

    # Operations that may have floating-point differences
    result = df.select(
        (pl.col("x") / pl.col("y")).alias("ratio")
    )

    # Use custom comparison with tolerance
    def compare_polars(actual, expected):
        if actual.shape != expected.shape:
            return False
        for col in actual.columns:
            a_col = actual[col].to_list()
            e_col = expected[col].to_list()
            for a, e in zip(a_col, e_col):
                if abs(a - e) > 1e-10:
                    return False
        return True

    context_assert(result, name="ratios", compare=compare_polars)
```

---

## Custom Comparison Functions

### Case-Insensitive String Comparison

```python
def test_case_insensitive(context_assert):
    result = get_message().upper()

    context_assert(
        result,
        name="message",
        compare=lambda a, b: a.lower() == b.lower()
    )
```

### Percentage Tolerance

```python
def test_within_percentage(context_assert):
    result = benchmark_function()

    def within_5_percent(actual, expected):
        if expected == 0:
            return actual == 0
        return abs(actual - expected) / abs(expected) <= 0.05

    context_assert(result, name="benchmark", compare=within_5_percent)
```

### Comparing Complex Objects

```python
from dataclasses import dataclass

@dataclass
class Result:
    value: float
    confidence: float
    label: str

def test_ml_result(context_assert):
    result = run_model()

    def compare_results(actual, expected):
        # Compare value and confidence within tolerance
        value_ok = abs(actual.value - expected.value) < 0.01
        conf_ok = abs(actual.confidence - expected.confidence) < 0.05
        # Label must match exactly
        label_ok = actual.label == expected.label
        return value_ok and conf_ok and label_ok

    context_assert(result, name="model_output", compare=compare_results)
```

### Ignoring Certain Fields

```python
def test_ignore_timestamp(context_assert):
    result = {"data": [1, 2, 3], "timestamp": time.time()}

    def compare_without_timestamp(actual, expected):
        a = {k: v for k, v in actual.items() if k != "timestamp"}
        b = {k: v for k, v in expected.items() if k != "timestamp"}
        return a == b

    context_assert(result, name="data", compare=compare_without_timestamp)
```

---

## Platform-Specific Tests

### Different Expected Values per Platform

When you run `--context-assert-update` on different platforms, each platform gets its own expected value in the snapshot:

```yaml
# __snapshots__/test_module/test_platform_specific.yaml
contexts:
  darwin-arm64-accelerate:
    value: 1.0000001
    type: float
  linux-x86_64-mkl:
    value: 1.0000002
    type: float
  windows-x86_64:
    value: 1.0
    type: float
  default:
    value: 1.0
    type: float
```

### Manually Setting Platform-Specific Values

```python
def test_with_default(context_assert):
    # Set a default that works on most platforms
    context_assert.set_default(1.0, name="result")

    # The actual comparison will use the platform-specific
    # value if available, otherwise the default
    context_assert(compute(), name="result")
```

---

## Context Customization

### Adding Custom Context

```python
def test_gpu_context(context_assert):
    # Add GPU information to context
    custom = context_assert.with_context(
        gpu="nvidia",
        cuda_version="11.8"
    )

    result = gpu_compute()
    custom(result, name="gpu_result")
```

### Using Markers

```python
import pytest

@pytest.mark.context_assert_context(backend="tensorflow")
def test_tf_model(context_assert):
    result = run_tf_model()
    context_assert(result)

@pytest.mark.context_assert_context(backend="pytorch")
def test_pytorch_model(context_assert):
    result = run_pytorch_model()
    context_assert(result)
```

### Global Custom Context via Hook

In `conftest.py`:

```python
def pytest_context_assert_get_context():
    """Add environment-specific context."""
    import os

    context = {}

    # Add CUDA version if available
    cuda = os.environ.get("CUDA_VERSION")
    if cuda:
        context["cuda"] = cuda

    # Add custom environment marker
    if os.environ.get("CI"):
        context["env"] = "ci"

    return context
```

---

## Working with Snapshots

### Updating Snapshots

```bash
# Update all snapshots for current context
pytest --context-assert-update

# Update specific test
pytest tests/test_math.py::test_calculation --context-assert-update
```

### Using a Different Snapshot Directory

```bash
pytest --context-assert-dir=golden_files
```

Or configure in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
context_assert_dir = "golden_files"
```

### Overriding Context Detection

```bash
# Test as if running on Linux
pytest --context-assert-context=linux-x86_64-mkl
```

---

## Integration Patterns

### With pytest-xdist (Parallel Testing)

The plugin works with parallel test execution:

```bash
pytest -n auto --context-assert-update
```

### With CI/CD

Example GitHub Actions workflow:

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python: ['3.10', '3.11', '3.12']

    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}

      - name: Install dependencies
        run: pip install -e .[dev]

      - name: Run tests
        run: pytest

      # Optionally update snapshots on main branch
      - name: Update snapshots
        if: github.ref == 'refs/heads/main'
        run: pytest --context-assert-update
```

### Snapshot Review in PRs

Add snapshots to version control and review changes in pull requests:

```bash
git add tests/__snapshots__/
git commit -m "Update test snapshots"
```

---

## Error Handling

### Missing Snapshot Error

```python
def test_new_feature(context_assert):
    result = new_feature()
    # First run without --context-assert-update will raise:
    # MissingSnapshotError: No snapshot found for context 'darwin-arm64'.
    # Run with --context-assert-update to create it.
    context_assert(result)
```

### Assertion Failure

```python
def test_changed_value(context_assert):
    # If the computed value differs from snapshot:
    # ContextAssertionError: Assertion failed for context 'darwin-arm64':
    # Float comparison failed: 2.0 != 1.0 (diff=1.0, rtol=1e-07, atol=0)
    context_assert(compute())
```

### Handling Exceptions in Custom Compare

```python
def test_safe_compare(context_assert):
    def safe_compare(actual, expected):
        try:
            return actual.matches(expected)
        except AttributeError:
            return actual == expected

    context_assert(result, compare=safe_compare)
```

---

## Advanced Patterns

### Parametrized Tests

```python
import pytest

@pytest.mark.parametrize("input,expected_name", [
    (1, "one"),
    (2, "two"),
    (3, "three"),
])
def test_parametrized(context_assert, input, expected_name):
    result = process(input)
    # Each parameter combination gets its own snapshot file
    context_assert(result)
```

### Conditional Assertions

```python
import sys

def test_platform_specific(context_assert):
    result = compute()

    if sys.platform == "darwin":
        # macOS might have slightly different results
        context_assert(result, rtol=1e-5)
    else:
        context_assert(result, rtol=1e-7)
```

### Combining with Other Assertions

```python
def test_combined(context_assert):
    result = compute()

    # Regular assertions for invariants
    assert result > 0
    assert result < 100

    # Context-aware assertion for exact value
    context_assert(result, name="exact_value")
```
