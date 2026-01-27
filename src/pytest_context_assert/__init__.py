"""pytest-context-assert: Context-aware assertions with snapshot-style storage.

A pytest plugin for context-aware assertions with snapshot-style storage.
Expected values are stored in YAML files and can vary based on platform,
architecture, BLAS library, or custom user-defined contexts.

Features:
    - Context-aware snapshots for different platforms/architectures
    - YAML storage for human-readable snapshot files
    - Tolerance support for floating-point comparisons (rtol/atol)
    - Native support for NumPy arrays, pandas DataFrames/Series, and Polars DataFrames/Series
    - Special float handling (inf, -inf, nan)
    - Custom serialization and comparison functions
    - Syrupy-style API with assert_match

Example:
    >>> def test_calculation(context_assert):
    ...     result = my_function()
    ...     context_assert(result, name="result", rtol=1e-7)

See Also:
    - README.md for full documentation
    - docs/api.md for API reference
    - docs/examples.md for usage examples
"""

from pytest_context_assert.comparators import (
    Comparator,
    FunctionComparator,
    GenericComparator,
    NumpyArrayComparator,
    ScalarComparator,
    get_comparator,
)
from pytest_context_assert.context import ContextResolver
from pytest_context_assert.fixture import ContextAssert
from pytest_context_assert.helpers import (
    build_context,
    context_from_env,
    detect_numpy_blas,
    env,
    env_or_skip,
    get_arch,
    get_blas_num_threads,
    get_ci_platform,
    get_conda_env,
    get_mkl_verbose,
    get_omp_num_threads,
    get_openblas_coretype,
    get_platform,
    get_python_version,
)
from pytest_context_assert.plugin import set_context
from pytest_context_assert.serializers import (
    DictSerializer,
    GenericSerializer,
    ListSerializer,
    NumpySerializer,
    PandasSerializer,
    PolarsSerializer,
    ScalarSerializer,
    Serializer,
    deserialize_value,
    get_serializer,
    serialize_value,
)
from pytest_context_assert.storage import SnapshotStorage

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Decorator
    "set_context",
    # Helper functions for dynamic context
    "env",
    "env_or_skip",
    "get_platform",
    "get_arch",
    "get_python_version",
    "get_openblas_coretype",
    "get_mkl_verbose",
    "get_omp_num_threads",
    "get_blas_num_threads",
    "get_conda_env",
    "get_ci_platform",
    "context_from_env",
    "detect_numpy_blas",
    "build_context",
    # Core classes
    "ContextResolver",
    "ContextAssert",
    # Comparators
    "Comparator",
    "ScalarComparator",
    "NumpyArrayComparator",
    "GenericComparator",
    "FunctionComparator",
    "get_comparator",
    # Serializers
    "Serializer",
    "ScalarSerializer",
    "ListSerializer",
    "DictSerializer",
    "NumpySerializer",
    "PandasSerializer",
    "PolarsSerializer",
    "GenericSerializer",
    "get_serializer",
    "serialize_value",
    "deserialize_value",
    # Storage
    "SnapshotStorage",
]
