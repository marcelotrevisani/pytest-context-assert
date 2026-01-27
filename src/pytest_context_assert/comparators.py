"""Comparison logic for different value types.

This module provides comparators that determine whether two values match
for snapshot testing purposes. Each comparator handles specific types
and supports tolerance-based comparison for floating-point values.

Built-in Comparators:
    ScalarComparator: Handles int, float, str, bool, None.
        Float comparison uses: abs(actual - expected) <= atol + rtol * abs(expected)
        Special floats (inf, -inf, nan) are compared by identity.
    NumpyArrayComparator: Handles numpy.ndarray using np.allclose() for floats.
    ListComparator: Handles list and tuple with element-wise recursive comparison.
    DictComparator: Handles dict with key matching and recursive value comparison.
    GenericComparator: Fallback using == operator.
    FunctionComparator: Uses a user-provided comparison function.

Special Float Handling:
    For snapshot testing, the following special float comparisons are used:
    - inf == inf → True
    - -inf == -inf → True
    - nan == nan → True (unlike standard Python where nan != nan)
    - inf != -inf → True

Usage:
    >>> from pytest_context_assert.comparators import get_comparator
    >>> comparator = get_comparator(3.14)
    >>> result = comparator.compare(3.14, 3.14, rtol=1e-7, atol=0)
    >>> assert result.equal
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

CompareFunc = Callable[[Any, Any], bool]


@dataclass
class ComparisonResult:
    """Result of a comparison operation."""

    equal: bool
    message: str = ""
    actual: Any = None
    expected: Any = None
    diff: str | None = None

    def __bool__(self) -> bool:
        return self.equal


class Comparator(ABC):
    """Abstract base class for value comparators."""

    @abstractmethod
    def can_compare(self, value: Any) -> bool:
        """Check if this comparator can handle the given value type."""
        ...

    @abstractmethod
    def compare(
        self,
        actual: Any,
        expected: Any,
        *,
        rtol: float = 1e-7,
        atol: float = 0,
    ) -> ComparisonResult:
        """Compare actual value against expected value.

        Args:
            actual: The actual value from the test.
            expected: The expected value from the snapshot.
            rtol: Relative tolerance for floating point comparisons.
            atol: Absolute tolerance for floating point comparisons.

        Returns:
            ComparisonResult indicating whether values match.
        """
        ...


class ScalarComparator(Comparator):
    """Comparator for scalar types (int, float, str, bool, None)."""

    SCALAR_TYPES = (int, float, str, bool, type(None))

    def can_compare(self, value: Any) -> bool:
        return isinstance(value, self.SCALAR_TYPES)

    def _compare_special_floats(self, actual: float, expected: float) -> bool | None:
        """Compare special float values (inf, -inf, nan).

        Returns True if both are the same special value, False if one is special
        and the other isn't (or different special values), None if neither is special.
        """
        import math

        actual_is_inf = math.isinf(actual)
        expected_is_inf = math.isinf(expected)
        actual_is_nan = math.isnan(actual)
        expected_is_nan = math.isnan(expected)

        # Both are NaN - consider equal for snapshot testing
        if actual_is_nan and expected_is_nan:
            return True

        # One is NaN and the other isn't
        if actual_is_nan or expected_is_nan:
            return False

        # Both are infinity - check if same sign
        if actual_is_inf and expected_is_inf:
            return (actual > 0) == (expected > 0)

        # One is infinity and the other isn't
        if actual_is_inf or expected_is_inf:
            return False

        # Neither is special
        return None

    def compare(
        self,
        actual: Any,
        expected: Any,
        *,
        rtol: float = 1e-7,
        atol: float = 0,
    ) -> ComparisonResult:
        if isinstance(actual, float) and isinstance(expected, (int, float)):
            expected_float = float(expected)

            # Handle special float values first
            special_result = self._compare_special_floats(actual, expected_float)
            if special_result is not None:
                if special_result:
                    return ComparisonResult(equal=True)
                else:
                    return ComparisonResult(
                        equal=False,
                        message=f"Special float comparison failed: {actual} != {expected}",
                        actual=actual,
                        expected=expected,
                    )

            # Regular float comparison
            if abs(expected_float) == 0:
                equal = abs(actual) <= atol
            else:
                equal = abs(actual - expected_float) <= atol + rtol * abs(expected_float)

            if equal:
                return ComparisonResult(equal=True)
            else:
                diff = actual - expected_float
                return ComparisonResult(
                    equal=False,
                    message=f"Float comparison failed: {actual} != {expected} (diff={diff}, rtol={rtol}, atol={atol})",
                    actual=actual,
                    expected=expected,
                    diff=f"difference: {diff}",
                )

        if actual == expected:
            return ComparisonResult(equal=True)

        return ComparisonResult(
            equal=False,
            message=f"Values do not match: {actual!r} != {expected!r}",
            actual=actual,
            expected=expected,
        )


class NumpyArrayComparator(Comparator):
    """Comparator for numpy arrays using np.allclose()."""

    def can_compare(self, value: Any) -> bool:
        try:
            import numpy as np

            return isinstance(value, np.ndarray)
        except ImportError:
            return False

    def compare(
        self,
        actual: Any,
        expected: Any,
        *,
        rtol: float = 1e-7,
        atol: float = 0,
    ) -> ComparisonResult:
        import numpy as np

        if not isinstance(expected, np.ndarray):
            expected = np.array(expected)

        if actual.shape != expected.shape:
            return ComparisonResult(
                equal=False,
                message=f"Array shapes do not match: {actual.shape} != {expected.shape}",
                actual=actual,
                expected=expected,
                diff=f"shape mismatch: {actual.shape} vs {expected.shape}",
            )

        if actual.dtype != expected.dtype:
            try:
                expected = expected.astype(actual.dtype)
            except (TypeError, ValueError):
                pass

        if np.issubdtype(actual.dtype, np.floating) or np.issubdtype(
            actual.dtype, np.complexfloating
        ):
            equal = np.allclose(actual, expected, rtol=rtol, atol=atol)
        else:
            equal = np.array_equal(actual, expected)

        if equal:
            return ComparisonResult(equal=True)

        if np.issubdtype(actual.dtype, np.floating):
            diff_array = actual - expected
            max_diff = np.max(np.abs(diff_array))
            max_diff_idx = np.unravel_index(np.argmax(np.abs(diff_array)), diff_array.shape)
            diff_msg = f"max absolute diff: {max_diff} at index {max_diff_idx}"
        else:
            mismatch = actual != expected
            mismatch_count = np.sum(mismatch)
            diff_msg = f"{mismatch_count} elements differ"

        return ComparisonResult(
            equal=False,
            message=f"Arrays do not match (rtol={rtol}, atol={atol})",
            actual=actual,
            expected=expected,
            diff=diff_msg,
        )


class ListComparator(Comparator):
    """Comparator for lists and tuples."""

    def can_compare(self, value: Any) -> bool:
        return isinstance(value, (list, tuple))

    def compare(
        self,
        actual: Any,
        expected: Any,
        *,
        rtol: float = 1e-7,
        atol: float = 0,
    ) -> ComparisonResult:
        if type(actual) is not type(expected):
            if not (isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple))):
                return ComparisonResult(
                    equal=False,
                    message=f"Type mismatch: {type(actual).__name__} != {type(expected).__name__}",
                    actual=actual,
                    expected=expected,
                )

        if len(actual) != len(expected):
            return ComparisonResult(
                equal=False,
                message=f"Length mismatch: {len(actual)} != {len(expected)}",
                actual=actual,
                expected=expected,
                diff=f"length: {len(actual)} vs {len(expected)}",
            )

        for i, (a, e) in enumerate(zip(actual, expected)):
            comparator = get_comparator(a)
            result = comparator.compare(a, e, rtol=rtol, atol=atol)
            if not result.equal:
                return ComparisonResult(
                    equal=False,
                    message=f"Element at index {i} differs: {result.message}",
                    actual=actual,
                    expected=expected,
                    diff=f"index {i}: {result.diff}" if result.diff else f"index {i}",
                )

        return ComparisonResult(equal=True)


class DictComparator(Comparator):
    """Comparator for dictionaries."""

    def can_compare(self, value: Any) -> bool:
        return isinstance(value, dict)

    def compare(
        self,
        actual: Any,
        expected: Any,
        *,
        rtol: float = 1e-7,
        atol: float = 0,
    ) -> ComparisonResult:
        if not isinstance(expected, dict):
            return ComparisonResult(
                equal=False,
                message=f"Type mismatch: {type(actual).__name__} != {type(expected).__name__}",
                actual=actual,
                expected=expected,
            )

        actual_keys = set(actual.keys())
        expected_keys = set(expected.keys())

        if actual_keys != expected_keys:
            missing = expected_keys - actual_keys
            extra = actual_keys - expected_keys
            parts = []
            if missing:
                parts.append(f"missing keys: {missing}")
            if extra:
                parts.append(f"extra keys: {extra}")
            return ComparisonResult(
                equal=False,
                message=f"Dictionary keys do not match: {', '.join(parts)}",
                actual=actual,
                expected=expected,
                diff="; ".join(parts),
            )

        for key in actual_keys:
            comparator = get_comparator(actual[key])
            result = comparator.compare(actual[key], expected[key], rtol=rtol, atol=atol)
            if not result.equal:
                return ComparisonResult(
                    equal=False,
                    message=f"Value at key '{key}' differs: {result.message}",
                    actual=actual,
                    expected=expected,
                    diff=f"key '{key}': {result.diff}" if result.diff else f"key '{key}'",
                )

        return ComparisonResult(equal=True)


class GenericComparator(Comparator):
    """Fallback comparator using equality."""

    def can_compare(self, value: Any) -> bool:
        return True

    def compare(
        self,
        actual: Any,
        expected: Any,
        *,
        rtol: float = 1e-7,
        atol: float = 0,
    ) -> ComparisonResult:
        try:
            equal = actual == expected
            if hasattr(equal, "__bool__"):
                equal = bool(equal)
            elif hasattr(equal, "all"):
                equal = equal.all()
        except Exception as e:
            return ComparisonResult(
                equal=False,
                message=f"Comparison raised exception: {e}",
                actual=actual,
                expected=expected,
            )

        if equal:
            return ComparisonResult(equal=True)

        return ComparisonResult(
            equal=False,
            message=f"Values do not match: {actual!r} != {expected!r}",
            actual=actual,
            expected=expected,
        )


class FunctionComparator(Comparator):
    """Comparator that uses a user-provided comparison function."""

    def __init__(self, compare_func: CompareFunc):
        """Initialize with a custom comparison function.

        Args:
            compare_func: A function that takes (actual, expected) and returns
                True if they match, False otherwise.
        """
        self._compare_func = compare_func

    def can_compare(self, value: Any) -> bool:
        return True

    def compare(
        self,
        actual: Any,
        expected: Any,
        *,
        rtol: float = 1e-7,
        atol: float = 0,
    ) -> ComparisonResult:
        try:
            equal = self._compare_func(actual, expected)
            if not isinstance(equal, bool):
                equal = bool(equal)
        except Exception as e:
            return ComparisonResult(
                equal=False,
                message=f"Custom comparison function raised exception: {e}",
                actual=actual,
                expected=expected,
            )

        if equal:
            return ComparisonResult(equal=True)

        return ComparisonResult(
            equal=False,
            message=f"Custom comparison failed: {actual!r} != {expected!r}",
            actual=actual,
            expected=expected,
        )


_COMPARATORS: list[Comparator] = [
    ScalarComparator(),
    NumpyArrayComparator(),
    ListComparator(),
    DictComparator(),
    GenericComparator(),
]


def get_comparator(value: Any) -> Comparator:
    """Get the appropriate comparator for a value.

    Args:
        value: The value to compare.

    Returns:
        A Comparator instance capable of handling the value type.
    """
    for comparator in _COMPARATORS:
        if comparator.can_compare(value):
            return comparator
    return GenericComparator()
