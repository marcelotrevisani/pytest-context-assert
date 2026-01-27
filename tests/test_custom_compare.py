"""Tests for custom compare function in context_assert fixture."""

from __future__ import annotations

import pytest

from pytest_context_assert.fixture import ContextAssertionError


class TestCustomCompareInFixture:
    """Tests for custom compare function in context_assert fixture."""

    def test_custom_compare_success(self, context_assert):
        """Test that custom compare function works for matching values."""
        context_assert._update_snapshots = True
        context_assert(42, name="custom_compare_test")

        context_assert._update_snapshots = False
        # Use custom compare that allows any value
        context_assert(100, name="custom_compare_test", compare=lambda a, b: True)

    def test_custom_compare_failure(self, context_assert):
        """Test that custom compare function can fail."""
        context_assert._update_snapshots = True
        context_assert(42, name="custom_compare_fail_test")

        context_assert._update_snapshots = False
        # Use custom compare that always fails
        with pytest.raises(ContextAssertionError):
            context_assert(42, name="custom_compare_fail_test", compare=lambda a, b: False)

    def test_custom_compare_with_tolerance(self, context_assert):
        """Test custom compare with tolerance logic."""
        context_assert._update_snapshots = True
        context_assert(100.0, name="custom_tol_test")

        context_assert._update_snapshots = False

        # Custom 10% tolerance
        def within_10_percent(a, b):
            return abs(a - b) / abs(b) <= 0.1

        context_assert(105.0, name="custom_tol_test", compare=within_10_percent)

        with pytest.raises(ContextAssertionError):
            context_assert(120.0, name="custom_tol_test", compare=within_10_percent)

    def test_custom_compare_case_insensitive(self, context_assert):
        """Test custom compare for case-insensitive string comparison."""
        context_assert._update_snapshots = True
        context_assert("Hello World", name="case_test")

        context_assert._update_snapshots = False
        context_assert("HELLO WORLD", name="case_test", compare=lambda a, b: a.lower() == b.lower())

    def test_custom_compare_ignores_rtol_atol(self, context_assert):
        """Test that custom compare ignores rtol/atol parameters."""
        context_assert._update_snapshots = True
        context_assert(1.0, name="ignore_tol_test")

        context_assert._update_snapshots = False
        # Even with strict tolerance, custom compare takes precedence
        context_assert(
            2.0,
            name="ignore_tol_test",
            rtol=1e-10,
            atol=0,
            compare=lambda a, b: True,
        )

    def test_custom_compare_with_dict(self, context_assert):
        """Test custom compare with dictionary values."""
        context_assert._update_snapshots = True
        context_assert({"a": 1, "b": 2}, name="dict_compare")

        context_assert._update_snapshots = False
        # Compare only specific keys
        context_assert(
            {"a": 1, "b": 3, "c": 4},
            name="dict_compare",
            compare=lambda a, b: a["a"] == b["a"],
        )

    def test_custom_compare_with_list(self, context_assert):
        """Test custom compare with list values."""
        context_assert._update_snapshots = True
        context_assert([1, 2, 3, 4, 5], name="list_compare")

        context_assert._update_snapshots = False
        # Compare only first 3 elements
        context_assert(
            [1, 2, 3, 99, 99],
            name="list_compare",
            compare=lambda a, b: a[:3] == b[:3],
        )

    def test_custom_compare_structural(self, context_assert):
        """Test custom compare for structural comparison."""
        context_assert._update_snapshots = True
        context_assert({"users": [{"name": "Alice"}]}, name="struct_compare")

        context_assert._update_snapshots = False

        # Check structure exists, ignore actual values
        def same_structure(a, b):
            if type(a) is not type(b):
                return False
            if isinstance(a, dict):
                return a.keys() == b.keys()
            if isinstance(a, list):
                return len(a) == len(b)
            return True

        context_assert({"users": [{"name": "Bob"}]}, name="struct_compare", compare=same_structure)

    def test_custom_compare_with_numpy(self, context_assert, numpy):
        """Test custom compare with numpy arrays."""
        arr1 = numpy.array([1.0, 2.0, 3.0])

        context_assert._update_snapshots = True
        context_assert(arr1, name="numpy_custom")

        context_assert._update_snapshots = False

        # Custom comparison: only check sum
        def same_sum(a, b):
            return numpy.sum(a) == numpy.sum(b)

        arr2 = numpy.array([3.0, 2.0, 1.0])  # Same sum, different order
        context_assert(arr2, name="numpy_custom", compare=same_sum)

    def test_custom_compare_with_exception_handling(self, context_assert):
        """Test that custom compare handles exceptions gracefully."""
        context_assert._update_snapshots = True
        context_assert("hello", name="exception_test")

        context_assert._update_snapshots = False

        def bad_compare(a, b):
            raise ValueError("Intentional error")

        with pytest.raises(ContextAssertionError) as exc_info:
            context_assert("hello", name="exception_test", compare=bad_compare)

        assert "exception" in str(exc_info.value).lower()

    def test_custom_compare_with_none(self, context_assert):
        """Test custom compare with None values."""
        context_assert._update_snapshots = True
        context_assert(None, name="none_compare")

        context_assert._update_snapshots = False
        # Custom compare that treats None specially
        context_assert(
            "not none",
            name="none_compare",
            compare=lambda a, b: b is None,  # Only check expected is None
        )

    def test_custom_compare_complex_object(self, context_assert):
        """Test custom compare with complex objects."""

        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        context_assert._update_snapshots = True
        context_assert({"x": 1.0, "y": 2.0}, name="point_compare")

        context_assert._update_snapshots = False

        def point_matches_dict(point, expected_dict):
            return point.x == expected_dict["x"] and point.y == expected_dict["y"]

        context_assert(Point(1.0, 2.0), name="point_compare", compare=point_matches_dict)
