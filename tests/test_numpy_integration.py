"""Integration tests with numpy arrays."""

from __future__ import annotations

import pytest

from pytest_context_assert.fixture import ContextAssertionError


class TestNumpyIntegration:
    """Integration tests with numpy arrays."""

    def test_array_update_and_compare(self, context_assert, numpy):
        """Test updating and comparing numpy arrays."""
        arr = numpy.array([1.0, 2.0, 3.0])

        context_assert._update_snapshots = True
        context_assert(arr, name="array_test")

        context_assert._update_snapshots = False
        context_assert(arr.copy(), name="array_test")

    def test_array_with_tolerance(self, context_assert, numpy):
        """Test array comparison with tolerance."""
        arr1 = numpy.array([1.0, 2.0, 3.0])
        arr2 = numpy.array([1.0000001, 2.0000001, 3.0000001])

        context_assert._update_snapshots = True
        context_assert(arr1, name="array_tol_test")

        context_assert._update_snapshots = False
        context_assert(arr2, rtol=1e-5, name="array_tol_test")

    def test_2d_array(self, context_assert, numpy):
        """Test 2D array handling."""
        arr = numpy.array([[1, 2], [3, 4]])

        context_assert._update_snapshots = True
        context_assert(arr, name="array_2d_test")

        context_assert._update_snapshots = False
        context_assert(arr.copy(), name="array_2d_test")

    def test_3d_array(self, context_assert, numpy):
        """Test 3D array handling."""
        arr = numpy.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

        context_assert._update_snapshots = True
        context_assert(arr, name="array_3d_test")

        context_assert._update_snapshots = False
        context_assert(arr.copy(), name="array_3d_test")

    def test_array_mismatch_fails(self, context_assert, numpy):
        """Test that mismatched arrays fail."""
        arr1 = numpy.array([1.0, 2.0, 3.0])
        arr2 = numpy.array([1.0, 2.0, 4.0])

        context_assert._update_snapshots = True
        context_assert(arr1, name="array_mismatch")

        context_assert._update_snapshots = False
        with pytest.raises(ContextAssertionError):
            context_assert(arr2, name="array_mismatch")

    def test_array_shape_mismatch_fails(self, context_assert, numpy):
        """Test that shape mismatches fail."""
        arr1 = numpy.array([1, 2, 3])
        arr2 = numpy.array([[1, 2], [3, 4]])

        context_assert._update_snapshots = True
        context_assert(arr1, name="shape_mismatch")

        context_assert._update_snapshots = False
        with pytest.raises(ContextAssertionError):
            context_assert(arr2, name="shape_mismatch")

    def test_integer_array(self, context_assert, numpy):
        """Test integer array handling."""
        arr = numpy.array([1, 2, 3], dtype=numpy.int32)

        context_assert._update_snapshots = True
        context_assert(arr, name="int_array")

        context_assert._update_snapshots = False
        context_assert(arr.copy(), name="int_array")

    def test_complex_array(self, context_assert, numpy):
        """Test complex array handling."""
        arr = numpy.array([1 + 2j, 3 + 4j, 5 + 6j])

        context_assert._update_snapshots = True
        context_assert(arr, name="complex_array")

        context_assert._update_snapshots = False
        context_assert(arr.copy(), name="complex_array")

    def test_empty_array(self, context_assert, numpy):
        """Test empty array handling."""
        arr = numpy.array([])

        context_assert._update_snapshots = True
        context_assert(arr, name="empty_array")

        context_assert._update_snapshots = False
        context_assert(numpy.array([]), name="empty_array")

    def test_large_array_tolerance(self, context_assert, numpy):
        """Test large array with small numerical differences."""
        numpy.random.seed(42)
        arr1 = numpy.random.rand(100, 100)
        # Add very small noise
        arr2 = arr1 + numpy.random.rand(100, 100) * 1e-10

        context_assert._update_snapshots = True
        context_assert(arr1, name="large_array")

        context_assert._update_snapshots = False
        context_assert(arr2, rtol=1e-7, atol=1e-9, name="large_array")

    def test_array_with_nan(self, context_assert, numpy):
        """Test array with NaN values (should fail comparison)."""
        arr1 = numpy.array([1.0, numpy.nan, 3.0])
        arr2 = numpy.array([1.0, numpy.nan, 3.0])

        context_assert._update_snapshots = True
        context_assert(arr1, name="nan_array")

        context_assert._update_snapshots = False
        # NaN != NaN, so this should fail without special handling
        with pytest.raises(ContextAssertionError):
            context_assert(arr2, name="nan_array")

    def test_array_with_inf(self, context_assert, numpy):
        """Test array with infinity values."""
        arr = numpy.array([1.0, numpy.inf, -numpy.inf])

        context_assert._update_snapshots = True
        context_assert(arr, name="inf_array")

        context_assert._update_snapshots = False
        context_assert(arr.copy(), name="inf_array")

    def test_boolean_array(self, context_assert, numpy):
        """Test boolean array handling."""
        arr = numpy.array([True, False, True, False])

        context_assert._update_snapshots = True
        context_assert(arr, name="bool_array")

        context_assert._update_snapshots = False
        context_assert(arr.copy(), name="bool_array")

    def test_string_array(self, context_assert, numpy):
        """Test string array handling."""
        arr = numpy.array(["hello", "world", "test"])

        context_assert._update_snapshots = True
        context_assert(arr, name="string_array")

        context_assert._update_snapshots = False
        context_assert(arr.copy(), name="string_array")
