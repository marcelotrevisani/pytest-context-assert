"""Tests for syrupy-style assert_match method."""

from __future__ import annotations

import pytest

from pytest_context_assert.fixture import ContextAssertionError


class TestSyrupyStyleAPI:
    """Tests for syrupy-style assert_match method."""

    def test_assert_match_basic(self, context_assert):
        """Test assert_match as alias for __call__."""
        context_assert._update_snapshots = True
        context_assert.assert_match(42, name="syrupy_style")

        context_assert._update_snapshots = False
        context_assert.assert_match(42, name="syrupy_style")

    def test_assert_match_with_tolerance(self, context_assert):
        """Test assert_match with tolerance parameters."""
        context_assert._update_snapshots = True
        context_assert.assert_match(1.0, name="syrupy_tol")

        context_assert._update_snapshots = False
        context_assert.assert_match(1.0000001, name="syrupy_tol", rtol=1e-5)

    def test_assert_match_with_custom_compare(self, context_assert):
        """Test assert_match with custom comparison."""
        context_assert._update_snapshots = True
        context_assert.assert_match("hello", name="syrupy_compare")

        context_assert._update_snapshots = False
        context_assert.assert_match(
            "HELLO", name="syrupy_compare", compare=lambda a, b: a.lower() == b.lower()
        )

    def test_assert_match_failure(self, context_assert):
        """Test assert_match raises on mismatch."""
        context_assert._update_snapshots = True
        context_assert.assert_match(42, name="syrupy_fail")

        context_assert._update_snapshots = False
        with pytest.raises(ContextAssertionError):
            context_assert.assert_match(43, name="syrupy_fail")

    def test_assert_match_with_dict(self, context_assert):
        """Test assert_match with dictionary values."""
        context_assert._update_snapshots = True
        context_assert.assert_match({"key": "value", "count": 42}, name="dict_match")

        context_assert._update_snapshots = False
        context_assert.assert_match({"key": "value", "count": 42}, name="dict_match")

    def test_assert_match_with_list(self, context_assert):
        """Test assert_match with list values."""
        context_assert._update_snapshots = True
        context_assert.assert_match([1, 2, 3, "four"], name="list_match")

        context_assert._update_snapshots = False
        context_assert.assert_match([1, 2, 3, "four"], name="list_match")

    def test_assert_match_with_nested_structure(self, context_assert):
        """Test assert_match with nested data structures."""
        data = {
            "users": [
                {"name": "Alice", "scores": [85, 90, 95]},
                {"name": "Bob", "scores": [70, 75, 80]},
            ],
            "metadata": {"version": 1},
        }

        context_assert._update_snapshots = True
        context_assert.assert_match(data, name="nested_match")

        context_assert._update_snapshots = False
        context_assert.assert_match(data, name="nested_match")

    def test_assert_match_auto_naming(self, context_assert):
        """Test assert_match with auto-generated names."""
        context_assert._update_snapshots = True
        context_assert.assert_match(1)  # assertion_1
        context_assert.assert_match(2)  # assertion_2
        context_assert.assert_match(3)  # assertion_3

        context_assert._assertion_count = 0
        context_assert._update_snapshots = False
        context_assert.assert_match(1)
        context_assert.assert_match(2)
        context_assert.assert_match(3)

    def test_assert_match_with_numpy(self, context_assert, numpy):
        """Test assert_match with numpy arrays."""
        arr = numpy.array([1.0, 2.0, 3.0])

        context_assert._update_snapshots = True
        context_assert.assert_match(arr, name="numpy_match")

        context_assert._update_snapshots = False
        context_assert.assert_match(arr.copy(), name="numpy_match")

    def test_assert_match_with_atol(self, context_assert):
        """Test assert_match with absolute tolerance."""
        context_assert._update_snapshots = True
        context_assert.assert_match(0.0, name="atol_match")

        context_assert._update_snapshots = False
        context_assert.assert_match(0.0001, name="atol_match", atol=0.001)

    def test_assert_match_with_serialize(self, context_assert):
        """Test assert_match with custom serialization."""

        class Custom:
            def __init__(self, val):
                self.val = val

        context_assert._update_snapshots = True
        context_assert.assert_match(
            Custom(42),
            name="serialize_match",
            serialize=lambda x: {"val": x.val, "type": "Custom"},
        )

        context_assert._update_snapshots = False
        context_assert.assert_match(
            Custom(42),
            name="serialize_match",
            deserialize=lambda d: Custom(d["val"]),
            compare=lambda a, b: a.val == b.val,
        )
