"""Tests for the context_assert fixture."""

from __future__ import annotations

import pytest

from pytest_context_assert.fixture import ContextAssertionError, MissingSnapshotError


class TestContextAssertFixture:
    """Integration tests for the context_assert fixture."""

    def test_fixture_available(self, context_assert):
        """Test that the fixture is available."""
        assert context_assert is not None

    def test_update_and_compare(self, context_assert):
        """Test updating and comparing values."""
        context_assert._update_snapshots = True
        context_assert(42, name="value")

        context_assert._update_snapshots = False
        context_assert(42, name="value")

    def test_comparison_failure(self, context_assert):
        """Test that mismatched values raise an error."""
        context_assert._update_snapshots = True
        context_assert(42, name="value")

        context_assert._update_snapshots = False
        with pytest.raises(ContextAssertionError):
            context_assert(43, name="value")

    def test_missing_snapshot_error(self, context_assert):
        """Test that missing snapshots raise an error."""
        with pytest.raises(MissingSnapshotError):
            context_assert(42, name="nonexistent")

    def test_auto_named_assertions(self, context_assert):
        """Test that assertions without name get auto-generated names based on order."""
        context_assert._update_snapshots = True
        context_assert(42)  # Gets name "assertion_1"
        context_assert("hello")  # Gets name "assertion_2"
        context_assert(3.14)  # Gets name "assertion_3"

        # Reset for comparison
        context_assert._assertion_count = 0
        context_assert._update_snapshots = False
        context_assert(42)  # Compares with "assertion_1"
        context_assert("hello")  # Compares with "assertion_2"
        context_assert(3.14)  # Compares with "assertion_3"

    def test_named_assertions(self, context_assert):
        """Test multiple named assertions in one test."""
        context_assert._update_snapshots = True
        context_assert(42, name="first")
        context_assert("hello", name="second")

        context_assert._update_snapshots = False
        context_assert(42, name="first")
        context_assert("hello", name="second")

    def test_float_with_tolerance(self, context_assert):
        """Test float comparison with tolerance."""
        context_assert._update_snapshots = True
        context_assert(1.0, name="float_test")

        context_assert._update_snapshots = False
        context_assert(1.0000001, rtol=1e-5, name="float_test")

    def test_with_context_override(self, context_assert):
        """Test context override."""
        context_assert._update_snapshots = True
        custom = context_assert.with_context(custom="value")
        assert "custom" in custom.context_key

    def test_context_key_property(self, context_assert):
        """Test that context_key property works."""
        key = context_assert.context_key
        assert isinstance(key, str)
        assert "-" in key

    def test_storage_property(self, context_assert):
        """Test that storage property works."""
        storage = context_assert.storage
        assert storage is not None

    def test_update_method(self, context_assert):
        """Test explicit update method."""
        context_assert.update(42, name="explicit_update")
        context_assert._update_snapshots = False
        context_assert(42, name="explicit_update")

    def test_set_default_method(self, context_assert):
        """Test set_default method."""
        context_assert.set_default(42, name="default_value")
        # The value is stored under "default" context
        value = context_assert.storage.get_value("default", name="default_value")
        assert value == 42

    def test_assertion_count_increments(self, context_assert):
        """Test that assertion count increments correctly."""
        context_assert._update_snapshots = True
        assert context_assert._assertion_count == 0

        context_assert(1)
        assert context_assert._assertion_count == 1

        context_assert(2)
        assert context_assert._assertion_count == 2

        context_assert(3)
        assert context_assert._assertion_count == 3

    def test_named_assertion_doesnt_increment_counter(self, context_assert):
        """Test that named assertions don't increment the counter."""
        context_assert._update_snapshots = True
        assert context_assert._assertion_count == 0

        context_assert(1, name="named")
        assert context_assert._assertion_count == 0

    def test_with_context_preserves_assertion_count(self, context_assert):
        """Test that with_context preserves assertion count."""
        context_assert._update_snapshots = True
        context_assert(1)  # counter = 1
        context_assert(2)  # counter = 2

        custom = context_assert.with_context(env="test")
        assert custom._assertion_count == 2

    def test_error_includes_context_key(self, context_assert):
        """Test that error includes context key."""
        context_assert._update_snapshots = True
        context_assert(42, name="error_test")

        context_assert._update_snapshots = False
        with pytest.raises(ContextAssertionError) as exc_info:
            context_assert(43, name="error_test")

        assert exc_info.value.context_key is not None

    def test_error_includes_actual_and_expected(self, context_assert):
        """Test that error includes actual and expected values."""
        context_assert._update_snapshots = True
        context_assert(42, name="values_test")

        context_assert._update_snapshots = False
        with pytest.raises(ContextAssertionError) as exc_info:
            context_assert(43, name="values_test")

        assert exc_info.value.actual == 43
        assert exc_info.value.expected == 42

    def test_missing_snapshot_error_includes_path(self, context_assert):
        """Test that MissingSnapshotError includes snapshot path."""
        with pytest.raises(MissingSnapshotError) as exc_info:
            context_assert(42, name="missing_test")

        assert exc_info.value.snapshot_path is not None
        assert exc_info.value.context_key is not None
