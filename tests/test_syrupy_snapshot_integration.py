"""Integration tests for syrupy-style snapshot files.

These tests verify the full workflow of creating, reading, and comparing
snapshot files in the syrupy style.
"""

from __future__ import annotations

import math
from pathlib import Path

import yaml

from pytest_context_assert import set_context
from pytest_context_assert.storage import SnapshotStorage


class TestSnapshotFileCreation:
    """Tests for snapshot file creation via storage API."""

    def test_snapshot_file_created_on_set_value(self, tmp_path):
        """Verify that snapshot files are created when setting values."""
        storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_snapshot_creation",
            snapshot_dir=str(tmp_path),
        )

        # Initially no file exists
        assert not storage.exists()

        # Save a value - set_value automatically saves
        storage.set_value(
            context={"platform": "darwin", "arch": "arm64"},
            value=42,
            name="test_value",
        )

        # Now file exists
        assert storage.exists()

        # Verify file content
        snapshot_file = (
            tmp_path / "test_syrupy_snapshot_integration" / "test_snapshot_creation.yaml"
        )
        assert snapshot_file.exists()

        with open(snapshot_file) as f:
            content = yaml.safe_load(f)

        assert "_metadata" in content
        assert content["_metadata"]["test_name"] == "test_snapshot_creation"
        assert "assertions" in content
        assert "test_value" in content["assertions"]

    def test_snapshot_file_format_is_valid_yaml(self, tmp_path):
        """Verify that snapshot files are valid YAML."""
        storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_yaml_format",
            snapshot_dir=str(tmp_path),
        )

        ctx = {"platform": "test"}

        # Save various types
        storage.set_value(ctx, 42, name="integer")
        storage.set_value(ctx, "hello world", name="string")
        storage.set_value(ctx, 3.14159, name="float")
        storage.set_value(ctx, True, name="boolean")
        storage.set_value(ctx, None, name="none")

        # Read back and verify
        snapshot_file = tmp_path / "test_syrupy_snapshot_integration" / "test_yaml_format.yaml"
        with open(snapshot_file) as f:
            content = yaml.safe_load(f)

        assert len(content["assertions"]) == 5

    def test_snapshot_preserves_metadata(self, tmp_path):
        """Verify that snapshot metadata is preserved."""
        storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_metadata",
            snapshot_dir=str(tmp_path),
        )

        storage.set_value({"platform": "test"}, 1, name="value")

        # Read back
        snapshot_file = tmp_path / "test_syrupy_snapshot_integration" / "test_metadata.yaml"
        with open(snapshot_file) as f:
            content = yaml.safe_load(f)

        metadata = content["_metadata"]
        assert "version" in metadata
        assert "test_name" in metadata
        assert "created" in metadata
        assert metadata["test_name"] == "test_metadata"


class TestSnapshotFileReading:
    """Tests for reading and comparing snapshot files."""

    def test_read_existing_snapshot(self, tmp_path):
        """Verify that existing snapshots can be read."""
        # Create a snapshot file manually
        snapshot_dir = tmp_path / "test_syrupy_snapshot_integration"
        snapshot_dir.mkdir(parents=True)
        snapshot_file = snapshot_dir / "test_read_snapshot.yaml"

        snapshot_content = {
            "_metadata": {
                "version": 2,
                "test_name": "test_read_snapshot",
                "created": "2026-01-27T00:00:00+00:00",
            },
            "assertions": {
                "my_value": [
                    {
                        "__context__": {"default": True},
                        "value": 42,
                        "type": "int",
                    }
                ]
            },
        }

        with open(snapshot_file, "w") as f:
            yaml.dump(snapshot_content, f)

        # Read it back via storage
        storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_read_snapshot",
            snapshot_dir=str(tmp_path),
        )

        assert storage.exists()
        value = storage.get_value(context={"platform": "any"}, name="my_value")
        assert value == 42

    def test_context_matching_in_snapshot(self, tmp_path):
        """Verify that context matching works correctly."""
        snapshot_dir = tmp_path / "test_syrupy_snapshot_integration"
        snapshot_dir.mkdir(parents=True)
        snapshot_file = snapshot_dir / "test_context_matching.yaml"

        # Create snapshot with multiple contexts
        snapshot_content = {
            "_metadata": {
                "version": 2,
                "test_name": "test_context_matching",
            },
            "assertions": {
                "value": [
                    {
                        "__context__": {"platform": "linux", "arch": "x86_64"},
                        "value": "linux_value",
                        "type": "str",
                    },
                    {
                        "__context__": {"platform": "darwin", "arch": "arm64"},
                        "value": "darwin_arm_value",
                        "type": "str",
                    },
                    {
                        "__context__": {"default": True},
                        "value": "default_value",
                        "type": "str",
                    },
                ]
            },
        }

        with open(snapshot_file, "w") as f:
            yaml.dump(snapshot_content, f)

        storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_context_matching",
            snapshot_dir=str(tmp_path),
        )

        # Test with specific context
        value = storage.get_value(
            context={"platform": "linux", "arch": "x86_64"},
            name="value",
        )
        assert value == "linux_value"

        value = storage.get_value(
            context={"platform": "darwin", "arch": "arm64"},
            name="value",
        )
        assert value == "darwin_arm_value"

        # Test fallback to default
        value = storage.get_value(
            context={"platform": "windows", "arch": "x86_64"},
            name="value",
        )
        assert value == "default_value"


class TestSnapshotWithSpecialFloats:
    """Tests for snapshot files with special float values."""

    def test_snapshot_with_infinity(self, tmp_path):
        """Verify that infinity values are stored and retrieved correctly."""
        storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_infinity",
            snapshot_dir=str(tmp_path),
        )

        ctx = {"platform": "test"}

        # Store positive infinity
        storage.set_value(ctx, math.inf, name="pos_inf")

        # Store negative infinity
        storage.set_value(ctx, -math.inf, name="neg_inf")

        # Read back and verify
        new_storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_infinity",
            snapshot_dir=str(tmp_path),
        )

        pos_inf = new_storage.get_value(ctx, name="pos_inf")
        assert math.isinf(pos_inf) and pos_inf > 0

        neg_inf = new_storage.get_value(ctx, name="neg_inf")
        assert math.isinf(neg_inf) and neg_inf < 0

    def test_snapshot_with_nan(self, tmp_path):
        """Verify that NaN values are stored and retrieved correctly."""
        storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_nan",
            snapshot_dir=str(tmp_path),
        )

        ctx = {"platform": "test"}
        storage.set_value(ctx, math.nan, name="nan_value")

        # Read back
        new_storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_nan",
            snapshot_dir=str(tmp_path),
        )

        nan_val = new_storage.get_value(ctx, name="nan_value")
        assert math.isnan(nan_val)

    def test_full_workflow_with_special_floats(self, context_assert):
        """Test the full workflow with special float values through the fixture."""
        context_assert(math.inf, name="positive_infinity")
        context_assert(-math.inf, name="negative_infinity")
        context_assert(math.nan, name="nan_value")


class TestSnapshotWithNumpyArrays:
    """Tests for snapshot files with numpy arrays."""

    def test_snapshot_with_1d_array(self, numpy, tmp_path):
        """Verify that 1D numpy arrays are stored correctly."""
        storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_1d_array",
            snapshot_dir=str(tmp_path),
        )

        ctx = {"platform": "test"}
        arr = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0])
        storage.set_value(ctx, arr, name="array")

        # Read back
        new_storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_1d_array",
            snapshot_dir=str(tmp_path),
        )

        result = new_storage.get_value(ctx, name="array")
        assert isinstance(result, numpy.ndarray)
        assert result.shape == (5,)
        assert numpy.array_equal(result, arr)

    def test_snapshot_with_2d_array(self, numpy, tmp_path):
        """Verify that 2D numpy arrays are stored correctly."""
        storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_2d_array",
            snapshot_dir=str(tmp_path),
        )

        ctx = {"platform": "test"}
        arr = numpy.array([[1, 2, 3], [4, 5, 6]])
        storage.set_value(ctx, arr, name="matrix")

        # Read back
        new_storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_2d_array",
            snapshot_dir=str(tmp_path),
        )

        result = new_storage.get_value(ctx, name="matrix")
        assert result.shape == (2, 3)
        assert numpy.array_equal(result, arr)

    def test_full_workflow_with_numpy(self, numpy, context_assert):
        """Test the full workflow with numpy arrays through the fixture."""
        arr = numpy.array([1.0, 2.0, 3.0])
        context_assert(arr, name="simple_array", rtol=1e-10)

        matrix = numpy.array([[1, 2], [3, 4]])
        context_assert(matrix, name="simple_matrix")


class TestSnapshotWithComplexStructures:
    """Tests for snapshot files with complex nested structures."""

    def test_snapshot_with_nested_dict(self, tmp_path):
        """Verify that nested dicts are stored correctly."""
        storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_nested_dict",
            snapshot_dir=str(tmp_path),
        )

        ctx = {"platform": "test"}
        data = {
            "database": {
                "host": "localhost",
                "port": 5432,
            },
            "cache": {
                "enabled": True,
            },
        }
        storage.set_value(ctx, data, name="config")

        # Read back
        new_storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_nested_dict",
            snapshot_dir=str(tmp_path),
        )

        result = new_storage.get_value(ctx, name="config")
        assert result["database"]["host"] == "localhost"
        assert result["database"]["port"] == 5432
        assert result["cache"]["enabled"] is True

    def test_snapshot_with_mixed_list(self, tmp_path):
        """Verify that lists with mixed types are stored correctly."""
        storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_mixed_list",
            snapshot_dir=str(tmp_path),
        )

        ctx = {"platform": "test"}
        data = [1, "hello", 3.14, True, None]
        storage.set_value(ctx, data, name="items")

        # Read back
        new_storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_mixed_list",
            snapshot_dir=str(tmp_path),
        )

        result = new_storage.get_value(ctx, name="items")
        assert result == [1, "hello", 3.14, True, None]

    def test_full_workflow_with_complex_structure(self, context_assert):
        """Test the full workflow with complex structures through the fixture."""
        data = {
            "users": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25},
            ],
            "settings": {
                "theme": "dark",
                "notifications": True,
            },
            "version": 1.0,
        }
        context_assert(data, name="complex_config")


class TestSnapshotContextVariations:
    """Tests for snapshot files with different context configurations."""

    @set_context({"environment": "production"})
    def test_custom_context_in_snapshot(self, context_assert):
        """Test that custom context is stored in snapshots."""
        context_assert(42, name="prod_value")

    @set_context({"environment": "staging", "region": "us-east"})
    def test_multiple_custom_context_keys(self, context_assert):
        """Test multiple custom context keys in snapshots."""
        context_assert("staging_data", name="env_data")

    def test_runtime_context_override(self, context_assert):
        """Test runtime context override in snapshots."""
        custom = context_assert.with_context(feature_flag="enabled")
        custom(100, name="feature_value")

    def test_default_context_fallback(self, tmp_path):
        """Verify default context fallback behavior."""
        snapshot_dir = tmp_path / "test_syrupy_snapshot_integration"
        snapshot_dir.mkdir(parents=True)
        snapshot_file = snapshot_dir / "test_default_fallback.yaml"

        # Create snapshot with only default context
        snapshot_content = {
            "_metadata": {"version": 2, "test_name": "test_default_fallback"},
            "assertions": {
                "value": [
                    {
                        "__context__": {"default": True},
                        "value": "fallback",
                        "type": "str",
                    }
                ]
            },
        }

        with open(snapshot_file, "w") as f:
            yaml.dump(snapshot_content, f)

        storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_default_fallback",
            snapshot_dir=str(tmp_path),
        )

        # Any context should fall back to default
        for ctx in [
            {"platform": "linux"},
            {"platform": "darwin", "arch": "arm64"},
            {"custom": "context"},
        ]:
            value = storage.get_value(ctx, name="value")
            assert value == "fallback"


class TestSnapshotUpdateWorkflow:
    """Tests for snapshot update workflow."""

    def test_update_adds_new_assertion(self, tmp_path):
        """Verify that updating adds new assertions."""
        storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_update_new",
            snapshot_dir=str(tmp_path),
        )

        ctx = {"platform": "test"}

        # Add first assertion
        storage.set_value(ctx, 1, name="first")

        # Add second assertion
        storage.set_value(ctx, 2, name="second")

        # Both should exist
        assert storage.has_value(ctx, name="first")
        assert storage.has_value(ctx, name="second")

    def test_update_modifies_existing_assertion(self, tmp_path):
        """Verify that updating modifies existing assertions."""
        storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_update_modify",
            snapshot_dir=str(tmp_path),
        )

        ctx = {"platform": "test"}

        # Add initial value
        storage.set_value(ctx, 1, name="value")

        # Modify value
        storage.set_value(ctx, 2, name="value")

        # Should have new value
        data = storage.get_value(ctx, name="value")
        assert data == 2

    def test_update_preserves_other_contexts(self, tmp_path):
        """Verify that updating preserves other contexts."""
        snapshot_dir = tmp_path / "test_syrupy_snapshot_integration"
        snapshot_dir.mkdir(parents=True)
        snapshot_file = snapshot_dir / "test_preserve_contexts.yaml"

        # Create snapshot with multiple contexts
        snapshot_content = {
            "_metadata": {"version": 2, "test_name": "test_preserve_contexts"},
            "assertions": {
                "value": [
                    {
                        "__context__": {"platform": "linux"},
                        "value": "linux",
                        "type": "str",
                    },
                    {
                        "__context__": {"platform": "darwin"},
                        "value": "darwin",
                        "type": "str",
                    },
                ]
            },
        }

        with open(snapshot_file, "w") as f:
            yaml.dump(snapshot_content, f)

        # Update for a new context
        storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_preserve_contexts",
            snapshot_dir=str(tmp_path),
        )

        storage.set_value(
            {"platform": "windows"},
            "windows",
            name="value",
        )

        # All contexts should exist
        assert storage.get_value({"platform": "linux"}, name="value") == "linux"
        assert storage.get_value({"platform": "darwin"}, name="value") == "darwin"
        assert storage.get_value({"platform": "windows"}, name="value") == "windows"


class TestSnapshotFileIntegrity:
    """Tests for snapshot file integrity and edge cases."""

    def test_snapshot_handles_unicode(self, context_assert):
        """Verify that unicode characters are handled correctly."""
        context_assert("Hello, 世界! 🌍", name="unicode_string")

    def test_snapshot_handles_empty_string(self, context_assert):
        """Verify that empty strings are handled correctly."""
        context_assert("", name="empty_string")

    def test_snapshot_handles_empty_list(self, context_assert):
        """Verify that empty lists are handled correctly."""
        context_assert([], name="empty_list")

    def test_snapshot_handles_empty_dict(self, context_assert):
        """Verify that empty dicts are handled correctly."""
        context_assert({}, name="empty_dict")

    def test_snapshot_handles_zero_values(self, context_assert):
        """Verify that zero values are handled correctly."""
        context_assert(0, name="zero_int")
        context_assert(0.0, name="zero_float")

    def test_snapshot_handles_negative_values(self, context_assert):
        """Verify that negative values are handled correctly."""
        context_assert(-42, name="negative_int")
        context_assert(-3.14, name="negative_float")

    def test_snapshot_handles_very_large_numbers(self, context_assert):
        """Verify that very large numbers are handled correctly."""
        context_assert(10**100, name="large_int")
        context_assert(1e308, name="large_float")

    def test_snapshot_handles_very_small_numbers(self, context_assert):
        """Verify that very small numbers are handled correctly."""
        context_assert(1e-308, name="small_float")


class TestSnapshotComparisonBehavior:
    """Tests for snapshot comparison behavior."""

    def test_comparison_succeeds_for_matching_values(self, context_assert):
        """Verify that comparison succeeds when values match."""
        context_assert(42, name="matching_value")

    def test_comparison_with_tolerance(self, context_assert):
        """Verify that comparison with tolerance works."""
        context_assert(3.14159265, name="pi_approx", rtol=1e-5)

    def test_comparison_with_custom_compare(self, context_assert):
        """Verify that custom comparison functions work."""
        context_assert(
            "HELLO WORLD",
            name="case_insensitive",
            compare=lambda a, b: a.lower() == b.lower(),
        )


class TestMultipleAssertionsPerTest:
    """Tests for multiple assertions in a single test."""

    def test_multiple_named_assertions(self, context_assert):
        """Test multiple named assertions in one test."""
        context_assert(1, name="first")
        context_assert(2, name="second")
        context_assert(3, name="third")

    def test_multiple_auto_named_assertions(self, context_assert):
        """Test multiple auto-named assertions in one test."""
        context_assert(10)  # assertion_1
        context_assert(20)  # assertion_2
        context_assert(30)  # assertion_3

    def test_mixed_named_and_auto_assertions(self, context_assert):
        """Test mix of named and auto-named assertions."""
        context_assert(100, name="explicit")
        context_assert(200)  # assertion_1
        context_assert(300, name="another_explicit")
        context_assert(400)  # assertion_2


class TestSnapshotWithTuples:
    """Tests for snapshot files with tuples."""

    def test_tuple_preserved_as_tuple(self, context_assert):
        """Verify that tuples are preserved as tuples."""
        context_assert((1, 2, 3), name="simple_tuple")

    def test_nested_tuple(self, context_assert):
        """Verify that nested tuples work."""
        context_assert(((1, 2), (3, 4)), name="nested_tuple")

    def test_tuple_in_dict(self, context_assert):
        """Verify that tuples in dicts work."""
        context_assert({"coords": (10, 20)}, name="tuple_in_dict")


class TestSnapshotYAMLReadability:
    """Tests to verify YAML files are human-readable."""

    def test_yaml_file_is_readable(self, tmp_path):
        """Verify that generated YAML is human-readable."""
        storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_readable_yaml",
            snapshot_dir=str(tmp_path),
        )

        ctx = {"platform": "test"}
        storage.set_value(ctx, {"name": "test", "count": 42}, name="config")

        # Read the raw file content
        snapshot_file = tmp_path / "test_syrupy_snapshot_integration" / "test_readable_yaml.yaml"
        content = snapshot_file.read_text()

        # Verify it's readable (no binary, proper indentation)
        assert "_metadata:" in content
        assert "assertions:" in content
        assert "config:" in content
        # Should not have Python-specific tags
        assert "!!python" not in content


class TestSnapshotGetAllContexts:
    """Tests for getting all contexts from a snapshot."""

    def test_get_all_contexts_single(self, tmp_path):
        """Test getting all contexts with single context."""
        storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_single_context",
            snapshot_dir=str(tmp_path),
        )

        ctx = {"platform": "linux", "arch": "x86_64"}
        storage.set_value(ctx, 42, name="value")

        contexts = storage.get_all_contexts(name="value")
        assert len(contexts) == 1
        assert contexts[0] == ctx

    def test_get_all_contexts_multiple(self, tmp_path):
        """Test getting all contexts with multiple contexts."""
        storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_multiple_contexts",
            snapshot_dir=str(tmp_path),
        )

        ctx1 = {"platform": "linux"}
        ctx2 = {"platform": "darwin"}
        ctx3 = {"platform": "windows"}

        storage.set_value(ctx1, "linux_val", name="value")
        storage.set_value(ctx2, "darwin_val", name="value")
        storage.set_value(ctx3, "windows_val", name="value")

        contexts = storage.get_all_contexts(name="value")
        assert len(contexts) == 3


class TestSnapshotDeleteValue:
    """Tests for deleting values from snapshots."""

    def test_delete_existing_value(self, tmp_path):
        """Test deleting an existing value."""
        storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_delete",
            snapshot_dir=str(tmp_path),
        )

        ctx = {"platform": "test"}
        storage.set_value(ctx, 42, name="value")
        assert storage.has_value(ctx, name="value")

        result = storage.delete_value(ctx, name="value")
        assert result is True
        assert not storage.has_value(ctx, name="value")

    def test_delete_nonexistent_value(self, tmp_path):
        """Test deleting a value that doesn't exist."""
        storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_delete_nonexistent",
            snapshot_dir=str(tmp_path),
        )

        ctx = {"platform": "test"}
        storage.set_value(ctx, 42, name="other")

        result = storage.delete_value(ctx, name="nonexistent")
        assert result is False


class TestSnapshotHasValue:
    """Tests for checking if values exist in snapshots."""

    def test_has_value_true(self, tmp_path):
        """Test has_value returns True for existing value."""
        storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_has_value",
            snapshot_dir=str(tmp_path),
        )

        ctx = {"platform": "test"}
        storage.set_value(ctx, 42, name="value")
        assert storage.has_value(ctx, name="value") is True

    def test_has_value_false_no_file(self, tmp_path):
        """Test has_value returns False when no file exists."""
        storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_no_file",
            snapshot_dir=str(tmp_path),
        )

        ctx = {"platform": "test"}
        assert storage.has_value(ctx, name="value") is False

    def test_has_value_false_no_assertion(self, tmp_path):
        """Test has_value returns False when assertion doesn't exist."""
        storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_no_assertion",
            snapshot_dir=str(tmp_path),
        )

        ctx = {"platform": "test"}
        storage.set_value(ctx, 42, name="other")
        assert storage.has_value(ctx, name="nonexistent") is False

    def test_has_value_with_default_fallback(self, tmp_path):
        """Test has_value with default context fallback."""
        snapshot_dir = tmp_path / "test_syrupy_snapshot_integration"
        snapshot_dir.mkdir(parents=True)
        snapshot_file = snapshot_dir / "test_has_default.yaml"

        snapshot_content = {
            "_metadata": {"version": 2, "test_name": "test_has_default"},
            "assertions": {
                "value": [
                    {
                        "__context__": {"default": True},
                        "value": 42,
                        "type": "int",
                    }
                ]
            },
        }

        with open(snapshot_file, "w") as f:
            yaml.dump(snapshot_content, f)

        storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_has_default",
            snapshot_dir=str(tmp_path),
        )

        # Should find value via default fallback
        assert storage.has_value({"platform": "any"}, name="value") is True


class TestSnapshotWithNumpySpecialValues:
    """Tests for numpy arrays with special values in snapshots."""

    def test_numpy_array_with_inf(self, numpy, tmp_path):
        """Test numpy array containing infinity."""
        storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_np_inf",
            snapshot_dir=str(tmp_path),
        )

        ctx = {"platform": "test"}
        arr = numpy.array([1.0, numpy.inf, -numpy.inf, 2.0])
        storage.set_value(ctx, arr, name="array")

        new_storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_np_inf",
            snapshot_dir=str(tmp_path),
        )

        result = new_storage.get_value(ctx, name="array")
        assert result[0] == 1.0
        assert numpy.isinf(result[1]) and result[1] > 0
        assert numpy.isinf(result[2]) and result[2] < 0
        assert result[3] == 2.0

    def test_numpy_array_with_nan(self, numpy, tmp_path):
        """Test numpy array containing NaN."""
        storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_np_nan",
            snapshot_dir=str(tmp_path),
        )

        ctx = {"platform": "test"}
        arr = numpy.array([1.0, numpy.nan, 2.0])
        storage.set_value(ctx, arr, name="array")

        new_storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_np_nan",
            snapshot_dir=str(tmp_path),
        )

        result = new_storage.get_value(ctx, name="array")
        assert result[0] == 1.0
        assert numpy.isnan(result[1])
        assert result[2] == 2.0


class TestSnapshotParseContextKey:
    """Tests for parsing legacy context key strings."""

    def test_parse_simple_context_key(self, tmp_path):
        """Test parsing simple context key."""
        storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_parse",
            snapshot_dir=str(tmp_path),
        )

        # Use string context key
        storage.set_value("linux-x86_64", 42, name="value")

        # Should be retrievable with dict context
        value = storage.get_value({"platform": "linux", "arch": "x86_64"}, name="value")
        assert value == 42

    def test_parse_context_key_with_blas(self, tmp_path):
        """Test parsing context key with blas."""
        storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_parse_blas",
            snapshot_dir=str(tmp_path),
        )

        storage.set_value("darwin-arm64-accelerate", 100, name="value")

        value = storage.get_value(
            {"platform": "darwin", "arch": "arm64", "blas": "accelerate"},
            name="value",
        )
        assert value == 100

    def test_default_string_context(self, tmp_path):
        """Test using 'default' string context."""
        storage = SnapshotStorage(
            test_file=Path(__file__),
            test_name="test_default_string",
            snapshot_dir=str(tmp_path),
        )

        storage.set_value("default", 999, name="value")

        # Should be retrievable as fallback
        value = storage.get_value({"platform": "any"}, name="value")
        assert value == 999
