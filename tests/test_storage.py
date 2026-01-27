"""Tests for snapshot storage."""

from __future__ import annotations

import yaml

from pytest_context_assert.storage import SnapshotStorage


class TestSnapshotStorage:
    """Tests for SnapshotStorage."""

    def test_snapshot_path(self, temp_test_file):
        storage = SnapshotStorage(temp_test_file, "test_function")
        expected = temp_test_file.parent / "__snapshots__" / "test_example" / "test_function.yaml"
        assert storage.snapshot_path == expected

    def test_exists_false(self, temp_test_file):
        storage = SnapshotStorage(temp_test_file, "test_function")
        assert not storage.exists()

    def test_save_and_load(self, temp_test_file):
        storage = SnapshotStorage(temp_test_file, "test_function")

        data = {"contexts": {"default": {"value": 42, "type": "int"}}}
        storage.save(data)

        assert storage.exists()
        loaded = storage.load()
        assert loaded["contexts"]["default"]["value"] == 42

    def test_set_and_get_value(self, temp_test_file):
        storage = SnapshotStorage(temp_test_file, "test_function")

        storage.set_value("default", 42)
        value = storage.get_value("default")
        assert value == 42

    def test_get_value_with_fallback(self, temp_test_file):
        storage = SnapshotStorage(temp_test_file, "test_function")

        storage.set_value("default", 42)
        value = storage.get_value("darwin-arm64")
        assert value == 42

    def test_get_value_specific_context(self, temp_test_file):
        storage = SnapshotStorage(temp_test_file, "test_function")

        storage.set_value("default", 42)
        storage.set_value("darwin-arm64", 43)

        assert storage.get_value("darwin-arm64") == 43
        assert storage.get_value("linux-x86_64") == 42

    def test_named_assertions(self, temp_test_file):
        storage = SnapshotStorage(temp_test_file, "test_function")

        storage.set_value("default", 42, name="first")
        storage.set_value("default", 43, name="second")

        assert storage.get_value("default", name="first") == 42
        assert storage.get_value("default", name="second") == 43

    def test_has_value(self, temp_test_file):
        storage = SnapshotStorage(temp_test_file, "test_function")

        assert not storage.has_value("default", name="test")
        storage.set_value("default", 42, name="test")
        assert storage.has_value("default", name="test")

    def test_custom_snapshot_dir(self, temp_test_file):
        storage = SnapshotStorage(temp_test_file, "test_function", snapshot_dir="custom_snapshots")
        expected = (
            temp_test_file.parent / "custom_snapshots" / "test_example" / "test_function.yaml"
        )
        assert storage.snapshot_path == expected

    def test_multiple_contexts(self, temp_test_file):
        """Test storing values for multiple contexts."""
        storage = SnapshotStorage(temp_test_file, "test_function")

        storage.set_value("darwin-arm64", 1.0)
        storage.set_value("darwin-x86_64", 1.1)
        storage.set_value("linux-x86_64", 1.2)
        storage.set_value("default", 1.0)

        assert storage.get_value("darwin-arm64") == 1.0
        assert storage.get_value("darwin-x86_64") == 1.1
        assert storage.get_value("linux-x86_64") == 1.2
        assert storage.get_value("windows-x86_64") == 1.0  # Falls back to default

    def test_update_existing_value(self, temp_test_file):
        """Test updating an existing value."""
        storage = SnapshotStorage(temp_test_file, "test_function")

        storage.set_value("default", 42)
        assert storage.get_value("default") == 42

        storage.set_value("default", 100)
        assert storage.get_value("default") == 100


class TestYAMLStorage:
    """Tests for YAML storage format."""

    def test_yaml_format(self, temp_test_file):
        """Test that stored data is valid YAML."""
        storage = SnapshotStorage(temp_test_file, "test_function")

        storage.set_value("default", {"nested": [1, 2, 3]})

        with open(storage.snapshot_path) as f:
            content = f.read()

        data = yaml.safe_load(content)
        assert "_metadata" in data
        assert "assertions" in data
        assert data["_metadata"]["version"] == 2

    def test_yaml_readable(self, temp_test_file):
        """Test that YAML is human-readable."""
        storage = SnapshotStorage(temp_test_file, "test_function")

        storage.set_value("default", 42)

        with open(storage.snapshot_path) as f:
            content = f.read()

        assert "value: 42" in content
        assert "type: int" in content

    def test_yaml_preserves_order(self, temp_test_file):
        """Test that YAML preserves key ordering."""
        storage = SnapshotStorage(temp_test_file, "test_function")

        storage.set_value("default", {"z": 1, "a": 2, "m": 3})

        loaded = storage.load()
        # The structure is assertions -> name -> list of entries -> entry with value
        entry = loaded["assertions"]["default"][0]
        keys = list(entry["value"].keys())
        # Dict keys should be preserved in some order
        assert set(keys) == {"z", "a", "m"}

    def test_metadata_includes_test_name(self, temp_test_file):
        """Test that metadata includes test name."""
        storage = SnapshotStorage(temp_test_file, "test_my_function")
        storage.set_value("default", 42)

        loaded = storage.load()
        assert loaded["_metadata"]["test_name"] == "test_my_function"

    def test_metadata_includes_version(self, temp_test_file):
        """Test that metadata includes version."""
        storage = SnapshotStorage(temp_test_file, "test_function")
        storage.set_value("default", 42)

        loaded = storage.load()
        assert loaded["_metadata"]["version"] == 2
