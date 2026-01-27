"""Snapshot file management for storing and retrieving expected values."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from pytest_context_assert.serializers import deserialize_value, serialize_value

SerializeFunc = Callable[[Any], dict[str, Any]]
DeserializeFunc = Callable[[dict[str, Any]], Any]


class SnapshotStorage:
    """Manages reading and writing snapshot files."""

    SNAPSHOT_VERSION = 2

    def __init__(
        self,
        test_file: Path,
        test_name: str,
        snapshot_dir: str = "__snapshots__",
    ):
        """Initialize snapshot storage for a specific test.

        Args:
            test_file: Path to the test file.
            test_name: Name of the test function.
            snapshot_dir: Name of the snapshot directory (relative to test file).
        """
        self.test_file = Path(test_file)
        self.test_name = test_name
        self.snapshot_dir = snapshot_dir
        self._snapshot_path: Path | None = None

    @property
    def snapshot_path(self) -> Path:
        """Get the path to the snapshot file for this test."""
        if self._snapshot_path is None:
            test_dir = self.test_file.parent
            test_file_stem = self.test_file.stem
            self._snapshot_path = (
                test_dir / self.snapshot_dir / test_file_stem / f"{self.test_name}.yaml"
            )
        return self._snapshot_path

    def exists(self) -> bool:
        """Check if a snapshot file exists for this test."""
        return self.snapshot_path.exists()

    def load(self) -> dict[str, Any]:
        """Load the snapshot data from file.

        Returns:
            Dictionary containing snapshot data with 'assertions' key.

        Raises:
            FileNotFoundError: If snapshot file doesn't exist.
        """
        if not self.exists():
            raise FileNotFoundError(f"Snapshot file not found: {self.snapshot_path}")

        with open(self.snapshot_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return data or {"_metadata": {}, "assertions": {}}

    def save(self, data: dict[str, Any]) -> None:
        """Save snapshot data to file.

        Args:
            data: Dictionary containing snapshot data with 'assertions' key.
        """
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)

        if "_metadata" not in data:
            data["_metadata"] = {}

        data["_metadata"]["version"] = self.SNAPSHOT_VERSION
        data["_metadata"]["test_name"] = self.test_name

        if "created" not in data["_metadata"]:
            data["_metadata"]["created"] = datetime.now(timezone.utc).isoformat()
        data["_metadata"]["updated"] = datetime.now(timezone.utc).isoformat()

        with open(self.snapshot_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def _context_matches(
        self, stored_context: dict[str, Any], current_context: dict[str, str]
    ) -> bool:
        """Check if a stored context matches the current context.

        A stored context matches if all its keys (except 'default') match
        the corresponding keys in the current context.

        Args:
            stored_context: The __context__ dict from storage.
            current_context: The current execution context.

        Returns:
            True if the stored context matches the current context.
        """
        if stored_context.get("default"):
            return False  # Default is only used as fallback

        for key, value in stored_context.items():
            if key == "default":
                continue
            if key not in current_context:
                return False
            if current_context[key] != str(value):
                return False

        return True

    def _find_matching_entry(
        self,
        entries: list[dict[str, Any]],
        context: dict[str, str],
    ) -> dict[str, Any] | None:
        """Find the best matching entry for a given context.

        Matching priority:
        1. Exact match (all context keys match)
        2. Partial match with most specific context (most keys matching)
        3. Default entry (has __context__.default = true)

        Args:
            entries: List of assertion entries.
            context: The current execution context.

        Returns:
            The best matching entry, or None if no match found.
        """
        best_match = None
        best_match_score = -1
        default_entry = None

        for entry in entries:
            stored_ctx = entry.get("__context__", {})

            # Check for default entry
            if stored_ctx.get("default"):
                default_entry = entry
                continue

            # Calculate match score (number of matching keys)
            if self._context_matches(stored_ctx, context):
                score = len(stored_ctx)
                if score > best_match_score:
                    best_match = entry
                    best_match_score = score

        return best_match if best_match is not None else default_entry

    def get_value(
        self,
        context: dict[str, str] | str,
        name: str = "default",
        deserialize: DeserializeFunc | None = None,
    ) -> Any | None:
        """Get the expected value for a context.

        Args:
            context: The context dict or context key string.
            name: Name for the assertion (default: "default").
            deserialize: Optional custom deserialization function.

        Returns:
            The deserialized expected value, or None if not found.
        """
        if not self.exists():
            return None

        # Handle legacy string context keys
        if isinstance(context, str):
            context = self._parse_context_key(context)

        data = self.load()
        assertions = data.get("assertions", {})

        if name not in assertions:
            return None

        entries = assertions[name]
        if not isinstance(entries, list):
            # Handle legacy format
            entries = [entries]

        entry = self._find_matching_entry(entries, context)
        if entry is None:
            return None

        # Extract value data (everything except __context__)
        value_data = {k: v for k, v in entry.items() if k != "__context__"}

        if deserialize is not None:
            return deserialize(value_data)
        return deserialize_value(value_data)

    def has_value(self, context: dict[str, str] | str, name: str = "default") -> bool:
        """Check if a value exists for the given context.

        Args:
            context: The context dict or context key string.
            name: Name for the assertion (default: "default").

        Returns:
            True if a value exists (either exact match or default).
        """
        if not self.exists():
            return False

        # Handle legacy string context keys
        if isinstance(context, str):
            context = self._parse_context_key(context)

        data = self.load()
        assertions = data.get("assertions", {})

        if name not in assertions:
            return False

        entries = assertions[name]
        if not isinstance(entries, list):
            entries = [entries]

        return self._find_matching_entry(entries, context) is not None

    def set_value(
        self,
        context: dict[str, str] | str,
        value: Any,
        name: str = "default",
        serialize: SerializeFunc | None = None,
    ) -> None:
        """Store a value for a specific context.

        Args:
            context: The context dict or context key string.
            value: The value to store.
            name: Name for the assertion (default: "default").
            serialize: Optional custom serialization function.
        """
        # Handle string context keys (including "default")
        if isinstance(context, str):
            if context == "default":
                context_dict: dict[str, Any] = {"default": True}
            else:
                context_dict = self._parse_context_key(context)
        else:
            context_dict = dict(context)

        if self.exists():
            data = self.load()
        else:
            data = {"_metadata": {}, "assertions": {}}

        if "assertions" not in data:
            data["assertions"] = {}

        if serialize is not None:
            serialized = serialize(value)
        else:
            serialized = serialize_value(value)

        # Create the entry with __context__ and value data
        entry = {"__context__": context_dict, **serialized}

        # Get or create the assertion list
        if name not in data["assertions"]:
            data["assertions"][name] = []

        entries = data["assertions"][name]
        if not isinstance(entries, list):
            entries = [entries]
            data["assertions"][name] = entries

        # Find and update existing entry with same context, or append new one
        found = False
        for i, existing in enumerate(entries):
            existing_ctx = existing.get("__context__", {})
            if existing_ctx == context_dict:
                entries[i] = entry
                found = True
                break

        if not found:
            entries.append(entry)

        self.save(data)

    def delete_value(self, context: dict[str, str] | str, name: str = "default") -> bool:
        """Delete a value for a specific context.

        Args:
            context: The context dict or context key string.
            name: Name for the assertion (default: "default").

        Returns:
            True if a value was deleted, False if it didn't exist.
        """
        if not self.exists():
            return False

        # Handle string context keys
        if isinstance(context, str):
            if context == "default":
                context_dict: dict[str, Any] = {"default": True}
            else:
                context_dict = self._parse_context_key(context)
        else:
            context_dict = dict(context)

        data = self.load()
        assertions = data.get("assertions", {})

        if name not in assertions:
            return False

        entries = assertions[name]
        if not isinstance(entries, list):
            entries = [entries]

        # Find and remove entry with matching context
        for i, entry in enumerate(entries):
            if entry.get("__context__", {}) == context_dict:
                entries.pop(i)
                if not entries:
                    del assertions[name]
                else:
                    assertions[name] = entries
                self.save(data)
                return True

        return False

    def get_all_contexts(self, name: str = "default") -> list[dict[str, Any]]:
        """Get all contexts that have stored values for an assertion.

        Args:
            name: The assertion name to get contexts for.

        Returns:
            List of context dictionaries.
        """
        if not self.exists():
            return []

        data = self.load()
        assertions = data.get("assertions", {})

        if name not in assertions:
            return []

        entries = assertions[name]
        if not isinstance(entries, list):
            entries = [entries]

        return [entry.get("__context__", {}) for entry in entries]

    def _parse_context_key(self, context_key: str) -> dict[str, str]:
        """Parse a legacy context key string into a context dict.

        Args:
            context_key: String like "darwin-arm64-accelerate" or
                        "linux-x86_64-openblas-openblas_target_haswell"

        Returns:
            Context dictionary with platform, arch, etc.
        """
        if context_key == "default":
            return {}

        parts = context_key.split("-")
        context: dict[str, str] = {}

        if len(parts) >= 1:
            context["platform"] = parts[0]
        if len(parts) >= 2:
            context["arch"] = parts[1]
        if len(parts) >= 3:
            # Third part could be blas or a custom key
            third = parts[2]
            if third in ("mkl", "openblas", "accelerate", "blis"):
                context["blas"] = third
            elif "_" in third:
                key, value = third.split("_", 1)
                context[key] = value
            else:
                context["blas"] = third

        # Handle remaining parts (custom context)
        for part in parts[3:]:
            if "_" in part:
                key, value = part.split("_", 1)
                context[key] = value

        return context


def get_snapshot_storage(
    test_file: Path | str,
    test_name: str,
    snapshot_dir: str = "__snapshots__",
) -> SnapshotStorage:
    """Factory function to create a SnapshotStorage instance.

    Args:
        test_file: Path to the test file.
        test_name: Name of the test function.
        snapshot_dir: Name of the snapshot directory.

    Returns:
        SnapshotStorage instance.
    """
    return SnapshotStorage(
        test_file=Path(test_file),
        test_name=test_name,
        snapshot_dir=snapshot_dir,
    )
