"""Advanced syrupy-style API tests - comprehensive snapshot testing scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pytest

from pytest_context_assert import set_context, build_context
from pytest_context_assert.fixture import ContextAssertionError, MissingSnapshotError


# ============================================================================
# Helper Classes
# ============================================================================


@dataclass
class APIResponse:
    """Simulated API response."""

    status: int
    data: dict[str, Any]
    timestamp: datetime | None = None


@dataclass
class UserProfile:
    """User profile for testing."""

    username: str
    email: str
    roles: list[str]
    settings: dict[str, Any]


# ============================================================================
# Syrupy-Style Tests
# ============================================================================


class TestAssertMatchChaining:
    """Tests for chaining multiple assert_match calls."""

    def test_multiple_assertions_same_test(self, context_assert):
        """Test multiple assertions in the same test."""
        context_assert._update_snapshots = True
        context_assert.assert_match(1, name="first")
        context_assert.assert_match("hello", name="second")
        context_assert.assert_match([1, 2, 3], name="third")
        context_assert.assert_match({"key": "value"}, name="fourth")

        context_assert._update_snapshots = False
        context_assert._assertion_count = 0  # Reset for comparison
        context_assert.assert_match(1, name="first")
        context_assert.assert_match("hello", name="second")
        context_assert.assert_match([1, 2, 3], name="third")
        context_assert.assert_match({"key": "value"}, name="fourth")

    def test_auto_naming_sequence(self, context_assert):
        """Test auto-naming for sequential assertions."""
        context_assert._update_snapshots = True
        for i in range(5):
            context_assert.assert_match(i * 10)

        context_assert._assertion_count = 0
        context_assert._update_snapshots = False
        for i in range(5):
            context_assert.assert_match(i * 10)

    def test_mixed_named_and_auto(self, context_assert):
        """Test mixing named and auto-named assertions."""
        context_assert._update_snapshots = True
        context_assert.assert_match(1, name="explicit_first")
        context_assert.assert_match(2)  # auto: assertion_1
        context_assert.assert_match(3, name="explicit_second")
        context_assert.assert_match(4)  # auto: assertion_2

        context_assert._assertion_count = 0
        context_assert._update_snapshots = False
        context_assert.assert_match(1, name="explicit_first")
        context_assert.assert_match(2)
        context_assert.assert_match(3, name="explicit_second")
        context_assert.assert_match(4)


class TestAssertMatchWithTolerance:
    """Tests for assert_match with various tolerance scenarios."""

    def test_rtol_only(self, context_assert):
        """Test with relative tolerance only."""
        context_assert._update_snapshots = True
        context_assert.assert_match(1.0, name="rtol_test")

        context_assert._update_snapshots = False
        context_assert.assert_match(1.000001, name="rtol_test", rtol=1e-5)

    def test_atol_only(self, context_assert):
        """Test with absolute tolerance only."""
        context_assert._update_snapshots = True
        context_assert.assert_match(0.0, name="atol_test")

        context_assert._update_snapshots = False
        context_assert.assert_match(0.00001, name="atol_test", atol=1e-4)

    def test_rtol_and_atol(self, context_assert):
        """Test with both tolerances."""
        context_assert._update_snapshots = True
        context_assert.assert_match(1.0, name="both_tol")

        context_assert._update_snapshots = False
        context_assert.assert_match(1.0001, name="both_tol", rtol=1e-3, atol=1e-4)

    def test_tolerance_with_list(self, context_assert):
        """Test tolerance with list of floats."""
        context_assert._update_snapshots = True
        context_assert.assert_match([1.0, 2.0, 3.0], name="list_tol")

        context_assert._update_snapshots = False
        context_assert.assert_match([1.000001, 2.000001, 3.000001], name="list_tol", rtol=1e-5)

    def test_tolerance_fails_when_exceeded(self, context_assert):
        """Test that comparison fails when tolerance is exceeded."""
        context_assert._update_snapshots = True
        context_assert.assert_match(1.0, name="exceed_tol")

        context_assert._update_snapshots = False
        with pytest.raises(ContextAssertionError):
            context_assert.assert_match(1.1, name="exceed_tol", rtol=1e-5)


class TestAssertMatchWithCustomCompare:
    """Tests for assert_match with custom comparison functions."""

    def test_case_insensitive_compare(self, context_assert):
        """Test case-insensitive string comparison."""
        context_assert._update_snapshots = True
        context_assert.assert_match("Hello World", name="case_insensitive")

        context_assert._update_snapshots = False
        context_assert.assert_match(
            "HELLO WORLD",
            name="case_insensitive",
            compare=lambda a, b: a.lower() == b.lower(),
        )

    def test_substring_compare(self, context_assert):
        """Test substring comparison."""
        context_assert._update_snapshots = True
        context_assert.assert_match("important message", name="substring")

        context_assert._update_snapshots = False
        context_assert.assert_match(
            "The important message is here",
            name="substring",
            compare=lambda a, b: b in a,
        )

    def test_length_only_compare(self, context_assert):
        """Test comparison based on length only."""
        context_assert._update_snapshots = True
        context_assert.assert_match([1, 2, 3, 4, 5], name="length_only")

        context_assert._update_snapshots = False
        context_assert.assert_match(
            ["a", "b", "c", "d", "e"],
            name="length_only",
            compare=lambda a, b: len(a) == len(b),
        )

    def test_key_subset_compare(self, context_assert):
        """Test dictionary comparison with key subset."""
        context_assert._update_snapshots = True
        context_assert.assert_match({"important": 1, "also_important": 2}, name="key_subset")

        context_assert._update_snapshots = False
        context_assert.assert_match(
            {"important": 1, "also_important": 2, "extra": 3, "more_extra": 4},
            name="key_subset",
            compare=lambda a, b: all(a.get(k) == v for k, v in b.items()),
        )

    def test_numeric_range_compare(self, context_assert):
        """Test numeric range comparison."""
        context_assert._update_snapshots = True
        context_assert.assert_match(50, name="range_compare")

        context_assert._update_snapshots = False

        def in_range(actual, expected, margin=10):
            return expected - margin <= actual <= expected + margin

        context_assert.assert_match(55, name="range_compare", compare=lambda a, b: in_range(a, b))


class TestAssertMatchWithSerialization:
    """Tests for assert_match with custom serialization."""

    def test_api_response_serialization(self, context_assert):
        """Test serializing API response objects."""
        response = APIResponse(
            status=200,
            data={"users": [{"id": 1, "name": "Alice"}]},
            timestamp=datetime(2026, 1, 20, 12, 0, 0),
        )

        def serialize(r):
            return {
                "status": r.status,
                "data": r.data,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "type": "APIResponse",
            }

        def deserialize(d):
            return APIResponse(
                status=d["status"],
                data=d["data"],
                timestamp=datetime.fromisoformat(d["timestamp"]) if d["timestamp"] else None,
            )

        context_assert._update_snapshots = True
        context_assert.assert_match(response, name="api_response", serialize=serialize)

        context_assert._update_snapshots = False
        context_assert.assert_match(
            response,
            name="api_response",
            deserialize=deserialize,
            compare=lambda a, b: a.status == b.status and a.data == b.data,
        )

    def test_user_profile_serialization(self, context_assert):
        """Test serializing user profile objects."""
        profile = UserProfile(
            username="testuser",
            email="test@example.com",
            roles=["admin", "user"],
            settings={"theme": "dark", "notifications": True},
        )

        def serialize(p):
            return {
                "username": p.username,
                "email": p.email,
                "roles": p.roles,
                "settings": p.settings,
                "type": "UserProfile",
            }

        def deserialize(d):
            return UserProfile(
                username=d["username"],
                email=d["email"],
                roles=d["roles"],
                settings=d["settings"],
            )

        context_assert._update_snapshots = True
        context_assert.assert_match(profile, name="user_profile", serialize=serialize)

        context_assert._update_snapshots = False
        context_assert.assert_match(
            profile,
            name="user_profile",
            deserialize=deserialize,
            compare=lambda a, b: a == b,
        )

    def test_partial_serialization(self, context_assert):
        """Test serializing only specific fields."""
        data = {
            "id": 123,
            "name": "Test",
            "created_at": datetime.now().isoformat(),  # Dynamic field
            "random_value": 42,  # Changes each run
        }

        def serialize_stable(d):
            # Only serialize stable fields
            return {"id": d["id"], "name": d["name"], "type": "partial"}

        def deserialize_stable(stored):
            return {"id": stored["id"], "name": stored["name"]}

        context_assert._update_snapshots = True
        context_assert.assert_match(data, name="partial", serialize=serialize_stable)

        # Different dynamic values should still match
        data2 = {
            "id": 123,
            "name": "Test",
            "created_at": "different",
            "random_value": 999,
        }

        context_assert._update_snapshots = False
        context_assert.assert_match(
            data2,
            name="partial",
            serialize=serialize_stable,
            deserialize=deserialize_stable,
            compare=lambda a, b: a["id"] == b["id"] and a["name"] == b["name"],
        )


class TestAssertMatchWithContext:
    """Tests for assert_match with context decorators."""

    @set_context({"env": "test", "version": "1.0"})
    def test_with_static_context(self, context_assert):
        """Test assert_match with static context."""
        context_assert._update_snapshots = True
        context_assert.assert_match({"result": "success"}, name="static_ctx")

        context_assert._update_snapshots = False
        context_assert.assert_match({"result": "success"}, name="static_ctx")

    @set_context(build_context(include_platform=True, include_arch=True))
    def test_with_dynamic_context(self, context_assert):
        """Test assert_match with dynamically built context."""
        context_assert._update_snapshots = True
        context_assert.assert_match({"platform_test": True}, name="dynamic_ctx")

        context_assert._update_snapshots = False
        context_assert.assert_match({"platform_test": True}, name="dynamic_ctx")

    def test_with_runtime_context(self, context_assert):
        """Test assert_match with runtime context via with_context."""
        ctx = context_assert.with_context(runtime_key="runtime_value")

        ctx._update_snapshots = True
        ctx.assert_match({"data": 123}, name="runtime_ctx")

        ctx._update_snapshots = False
        ctx.assert_match({"data": 123}, name="runtime_ctx")


class TestUpdateAndSetDefault:
    """Tests for update() and set_default() methods."""

    def test_explicit_update(self, context_assert):
        """Test explicit update() method."""
        context_assert.update({"key": "value"}, name="explicit_update")

        # Should be able to retrieve it
        assert context_assert.storage.has_value(context_assert.context, name="explicit_update")

    def test_set_default(self, context_assert):
        """Test set_default() method."""
        context_assert.set_default(42, name="default_value")

        # Should be stored under 'default' context
        assert context_assert.storage.has_value("default", name="default_value")

    def test_update_with_specific_context(self, context_assert):
        """Test update() with specific context."""
        context_assert.update(
            {"platform_specific": True},
            name="specific_ctx",
            context={"platform": "linux", "arch": "x86_64"},
        )

        # Verify stored
        assert context_assert.storage.has_value(
            {"platform": "linux", "arch": "x86_64"}, name="specific_ctx"
        )


class TestFailureScenarios:
    """Tests for various failure scenarios."""

    def test_missing_snapshot_error(self, context_assert, tmp_path):
        """Test MissingSnapshotError is raised correctly."""
        # Don't update snapshots, so it should fail
        context_assert._update_snapshots = False

        # Use a unique name that won't exist
        with pytest.raises(MissingSnapshotError) as exc_info:
            context_assert.assert_match("value", name="nonexistent_snapshot_12345")

        # Check that it mentions the context or snapshot path
        error_str = str(exc_info.value)
        assert "No snapshot found" in error_str or "context" in error_str.lower()

    def test_assertion_error_includes_details(self, context_assert):
        """Test that assertion errors include useful details."""
        context_assert._update_snapshots = True
        context_assert.assert_match({"expected": "value"}, name="detail_test")

        context_assert._update_snapshots = False
        with pytest.raises(ContextAssertionError) as exc_info:
            context_assert.assert_match({"expected": "different"}, name="detail_test")

        error = exc_info.value
        assert error.actual == {"expected": "different"}
        assert error.expected == {"expected": "value"}

    def test_type_mismatch_error(self, context_assert):
        """Test error on type mismatch."""
        context_assert._update_snapshots = True
        context_assert.assert_match(42, name="type_mismatch")

        context_assert._update_snapshots = False
        with pytest.raises(ContextAssertionError):
            context_assert.assert_match("42", name="type_mismatch")


class TestComplexWorkflows:
    """Tests for complex real-world workflows."""

    def test_api_response_workflow(self, context_assert):
        """Test complete API response testing workflow."""
        # Simulate API responses
        responses = [
            {"status": 200, "data": {"users": []}},
            {"status": 201, "data": {"id": 1, "message": "Created"}},
            {"status": 400, "data": {"error": "Bad Request"}},
        ]

        context_assert._update_snapshots = True
        for i, resp in enumerate(responses):
            context_assert.assert_match(resp, name=f"api_response_{i}")

        context_assert._assertion_count = 0
        context_assert._update_snapshots = False
        for i, resp in enumerate(responses):
            context_assert.assert_match(resp, name=f"api_response_{i}")

    def test_data_pipeline_workflow(self, context_assert):
        """Test data pipeline stages."""
        # Stage 1: Raw data
        raw_data = [{"id": 1, "value": 10}, {"id": 2, "value": 20}]

        # Stage 2: Transformed
        transformed = [{"id": i["id"], "value": i["value"] * 2} for i in raw_data]

        # Stage 3: Aggregated
        aggregated = {"total": sum(i["value"] for i in transformed), "count": len(transformed)}

        context_assert._update_snapshots = True
        context_assert.assert_match(raw_data, name="pipeline_raw")
        context_assert.assert_match(transformed, name="pipeline_transformed")
        context_assert.assert_match(aggregated, name="pipeline_aggregated")

        context_assert._update_snapshots = False
        context_assert.assert_match(raw_data, name="pipeline_raw")
        context_assert.assert_match(transformed, name="pipeline_transformed")
        context_assert.assert_match(aggregated, name="pipeline_aggregated")

    def test_configuration_validation_workflow(self, context_assert):
        """Test configuration validation workflow."""
        config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "testdb",
            },
            "cache": {
                "enabled": True,
                "ttl": 3600,
            },
            "features": ["feature_a", "feature_b"],
        }

        # Validate each section
        context_assert._update_snapshots = True
        context_assert.assert_match(config["database"], name="config_database")
        context_assert.assert_match(config["cache"], name="config_cache")
        context_assert.assert_match(config["features"], name="config_features")

        context_assert._update_snapshots = False
        context_assert.assert_match(config["database"], name="config_database")
        context_assert.assert_match(config["cache"], name="config_cache")
        context_assert.assert_match(config["features"], name="config_features")

    def test_multi_format_output_workflow(self, context_assert):
        """Test validating output in multiple formats."""
        data = {"name": "Test", "value": 42, "items": [1, 2, 3]}

        # Different output formats
        json_output = data
        list_output = [data["name"], data["value"], data["items"]]
        summary_output = f"Name: {data['name']}, Value: {data['value']}"

        context_assert._update_snapshots = True
        context_assert.assert_match(json_output, name="format_json")
        context_assert.assert_match(list_output, name="format_list")
        context_assert.assert_match(summary_output, name="format_summary")

        context_assert._update_snapshots = False
        context_assert.assert_match(json_output, name="format_json")
        context_assert.assert_match(list_output, name="format_list")
        context_assert.assert_match(summary_output, name="format_summary")
