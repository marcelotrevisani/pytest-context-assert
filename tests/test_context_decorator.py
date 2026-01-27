"""Tests for the @pytest_context_assert.set_context decorator."""

from __future__ import annotations

import pytest

import pytest_context_assert
from pytest_context_assert import set_context


class TestContextDecoratorImport:
    """Tests for importing the set_context decorator."""

    def test_import_from_package(self):
        """Test that set_context can be imported from the main package."""
        assert hasattr(pytest_context_assert, "set_context")
        assert callable(pytest_context_assert.set_context)

    def test_import_directly(self):
        """Test that set_context can be imported directly."""
        from pytest_context_assert import set_context

        assert callable(set_context)


class TestContextDecoratorWithDict:
    """Tests for using context decorator with a dictionary."""

    @set_context({"custom_key": "custom_value"})
    def test_dict_style_context(self, context_assert):
        """Test that dict-style context is applied."""
        assert "custom_key" in context_assert.context
        assert context_assert.context["custom_key"] == "custom_value"

    @set_context({"openblas_target": "haswell", "cpu_vendor": "intel"})
    def test_multiple_context_values(self, context_assert):
        """Test multiple context values with dict."""
        ctx = context_assert.context
        assert ctx["openblas_target"] == "haswell"
        assert ctx["cpu_vendor"] == "intel"

    @set_context({"platform": "linux", "arch": "x86_64"})
    def test_override_detected_context(self, context_assert):
        """Test that dict values override auto-detected context."""
        ctx = context_assert.context
        assert ctx["platform"] == "linux"
        assert ctx["arch"] == "x86_64"


class TestContextDecoratorWithKwargs:
    """Tests for using context decorator with keyword arguments."""

    @set_context(custom_key="custom_value")
    def test_kwargs_style_context(self, context_assert):
        """Test that kwargs-style context is applied."""
        assert "custom_key" in context_assert.context
        assert context_assert.context["custom_key"] == "custom_value"

    @set_context(openblas_target="zen3", cpu_vendor="amd")
    def test_multiple_kwargs_context(self, context_assert):
        """Test multiple context values with kwargs."""
        ctx = context_assert.context
        assert ctx["openblas_target"] == "zen3"
        assert ctx["cpu_vendor"] == "amd"


class TestContextDecoratorMixed:
    """Tests for using context decorator with both dict and kwargs."""

    @set_context({"openblas_target": "haswell"}, cpu_vendor="intel")
    def test_mixed_dict_and_kwargs(self, context_assert):
        """Test that both dict and kwargs are applied."""
        ctx = context_assert.context
        assert ctx["openblas_target"] == "haswell"
        assert ctx["cpu_vendor"] == "intel"

    @set_context({"key1": "from_dict"}, key1="from_kwargs")
    def test_kwargs_override_dict(self, context_assert):
        """Test that kwargs override dict values."""
        # kwargs should override dict values due to dict merge order
        ctx = context_assert.context
        assert ctx["key1"] == "from_kwargs"


class TestContextDecoratorWithSnapshots:
    """Tests for context decorator with actual snapshot assertions."""

    @set_context({"test_env": "ci"})
    def test_snapshot_with_custom_context(self, context_assert):
        """Test that custom context is used in snapshots."""
        # The context should include test_env=ci
        assert context_assert.context.get("test_env") == "ci"
        # Verify context_key includes the custom context
        assert "test_env_ci" in context_assert.context_key

    @set_context({"microarch": "sapphirerapids", "simd": "avx512"})
    def test_detailed_context(self, context_assert):
        """Test with detailed microarchitecture context."""
        ctx = context_assert.context
        assert ctx["microarch"] == "sapphirerapids"
        assert ctx["simd"] == "avx512"
        # Verify context_key includes both
        key = context_assert.context_key
        assert "microarch_sapphirerapids" in key
        assert "simd_avx512" in key


class TestContextDecoratorWithMarker:
    """Tests for backward compatibility with pytest markers."""

    @pytest.mark.context_assert_context(legacy_key="legacy_value")
    def test_legacy_marker_style(self, context_assert):
        """Test that legacy pytest.mark style still works."""
        ctx = context_assert.context
        assert ctx["legacy_key"] == "legacy_value"


class TestContextDecoratorEdgeCases:
    """Tests for edge cases in context decorator."""

    @set_context({})
    def test_empty_dict_context(self, context_assert):
        """Test with empty dict - should use auto-detected context."""
        # Should still have platform and arch from auto-detection
        ctx = context_assert.context
        assert "platform" in ctx
        assert "arch" in ctx

    @set_context({"value_with_special_chars": "foo-bar_baz.qux"})
    def test_special_chars_in_value(self, context_assert):
        """Test context values with special characters."""
        ctx = context_assert.context
        assert ctx["value_with_special_chars"] == "foo-bar_baz.qux"

    @set_context({"numeric_value": "123"})
    def test_numeric_string_value(self, context_assert):
        """Test context with numeric string value."""
        ctx = context_assert.context
        assert ctx["numeric_value"] == "123"
