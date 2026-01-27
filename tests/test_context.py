"""Tests for ContextResolver."""

from __future__ import annotations

from pytest_context_assert.context import ContextResolver


class TestContextResolver:
    """Tests for basic ContextResolver functionality."""

    def test_get_platform(self):
        resolver = ContextResolver()
        plat = resolver.get_platform()
        assert plat in ("linux", "darwin", "windows") or isinstance(plat, str)

    def test_get_architecture(self):
        resolver = ContextResolver()
        arch = resolver.get_architecture()
        assert arch in ("x86_64", "arm64", "x86") or isinstance(arch, str)

    def test_get_context_key(self):
        resolver = ContextResolver()
        key = resolver.get_context_key()
        assert isinstance(key, str)
        assert "-" in key

    def test_custom_context(self):
        resolver = ContextResolver(custom_context={"custom_key": "custom_value"})
        context = resolver.get_context()
        assert "custom_key" in context
        assert context["custom_key"] == "custom_value"

    def test_with_custom_context(self):
        resolver = ContextResolver()
        new_resolver = resolver.with_custom_context(extra="value")
        context = new_resolver.get_context()
        assert "extra" in context

    def test_matches_context(self):
        resolver = ContextResolver()
        key = resolver.get_context_key()
        assert resolver.matches_context(key)
        assert not resolver.matches_context("nonexistent-context")

    def test_get_python_version(self):
        resolver = ContextResolver()
        version = resolver.get_python_version()
        assert "." in version
        parts = version.split(".")
        assert len(parts) == 2
        assert int(parts[0]) >= 3

    def test_context_caching(self):
        """Test that context is cached after first call."""
        resolver = ContextResolver()
        context1 = resolver.get_context()
        context2 = resolver.get_context()
        assert context1 is context2

    def test_multiple_custom_context_values(self):
        """Test multiple custom context values."""
        resolver = ContextResolver(
            custom_context={"blas_target": "haswell", "compiler": "gcc", "version": "12"}
        )
        context = resolver.get_context()
        assert context["blas_target"] == "haswell"
        assert context["compiler"] == "gcc"
        assert context["version"] == "12"

    def test_custom_context_in_key(self):
        """Test that custom context appears in context key."""
        resolver = ContextResolver(custom_context={"custom": "value"})
        key = resolver.get_context_key()
        assert "custom_value" in key

    def test_with_custom_context_preserves_original(self):
        """Test that with_custom_context doesn't modify original."""
        resolver = ContextResolver(custom_context={"a": "1"})
        new_resolver = resolver.with_custom_context(b="2")

        original_context = resolver.get_context()
        new_context = new_resolver.get_context()

        assert "b" not in original_context
        assert "a" in new_context
        assert "b" in new_context

    def test_with_custom_context_overrides(self):
        """Test that with_custom_context can override values."""
        resolver = ContextResolver(custom_context={"key": "original"})
        new_resolver = resolver.with_custom_context(key="overridden")

        assert new_resolver.get_context()["key"] == "overridden"
