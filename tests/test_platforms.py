"""Tests for platform simulation and context-based value selection."""

from __future__ import annotations

from pytest_context_assert.context import ContextResolver
from pytest_context_assert.storage import SnapshotStorage


class TestPlatformDetection:
    """Tests for platform detection with mocked values."""

    def test_linux_platform(self, mock_platform):
        """Test Linux platform detection."""
        mock_platform(system="Linux", machine="x86_64")
        resolver = ContextResolver()
        assert resolver.get_platform() == "linux"

    def test_darwin_platform(self, mock_platform):
        """Test macOS platform detection."""
        mock_platform(system="Darwin", machine="arm64")
        resolver = ContextResolver()
        assert resolver.get_platform() == "darwin"

    def test_windows_platform(self, mock_platform):
        """Test Windows platform detection."""
        mock_platform(system="Windows", machine="AMD64")
        resolver = ContextResolver()
        assert resolver.get_platform() == "windows"

    def test_freebsd_platform(self, mock_platform):
        """Test FreeBSD platform detection (falls through)."""
        mock_platform(system="FreeBSD", machine="amd64")
        resolver = ContextResolver()
        assert resolver.get_platform() == "freebsd"


class TestArchitectureDetection:
    """Tests for CPU architecture detection."""

    def test_x86_64_architecture(self, mock_platform):
        """Test x86_64 architecture detection."""
        mock_platform(system="Linux", machine="x86_64")
        resolver = ContextResolver()
        assert resolver.get_architecture() == "x86_64"

    def test_amd64_maps_to_x86_64(self, mock_platform):
        """Test AMD64 maps to x86_64."""
        mock_platform(system="Windows", machine="AMD64")
        resolver = ContextResolver()
        assert resolver.get_architecture() == "x86_64"

    def test_arm64_architecture(self, mock_platform):
        """Test ARM64 architecture detection."""
        mock_platform(system="Darwin", machine="arm64")
        resolver = ContextResolver()
        assert resolver.get_architecture() == "arm64"

    def test_aarch64_maps_to_arm64(self, mock_platform):
        """Test aarch64 maps to arm64."""
        mock_platform(system="Linux", machine="aarch64")
        resolver = ContextResolver()
        assert resolver.get_architecture() == "arm64"

    def test_i686_maps_to_x86(self, mock_platform):
        """Test i686 maps to x86."""
        mock_platform(system="Linux", machine="i686")
        resolver = ContextResolver()
        assert resolver.get_architecture() == "x86"

    def test_i386_maps_to_x86(self, mock_platform):
        """Test i386 maps to x86."""
        mock_platform(system="Linux", machine="i386")
        resolver = ContextResolver()
        assert resolver.get_architecture() == "x86"

    def test_unknown_architecture_passthrough(self, mock_platform):
        """Test unknown architecture passes through."""
        mock_platform(system="Linux", machine="riscv64")
        resolver = ContextResolver()
        assert resolver.get_architecture() == "riscv64"


class TestContextKeyGeneration:
    """Tests for context key generation with different platforms."""

    def test_linux_x86_64_context_key(self, mock_platform, mock_blas):
        """Test context key for Linux x86_64."""
        mock_platform(system="Linux", machine="x86_64")
        mock_blas(None)
        resolver = ContextResolver()
        assert resolver.get_context_key() == "linux-x86_64"

    def test_darwin_arm64_context_key(self, mock_platform, mock_blas):
        """Test context key for macOS ARM64."""
        mock_platform(system="Darwin", machine="arm64")
        mock_blas(None)
        resolver = ContextResolver()
        assert resolver.get_context_key() == "darwin-arm64"

    def test_windows_x86_64_context_key(self, mock_platform, mock_blas):
        """Test context key for Windows x86_64."""
        mock_platform(system="Windows", machine="AMD64")
        mock_blas(None)
        resolver = ContextResolver()
        assert resolver.get_context_key() == "windows-x86_64"

    def test_linux_arm64_with_blas_context_key(self, mock_platform, mock_blas):
        """Test context key with BLAS included."""
        mock_platform(system="Linux", machine="aarch64")
        mock_blas("openblas")
        resolver = ContextResolver()
        assert resolver.get_context_key() == "linux-arm64-openblas"


class TestPlatformSpecificSnapshots:
    """Tests for platform-specific snapshot retrieval."""

    def test_platform_specific_value(self, temp_test_file, mock_platform, mock_blas):
        """Test that platform-specific values are retrieved correctly."""
        storage = SnapshotStorage(temp_test_file, "test_function")

        # Store different values for different platforms
        storage.set_value("linux-x86_64", 1.0)
        storage.set_value("darwin-arm64", 1.1)
        storage.set_value("windows-x86_64", 1.2)
        storage.set_value("default", 0.9)

        # Test Linux retrieval
        mock_platform(system="Linux", machine="x86_64")
        mock_blas(None)
        resolver = ContextResolver()
        value = storage.get_value(resolver.get_context_key())
        assert value == 1.0

        # Reset cache for new context
        resolver._cached_context = None

    def test_fallback_to_default(self, temp_test_file, mock_platform, mock_blas):
        """Test fallback to default when specific context not found."""
        storage = SnapshotStorage(temp_test_file, "test_function")

        storage.set_value("linux-x86_64", 1.0)
        storage.set_value("default", 0.9)

        # Request FreeBSD context which doesn't exist
        mock_platform(system="FreeBSD", machine="amd64")
        mock_blas(None)
        resolver = ContextResolver()
        value = storage.get_value(resolver.get_context_key())
        assert value == 0.9  # Falls back to default


class TestCrossPlatformSnapshots:
    """Tests for cross-platform snapshot scenarios."""

    def test_store_multiple_platforms(self, temp_test_file):
        """Test storing snapshots for multiple platforms."""
        storage = SnapshotStorage(temp_test_file, "test_cross_platform")

        # Simulate results from different platforms
        platform_results = {
            "linux-x86_64": 1.0000000001,
            "linux-x86_64-mkl": 1.0000000002,
            "linux-x86_64-openblas": 1.0000000003,
            "darwin-arm64": 1.0000000010,
            "darwin-arm64-accelerate": 1.0000000011,
            "darwin-x86_64": 1.0000000020,
            "windows-x86_64": 1.0000000030,
            "windows-x86_64-mkl": 1.0000000031,
            "default": 1.0,
        }

        for context, value in platform_results.items():
            storage.set_value(context, value)

        # Verify each can be retrieved
        for context, expected in platform_results.items():
            actual = storage.get_value(context)
            assert actual == expected, f"Mismatch for {context}"

    def test_partial_platform_coverage(self, temp_test_file, mock_platform, mock_blas):
        """Test when only some platforms have specific values."""
        storage = SnapshotStorage(temp_test_file, "test_partial")

        # Only Linux has a specific value
        storage.set_value("linux-x86_64", 42)
        storage.set_value("default", 0)

        # Linux should get specific value
        mock_platform(system="Linux", machine="x86_64")
        mock_blas(None)
        resolver = ContextResolver()
        assert storage.get_value(resolver.get_context_key()) == 42

        # macOS should fall back to default
        mock_platform(system="Darwin", machine="arm64")
        resolver = ContextResolver()
        resolver._cached_context = None
        assert storage.get_value(resolver.get_context_key()) == 0

        # Windows should fall back to default
        mock_platform(system="Windows", machine="AMD64")
        resolver = ContextResolver()
        resolver._cached_context = None
        assert storage.get_value(resolver.get_context_key()) == 0


class TestPlatformContextMatching:
    """Tests for context matching across platforms."""

    def test_matches_own_context(self, mock_platform, mock_blas):
        """Test that resolver matches its own context key."""
        mock_platform(system="Linux", machine="x86_64")
        mock_blas("openblas")
        resolver = ContextResolver()
        key = resolver.get_context_key()
        assert resolver.matches_context(key)

    def test_does_not_match_other_platform(self, mock_platform, mock_blas):
        """Test that resolver doesn't match other platform's context."""
        mock_platform(system="Linux", machine="x86_64")
        mock_blas(None)
        resolver = ContextResolver()
        assert not resolver.matches_context("darwin-arm64")
        assert not resolver.matches_context("windows-x86_64")

    def test_does_not_match_partial_context(self, mock_platform, mock_blas):
        """Test that partial matches don't work."""
        mock_platform(system="Linux", machine="x86_64")
        mock_blas(None)
        resolver = ContextResolver()
        assert not resolver.matches_context("linux")
        assert not resolver.matches_context("x86_64")
