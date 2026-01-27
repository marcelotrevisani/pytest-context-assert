"""Tests for dynamic context values in @set_context decorator."""

from __future__ import annotations

import os

import pytest

from pytest_context_assert import (
    build_context,
    context_from_env,
    env,
    env_or_skip,
    get_arch,
    get_blas_num_threads,
    get_ci_platform,
    get_openblas_coretype,
    get_platform,
    get_python_version,
    set_context,
)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_env_returns_value(self, monkeypatch):
        """Test env() returns environment variable value."""
        monkeypatch.setenv("TEST_VAR", "test_value")
        assert env("TEST_VAR") == "test_value"

    def test_env_returns_default(self):
        """Test env() returns default when var not set."""
        assert env("NONEXISTENT_VAR", "default") == "default"

    def test_env_returns_none_without_default(self):
        """Test env() returns None when var not set and no default."""
        assert env("NONEXISTENT_VAR") is None

    def test_env_or_skip(self, monkeypatch):
        """Test env_or_skip returns value or skip marker."""
        monkeypatch.setenv("TEST_VAR", "value")
        assert env_or_skip("TEST_VAR") == "value"
        assert env_or_skip("NONEXISTENT") == "__SKIP__"

    def test_get_platform(self):
        """Test get_platform returns lowercase platform."""
        platform = get_platform()
        assert platform in ("linux", "darwin", "windows")
        assert platform == platform.lower()

    def test_get_arch(self):
        """Test get_arch returns normalized architecture."""
        arch = get_arch()
        assert arch in ("x86_64", "arm64", "aarch64", "i386", "i686")

    def test_get_python_version(self):
        """Test get_python_version returns major.minor format."""
        version = get_python_version()
        assert "." in version
        parts = version.split(".")
        assert len(parts) == 2
        assert all(p.isdigit() for p in parts)

    def test_get_openblas_coretype(self, monkeypatch):
        """Test get_openblas_coretype."""
        monkeypatch.setenv("OPENBLAS_CORETYPE", "HASWELL")
        assert get_openblas_coretype() == "haswell"

        monkeypatch.delenv("OPENBLAS_CORETYPE", raising=False)
        assert get_openblas_coretype() is None

    def test_get_blas_num_threads(self, monkeypatch):
        """Test get_blas_num_threads checks multiple vars."""
        # Clear all
        for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
            monkeypatch.delenv(var, raising=False)

        assert get_blas_num_threads() is None

        monkeypatch.setenv("OMP_NUM_THREADS", "4")
        assert get_blas_num_threads() == "4"

        monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")
        assert get_blas_num_threads() == "8"  # OPENBLAS takes precedence

    def test_get_ci_platform(self, monkeypatch):
        """Test CI platform detection."""
        # Clear CI env vars
        for var in ("GITHUB_ACTIONS", "GITLAB_CI", "JENKINS_URL", "CIRCLECI", "TRAVIS"):
            monkeypatch.delenv(var, raising=False)

        assert get_ci_platform() is None

        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        assert get_ci_platform() == "github"

    def test_context_from_env(self, monkeypatch):
        """Test context_from_env builds dict from env vars."""
        monkeypatch.setenv("OPENBLAS_CORETYPE", "HASWELL")
        monkeypatch.setenv("OPENBLAS_NUM_THREADS", "4")

        ctx = context_from_env(
            "OPENBLAS_CORETYPE",
            "OPENBLAS_NUM_THREADS",
            "NONEXISTENT_VAR",
            prefix="OPENBLAS_",
        )

        assert ctx == {"coretype": "HASWELL", "num_threads": "4"}
        assert "nonexistent_var" not in ctx

    def test_build_context(self, monkeypatch):
        """Test build_context creates comprehensive context."""
        monkeypatch.setenv("MY_VAR", "my_value")

        ctx = build_context(
            include_platform=True,
            include_arch=True,
            include_python=True,
            env_vars=["MY_VAR"],
            custom_key="custom_value",
        )

        assert "platform" in ctx
        assert "arch" in ctx
        assert "python" in ctx
        assert ctx["my_var"] == "my_value"
        assert ctx["custom_key"] == "custom_value"


class TestDynamicContextDecorator:
    """Tests for using dynamic values in @set_context decorator."""

    @set_context({"platform": get_platform(), "arch": get_arch()})
    def test_platform_and_arch_from_helpers(self, context_assert):
        """Test using get_platform() and get_arch() in decorator."""
        ctx = context_assert.context
        assert ctx["platform"] == get_platform()
        assert ctx["arch"] == get_arch()

    @set_context({"python_version": get_python_version()})
    def test_python_version_in_context(self, context_assert):
        """Test using get_python_version() in decorator."""
        ctx = context_assert.context
        assert ctx["python_version"] == get_python_version()


class TestDynamicContextWithEnvVars:
    """Tests for dynamic context from environment variables."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Set up test environment variables."""
        monkeypatch.setenv("TEST_OPENBLAS_CORETYPE", "haswell")
        monkeypatch.setenv("TEST_NUM_THREADS", "4")

    def test_env_helper_in_decorator(self, context_assert, monkeypatch):
        """Test using env() helper - must be evaluated at runtime."""
        # Since decorator is evaluated at import time, we need to
        # demonstrate runtime context modification via with_context
        monkeypatch.setenv("RUNTIME_VAR", "runtime_value")

        ctx_assert = context_assert.with_context(
            runtime_var=os.environ.get("RUNTIME_VAR", "default")
        )
        assert ctx_assert.context["runtime_var"] == "runtime_value"

    def test_context_from_env_helper(self, context_assert, monkeypatch):
        """Test building context from multiple env vars at runtime."""
        monkeypatch.setenv("BLAS_TYPE", "openblas")
        monkeypatch.setenv("BLAS_TARGET", "zen3")

        # Use with_context for runtime evaluation
        env_context = context_from_env("BLAS_TYPE", "BLAS_TARGET")
        ctx_assert = context_assert.with_context(**env_context)

        assert ctx_assert.context["blas_type"] == "openblas"
        assert ctx_assert.context["blas_target"] == "zen3"


class TestBuildContextDecorator:
    """Tests for build_context helper in decorator."""

    @set_context(build_context(include_platform=True, include_arch=True))
    def test_build_context_basic(self, context_assert):
        """Test build_context with basic options."""
        ctx = context_assert.context
        assert "platform" in ctx
        assert "arch" in ctx

    @set_context(
        build_context(
            include_platform=True, include_arch=True, include_python=True, custom_key="test_value"
        )
    )
    def test_build_context_with_extras(self, context_assert):
        """Test build_context with extra key-value pairs."""
        ctx = context_assert.context
        assert ctx["custom_key"] == "test_value"
        assert "python" in ctx


class TestRealWorldScenarios:
    """Tests simulating real-world usage scenarios."""

    @set_context(
        {
            "scenario": "cpu_specific",
            "platform": get_platform(),
            "arch": get_arch(),
        }
    )
    def test_cpu_specific_context(self, context_assert):
        """Simulate CPU-specific test context."""
        ctx = context_assert.context
        assert ctx["scenario"] == "cpu_specific"
        # The context key should include all specified values
        key = context_assert.context_key
        assert "scenario_cpu_specific" in key
        # Platform is included in the base key format (e.g., "darwin-arm64-...")
        assert get_platform() in key

    def test_openblas_coretype_scenario(self, context_assert, monkeypatch):
        """Simulate OpenBLAS coretype-specific testing."""
        # Simulate different OpenBLAS configurations
        monkeypatch.setenv("OPENBLAS_CORETYPE", "SAPPHIRERAPIDS")

        coretype = get_openblas_coretype()
        ctx_assert = context_assert.with_context(openblas_coretype=coretype)

        assert ctx_assert.context["openblas_coretype"] == "sapphirerapids"
        assert "openblas_coretype_sapphirerapids" in ctx_assert.context_key

    def test_ci_specific_context(self, context_assert, monkeypatch):
        """Simulate CI-specific test context."""
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.setenv("GITHUB_RUN_ID", "12345")

        ci = get_ci_platform()
        ctx_assert = context_assert.with_context(
            ci_platform=ci,
            ci_run_id=os.environ.get("GITHUB_RUN_ID"),
        )

        assert ctx_assert.context["ci_platform"] == "github"
        assert ctx_assert.context["ci_run_id"] == "12345"


class TestConditionalContext:
    """Tests for conditionally applying context."""

    def test_conditional_env_context(self, context_assert, monkeypatch):
        """Test conditionally adding context based on env vars."""
        # Only add openblas_coretype if it's set
        monkeypatch.setenv("OPENBLAS_CORETYPE", "ZEN3")

        coretype = get_openblas_coretype()
        if coretype:
            ctx_assert = context_assert.with_context(openblas_coretype=coretype)
            assert "openblas_coretype" in ctx_assert.context
        else:
            assert "openblas_coretype" not in context_assert.context

    def test_platform_specific_context(self, context_assert):
        """Test adding platform-specific context."""
        platform = get_platform()

        if platform == "darwin":
            ctx_assert = context_assert.with_context(
                macos_version=os.environ.get("MACOS_VERSION", "unknown")
            )
        elif platform == "linux":
            ctx_assert = context_assert.with_context(
                linux_distro=os.environ.get("LINUX_DISTRO", "unknown")
            )
        else:
            ctx_assert = context_assert

        # Just verify the context is accessible
        assert "platform" in ctx_assert.context
