"""Shared fixtures for pytest-context-assert tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def numpy():
    """Fixture to skip tests if numpy is not available."""
    pytest.importorskip("numpy")
    import numpy as np

    return np


@pytest.fixture
def mock_platform(monkeypatch):
    """Factory fixture for mocking platform detection.

    Usage:
        def test_linux(mock_platform):
            mock_platform(system="Linux", machine="x86_64")
    """

    def _mock_platform(system: str = "Darwin", machine: str = "arm64"):
        monkeypatch.setattr("platform.system", lambda: system)
        monkeypatch.setattr("platform.machine", lambda: machine)

    return _mock_platform


@pytest.fixture
def mock_blas(monkeypatch):
    """Factory fixture for mocking BLAS detection.

    Usage:
        def test_mkl(mock_blas):
            mock_blas("mkl")
    """

    def _mock_blas(blas_name: str | None):
        def fake_detect_blas():
            return blas_name

        monkeypatch.setattr(
            "pytest_context_assert.context._detect_blas",
            fake_detect_blas,
        )

    return _mock_blas


@pytest.fixture
def temp_test_file(tmp_path):
    """Create a temporary test file for storage tests."""
    test_file = tmp_path / "test_example.py"
    test_file.touch()
    return test_file
