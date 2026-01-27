"""Tests for BLAS library-specific contexts (OpenBLAS, MKL, Accelerate, BLIS)."""

from __future__ import annotations

import pytest

from pytest_context_assert.context import ContextResolver
from pytest_context_assert.storage import SnapshotStorage


class TestBLASDetection:
    """Tests for BLAS library detection."""

    def test_mkl_blas_context(self, mock_platform, mock_blas):
        """Test context with Intel MKL."""
        mock_platform(system="Linux", machine="x86_64")
        mock_blas("mkl")
        resolver = ContextResolver()
        assert resolver.get_blas() == "mkl"
        assert resolver.get_context_key() == "linux-x86_64-mkl"

    def test_openblas_context(self, mock_platform, mock_blas):
        """Test context with OpenBLAS."""
        mock_platform(system="Linux", machine="x86_64")
        mock_blas("openblas")
        resolver = ContextResolver()
        assert resolver.get_blas() == "openblas"
        assert resolver.get_context_key() == "linux-x86_64-openblas"

    def test_accelerate_context(self, mock_platform, mock_blas):
        """Test context with Apple Accelerate."""
        mock_platform(system="Darwin", machine="arm64")
        mock_blas("accelerate")
        resolver = ContextResolver()
        assert resolver.get_blas() == "accelerate"
        assert resolver.get_context_key() == "darwin-arm64-accelerate"

    def test_blis_context(self, mock_platform, mock_blas):
        """Test context with BLIS."""
        mock_platform(system="Linux", machine="x86_64")
        mock_blas("blis")
        resolver = ContextResolver()
        assert resolver.get_blas() == "blis"
        assert resolver.get_context_key() == "linux-x86_64-blis"

    def test_no_blas_context(self, mock_platform, mock_blas):
        """Test context without BLAS (numpy not installed or no BLAS)."""
        mock_platform(system="Linux", machine="x86_64")
        mock_blas(None)
        resolver = ContextResolver()
        assert resolver.get_blas() is None
        assert resolver.get_context_key() == "linux-x86_64"


class TestOpenBLASTargets:
    """Tests for OpenBLAS target-specific contexts (Haswell, Sapphire Rapids, etc.)."""

    @pytest.mark.parametrize(
        "target,description",
        [
            ("haswell", "Intel Haswell / 4th Gen Core"),
            ("skylakex", "Intel Skylake-X / Xeon Scalable"),
            ("cascadelake", "Intel Cascade Lake"),
            ("cooperlake", "Intel Cooper Lake"),
            ("sapphirerapids", "Intel Sapphire Rapids"),
            ("zen", "AMD Zen / Ryzen 1000"),
            ("zen2", "AMD Zen 2 / Ryzen 3000"),
            ("zen3", "AMD Zen 3 / Ryzen 5000"),
            ("zen4", "AMD Zen 4 / Ryzen 7000"),
            ("thunderx2t99", "Marvell ThunderX2"),
            ("cortexa57", "ARM Cortex-A57"),
            ("cortexa72", "ARM Cortex-A72"),
            ("neoversen1", "ARM Neoverse N1 / Graviton 2"),
            ("neoversev1", "ARM Neoverse V1 / Graviton 3"),
            ("armv8", "Generic ARMv8"),
        ],
    )
    def test_openblas_target_context(self, mock_platform, mock_blas, target, description):
        """Test context with various OpenBLAS targets."""
        mock_platform(system="Linux", machine="x86_64")
        mock_blas("openblas")
        resolver = ContextResolver(custom_context={"openblas_target": target})
        key = resolver.get_context_key()
        assert f"openblas_target_{target}" in key

    def test_haswell_vs_zen3_values(self, temp_test_file, mock_platform, mock_blas):
        """Test different values for Haswell vs Zen3 OpenBLAS targets."""
        storage = SnapshotStorage(temp_test_file, "test_openblas_targets")

        mock_platform(system="Linux", machine="x86_64")
        mock_blas("openblas")

        # OpenBLAS compiled for Haswell
        haswell_resolver = ContextResolver(custom_context={"openblas_target": "haswell"})
        storage.set_value(haswell_resolver.get_context_key(), 1.7320508075688772)

        # OpenBLAS compiled for Zen3
        zen3_resolver = ContextResolver(custom_context={"openblas_target": "zen3"})
        storage.set_value(zen3_resolver.get_context_key(), 1.7320508075688774)

        storage.set_value("default", 1.732050807)

        assert storage.get_value(haswell_resolver.get_context_key()) == 1.7320508075688772
        assert storage.get_value(zen3_resolver.get_context_key()) == 1.7320508075688774

    def test_sapphirerapids_vs_neoversen1(self, temp_test_file, mock_platform, mock_blas):
        """Test different values for Sapphire Rapids vs Neoverse N1."""
        storage = SnapshotStorage(temp_test_file, "test_cross_arch_openblas")

        # Intel Sapphire Rapids
        mock_platform(system="Linux", machine="x86_64")
        mock_blas("openblas")
        spr_resolver = ContextResolver(custom_context={"openblas_target": "sapphirerapids"})
        storage.set_value(spr_resolver.get_context_key(), [1.0, 2.0, 3.0])

        # ARM Neoverse N1
        mock_platform(system="Linux", machine="aarch64")
        n1_resolver = ContextResolver(custom_context={"openblas_target": "neoversen1"})
        storage.set_value(n1_resolver.get_context_key(), [1.0000001, 2.0, 3.0])

        storage.set_value("default", [1.0, 2.0, 3.0])

        assert storage.get_value(spr_resolver.get_context_key()) == [1.0, 2.0, 3.0]
        assert storage.get_value(n1_resolver.get_context_key()) == [1.0000001, 2.0, 3.0]


class TestMKLConfigurations:
    """Tests for Intel MKL-specific contexts."""

    @pytest.mark.parametrize(
        "interface",
        ["lp64", "ilp64"],
    )
    def test_mkl_interface_context(self, mock_platform, mock_blas, interface):
        """Test context with MKL interface variants."""
        mock_platform(system="Linux", machine="x86_64")
        mock_blas("mkl")
        resolver = ContextResolver(custom_context={"mkl_interface": interface})
        key = resolver.get_context_key()
        assert f"mkl_interface_{interface}" in key

    @pytest.mark.parametrize(
        "threading",
        ["sequential", "intel_omp", "gnu_omp", "tbb"],
    )
    def test_mkl_threading_context(self, mock_platform, mock_blas, threading):
        """Test context with MKL threading variants."""
        mock_platform(system="Linux", machine="x86_64")
        mock_blas("mkl")
        resolver = ContextResolver(custom_context={"mkl_threading": threading})
        key = resolver.get_context_key()
        assert f"mkl_threading_{threading}" in key

    def test_mkl_full_config(self, temp_test_file, mock_platform, mock_blas):
        """Test MKL with full configuration context."""
        storage = SnapshotStorage(temp_test_file, "test_mkl_config")

        mock_platform(system="Linux", machine="x86_64")
        mock_blas("mkl")

        # MKL with ILP64 and Intel OpenMP
        config1_resolver = ContextResolver(
            custom_context={"mkl_interface": "ilp64", "mkl_threading": "intel_omp"}
        )
        storage.set_value(config1_resolver.get_context_key(), 2.718281828459045)

        # MKL with LP64 and TBB
        config2_resolver = ContextResolver(
            custom_context={"mkl_interface": "lp64", "mkl_threading": "tbb"}
        )
        storage.set_value(config2_resolver.get_context_key(), 2.7182818284590452)

        storage.set_value("default", 2.71828)

        assert storage.get_value(config1_resolver.get_context_key()) == 2.718281828459045
        assert storage.get_value(config2_resolver.get_context_key()) == 2.7182818284590452


class TestBLASComparisonScenarios:
    """Tests for comparing results across different BLAS implementations."""

    def test_mkl_vs_openblas_values(self, temp_test_file, mock_platform, mock_blas):
        """Test different expected values for MKL vs OpenBLAS."""
        storage = SnapshotStorage(temp_test_file, "test_blas_comparison")

        mock_platform(system="Linux", machine="x86_64")

        # MKL result
        mock_blas("mkl")
        mkl_resolver = ContextResolver()
        storage.set_value(mkl_resolver.get_context_key(), 1.00000000000001)

        # OpenBLAS result
        mock_blas("openblas")
        openblas_resolver = ContextResolver()
        storage.set_value(openblas_resolver.get_context_key(), 1.00000000000002)

        storage.set_value("default", 1.0)

        assert storage.get_value(mkl_resolver.get_context_key()) == 1.00000000000001
        assert storage.get_value(openblas_resolver.get_context_key()) == 1.00000000000002

    def test_accelerate_vs_openblas_arm(self, temp_test_file, mock_platform, mock_blas):
        """Test Accelerate vs OpenBLAS on ARM."""
        storage = SnapshotStorage(temp_test_file, "test_arm_blas")

        # macOS with Accelerate
        mock_platform(system="Darwin", machine="arm64")
        mock_blas("accelerate")
        accel_resolver = ContextResolver()
        storage.set_value(accel_resolver.get_context_key(), 3.141592653589793)

        # Linux ARM with OpenBLAS
        mock_platform(system="Linux", machine="aarch64")
        mock_blas("openblas")
        openblas_resolver = ContextResolver()
        storage.set_value(openblas_resolver.get_context_key(), 3.1415926535897932)

        storage.set_value("default", 3.14159265)

        assert storage.get_value(accel_resolver.get_context_key()) == 3.141592653589793
        assert storage.get_value(openblas_resolver.get_context_key()) == 3.1415926535897932


class TestBLASVersions:
    """Tests for BLAS version-specific contexts."""

    @pytest.mark.parametrize(
        "version",
        ["0.3.20", "0.3.21", "0.3.23", "0.3.24", "0.3.25", "0.3.26"],
    )
    def test_openblas_version_context(self, mock_platform, mock_blas, version):
        """Test context with OpenBLAS versions."""
        mock_platform(system="Linux", machine="x86_64")
        mock_blas("openblas")
        resolver = ContextResolver(custom_context={"openblas_version": version})
        key = resolver.get_context_key()
        assert f"openblas_version_{version}" in key

    @pytest.mark.parametrize(
        "version",
        ["2021.4", "2022.1", "2023.0", "2023.2", "2024.0", "2024.1"],
    )
    def test_mkl_version_context(self, mock_platform, mock_blas, version):
        """Test context with MKL versions."""
        mock_platform(system="Linux", machine="x86_64")
        mock_blas("mkl")
        resolver = ContextResolver(custom_context={"mkl_version": version})
        key = resolver.get_context_key()
        assert f"mkl_version_{version}" in key

    def test_version_specific_values(self, temp_test_file, mock_platform, mock_blas):
        """Test that different BLAS versions can have different values."""
        storage = SnapshotStorage(temp_test_file, "test_blas_versions")

        mock_platform(system="Linux", machine="x86_64")
        mock_blas("openblas")

        # Older OpenBLAS
        old_resolver = ContextResolver(custom_context={"openblas_version": "0.3.20"})
        storage.set_value(old_resolver.get_context_key(), 1.414213562373095)

        # Newer OpenBLAS with potential bug fix
        new_resolver = ContextResolver(custom_context={"openblas_version": "0.3.26"})
        storage.set_value(new_resolver.get_context_key(), 1.4142135623730951)

        storage.set_value("default", 1.41421356)

        assert storage.get_value(old_resolver.get_context_key()) == 1.414213562373095
        assert storage.get_value(new_resolver.get_context_key()) == 1.4142135623730951


class TestBLASWithNumThreads:
    """Tests for BLAS with different thread configurations."""

    @pytest.mark.parametrize(
        "num_threads",
        ["1", "2", "4", "8", "16", "32"],
    )
    def test_thread_count_context(self, mock_platform, mock_blas, num_threads):
        """Test context with different thread counts."""
        mock_platform(system="Linux", machine="x86_64")
        mock_blas("openblas")
        resolver = ContextResolver(custom_context={"num_threads": num_threads})
        key = resolver.get_context_key()
        assert f"num_threads_{num_threads}" in key

    def test_single_vs_multi_thread_values(self, temp_test_file, mock_platform, mock_blas):
        """Test potentially different values between single and multi-threaded."""
        storage = SnapshotStorage(temp_test_file, "test_threading")

        mock_platform(system="Linux", machine="x86_64")
        mock_blas("openblas")

        # Single-threaded (deterministic)
        single_resolver = ContextResolver(custom_context={"num_threads": "1"})
        storage.set_value(single_resolver.get_context_key(), 2.0)

        # Multi-threaded (might have slight differences due to FP ordering)
        multi_resolver = ContextResolver(custom_context={"num_threads": "8"})
        storage.set_value(multi_resolver.get_context_key(), 2.0000000000000004)

        storage.set_value("default", 2.0)

        assert storage.get_value(single_resolver.get_context_key()) == 2.0
        assert storage.get_value(multi_resolver.get_context_key()) == 2.0000000000000004


class TestComprehensiveBLASContext:
    """Tests for comprehensive BLAS context combinations."""

    def test_full_blas_context(self, mock_platform, mock_blas):
        """Test fully specified BLAS context."""
        mock_platform(system="Linux", machine="x86_64")
        mock_blas("openblas")

        resolver = ContextResolver(
            custom_context={
                "openblas_target": "sapphirerapids",
                "openblas_version": "0.3.26",
                "num_threads": "1",
                "cpu_vendor": "intel",
            }
        )

        key = resolver.get_context_key()
        assert "linux-x86_64-openblas" in key
        assert "openblas_target_sapphirerapids" in key
        assert "openblas_version_0.3.26" in key
        assert "num_threads_1" in key
        assert "cpu_vendor_intel" in key

    def test_fallback_chain_with_blas(self, temp_test_file, mock_platform, mock_blas):
        """Test fallback chain with BLAS contexts."""
        storage = SnapshotStorage(temp_test_file, "test_blas_fallback")

        mock_platform(system="Linux", machine="x86_64")
        mock_blas("openblas")

        # Most specific - use context dict directly
        specific_ctx = {
            "platform": "linux",
            "arch": "x86_64",
            "blas": "openblas",
            "openblas_target": "haswell",
            "openblas_version": "0.3.26",
        }
        storage.set_value(specific_ctx, 1.0)

        # Just target
        target_ctx = {
            "platform": "linux",
            "arch": "x86_64",
            "blas": "openblas",
            "openblas_target": "haswell",
        }
        storage.set_value(target_ctx, 2.0)

        # Just BLAS
        blas_ctx = {"platform": "linux", "arch": "x86_64", "blas": "openblas"}
        storage.set_value(blas_ctx, 3.0)

        # Default
        storage.set_value("default", 4.0)

        # Each context gets its specific value
        assert storage.get_value(specific_ctx) == 1.0
        assert storage.get_value(target_ctx) == 2.0
        assert storage.get_value(blas_ctx) == 3.0

        # Unknown specific context: with openblas_target=unknown, falls back to less specific match
        # Since blas_ctx (platform+arch+blas) matches, it returns 3.0
        unknown_ctx = {
            "platform": "linux",
            "arch": "x86_64",
            "blas": "openblas",
            "openblas_target": "unknown",
        }
        assert storage.get_value(unknown_ctx) == 3.0  # Falls back to blas_ctx match
