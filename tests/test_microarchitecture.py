"""Tests for microarchitecture-specific contexts (CPU variants, vendors)."""

from __future__ import annotations

import pytest

from pytest_context_assert.context import ContextResolver
from pytest_context_assert.storage import SnapshotStorage


class TestCPUVendorContext:
    """Tests for CPU vendor-specific contexts (Intel vs AMD)."""

    def test_intel_cpu_context(self, mock_platform, mock_blas):
        """Test context with Intel CPU specification."""
        mock_platform(system="Linux", machine="x86_64")
        mock_blas(None)
        resolver = ContextResolver(custom_context={"cpu_vendor": "intel"})
        key = resolver.get_context_key()
        assert "cpu_vendor_intel" in key

    def test_amd_cpu_context(self, mock_platform, mock_blas):
        """Test context with AMD CPU specification."""
        mock_platform(system="Linux", machine="x86_64")
        mock_blas(None)
        resolver = ContextResolver(custom_context={"cpu_vendor": "amd"})
        key = resolver.get_context_key()
        assert "cpu_vendor_amd" in key

    def test_different_results_for_vendors(self, temp_test_file, mock_platform, mock_blas):
        """Test that different CPU vendors can have different expected values."""
        storage = SnapshotStorage(temp_test_file, "test_vendor_specific")

        mock_platform(system="Linux", machine="x86_64")
        mock_blas(None)

        # Intel-specific result
        intel_resolver = ContextResolver(custom_context={"cpu_vendor": "intel"})
        storage.set_value(intel_resolver.get_context_key(), 1.00000001)

        # AMD-specific result
        amd_resolver = ContextResolver(custom_context={"cpu_vendor": "amd"})
        storage.set_value(amd_resolver.get_context_key(), 1.00000002)

        # Default fallback
        storage.set_value("default", 1.0)

        # Verify retrieval
        assert storage.get_value(intel_resolver.get_context_key()) == 1.00000001
        assert storage.get_value(amd_resolver.get_context_key()) == 1.00000002


class TestIntelMicroarchitectures:
    """Tests for Intel microarchitecture-specific contexts."""

    @pytest.mark.parametrize(
        "microarch,codename",
        [
            ("haswell", "4th Gen Core"),
            ("broadwell", "5th Gen Core"),
            ("skylake", "6th Gen Core"),
            ("cascadelake", "2nd Gen Xeon Scalable"),
            ("icelake", "10th Gen Core / 3rd Gen Xeon"),
            ("tigerlake", "11th Gen Core"),
            ("alderlake", "12th Gen Core"),
            ("raptorlake", "13th Gen Core"),
            ("sapphirerapids", "4th Gen Xeon Scalable"),
            ("emeraldrapids", "5th Gen Xeon Scalable"),
        ],
    )
    def test_intel_microarch_context(self, mock_platform, mock_blas, microarch, codename):
        """Test context with various Intel microarchitectures."""
        mock_platform(system="Linux", machine="x86_64")
        mock_blas("mkl")
        resolver = ContextResolver(custom_context={"cpu_vendor": "intel", "microarch": microarch})
        key = resolver.get_context_key()
        assert f"microarch_{microarch}" in key
        assert "cpu_vendor_intel" in key

    def test_haswell_vs_sapphirerapids_values(self, temp_test_file, mock_platform, mock_blas):
        """Test different values for Haswell vs Sapphire Rapids."""
        storage = SnapshotStorage(temp_test_file, "test_microarch_comparison")

        mock_platform(system="Linux", machine="x86_64")
        mock_blas("mkl")

        # Older Haswell might have slightly different FP behavior
        haswell_resolver = ContextResolver(
            custom_context={"cpu_vendor": "intel", "microarch": "haswell"}
        )
        storage.set_value(haswell_resolver.get_context_key(), 2.718281828459045)

        # Sapphire Rapids with newer FP unit
        spr_resolver = ContextResolver(
            custom_context={"cpu_vendor": "intel", "microarch": "sapphirerapids"}
        )
        storage.set_value(spr_resolver.get_context_key(), 2.7182818284590452)

        storage.set_value("default", 2.718281828)

        # Verify distinct values
        assert storage.get_value(haswell_resolver.get_context_key()) == 2.718281828459045
        assert storage.get_value(spr_resolver.get_context_key()) == 2.7182818284590452


class TestAMDMicroarchitectures:
    """Tests for AMD microarchitecture-specific contexts."""

    @pytest.mark.parametrize(
        "microarch,family",
        [
            ("zen", "Ryzen 1000 / EPYC 7001"),
            ("zen+", "Ryzen 2000 / EPYC 7002"),
            ("zen2", "Ryzen 3000 / EPYC 7002"),
            ("zen3", "Ryzen 5000 / EPYC 7003"),
            ("zen4", "Ryzen 7000 / EPYC 9004"),
            ("zen5", "Ryzen 9000 / EPYC 9005"),
        ],
    )
    def test_amd_microarch_context(self, mock_platform, mock_blas, microarch, family):
        """Test context with various AMD microarchitectures."""
        mock_platform(system="Linux", machine="x86_64")
        mock_blas("openblas")
        resolver = ContextResolver(custom_context={"cpu_vendor": "amd", "microarch": microarch})
        key = resolver.get_context_key()
        assert f"microarch_{microarch}" in key
        assert "cpu_vendor_amd" in key

    def test_zen2_vs_zen4_values(self, temp_test_file, mock_platform, mock_blas):
        """Test different values for Zen2 vs Zen4."""
        storage = SnapshotStorage(temp_test_file, "test_amd_comparison")

        mock_platform(system="Linux", machine="x86_64")
        mock_blas("openblas")

        # Zen2
        zen2_resolver = ContextResolver(custom_context={"cpu_vendor": "amd", "microarch": "zen2"})
        storage.set_value(zen2_resolver.get_context_key(), 3.141592653589793)

        # Zen4 with potentially different rounding
        zen4_resolver = ContextResolver(custom_context={"cpu_vendor": "amd", "microarch": "zen4"})
        storage.set_value(zen4_resolver.get_context_key(), 3.1415926535897932)

        storage.set_value("default", 3.14159265)

        assert storage.get_value(zen2_resolver.get_context_key()) == 3.141592653589793
        assert storage.get_value(zen4_resolver.get_context_key()) == 3.1415926535897932


class TestARMMicroarchitectures:
    """Tests for ARM microarchitecture-specific contexts."""

    @pytest.mark.parametrize(
        "microarch,description",
        [
            ("cortex-a72", "Raspberry Pi 4"),
            ("cortex-a76", "AWS Graviton 2"),
            ("neoverse-n1", "AWS Graviton 2 / Ampere Altra"),
            ("neoverse-v1", "AWS Graviton 3"),
            ("neoverse-v2", "AWS Graviton 4"),
            ("apple-m1", "Apple M1"),
            ("apple-m2", "Apple M2"),
            ("apple-m3", "Apple M3"),
            ("apple-m4", "Apple M4"),
        ],
    )
    def test_arm_microarch_context(self, mock_platform, mock_blas, microarch, description):
        """Test context with various ARM microarchitectures."""
        if "apple" in microarch:
            mock_platform(system="Darwin", machine="arm64")
            mock_blas("accelerate")
        else:
            mock_platform(system="Linux", machine="aarch64")
            mock_blas("openblas")

        resolver = ContextResolver(custom_context={"microarch": microarch})
        key = resolver.get_context_key()
        assert f"microarch_{microarch}" in key

    def test_graviton2_vs_graviton3_values(self, temp_test_file, mock_platform, mock_blas):
        """Test different values for AWS Graviton generations."""
        storage = SnapshotStorage(temp_test_file, "test_graviton_comparison")

        mock_platform(system="Linux", machine="aarch64")
        mock_blas("openblas")

        # Graviton 2 (Neoverse N1) - use context dict directly to handle hyphen in value
        g2_ctx = {
            "platform": "linux",
            "arch": "arm64",
            "blas": "openblas",
            "microarch": "neoverse-n1",
        }
        storage.set_value(g2_ctx, 1.4142135623730951)

        # Graviton 3 (Neoverse V1)
        g3_ctx = {
            "platform": "linux",
            "arch": "arm64",
            "blas": "openblas",
            "microarch": "neoverse-v1",
        }
        storage.set_value(g3_ctx, 1.4142135623730950)

        storage.set_value("default", 1.41421356)

        assert storage.get_value(g2_ctx) == 1.4142135623730951
        assert storage.get_value(g3_ctx) == 1.4142135623730950

    def test_apple_m1_vs_m3_values(self, temp_test_file, mock_platform, mock_blas):
        """Test different values for Apple Silicon generations."""
        storage = SnapshotStorage(temp_test_file, "test_apple_comparison")

        mock_platform(system="Darwin", machine="arm64")
        mock_blas("accelerate")

        # Apple M1
        m1_resolver = ContextResolver(custom_context={"microarch": "apple-m1"})
        storage.set_value(m1_resolver.get_context_key(), 2.302585092994046)

        # Apple M3
        m3_resolver = ContextResolver(custom_context={"microarch": "apple-m3"})
        storage.set_value(m3_resolver.get_context_key(), 2.3025850929940459)

        storage.set_value("default", 2.30258509)

        assert storage.get_value(m1_resolver.get_context_key()) == 2.302585092994046
        assert storage.get_value(m3_resolver.get_context_key()) == 2.3025850929940459


class TestSIMDExtensions:
    """Tests for SIMD extension-specific contexts."""

    @pytest.mark.parametrize(
        "simd_ext",
        ["sse4.2", "avx", "avx2", "avx512", "avx512vnni", "amx"],
    )
    def test_intel_simd_context(self, mock_platform, mock_blas, simd_ext):
        """Test context with Intel SIMD extensions."""
        mock_platform(system="Linux", machine="x86_64")
        mock_blas("mkl")
        resolver = ContextResolver(custom_context={"simd": simd_ext})
        key = resolver.get_context_key()
        assert f"simd_{simd_ext}" in key

    def test_avx2_vs_avx512_values(self, temp_test_file, mock_platform, mock_blas):
        """Test different values for AVX2 vs AVX-512."""
        storage = SnapshotStorage(temp_test_file, "test_simd_comparison")

        mock_platform(system="Linux", machine="x86_64")
        mock_blas("mkl")

        # AVX2 result
        avx2_resolver = ContextResolver(custom_context={"simd": "avx2"})
        storage.set_value(avx2_resolver.get_context_key(), [1.0, 2.0, 3.0, 4.0])

        # AVX-512 might have different precision characteristics
        avx512_resolver = ContextResolver(custom_context={"simd": "avx512"})
        storage.set_value(avx512_resolver.get_context_key(), [1.0000000001, 2.0, 3.0, 4.0])

        storage.set_value("default", [1.0, 2.0, 3.0, 4.0])

        assert storage.get_value(avx2_resolver.get_context_key()) == [1.0, 2.0, 3.0, 4.0]
        assert storage.get_value(avx512_resolver.get_context_key()) == [
            1.0000000001,
            2.0,
            3.0,
            4.0,
        ]


class TestCombinedMicroarchContext:
    """Tests for combined microarchitecture contexts."""

    def test_full_context_specification(self, mock_platform, mock_blas):
        """Test fully specified context with all details."""
        mock_platform(system="Linux", machine="x86_64")
        mock_blas("mkl")
        resolver = ContextResolver(
            custom_context={
                "cpu_vendor": "intel",
                "microarch": "sapphirerapids",
                "simd": "avx512",
                "numa_nodes": "2",
            }
        )
        key = resolver.get_context_key()
        assert "linux-x86_64-mkl" in key
        assert "cpu_vendor_intel" in key
        assert "microarch_sapphirerapids" in key
        assert "simd_avx512" in key
        assert "numa_nodes_2" in key

    def test_hierarchical_fallback(self, temp_test_file, mock_platform, mock_blas):
        """Test fallback through context hierarchy."""
        storage = SnapshotStorage(temp_test_file, "test_hierarchy")

        mock_platform(system="Linux", machine="x86_64")
        mock_blas("mkl")

        # Most specific - microarch + simd
        specific_resolver = ContextResolver(
            custom_context={"microarch": "sapphirerapids", "simd": "avx512"}
        )
        storage.set_value(specific_resolver.get_context_key(), 1.0)

        # Less specific - just microarch
        microarch_resolver = ContextResolver(custom_context={"microarch": "sapphirerapids"})
        storage.set_value(microarch_resolver.get_context_key(), 2.0)

        # Platform only
        storage.set_value("linux-x86_64-mkl", 3.0)

        # Default
        storage.set_value("default", 4.0)

        # Each should get its specific value
        assert storage.get_value(specific_resolver.get_context_key()) == 1.0
        assert storage.get_value(microarch_resolver.get_context_key()) == 2.0
        assert storage.get_value("linux-x86_64-mkl") == 3.0
