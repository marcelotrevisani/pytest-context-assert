"""Advanced numpy integration tests - linear algebra, matrix operations, etc."""

from __future__ import annotations


from pytest_context_assert import set_context, get_platform, get_arch


class TestLinearAlgebra:
    """Tests for linear algebra operations with platform-specific results."""

    def test_matrix_multiplication(self, context_assert, numpy):
        """Test matrix multiplication results."""
        A = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        B = numpy.array([[5.0, 6.0], [7.0, 8.0]])
        result = numpy.matmul(A, B)

        context_assert._update_snapshots = True
        context_assert(result, name="matmul_result")

        context_assert._update_snapshots = False
        context_assert(result.copy(), name="matmul_result")

    def test_eigenvalues(self, context_assert, numpy):
        """Test eigenvalue computation."""
        matrix = numpy.array([[4.0, 2.0], [1.0, 3.0]])
        eigenvalues = numpy.linalg.eigvals(matrix)
        # Sort for consistency across platforms
        eigenvalues = numpy.sort(eigenvalues)

        context_assert._update_snapshots = True
        context_assert(eigenvalues, name="eigenvalues", rtol=1e-10)

        context_assert._update_snapshots = False
        context_assert(eigenvalues.copy(), name="eigenvalues", rtol=1e-10)

    def test_svd(self, context_assert, numpy):
        """Test singular value decomposition."""
        matrix = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        U, S, Vh = numpy.linalg.svd(matrix)

        context_assert._update_snapshots = True
        context_assert(S, name="svd_singular_values", rtol=1e-10)

        context_assert._update_snapshots = False
        context_assert(S.copy(), name="svd_singular_values", rtol=1e-10)

    def test_matrix_inverse(self, context_assert, numpy):
        """Test matrix inversion."""
        matrix = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        inverse = numpy.linalg.inv(matrix)

        context_assert._update_snapshots = True
        context_assert(inverse, name="matrix_inverse", rtol=1e-10)

        context_assert._update_snapshots = False
        context_assert(inverse.copy(), name="matrix_inverse", rtol=1e-10)

    def test_determinant(self, context_assert, numpy):
        """Test determinant calculation."""
        matrix = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
        det = numpy.linalg.det(matrix)

        context_assert._update_snapshots = True
        context_assert(det, name="determinant", rtol=1e-10)

        context_assert._update_snapshots = False
        context_assert(det, name="determinant", rtol=1e-10)

    def test_cholesky_decomposition(self, context_assert, numpy):
        """Test Cholesky decomposition of positive definite matrix."""
        # Create a positive definite matrix
        A = numpy.array([[4.0, 2.0], [2.0, 3.0]])
        L = numpy.linalg.cholesky(A)

        context_assert._update_snapshots = True
        context_assert(L, name="cholesky", rtol=1e-10)

        context_assert._update_snapshots = False
        context_assert(L.copy(), name="cholesky", rtol=1e-10)

    def test_qr_decomposition(self, context_assert, numpy):
        """Test QR decomposition."""
        matrix = numpy.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Q, R = numpy.linalg.qr(matrix)

        context_assert._update_snapshots = True
        context_assert(R, name="qr_r_matrix", rtol=1e-10)

        context_assert._update_snapshots = False
        context_assert(R.copy(), name="qr_r_matrix", rtol=1e-10)

    def test_solve_linear_system(self, context_assert, numpy):
        """Test solving linear systems."""
        A = numpy.array([[3.0, 1.0], [1.0, 2.0]])
        b = numpy.array([9.0, 8.0])
        x = numpy.linalg.solve(A, b)

        context_assert._update_snapshots = True
        context_assert(x, name="linear_solve", rtol=1e-10)

        context_assert._update_snapshots = False
        context_assert(x.copy(), name="linear_solve", rtol=1e-10)


class TestStatisticalFunctions:
    """Tests for statistical computations."""

    def test_mean_std(self, context_assert, numpy):
        """Test mean and standard deviation."""
        data = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = {"mean": float(numpy.mean(data)), "std": float(numpy.std(data))}

        context_assert._update_snapshots = True
        context_assert(result, name="mean_std")

        context_assert._update_snapshots = False
        context_assert(result, name="mean_std")

    def test_percentiles(self, context_assert, numpy):
        """Test percentile calculations."""
        numpy.random.seed(42)
        data = numpy.random.randn(1000)
        percentiles = numpy.percentile(data, [25, 50, 75])

        context_assert._update_snapshots = True
        context_assert(percentiles, name="percentiles", rtol=1e-10)

        context_assert._update_snapshots = False
        context_assert(percentiles.copy(), name="percentiles", rtol=1e-10)

    def test_correlation_matrix(self, context_assert, numpy):
        """Test correlation matrix computation."""
        numpy.random.seed(42)
        data = numpy.random.randn(100, 3)
        corr = numpy.corrcoef(data.T)

        context_assert._update_snapshots = True
        context_assert(corr, name="correlation", rtol=1e-10)

        context_assert._update_snapshots = False
        context_assert(corr.copy(), name="correlation", rtol=1e-10)

    def test_histogram(self, context_assert, numpy):
        """Test histogram computation."""
        numpy.random.seed(42)
        data = numpy.random.randn(1000)
        hist, bin_edges = numpy.histogram(data, bins=10)

        context_assert._update_snapshots = True
        context_assert(hist, name="histogram_counts")
        context_assert(bin_edges, name="histogram_edges", rtol=1e-10)

        context_assert._update_snapshots = False
        context_assert(hist.copy(), name="histogram_counts")
        context_assert(bin_edges.copy(), name="histogram_edges", rtol=1e-10)

    def test_covariance(self, context_assert, numpy):
        """Test covariance calculation."""
        numpy.random.seed(42)
        x = numpy.random.randn(100)
        y = x + 0.5 * numpy.random.randn(100)
        cov = numpy.cov(x, y)

        context_assert._update_snapshots = True
        context_assert(cov, name="covariance", rtol=1e-10)

        context_assert._update_snapshots = False
        context_assert(cov.copy(), name="covariance", rtol=1e-10)


class TestFFTOperations:
    """Tests for FFT operations."""

    def test_fft_1d(self, context_assert, numpy):
        """Test 1D FFT."""
        signal = numpy.sin(numpy.linspace(0, 2 * numpy.pi, 64))
        fft_result = numpy.fft.fft(signal)
        magnitudes = numpy.abs(fft_result)

        context_assert._update_snapshots = True
        context_assert(magnitudes, name="fft_magnitudes", rtol=1e-10)

        context_assert._update_snapshots = False
        context_assert(magnitudes.copy(), name="fft_magnitudes", rtol=1e-10)

    def test_fft_2d(self, context_assert, numpy):
        """Test 2D FFT."""
        image = numpy.random.rand(16, 16)
        numpy.random.seed(42)
        image = numpy.random.rand(16, 16)
        fft_result = numpy.fft.fft2(image)
        magnitudes = numpy.abs(fft_result)

        context_assert._update_snapshots = True
        context_assert(magnitudes, name="fft2d_magnitudes", rtol=1e-10)

        context_assert._update_snapshots = False
        context_assert(magnitudes.copy(), name="fft2d_magnitudes", rtol=1e-10)

    def test_ifft_reconstruction(self, context_assert, numpy):
        """Test inverse FFT reconstruction."""
        original = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        fft_result = numpy.fft.fft(original)
        reconstructed = numpy.fft.ifft(fft_result).real

        context_assert._update_snapshots = True
        context_assert(reconstructed, name="ifft_reconstructed", rtol=1e-10)

        context_assert._update_snapshots = False
        context_assert(reconstructed.copy(), name="ifft_reconstructed", rtol=1e-10)


class TestSpecialArrayTypes:
    """Tests for special array types and operations."""

    def test_masked_array(self, context_assert, numpy):
        """Test masked array handling."""
        data = numpy.array([1.0, 2.0, -999.0, 4.0, 5.0])
        masked = numpy.ma.masked_equal(data, -999.0)
        mean_val = float(masked.mean())

        context_assert._update_snapshots = True
        context_assert(mean_val, name="masked_mean")

        context_assert._update_snapshots = False
        context_assert(mean_val, name="masked_mean")

    def test_structured_array(self, context_assert, numpy):
        """Test structured array with custom dtype."""
        dt = numpy.dtype([("name", "U10"), ("age", "i4"), ("score", "f8")])
        data = numpy.array([("Alice", 25, 85.5), ("Bob", 30, 90.2)], dtype=dt)

        # Extract numerical part for comparison
        scores = data["score"].tolist()

        context_assert._update_snapshots = True
        context_assert(scores, name="struct_scores")

        context_assert._update_snapshots = False
        context_assert(scores, name="struct_scores")

    def test_record_array(self, context_assert, numpy):
        """Test record array."""
        data = numpy.rec.array(
            [(1, 2.0, "Hello"), (3, 4.0, "World")],
            dtype=[("x", "i4"), ("y", "f8"), ("z", "U10")],
        )
        x_sum = int(data.x.sum())

        context_assert._update_snapshots = True
        context_assert(x_sum, name="rec_x_sum")

        context_assert._update_snapshots = False
        context_assert(x_sum, name="rec_x_sum")

    def test_datetime_array(self, context_assert, numpy):
        """Test datetime64 array."""
        dates = numpy.array(["2026-01-01", "2026-01-02", "2026-01-03"], dtype="datetime64")
        # Convert to string list for serialization
        date_strs = [str(d) for d in dates]

        context_assert._update_snapshots = True
        context_assert(date_strs, name="datetime_strings")

        context_assert._update_snapshots = False
        context_assert(date_strs, name="datetime_strings")

    def test_different_dtypes(self, context_assert, numpy):
        """Test various numpy dtypes."""
        dtypes_results = {
            "float32": float(numpy.array([1.5], dtype=numpy.float32)[0]),
            "float64": float(numpy.array([1.5], dtype=numpy.float64)[0]),
            "int8": int(numpy.array([127], dtype=numpy.int8)[0]),
            "int16": int(numpy.array([32767], dtype=numpy.int16)[0]),
            "int32": int(numpy.array([2147483647], dtype=numpy.int32)[0]),
            "uint8": int(numpy.array([255], dtype=numpy.uint8)[0]),
        }

        context_assert._update_snapshots = True
        context_assert(dtypes_results, name="dtype_values")

        context_assert._update_snapshots = False
        context_assert(dtypes_results, name="dtype_values")


class TestNumpyWithContext:
    """Tests for numpy operations with custom context."""

    @set_context({"computation_type": "matrix_ops"})
    def test_context_aware_matrix_multiply(self, context_assert, numpy):
        """Test matrix multiplication with custom context."""
        A = numpy.random.seed(42)
        A = numpy.random.rand(3, 3)
        B = numpy.random.rand(3, 3)
        result = A @ B

        context_assert._update_snapshots = True
        context_assert(result, name="ctx_matmul", rtol=1e-10)

        context_assert._update_snapshots = False
        context_assert(result.copy(), name="ctx_matmul", rtol=1e-10)

    @set_context({"precision": "high", "algorithm": "default"})
    def test_high_precision_computation(self, context_assert, numpy):
        """Test high-precision computation with context."""
        # Test with values that might have precision issues
        a = numpy.float64(1.0)
        b = numpy.float64(1e-15)
        result = float(a + b - a)

        context_assert._update_snapshots = True
        context_assert(result, name="high_precision", rtol=1e-14)

        context_assert._update_snapshots = False
        context_assert(result, name="high_precision", rtol=1e-14)

    def test_platform_specific_computation(self, context_assert, numpy):
        """Test computation that might vary by platform."""
        numpy.random.seed(42)
        data = numpy.random.randn(100)

        # Use with_context for runtime context
        ctx = context_assert.with_context(
            platform=get_platform(),
            arch=get_arch(),
        )

        result = {
            "sum": float(numpy.sum(data)),
            "mean": float(numpy.mean(data)),
            "std": float(numpy.std(data)),
        }

        ctx._update_snapshots = True
        ctx(result, name="platform_stats", rtol=1e-10)

        ctx._update_snapshots = False
        ctx(result, name="platform_stats", rtol=1e-10)


class TestBroadcastingAndVectorization:
    """Tests for broadcasting and vectorized operations."""

    def test_broadcasting(self, context_assert, numpy):
        """Test array broadcasting."""
        a = numpy.array([[1], [2], [3]])  # 3x1
        b = numpy.array([10, 20, 30])  # 1x3
        result = a + b  # 3x3 broadcast

        context_assert._update_snapshots = True
        context_assert(result, name="broadcast_result")

        context_assert._update_snapshots = False
        context_assert(result.copy(), name="broadcast_result")

    def test_vectorized_function(self, context_assert, numpy):
        """Test numpy vectorized functions."""

        def custom_func(x):
            return x**2 + 2 * x + 1

        vec_func = numpy.vectorize(custom_func)
        data = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = vec_func(data)

        context_assert._update_snapshots = True
        context_assert(result, name="vectorized")

        context_assert._update_snapshots = False
        context_assert(result.copy(), name="vectorized")

    def test_ufunc_operations(self, context_assert, numpy):
        """Test universal function operations."""
        data = numpy.array([0.0, numpy.pi / 6, numpy.pi / 4, numpy.pi / 3, numpy.pi / 2])

        results = {
            "sin": numpy.sin(data).tolist(),
            "cos": numpy.cos(data).tolist(),
            "tan": numpy.tan(data[:4]).tolist(),  # Exclude pi/2 for tan
            "exp": numpy.exp(data).tolist(),
            "log": numpy.log(data[1:] + 1).tolist(),  # Avoid log(0)
        }

        context_assert._update_snapshots = True
        context_assert(results, name="ufunc_results")

        context_assert._update_snapshots = False
        context_assert(results, name="ufunc_results")


class TestToleranceEdgeCases:
    """Tests for edge cases in tolerance comparisons."""

    def test_very_small_values(self, context_assert, numpy):
        """Test comparison of very small values."""
        arr1 = numpy.array([1e-15, 1e-16, 1e-17])
        arr2 = numpy.array([1.000001e-15, 1.000001e-16, 1.000001e-17])

        context_assert._update_snapshots = True
        context_assert(arr1, name="tiny_values")

        context_assert._update_snapshots = False
        context_assert(arr2, name="tiny_values", rtol=1e-5)

    def test_very_large_values(self, context_assert, numpy):
        """Test comparison of very large values."""
        arr1 = numpy.array([1e15, 1e16, 1e17])
        arr2 = numpy.array([1.000001e15, 1.000001e16, 1.000001e17])

        context_assert._update_snapshots = True
        context_assert(arr1, name="huge_values")

        context_assert._update_snapshots = False
        context_assert(arr2, name="huge_values", rtol=1e-5)

    def test_mixed_scale_values(self, context_assert, numpy):
        """Test comparison with mixed scale values."""
        arr1 = numpy.array([1e-10, 1.0, 1e10])
        arr2 = numpy.array([1.00001e-10, 1.00001, 1.00001e10])

        context_assert._update_snapshots = True
        context_assert(arr1, name="mixed_scale")

        context_assert._update_snapshots = False
        context_assert(arr2, name="mixed_scale", rtol=1e-4)

    def test_near_zero_with_atol(self, context_assert, numpy):
        """Test near-zero comparisons using atol."""
        arr1 = numpy.array([0.0, 0.0, 0.0])
        arr2 = numpy.array([1e-12, -1e-12, 1e-13])

        context_assert._update_snapshots = True
        context_assert(arr1, name="near_zero")

        context_assert._update_snapshots = False
        context_assert(arr2, name="near_zero", atol=1e-10)
