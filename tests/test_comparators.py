"""Tests for comparators."""

from __future__ import annotations

from pytest_context_assert.comparators import (
    DictComparator,
    FunctionComparator,
    GenericComparator,
    ListComparator,
    NumpyArrayComparator,
    ScalarComparator,
    get_comparator,
)


class TestScalarComparator:
    """Tests for ScalarComparator."""

    def test_can_compare(self):
        comparator = ScalarComparator()
        assert comparator.can_compare(42)
        assert comparator.can_compare(3.14)
        assert comparator.can_compare("hello")
        assert not comparator.can_compare([1, 2, 3])

    def test_compare_equal_ints(self):
        comparator = ScalarComparator()
        result = comparator.compare(42, 42)
        assert result.equal

    def test_compare_unequal_ints(self):
        comparator = ScalarComparator()
        result = comparator.compare(42, 43)
        assert not result.equal

    def test_compare_floats_within_tolerance(self):
        comparator = ScalarComparator()
        result = comparator.compare(1.0000001, 1.0, rtol=1e-5)
        assert result.equal

    def test_compare_floats_outside_tolerance(self):
        comparator = ScalarComparator()
        result = comparator.compare(1.001, 1.0, rtol=1e-5)
        assert not result.equal

    def test_compare_strings(self):
        comparator = ScalarComparator()
        assert comparator.compare("hello", "hello").equal
        assert not comparator.compare("hello", "world").equal

    def test_compare_booleans(self):
        comparator = ScalarComparator()
        assert comparator.compare(True, True).equal
        assert comparator.compare(False, False).equal
        assert not comparator.compare(True, False).equal

    def test_compare_none(self):
        comparator = ScalarComparator()
        assert comparator.compare(None, None).equal
        assert not comparator.compare(None, 0).equal

    def test_compare_float_with_zero_expected(self):
        """Test float comparison when expected is zero."""
        comparator = ScalarComparator()
        # With zero expected, only atol matters
        result = comparator.compare(0.0001, 0, atol=0.001)
        assert result.equal

        result = comparator.compare(0.01, 0, atol=0.001)
        assert not result.equal

    def test_compare_float_with_atol(self):
        """Test float comparison with absolute tolerance."""
        comparator = ScalarComparator()
        result = comparator.compare(1.0001, 1.0, rtol=0, atol=0.001)
        assert result.equal

    def test_comparison_result_has_diff(self):
        """Test that failed comparison includes diff info."""
        comparator = ScalarComparator()
        result = comparator.compare(1.5, 1.0, rtol=1e-9, atol=0)
        assert not result.equal
        assert result.diff is not None
        assert "difference" in result.diff


class TestNumpyArrayComparator:
    """Tests for NumpyArrayComparator."""

    def test_can_compare(self, numpy):
        comparator = NumpyArrayComparator()
        assert comparator.can_compare(numpy.array([1, 2, 3]))
        assert not comparator.can_compare([1, 2, 3])

    def test_compare_equal_arrays(self, numpy):
        comparator = NumpyArrayComparator()
        arr = numpy.array([1.0, 2.0, 3.0])
        result = comparator.compare(arr, arr.copy())
        assert result.equal

    def test_compare_arrays_within_tolerance(self, numpy):
        comparator = NumpyArrayComparator()
        arr1 = numpy.array([1.0, 2.0, 3.0])
        arr2 = numpy.array([1.0000001, 2.0000001, 3.0000001])
        result = comparator.compare(arr1, arr2, rtol=1e-5)
        assert result.equal

    def test_compare_arrays_outside_tolerance(self, numpy):
        comparator = NumpyArrayComparator()
        arr1 = numpy.array([1.0, 2.0, 3.0])
        arr2 = numpy.array([1.01, 2.01, 3.01])
        result = comparator.compare(arr1, arr2, rtol=1e-5)
        assert not result.equal

    def test_compare_different_shapes(self, numpy):
        comparator = NumpyArrayComparator()
        arr1 = numpy.array([1, 2, 3])
        arr2 = numpy.array([[1, 2], [3, 4]])
        result = comparator.compare(arr1, arr2)
        assert not result.equal
        assert "shape" in result.message.lower()

    def test_compare_integer_arrays(self, numpy):
        comparator = NumpyArrayComparator()
        arr1 = numpy.array([1, 2, 3])
        arr2 = numpy.array([1, 2, 3])
        result = comparator.compare(arr1, arr2)
        assert result.equal

    def test_compare_integer_arrays_unequal(self, numpy):
        comparator = NumpyArrayComparator()
        arr1 = numpy.array([1, 2, 3])
        arr2 = numpy.array([1, 2, 4])
        result = comparator.compare(arr1, arr2)
        assert not result.equal
        assert "differ" in result.diff

    def test_compare_2d_arrays(self, numpy):
        comparator = NumpyArrayComparator()
        arr1 = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        arr2 = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        result = comparator.compare(arr1, arr2)
        assert result.equal

    def test_compare_with_dtype_conversion(self, numpy):
        """Test comparison with compatible dtypes."""
        comparator = NumpyArrayComparator()
        arr1 = numpy.array([1.0, 2.0, 3.0], dtype=numpy.float64)
        arr2 = numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32)
        result = comparator.compare(arr1, arr2)
        assert result.equal

    def test_compare_from_list(self, numpy):
        """Test comparison when expected is a list."""
        comparator = NumpyArrayComparator()
        arr = numpy.array([1.0, 2.0, 3.0])
        result = comparator.compare(arr, [1.0, 2.0, 3.0])
        assert result.equal


class TestListComparator:
    """Tests for ListComparator."""

    def test_can_compare(self):
        comparator = ListComparator()
        assert comparator.can_compare([1, 2, 3])
        assert comparator.can_compare((1, 2, 3))
        assert not comparator.can_compare("hello")

    def test_compare_equal_lists(self):
        comparator = ListComparator()
        result = comparator.compare([1, 2, 3], [1, 2, 3])
        assert result.equal

    def test_compare_unequal_lists(self):
        comparator = ListComparator()
        result = comparator.compare([1, 2, 3], [1, 2, 4])
        assert not result.equal

    def test_compare_different_lengths(self):
        comparator = ListComparator()
        result = comparator.compare([1, 2], [1, 2, 3])
        assert not result.equal
        assert "length" in result.message.lower()

    def test_compare_tuples(self):
        comparator = ListComparator()
        result = comparator.compare((1, 2, 3), (1, 2, 3))
        assert result.equal

    def test_compare_nested_lists(self):
        comparator = ListComparator()
        result = comparator.compare([[1, 2], [3, 4]], [[1, 2], [3, 4]])
        assert result.equal

    def test_compare_nested_lists_unequal(self):
        comparator = ListComparator()
        result = comparator.compare([[1, 2], [3, 4]], [[1, 2], [3, 5]])
        assert not result.equal

    def test_compare_list_tuple_mixed(self):
        """Test that list and tuple can be compared."""
        comparator = ListComparator()
        result = comparator.compare([1, 2, 3], (1, 2, 3))
        assert result.equal

    def test_compare_empty_lists(self):
        comparator = ListComparator()
        result = comparator.compare([], [])
        assert result.equal

    def test_compare_with_floats_tolerance(self):
        """Test list comparison with float tolerance."""
        comparator = ListComparator()
        result = comparator.compare([1.0, 2.0], [1.0000001, 2.0], rtol=1e-5)
        assert result.equal


class TestDictComparator:
    """Tests for DictComparator."""

    def test_can_compare(self):
        comparator = DictComparator()
        assert comparator.can_compare({"a": 1})
        assert not comparator.can_compare([1, 2])

    def test_compare_equal_dicts(self):
        comparator = DictComparator()
        result = comparator.compare({"a": 1, "b": 2}, {"a": 1, "b": 2})
        assert result.equal

    def test_compare_unequal_dicts(self):
        comparator = DictComparator()
        result = comparator.compare({"a": 1}, {"a": 2})
        assert not result.equal

    def test_compare_different_keys(self):
        comparator = DictComparator()
        result = comparator.compare({"a": 1}, {"b": 1})
        assert not result.equal
        assert "missing" in result.message.lower() or "extra" in result.message.lower()

    def test_compare_extra_keys(self):
        comparator = DictComparator()
        result = comparator.compare({"a": 1, "b": 2}, {"a": 1})
        assert not result.equal
        assert "extra" in result.message.lower()

    def test_compare_missing_keys(self):
        comparator = DictComparator()
        result = comparator.compare({"a": 1}, {"a": 1, "b": 2})
        assert not result.equal
        assert "missing" in result.message.lower()

    def test_compare_nested_dicts(self):
        comparator = DictComparator()
        result = comparator.compare(
            {"outer": {"inner": 1}},
            {"outer": {"inner": 1}},
        )
        assert result.equal

    def test_compare_empty_dicts(self):
        comparator = DictComparator()
        result = comparator.compare({}, {})
        assert result.equal

    def test_compare_with_type_mismatch(self):
        comparator = DictComparator()
        result = comparator.compare({"a": 1}, [1, 2])
        assert not result.equal
        assert "type" in result.message.lower()


class TestGenericComparator:
    """Tests for GenericComparator."""

    def test_can_compare_anything(self):
        comparator = GenericComparator()
        assert comparator.can_compare(42)
        assert comparator.can_compare("hello")
        assert comparator.can_compare(object())

    def test_compare_equal_objects(self):
        comparator = GenericComparator()
        result = comparator.compare(42, 42)
        assert result.equal

    def test_compare_unequal_objects(self):
        comparator = GenericComparator()
        result = comparator.compare(42, 43)
        assert not result.equal

    def test_compare_with_custom_eq(self):
        """Test comparison with objects that have custom __eq__."""

        class AlwaysEqual:
            def __eq__(self, other):
                return True

        comparator = GenericComparator()
        result = comparator.compare(AlwaysEqual(), "anything")
        assert result.equal

    def test_compare_handles_exception(self):
        """Test that comparison handles exceptions gracefully."""

        class RaisesOnCompare:
            def __eq__(self, other):
                raise ValueError("Cannot compare")

        comparator = GenericComparator()
        result = comparator.compare(RaisesOnCompare(), "anything")
        assert not result.equal
        assert "exception" in result.message.lower()


class TestFunctionComparator:
    """Tests for FunctionComparator."""

    def test_can_compare_any_value(self):
        """FunctionComparator should accept any value."""
        comparator = FunctionComparator(lambda a, b: a == b)
        assert comparator.can_compare(42)
        assert comparator.can_compare("hello")
        assert comparator.can_compare([1, 2, 3])
        assert comparator.can_compare({"a": 1})

    def test_compare_with_simple_function(self):
        """Test comparison with a simple equality function."""
        comparator = FunctionComparator(lambda a, b: a == b)
        result = comparator.compare(42, 42)
        assert result.equal

        result = comparator.compare(42, 43)
        assert not result.equal

    def test_compare_with_custom_logic(self):
        """Test comparison with custom logic."""

        def case_insensitive(a, b):
            return a.lower() == b.lower()

        comparator = FunctionComparator(case_insensitive)
        result = comparator.compare("Hello", "hello")
        assert result.equal

        result = comparator.compare("Hello", "world")
        assert not result.equal

    def test_compare_with_tolerance(self):
        """Test comparison with custom tolerance logic."""

        def within_10_percent(a, b):
            if b == 0:
                return a == 0
            return abs(a - b) / abs(b) <= 0.1

        comparator = FunctionComparator(within_10_percent)
        result = comparator.compare(105, 100)
        assert result.equal

        result = comparator.compare(120, 100)
        assert not result.equal

    def test_compare_with_complex_objects(self):
        """Test comparison with complex objects."""

        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        def points_close(a, b):
            return abs(a.x - b.x) < 0.01 and abs(a.y - b.y) < 0.01

        comparator = FunctionComparator(points_close)
        result = comparator.compare(Point(1.0, 2.0), Point(1.005, 2.005))
        assert result.equal

        result = comparator.compare(Point(1.0, 2.0), Point(2.0, 3.0))
        assert not result.equal

    def test_compare_function_raises_exception(self):
        """Test that exceptions in compare function are handled."""

        def bad_compare(a, b):
            raise ValueError("Something went wrong")

        comparator = FunctionComparator(bad_compare)
        result = comparator.compare(1, 2)
        assert not result.equal
        assert "exception" in result.message.lower()

    def test_compare_function_returns_non_bool(self):
        """Test that non-bool return values are converted to bool."""
        comparator = FunctionComparator(lambda a, b: 1 if a == b else 0)
        result = comparator.compare(5, 5)
        assert result.equal

        result = comparator.compare(5, 3)
        assert not result.equal

        comparator = FunctionComparator(lambda a, b: "match" if a == b else "")
        result = comparator.compare(5, 5)
        assert result.equal

        result = comparator.compare(5, 3)
        assert not result.equal


class TestGetComparator:
    """Tests for get_comparator function."""

    def test_get_scalar_comparator(self):
        assert isinstance(get_comparator(42), ScalarComparator)
        assert isinstance(get_comparator(3.14), ScalarComparator)
        assert isinstance(get_comparator("hello"), ScalarComparator)

    def test_get_list_comparator(self):
        assert isinstance(get_comparator([1, 2, 3]), ListComparator)

    def test_get_dict_comparator(self):
        assert isinstance(get_comparator({"a": 1}), DictComparator)

    def test_get_numpy_comparator(self, numpy):
        arr = numpy.array([1, 2, 3])
        assert isinstance(get_comparator(arr), NumpyArrayComparator)

    def test_get_generic_comparator_fallback(self):
        """Test that unknown types get GenericComparator."""

        class CustomClass:
            pass

        assert isinstance(get_comparator(CustomClass()), GenericComparator)
