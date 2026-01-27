"""Tests for custom serialization/deserialization."""

from __future__ import annotations


class TestCustomSerialization:
    """Tests for custom serialization/deserialization."""

    def test_custom_serialize(self, context_assert):
        """Test custom serialization function."""

        class CustomObject:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        def custom_serialize(obj):
            return {"x": obj.x, "y": obj.y, "type": "CustomObject"}

        def custom_deserialize(data):
            return CustomObject(data["x"], data["y"])

        obj = CustomObject(10, 20)

        context_assert._update_snapshots = True
        context_assert(
            obj,
            name="custom_obj",
            serialize=custom_serialize,
        )

        context_assert._update_snapshots = False
        # Use custom compare since default won't work for custom objects
        context_assert(
            CustomObject(10, 20),
            name="custom_obj",
            deserialize=custom_deserialize,
            compare=lambda a, b: a.x == b.x and a.y == b.y,
        )

    def test_custom_serialize_complex(self, context_assert):
        """Test custom serialization for complex nested data."""
        data = {
            "timestamp": "2026-01-20",
            "values": [1, 2, 3],
            "metadata": {"version": 1},
        }

        def serialize_without_timestamp(obj):
            return {
                "values": obj["values"],
                "metadata": obj["metadata"],
                "type": "filtered_dict",
            }

        def deserialize_without_timestamp(stored):
            return {
                "values": stored["values"],
                "metadata": stored["metadata"],
            }

        context_assert._update_snapshots = True
        context_assert(
            data,
            name="filtered_data",
            serialize=serialize_without_timestamp,
        )

        context_assert._update_snapshots = False
        # Different timestamp should still match
        data2 = {
            "timestamp": "2026-01-21",
            "values": [1, 2, 3],
            "metadata": {"version": 1},
        }
        context_assert(
            data2,
            name="filtered_data",
            deserialize=deserialize_without_timestamp,
            compare=lambda a, b: a["values"] == b["values"] and a["metadata"] == b["metadata"],
        )

    def test_custom_serialize_datetime(self, context_assert):
        """Test custom serialization for datetime-like objects."""
        from datetime import date

        test_date = date(2026, 1, 20)

        def serialize_date(d):
            return {"year": d.year, "month": d.month, "day": d.day, "type": "date"}

        def deserialize_date(data):
            return date(data["year"], data["month"], data["day"])

        context_assert._update_snapshots = True
        context_assert(test_date, name="date_test", serialize=serialize_date)

        context_assert._update_snapshots = False
        context_assert(
            date(2026, 1, 20),
            name="date_test",
            deserialize=deserialize_date,
            compare=lambda a, b: a == b,
        )

    def test_custom_serialize_with_numpy(self, context_assert, numpy):
        """Test custom serialization with numpy-based objects."""

        class DataContainer:
            def __init__(self, data):
                self.data = numpy.array(data)

        def serialize_container(obj):
            return {"data": obj.data.tolist(), "type": "DataContainer"}

        def deserialize_container(stored):
            return DataContainer(stored["data"])

        container = DataContainer([1.0, 2.0, 3.0])

        context_assert._update_snapshots = True
        context_assert(container, name="numpy_container", serialize=serialize_container)

        context_assert._update_snapshots = False
        context_assert(
            DataContainer([1.0, 2.0, 3.0]),
            name="numpy_container",
            deserialize=deserialize_container,
            compare=lambda a, b: numpy.array_equal(a.data, b.data),
        )

    def test_custom_serialize_partial_data(self, context_assert):
        """Test serializing only part of the data."""

        def serialize_values_only(obj):
            # Only store the 'important' field
            return {"important": obj["important"], "type": "partial"}

        def deserialize_values_only(stored):
            return {"important": stored["important"]}

        context_assert._update_snapshots = True
        context_assert(
            {"important": 42, "transient": "ignored", "timestamp": "now"},
            name="partial_test",
            serialize=serialize_values_only,
        )

        context_assert._update_snapshots = False
        context_assert(
            {"important": 42, "transient": "different", "timestamp": "later"},
            name="partial_test",
            deserialize=deserialize_values_only,
            compare=lambda a, b: a["important"] == b["important"],
        )

    def test_custom_serialize_with_validation(self, context_assert):
        """Test custom serialization with validation."""

        def serialize_with_checksum(obj):
            checksum = sum(obj["values"])
            return {"values": obj["values"], "checksum": checksum, "type": "validated"}

        def deserialize_with_checksum(stored):
            # Verify checksum on deserialize
            expected_checksum = sum(stored["values"])
            if stored["checksum"] != expected_checksum:
                raise ValueError("Checksum mismatch")
            return {"values": stored["values"]}

        context_assert._update_snapshots = True
        context_assert(
            {"values": [1, 2, 3]},
            name="checksum_test",
            serialize=serialize_with_checksum,
        )

        context_assert._update_snapshots = False
        context_assert(
            {"values": [1, 2, 3]},
            name="checksum_test",
            deserialize=deserialize_with_checksum,
            compare=lambda a, b: a["values"] == b["values"],
        )

    def test_custom_serialize_precision_control(self, context_assert):
        """Test custom serialization for controlling precision."""

        def serialize_rounded(value):
            # Round to 2 decimal places for storage
            return {"value": round(value, 2), "type": "rounded_float"}

        def deserialize_rounded(stored):
            return stored["value"]

        context_assert._update_snapshots = True
        context_assert(3.14159265359, name="precision_test", serialize=serialize_rounded)

        context_assert._update_snapshots = False
        # Slightly different value should match due to rounding (3.14159 rounds to 3.14)
        context_assert(
            3.14159,
            name="precision_test",
            deserialize=deserialize_rounded,
            compare=lambda a, b: round(a, 2) == b,
        )

    def test_custom_serialize_nested_objects(self, context_assert):
        """Test custom serialization for nested objects."""

        class Inner:
            def __init__(self, value):
                self.value = value

        class Outer:
            def __init__(self, inner):
                self.inner = inner

        def serialize_nested(obj):
            return {
                "inner_value": obj.inner.value,
                "type": "nested",
            }

        def deserialize_nested(stored):
            return Outer(Inner(stored["inner_value"]))

        context_assert._update_snapshots = True
        context_assert(
            Outer(Inner(42)),
            name="nested_test",
            serialize=serialize_nested,
        )

        context_assert._update_snapshots = False
        context_assert(
            Outer(Inner(42)),
            name="nested_test",
            deserialize=deserialize_nested,
            compare=lambda a, b: a.inner.value == b.inner.value,
        )
