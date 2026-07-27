"""Tests for custom objects - dataclasses, pydantic, attrs, namedtuples, etc."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum, IntEnum
from fractions import Fraction
from pathlib import Path
from typing import Any, NamedTuple
from uuid import UUID

from pytest_context_assert import set_context

# ============================================================================
# Custom Classes for Testing
# ============================================================================


@dataclass
class Point:
    """Simple 2D point."""

    x: float
    y: float


@dataclass
class Point3D:
    """3D point with additional metadata."""

    x: float
    y: float
    z: float
    label: str = ""


@dataclass
class Person:
    """Person with nested data."""

    name: str
    age: int
    email: str
    tags: list[str] = field(default_factory=list)


@dataclass
class Company:
    """Company with nested Person objects."""

    name: str
    employees: list[Person] = field(default_factory=list)
    founded: date | None = None


class Color(Enum):
    """Color enumeration."""

    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class Priority(IntEnum):
    """Priority levels."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3


class Coordinate(NamedTuple):
    """Named tuple for coordinates."""

    lat: float
    lon: float
    name: str = ""


class Vector:
    """Custom vector class with math operations."""

    def __init__(self, *components: float):
        self.components = list(components)

    def magnitude(self) -> float:
        return sum(c**2 for c in self.components) ** 0.5

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return False
        return self.components == other.components

    def __repr__(self) -> str:
        return f"Vector({', '.join(map(str, self.components))})"


class Matrix:
    """Custom matrix class."""

    def __init__(self, rows: list[list[float]]):
        self.rows = rows
        self.n_rows = len(rows)
        self.n_cols = len(rows[0]) if rows else 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return False
        return self.rows == other.rows


class Tree:
    """Binary tree node."""

    def __init__(self, value: Any, left: Tree | None = None, right: Tree | None = None):
        self.value = value
        self.left = left
        self.right = right


class LinkedListNode:
    """Linked list node."""

    def __init__(self, value: Any, next_node: LinkedListNode | None = None):
        self.value = value
        self.next = next_node


# ============================================================================
# Test Classes
# ============================================================================


class TestDataclassObjects:
    """Tests for dataclass objects."""

    def test_simple_dataclass(self, context_assert):
        """Test simple dataclass serialization."""
        point = Point(3.0, 4.0)

        def serialize_point(p):
            return {"x": p.x, "y": p.y, "type": "Point"}

        def deserialize_point(data):
            return Point(data["x"], data["y"])

        context_assert._update_snapshots = True
        context_assert(point, name="simple_dataclass", serialize=serialize_point)

        context_assert._update_snapshots = False
        context_assert(
            Point(3.0, 4.0),
            name="simple_dataclass",
            deserialize=deserialize_point,
            compare=lambda a, b: a == b,
        )

    def test_dataclass_with_defaults(self, context_assert):
        """Test dataclass with default values."""
        point = Point3D(1.0, 2.0, 3.0, "origin")

        def serialize(p):
            return {"x": p.x, "y": p.y, "z": p.z, "label": p.label, "type": "Point3D"}

        def deserialize(data):
            return Point3D(data["x"], data["y"], data["z"], data["label"])

        context_assert._update_snapshots = True
        context_assert(point, name="dataclass_defaults", serialize=serialize)

        context_assert._update_snapshots = False
        context_assert(
            Point3D(1.0, 2.0, 3.0, "origin"),
            name="dataclass_defaults",
            deserialize=deserialize,
            compare=lambda a, b: a == b,
        )

    def test_nested_dataclass(self, context_assert):
        """Test nested dataclass structures."""
        company = Company(
            name="TechCorp",
            employees=[
                Person("Alice", 30, "alice@tech.com", ["dev", "lead"]),
                Person("Bob", 25, "bob@tech.com", ["dev"]),
            ],
            founded=date(2020, 1, 15),
        )

        def serialize(c):
            return {
                "name": c.name,
                "employees": [
                    {"name": e.name, "age": e.age, "email": e.email, "tags": e.tags}
                    for e in c.employees
                ],
                "founded": c.founded.isoformat() if c.founded else None,
                "type": "Company",
            }

        def deserialize(data):
            return Company(
                name=data["name"],
                employees=[
                    Person(e["name"], e["age"], e["email"], e["tags"]) for e in data["employees"]
                ],
                founded=date.fromisoformat(data["founded"]) if data["founded"] else None,
            )

        context_assert._update_snapshots = True
        context_assert(company, name="nested_dataclass", serialize=serialize)

        context_assert._update_snapshots = False
        context_assert(
            company,
            name="nested_dataclass",
            deserialize=deserialize,
            compare=lambda a, b: a == b,
        )

    def test_dataclass_list(self, context_assert):
        """Test list of dataclass objects."""
        points = [Point(1.0, 1.0), Point(2.0, 2.0), Point(3.0, 3.0)]

        def serialize(pts):
            return {"points": [{"x": p.x, "y": p.y} for p in pts], "type": "PointList"}

        def deserialize(data):
            return [Point(p["x"], p["y"]) for p in data["points"]]

        context_assert._update_snapshots = True
        context_assert(points, name="dataclass_list", serialize=serialize)

        context_assert._update_snapshots = False
        context_assert(
            points,
            name="dataclass_list",
            deserialize=deserialize,
            compare=lambda a, b: a == b,
        )


class TestEnumObjects:
    """Tests for enum objects."""

    def test_string_enum(self, context_assert):
        """Test string enum serialization."""
        color = Color.RED

        def serialize(c):
            return {"value": c.value, "name": c.name, "type": "Color"}

        def deserialize(data):
            return Color(data["value"])

        context_assert._update_snapshots = True
        context_assert(color, name="string_enum", serialize=serialize)

        context_assert._update_snapshots = False
        context_assert(
            Color.RED,
            name="string_enum",
            deserialize=deserialize,
            compare=lambda a, b: a == b,
        )

    def test_int_enum(self, context_assert):
        """Test integer enum serialization."""
        priority = Priority.HIGH

        def serialize(p):
            return {"value": int(p), "name": p.name, "type": "Priority"}

        def deserialize(data):
            return Priority(data["value"])

        context_assert._update_snapshots = True
        context_assert(priority, name="int_enum", serialize=serialize)

        context_assert._update_snapshots = False
        context_assert(
            Priority.HIGH,
            name="int_enum",
            deserialize=deserialize,
            compare=lambda a, b: a == b,
        )

    def test_enum_in_dict(self, context_assert):
        """Test enum as dictionary value."""
        config = {"theme": Color.BLUE, "priority": Priority.MEDIUM}

        def serialize(cfg):
            return {
                "theme": cfg["theme"].value,
                "priority": int(cfg["priority"]),
                "type": "config",
            }

        def deserialize(data):
            return {
                "theme": Color(data["theme"]),
                "priority": Priority(data["priority"]),
            }

        context_assert._update_snapshots = True
        context_assert(config, name="enum_in_dict", serialize=serialize)

        context_assert._update_snapshots = False
        context_assert(
            config,
            name="enum_in_dict",
            deserialize=deserialize,
            compare=lambda a, b: a == b,
        )


class TestNamedTupleObjects:
    """Tests for named tuple objects."""

    def test_namedtuple(self, context_assert):
        """Test named tuple serialization."""
        coord = Coordinate(40.7128, -74.0060, "New York")

        def serialize(c):
            return {"lat": c.lat, "lon": c.lon, "name": c.name, "type": "Coordinate"}

        def deserialize(data):
            return Coordinate(data["lat"], data["lon"], data["name"])

        context_assert._update_snapshots = True
        context_assert(coord, name="namedtuple", serialize=serialize)

        context_assert._update_snapshots = False
        context_assert(
            Coordinate(40.7128, -74.0060, "New York"),
            name="namedtuple",
            deserialize=deserialize,
            compare=lambda a, b: a == b,
        )

    def test_namedtuple_list(self, context_assert):
        """Test list of named tuples."""
        coords = [
            Coordinate(40.7128, -74.0060, "NYC"),
            Coordinate(34.0522, -118.2437, "LA"),
            Coordinate(41.8781, -87.6298, "Chicago"),
        ]

        def serialize(cs):
            return {
                "coordinates": [{"lat": c.lat, "lon": c.lon, "name": c.name} for c in cs],
                "type": "CoordinateList",
            }

        def deserialize(data):
            return [Coordinate(c["lat"], c["lon"], c["name"]) for c in data["coordinates"]]

        context_assert._update_snapshots = True
        context_assert(coords, name="namedtuple_list", serialize=serialize)

        context_assert._update_snapshots = False
        context_assert(
            coords,
            name="namedtuple_list",
            deserialize=deserialize,
            compare=lambda a, b: a == b,
        )


class TestCustomMathObjects:
    """Tests for custom mathematical objects."""

    def test_vector(self, context_assert):
        """Test custom Vector class."""
        v = Vector(3.0, 4.0, 0.0)

        def serialize(vec):
            return {
                "components": vec.components,
                "magnitude": vec.magnitude(),
                "type": "Vector",
            }

        def deserialize(data):
            return Vector(*data["components"])

        context_assert._update_snapshots = True
        context_assert(v, name="vector", serialize=serialize)

        context_assert._update_snapshots = False
        context_assert(
            Vector(3.0, 4.0, 0.0),
            name="vector",
            deserialize=deserialize,
            compare=lambda a, b: a == b,
        )

    def test_matrix(self, context_assert):
        """Test custom Matrix class."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])

        def serialize(mat):
            return {
                "rows": mat.rows,
                "shape": [mat.n_rows, mat.n_cols],
                "type": "Matrix",
            }

        def deserialize(data):
            return Matrix(data["rows"])

        context_assert._update_snapshots = True
        context_assert(m, name="matrix", serialize=serialize)

        context_assert._update_snapshots = False
        context_assert(
            Matrix([[1, 2, 3], [4, 5, 6]]),
            name="matrix",
            deserialize=deserialize,
            compare=lambda a, b: a == b,
        )


class TestDateTimeObjects:
    """Tests for datetime objects."""

    def test_datetime(self, context_assert):
        """Test datetime serialization."""
        dt = datetime(2026, 1, 20, 15, 30, 45)

        def serialize(d):
            return {"iso": d.isoformat(), "type": "datetime"}

        def deserialize(data):
            return datetime.fromisoformat(data["iso"])

        context_assert._update_snapshots = True
        context_assert(dt, name="datetime", serialize=serialize)

        context_assert._update_snapshots = False
        context_assert(
            datetime(2026, 1, 20, 15, 30, 45),
            name="datetime",
            deserialize=deserialize,
            compare=lambda a, b: a == b,
        )

    def test_date(self, context_assert):
        """Test date serialization."""
        d = date(2026, 1, 20)

        def serialize(d):
            return {"iso": d.isoformat(), "type": "date"}

        def deserialize(data):
            return date.fromisoformat(data["iso"])

        context_assert._update_snapshots = True
        context_assert(d, name="date", serialize=serialize)

        context_assert._update_snapshots = False
        context_assert(
            date(2026, 1, 20),
            name="date",
            deserialize=deserialize,
            compare=lambda a, b: a == b,
        )

    def test_time(self, context_assert):
        """Test time serialization."""
        t = time(15, 30, 45)

        def serialize(t):
            return {"iso": t.isoformat(), "type": "time"}

        def deserialize(data):
            return time.fromisoformat(data["iso"])

        context_assert._update_snapshots = True
        context_assert(t, name="time", serialize=serialize)

        context_assert._update_snapshots = False
        context_assert(
            time(15, 30, 45),
            name="time",
            deserialize=deserialize,
            compare=lambda a, b: a == b,
        )

    def test_timedelta(self, context_assert):
        """Test timedelta serialization."""
        td = timedelta(days=5, hours=3, minutes=30)

        def serialize(td):
            return {"total_seconds": td.total_seconds(), "type": "timedelta"}

        def deserialize(data):
            return timedelta(seconds=data["total_seconds"])

        context_assert._update_snapshots = True
        context_assert(td, name="timedelta", serialize=serialize)

        context_assert._update_snapshots = False
        context_assert(
            timedelta(days=5, hours=3, minutes=30),
            name="timedelta",
            deserialize=deserialize,
            compare=lambda a, b: a == b,
        )


class TestSpecialTypes:
    """Tests for special Python types."""

    def test_decimal(self, context_assert):
        """Test Decimal serialization."""
        d = Decimal("3.14159265358979323846")

        def serialize(d):
            return {"value": str(d), "type": "Decimal"}

        def deserialize(data):
            return Decimal(data["value"])

        context_assert._update_snapshots = True
        context_assert(d, name="decimal", serialize=serialize)

        context_assert._update_snapshots = False
        context_assert(
            Decimal("3.14159265358979323846"),
            name="decimal",
            deserialize=deserialize,
            compare=lambda a, b: a == b,
        )

    def test_fraction(self, context_assert):
        """Test Fraction serialization."""
        f = Fraction(22, 7)

        def serialize(f):
            return {
                "numerator": f.numerator,
                "denominator": f.denominator,
                "type": "Fraction",
            }

        def deserialize(data):
            return Fraction(data["numerator"], data["denominator"])

        context_assert._update_snapshots = True
        context_assert(f, name="fraction", serialize=serialize)

        context_assert._update_snapshots = False
        context_assert(
            Fraction(22, 7),
            name="fraction",
            deserialize=deserialize,
            compare=lambda a, b: a == b,
        )

    def test_uuid(self, context_assert):
        """Test UUID serialization."""
        u = UUID("12345678-1234-5678-1234-567812345678")

        def serialize(u):
            return {"value": str(u), "type": "UUID"}

        def deserialize(data):
            return UUID(data["value"])

        context_assert._update_snapshots = True
        context_assert(u, name="uuid", serialize=serialize)

        context_assert._update_snapshots = False
        context_assert(
            UUID("12345678-1234-5678-1234-567812345678"),
            name="uuid",
            deserialize=deserialize,
            compare=lambda a, b: a == b,
        )

    def test_path(self, context_assert):
        """Test Path serialization."""
        p = Path("/usr/local/bin/python")

        def serialize(p):
            return {"path": str(p), "type": "Path"}

        def deserialize(data):
            return Path(data["path"])

        context_assert._update_snapshots = True
        context_assert(p, name="path", serialize=serialize)

        context_assert._update_snapshots = False
        context_assert(
            Path("/usr/local/bin/python"),
            name="path",
            deserialize=deserialize,
            compare=lambda a, b: a == b,
        )


class TestTreeStructures:
    """Tests for tree and graph structures."""

    def test_binary_tree(self, context_assert):
        """Test binary tree serialization."""
        tree = Tree(
            1,
            left=Tree(2, left=Tree(4), right=Tree(5)),
            right=Tree(3, left=Tree(6), right=Tree(7)),
        )

        def serialize_tree(node):
            if node is None:
                return None
            return {
                "value": node.value,
                "left": serialize_tree(node.left),
                "right": serialize_tree(node.right),
            }

        def serialize(t):
            return {"tree": serialize_tree(t), "type": "BinaryTree"}

        def deserialize_tree(data):
            if data is None:
                return None
            return Tree(
                data["value"],
                left=deserialize_tree(data["left"]),
                right=deserialize_tree(data["right"]),
            )

        def deserialize(data):
            return deserialize_tree(data["tree"])

        def trees_equal(a, b):
            if a is None and b is None:
                return True
            if a is None or b is None:
                return False
            return (
                a.value == b.value and trees_equal(a.left, b.left) and trees_equal(a.right, b.right)
            )

        context_assert._update_snapshots = True
        context_assert(tree, name="binary_tree", serialize=serialize)

        context_assert._update_snapshots = False
        context_assert(
            tree,
            name="binary_tree",
            deserialize=deserialize,
            compare=trees_equal,
        )

    def test_linked_list(self, context_assert):
        """Test linked list serialization."""
        # Create: 1 -> 2 -> 3 -> 4 -> None
        head = LinkedListNode(1, LinkedListNode(2, LinkedListNode(3, LinkedListNode(4))))

        def serialize(node):
            values = []
            current = node
            while current:
                values.append(current.value)
                current = current.next
            return {"values": values, "type": "LinkedList"}

        def deserialize(data):
            if not data["values"]:
                return None
            head = LinkedListNode(data["values"][0])
            current = head
            for val in data["values"][1:]:
                current.next = LinkedListNode(val)
                current = current.next
            return head

        def lists_equal(a, b):
            while a and b:
                if a.value != b.value:
                    return False
                a = a.next
                b = b.next
            return a is None and b is None

        context_assert._update_snapshots = True
        context_assert(head, name="linked_list", serialize=serialize)

        context_assert._update_snapshots = False
        context_assert(
            head,
            name="linked_list",
            deserialize=deserialize,
            compare=lists_equal,
        )


class TestComplexNestedStructures:
    """Tests for complex nested data structures."""

    def test_deeply_nested(self, context_assert):
        """Test deeply nested structure."""
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {"value": 42, "items": [1, 2, 3]},
                        "other": "data",
                    }
                },
                "sibling": [{"a": 1}, {"b": 2}],
            }
        }

        context_assert._update_snapshots = True
        context_assert(data, name="deeply_nested")

        context_assert._update_snapshots = False
        context_assert(data, name="deeply_nested")

    def test_mixed_types_structure(self, context_assert):
        """Test structure with mixed types."""
        data = {
            "string": "hello",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "list": [1, "two", 3.0],
            "nested_list": [[1, 2], [3, 4]],
            "nested_dict": {"a": {"b": {"c": 1}}},
        }

        context_assert._update_snapshots = True
        context_assert(data, name="mixed_types")

        context_assert._update_snapshots = False
        context_assert(data, name="mixed_types")

    @set_context({"test_type": "complex_structure"})
    def test_context_with_complex_object(self, context_assert):
        """Test complex object with custom context."""
        company = Company(
            name="TestCorp",
            employees=[Person("Test", 25, "test@test.com", ["test"])],
            founded=date(2025, 1, 1),
        )

        def serialize(c):
            return {
                "name": c.name,
                "employee_count": len(c.employees),
                "founded_year": c.founded.year if c.founded else None,
                "type": "CompanySummary",
            }

        context_assert._update_snapshots = True
        context_assert(company, name="ctx_complex", serialize=serialize)

        context_assert._update_snapshots = False
        context_assert(
            company,
            name="ctx_complex",
            serialize=serialize,
            compare=lambda a, b: True,  # Using same serialization
        )
