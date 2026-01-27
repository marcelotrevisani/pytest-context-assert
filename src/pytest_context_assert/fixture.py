"""The context_assert fixture and ContextAssert class."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pytest_context_assert.comparators import FunctionComparator, get_comparator
from pytest_context_assert.context import ContextResolver
from pytest_context_assert.storage import SnapshotStorage

if TYPE_CHECKING:
    import pytest

CompareFunc = Callable[[Any, Any], bool]
SerializeFunc = Callable[[Any], dict[str, Any]]
DeserializeFunc = Callable[[dict[str, Any]], Any]


class ContextAssertionError(AssertionError):
    """Raised when a context assertion fails."""

    def __init__(
        self,
        message: str,
        actual: Any = None,
        expected: Any = None,
        context_key: str | None = None,
        diff: str | None = None,
    ):
        super().__init__(message)
        self.actual = actual
        self.expected = expected
        self.context_key = context_key
        self.diff = diff


class MissingSnapshotError(Exception):
    """Raised when no snapshot exists and update mode is not enabled."""

    def __init__(self, message: str, context_key: str, snapshot_path: Path):
        super().__init__(message)
        self.context_key = context_key
        self.snapshot_path = snapshot_path


class ContextAssert:
    """Main assertion class for context-aware comparisons."""

    def __init__(
        self,
        request: pytest.FixtureRequest,
        resolver: ContextResolver | None = None,
        snapshot_dir: str = "__snapshots__",
        default_rtol: float = 1e-7,
        default_atol: float = 0.0,
        update_snapshots: bool = False,
    ):
        """Initialize the ContextAssert fixture.

        Args:
            request: The pytest request fixture.
            resolver: Optional custom ContextResolver.
            snapshot_dir: Directory name for storing snapshots.
            default_rtol: Default relative tolerance for comparisons.
            default_atol: Default absolute tolerance for comparisons.
            update_snapshots: Whether to update snapshots instead of comparing.
        """
        self._request = request
        self._resolver = resolver or ContextResolver()
        self._snapshot_dir = snapshot_dir
        self._default_rtol = default_rtol
        self._default_atol = default_atol
        self._update_snapshots = update_snapshots
        self._assertion_count = 0
        self._custom_context: dict[str, Any] = {}

    @property
    def context(self) -> dict[str, str]:
        """Get the current context dictionary."""
        if self._custom_context:
            resolver = self._resolver.with_custom_context(**self._custom_context)
            return resolver.get_context()
        return self._resolver.get_context()

    @property
    def context_key(self) -> str:
        """Get the current context key (for display purposes)."""
        if self._custom_context:
            resolver = self._resolver.with_custom_context(**self._custom_context)
            return resolver.get_context_key()
        return self._resolver.get_context_key()

    @property
    def storage(self) -> SnapshotStorage:
        """Get the snapshot storage for the current test."""
        test_file = Path(self._request.fspath)
        test_name = self._request.node.name

        if "[" in test_name:
            test_name = test_name.split("[")[0]

        return SnapshotStorage(
            test_file=test_file,
            test_name=test_name,
            snapshot_dir=self._snapshot_dir,
        )

    def __call__(
        self,
        value: Any,
        *,
        rtol: float | None = None,
        atol: float | None = None,
        name: str | None = None,
        compare: CompareFunc | None = None,
        serialize: SerializeFunc | None = None,
        deserialize: DeserializeFunc | None = None,
    ) -> None:
        """Assert that a value matches the stored snapshot.

        Args:
            value: The actual value to compare.
            rtol: Relative tolerance (overrides default).
            atol: Absolute tolerance (overrides default).
            name: Optional name for this assertion (allows multiple per test).
            compare: Optional custom comparison function. Should take two arguments
                (actual, expected) and return True if they match, False otherwise.
                When provided, rtol and atol are ignored.
            serialize: Optional custom serialization function. Should take the value
                and return a dict suitable for YAML storage.
            deserialize: Optional custom deserialization function. Should take the
                stored dict and return the original value type.

        Raises:
            ContextAssertionError: If the value doesn't match the expected.
            MissingSnapshotError: If no snapshot exists and not in update mode.
        """
        rtol = rtol if rtol is not None else self._default_rtol
        atol = atol if atol is not None else self._default_atol

        if name is None:
            self._assertion_count += 1
            name = f"assertion_{self._assertion_count}"

        context = self.context
        context_key = self.context_key  # For display purposes
        storage = self.storage

        if self._update_snapshots:
            storage.set_value(context, value, name=name, serialize=serialize)
            return

        if not storage.has_value(context, name=name):
            raise MissingSnapshotError(
                f"No snapshot found for context '{context_key}'. "
                f"Run with --context-assert-update to create it.\n"
                f"Snapshot path: {storage.snapshot_path}",
                context_key=context_key,
                snapshot_path=storage.snapshot_path,
            )

        expected = storage.get_value(context, name=name, deserialize=deserialize)

        if compare is not None:
            comparator = FunctionComparator(compare)
            result = comparator.compare(value, expected)
        else:
            comparator = get_comparator(value)
            result = comparator.compare(value, expected, rtol=rtol, atol=atol)

        if not result.equal:
            raise ContextAssertionError(
                f"Assertion failed for context '{context_key}':\n{result.message}",
                actual=value,
                expected=expected,
                context_key=context_key,
                diff=result.diff,
            )

    def assert_match(
        self,
        value: Any,
        *,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Syrupy-style assertion method.

        This is an alias for __call__ with a more explicit name.

        Args:
            value: The actual value to compare.
            name: Optional name for this assertion.
            **kwargs: Additional arguments passed to __call__ (rtol, atol, compare, etc.)
        """
        return self.__call__(value, name=name, **kwargs)

    def with_context(self, **custom_context: Any) -> ContextAssert:
        """Create a new ContextAssert with additional custom context.

        Args:
            **custom_context: Additional context key-value pairs.

        Returns:
            New ContextAssert instance with merged context.
        """
        new_instance = ContextAssert(
            request=self._request,
            resolver=self._resolver,
            snapshot_dir=self._snapshot_dir,
            default_rtol=self._default_rtol,
            default_atol=self._default_atol,
            update_snapshots=self._update_snapshots,
        )
        new_instance._custom_context = {**self._custom_context, **custom_context}
        new_instance._assertion_count = self._assertion_count
        return new_instance

    def update(
        self,
        value: Any,
        *,
        name: str | None = None,
        context: dict[str, str] | str | None = None,
    ) -> None:
        """Explicitly update a snapshot value.

        Args:
            value: The value to store.
            name: Optional assertion name.
            context: Optional context override (dict or "default"). Defaults to current.
        """
        if context is None:
            context = self.context

        if name is None:
            self._assertion_count += 1
            name = f"assertion_{self._assertion_count}"

        self.storage.set_value(context, value, name=name)

    def set_default(
        self,
        value: Any,
        *,
        name: str | None = None,
    ) -> None:
        """Set the default value for this assertion.

        Args:
            value: The default value to store.
            name: Optional assertion name.
        """
        self.update(value, name=name, context="default")
