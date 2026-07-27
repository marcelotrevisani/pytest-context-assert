"""pytest plugin hooks and configuration."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, overload

import pytest

from pytest_context_assert.context import ContextResolver
from pytest_context_assert.fixture import ContextAssert

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Any

    F = Callable[..., Any]


@overload
def set_context(ctx: dict[str, Any]) -> Callable[[F], F]: ...


@overload
def set_context(**kwargs: Any) -> Callable[[F], F]: ...


def set_context(ctx: dict[str, Any] | None = None, **kwargs: Any) -> Callable[[F], F]:
    """Decorator to specify custom context for a test.

    Can be used in two ways:

    1. With a dictionary:
        @pytest_context_assert.set_context({"openblas": "haswell", "platform": "linux"})
        def test_my_func(context_assert):
            ...

    2. With keyword arguments:
        @pytest_context_assert.set_context(openblas="haswell", platform="linux")
        def test_my_func(context_assert):
            ...

    The context values will be merged with auto-detected context,
    with explicitly specified values taking precedence.

    Args:
        ctx: Dictionary of context key-value pairs.
        **kwargs: Context key-value pairs as keyword arguments.

    Returns:
        A decorator that marks the test with the specified context.
    """
    if ctx is not None:
        context_dict = {**ctx, **kwargs}
    else:
        context_dict = kwargs

    return pytest.mark.context_assert_context(_context_dict=context_dict)


# Alias for backward compatibility
context = set_context


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add command-line options for the plugin."""
    group = parser.getgroup("context-assert")
    group.addoption(
        "--context-assert-update",
        action="store_true",
        default=False,
        dest="context_assert_update",
        help="Update snapshots for the current context instead of comparing",
    )
    group.addoption(
        "--context-assert-update-all",
        action="store_true",
        default=False,
        dest="context_assert_update_all",
        help="Update all context snapshots (dangerous, removes other contexts)",
    )
    group.addoption(
        "--context-assert-context",
        action="store",
        default=None,
        dest="context_assert_context",
        help="Override context detection with a specific context key",
    )
    group.addoption(
        "--context-assert-dir",
        action="store",
        default=None,
        dest="context_assert_dir",
        help="Custom snapshot directory name (default: __snapshots__)",
    )

    parser.addini(
        "context_assert_dir",
        "Snapshot directory name",
        type="string",
        default="__snapshots__",
    )
    parser.addini(
        "context_assert_default_rtol",
        "Default relative tolerance for comparisons",
        type="string",
        default="1e-7",
    )
    parser.addini(
        "context_assert_default_atol",
        "Default absolute tolerance for comparisons",
        type="string",
        default="0",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Configure the plugin."""
    config.addinivalue_line(
        "markers",
        "context_assert_context(**kwargs): Override context for specific tests",
    )


class ContextAssertHookspec:
    """Hook specifications for pytest-context-assert plugin."""

    @staticmethod
    def pytest_context_assert_get_context() -> dict[str, Any] | None:
        """Hook to allow users to add custom context.

        Implement this hook in conftest.py to add custom context keys.

        Returns:
            Dictionary of custom context key-value pairs, or None.
        """


def pytest_addhooks(pluginmanager: pytest.PytestPluginManager) -> None:
    """Register custom hooks."""
    pluginmanager.add_hookspecs(ContextAssertHookspec)


@pytest.fixture
def context_assert(request: pytest.FixtureRequest) -> Generator[ContextAssert, None, None]:
    """Fixture providing context-aware assertions.

    Usage:
        def test_calculation(context_assert):
            result = my_function()
            context_assert(result)  # Compares with stored snapshot

        def test_array(context_assert):
            arr = compute_array()
            context_assert(arr, rtol=1e-5, atol=1e-8)

    The fixture automatically:
    - Detects the current platform, architecture, and BLAS library
    - Loads expected values from YAML snapshot files
    - Compares actual values with appropriate tolerances
    - Falls back to 'default' context if specific context not found

    Command-line options:
        --context-assert-update: Update snapshots instead of comparing
        --context-assert-context=KEY: Override auto-detected context
        --context-assert-dir=PATH: Custom snapshot directory
    """
    config = request.config

    update_snapshots = config.getoption("context_assert_update", False)
    context_override = config.getoption("context_assert_context", None)
    snapshot_dir = config.getoption("context_assert_dir") or config.getini("context_assert_dir")

    try:
        default_rtol = float(config.getini("context_assert_default_rtol"))
    except (ValueError, TypeError):
        default_rtol = 1e-7

    try:
        default_atol = float(config.getini("context_assert_default_atol"))
    except (ValueError, TypeError):
        default_atol = 0.0

    custom_context: dict[str, Any] = {}

    marker = request.node.get_closest_marker("context_assert_context")
    if marker:
        # Support both decorator-style (with _context_dict) and marker-style (with kwargs)
        if marker.kwargs.get("_context_dict"):
            custom_context.update(marker.kwargs["_context_dict"])
        elif marker.kwargs:
            # Filter out internal keys
            custom_context.update({k: v for k, v in marker.kwargs.items() if not k.startswith("_")})

    hook_result = request.config.hook.pytest_context_assert_get_context()
    if hook_result:
        custom_context.update(hook_result)

    if context_override:
        resolver = _OverrideResolver(context_override)
    else:
        resolver = ContextResolver(custom_context=custom_context if custom_context else None)

    ctx_assert = ContextAssert(
        request=request,
        resolver=resolver,
        snapshot_dir=snapshot_dir,
        default_rtol=default_rtol,
        default_atol=default_atol,
        update_snapshots=update_snapshots,
    )

    yield ctx_assert


class _OverrideResolver(ContextResolver):
    """A resolver that returns a fixed context key."""

    def __init__(self, context_key: str):
        super().__init__()
        self._override_key = context_key

    def get_context_key(self) -> str:
        return self._override_key


def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Modify test collection if needed."""


def pytest_report_header(config: pytest.Config) -> list[str]:
    """Add plugin info to pytest header."""
    lines = []

    resolver = ContextResolver()
    context_key = resolver.get_context_key()
    lines.append(f"context-assert: context={context_key}")

    if config.getoption("context_assert_update", False):
        lines.append("context-assert: UPDATE MODE ENABLED")

    return lines
