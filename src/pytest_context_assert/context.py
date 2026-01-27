"""Context detection logic for determining execution environment."""

from __future__ import annotations

import logging
import platform
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


def _detect_blas() -> str | None:
    """Detect the BLAS library in use by numpy if available."""
    try:
        import numpy as np

        if hasattr(np, "__config__"):
            config_info = np.__config__
            if hasattr(config_info, "show"):
                import io
                from contextlib import redirect_stdout

                f = io.StringIO()
                with redirect_stdout(f):
                    config_info.show()
                config_str = f.getvalue().lower()

                if "mkl" in config_str:
                    return "mkl"
                elif "openblas" in config_str:
                    return "openblas"
                elif "accelerate" in config_str:
                    return "accelerate"
                elif "blis" in config_str:
                    return "blis"

        if hasattr(np, "show_config"):
            import io
            from contextlib import redirect_stdout

            f = io.StringIO()
            with redirect_stdout(f):
                np.show_config()
            config_str = f.getvalue().lower()

            if "mkl" in config_str:
                return "mkl"
            elif "openblas" in config_str:
                return "openblas"
            elif "accelerate" in config_str:
                return "accelerate"
            elif "blis" in config_str:
                return "blis"
    except Exception as err:
        logging.warning(f"Failed to detect BLAS: {err}")

    return None


class ContextResolver:
    """Resolves the current execution context for snapshot matching."""

    def __init__(self, custom_context: dict[str, Any] | None = None):
        """Initialize with optional custom context overrides.

        Args:
            custom_context: Optional dictionary of custom context key-value pairs
                to merge with the detected context.
        """
        self._custom_context = custom_context or {}
        self._cached_context: dict[str, str] | None = None

    def get_platform(self) -> str:
        """Get the current platform (linux, darwin, windows)."""
        return platform.system().lower()

    def get_architecture(self) -> str:
        """Get the current CPU architecture."""
        machine = platform.machine().lower()
        if machine in ("x86_64", "amd64"):
            machine = "x86_64"
        elif machine in ("arm64", "aarch64"):
            machine = "arm64"
        elif machine in ("i386", "i686"):
            machine = "x86"
        return machine

    def get_blas(self) -> str | None:
        """Get the BLAS library in use, if detectable."""
        return _detect_blas()

    def get_python_version(self) -> str:
        """Get the Python major.minor version."""
        return f"{sys.version_info.major}.{sys.version_info.minor}"

    def get_context(self) -> dict[str, str]:
        """Get the full context dictionary.

        Returns:
            Dictionary containing all detected context values plus any custom
            context provided at initialization.
        """
        if self._cached_context is not None:
            return self._cached_context

        context = {
            "platform": self.get_platform(),
            "arch": self.get_architecture(),
        }

        blas = self.get_blas()
        if blas:
            context["blas"] = blas

        context.update({k: str(v) for k, v in self._custom_context.items()})

        self._cached_context = context
        return context

    def get_context_key(self) -> str:
        """Get a string key representing the current context.

        Returns:
            A hyphen-separated string of context values suitable for use
            as a key in snapshot files. Example: "darwin-arm64-accelerate"
        """
        context = self.get_context()

        parts = [context["platform"], context["arch"]]

        if "blas" in context:
            parts.append(context["blas"])

        for key in sorted(context.keys()):
            if key not in ("platform", "arch", "blas"):
                parts.append(f"{key}_{context[key]}")

        return "-".join(parts)

    def matches_context(self, stored_context: str) -> bool:
        """Check if a stored context key matches the current context.

        Args:
            stored_context: The context key string from a snapshot file.

        Returns:
            True if the stored context matches the current execution context.
        """
        return stored_context == self.get_context_key()

    def with_custom_context(self, **kwargs: Any) -> ContextResolver:
        """Create a new resolver with additional custom context.

        Args:
            **kwargs: Additional context key-value pairs.

        Returns:
            A new ContextResolver with the combined context.
        """
        merged = {**self._custom_context, **kwargs}
        return ContextResolver(custom_context=merged)
