"""Helper functions for getting dynamic context values.

These helpers make it easy to use environment variables, platform detection,
and other system information in the @set_context decorator.

Example usage:
    from pytest_context_assert import set_context
    from pytest_context_assert.helpers import env, get_platform, get_arch, get_openblas_coretype

    @set_context({
        "openblas_coretype": get_openblas_coretype(),
        "platform": get_platform(),
        "arch": get_arch(),
        "custom_env": env("MY_CUSTOM_VAR", "default_value"),
    })
    def test_my_function(context_assert):
        ...
"""

from __future__ import annotations

import os
import platform as platform_module
from typing import Any


def env(key: str, default: str | None = None) -> str | None:
    """Get an environment variable value.

    Args:
        key: The environment variable name.
        default: Default value if not set.

    Returns:
        The environment variable value or default.

    Example:
        @set_context({"openblas": env("OPENBLAS_CORETYPE", "haswell")})
        def test_func(context_assert):
            ...
    """
    return os.environ.get(key, default)


def env_or_skip(key: str) -> str:
    """Get an environment variable or return a skip marker value.

    If the environment variable is not set, returns "__SKIP__" which
    can be used to conditionally skip context-specific tests.

    Args:
        key: The environment variable name.

    Returns:
        The environment variable value or "__SKIP__".
    """
    return os.environ.get(key, "__SKIP__")


def get_platform() -> str:
    """Get the current platform (linux, darwin, windows).

    Returns:
        Lowercase platform name.

    Example:
        @set_context({"os": get_platform()})
        def test_func(context_assert):
            ...
    """
    return platform_module.system().lower()


def get_arch() -> str:
    """Get the current architecture.

    Normalizes common architecture names:
    - x86_64, AMD64 -> x86_64
    - arm64, aarch64 -> arm64

    Returns:
        Normalized architecture name.
    """
    machine = platform_module.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x86_64"
    elif machine in ("arm64", "aarch64"):
        return "arm64"
    return machine


def get_python_version() -> str:
    """Get the Python version as major.minor string.

    Returns:
        Python version like "3.10" or "3.11".
    """
    import sys

    return f"{sys.version_info.major}.{sys.version_info.minor}"


def get_openblas_coretype() -> str | None:
    """Get the OpenBLAS core type from environment.

    OpenBLAS sets OPENBLAS_CORETYPE to indicate the CPU target.
    Common values: HASWELL, ZEN, SKYLAKEX, NEOVERSEN1, etc.

    Returns:
        The OPENBLAS_CORETYPE value (lowercase) or None.
    """
    coretype = os.environ.get("OPENBLAS_CORETYPE")
    return coretype.lower() if coretype else None


def get_mkl_verbose() -> str | None:
    """Get MKL threading/interface info from MKL_VERBOSE if enabled.

    Returns:
        MKL configuration string or None.
    """
    return os.environ.get("MKL_VERBOSE")


def get_omp_num_threads() -> str | None:
    """Get the OMP_NUM_THREADS value.

    Returns:
        Number of OpenMP threads or None.
    """
    return os.environ.get("OMP_NUM_THREADS")


def get_blas_num_threads() -> str | None:
    """Get BLAS thread count from common environment variables.

    Checks: OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, OMP_NUM_THREADS

    Returns:
        Number of BLAS threads or None.
    """
    for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        value = os.environ.get(var)
        if value:
            return value
    return None


def get_conda_env() -> str | None:
    """Get the current conda environment name.

    Returns:
        Conda environment name or None.
    """
    return os.environ.get("CONDA_DEFAULT_ENV")


def get_ci_platform() -> str | None:
    """Detect the CI platform from environment variables.

    Returns:
        CI platform name (github, gitlab, jenkins, circleci, travis) or None.
    """
    if os.environ.get("GITHUB_ACTIONS"):
        return "github"
    elif os.environ.get("GITLAB_CI"):
        return "gitlab"
    elif os.environ.get("JENKINS_URL"):
        return "jenkins"
    elif os.environ.get("CIRCLECI"):
        return "circleci"
    elif os.environ.get("TRAVIS"):
        return "travis"
    return None


def context_from_env(*env_vars: str, prefix: str = "") -> dict[str, str]:
    """Build a context dict from multiple environment variables.

    Only includes variables that are set.

    Args:
        *env_vars: Environment variable names to include.
        prefix: Optional prefix to strip from variable names in the dict keys.

    Returns:
        Dictionary with env var names (lowercase, prefix stripped) as keys.

    Example:
        @set_context(context_from_env(
            "OPENBLAS_CORETYPE",
            "OPENBLAS_NUM_THREADS",
            prefix="OPENBLAS_"
        ))
        def test_func(context_assert):
            # Results in {"coretype": "haswell", "num_threads": "4"}
            ...
    """
    result: dict[str, str] = {}
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            key = var.lower()
            if prefix and key.startswith(prefix.lower()):
                key = key[len(prefix) :]
            result[key] = value
    return result


def detect_numpy_blas() -> str | None:
    """Detect the BLAS library used by numpy.

    Returns:
        BLAS library name (mkl, openblas, accelerate, blis) or None.
    """
    try:
        import numpy as np

        config = np.__config__

        if hasattr(config, "blas_ilp64_opt_info"):
            blas_info = config.blas_ilp64_opt_info
        elif hasattr(config, "blas_opt_info"):
            blas_info = config.blas_opt_info
        else:
            # Try newer numpy config API
            if hasattr(config, "show"):
                import io
                import sys

                old_stdout = sys.stdout
                sys.stdout = buffer = io.StringIO()
                config.show()
                sys.stdout = old_stdout
                output = buffer.getvalue().lower()

                if "mkl" in output:
                    return "mkl"
                elif "openblas" in output:
                    return "openblas"
                elif "accelerate" in output:
                    return "accelerate"
                elif "blis" in output:
                    return "blis"
            return None

        blas_info_str = str(blas_info).lower()
        if "mkl" in blas_info_str:
            return "mkl"
        elif "openblas" in blas_info_str:
            return "openblas"
        elif "accelerate" in blas_info_str:
            return "accelerate"
        elif "blis" in blas_info_str:
            return "blis"

    except (ImportError, AttributeError):
        pass

    return None


def build_context(
    include_platform: bool = True,
    include_arch: bool = True,
    include_blas: bool = False,
    include_python: bool = False,
    env_vars: list[str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build a comprehensive context dictionary.

    Args:
        include_platform: Include platform (os) in context.
        include_arch: Include architecture in context.
        include_blas: Include detected BLAS library.
        include_python: Include Python version.
        env_vars: List of environment variables to include.
        **extra: Additional key-value pairs to include.

    Returns:
        Context dictionary with requested values.

    Example:
        @set_context(build_context(
            include_blas=True,
            env_vars=["OPENBLAS_CORETYPE"],
            custom_key="custom_value"
        ))
        def test_func(context_assert):
            ...
    """
    ctx: dict[str, Any] = {}

    if include_platform:
        ctx["platform"] = get_platform()

    if include_arch:
        ctx["arch"] = get_arch()

    if include_blas:
        blas = detect_numpy_blas()
        if blas:
            ctx["blas"] = blas

    if include_python:
        ctx["python"] = get_python_version()

    if env_vars:
        for var in env_vars:
            value = os.environ.get(var)
            if value:
                ctx[var.lower()] = value

    ctx.update(extra)

    return ctx
