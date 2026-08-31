"""geometry.cad_export.kernel: locating an available CAD kernel.

CAD is an optional dependency. The analysis stack does not need it, and the
OpenCascade build behind it is several hundred megabytes, so a user who only
runs simulations should not be made to download it.

build123d is preferred and cadquery is the fallback. Both wrap the same OCCT
kernel, so either can build the B-rep and write STEP; only the calling code
differs.
"""

from __future__ import annotations

from dataclasses import dataclass

INSTALL_HINT = (
    "CAD export needs an OpenCascade based kernel, which is an optional "
    "dependency. Install it with:\n"
    "    env -u PYTHONPATH .venv/bin/pip install -r requirements-cad.txt\n"
    "The analysis stack does not require it."
)


@dataclass(frozen=True)
class Kernel:
    name: str
    module: object


def find_kernel() -> Kernel | None:
    """Return the first available CAD kernel, or None."""
    try:
        import build123d
        return Kernel("build123d", build123d)
    except ImportError:
        pass
    try:
        import cadquery
        return Kernel("cadquery", cadquery)
    except ImportError:
        pass
    return None


def require_kernel() -> Kernel:
    kernel = find_kernel()
    if kernel is None:
        raise ModuleNotFoundError(INSTALL_HINT)
    return kernel


def kernel_available() -> bool:
    return find_kernel() is not None
