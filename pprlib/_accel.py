"""Optional JIT acceleration.

Some kernels (the alias-table build and the loop-erased walk behind
spanning-forest sampling) are inherently sequential, so they cannot be
vectorised with NumPy. If numba is installed they are JIT-compiled. Otherwise
the same Python code runs unchanged, only slower.

Set the environment variable ``PPRLIB_DISABLE_NUMBA=1`` to force pure Python.
"""

import os

HAVE_NUMBA = False

if os.environ.get("PPRLIB_DISABLE_NUMBA", "0") != "1":
    try:
        from numba import njit as _numba_njit

        HAVE_NUMBA = True
    except ImportError:  # pragma: no cover - depends on the environment
        pass


def njit(*args, **kwargs):
    """``numba.njit`` if available, otherwise a no-op decorator."""
    if HAVE_NUMBA:
        return _numba_njit(*args, **kwargs)
    if args and callable(args[0]):
        return args[0]
    return lambda f: f
