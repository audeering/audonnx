"""Sybil-based doctest collection for ``docs/usage.rst``.

Also provides ``install_doctest_linecache_patch()`` which is shared
with ``audonnx/conftest.py``.
"""

import doctest
from doctest import ELLIPSIS
from doctest import NORMALIZE_WHITESPACE
import linecache
import os
import shutil

import pytest
from sybil import Sybil
from sybil.parsers.rest import DocTestParser
from sybil.parsers.rest import PythonCodeBlockParser
from sybil.parsers.rest import SkipParser


# ---------------------------------------------------------------------------
# Shared helper: make ``inspect.getsource`` work inside doctest examples
# ---------------------------------------------------------------------------
#
# ``audonnx.Function`` uses ``audobject.resolver.Function``, which calls
# ``inspect.getsource(func)`` to serialise a function's source code.
# ``inspect.getsource`` reads source lines from ``linecache`` keyed by
# ``func.__code__.co_filename``.
#
# Python's ``doctest`` module compiles each example with a synthetic
# filename (``<doctest name[N]>``) but does *not* register the source
# in ``linecache``, so ``inspect.getsource`` raises ``OSError`` for
# functions defined inline in doctests.  We fix this by wrapping
# ``DocTestRunner.run`` to populate ``linecache`` before each run.

_PATCH_ATTR = "_audonnx_linecache_patch"


def install_doctest_linecache_patch() -> None:
    """Patch ``doctest.DocTestRunner.run`` to populate ``linecache``.

    The patch is guarded by a sentinel attribute so that it is only
    applied once, regardless of how many times this function is called.
    """
    if getattr(doctest.DocTestRunner.run, _PATCH_ATTR, False):
        return

    orig_run = doctest.DocTestRunner.run

    def run_with_linecache(self, test, *args, **kwargs):
        counter = getattr(self, "_example_counter", 0) + 1
        self._example_counter = counter
        test.name = f"{test.name}-{counter}"
        for examplenum, example in enumerate(test.examples):
            filename = "<doctest %s[%d]>" % (test.name, examplenum)
            lines = example.source.splitlines(keepends=True)
            linecache.cache[filename] = (
                len(example.source),
                None,
                lines,
                filename,
            )
        return orig_run(self, test, *args, **kwargs)

    setattr(run_with_linecache, _PATCH_ATTR, True)
    doctest.DocTestRunner.run = run_with_linecache


# Install the patch so ``inspect.getsource`` works for functions
# defined inline in doctests.
install_doctest_linecache_patch()


# Path to the test audio file shipped with the documentation.
_TEST_WAV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "_static",
    "test.wav",
)


def imports(namespace):
    """Provide deterministic setup for the doctest namespace."""
    # Seed torch so randomly initialised model weights
    # (and the ONNX models exported from them) are deterministic
    # across runs and platforms.
    import torch

    torch.manual_seed(0)


@pytest.fixture(scope="module")
def run_in_tmpdir(tmpdir_factory):
    """Move to a persistent tmpdir for execution of a whole file.

    Copies the documentation's test audio file into the tmpdir
    so examples can refer to it via the relative path ``test.wav``.
    """
    tmpdir = tmpdir_factory.mktemp("tmp")
    shutil.copy(_TEST_WAV, os.path.join(tmpdir, "test.wav"))
    current_dir = os.getcwd()
    os.chdir(tmpdir)

    yield

    os.chdir(current_dir)


# Collect doctests
pytest_collect_file = Sybil(
    parsers=[
        DocTestParser(optionflags=ELLIPSIS | NORMALIZE_WHITESPACE),
        PythonCodeBlockParser(),
        SkipParser(),
    ],
    patterns=["usage.rst"],
    fixtures=["run_in_tmpdir"],
    setup=imports,
).pytest()
