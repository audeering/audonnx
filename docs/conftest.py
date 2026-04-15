import doctest
from doctest import ELLIPSIS
from doctest import NORMALIZE_WHITESPACE
import linecache
import os
import shutil
import warnings

import pytest
from sybil import Sybil
from sybil.parsers.rest import DocTestParser
from sybil.parsers.rest import PythonCodeBlockParser
from sybil.parsers.rest import SkipParser


# --- Make ``inspect.getsource`` work for doctest-defined functions -------
#
# ``audonnx.Function`` uses ``audobject.resolver.Function``, which calls
# ``inspect.getsource(func)`` to serialise the source code of a function.
# ``inspect.getsource`` reads source lines from ``linecache`` keyed by
# ``func.__code__.co_filename``.
#
# Python's ``doctest`` module compiles each example with a synthetic
# filename (``<doctest name[N]>``) but does *not* register the source
# in ``linecache``, so ``inspect.getsource`` raises ``OSError`` for
# functions defined inline in doctests. IPython/Jupyter avoids this by
# registering cell sources in ``linecache`` — we do the same here by
# wrapping ``DocTestRunner.run``.
#
# sybil feeds every example through ``DocTestRunner.run`` as its own
# single-example ``DocTest``, and ``doctest`` builds the filename as
# ``<doctest <test.name>[<examplenum>]>``. If we left ``test.name``
# alone, every example from the same rst file would end up with the
# same filename (``[examplenum]`` is always 0) and each linecache
# entry would overwrite the previous one. We therefore make the name
# unique per example using a counter stored on the runner instance.
#
# The patch is made idempotent via a sentinel attribute so that
# ``audonnx/conftest.py`` (which needs the same patch for module
# docstring doctests) can apply it too without double-wrapping,
# regardless of conftest import order.
_PATCH_ATTR = "_audonnx_linecache_patch"

if not getattr(doctest.DocTestRunner.run, _PATCH_ATTR, False):
    _orig_doctest_run = doctest.DocTestRunner.run

    def _run_with_linecache(self, test, *args, **kwargs):
        counter = getattr(self, "_example_counter", 0) + 1
        self._example_counter = counter
        test.name = f"{test.name}-{counter}"
        for examplenum, example in enumerate(test.examples):
            filename = "<doctest %s[%d]>" % (test.name, examplenum)
            lines = example.source.splitlines(keepends=True)
            linecache.cache[filename] = (len(example.source), None, lines, filename)
        return _orig_doctest_run(self, test, *args, **kwargs)

    setattr(_run_with_linecache, _PATCH_ATTR, True)
    doctest.DocTestRunner.run = _run_with_linecache


# Path to the test audio file shipped with the documentation.
_TEST_WAV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "_static",
    "test.wav",
)


def imports(namespace):
    """Provide deterministic setup for the doctest namespace."""
    # Silence the legacy torch -> onnx exporter deprecation warning
    # that would otherwise pollute doctest output.
    warnings.simplefilter(action="ignore", category=DeprecationWarning)
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
