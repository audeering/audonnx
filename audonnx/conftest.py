import doctest
from doctest import ELLIPSIS
from doctest import NORMALIZE_WHITESPACE
import linecache

from sybil import Sybil
from sybil.parsers.rest import DocTestParser


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
# functions defined inline in doctests (e.g. the ``feature_addition``
# example in :func:`audonnx.Function`'s docstring).
#
# The same patch is also installed by ``docs/conftest.py``; both
# conftest files guard the installation with a sentinel attribute so
# that the wrapper is applied exactly once, regardless of which
# conftest pytest imports first.
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


# Collect doctests from module docstrings in ``audonnx/core/*.py``.
pytest_collect_file = Sybil(
    parsers=[DocTestParser(optionflags=ELLIPSIS | NORMALIZE_WHITESPACE)],
    patterns=["core/*.py"],
).pytest()
