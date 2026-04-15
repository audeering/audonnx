"""Shared helper to make ``inspect.getsource`` work in doctest examples.

``audonnx.Function`` uses ``audobject.resolver.Function``, which calls
``inspect.getsource(func)`` to serialise the source code of a function.
``inspect.getsource`` reads source lines from ``linecache`` keyed by
``func.__code__.co_filename``.

Python's ``doctest`` module compiles each example with a synthetic
filename (``<doctest name[N]>``) but does *not* register the source
in ``linecache``, so ``inspect.getsource`` raises ``OSError`` for
functions defined inline in doctests. IPython/Jupyter avoids this by
registering cell sources in ``linecache`` - we do the same here by
wrapping ``DocTestRunner.run``.

sybil feeds every example through ``DocTestRunner.run`` as its own
single-example ``DocTest``, and ``doctest`` builds the filename as
``<doctest <test.name>[<examplenum>]>``. If we left ``test.name``
alone, every example from the same rst (or Python) file would end up
with the same filename (``[examplenum]`` is always 0) and each
linecache entry would overwrite the previous one. We therefore make
the name unique per example using a counter stored on the runner
instance.

The patch is needed from both ``docs/conftest.py`` (for usage.rst
doctests) and ``audonnx/conftest.py`` (for module docstring
doctests). Centralising it here keeps the two call sites in sync.
"""

import doctest
import linecache


_PATCH_ATTR = "_audonnx_linecache_patch"


def install_doctest_linecache_patch() -> None:
    """Patch ``doctest.DocTestRunner.run`` to populate ``linecache``.

    The patch is guarded by a sentinel attribute so that it is only
    applied once, regardless of how many times this function is
    imported or called.
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
            linecache.cache[filename] = (len(example.source), None, lines, filename)
        return orig_run(self, test, *args, **kwargs)

    setattr(run_with_linecache, _PATCH_ATTR, True)
    doctest.DocTestRunner.run = run_with_linecache
