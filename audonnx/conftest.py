from doctest import ELLIPSIS
from doctest import NORMALIZE_WHITESPACE

from sybil import Sybil
from sybil.parsers.rest import DocTestParser

from audonnx._doctest_linecache import install_doctest_linecache_patch


# Install the shared ``linecache`` patch so ``inspect.getsource``
# works for functions defined inline in module docstring doctests
# (e.g. the ``feature_addition`` example in ``audonnx.Function``).
install_doctest_linecache_patch()


# Collect doctests from module docstrings in ``audonnx/core/*.py``.
pytest_collect_file = Sybil(
    parsers=[DocTestParser(optionflags=ELLIPSIS | NORMALIZE_WHITESPACE)],
    patterns=["core/*.py"],
).pytest()
