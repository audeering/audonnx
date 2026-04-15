from doctest import ELLIPSIS
from doctest import NORMALIZE_WHITESPACE
import os
import shutil

import pytest
from sybil import Sybil
from sybil.parsers.rest import DocTestParser
from sybil.parsers.rest import PythonCodeBlockParser
from sybil.parsers.rest import SkipParser

from audonnx._doctest_linecache import install_doctest_linecache_patch


# Install the shared ``linecache`` patch so ``inspect.getsource``
# works for functions defined inline in doctests (needed by
# ``audonnx.Function``/``audobject.resolver.Function`` when they
# serialise a function's source code).
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
