from audonnx.core.api import load
from audonnx.core.function import Function
from audonnx.core.model import Model
from audonnx.core.node import (
    InputNode,
    OutputNode,
)


__all__ = []


# Dynamically get the version of the installed module
try:
    import pkg_resources
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:  # pragma: no cover
    pkg_resources = None  # pragma: no cover
finally:
    del pkg_resources
