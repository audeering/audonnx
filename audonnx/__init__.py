from audonnx.core.api import load
from audonnx.core.function import Function
from audonnx.core.model import Model
from audonnx.core.node import InputNode
from audonnx.core.node import OutputNode
from audonnx.core.ort import device_to_providers


__all__ = []


# Dynamically get the version of the installed module
try:
    import importlib.metadata
    __version__ = importlib.metadata.version(__name__)
except Exception:  # pragma: no cover
    importlib = None  # pragma: no cover
finally:
    del importlib
