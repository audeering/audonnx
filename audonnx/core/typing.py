from collections.abc import Callable
from collections.abc import Sequence

import numpy as np

from audonnx.core.function import Function


Device = str | tuple[str, dict] | Sequence[str | tuple[str, dict]]
Labels = Sequence[str] | dict[str, Sequence[str] | None]
_SignalTransform = Callable[[np.ndarray, int], np.ndarray]

Transform = _SignalTransform | Function | dict[str, _SignalTransform | Function]
