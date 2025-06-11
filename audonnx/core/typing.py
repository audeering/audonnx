from collections.abc import Callable
from collections.abc import Sequence

import numpy as np


Device = str | tuple[str, dict] | Sequence[str | tuple[str, dict]]
Labels = Sequence[str] | dict[str, Sequence[str] | None]
_Transform = Callable[[np.ndarray, int], np.ndarray]
Transform = _Transform | dict[str, _Transform]
