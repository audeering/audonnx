from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence

import numpy as np


Device = (
    str
    | tuple[str, dict]
    | Sequence[str | tuple[str, dict]]
)


Labels = (
    Sequence[str]
    | dict[
        str,
        Sequence[str] | None,
    ]
)


Transform = (
    Callable[
        [np.ndarray, int],
        np.ndarray,
    ]
    | dict[
        str,
        Callable[
            [np.ndarray, int],
            np.ndarray,
        ] | None,
    ]
)
