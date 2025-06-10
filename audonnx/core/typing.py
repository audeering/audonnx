from collections.abc import Callable
from collections.abc import Sequence
from typing import Optional
from typing import Union

import numpy as np


Device = Union[
    str,
    tuple[str, dict],
    Sequence[Union[str, tuple[str, dict]]],
]


Labels = Union[
    Sequence[str],
    dict[str, Optional[Sequence[str]]],
]


Transform = Union[
    Callable[
        [np.ndarray, int],
        np.ndarray,
    ],
    dict[
        str,
        Optional[Callable[[np.ndarray, int], np.ndarray]],
    ],
]
