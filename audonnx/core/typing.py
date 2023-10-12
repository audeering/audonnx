import typing

import numpy as np


Device = typing.Union[
    str,
    typing.Tuple[str, typing.Dict],
    typing.Sequence[
        typing.Union[str, typing.Tuple[str, typing.Dict]]],
]


Labels = typing.Union[
    typing.Sequence[str],
    typing.Dict[
        str,
        typing.Optional[
            typing.Sequence[str]
        ],
    ],
]


Transform = typing.Union[
    typing.Callable[
        [np.ndarray, int],
        np.ndarray,
    ],
    typing.Dict[
        str,
        typing.Optional[
            typing.Callable[
                [np.ndarray, int],
                np.ndarray,
            ]
        ],
    ],
]
