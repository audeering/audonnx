import typing

import numpy as np

import audobject


class Function(audobject.Object):
    r"""Turn function into an :class:`audobject.Object`.

    Args:
        func: function that expects input signal and sampling rate as
            first two arguments
        func_args: additional arguments that will be passed to the function

    Examples:
        >>> object = Function(lambda x, sr: x.mean())
        >>> object
        {'$audonnx.core.function.Function': {'func': 'lambda x, sr: x.mean()', 'func_args': {}}}
        >>> object(np.array([1, 2, 3]), 10)
        2.0

    """  # noqa: E501
    @audobject.init_decorator(
        resolvers={
            'func': audobject.resolver.Function,
        }
    )
    def __init__(
            self,
            func: typing.Callable,
            *,
            func_args: typing.Dict[str, typing.Any] = None,
    ):
        self.func = func
        r"""Function"""
        self.func_args = func_args or {}
        r"""Additional function arguments"""

    def __call__(
            self,
            signal: np.ndarray,
            sampling_rate: int,
    ) -> np.ndarray:
        r"""Transform input signal.

        Args:
            signal: input signal
            sampling_rate: sampling rate in Hz

        Returns:
            transformed signal

        """
        return self.func(
            signal,
            sampling_rate,
            **self.func_args,
        )
