from collections.abc import Callable
from collections.abc import Sequence
import inspect

import numpy as np

import audobject


class Function(audobject.Object):
    r"""Turn function into an :class:`audobject.Object`.

    Args:
        func: function that expects input signal and sampling rate as
            first two arguments
        func_args: additional arguments that will be passed to the function

    Examples:
        >>> object = Function(lambda x, sr: float(x.mean()))
        >>> object
        {'$audonnx.core.function.Function': {'func': 'lambda x, sr: float(x.mean())', 'func_args': {}}}
        >>> object(np.array([1, 2, 3]), 10)
        2.0

    """  # noqa: E501

    @audobject.init_decorator(
        resolvers={
            "func": audobject.resolver.Function,
        }
    )
    def __init__(
        self,
        func: Callable,
        *,
        func_args: dict[str, object] | None = None,
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


class VariableFunction(audobject.Object):
    r"""Turn function with variable arguments into an :class:`audobject.Object`.

    The exact names of the function's arguments
    must be used when calling this
    object with keyword arguments.

    Args:
        func: function with variable arguments.
        default_args: default arguments that will be passed to the function.
            Can be overridden by arguments passed in the object call

    Examples:
        >>> object = VariableFunction(lambda x, sr, offset: float(x.mean() + offset))
        >>> object
        {'$audonnx.core.function.VariableFunction': {'func': 'lambda x, sr, offset: float(x.mean() + offset)', 'default_args': {}}}
        >>> object(x=np.array([1, 2, 3]), sr=10, offset=1)
        3.0

    """  # noqa: E501

    @audobject.init_decorator(
        resolvers={
            "func": audobject.resolver.Function,
        }
    )
    def __init__(
        self,
        func: Callable,
        *,
        default_args: dict[str, object] | None = None,
    ):
        self.func = func
        r"""Function"""
        self.default_args = default_args or {}
        r"""Default set function arguments"""
        self._signature = inspect.signature(func)
        self.parameters = self._signature.parameters
        r"""Function parameters"""

    def _match_arguments(self, *args, **kwargs) -> tuple[Sequence, dict]:
        r"""Match the inputs to the function arguments and keyword arguments.

        Args:
            args: positional arguments
            kwargs: keyword arguments

        Raises:
            TypeError: if a required function argument is missing
        """
        combined_kwargs = self.default_args.copy()
        # Passed kwargs should overwrite the defaults in func_args
        combined_kwargs.update(**kwargs)
        bound = self._signature.bind(*args, **combined_kwargs)
        return bound.args, bound.kwargs

    def __call__(self, *args, **kwargs) -> np.ndarray:
        r"""Apply function on inputs.

        All required arguments of the function
        that are not already set in the :attr:`func_args`
        must be provided.
        Arguments that don't appear in the function are ignored.

        Args:
            args: positional arguments
            kwargs: keyword arguments

        Returns:
            transformed inputs

        Raises:
            TypeError: if a required function argument is missing
        """
        args, kwargs = self._match_arguments(*args, **kwargs)
        return self.func(*args, **kwargs)
