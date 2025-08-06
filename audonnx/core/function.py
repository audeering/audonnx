from collections.abc import Callable
import inspect

import numpy as np

import audobject


class Function(audobject.Object):
    r"""Turn function into an :class:`audobject.Object`.

    Args:
        func: function to apply
        func_args: default values for arguments that will be passed to the function

    Examples:
        >>> object = Function(lambda x, sr: float(x.mean()))
        >>> object
        {'$audonnx.core.function.Function': {'func': 'lambda x, sr: float(x.mean())', 'func_args': {}}}
        >>> object(np.array([1, 2, 3]), 10)
        2.0
        >>> def feature_addition(x, offset=1):
        ...     return float(x.mean() + offset)
        >>> object = Function(feature_addition)
        >>> object
        {'$audonnx.core.function.Function': {'func': 'def feature_addition(x, offset=1):\n    return float(x.mean() + offset)\n', 'func_args': {}}}
        >>> object({"x": np.array([1, 2, 3]), "offset": 1})
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
        func_args: dict[str, object] | None = None,
    ):
        self.func = func
        r"""Function"""
        self.func_args = func_args or {}
        r"""Default set function arguments"""
        self._signature = inspect.signature(func)
        self.parameters = self._signature.parameters
        r"""Function parameters"""

    def _match_args(
        self, inputs: np.ndarray | dict[str, object], sampling_rate: int | None = None
    ) -> tuple[tuple, dict[str, object]]:
        r"""Return the matching positional and keyword arguments for the function."""
        kwargs = self.func_args
        # In case there is only one input,
        # it is passed as a positional argument
        # alongside the sampling rate, if provided
        if isinstance(inputs, np.ndarray):
            if sampling_rate is None:
                args = (inputs,)
            else:
                args = (inputs, sampling_rate)
            return args, kwargs
        # In case the input is a dictionary,
        # we add all inputs that could be used
        # as keyword arguments
        args = ()
        kwargs = kwargs | inputs
        if sampling_rate is not None:
            kwargs["sampling_rate"] = sampling_rate
        # Filter out arguments that are not required for this transformation
        kwargs = {k: v for k, v in kwargs.items() if k in self.parameters}
        return args, kwargs

    def __call__(
        self, inputs: np.ndarray | dict[str, object], sampling_rate: int | None = None
    ) -> np.ndarray:
        r"""Apply function on inputs.

        All required arguments of the function
        that are not already set in the :attr:`func_args`
        must be provided.

        If ``inputs`` is a :class:`numpy.ndarray`,
        it is passed as a positional argument to the function,
        and the ``sampling_rate``, if provided,
        is passed as the second positional argument.

        If ``inputs`` is a dictionary,
        the dictionary entries
        are matched to the function arguments
        and passed as keyword arguments.
        Keys that don't occur in the function arguments
        are ignored.

        Args:
            inputs: input signal
                or dictionary with multiple inputs
            sampling_rate: sampling rate in Hz

        Returns:
            transformed inputs

        Raises:
            ValueError: if the passed input could not be matched
                to the function arguments
        """
        res = self._match_args(inputs, sampling_rate)
        error_message = (
            "The passed inputs and sampling rate "
            "could not be matched to the function arguments."
        )
        args, kwargs = res
        try:
            bound = self._signature.bind(*args, **kwargs)
        except TypeError as e:
            raise ValueError(error_message) from e
        return self.func(*bound.args, **bound.kwargs)
