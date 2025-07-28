from collections.abc import Callable
import inspect

import numpy as np

import audobject


class Function(audobject.Object):
    r"""Turn function into an :class:`audobject.Object`.

    Args:
        func: function that expects input signal and sampling rate as
            first two arguments
        func_args: default values for arguments that will be passed to the function
        fixed_signature: whether the function expects only two positional arguments:
            signal and sampling rate.
            If ``False``, any arguments can be used but
            the argument names must match exactly when calling this object

    Examples:
        >>> object = Function(lambda x, sr: float(x.mean()))
        >>> object
        {'$audonnx.core.function.Function': {'func': 'lambda x, sr: float(x.mean())', 'func_args': {}, 'fixed_signature': True}}
        >>> object(np.array([1, 2, 3]), 10)
        2.0
        >>> def feature_addition(x, offset=1):
        ...     return float(x.mean() + offset)
        >>> object = Function(feature_addition, fixed_signature=False)
        >>> object
        {'$audonnx.core.function.Function': {'func': 'def feature_addition(x, offset=1):\n    return float(x.mean() + offset)\n', 'func_args': {}, 'fixed_signature': False}}
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
        fixed_signature: bool = True,
    ):
        self.func = func
        r"""Function"""
        self.func_args = func_args or {}
        r"""Default set function arguments"""
        self.fixed_signature = fixed_signature
        r"""Whether the function has fixed arguments of signal and sampling rate"""
        self._signature = inspect.signature(func)
        self.parameters = self._signature.parameters
        r"""Function parameters"""

    def _match_args(
        self, inputs: np.ndarray | dict[str, object], sampling_rate: int | None = None
    ) -> tuple[tuple, dict[str, object]] | None:
        r"""Return the matching positional and keyword arguments for the function."""
        # Fixed transform signature with two args
        # one for signal, one for sampling_rate
        if self.fixed_signature:
            if isinstance(inputs, np.ndarray):
                signal = inputs
            elif "signal" not in inputs:
                return None
            else:
                signal = inputs["signal"]
            if sampling_rate is None:
                return None
            else:
                return (signal, sampling_rate), self.func_args
        # Custom function signature
        # We need to pass all arguments that occur in the transform's parameters
        else:
            kwargs = self.func_args
            # In case there is only one input,
            # it is passed as a positional argument
            if isinstance(inputs, np.ndarray):
                args = (inputs,)
            else:
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
        if res is None:
            raise ValueError(error_message)
        args, kwargs = res
        try:
            bound = self._signature.bind(*args, **kwargs)
        except TypeError as e:
            raise ValueError(error_message) from e
        return self.func(*bound.args, **bound.kwargs)
