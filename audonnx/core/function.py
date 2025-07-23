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
    must be provided as keys of the dictionary of inputs
    when calling this object.
    This is not required when only a single value is provided
    in the call and the function only has one argument.

    Args:
        func: function with variable arguments.
        func_args: fixed arguments that will be passed to the function

    Examples:
        >>> object = VariableFunction(lambda x, sr, offset: float(x.mean() + offset))
        >>> object
        {'$audonnx.core.function.VariableFunction': {'func': 'lambda x, sr, offset: float(x.mean() + offset)', 'func_args': {}}}
        >>> object({"x": np.array([1, 2, 3]), "sr": 10, "offset": 1})
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
        r"""Function arguments"""
        self._signature = inspect.signature(func)
        self._required_args = [
            name
            for name, p in self._signature.parameters.items()
            if p.default == inspect.Parameter.empty
        ]

    def _match_arguments(
        self, inputs: np.ndarray | dict[str, object]
    ) -> tuple[Sequence, dict]:
        r"""Match the inputs to the function arguments and keyword arguments."""
        args = ()
        kwargs = self.func_args
        if isinstance(inputs, dict):
            # Try to match the input names to the argument names
            matching_args = [
                arg
                for arg in self._required_args
                if arg in inputs or arg in self.func_args
            ]
            if len(matching_args) < len(self._required_args):
                raise ValueError(
                    "The input is missing at least one of the required arguments: "
                    f"{self._required_args}."
                )
            else:
                kwargs = self.func_args.copy()
                # Remove parameters from the inputs
                # that are not used for this function
                kwargs.update(
                    {k: v for k, v in inputs.items() if k in self._signature.parameters}
                )
        else:
            args = (inputs,)
            if len(self._required_args) > len(args) + len(kwargs):
                raise ValueError(
                    "The input is missing at least one of the required arguments: "
                    f"{self._required_args}."
                )
        return args, kwargs

    def __call__(self, inputs: np.ndarray | dict[str, object]) -> np.ndarray:
        r"""Apply function on inputs.

        When ``inputs`` is a dictionary,
        it must include the names of all required arguments
        of the function that are not already set in the :attr:`func_args`.
        Keys that don't appear as arguments in the function are ignored.

        Args:
            inputs: function inputs, either as a single input value
                or a dictionary mapping from function argument name to the value

        Returns:
            transformed inputs

        Raises:
            ValueError: if a required function argument is missing from the input
        """
        args, kwargs = self._match_arguments(inputs)
        return self.func(*args, **kwargs)
