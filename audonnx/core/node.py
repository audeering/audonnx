import typing

import numpy as np
import oyaml as yaml

from audonnx.core.function import Function


class InputNode:
    r"""Input node.

    Args:
        shape: list with dimensions
        dtype: data type
        transform: callable object that transforms the raw signal
            into the desired representation

    """
    def __init__(
            self,
            shape: typing.List[typing.Union[int]],
            dtype: str,
            transform: typing.Optional[
                typing.Callable[
                    [np.ndarray, int],
                    np.ndarray,
                ]
            ],
    ):
        self.dtype = dtype
        r"""Data type of node"""

        self.shape = shape
        r"""Shape of node"""

        self.transform = transform
        r"""Transform object"""

    def _dict(self) -> typing.Dict:

        if self.transform is None:
            transform = 'None'
        else:
            transform = f'{self.transform.__class__.__module__}' \
                        f'.' \
                        f'{self.transform.__class__.__name__}'

        if isinstance(self.transform, Function):
            transform += f'({self.transform.func.__name__})'

        return {
            'shape': self.shape,
            'dtype': self.dtype,
            'transform': transform,
        }

    def __repr__(self):
        r"""Printable presentation of input node."""
        return yaml.dump(self._dict(), default_flow_style=True).strip()


class OutputNode:
    r"""Output node.

    Args:
        shape: list with dimensions
        dtype: data type
        labels: list with names of last non-dynamic output dimension

    """
    def __init__(
            self,
            shape: typing.List[int],
            dtype: str,
            labels: typing.List[str],
    ):
        self.shape = shape
        r"""Shape of node"""

        self.dtype = dtype
        r"""Data type of node"""

        self.labels = labels
        r"""Labels of last non-dynamic output dimension."""

    def _dict(self) -> typing.Dict:

        if len(self.labels) > 6:
            labels = self.labels[:3] + ['(...)'] + self.labels[-3:]
        else:
            labels = self.labels

        return {
            'shape': self.shape,
            'dtype': self.dtype,
            'labels': labels,
        }

    def __repr__(self):
        r"""Printable presentation of output node."""
        return yaml.dump(self._dict(), default_flow_style=True).strip()
