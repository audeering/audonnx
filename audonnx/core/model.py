import typing

import numpy as np
import onnxruntime

import audeer


class Model:
    r"""ONNX model.

    Args:
        path: path model file
        labels: list of labels (single-head)
            or dictionary with labels (multi-head)
        transform: callable object that transforms the raw signal
            into the desired representation

    """
    def __init__(
            self,
            path: str,
            *,
            labels: typing.Union[
                typing.Sequence[str],
                typing.Dict[str, typing.Sequence[str]],
            ] = None,
            transform: typing.Callable[[np.ndarray, int], np.ndarray] = None,
    ):

        self.path = audeer.safe_path(path)
        r"""Model path"""

        self.sess = onnxruntime.InferenceSession(self.path)
        r"""Interference session"""

        self._input = self.sess.get_inputs()[0]
        self._outputs = {
            output.name: output for output in self.sess.get_outputs()
        }

        self.labels = self._labels(labels)
        r"""Label names"""

        self.transform = transform
        r"""Transform object"""

    @property
    def input_name(self) -> str:
        r"""Name of input node."""
        return self._input.name

    @property
    def input_shape(self) -> list:
        r"""Shape of input node."""
        return self._input.shape

    @property
    def input_type(self) -> list:
        r"""Type of input node."""
        return self._input.type

    @property
    def output_names(self) -> typing.List[str]:
        r"""Names of output nodes."""
        return list(self._outputs)

    def output_shape(self, name: str) -> list:
        r"""Shape of output node."""
        return self._outputs[name].shape

    def output_type(self, name: str) -> list:
        r"""Type of output node."""
        return self._outputs[name].type

    def forward(
            self,
            signal: np.ndarray,
            sampling_rate: int,
            *,
            output_names: typing.Union[str, typing.Sequence[str]] = None,
    ) -> typing.Union[np.ndarray, typing.Dict[str, np.ndarray]]:
        r"""Compute raw predictions for one or more output nodes.

        Args:
            signal: input signal
            sampling_rate: sampling rate in Hz
            output_names: name of output or list with output names

        Returns:
            array if single output name or dictionary with
            output names as keys and predictions as values

        """
        if output_names is None:
            output_names = list(self.output_names)
            if len(output_names) == 1:  # pragma: no cover
                output_names = output_names[0]

        if self.transform is not None:
            signal = self.transform(signal, sampling_rate)

        y = signal.reshape([1] + list(signal.shape))
        y = self.sess.run(
            audeer.to_list(output_names),
            {self.input_name: y},
        )

        if isinstance(output_names, str):
            y = y[0]
        else:
            y = {
                name: values for name, values in zip(output_names, y)
            }

        return y

    def predict(
            self,
            signal: np.ndarray,
            sampling_rate: int,
            *,
            output_names: typing.Union[str, typing.Sequence[str]] = None,
    ) -> typing.Union[
        typing.Union[int, str],
        typing.Dict[str, typing.Union[int, str]]
    ]:
        r"""Predict labels for one or more output nodes.

        Args:
            signal: input signal
            sampling_rate: sampling rate in Hz
            output_names: name of output or list with output names

        Returns:
            label if single output name or dictionary with
            output names as keys and labels as values

        """
        if output_names is None:
            output_names = list(self.output_names)
            if len(output_names) == 1:  # pragma: no cover
                output_names = output_names[0]

        y = self.forward(
            signal,
            sampling_rate,
            output_names=audeer.to_list(output_names),
        )
        for name, values in y.items():
            y[name] = values.argmax()
            y[name] = self.labels[name][y[name]]

        if isinstance(output_names, str):
            y = y[output_names]

        return y

    def _labels(
            self,
            labels: typing.Optional[
                typing.Union[
                    typing.Sequence[str],
                    typing.Dict[str, typing.Sequence[str]],
                ]
            ],
    ):
        r"""Assign missing labels to '<output-name>-<dim>'"""
        labels = labels or {}
        if not isinstance(labels, dict):
            labels = {
                self.output_names[0]: labels
            }
        result = {}
        for name in self.output_names:
            dim = self.output_shape(name)[-1]
            if name in labels:
                if len(labels[name]) != dim:
                    raise ValueError(
                        f"Cannot assign "
                        f"{len(labels[name])} "
                        f"labels to output "
                        f"'{name}' "
                        f"with dimension "
                        f"{dim}."
                    )
                result[name] = labels[name]
            else:
                result[name] = [f'{name}-{idx}' for idx in range(dim)]
        return result
