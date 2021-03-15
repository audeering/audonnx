import typing

import numpy as np
import onnxruntime

import audeer


class Model:

    def __init__(
            self,
            path: str,
            *,
            device: str = 'cpu',
            labels: typing.Sequence[str] = None,
            transform: typing.Callable[[np.ndarray, int], np.ndarray] = None,
    ):
        self.device = device
        r"""Device string"""

        self.labels = labels
        r"""Label names"""

        self.path = audeer.safe_path(path)
        r"""Model path"""

        self.sess = onnxruntime.InferenceSession(self.path)
        r"""Interference session"""

        self.transform = transform
        r"""Transform object"""

    def forward(
            self,
            signal: np.ndarray,
            sampling_rate: int,
            *,
            input_name: str = None,
            output_name: str = None,
    ) -> np.ndarray:

        if self.transform is not None:
            y = self.transform(signal, sampling_rate)
        else:
            y = signal

        y = y.reshape([1] + list(y.shape))
        if self.device != 'cpu':  # pragma: no cover
            # TODO: not tested
            y = onnxruntime.OrtValue.ortvalue_from_numpy(
                y, 'cuda', int(self.device[-1]),
            )

        if input_name is None:
            input_name = self.sess.get_inputs()[0].name
        if output_name is None:
            output_name = self.sess.get_outputs()[0].name

        y = self.sess.run([output_name], {input_name: y})

        return y[0]

    def predict(
            self,
            signal: np.ndarray,
            sampling_rate: int,
            *,
            input_name: str = None,
            output_name: str = None,
    ) -> typing.Union[int, str]:

        y = self.forward(
            signal,
            sampling_rate,
            input_name=input_name,
            output_name=output_name,
        ).argmax()

        if self.labels is not None:
            y = self.labels[y]

        return y
