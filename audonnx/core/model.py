import os
import re
import typing

import numpy as np
import onnxruntime

import audeer
import audobject
import yaml

from audonnx.core.node import (
    InputNode,
    OutputNode,
)
from audonnx.core.typing import (
    Labels,
    Transform,
)


class Model(audobject.Object):
    r"""Model with multiple input and output nodes.

    For input nodes an optional transform can be given
    that transforms the raw signal into the desired representation,
    otherwise the raw signal is used as input.
    Use dictionary to assign transform objects to specific nodes
    if model has multiple input nodes.

    For output nodes an optional list of labels can be given,
    where each label corresponds to a dimension in the output,
    i.e. the number of labels must match the dimension of the output node.
    Use dictionary to assign labels to specific nodes
    if model has multiple output nodes.

    :class:`Model` inherites from :class:`audobject.Object`,
    which means you can seralize to
    and instantiate the class
    from a YAML file.
    Have a look at :class:`audobject.Object`
    to see all available methods.

    Args:
        path: path to model file
        labels: list of labels or dictionary with labels
        transform: callable object or a dictionary of callable objects
        device: set device
            (``'cpu'``, ``'cuda'``, or ``'cuda:<id>'``)
            or a (list of) provider(s)_

    .. _provider(s): https://onnxruntime.ai/docs/execution-providers/

    """
    @audobject.init_decorator(
        borrow={
            'labels': '_original_args',
            'transform': '_original_args',
        },
        resolvers={
            'path': audobject.resolver.FilePath,
        },
        hide=[
            'device',
        ],
    )
    def __init__(
            self,
            path: str,
            *,
            labels: Labels = None,
            transform: Transform = None,
            device: typing.Union[
                str,
                typing.Tuple[str, typing.Dict],
                typing.Sequence[
                    typing.Union[str, typing.Tuple[str, typing.Dict]]],
            ] = 'cpu',
    ):
        # keep original arguments to store them
        # when object is serialized
        self._original_args = {
            'labels': labels,
            'transform': transform,
        }

        self.path = audeer.safe_path(path)
        r"""Model path"""

        providers = _device_to_providers(device)
        self.sess = onnxruntime.InferenceSession(
            self.path,
            providers=providers,
        )
        r"""Interference session"""

        inputs = self.sess.get_inputs()
        outputs = self.sess.get_outputs()

        transform = transform or {}
        if not isinstance(transform, dict):
            if not len(inputs) == 1:
                names = [input.name for input in inputs]
                raise ValueError(
                    f'Model has multiple input nodes. '
                    f'Please use a dictionary to assign '
                    f'transform object to one of '
                    f'{names}.'
                )
            transform = {inputs[0].name: transform}

        labels = labels or {}
        if not isinstance(labels, dict):
            if not len(outputs) == 1:
                names = [output.name for output in outputs]
                raise ValueError(
                    f'Model has multiple output nodes. '
                    f'Please use a dictionary to assign '
                    f'labels to one of '
                    f'{names}.'
                )
            labels = {outputs[0].name: labels}

        self.inputs = {}
        r"""Input nodes"""
        for input in inputs:
            trans = transform[input.name] if input.name in transform else None
            self.inputs[str(input.name)] = InputNode(
                [-1 if x == 'time' else x for x in input.shape],
                input.type,
                trans,
            )

        self.outputs = {}
        r"""Output nodes"""
        for output in outputs:
            shape = output.shape or [1]
            dim = shape[-1]
            if output.name in labels:
                lab = labels[output.name]
                if len(lab) != dim:
                    raise ValueError(
                        f"Cannot assign "
                        f"{len(lab)} "
                        f"labels to output "
                        f"'{output.name}' "
                        f"with dimension "
                        f"{dim}."
                    )
            elif dim == 1:
                lab = [output.name]
            else:
                lab = [f'{output.name}-{idx}' for idx in range(dim)]
            self.outputs[output.name] = OutputNode(
                shape,
                output.type,
                lab,
            )

    def __call__(
            self,
            signal: np.ndarray,
            sampling_rate: int,
            *,
            output_names: typing.Union[str, typing.Sequence[str]] = None,
    ) -> typing.Union[
        np.ndarray,
        typing.Dict[str, np.ndarray],
    ]:
        r"""Compute output for one or more nodes.

        If ``output_names`` is a plain string,
        the output of the according node is returned.

        If ``output_names`` is a list of strings,
        a dictionary with according nodes as keys and
        their outputs as values is returned.

        If ``output_names`` is not set
        and the model has a single output node,
        the output of that node is returned.
        Otherwise a dictionary with outputs of all nodes is returned.

        Use :attr:`audonnx.Model.output_names` to get a list of available
        output nodes.

        Args:
            signal: input signal
            sampling_rate: sampling rate in Hz
            output_names: name of output or list with output names

        Returns:
            model output

        Examples:
            >>> import audiofile
            >>> audio_path = os.path.join('tests', 'test.wav')
            >>> signal, sampling_rate = audiofile.read(audio_path)
            >>> model_path = os.path.join('tests', 'model.yaml')
            >>> import audobject
            >>> model = audobject.from_yaml(model_path)
            >>> model(
            ...     signal,
            ...     sampling_rate,
            ...     output_names='gender',
            ... ).round(1)
            array([-195.1,   73.3], dtype=float32)

        """
        if output_names is None:
            output_names = list(self.outputs)
            if len(output_names) == 1:
                output_names = output_names[0]

        y = {}
        for name, input in self.inputs.items():
            if input.transform is not None:
                x = input.transform(signal, sampling_rate)
            else:
                x = signal
            y[name] = x.reshape(self.inputs[name].shape)

        z = self.sess.run(
            audeer.to_list(output_names),
            y,
        )

        if isinstance(output_names, str):
            z = z[0]
        else:
            z = {
                name: values for name, values in zip(output_names, z)
            }

        return z

    def __repr__(self) -> str:

        d = {
            'Input': {
                name: node._dict() for name, node in self.inputs.items()
            },
            'Output': {
                name: node._dict() for name, node in self.outputs.items()
            }
        }

        return yaml.dump(d, default_flow_style=None).strip()

    def __str__(self) -> str:
        return repr(self)

    def to_yaml(
            self,
            path: str,
            *,
            include_version: bool = True,
    ):
        r"""Save model to YAML file.

        Args:
            path: file path, must end on ``.yaml``
            include_version: add version to class name

        Raises:
            ValueError: if file path does not end on ``.yaml``

       """
        path = audeer.safe_path(path)
        if not audeer.file_extension(path) == 'yaml':
            raise ValueError(f"Model path {path} does not end on '.yaml'")
        with open(path, 'w') as fp:
            super().to_yaml(fp, include_version=include_version)


def _device_to_providers(
        device: typing.Union[
            str,
            typing.Tuple[str, typing.Dict],
            typing.Sequence[typing.Union[str, typing.Tuple[str, typing.Dict]]],
        ],
) -> typing.Sequence[typing.Union[str, typing.Tuple[str, typing.Dict]]]:
    r"""Converts device into a list of providers."""
    if isinstance(device, str):
        if device == 'cpu':
            providers = ['CPUExecutionProvider']
        elif device.startswith('cuda'):
            match = re.search(r'^cuda:(\d+)$', device)
            if match:
                device_id = match.group(1)
                providers = [
                    (
                        'CUDAExecutionProvider', {
                            'device_id': device_id,
                        }
                    ),
                ]
            else:
                providers = ['CUDAExecutionProvider']
        else:
            providers = [device]
    elif isinstance(device, tuple):
        providers = [device]
    else:
        providers = device
    return providers
