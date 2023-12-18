import os
import typing

import numpy as np
import onnx
import onnxruntime
import yaml

import audeer
import audobject

from audonnx.core.node import InputNode
from audonnx.core.node import OutputNode
from audonnx.core.ort import device_to_providers
from audonnx.core.typing import Device
from audonnx.core.typing import Labels
from audonnx.core.typing import Transform


class Model(audobject.Object):
    r"""Model with multiple input and output nodes.

    For input nodes an optional transform can be given
    that transforms the raw signal into the desired representation,
    otherwise the raw signal is used as input.
    Use dictionary to assign transform objects to specific nodes
    if model has multiple input nodes.

    For output nodes an optional list of labels can be given
    to assign names to the last non-dynamic dimension.
    E.g. if the shape of the output node is
    ``(1, 3, -1)``
    three labels can be assigned to the second dimension.
    By default,
    labels are generated from the name of the node.
    Use dictionary to assign labels to specific nodes
    if model has multiple output nodes.

    :class:`Model` inherites from :class:`audobject.Object`,
    which means you can serialize to
    and instantiate the class
    from a YAML file.
    Have a look at :class:`audobject.Object`
    to see all available methods.

    Args:
        path: ONNX proto object or path to ONNX file
        labels: list of labels or dictionary with labels
        transform: callable object or a dictionary of callable objects
        device: set device
            (``'cpu'``, ``'cuda'``, or ``'cuda:<id>'``)
            or a (list of) `provider(s)`_
        num_workers: number of threads for running
            onnxruntime inference on cpu.
            If ``None`` and ``session_options`` is ``None``,
            onnxruntime chooses the number of threads
        session_options: :class:`onnxruntime.SessionOptions`
            to use for inference.
            If ``None`` the default options are used
            and the number of threads
            for running inference on cpu
            is determined by ``num_workers``.
            Otherwise,
            the provided options are used
            and the ``session_options`` properties
            :attr:`~onnxruntime.SessionOptions.inter_op_num_threads`
            and :attr:`~onnxruntime.SessionOptions.intra_op_num_threads`
            determine the number of threads
            for inference on cpu
            and ``num_workers`` is ignored

    Examples:
        >>> import audiofile
        >>> import opensmile
        >>> transform = opensmile.Smile(
        ...    opensmile.FeatureSet.GeMAPSv01b,
        ...    opensmile.FeatureLevel.LowLevelDescriptors,
        ... )
        >>> path = os.path.join('tests', 'model.onnx')
        >>> model = Model(
        ...     path,
        ...     labels=['female', 'male'],
        ...     transform=transform,
        ... )
        >>> model
        Input:
          feature:
            shape: [18, -1]
            dtype: tensor(float)
            transform: opensmile.core.smile.Smile
        Output:
          gender:
            shape: [2]
            dtype: tensor(float)
            labels: [female, male]
        >>> path = os.path.join('tests', 'test.wav')
        >>> signal, sampling_rate = audiofile.read(path)
        >>> model(
        ...     signal,
        ...     sampling_rate,
        ...     outputs='gender',
        ... ).round(1)
        array([-195.1,   73.3], dtype=float32)

    .. _`provider(s)`: https://onnxruntime.ai/docs/execution-providers/

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
            'num_workers',
            'session_options',
        ],
    )
    def __init__(
            self,
            path: typing.Union[str, onnx.ModelProto],
            *,
            labels: Labels = None,
            transform: Transform = None,
            device: Device = 'cpu',
            num_workers: typing.Optional[int] = 1,
            session_options: typing.Optional[
                onnxruntime.SessionOptions
            ] = None,
    ):
        # keep original arguments to store them
        # when object is serialized
        self._original_args = {
            'labels': labels,
            'transform': transform,
        }

        self.path = audeer.path(path) if isinstance(path, str) else None
        r"""Model path"""

        if session_options is None:
            session_options = onnxruntime.SessionOptions()
            if num_workers is not None:
                session_options.inter_op_num_threads = num_workers
                session_options.intra_op_num_threads = num_workers

        providers = device_to_providers(device)
        self.sess = onnxruntime.InferenceSession(
            self.path if isinstance(path, str) else path.SerializeToString(),
            sess_options=session_options,
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
            shape = _shape(input.shape)
            self.inputs[str(input.name)] = InputNode(
                shape,
                input.type,
                trans,
            )

        self.outputs = {}
        r"""Output nodes"""
        for output in outputs:
            shape = output.shape or [1]
            shape = _shape(shape)
            dim_size = _last_static_dim_size(shape)
            if output.name in labels:
                lab = labels[output.name]
                if len(lab) != dim_size:
                    raise ValueError(
                        f"Cannot assign "
                        f"{len(lab)} "
                        f"labels to output node "
                        f"'{output.name}' "
                        f"when last non-dynamic dimension has size "
                        f"{dim_size}."
                    )
            elif dim_size == 1:
                lab = [output.name]
            else:
                lab = [f'{output.name}-{idx}'
                       for idx in range(dim_size)]
            self.outputs[output.name] = OutputNode(
                shape,
                output.type,
                lab,
            )

    @audeer.deprecated_keyword_argument(
        deprecated_argument='output_names',
        removal_version='1.2.0',
        new_argument='outputs',
    )
    def __call__(
            self,
            signal: np.ndarray,
            sampling_rate: int,
            *,
            outputs: typing.Union[str, typing.Sequence[str]] = None,
            concat: bool = False,
            squeeze: bool = False,
    ) -> typing.Union[
        np.ndarray,
        typing.Dict[str, np.ndarray],
    ]:
        r"""Compute output for one or more nodes.

        If ``outputs`` is a plain string,
        the output of the according node is returned.

        If ``outputs`` is a list of strings,
        a dictionary with according nodes as keys and
        their outputs as values is returned.

        If ``outputs`` is not set
        and the model has a single output node,
        the output of that node is returned.
        Otherwise a dictionary with outputs of all nodes is returned.

        If ``concat`` is set to ``True``,
        the output of the requested nodes is concatenated
        along the last non-dynamic axis
        and a single array is returned.
        This requires that the number of dimensions,
        the position of dynamic axis,
        and all dimensions except the last non-dynamic axis
        match for the requested nodes

        Use :attr:`audonnx.Model.outputs` to get a list of available
        output nodes.

        Args:
            signal: input signal
            sampling_rate: sampling rate in Hz
            outputs: name of output or list with output names
            concat: if ``True``,
                concatenate output of the requested nodes
            squeeze: if ``True``,
                remove axes of length one from the output(s)

        Returns:
            model output

        Raises:
            RuntimeError: if ``concat`` is ``True``,
                but output of requested nodes cannot be concatenated

        """
        if outputs is None:
            outputs = list(self.outputs)
            if len(outputs) == 1:
                outputs = outputs[0]

        y = {}
        for name, input in self.inputs.items():
            if input.transform is not None:
                x = input.transform(signal, sampling_rate)
            else:
                x = signal
            y[name] = x.reshape(self.inputs[name].shape)

        z = self.sess.run(
            audeer.to_list(outputs),
            y,
        )

        if isinstance(outputs, str):
            z = z[0]
        else:
            z = {
                name: values for name, values in zip(outputs, z)
            }
            if concat:
                shapes = [self.outputs[node].shape for node in outputs]
                z = _concat(z, shapes)

        if squeeze:
            if isinstance(z, dict):
                z = {
                    name: values.squeeze() for name, values in z.items()
                }
            else:
                z = z.squeeze()

        return z

    def __repr__(self) -> str:
        r"""Printable representation of model."""
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
        r"""String representation of model."""
        return repr(self)

    def labels(
            self,
            outputs: typing.Union[str, typing.Sequence[str]] = None,
    ) -> typing.Sequence[str]:
        r"""Collect labels of output nodes.

        Args:
            outputs: name of output or list with output names.
                Selects all output nodes by default

        Returns:
            list with labels

        """
        if outputs is None:
            outputs = list(self.outputs)
        outputs = audeer.to_list(outputs)

        names = [self.outputs[name].labels for name in outputs]
        names = audeer.flatten_list(names)

        return names

    def to_yaml(
            self,
            path: str,
            *,
            include_version: bool = True,
    ):
        r"""Save model to YAML file.

        If ONNX model was loaded from a byte stream,
        it will be saved in addition to the YAML file
        under the same path
        with file extension ``.onnx``.

        Args:
            path: file path, must end on ``.yaml``
            include_version: add version to class name

        Raises:
            ValueError: if file path does not end on ``.yaml``

        """
        path = audeer.path(path)
        if not audeer.file_extension(path) == 'yaml':
            raise ValueError(f"Model path {path} does not end on '.yaml'")

        if self.path is None:
            # if model was loaded from a byte stream,
            # we have to save it first
            self.path = audeer.replace_file_extension(path, 'onnx')
            audeer.mkdir(os.path.dirname(path))
            onnx.save(self.sess._model_bytes, self.path)

        with open(path, 'w') as fp:
            super().to_yaml(fp, include_version=include_version)


def _concat(
        y: typing.Dict[str, np.ndarray],
        shapes: typing.Sequence[typing.List[int]],
):
    r"""Flatten dictionary by concatenating values."""
    y = list(y.values())

    # special case if all shapes are [-1]
    if all(map([-1].__eq__, shapes)):
        return np.stack(y)

    axis = _concat_axis(shapes)
    if axis is None:
        raise RuntimeError(
            f'To concatenate outputs '
            f'number of dimensions, '
            f'position of dynamic axis, '
            f'and all dimensions except the last non-dynamic axis '
            f'must match. '
            f'This does not apply to: '
            f'{shapes}'
        )

    return np.concatenate(y, axis=axis)


def _concat_axis(shapes: typing.Sequence[int]) -> typing.Optional[int]:
    r"""Return concat dimension or None if not possible."""
    # number of dimensions do not match
    if not len(set(map(len, shapes))) == 1:
        return None

    # dynamic axis in different positions
    if not len(set(map(_dynamic_axis, shapes))) == 1:
        return None

    # select last non-dynamic axis
    axis = len(shapes[0]) - (2 if shapes[0][-1] == -1 else 1)

    # remove concat axis
    shapes_wo_axis = [shape[:axis] + shape[axis + 1:] for shape in shapes]

    # non-concat dimensions do not match
    if not all(map(shapes_wo_axis[0].__eq__, shapes_wo_axis)):
        return None

    return axis


def _dynamic_axis(shape: typing.Sequence[int]) -> typing.Optional[int]:
    r"""Return dimension of dynamic axis or None if none."""
    for idx, dim in enumerate(shape):
        if dim == -1:
            return idx
    return None


def _last_static_dim_size(
        shape: typing.List[int],
) -> int:
    r"""Return size of last static dimension."""
    shape = list(filter((-1).__ne__, shape))
    return shape[-1] if len(shape) else 0


def _shape(
        shape: typing.List[typing.Union[int, str]],
) -> typing.List[int]:
    r"""Replace dynamic dimensions with -1."""
    return [-1 if not isinstance(x, int) else x for x in shape]
