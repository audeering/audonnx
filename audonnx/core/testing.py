import typing

import numpy as np
import onnx

import audeer
import audobject
import audonnx


def create_model(
        root: str,
        shapes: typing.Sequence[typing.Sequence[int]],
        *,
        value: float = 0.,
        dtype: int = onnx.TensorProto.FLOAT,
        opset_version: int = 14,
) -> audonnx.Model:
    r"""Create test model.

    Creates a model that outputs
    arrays filled with ``value``
    of the given ``shapes``.
    For each entry an output node will be created.
    ``-1``, ``None`` or strings
    define a dynamic axis.
    Per node,
    one dynamic axis is allowed.
    The model is stored under ``root``
    and can be loaded with
    :func:`audonnx.load`.

    Args:
        root: folder where model is stored
        shapes: list with shapes defining the output nodes of the model.
            The model will have the same number of input nodes,
            copying the shapes from the output nodes
        value: fill value
        dtype: data type, see `supported data types`_
        opset_version: opset version

    Returns:
        model object

    Example:
        >>> shapes = [[3], [1, -1, 2]]
        >>> model = audonnx.testing.create_model('test', shapes)
        >>> model
        Input:
          input-0:
            shape: [3]
            dtype: tensor(float)
            transform: audonnx.core.function.Function
          input-1:
            shape: [1, -1, 2]
            dtype: tensor(float)
            transform: audonnx.core.function.Function
        Output:
          output-0:
            shape: [3]
            dtype: tensor(float)
            labels: [output-0-0, output-0-1, output-0-2]
          output-1:
            shape: [1, -1, 2]
            dtype: tensor(float)
            labels: [output-1-0, output-1-1]
        >>> signal = np.zeros((1, 5), np.float32)
        >>> model(signal, 8000)
        {'output-0': array([0., 0., 0.], dtype=float32), 'output-1': array([[[0., 0.],
                [0., 0.],
                [0., 0.],
                [0., 0.],
                [0., 0.]]], dtype=float32)}

    .. _`supported data types`: https://onnxruntime.ai/docs/reference/operators/custom-python-operator.html#supported-data-types

    """  # noqa: E501
    root = audeer.mkdir(root)
    path = audeer.path(root, 'model.onnx')

    # create graph

    graph = _identity_graph(shapes, dtype, opset_version)
    onnx.save(graph, path)

    # create transform objects

    transform = {}
    for idx, shape in enumerate(shapes):
        transform[f'input-{idx}'] = audonnx.Function(
            _shape_func,
            func_args={
                'shape': shape,
                'value': value,
            },
        )

    # create model

    model = audonnx.Model(
        path,
        transform=transform,
    )
    path = audeer.path(root, 'model.yaml')
    model.to_yaml(path)

    return model


def _identity_graph(
        shapes: typing.Sequence[typing.Sequence[int]],
        dtype: int,
        opset_version: int,
) -> onnx.ModelProto:
    r"""Create identity graph."""

    # nodes

    inputs = []
    outputs = []
    nodes = []

    for idx, shape in enumerate(shapes):

        input = onnx.helper.make_tensor_value_info(
            f'input-{idx}',
            dtype,
            shape,
        )
        inputs.append(input)

        output = onnx.helper.make_tensor_value_info(
            f'output-{idx}',
            dtype,
            shape,
        )
        outputs.append(output)

        node = onnx.helper.make_node(
            'Identity',
            [input.name],
            [output.name],
        )
        nodes.append(node)

    # graph

    graph = onnx.helper.make_graph(
        nodes,
        'test',
        inputs,
        outputs,
    )

    # model

    model = onnx.helper.make_model(graph, producer_name='test')
    model.opset_import[0].version = opset_version
    onnx.checker.check_model(model)

    return model


def _shape_func(signal, _, shape, value):
    r"""Return array with zeros of given shape."""
    import numpy as np  # noqa: F811
    shape = [signal.shape[-1] if not isinstance(s, int) or s < 0
             else s for s in shape]
    return (np.ones(shape) * value).astype(signal.dtype)
