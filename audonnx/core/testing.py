import typing

import numpy as np  # noqa: F401, needed for doctest
import onnx
import onnxruntime

from audonnx.core.function import Function
from audonnx.core.model import Model
from audonnx.core.typing import Device


def create_model(
        shapes: typing.Sequence[typing.Sequence[int]],
        *,
        value: float = 0.,
        dtype: int = onnx.TensorProto.FLOAT,
        opset_version: int = 14,
        device: Device = 'cpu',
        num_workers: typing.Optional[int] = 1,
        session_options: typing.Optional[onnxruntime.SessionOptions] = None,
) -> Model:
    r"""Create test model.

    Creates a model that outputs
    arrays filled with ``value``
    of the given ``shapes``.
    For each entry an output node will be created.
    ``-1``, ``None`` or strings
    define a dynamic axis.
    Per node,
    one dynamic axis is allowed.

    Args:
        shapes: list with shapes defining the identical
            input and output nodes of the model graph
        value: fill value
        dtype: data type, see `supported data types`_
        opset_version: opset version
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

    Returns:
        model object

    Examples:
        >>> shapes = [[3], [1, -1, 2]]
        >>> model = create_model(shapes)
        >>> model
        Input:
          input-0:
            shape: [3]
            dtype: tensor(float)
            transform: audonnx.core.function.Function(reshape)
          input-1:
            shape: [1, -1, 2]
            dtype: tensor(float)
            transform: audonnx.core.function.Function(reshape)
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
    .. _`provider(s)`: https://onnxruntime.ai/docs/execution-providers/

    """  # noqa: E501
    # create graph

    object = create_model_proto(
        shapes,
        dtype=dtype,
        opset_version=opset_version,
    )

    # create transform objects

    transform = {}
    for idx, shape in enumerate(shapes):
        transform[f'input-{idx}'] = Function(
            reshape,
            func_args={
                'shape': shape,
                'value': value,
            },
        )

    # create model

    model = Model(
        object,
        transform=transform,
        device=device,
        num_workers=num_workers,
        session_options=session_options,
    )

    return model


def create_model_proto(
        shapes: typing.Sequence[typing.Sequence[int]],
        *,
        dtype: int = onnx.TensorProto.FLOAT,
        opset_version: int = 14,
) -> onnx.ModelProto:
    r"""Create test ONNX proto object.

    Creates an identity graph
    with input and output nodes
    of the given ``shapes``.
    For each entry an input and output
    node will be created.
    ``-1``, ``None`` or strings
    define a dynamic axis.
    Per node,
    one dynamic axis is allowed.
    The identity graph can be used as ``path`` argument
    of :class:`audonnx.Model`.

    Args:
        shapes: list with shapes defining the identical
            input and output nodes of the model graph
        dtype: data type, see `supported data types`_
        opset_version: opset version

    Returns:
        ONNX object

    Examples:
        >>> create_model_proto([[2]])
        ir_version: 9
        opset_import {
          version: 14
        }
        producer_name: "test"
        graph {
          node {
            input: "input-0"
            output: "output-0"
            op_type: "Identity"
          }
          name: "test"
          input {
            name: "input-0"
            type {
              tensor_type {
                elem_type: 1
                shape {
                  dim {
                    dim_value: 2
                  }
                }
              }
            }
          }
          output {
            name: "output-0"
            type {
              tensor_type {
                elem_type: 1
                shape {
                  dim {
                    dim_value: 2
                  }
                }
              }
            }
          }
        }
        <BLANKLINE>

    .. _`supported data types`: https://onnxruntime.ai/docs/reference/operators/custom-python-operator.html#supported-data-types

    """  # noqa: E501
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


def reshape(signal, _, shape, value):
    r"""Return array with zeros of given shape."""
    import numpy as np  # noqa: F811
    shape = [signal.shape[-1] if not isinstance(s, int) or s < 0
             else s for s in shape]
    return (np.ones(shape) * value).astype(signal.dtype)
