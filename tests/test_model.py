import numpy as np
import onnxruntime
import pytest

import audeer

import audonnx.testing


def min_max(x, sr):
    return [x.min(), x.max()]


@pytest.mark.parametrize(
    'model, outputs, expected',
    [
        (
            audonnx.Model(audonnx.testing.create_model_proto([[1, -1]])),
            None,
            pytest.SIGNAL,
        ),
        (
            audonnx.testing.create_model([[2]]),
            None,
            np.array([0.0, 0.0], np.float32),
        ),
        (
            audonnx.testing.create_model([[2]]),
            'output-0',
            np.array([0.0, 0.0], np.float32),
        ),
        (
            audonnx.testing.create_model([[2]]),
            ['output-0'],
            {'output-0': np.array([0.0, 0.0], np.float32)},
        ),
        (
            audonnx.testing.create_model([[2], [1, 3]]),
            None,
            {
                'output-0': np.array([0.0, 0.0], np.float32),
                'output-1': np.array([[0.0, 0.0, 0.0]], np.float32),
            },
        ),
        (
            audonnx.testing.create_model([[2], [1, 3]]),
            'output-1',
            np.array([[0.0, 0.0, 0.0]], np.float32),
        ),
        (
            audonnx.testing.create_model([[2], [1, 3]]),
            ['output-1'],
            {
                'output-1': np.array([[0.0, 0.0, 0.0]], np.float32),
            },
        ),
        (
            audonnx.testing.create_model([[2], [1, 3]]),
            ['output-1', 'output-0'],
            {
                'output-1': np.array([[0.0, 0.0, 0.0]], np.float32),
                'output-0': np.array([0.0, 0.0], np.float32),
            },
        ),
        (
            audonnx.testing.create_model([[2], [1, 3]]),
            ['output-1', 'output-0', 'output-1'],
            {
                'output-1': np.array([[0.0, 0.0, 0.0]], np.float32),
                'output-0': np.array([0.0, 0.0], np.float32),
            },
        ),
    ]
)
def test_call(model, outputs, expected):
    for squeeze in [False, True]:
        y = model(
            pytest.SIGNAL,
            pytest.SAMPLING_RATE,
            outputs=outputs,
            squeeze=squeeze,
        )
        if isinstance(y, dict):
            for key, values in y.items():
                if squeeze:
                    np.testing.assert_equal(y[key], expected[key].squeeze())
                else:
                    np.testing.assert_equal(y[key], expected[key])
        else:
            if squeeze:
                np.testing.assert_equal(y, expected.squeeze())
            else:
                np.testing.assert_equal(y, expected)


@pytest.mark.parametrize(
    'model, output_names',
    [
        (
            audonnx.testing.create_model([[2], [1, 3]]),
            'output-1',
        ),
    ]
)
def test_call_deprecated(model, output_names):
    if (
            audeer.LooseVersion(audonnx.__version__)
            < audeer.LooseVersion('1.2.0')
    ):
        with pytest.warns(UserWarning, match='is deprecated'):
            model(
                pytest.SIGNAL,
                pytest.SAMPLING_RATE,
                output_names=output_names,
            )
    else:
        with pytest.raises(TypeError, match='unexpected keyword argument'):
            model(
                pytest.SIGNAL,
                pytest.SAMPLING_RATE,
                output_names=output_names,
            )


@pytest.mark.parametrize(
    'model, outputs, expected',
    [
        (
            audonnx.Model(audonnx.testing.create_model_proto([[1, -1]])),
            None,
            pytest.SIGNAL,
        ),
        (
            audonnx.testing.create_model([[-1], [-1]]),
            None,
            np.zeros([2, pytest.SIGNAL.shape[1]], dtype=np.float32),
        ),
        (
            audonnx.testing.create_model([[2]]),
            None,
            np.array([0, 0], dtype=np.float32),
        ),
        (
            audonnx.testing.create_model([[2], [3]]),
            None,
            np.array([0, 0, 0, 0, 0], dtype=np.float32),
        ),
        (
            audonnx.testing.create_model([[1, 2], [1, 3]]),
            None,
            np.array([[0, 0, 0, 0, 0]], dtype=np.float32),
        ),
        (
            audonnx.testing.create_model([[2, -1], [3, -1]]),
            None,
            np.zeros([5, pytest.SIGNAL.shape[1]], dtype=np.float32),
        ),
        (
            audonnx.testing.create_model([[-1, 2], [-1, 3]]),
            None,
            np.zeros([pytest.SIGNAL.shape[1], 5], dtype=np.float32),
        ),
        (
            audonnx.testing.create_model([[1, -1, 2], [1, -1, 3]]),
            None,
            np.zeros([1, pytest.SIGNAL.shape[1], 5], dtype=np.float32),
        ),
        (
            audonnx.testing.create_model([[1, 2, -1], [1, 3, -1]]),
            None,
            np.zeros([1, 5, pytest.SIGNAL.shape[1]], dtype=np.float32),
        ),
        (
            audonnx.testing.create_model([[1, 3], [2]]),
            'output-0',
            np.array([[0, 0, 0]], dtype=np.float32),
        ),
        (
            audonnx.testing.create_model([[1, 3], [2]]),
            ['output-0'],
            np.array([[0, 0, 0]], dtype=np.float32),
        ),
        (
            audonnx.testing.create_model([[1, 3], [2], [3]]),
            ['output-1', 'output-2'],
            np.array([0, 0, 0, 0, 0], dtype=np.float32),
        ),
        # shapes do not match
        pytest.param(
            audonnx.testing.create_model([[1, 3], [2]]),
            None,
            None,
            marks=pytest.mark.xfail(raises=RuntimeError),
        ),
        # position of dynamic axis does not match
        pytest.param(
            audonnx.testing.create_model([[-1, 1, 3], [1, -1, 2]]),
            None,
            None,
            marks=pytest.mark.xfail(raises=RuntimeError),
        ),
        pytest.param(
            audonnx.testing.create_model([[1, 3], [-1, 2]]),
            None,
            None,
            marks=pytest.mark.xfail(raises=RuntimeError),
        ),
        # non-concat dimensions do not match
        pytest.param(
            audonnx.testing.create_model([[1, 3], [2, 3]]),
            None,
            None,
            marks=pytest.mark.xfail(raises=RuntimeError),
        ),
    ]
)
def test_call_concat(model, outputs, expected):
    for squeeze in [False, True]:
        y = model(
            pytest.SIGNAL,
            pytest.SAMPLING_RATE,
            outputs=outputs,
            concat=True,
            squeeze=squeeze,
        )
        if squeeze:
            np.testing.assert_equal(y, expected.squeeze())
        else:
            np.testing.assert_equal(y, expected)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('num_workers', [None, 1, 2])
@pytest.mark.parametrize(
    'session_options',
    [
        None,
        onnxruntime.SessionOptions(),
    ]
)
def test_call_num_workers_session_options(
    device, num_workers, session_options
):
    model = audonnx.testing.create_model(
        [[2]],
        device=device,
        num_workers=num_workers,
        session_options=session_options,
    )
    y = model(
        pytest.SIGNAL,
        pytest.SAMPLING_RATE,
    )
    expected = np.array([0.0, 0.0], np.float32)
    np.testing.assert_equal(y, expected)


@pytest.mark.parametrize(
    'device',
    [
        'cpu',
        'CPUExecutionProvider',
        'cuda',
        'cuda:0',
        (
            'CUDAExecutionProvider',
            {
                'device_id': 0,
            },
        ),
        [
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
        [
            (
                'CUDAExecutionProvider',
                {
                    'device_id': 0,
                },
            ),
            'CPUExecutionProvider',
        ],
    ]
)
def test_device_or_providers(device):
    model = audonnx.testing.create_model([[2]], device=device)
    y = model(
        pytest.SIGNAL,
        pytest.SAMPLING_RATE,
    )
    expected = np.array([0.0, 0.0], np.float32)
    np.testing.assert_equal(y, expected)


def test_init(tmpdir):

    # create model from ONNX object

    object = audonnx.testing.create_model_proto([[1, -1]])
    model = audonnx.Model(object)
    assert model.path is None

    # save model to YAML

    yaml_path = audeer.path(tmpdir, 'model.yaml')
    model.to_yaml(yaml_path)
    onnx_path = audeer.replace_file_extension(yaml_path, 'onnx')
    assert model.path == onnx_path

    # create model from ONNX file

    model_2 = audonnx.Model(onnx_path)
    assert model_2.path == onnx_path

    # load from YAML

    model_3 = audonnx.load(tmpdir)
    assert model_3.path == onnx_path


@pytest.mark.parametrize(
    'model, outputs, expected',
    [
        (
            audonnx.testing.create_model([[2]]),
            None,
            ['output-0-0', 'output-0-1'],
        ),
        (
            audonnx.testing.create_model([[2], [1]]),
            None,
            ['output-0-0', 'output-0-1', 'output-1'],
        ),
        (
            audonnx.testing.create_model([[2], [1]]),
            'output-1',
            ['output-1'],
        ),
        (
            audonnx.testing.create_model([[2], [1]]),
            ['output-1', 'output-0'],
            ['output-1', 'output-0-0', 'output-0-1'],
        ),
    ]
)
def test_labels(model, outputs, expected):
    labels = model.labels(outputs)
    assert labels == expected


@pytest.mark.parametrize(
    'object, transform, labels, expected',
    [
        (
            audonnx.testing.create_model_proto([[1, -1]]),
            None,
            None,
            {'output-0': ['output-0']},
        ),
        (
            audonnx.testing.create_model_proto([[2]]),
            min_max,
            None,
            {'output-0': ['output-0-0', 'output-0-1']},
        ),
        (
            audonnx.testing.create_model_proto([[2]]),
            min_max,
            ['min', 'max'],
            {'output-0': ['min', 'max']},
        ),
        (
            audonnx.testing.create_model_proto([[1, -1], [2]]),
            {'input-1': min_max},
            None,
            {
                'output-0': ['output-0'],
                'output-1': ['output-1-0', 'output-1-1'],
            },
        ),
        (
            audonnx.testing.create_model_proto([[1, -1], [2]]),
            {'input-1': min_max},
            {'output-1': ['min', 'max']},
            {
                'output-0': ['output-0'],
                'output-1': ['min', 'max'],
            },
        ),
        pytest.param(
            audonnx.testing.create_model_proto([[2]]),
            min_max,
            ['too', 'many', 'labels'],
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ]
)
def test_outputs(object, transform, labels, expected):
    model = audonnx.Model(
        object,
        labels=labels,
        transform=transform,
    )
    assert list(expected) == list(model.outputs)
    for name, expected_labels in expected.items():
        assert model.outputs[name].labels == expected_labels


@pytest.mark.parametrize(
    'object, labels, transform',
    [
        (
            audonnx.testing.create_model_proto([[1, -1]]),
            None,
            None,
        ),
        (
            audonnx.testing.create_model_proto([[1, -1]]),
            ['signal'],
            None,
        ),
        (
            audonnx.testing.create_model_proto([[1, -1]]),
            ['signal'],
            lambda x, sr: x.T,
        ),
        (
            audonnx.testing.create_model_proto([[1, -1]]),
            ['signal'],
            lambda x, sr: np.atleast_3d(),
        ),
        (
            audonnx.testing.create_model_proto([[2]]),
            ['min', 'max'],
            min_max,
        ),
        (
            audonnx.testing.create_model_proto([[2], [1, -1]]),
            {
                'input-0': ['min', 'max'],
                'input-1': None,
            },
            {
                'output-0': min_max,
                'output-1': None,
            },
        ),
        pytest.param(  # plain list of labels but multiple output nodes
            audonnx.testing.create_model_proto([[2], [1, -1]]),
            ['min', 'max'],
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(  # single transform object but multiple input nodes
            audonnx.testing.create_model_proto([[2], [1, -1]]),
            None,
            min_max,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ],
)
def test_nodes(object, labels, transform):

    model = audonnx.Model(
        object,
        labels=labels,
        transform=transform,
    )

    inputs = model.sess.get_inputs()
    outputs = model.sess.get_outputs()

    for idx, (name, node) in enumerate(model.inputs.items()):
        assert name == inputs[idx].name
        assert node.shape == [
            -1 if not isinstance(x, int) else x for x in inputs[idx].shape
        ]
        assert node.dtype == inputs[idx].type

    for idx, (name, node) in enumerate(model.outputs.items()):
        assert name == outputs[idx].name
        if outputs[idx].shape:
            assert node.shape == [
                -1 if not isinstance(x, int) else x for x in outputs[idx].shape
            ]
        else:
            assert node.shape == [1]
        assert node.dtype == outputs[idx].type


@pytest.mark.parametrize(
    'model, expected',
    [
        (
            audonnx.Model(
                audonnx.testing.create_model_proto([[1, -1]]),
            ), r'''Input:
  input-0:
    shape: [1, -1]
    dtype: tensor(float)
    transform: None
Output:
  output-0:
    shape: [1, -1]
    dtype: tensor(float)
    labels: [output-0]'''
        ),
        (
            audonnx.Model(
                audonnx.testing.create_model_proto([[2]]),
                labels=['min', 'max'],
                transform=lambda x, sr: [x.min(), x.max()],
            ), r'''Input:
  input-0:
    shape: [2]
    dtype: tensor(float)
    transform: builtins.function
Output:
  output-0:
    shape: [2]
    dtype: tensor(float)
    labels: [min, max]'''
        ),
        (
            audonnx.Model(
                audonnx.testing.create_model_proto([[2]]),
                labels=['min', 'max'],
                transform=audonnx.Function(lambda x, sr: [x.min(), x.max()]),
            ), r'''Input:
  input-0:
    shape: [2]
    dtype: tensor(float)
    transform: audonnx.core.function.Function(<lambda>)
Output:
  output-0:
    shape: [2]
    dtype: tensor(float)
    labels: [min, max]'''
        ),
        (
            audonnx.Model(
                audonnx.testing.create_model_proto([[2]]),
                labels=['min', 'max'],
                transform=audonnx.Function(min_max),
            ), r'''Input:
  input-0:
    shape: [2]
    dtype: tensor(float)
    transform: audonnx.core.function.Function(min_max)
Output:
  output-0:
    shape: [2]
    dtype: tensor(float)
    labels: [min, max]'''
        ),
        (
            audonnx.testing.create_model(
                [[2], [None], [1, -1, 3], [99, 'time']],
            ), r'''Input:
  input-0:
    shape: [2]
    dtype: tensor(float)
    transform: audonnx.core.function.Function(reshape)
  input-1:
    shape: [-1]
    dtype: tensor(float)
    transform: audonnx.core.function.Function(reshape)
  input-2:
    shape: [1, -1, 3]
    dtype: tensor(float)
    transform: audonnx.core.function.Function(reshape)
  input-3:
    shape: [99, -1]
    dtype: tensor(float)
    transform: audonnx.core.function.Function(reshape)
Output:
  output-0:
    shape: [2]
    dtype: tensor(float)
    labels: [output-0-0, output-0-1]
  output-1:
    shape: [-1]
    dtype: tensor(float)
    labels: []
  output-2:
    shape: [1, -1, 3]
    dtype: tensor(float)
    labels: [output-2-0, output-2-1, output-2-2]
  output-3:
    shape: [99, -1]
    dtype: tensor(float)
    labels: [output-3-0, output-3-1, output-3-2, (...), output-3-96, output-3-97,
      output-3-98]'''  # noqa: E501
        )
    ],
)
def test_str(model, expected):
    assert str(model) == expected
