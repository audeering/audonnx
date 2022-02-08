import numpy as np
import pytest

import audonnx


@pytest.mark.parametrize(
    'path, transform, labels, expected',
    [
        (
            pytest.MODEL_SINGLE_PATH,
            pytest.FEATURE,
            None,
            {'gender': ['gender-0', 'gender-1']},
        ),
        (
            pytest.MODEL_SINGLE_PATH,
            pytest.FEATURE,
            ['female', 'male'],
            {'gender': ['female', 'male']},
        ),
        pytest.param(
            pytest.MODEL_SINGLE_PATH,
            None,
            ['too', 'many', 'labels'],
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ]
)
def test_labels(path, transform, labels, expected):
    model = audonnx.Model(
        path,
        labels=labels,
        transform=transform,
    )
    for name, l in expected.items():
        assert model.outputs[name].labels == l


@pytest.mark.parametrize(
    'model, output_names, expected',
    [
        (
            audonnx.Model(
                pytest.MODEL_SINGLE_PATH,
                transform=pytest.FEATURE,
            ),
            None,
            np.array([-195.1, 73.3], np.float32),
        ),
        (
            audonnx.Model(
                pytest.MODEL_SINGLE_PATH,
                transform=pytest.FEATURE,
            ),
            'gender',
            np.array([-195.1, 73.3], np.float32),
        ),
        (
            audonnx.Model(
                pytest.MODEL_SINGLE_PATH,
                transform=pytest.FEATURE,
            ),
            ['gender'],
            {'gender': np.array([-195.1, 73.3], np.float32)},
        ),
        (
            audonnx.Model(
                pytest.MODEL_MULTI_PATH,
                transform={
                    'feature': pytest.FEATURE,
                }
            ),
            None,
            {
                'hidden': np.array([
                    1.3299127e-01, 2.1280064e-01,
                    -4.7600174e-01, -3.7167081e-01,
                    6.6870685e+02, -4.2869656e+02,
                    -4.5552551e+02, 8.6153650e+02,
                ], np.float32),
                'gender': np.array([224.83, -12.72], np.float32),
                'confidence': np.array(-311.88, np.float32),
            },
        ),
        (
            audonnx.Model(
                pytest.MODEL_MULTI_PATH,
                transform={
                    'feature': pytest.FEATURE,
                }
            ),
            ['confidence', 'gender'],
            {
                'confidence': np.array(-311.88, np.float32),
                'gender': np.array([224.83, -12.72], np.float32),
            },
        ),
        (
            audonnx.Model(
                pytest.MODEL_MULTI_PATH,
                transform={
                    'feature': pytest.FEATURE,
                }
            ),
            'confidence',
            np.array(-311.88, np.float32),
        ),
    ],
)
def test_call(model, output_names, expected):
    y = model(
        pytest.SIGNAL,
        pytest.SAMPLING_RATE,
        output_names=output_names,
    )
    if isinstance(y, dict):
        for key, values in y.items():
            np.testing.assert_almost_equal(y[key], expected[key], decimal=1)
    else:
        np.testing.assert_almost_equal(y, expected, decimal=1)


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
    model = audonnx.Model(
        pytest.MODEL_SINGLE_PATH,
        transform=pytest.FEATURE,
        device=device,
    )
    y = model(
        pytest.SIGNAL,
        pytest.SAMPLING_RATE,
    )
    expected = np.array([-195.1, 73.3], np.float32)
    np.testing.assert_almost_equal(y, expected, decimal=1)


@pytest.mark.parametrize(
    'path, labels, transform',
    [
        (
            pytest.MODEL_SINGLE_PATH,
            None,
            pytest.FEATURE,
        ),
        (
            pytest.MODEL_MULTI_PATH,
            {
                'gender': ['female', 'male'],
            },
            {
                'feature': pytest.FEATURE,
            }
        ),
        pytest.param(  # list of labels but multiple output nodes
            pytest.MODEL_MULTI_PATH,
            ['female', 'male'],
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(  # transform object but multiple input nodes
            pytest.MODEL_MULTI_PATH,
            None,
            pytest.FEATURE,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ],
)
def test_nodes(path, labels, transform):

    model = audonnx.Model(
        path,
        labels=labels,
        transform=transform,
    )

    inputs = model.sess.get_inputs()
    outputs = model.sess.get_outputs()

    for idx, (name, node) in enumerate(model.inputs.items()):
        assert name == inputs[idx].name
        assert node.shape == [
            -1 if x == 'time' else x for x in inputs[idx].shape
        ]
        assert node.dtype == inputs[idx].type

    for idx, (name, node) in enumerate(model.outputs.items()):
        assert name == outputs[idx].name
        if outputs[idx].shape:
            assert node.shape == outputs[idx].shape
        else:
            assert node.shape == [1]
        assert node.dtype == outputs[idx].type


@pytest.mark.parametrize(
    'model, expected',
    [
        (
            audonnx.Model(
                pytest.MODEL_SINGLE_PATH,
                labels=['female', 'male'],
                transform=pytest.FEATURE,
            ), r'''Input:
  feature:
    shape: [18, -1]
    dtype: tensor(float)
    transform: opensmile.core.smile.Smile
Output:
  gender:
    shape: [2]
    dtype: tensor(float)
    labels: [female, male]'''
        ),
        (
            audonnx.Model(
                pytest.MODEL_MULTI_PATH,
                transform={
                    'feature': pytest.FEATURE,
                }
            ), r'''Input:
  signal:
    shape: [1, -1]
    dtype: tensor(float)
    transform: None
  feature:
    shape: [18, -1]
    dtype: tensor(float)
    transform: opensmile.core.smile.Smile
Output:
  hidden:
    shape: [8]
    dtype: tensor(float)
    labels: [hidden-0, hidden-1, hidden-2, (...), hidden-5, hidden-6, hidden-7]
  gender:
    shape: [2]
    dtype: tensor(float)
    labels: [gender-0, gender-1]
  confidence:
    shape: [1]
    dtype: tensor(float)
    labels: [confidence]'''  # noqa
        )
    ],
)
def test_str(model, expected):
    assert str(model) == expected
