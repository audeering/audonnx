import numpy as np
import pytest

import audonnx


@pytest.mark.parametrize(
    'path, transform, labels, expected',
    [
        (
            pytest.MODEL_SINGLE_PATH,
            pytest.SPECTROGRAM,
            None,
            {'gender': ['gender-0', 'gender-1']},
        ),
        (
            pytest.MODEL_SINGLE_PATH,
            pytest.SPECTROGRAM,
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
                transform=pytest.SPECTROGRAM,
            ),
            None,
            np.array([2.29, 1.21], np.float32),
        ),
        (
            audonnx.Model(
                pytest.MODEL_SINGLE_PATH,
                transform=pytest.SPECTROGRAM,
            ),
            'gender',
            np.array([2.29, 1.21], np.float32),
        ),
        (
            audonnx.Model(
                pytest.MODEL_SINGLE_PATH,
                transform=pytest.SPECTROGRAM,
            ),
            ['gender'],
            {'gender': np.array([2.29, 1.21], np.float32)},
        ),
        (
            audonnx.Model(
                pytest.MODEL_MULTI_PATH,
                transform={
                    'spectrogram': pytest.SPECTROGRAM,
                }
            ),
            None,
            {
                'hidden': np.array([-7.31e-03, -3.56e-01, -2.38e-01,
                                    3.30e-01, -2.77e+00, 2.63e+00,
                                    -9.87e+00, 3.06e-01], np.float32),
                'gender': np.array([-3.53, -3.55], np.float32),
                'confidence': np.array(-1.33, np.float32),
            },
        ),
        (
            audonnx.Model(
                pytest.MODEL_MULTI_PATH,
                transform={
                    'spectrogram': pytest.SPECTROGRAM,
                }
            ),
            ['confidence', 'gender'],
            {
                'confidence': np.array(-1.33, np.float32),
                'gender': np.array([-3.53, -3.55], np.float32),
            },
        ),
        (
            audonnx.Model(
                pytest.MODEL_MULTI_PATH,
                transform={
                    'spectrogram': pytest.SPECTROGRAM,
                }
            ),
            'confidence',
            np.array(-1.33, np.float32),
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
            np.testing.assert_almost_equal(y[key], expected[key], decimal=2)
    else:
        np.testing.assert_almost_equal(y, expected, decimal=2)


@pytest.mark.parametrize(
    'path, labels, transform',
    [
        (
            pytest.MODEL_SINGLE_PATH,
            None,
            pytest.SPECTROGRAM,
        ),
        (
            pytest.MODEL_MULTI_PATH,
            {
                'gender': ['female', 'male'],
            },
            {
                'spectrogram': pytest.SPECTROGRAM,
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
            pytest.SPECTROGRAM,
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
                transform=pytest.SPECTROGRAM,
            ), r'''Input:
  spectrogram:
    shape: [8, -1]
    dtype: tensor(float)
    transform: audsp.core.spectrogram.Spectrogram
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
                    'spectrogram': pytest.SPECTROGRAM,
                }
            ), r'''Input:
  signal:
    shape: [1, -1]
    dtype: tensor(float)
    transform: None
  spectrogram:
    shape: [8, -1]
    dtype: tensor(float)
    transform: audsp.core.spectrogram.Spectrogram
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
