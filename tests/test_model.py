import numpy as np
import pytest

import audonnx


@pytest.mark.parametrize(
    'path, labels, expected',
    [
        (
            pytest.MODEL_SINGLE_PATH,
            None,
            {'gender': ['gender-0', 'gender-1']},
        ),
        (
            pytest.MODEL_SINGLE_PATH,
            ['female', 'male'],
            {'gender': ['female', 'male']},
        ),
        pytest.param(
            pytest.MODEL_SINGLE_PATH,
            ['too', 'many', 'labels'],
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ]
)
def test_labels(path, labels, expected):
    model = audonnx.Model(
        path,
        labels=labels,
        transform=pytest.FEATURE,
    )
    assert model.labels == expected


@pytest.mark.parametrize(
    'model, output_names, expected',
    [
        (
            audonnx.Model(
                pytest.MODEL_SINGLE_PATH,
                transform=pytest.FEATURE,
            ),
            None,
            np.array([2.29, 1.21], np.float32),
        ),
        (
            audonnx.Model(
                pytest.MODEL_SINGLE_PATH,
                transform=pytest.FEATURE,
            ),
            'gender',
            np.array([2.29, 1.21], np.float32),
        ),
        (
            audonnx.Model(
                pytest.MODEL_SINGLE_PATH,
                transform=pytest.FEATURE,
            ),
            ['gender'],
            {'gender': np.array([2.29, 1.21], np.float32)},
        ),
        (
            audonnx.Model(
                pytest.MODEL_MULTI_PATH,
                transform=pytest.FEATURE,
            ),
            None,
            {
                'hidden': np.array([-0.7, -2.04, -3.12, 3.84,
                                    -0.06, -6.34, 1.36, -5.46], np.float32),
                'gender': np.array([1.78, 1.22], np.float32),
                'confidence': np.array(2.6, np.float32),
            },
        ),
        (
            audonnx.Model(
                pytest.MODEL_MULTI_PATH,
                transform=pytest.FEATURE,
            ),
            ['confidence', 'gender'],
            {
                'confidence': np.array(2.6, np.float32),
                'gender': np.array([1.78, 1.22], np.float32),
            },
        ),
        (
            audonnx.Model(
                pytest.MODEL_MULTI_PATH,
                transform=pytest.FEATURE,
            ),
            'confidence',
            np.array(2.6, np.float32),
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
    'model',
    [
        audonnx.Model(
            pytest.MODEL_SINGLE_PATH,
            transform=pytest.FEATURE,
        ),
        audonnx.Model(
            pytest.MODEL_MULTI_PATH,
            transform=pytest.FEATURE,
        ),
        audonnx.Model(
            pytest.MODEL_MULTI_PATH,
            labels={
                'gender': ['female', 'male'],
            },
            transform=pytest.FEATURE,
        ),
    ],
)
def test_properties(model):

    inputs = model.sess.get_inputs()
    outputs = model.sess.get_outputs()

    assert model.input_name == inputs[0].name
    assert model.input_shape == inputs[0].shape
    assert model.input_type == inputs[0].type
    for idx, output_name in enumerate(model.output_names):
        assert output_name == outputs[idx].name
        if outputs[idx].shape:
            assert model.output_shape(output_name) == outputs[idx].shape
        else:
            assert model.output_shape(output_name) == [1]
        assert model.output_type(output_name) == outputs[idx].type


@pytest.mark.parametrize(
    'model, expected',
    [
        (
            audonnx.Model(
                pytest.MODEL_SINGLE_PATH,
                labels=['female', 'male'],
                transform=pytest.FEATURE,
            ), r'''audonnx.core.model.Model:
  Input:
  - input
  - [1, 1, 8, time]
  - tensor(float)
  Output(s):
    gender:
    - [2]
    - tensor(float)
    - [female, male]'''
        ),
        (
            audonnx.Model(
                pytest.MODEL_MULTI_PATH,
                transform=pytest.FEATURE,
            ), r'''audonnx.core.model.Model:
  Input:
  - input
  - [1, 1, 8, time]
  - tensor(float)
  Output(s):
    hidden:
    - [8]
    - tensor(float)
    - [hidden-0, hidden-1, hidden-2, '...', hidden-5, hidden-6, hidden-7]
    gender:
    - [2]
    - tensor(float)
    - [gender-0, gender-1]
    confidence:
    - [1]
    - tensor(float)
    - [confidence]'''
        )
    ],
)
def test_repr(model, expected):
    assert repr(model) == expected
