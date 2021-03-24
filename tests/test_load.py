import numpy as np
import pytest

import audonnx


@pytest.mark.parametrize(
    'path, labels, expected',
    [
        (
            pytest.MODEL_PATH,
            None,
            {
                'client-tone': ['client-tone-0', 'client-tone-1',
                                'client-tone-2', 'client-tone-3'],
                'client-gender': ['client-gender-0', 'client-gender-1'],
            }
        ),
        (
            pytest.MODEL_PATH,
            ['a', 'b', 'c', 'd'],
            {
                'client-tone': ['a', 'b', 'c', 'd'],
                'client-gender': ['client-gender-0', 'client-gender-1'],
            }
        ),
        (
            pytest.MODEL_PATH,
            {
                'client-tone': ['a', 'b', 'c', 'd'],
                'client-gender': ['e', 'f'],
            },
            {
                'client-tone': ['a', 'b', 'c', 'd'],
                'client-gender': ['e', 'f'],
            }
        ),
        (
            pytest.MODEL_PATH,
            {
                'client-tone': ['a', 'b', 'c', 'd'],
            },
            {
                'client-tone': ['a', 'b', 'c', 'd'],
                'client-gender': ['client-gender-0', 'client-gender-1'],
            }
        ),
        pytest.param(
            pytest.MODEL_PATH,
            ['a', 'b', 'c'],
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ]
)
def test_labels(path, labels, expected):

    model = audonnx.Model(path, labels=labels,)
    assert model.labels == expected


@pytest.mark.parametrize(
    'model, signal, sampling_rate',
    [
        (
            pytest.MODEL,
            pytest.SIGNAL,
            pytest.SAMPLING_RATE,
        )
    ]
)
@pytest.mark.parametrize(
    'output_names, expected',
    [
        (
            'client-tone',
            np.array(
                [1.37, 3.02, 2.28, -7.73],
                dtype=np.float32,
            ).reshape(1, 4),
        ),
        (
            ['client-tone'],
            {
                'client-tone':
                    np.array(
                        [1.37, 3.02, 2.28, -7.73],
                        dtype=np.float32,
                    ).reshape(1, 4),
            },
        ),
        (
            None,
            {
                'client-tone':
                    np.array(
                        [1.37, 3.02, 2.28, -7.73],
                        dtype=np.float32,
                    ).reshape(1, 4),
                'client-gender':
                    np.array(
                        [-3.58, 1.75],
                        dtype=np.float32,
                    ).reshape(1, 2),
            },
        ),
    ]
)
def test_forward(model, signal, sampling_rate, output_names, expected):

    y = model.forward(signal, sampling_rate, output_names=output_names)

    if isinstance(output_names, str):
        np.testing.assert_almost_equal(y, expected, decimal=2)
    else:
        if output_names is None:
            output_names = model.output_names
        for output_name in output_names:
            np.testing.assert_almost_equal(
                y[output_name],
                expected[output_name],
                decimal=2,
            )


@pytest.mark.parametrize(
    'model, signal, sampling_rate',
    [
        (
            pytest.MODEL,
            pytest.SIGNAL,
            pytest.SAMPLING_RATE,
        )
    ]
)
@pytest.mark.parametrize(
    'output_names, expected',
    [
        (
            'client-tone',
            'negative',
        ),
        (
            ['client-tone'],
            {
                'client-tone': 'negative',
            }
        ),
        (
            None,
            {
                'client-tone': 'negative',
                'client-gender': 'male',
            }
        ),
    ]
)
def test_predict(model, signal, sampling_rate, output_names, expected):

    y = model.predict(signal, sampling_rate, output_names=output_names)

    if isinstance(output_names, str):
        assert y == expected
    else:
        if output_names is None:
            output_names = model.output_names
        for output_name in output_names:
            assert y[output_name] == expected[output_name]


def test_properties():

    model = pytest.MODEL
    inputs = model.sess.get_inputs()
    outputs = model.sess.get_outputs()

    assert model.input_name == inputs[0].name
    assert model.input_shape == inputs[0].shape
    assert model.input_type == inputs[0].type
    for idx, output_name in enumerate(model.output_names):
        assert output_name == outputs[idx].name
        assert model.output_shape(output_name) == outputs[idx].shape
        assert model.output_type(output_name) == outputs[idx].type
