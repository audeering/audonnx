import os

import numpy as np
import oyaml as yaml
import pytest

import audonnx


@pytest.mark.parametrize(
    'path, labels, expected',
    [
        (
            pytest.MODEL_MULTI_PATH,
            {
                'gender': ['female', 'male'],
            },
            {
                'hidden': np.array([-0.7, -2.04, -3.12, 3.84,
                                    -0.06, -6.34, 1.36, -5.46], np.float32),
                'gender': np.array([1.78, 1.22], np.float32),
                'confidence': np.array(2.6, np.float32),
            },
        ),
    ]
)
def test_load(path, labels, expected):

    root = os.path.dirname(path)

    transform_path = os.path.join(root, 'transform.yaml')
    pytest.FEATURE.to_yaml(transform_path)

    labels_path = os.path.join(root, 'labels.yaml')
    with open(labels_path, 'w') as fp:
        yaml.dump(labels, fp)

    model = audonnx.load(
        root,
        model_file=os.path.basename(path),
        transform_file=os.path.basename(transform_path),
        labels_file=os.path.basename(labels_path),
    )
    y = model(pytest.SIGNAL, pytest.SAMPLING_RATE)

    for key, values in y.items():
        np.testing.assert_almost_equal(y[key], expected[key], decimal=2)
    for key, values in labels.items():
        assert model.labels[key] == labels[key]
