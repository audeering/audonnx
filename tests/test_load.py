import os

import audeer
import numpy as np
import oyaml as yaml
import pytest

import audobject
import audonnx


@pytest.mark.parametrize(
    'path, transform, labels, expected',
    [
        (
            pytest.MODEL_SINGLE_PATH,
            pytest.FEATURE,
            {
                'gender': ['female', 'male'],
            },
            np.array([-195.1, 73.3], np.float32),
        ),
    ]
)
def test_load_legacy(tmpdir, path, transform, labels, expected):

    root = os.path.dirname(path)

    # create from onnx

    transform_path = os.path.join(root, 'transform.yaml')
    transform.to_yaml(transform_path)

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

    np.testing.assert_almost_equal(y, expected, decimal=1)
    for key, values in labels.items():
        assert model.outputs[key].labels == labels[key]

    # legacy mode -> load from ONNX if YAML does not exist

    model = audonnx.load(
        root,
        model_file=audeer.replace_file_extension(path, 'yaml'),
    )
    y = model(pytest.SIGNAL, pytest.SAMPLING_RATE)

    np.testing.assert_almost_equal(y, expected, decimal=1)
    for key, values in labels.items():
        assert model.outputs[key].labels == labels[key]

    # store wrong model YAML

    model_path = os.path.join(tmpdir, 'single.yaml')
    audobject.Object().to_yaml(model_path)
    model = audonnx.load(
        root,
        model_file=os.path.basename(path),
    )

    # file extension does not end on '.yaml'

    model_path = os.path.join(tmpdir, 'model.yml')
    with pytest.raises(ValueError):
        model.to_yaml(model_path)

    # create from YAML

    model_path = os.path.join(tmpdir, 'model.yaml')
    model.to_yaml(model_path)

    model = audonnx.load(tmpdir)
    y = model(pytest.SIGNAL, pytest.SAMPLING_RATE)

    np.testing.assert_almost_equal(y, expected, decimal=1)
    for key, values in labels.items():
        assert model.outputs[key].labels == labels[key]
