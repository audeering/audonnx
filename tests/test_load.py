import os

import numpy as np
import onnx
import oyaml as yaml
import pytest

import audeer
import audobject

import audonnx.testing


def min_max(x, sr):
    import numpy as np
    return np.array([x.min(), x.max()], np.float32)


@pytest.mark.parametrize(
    'object, transform, labels, expected',
    [
        (
            audonnx.testing.create_model_proto([[2]]),
            audonnx.Function(min_max),
            {'output-0': ['min', 'max']},
            min_max(pytest.SIGNAL, pytest.SAMPLING_RATE),
        ),
    ]
)
def test_load_legacy(tmpdir, object, transform, labels, expected):

    root = tmpdir

    onnx_path = os.path.join(root, 'model.onnx')
    onnx.save(object, onnx_path)

    # create from onnx

    transform_path = os.path.join(root, 'transform.yaml')
    transform.to_yaml(transform_path)

    labels_path = os.path.join(root, 'labels.yaml')
    with open(labels_path, 'w') as fp:
        yaml.dump(labels, fp)

    model = audonnx.load(
        root,
        model_file=os.path.basename(onnx_path),
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
        model_file=audeer.replace_file_extension(onnx_path, 'yaml'),
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
        model_file=os.path.basename(onnx_path),
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

    np.testing.assert_equal(y, expected)
    for key, values in labels.items():
        assert model.outputs[key].labels == labels[key]
