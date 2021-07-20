import os

import oyaml as yaml

import audeer
import audobject

from audonnx.core.model import Model


def load(
        root: str,
        *,
        model_file: str = 'model.yaml',
        labels_file: str = 'labels.yaml',
        transform_file: str = 'transform.yaml',
) -> Model:
    r"""Load model from folder.

    Tries to load model from YAML file.
    Otherwise creates object from ONNX file (legacy mode).

    Args:
        root: root folder
        model_file: model file
        labels_file: yaml file with labels
        transform_file: yaml file with transformation

    Returns:
        model

    Examples:
        >>> model = load('tests')
        >>> model
        Input:
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
            labels: [female, male]
          confidence:
            shape: [1]
            dtype: tensor(float)
            labels: [confidence]

    """  # noqa: E501

    root = audeer.safe_path(root)

    # try to load object from YAML file

    path = os.path.join(root, model_file)
    if audeer.file_extension(path) == 'yaml':
        if os.path.exists(path):
            return Model.from_yaml(path)

    # otherwise create object from ONNX file

    model_file = audeer.replace_file_extension(model_file, 'onnx')
    labels = None
    transform = None

    path = os.path.join(root, labels_file)
    if os.path.exists(path):
        with open(path, 'r') as fp:
            labels = yaml.load(fp, yaml.BaseLoader)

    path = os.path.join(root, transform_file)
    if os.path.exists(path):
        transform = audobject.Object.from_yaml(path)

    model = Model(
        os.path.join(root, model_file),
        labels=labels,
        transform=transform,
    )

    return model
