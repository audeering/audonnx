import os

import oyaml as yaml

import audeer
import audobject

from audonnx.core.model import Model


def load(
        root: str,
        *,
        model_file: str = 'model.onnx',
        labels_file: str = 'labels.yaml',
        transform_file: str = 'transform.yaml',
) -> Model:
    r"""Load model from folder.

    Args:
        root: root folder
        model_file: model file
        labels_file: yaml file with labels
        transform_file: yaml file with transformation

    Returns:
        model

    """

    root = audeer.safe_path(root)
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
