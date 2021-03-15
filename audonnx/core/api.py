import os

import oyaml as yaml

import audeer
import audobject

from audonnx.core.model import Model


def load(
        root: str,
        *,
        device: str = 'cpu',
        name: str = 'model.onnx',
) -> Model:

    root = audeer.safe_path(root)
    labels = None
    transform = None

    path = os.path.join(root, 'labels.yaml')
    if os.path.exists(path):
        with open(path, 'r') as fp:
            labels = yaml.load(fp, yaml.BaseLoader)

    path = os.path.join(root, 'transform.yaml')
    if os.path.exists(path):
        transform = audobject.Object.from_yaml(path)

    model = Model(
        os.path.join(root, name),
        device=device,
        labels=labels,
        transform=transform,
    )

    return model
