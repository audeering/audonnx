import os
import typing

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
        device: typing.Union[
            str,
            typing.Tuple[str, typing.Dict],
            typing.Sequence[typing.Union[str, typing.Tuple[str, typing.Dict]]],
        ] = 'cpu',
) -> Model:
    r"""Load model from folder.

    Tries to load model from YAML file.

    Otherwise creates object from ONNX file (legacy mode).
    In this case it will load
    labels and the transform
    from the corresponding YAML files
    if provided.
    The model is expected to be located at
    ``audeer.replace_file_extension(model_file, 'onnx')``.

    Args:
        root: root folder
        model_file: model YAML file,
            that needs to end with ``.yaml``.
            In legacy mode path to model ONNX file
        labels_file: YAML file with labels
        transform_file: YAML file with transformation
        device: set device
            (``'cpu'``, ``'cuda'``, or ``'cuda:<id>'``)
            or a (list of) provider_

    .. _provider: https://onnxruntime.ai/docs/execution-providers/

    Returns:
        model

    Examples:
        >>> model = load('tests')
        >>> model
        Input:
          feature:
            shape: [18, -1]
            dtype: tensor(float)
            transform: opensmile.core.smile.Smile
        Output:
          gender:
            shape: [2]
            dtype: tensor(float)
            labels: [female, male]

    """  # noqa: E501

    root = audeer.safe_path(root)
    model_file = os.path.join(root, model_file)
    model_file_yaml = audeer.replace_file_extension(model_file, 'yaml')
    if audeer.file_extension(model_file) == 'yaml':
        model_file_onnx = audeer.replace_file_extension(model_file, 'onnx')
    else:
        model_file_onnx = model_file

    # Try to load object from YAML file

    if os.path.exists(model_file_yaml):
        with open(model_file_yaml) as f:
            first_line = f.readline()
        if first_line.startswith('$audonnx'):  # ensure correct object
            return audobject.from_yaml(
                model_file_yaml,
                override_args={
                    'device': device,
                },
            )
    # LEGACY support
    # Otherwise create object from ONNX file

    labels_file = os.path.join(root, labels_file)
    transform_file = os.path.join(root, transform_file)

    labels = None
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as fp:
            labels = yaml.load(fp, yaml.BaseLoader)

    transform = None
    if os.path.exists(transform_file):
        transform = audobject.from_yaml(transform_file)

    model = Model(
        model_file_onnx,
        labels=labels,
        transform=transform,
        device=device,
    )

    return model
