import re
import typing


def device_to_providers(
        device: typing.Union[
            str,
            typing.Tuple[str, typing.Dict],
            typing.Sequence[typing.Union[str, typing.Tuple[str, typing.Dict]]],
        ],
) -> typing.Sequence[typing.Union[str, typing.Tuple[str, typing.Dict]]]:
    r"""Converts device into a list of providers.

    Args:
        device: ``'cpu'``,
            ``'cuda'``,
            ``'cuda:<id>'``,
            or a (list of) `ONNX Runtime Execution Providers`_

    Returns:
        sequence of `ONNX Runtime Execution Providers`_

    Examples:
        >>> device_to_providers('cpu')
        ['CPUExecutionProvider']

    .. _ONNX Runtime Execution Providers: https://onnxruntime.ai/docs/execution-providers/

    """
    if isinstance(device, str):
        if device == 'cpu':
            providers = ['CPUExecutionProvider']
        elif device.startswith('cuda'):
            match = re.search(r'^cuda:(\d+)$', device)
            if match:
                device_id = match.group(1)
                providers = [
                    (
                        'CUDAExecutionProvider', {
                            'device_id': device_id,
                        }
                    ),
                ]
            else:
                providers = ['CUDAExecutionProvider']
        else:
            providers = [device]
    elif isinstance(device, tuple):
        providers = [device]
    else:
        providers = device
    return providers
