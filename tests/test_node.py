import pytest

import audonnx


@pytest.mark.parametrize(
    'node, expected',
    [
        (
            audonnx.InputNode([1, -1], 'tensor(float)', None),
            "{shape: [1, -1], dtype: tensor(float), transform: None}"
        ),
        (
            audonnx.InputNode([8, -1], 'tensor(float)', pytest.SPECTROGRAM),
            "{shape: [8, -1], dtype: tensor(float), "
            "transform: audsp.core.spectrogram.Spectrogram}"
        ),
        (
            audonnx.OutputNode([1, 2], 'tensor(float)', ['female', 'male']),
            "{shape: [1, 2], dtype: tensor(float), labels: [female, male]}"
        ),
    ],
)
def test_str(node, expected):
    assert str(node) == expected
