import pytest

import audonnx


@pytest.mark.parametrize(
    'node, expected',
    [
        (
            audonnx.InputNode(
                [1, -1],
                'tensor(float)',
                None,
            ),
            "{shape: [1, -1], dtype: tensor(float), transform: None}"
        ),
        (
            audonnx.InputNode(
                [pytest.FEATURE_SHAPE[0], -1],
                'tensor(float)',
                pytest.FEATURE,
            ),
            "{shape: [18, -1], dtype: tensor(float), "
            "transform: opensmile.core.smile.Smile}"
        ),
        (
            audonnx.OutputNode(
                [1, 2],
                'tensor(float)',
                ['female', 'male'],
            ),
            "{shape: [1, 2], dtype: tensor(float), labels: [female, male]}"
        ),
    ],
)
def test_str(node, expected):
    assert str(node) == expected
