import numpy as np
import pytest

import audonnx


@pytest.mark.parametrize(
    'root, signal, sampling_rate, expected',
    [
        (
            pytest.MODEL_ROOT,
            pytest.SIGNAL,
            pytest.SAMPLING_RATE,
            np.array(
                [-5.63, 0.85, 2.84, 2.96],
                dtype=np.float32,
            ).reshape(1, 4),
        ),
    ]
)
def test_load(root, signal, sampling_rate, expected):

    model = audonnx.load(root)

    y = model.forward(signal, sampling_rate)
    np.testing.assert_almost_equal(y, expected, decimal=2)

    y = model.predict(signal, sampling_rate)
    assert y == model.labels[expected.argmax()]

    x = model.transform(signal, sampling_rate)
    model.transform = None
    y = model.forward(x, 0)
    np.testing.assert_almost_equal(y, expected, decimal=2)
