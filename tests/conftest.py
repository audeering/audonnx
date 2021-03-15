import os

import pytest

import audiofile
import audmodel


pytest.MODEL_ID = 'e157969a-9898-6c62-f0b3-2d7b0f4ad9c7'
pytest.MODEL_ROOT = audmodel.load(pytest.MODEL_ID)
pytest.SIGNAL, pytest.SAMPLING_RATE = audiofile.read(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test.wav')
)
