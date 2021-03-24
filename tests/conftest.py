import os

import pytest

import audiofile
import audmodel
import audonnx


pytest.MODEL_ID = 'c3a709c9-0b58-48d1-7217-0aa3ea485d2e'
pytest.MODEL_ROOT = audmodel.load(pytest.MODEL_ID)
pytest.MODEL_PATH = os.path.join(pytest.MODEL_ROOT, 'model.onnx')
pytest.SIGNAL, pytest.SAMPLING_RATE = audiofile.read(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test.wav')
)
pytest.MODEL = audonnx.load(pytest.MODEL_ROOT)
