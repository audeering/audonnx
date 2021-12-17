import glob
import os
import random
import shutil

import numpy as np
import opensmile

import pytest
import torch

import audeer
import audiofile
import audobject


pytest.ROOT = audeer.safe_path(
    os.path.dirname(os.path.realpath(__file__))
)
pytest.TMP = audeer.mkdir(
    os.path.join(
        pytest.ROOT,
        audeer.uid(),
    )
)
pytest.SIGNAL, pytest.SAMPLING_RATE = audiofile.read(
    os.path.join(pytest.ROOT, 'test.wav'),
    always_2d=True,
)

# feature extractor

pytest.FEATURE = opensmile.Smile(
    feature_set=opensmile.FeatureSet.GeMAPSv01b,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)
pytest.FEATURE_SHAPE = pytest.FEATURE(
    pytest.SIGNAL,
    pytest.SAMPLING_RATE,
).shape[1:]

# fix seed

seed = 1234
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


# create model with single output node

class TorchModelSingle(torch.nn.Module):

    def __init__(
        self,
    ):
        super().__init__()
        self.hidden = torch.nn.Linear(pytest.FEATURE_SHAPE[0], 8)
        self.out = torch.nn.Linear(8, 2)

    def forward(self, x: torch.Tensor):
        y = self.hidden(x.mean(dim=-1))
        y = self.out(y)
        return y.squeeze()


pytest.MODEL_SINGLE_PATH = os.path.join(pytest.TMP, 'single.onnx')
torch.onnx.export(
    TorchModelSingle(),
    torch.randn(pytest.FEATURE_SHAPE),
    pytest.MODEL_SINGLE_PATH,
    input_names=['feature'],
    output_names=['gender'],
    dynamic_axes={'feature': {1: 'time'}},
    opset_version=12,
)


# create model with multiple input and output nodes

class TorchModelMulti(torch.nn.Module):

    def __init__(
        self,
    ):

        super().__init__()

        self.hidden_left = torch.nn.Linear(1, 4)
        self.hidden_right = torch.nn.Linear(pytest.FEATURE_SHAPE[0], 4)
        self.out = torch.nn.ModuleDict(
            {
                'gender': torch.nn.Linear(8, 2),
                'confidence': torch.nn.Linear(8, 1),
            }
        )

    def forward(self, signal: torch.Tensor, spectrogram: torch.Tensor):

        y_left = self.hidden_left(signal.mean(dim=-1))
        y_right = self.hidden_right(spectrogram.mean(dim=-1))
        y_hidden = torch.cat([y_left, y_right], dim=-1)
        y_gender = self.out['gender'](y_hidden)
        y_confidence = self.out['confidence'](y_hidden)

        return (
            y_hidden.squeeze(),
            y_gender.squeeze(),
            y_confidence.squeeze(),
        )


pytest.MODEL_MULTI_PATH = os.path.join(pytest.TMP, 'multi.onnx')
torch.onnx.export(
    TorchModelMulti(),
    (
        torch.randn(pytest.SIGNAL.shape),
        torch.randn(pytest.FEATURE_SHAPE),
    ),
    pytest.MODEL_MULTI_PATH,
    input_names=['signal', 'feature'],
    output_names=['hidden', 'gender', 'confidence'],
    dynamic_axes={
        'signal': {1: 'time'},
        'feature': {1: 'time'},
    },
    opset_version=12,
)


# clean up

@pytest.fixture(scope='session', autouse=True)
def cleanup_session():
    path = os.path.join(
        pytest.TMP,
        '..',
        '.coverage.*',
    )
    for file in glob.glob(path):
        os.remove(file)
    yield
    if os.path.exists(pytest.TMP):
        shutil.rmtree(pytest.TMP)
