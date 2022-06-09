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

class TorchModel(torch.nn.Module):

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


pytest.MODEL_PATH = os.path.join(pytest.TMP, 'model.onnx')
torch.onnx.export(
    TorchModel(),
    torch.randn(pytest.FEATURE_SHAPE),
    pytest.MODEL_PATH,
    input_names=['feature'],
    output_names=['gender'],
    dynamic_axes={'feature': {1: 'time'}},
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
