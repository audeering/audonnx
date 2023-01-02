import glob
import os
import shutil

import pytest

import audeer
import audiofile
import opensmile


pytest.ROOT = audeer.path(
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
