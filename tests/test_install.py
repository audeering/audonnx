import os
import pkg_resources
import subprocess
import sys

import pytest

import audonnx


def uninstall(
    package: str,
    module: str,
):
    # uninstall package
    subprocess.check_call(
        [
            sys.executable,
            '-m',
            'pip',
            'uninstall',
            '--yes',
            package,
        ]
    )
    # remove module
    for m in list(sys.modules):
        if m.startswith(package):
            sys.modules.pop(m)
    # force pkg_resources to re-scan site packages
    pkg_resources._initialize_master_working_set()


def test(tmpdir):

    model = audonnx.Model(
        pytest.MODEL_SINGLE_PATH,
        transform=pytest.FEATURE,
    )
    model_path = os.path.join(tmpdir, 'model.yaml')
    model.to_yaml(model_path)

    uninstall('opensmile', 'opensmile')

    with pytest.raises(ModuleNotFoundError):
        audonnx.load(tmpdir)

    audonnx.load(tmpdir, auto_install=True)
