import os
import subprocess
import sys

import pkg_resources
import pytest

import audonnx.testing


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

    object = audonnx.testing.create_model_proto([pytest.FEATURE_SHAPE])
    model = audonnx.Model(
        object,
        transform=pytest.FEATURE,
    )
    model_path = os.path.join(tmpdir, 'model.yaml')
    model.to_yaml(model_path)

    # Removing the package does not work under Windows
    # as the DLL file cannot be unloaded
    if not sys.platform == 'win32':
        uninstall('opensmile', 'opensmile')

        with pytest.raises(ModuleNotFoundError):
            audonnx.load(tmpdir)

    audonnx.load(tmpdir, auto_install=True)
