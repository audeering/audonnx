import importlib.metadata
import os
import subprocess
import sys

import pytest

import audonnx.testing


def force_package_rescan():
    """Force a rescan of installed packages."""
    # Clear importlib.metadata caches
    if hasattr(importlib.metadata, "_cache"):
        importlib.metadata._cache.clear()
    # Clear distribution cache
    if hasattr(importlib.metadata.distributions, "cache_clear"):
        importlib.metadata.distributions.cache_clear()
    # Rescan packages
    importlib.metadata.distributions()


def uninstall(
    package: str,
    module: str,
):
    # uninstall package
    subprocess.run(["uv", "pip", "uninstall", package])
    # remove module
    for m in list(sys.modules):
        if m.startswith(package):
            sys.modules.pop(m)

    # force pkg_resources to re-scan site packages
    force_package_rescan()


def test(tmpdir):
    object = audonnx.testing.create_model_proto([pytest.FEATURE_SHAPE])
    model = audonnx.Model(
        object,
        transform=pytest.FEATURE,
    )
    model_path = os.path.join(tmpdir, "model.yaml")
    model.to_yaml(model_path)

    # Removing the package does not work under Windows
    # as the DLL file cannot be unloaded
    if not sys.platform == "win32":
        uninstall("opensmile", "opensmile")

        with pytest.raises(ModuleNotFoundError):
            audonnx.load(tmpdir)

    audonnx.load(tmpdir, auto_install=True)
