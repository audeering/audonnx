=======
audonnx
=======

|tests| |coverage| |docs| |python-versions| |license|

**audonnx** deploys machine learning models stored in ONNX_ format.

Machine learning models can be trained
in a variety of frameworks,
e.g. scikit-learn_, TensorFlow_, Torch_.
To be independent of the training framework
and its version models can be exported in ONNX_ format,
which enables you to deploy and combine them easily.

**audonnx** allows you to name inputs and outputs
of your model,
and automatically loads the correct feature extraction
from a YAML file stored with your model.

Have a look at the installation_ and usage_ instructions.

.. _ONNX: https://onnx.ai/
.. _scikit-learn: https://scikit-learn.org
.. _TensorFlow: https://www.tensorflow.org
.. _Torch: https://pytorch.org/
.. _installation: https://audeering.github.io/audonnx/install.html
.. _usage: https://audeering.github.io/audonnx/usage.html


.. badges images and links:
.. |tests| image:: https://github.com/audeering/audonnx/workflows/Test/badge.svg
    :target: https://github.com/audeering/audonnx/actions?query=workflow%3ATest
    :alt: Test status
.. |coverage| image:: https://codecov.io/gh/audeering/audonnx/branch/master/graph/badge.svg?token=UGxnVQiKGK
    :target: https://codecov.io/gh/audeering/audonnx/
    :alt: code coverage
.. |docs| image:: https://img.shields.io/pypi/v/audonnx?label=docs
    :target: https://audeering.github.io/audonnx/
    :alt: audonnx's documentation
.. |license| image:: https://img.shields.io/badge/license-MIT-green.svg
    :target: https://github.com/audeering/audonnx/blob/master/LICENSE
    :alt: audonnx's MIT license
.. |python-versions| image:: https://img.shields.io/pypi/pyversions/audonnx.svg
    :target: https://pypi.org/project/audonnx/
    :alt: audonnx's supported Python versions
