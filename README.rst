=======
audonnx
=======

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
