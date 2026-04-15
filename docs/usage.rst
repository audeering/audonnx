Usage
=====

:mod:`audonnx` offers a simple interface
to load and use models in ONNX_ format.
Models with single or multiple input and output nodes are supported.

We begin with creating some test input -
a file path, a signal array and an index in audformat_.

.. code-block:: pycon

    >>> import audiofile
    >>> import pandas as pd
    >>> file = "test.wav"
    >>> signal, sampling_rate = audiofile.read(
    ...     file,
    ...     always_2d=True,
    ... )
    >>> index = pd.MultiIndex.from_arrays(
    ...     [
    ...         [file, file],
    ...         pd.to_timedelta(["0s", "3s"]),
    ...         pd.to_timedelta(["3s", "5s"]),
    ...     ],
    ...     names=["file", "start", "end"],
    ... )


Torch model
-----------

Create Torch_ model with a single input and output node.

.. code-block:: pycon

    >>> import torch
    >>> class TorchModelSingle(torch.nn.Module):
    ...
    ...     def __init__(
    ...         self,
    ...     ):
    ...         super().__init__()
    ...         self.hidden = torch.nn.Linear(18, 8)
    ...         self.out = torch.nn.Linear(8, 2)
    ...
    ...     def forward(self, x: torch.Tensor):
    ...         y = self.hidden(x.mean(dim=-1))
    ...         y = self.out(y)
    ...         return y.squeeze()
    >>> torch_model = TorchModelSingle()

Create OpenSMILE_ feature extractor to convert the
raw audio signal to a sequence of low-level descriptors.

.. code-block:: pycon

    >>> import opensmile
    >>> smile = opensmile.Smile(
    ...     feature_set=opensmile.FeatureSet.GeMAPSv01b,
    ...     feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    ... )

Calculate features and run Torch_ model.

.. code-block:: pycon

    >>> y = smile(signal, sampling_rate)
    >>> with torch.no_grad():
    ...     z = torch_model(torch.from_numpy(y))
    >>> z
    tensor([191.025..., 232.3628])


Export model
------------

To export the model to ONNX_ format,
we pass some dummy input,
which allows the function to figure out
correct input and output shapes.
Since the number of extracted feature frames
varies with the length of the input signal,
we tell the function that the last dimension
of the input has a dynamic size.
And we assign meaningful names to the nodes.

.. code-block:: pycon

    >>> import audeer
    >>> import os
    >>> onnx_root = audeer.mkdir("onnx")
    >>> onnx_model_path = os.path.join(onnx_root, "model.onnx")
    >>> dummy_input = torch.randn(y.shape[1:])
    >>> torch.onnx.export(
    ...     torch_model,
    ...     dummy_input,
    ...     onnx_model_path,
    ...     input_names=["feature"],  # assign custom name to input node
    ...     output_names=["gender"],  # assign custom name to output node
    ...     dynamic_axes={"feature": {1: "time"}},  # dynamic size
    ...     opset_version=12,
    ...     dynamo=False,
    ... )

From the exported model file
we now create an object of :class:`audonnx.Model`.
We pass the feature extractor,
so that the model can automatically convert the
input signal to the desired representation.
And we assign labels to the dimensions of the output node.
Printing the model provides a summary of
the input and output nodes.

.. code-block:: pycon

    >>> import audonnx
    >>> onnx_model = audonnx.Model(
    ...     onnx_model_path,
    ...     labels=["female", "male"],
    ...     transform=smile,
    ... )
    >>> onnx_model
    Input:
      feature:
        shape: [18, -1]
        dtype: tensor(float)
        transform: opensmile.core.smile.Smile
    Output:
      gender:
        shape: [2]
        dtype: tensor(float)
        labels: [female, male]

Get information for individual nodes.

.. code-block:: pycon

    >>> onnx_model.inputs["feature"]
    {shape: [18, -1], dtype: tensor(float), transform: opensmile.core.smile.Smile}

.. code-block:: pycon

    >>> print(onnx_model.inputs["feature"].transform)
    $opensmile.core.smile.Smile:
      feature_set: GeMAPSv01b
      feature_level: LowLevelDescriptors
      options: {}
      sampling_rate: null
      channels:
      - 0
      mixdown: false
      resample: false
    <BLANKLINE>

.. code-block:: pycon

    >>> onnx_model.outputs["gender"]
    {shape: [2], dtype: tensor(float), labels: [female, male]}

.. code-block:: pycon

    >>> onnx_model.outputs["gender"].labels
    ['female', 'male']

Check that the exported model gives the expected output.

.. code-block:: pycon

    >>> onnx_model(signal, sampling_rate)
    array([191.02567, 232.36278], dtype=float32)


Create interface
----------------

:class:`onnx.Model` does not come with a fancy interface itself,
but we can use audinterface_ to create one.

.. code-block:: pycon

    >>> import numpy as np
    >>> import audinterface
    >>> interface = audinterface.Feature(
    ...     feature_names=onnx_model.outputs["gender"].labels,
    ...     process_func=onnx_model,
    ... )
    >>> interface.process_index(index)
                                                  female        male
    file     start           end
    test.wav 0 days 00:00:00 0 days 00:00:03  191.204071  232.138229
             0 days 00:00:03 0 days 00:00:05  190.130585  230.189331

Or if we are only interested in the majority class.

.. code-block:: pycon

    >>> interface.process_index(index).idxmax(axis=1)
    file      start            end
    test.wav  0 days 00:00:00  0 days 00:00:03    male
              0 days 00:00:03  0 days 00:00:05    male
    dtype: object


Save and load
-------------

Save the model to a YAML file.

.. code-block:: pycon

    >>> onnx_meta_path = os.path.join(onnx_root, "model.yaml")
    >>> onnx_model.to_yaml(onnx_meta_path)

.. code-block:: pycon

    >>> import oyaml as yaml
    >>> with open(onnx_meta_path, "r") as fp:
    ...     d = yaml.load(fp, Loader=yaml.Loader)
    >>> print(yaml.dump(d))
    $audonnx.core.model.Model==...:
      path: model.onnx
      labels:
      - female
      - male
      transform:
        $opensmile.core.smile.Smile==...:
          feature_set: GeMAPSv01b
          feature_level: LowLevelDescriptors
          options: {}
          sampling_rate: null
          channels:
          - 0
          mixdown: false
          resample: false
    <BLANKLINE>

Load the model from a YAML file.

.. code-block:: pycon

    >>> import audobject
    >>> onnx_model_2 = audobject.from_yaml(onnx_meta_path)
    >>> onnx_model_2(signal, sampling_rate)
    array([191.02567, 232.36278], dtype=float32)

Or shorter:

.. code-block:: pycon

    >>> onnx_model_3 = audonnx.load(onnx_root)
    >>> onnx_model_3(signal, sampling_rate)
    array([191.02567, 232.36278], dtype=float32)


Quantize weights
----------------

To reduce the memory print of a model,
we can quantize it,
compare the `MobilenetV2 example`_.
For instance, we can store model weights as 8 bit integers.
For quantization make sure
you have installed
``onnx``
as well as
``onnxruntime``.

.. skip: start

.. code-block:: pycon

    >>> import onnxruntime.quantization
    >>> onnx_infer_path = os.path.join(onnx_root, "model_infer.onnx")
    >>> onnxruntime.quantization.quant_pre_process(
    ...     onnx_model_path,
    ...     onnx_infer_path,
    ... )
    >>> onnx_quant_path = os.path.join(onnx_root, "model_quant.onnx")
    >>> onnxruntime.quantization.quantize_dynamic(
    ...     onnx_infer_path,
    ...     onnx_quant_path,
    ...     weight_type=onnxruntime.quantization.QuantType.QUInt8,
    ... )

The output of the quantized model differs slightly.

.. code-block:: pycon

    >>> onnx_model_4 = audonnx.Model(
    ...     onnx_quant_path,
    ...     labels=["female", "male"],
    ...     transform=smile,
    ... )
    >>> onnx_model_4(signal, sampling_rate)
    array([191.16478, 232.2706 ], dtype=float32)

.. skip: end


Custom transform
----------------

So far,
we have used
:class:`opensmile.Smile`
as feature extractor.
It derives from
:class:`audobject.Object`
and is therefore serializable by default.
However,
using
:class:`audonnx.Function`
we can turn any function
into a serializable object.
For instance,
we can define a function that extracts
Mel-frequency cepstral coefficients (MFCCs)
with librosa_.

.. code-block:: pycon

    >>> def mfcc(x, sr):
    ...     import librosa  # import here to make function self-contained
    ...     y = librosa.feature.mfcc(
    ...         y=x.squeeze(),
    ...         sr=sr,
    ...         n_mfcc=18,
    ...     )
    ...     return y.reshape(1, 18, -1)

As long as the function is self-contained
(i.e. does not depend on external variables or imports)
we can turn it into a serializable object.

.. code-block:: pycon

    >>> transform = audonnx.Function(mfcc)
    >>> print(transform)
    $audonnx.core.function.Function:
      func: "def mfcc(x, sr):\n    import librosa  # import here to make function self-contained\n\
        \    y = librosa.feature.mfcc(\n        y=x.squeeze(),\n        sr=sr,\n     \
        \   n_mfcc=18,\n    )\n    return y.reshape(1, 18, -1)\n"
      func_args: {}
    <BLANKLINE>

And use it to initialize our model.

.. code-block:: pycon

    >>> onnx_model_5 = audonnx.Model(
    ...     onnx_model_path,
    ...     labels=["female", "male"],
    ...     transform=transform,
    ... )
    >>> onnx_model_5
    Input:
      feature:
        shape: [18, -1]
        dtype: tensor(float)
        transform: audonnx.core.function.Function(mfcc)
    Output:
      gender:
        shape: [2]
        dtype: tensor(float)
        labels: [female, male]

Then we can save and load the model as before.

.. code-block:: pycon

    >>> onnx_model_5.to_yaml(onnx_meta_path)
    >>> onnx_model_6 = audonnx.load(onnx_root)
    >>> onnx_model_6(signal, sampling_rate)
    array([43.0061..., -2.545765...], dtype=float32)


Multiple nodes
--------------

Define a model that takes as input the
raw audio in addition to the features
and provides two more output nodes -
the output from the hidden layer and a confidence value.

.. code-block:: pycon

    >>> class TorchModelMulti(torch.nn.Module):
    ...
    ...     def __init__(
    ...         self,
    ...     ):
    ...         super().__init__()
    ...         self.hidden_left = torch.nn.Linear(1, 4)
    ...         self.hidden_right = torch.nn.Linear(18, 4)
    ...         self.out = torch.nn.ModuleDict(
    ...             {
    ...                 "gender": torch.nn.Linear(8, 2),
    ...                 "confidence": torch.nn.Linear(8, 1),
    ...             }
    ...         )
    ...
    ...     def forward(self, signal: torch.Tensor, feature: torch.Tensor):
    ...         y_left = self.hidden_left(signal.mean(dim=-1))
    ...         y_right = self.hidden_right(feature.mean(dim=-1))
    ...         y_hidden = torch.cat([y_left, y_right], dim=-1)
    ...         y_gender = self.out["gender"](y_hidden)
    ...         y_confidence = self.out["confidence"](y_hidden)
    ...         return (
    ...             y_hidden.squeeze(),
    ...             y_gender.squeeze(),
    ...             y_confidence,
    ...         )

Export the new model to ONNX_ format and load it.
Note that we do not assign labels to all output nodes.
In that case, they are automatically created
from the name of the output node.
And since the first node expects the raw audio signal,
we do not set a transform for it.

.. code-block:: pycon

    >>> onnx_multi_root = audeer.mkdir("onnx_multi")
    >>> onnx_multi_path = os.path.join(onnx_multi_root, "model.onnx")
    >>> torch.onnx.export(
    ...     TorchModelMulti(),
    ...     (
    ...         torch.randn(signal.shape),
    ...         torch.randn(y.shape[1:]),
    ...     ),
    ...     onnx_multi_path,
    ...     input_names=["signal", "feature"],
    ...     output_names=["hidden", "gender", "confidence"],
    ...     dynamic_axes={
    ...         "signal": {1: "time"},
    ...         "feature": {1: "time"},
    ...     },
    ...     opset_version=12,
    ...     dynamo=False,
    ... )
    >>> onnx_model_7 = audonnx.Model(
    ...     onnx_multi_path,
    ...     labels={
    ...         "gender": ["female", "male"],
    ...     },
    ...     transform={
    ...         "feature": smile,
    ...     },
    ... )
    >>> onnx_model_7
    Input:
      signal:
        shape: [1, -1]
        dtype: tensor(float)
        transform: None
      feature:
        shape: [18, -1]
        dtype: tensor(float)
        transform: opensmile.core.smile.Smile
    Output:
      hidden:
        shape: [8]
        dtype: tensor(float)
        labels: [hidden-0, hidden-1, hidden-2, (...), hidden-5, hidden-6, hidden-7]
      gender:
        shape: [2]
        dtype: tensor(float)
        labels: [female, male]
      confidence:
        shape: [1]
        dtype: tensor(float)
        labels: [confidence]

By default,
returns a dictionary with output for every node.

.. code-block:: pycon

    >>> onnx_model_7(signal, sampling_rate)
    {'hidden': array([ 7.6037818e-01, -1.2064241e-02,  3.2603091e-01, -4.6754807e-01,
           -4.1000482e+02,  7.2107361e+01, -5.6038922e+02,  1.9108322e+01],
          dtype=float32), 'gender': array([307.07407 ,  22.489958], dtype=float32), 'confidence': array([-92.46997], dtype=float32)}

To request a specific node use the ``outputs`` argument.

.. code-block:: pycon

    >>> onnx_model_7(
    ...     signal,
    ...     sampling_rate,
    ...     outputs="gender",
    ... )
    array([307.07407 ,  22.489958], dtype=float32)

Or provide a list of names to request several outputs.

.. code-block:: pycon

    >>> onnx_model_7(
    ...     signal,
    ...     sampling_rate,
    ...     outputs=["gender", "confidence"],
    ... )
    {'gender': array([307.07407 ,  22.489958], dtype=float32), 'confidence': array([-92.46997], dtype=float32)}

To concatenate the outputs to a single array,
do:

.. code-block:: pycon

    >>> onnx_model_7(
    ...     signal,
    ...     sampling_rate,
    ...     outputs=["gender", "confidence"],
    ...     concat=True,
    ... )
    array([307.07407 ,  22.489958, -92.46997 ], dtype=float32)

Create interface and process a file.

.. code-block:: pycon

    >>> outputs = ["gender", "confidence"]
    >>> interface = audinterface.Feature(
    ...     feature_names=onnx_model_7.labels(outputs),
    ...     process_func=onnx_model_7,
    ...     process_func_args={
    ...         "outputs": outputs,
    ...         "concat": True,
    ...     },
    ... )
    >>> interface.process_file(file)
                                                   female       male  confidence
    file     start  end
    test.wav 0 days 0 days 00:00:05.247687500  307.074066  22.489958  -92.469971


Additional input values
-----------------------

In some cases it may be useful to
pass additional inputs to the model
without applying a transform
on a signal.

Here we create the same model as before
but without setting a transform
for the ``feature`` input.

.. code-block:: pycon

    >>> onnx_model_8 = audonnx.Model(
    ...     onnx_multi_path,
    ...     labels={
    ...         "gender": ["female", "male"],
    ...     },
    ... )
    >>> onnx_model_8
    Input:
      signal:
        shape: [1, -1]
        dtype: tensor(float)
        transform: None
      feature:
        shape: [18, -1]
        dtype: tensor(float)
        transform: None
    Output:
      hidden:
        shape: [8]
        dtype: tensor(float)
        labels: [hidden-0, hidden-1, hidden-2, (...), hidden-5, hidden-6, hidden-7]
      gender:
        shape: [2]
        dtype: tensor(float)
        labels: [female, male]
      confidence:
        shape: [1]
        dtype: tensor(float)
        labels: [confidence]

We can then pass all inputs
as a dictionary when calling the model.

.. code-block:: pycon

    >>> onnx_model_8(
    ...     {"signal": signal, "feature": y},
    ...     sampling_rate,
    ... )
    {'hidden': array([ 7.6037818e-01, -1.2064241e-02,  3.2603091e-01, -4.6754807e-01,
           -4.1000482e+02,  7.2107361e+01, -5.6038922e+02,  1.9108322e+01],
          dtype=float32), 'gender': array([307.07407 ,  22.489958], dtype=float32), 'confidence': array([-92.46997], dtype=float32)}

It is also possible to create a model
that doesn't use a ``signal`` as input.

.. code-block:: pycon

    >>> onnx_model_9 = audonnx.Model(
    ...     onnx_model_path,
    ...     labels=["female", "male"],
    ... )
    >>> onnx_model_9
    Input:
      feature:
        shape: [18, -1]
        dtype: tensor(float)
        transform: None
    Output:
      gender:
        shape: [2]
        dtype: tensor(float)
        labels: [female, male]

When calling this model,
we only need to supply the ``feature`` input
and can ignore the ``sampling_rate``.

.. code-block:: pycon

    >>> onnx_model_9(y)
    array([191.02567, 232.36278], dtype=float32)

We can also use :class:`audonnx.Function`
with a function with any arguments,
not just the arguments for signal and sampling rate.

.. code-block:: pycon

    >>> def feature_addition(my_input, offset=0):
    ...     return my_input + offset
    >>> transform = audonnx.Function(feature_addition)
    >>> print(transform)
    $audonnx.core.function.Function:
      func: "def feature_addition(my_input, offset=0):\n    return my_input + offset\n"
      func_args: {}
    <BLANKLINE>

We use this transform for the ``feature`` input
of our multi-input model:

.. code-block:: pycon

    >>> onnx_model_10 = audonnx.Model(
    ...     onnx_multi_path,
    ...     labels={
    ...         "gender": ["female", "male"],
    ...     },
    ...     transform={
    ...         "feature": transform,
    ...     },
    ... )
    >>> onnx_model_10
    Input:
      signal:
        shape: [1, -1]
        dtype: tensor(float)
        transform: None
      feature:
        shape: [18, -1]
        dtype: tensor(float)
        transform: audonnx.core.function.Function(feature_addition)
    Output:
      hidden:
        shape: [8]
        dtype: tensor(float)
        labels: [hidden-0, hidden-1, hidden-2, (...), hidden-5, hidden-6, hidden-7]
      gender:
        shape: [2]
        dtype: tensor(float)
        labels: [female, male]
      confidence:
        shape: [1]
        dtype: tensor(float)
        labels: [confidence]

When calling this model,
the keys of the input dictionary
need to match the signature of our function.
In this case, we need to pass the ``my_input``
input.

.. code-block:: pycon

    >>> onnx_model_10(
    ...     {"signal": signal, "my_input": y},
    ...     sampling_rate,
    ... )
    {'hidden': array([ 7.6037818e-01, -1.2064241e-02,  3.2603091e-01, -4.6754807e-01,
           -4.1000482e+02,  7.2107361e+01, -5.6038922e+02,  1.9108322e+01],
          dtype=float32), 'gender': array([307.07407 ,  22.489958], dtype=float32), 'confidence': array([-92.46997], dtype=float32)}

We can optionally set keyword arguments with default values,
in this case ``offset``.

.. code-block:: pycon

    >>> onnx_model_10(
    ...     {"signal": signal, "my_input": y, "offset": 1},
    ...     sampling_rate,
    ... )
    {'hidden': array([ 7.6037818e-01, -1.2064241e-02,  3.2603091e-01, -4.6754807e-01,
           -4.1040109e+02,  7.2496727e+01, -5.6054236e+02,  1.9019484e+01],
          dtype=float32), 'gender': array([307.29623 ,  22.639301], dtype=float32), 'confidence': array([-92.62186], dtype=float32)}


Run on the GPU
--------------

To run a model on the GPU install ``onnxruntime-gpu``.
Note that the version has to fit the CUDA installation.
We can get the information from this table_.

Then select CUDA device when loading the model:

.. skip: next

.. code-block:: python

    import os
    import audonnx

    model = audonnx.load(..., device='cuda:2')

With
``onnxruntime-gpu<1.8``
it is not possible to directly specify an ID.
In that case do:

.. skip: next

.. code-block:: python

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    model = audonnx.load(..., device='cuda')


.. _audformat: https://audeering.github.io/audformat/
.. _audinterface: https://audeering.github.io/audinterface/
.. _audobject: https://audeering.github.io/audobject/
.. _librosa: https://librosa.org/doc/main/index.html
.. _MobilenetV2 example: https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md
.. _ONNX: https://onnx.ai/
.. _OpenSMILE: https://github.com/audeering/opensmile-python
.. _table: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements
.. _Torch: https://pytorch.org/
