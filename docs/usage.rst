Usage
=====

.. jupyter-execute::
    :hide-code:
    :hide-output:

    import pandas as pd


    def series_to_html(self):
        df = self.to_frame()
        df.columns = ['']
        return df._repr_html_()
    setattr(pd.Series, '_repr_html_', series_to_html)


    def index_to_html(self):
        return self.to_frame(index=False)._repr_html_()
    setattr(pd.Index, '_repr_html_', index_to_html)


:mod:`audonnx` offers a simple interface
to load and use models in ONNX_ format.
Models with single or multiple input and output nodes are supported.

We begin with creating some test input -
a file path, a signal array and an index in audformat_.

.. jupyter-execute::

    import audiofile


    file = './docs/_static/test.wav'
    signal, sampling_rate = audiofile.read(
        file,
        always_2d=True,
    )
    index = pd.MultiIndex.from_arrays(
        [
            [file, file],
            pd.to_timedelta(['0s', '3s']),
            pd.to_timedelta(['3s', '5s']),
        ],
        names=['file', 'start', 'end'],
    )

.. jupyter-execute::
    :hide-code:

    import IPython
    IPython.display.Audio(file)


Torch model
-----------

Create Torch_ model with a single input and output node.

.. jupyter-execute::

    import torch


    class TorchModelSingle(torch.nn.Module):

        def __init__(
            self,
        ):
            super().__init__()
            self.hidden = torch.nn.Linear(8, 8)
            self.out = torch.nn.Linear(8, 2)

        def forward(self, x: torch.Tensor):
            y = self.hidden(x.mean(dim=-1))
            y = self.out(y)
            return y.squeeze()


    torch_model = TorchModelSingle()

Create feature extraction that converts raw audio signal to a spectrogram.

.. jupyter-execute::

    import audsp


    spectrogram = audsp.Spectrogram(
        16000,
        0.02,
        0.01,
        center=False,
        reflect=False,
        audspec=audsp.AuditorySpectrum(
            num_bands=8,
            scale=audsp.define.AuditorySpectrumScale.MEL,
        ),
    )

Calculate spectrogram and run Torch_ model.

.. jupyter-execute::

    y = spectrogram(signal, sampling_rate)
    with torch.no_grad():
        z = torch_model(torch.from_numpy(y))
    z


Export model
------------

To export the model to ONNX_ format,
we pass some dummy input,
which allows the function to figure out
correct input and output shapes.
Since the number of frames in the spectrogram
varies with the length of the input signal,
we tell the function that the last dimension
of the input has a dynamic size.
And we assign meaningful names to the nodes.

.. jupyter-execute::

    import audeer
    import os


    onnx_root = audeer.mkdir('onnx')
    onnx_model_path = os.path.join(onnx_root, 'model.onnx')

    dummy_input = torch.randn(spectrogram.shape(1.0))
    torch.onnx.export(
        torch_model,
        dummy_input,
        onnx_model_path,
        input_names=['spectrogram'],  # assign custom name to input node
        output_names=['gender'],      # assign custom name to output node
        dynamic_axes={'spectrogram': {1: 'time'}},  # dynamic size
        opset_version=12,
    )

From the exported model file
we now create an object of :class:`audonnx.Model`.
We pass the feature extractor,
so that the model can automatically convert the
input signal to the desired representation.
And we assign labels to the dimensions of the output node.
Printing the model provides a summary of
the input and output nodes.

.. jupyter-execute::

    import audonnx


    onnx_model = audonnx.Model(
        onnx_model_path,
        labels=['female', 'male'],
        transform=spectrogram,
    )
    onnx_model

Get information for individual nodes.

.. jupyter-execute::

    onnx_model.inputs['spectrogram']

.. jupyter-execute::

    print(onnx_model.inputs['spectrogram'].transform)

.. jupyter-execute::

    onnx_model.outputs['gender']

.. jupyter-execute::

    onnx_model.outputs['gender'].labels

Check that the exported model gives the expected output.

.. jupyter-execute::

    onnx_model(signal, sampling_rate)

Create interface
----------------

:class:`onnx.Model` does not come with a fancy interface itself,
but we can use audinterface_ to create one.

.. jupyter-execute::

    import numpy as np
    import audinterface


    interface = audinterface.Feature(
        # use labels of output node as feature names
        feature_names=onnx_model.outputs['gender'].labels,
        # TODO: simplify to 'process_func=onnx_model'
        # when https://github.com/audeering/audinterface/pull/22 is merged
        process_func=lambda x, sr: np.atleast_2d(onnx_model(x, sr)),
    )
    interface.process_index(index)

Or if we are only interested in the majority class.

.. jupyter-execute::

    interface.process_index(index).idxmax(axis=1)


Save and load
-------------

Save the model to a YAML file.

.. jupyter-execute::

    onnx_meta_path = os.path.join(onnx_root, 'model.yaml')
    onnx_model.to_yaml(onnx_meta_path)

.. jupyter-execute::
    :hide-code:

    import oyaml as yaml


    with open(onnx_meta_path, 'r') as fp:
        d = yaml.load(fp, Loader=yaml.Loader)
    print(yaml.dump(d))

Load the model from a YAML file.

.. jupyter-execute::

    onnx_model_2 = audonnx.Model.from_yaml(onnx_meta_path)
    onnx_model_2(signal, sampling_rate)

Or shorter:

.. jupyter-execute::

    onnx_model_3 = audonnx.load(onnx_root)
    onnx_model_3(signal, sampling_rate)


Quantize weights
----------------

To reduce the memory print of a model,
we can quantize it.
For instance, we can store model weights as 8 bit integers.

.. jupyter-execute::

    import onnxruntime.quantization


    onnx_quant_path = os.path.join(onnx_root, 'model_quant.onnx')
    onnxruntime.quantization.quantize_dynamic(
        onnx_model_path,
        onnx_quant_path,
        weight_type=onnxruntime.quantization.QuantType.QUInt8,
    )

The output of the quantized model differs slightly.

.. jupyter-execute::

    onnx_model_4 = audonnx.Model(
        onnx_quant_path,
        labels=['female', 'male'],
        transform=spectrogram,
    )
    onnx_model_4(signal, sampling_rate)


Model with multiple nodes
-------------------------

Define a model that takes as input the
raw audio in addition to the spectrogram
and provides two more output nodes -
the output from the hidden layer and a confidence value.

.. jupyter-execute::

    class TorchModelMulti(torch.nn.Module):

        def __init__(
            self,
        ):

            super().__init__()

            self.hidden_left = torch.nn.Linear(1, 4)
            self.hidden_right = torch.nn.Linear(8, 4)
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

Export the new model to ONNX_ format and load it.
Note that we do not assign labels to all output nodes.
In that case, they are automatically created
from the name of the output node.
And since the first node expects the raw audio signal,
we do not set a transform for it.

.. jupyter-execute::

    onnx_multi_path = os.path.join(onnx_root, 'model.onnx')

    torch.onnx.export(
        TorchModelMulti(),
        (
            torch.randn(signal.shape),
            torch.randn(spectrogram.shape(1.0)),
        ),
        onnx_multi_path,
        input_names=['signal', 'spectrogram'],
        output_names=['hidden', 'gender', 'confidence'],
        dynamic_axes={
            'signal': {1: 'time'},
            'spectrogram': {1: 'time'},
        },
        opset_version=12,
    )

    onnx_model_5 = audonnx.Model(
        onnx_multi_path,
        labels={
            'gender': ['female', 'male']
        },
        transform={
            'spectrogram': spectrogram,
        },
    )
    onnx_model_5

By default,
returns a dictionary with output for every node.

.. jupyter-execute::

    onnx_model_5(signal, sampling_rate)

To request specific nodes.

.. jupyter-execute::

    onnx_model_5(
        signal,
        sampling_rate,
        output_names=['gender', 'confidence'],
    )

Or a single node:

.. jupyter-execute::

    onnx_model_5(
        signal,
        sampling_rate,
        output_names='gender',
    )

Create interface and process a file.

.. jupyter-execute::

    def process_func(x, sr, output_names):
        y = onnx_model_5(x, sr, output_names=output_names)
        return np.atleast_2d(y)

    interface = audinterface.Feature(
        feature_names=onnx_model_5.outputs['gender'].labels,
        # TODO: simplify to 'process_func=onnx_model'
        # when https://github.com/audeering/audinterface/pull/22 is merged
        process_func=process_func,
        output_names='gender',
    )
    interface.process_file(file)


Run on the GPU
--------------

To run a model on the GPU install ``onnxruntime-gpu``.
Note that the version has to fit the CUDA installation.
We can get the information from this table_.

By default,
it uses the first GPU device it finds.
To select a specific CUDA device,
we can do:

.. code-block:: python

    import os
    import audonnx

    os.environ['CUDA_VISIBLE_DEVICES']='2'  # make cuda:2 default device
    model = audonnx.load(...)               # load model
    model(...)                              # run on cuda:2


.. _audformat: https://audeering.github.io/audformat/
.. _audinterface: http://tools.pp.audeering.com/audinterface/
.. _audobject: http://tools.pp.audeering.com/audobject/
.. _audpann: http://tools.pp.audeering.com/audpann/
.. _Torch: https://pytorch.org/
.. _ONNX: https://onnx.ai/
.. _table: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements
