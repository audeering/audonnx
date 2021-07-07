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
In the following we will use it to export a model
trained with PyTorch_ and write an interface for it.

But first, let's create some test input -
a file path, a signal array and an index in audformat_.

.. jupyter-execute::

    import audiofile


    file = './docs/_static/test.wav'
    signal, sampling_rate = audiofile.read(file)
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


Export to ONNX
--------------

The model we want to export to ONNX_
was trained with audpann_.
Let's load it and apply it to our test signal.

.. jupyter-execute::

    import audpann


    uid = '74c0af32-6acf-9f31-fe2a-3c05190a88f4'
    predictor = audpann.Predictor(
        uid=uid,
        get_logits=True,
    )
    predictor(signal, sampling_rate)

To export it, we pass the model and specify an output path.

.. jupyter-execute::

    import audeer
    import os
    import torch


    onnx_root = audeer.mkdir('onnx')
    onnx_path = os.path.join(onnx_root, 'model.onnx')

    dummy_input = torch.randn(1, 1, 64, 500)
    torch.onnx.export(
        predictor.model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {3: 'time'}},
        opset_version=12,
    )

``predictor.model`` is the implementation of the model
as a :class:`torch.nn.Module` object.

We can now load the exported model
and *voilà* we get the same output (well, almost :).

.. jupyter-execute::

    import audonnx


    onnx_model = audonnx.Model(
        onnx_path,
        labels=predictor.labels,
        transform=predictor.transform,
    )
    onnx_model(signal, sampling_rate)


Create an interface
-------------------

:class:`onnx.Model` does not come with a fancy interface itself,
but we can use audinterface_ to create one.

.. jupyter-execute::

    import audinterface


    interface = audinterface.Feature(
        feature_names=onnx_model.labels['output'],
        process_func=onnx_model,
    )
    interface.process_index(index)

Or if we are only interested in the majority class.

.. jupyter-execute::

    interface.process_index(index).idxmax(axis=1)


Save and load
-------------

The model we are using works on spectrograms.
Therefore, we have passed a callable object
to the ``transform`` argument.
If given, it is called to do the conversion from
raw audio to the desired representation.
Obviously, it is not a bad idea to store
the transformation with the model.
Since the callable is serializable class
from audobject_, we can achieve this easily.

.. jupyter-execute::

    transform_path = os.path.join(onnx_root, 'transform.yaml')
    onnx_model.transform.to_yaml(transform_path)

This stores the following yaml representation:

.. jupyter-execute::
    :hide-code:

    print(onnx_model.transform.to_yaml_s(include_version=True))

In addition, we also dump the labels to a yaml file.

.. jupyter-execute::

    import oyaml as yaml


    with open(os.path.join(onnx_root, 'labels.yaml'), 'w') as fp:
        yaml.dump(onnx_model.labels, fp)

Next time we want to load the model we can simply do:

.. jupyter-execute::

    onnx_model_2 = audonnx.load(onnx_root)
    onnx_model_2(signal, sampling_rate)


Quantize weights
----------------

To reduce the memory print of a model,
we can quantize it.
For instance, we can store model weights as 8 bit integers.

.. jupyter-execute::

    import onnxruntime.quantization


    quant_path = os.path.join(onnx_root, 'model_quant.onnx')
    quant_model = onnxruntime.quantization.quantize_dynamic(
        onnx_path,
        quant_path,
        weight_type=onnxruntime.quantization.QuantType.QUInt8,
    )

The converted model is significantly smaller.

.. jupyter-execute::

    f'{os.stat(quant_path).st_size} << {os.stat(onnx_path).st_size}'

The output of the quantized model will be slightly different, though.

.. jupyter-execute::

    onnx_model_3 = audonnx.load(onnx_root, model_file='model_quant.onnx')
    onnx_model_3(signal, sampling_rate)


Multi-head models
-----------------

The model we used so far has a single output node,
now let us switch to one with multiple output nodes,
a so called multi-head model.

.. jupyter-execute::

    import audmodel


    uid = 'c3a709c9-0b58-48d1-7217-0aa3ea485d2e'
    root = audmodel.load(uid)
    onnx_model_multi = audonnx.load(root)
    onnx_model_multi.output_names

For such a model,
we get a prediction for every output node:

.. jupyter-execute::

    onnx_model_multi(signal, sampling_rate)

We can also get predictions
for specific node(s):

.. jupyter-execute::

    onnx_model_multi(
        signal,
        sampling_rate,
        output_names=['client-gender'],
    )

Or:

.. jupyter-execute::

    onnx_model_multi(
        signal,
        sampling_rate,
        output_names='client-gender',
    )

And we can create an an interface for it, too:

.. jupyter-execute::

    interface = audinterface.Feature(
        feature_names=onnx_model_multi.labels['client-gender'],
        process_func=onnx_model_multi,
        output_names='client-gender',
    )
    interface.process_signal(signal, sampling_rate)

Or if we want to concatenate the predictions of all nodes:

.. jupyter-execute::

    import numpy as np


    interface = audinterface.Feature(
        feature_names=audeer.flatten_list(
            list(onnx_model_multi.labels.values())
        ),
        process_func=lambda x, sr: np.concatenate(
            list(onnx_model_multi(x, sr).values()),
            axis=1,
        ),
    )
    interface.process_signal(signal, sampling_rate)


Run on the GPU
--------------

If you want to run your model
on the GPU,
you have to install
``onnxruntime-gpu``.
Make sure you install the version
that fits your CUDA installation.
You can get the information
from this table_.

Note that it will pick the
first GPU device it finds.
To select a specific CUDA device,
you can do:

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
.. _PyTorch: https://pytorch.org/
.. _ONNX: https://onnx.ai/
.. _table: https://www.onnxruntime.ai/docs/reference/execution-providers/CUDA-ExecutionProvider.html#requirements
