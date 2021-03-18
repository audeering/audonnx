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


Export
------

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
        dynamic_axes={'input': {3: "time"}},
        opset_version=12,
    )

We can now load the exported model
and *voilà* we get the same output (well, almost :).

.. jupyter-execute::

    import audonnx

    onnx_model = audonnx.Model(
        onnx_path,
        labels=predictor.labels,
        transform=predictor.transform,
    )
    onnx_model.forward(signal, sampling_rate)

Or we directly output the majority class.

.. jupyter-execute::

    onnx_model.predict(signal, sampling_rate)


Interface
---------

:class:`onnx.Model` does not come with a fancy interface itself,
but we can use audinterface_ to create one.

.. jupyter-execute::

    import audinterface

    interface = audinterface.Process(
        process_func=onnx_model.predict,
    )
    interface.process_file(file)

Or if we are interested in the raw predictions.

.. jupyter-execute::

    import pandas as pd

    interface = audinterface.Feature(
        feature_names=onnx_model.labels,
        process_func=onnx_model.forward,
    )
    interface.process_index(index)


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
    onnx_model_2.predict(signal, sampling_rate)


Quantize
--------

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

    onnx_model_3 = audonnx.load(onnx_root, name='model_quant.onnx')
    onnx_model_3.forward(signal, sampling_rate)


.. _audformat: https://audeering.github.io/audformat/
.. _audinterface: http://tools.pp.audeering.com/audinterface/
.. _audobject: http://tools.pp.audeering.com/audobject/
.. _audpann: http://tools.pp.audeering.com/audpann/
.. _PyTorch: https://pytorch.org/
.. _ONNX: https://onnx.ai/
