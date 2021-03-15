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


Export
------


Load
----

.. jupyter-execute::

    import audonnx
    import audmodel

    uid = 'e157969a-9898-6c62-f0b3-2d7b0f4ad9c7'
    root = audmodel.load(uid)
    model = audonnx.load(root)


.. jupyter-execute::

    import audiofile

    file = './docs/_static/test.wav'
    signal, sampling_rate = audiofile.read(file)
    model.forward(signal, sampling_rate)


.. jupyter-execute::

    model.predict(signal, sampling_rate)


Interface
---------

.. jupyter-execute::

    import audinterface

    interface = audinterface.Process(
        process_func=model.predict,
    )
    interface.process_file(file)


.. jupyter-execute::

    import pandas as pd

    interface = audinterface.Feature(
        feature_names=model.labels,
        process_func=model.forward,
    )
    index = pd.MultiIndex.from_arrays(
        [
            [file, file],
            pd.to_timedelta(['0s', '3s']),
            pd.to_timedelta(['3s', '5s']),
        ],
        names=['file', 'start', 'end'],
    )
    interface.process_index(index)
