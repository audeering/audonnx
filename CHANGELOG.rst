Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_,
and this project adheres to `Semantic Versioning`_.


Version 0.7.0 (2023-12-18)
--------------------------

* Added: ``audonnx.device_to_providers()``
  to convert device names
  to ONNX runtime providers
* Added: ``num_workers`` argument
  to ``audonnx.Model``,
  ``audonnx.load()``,
  and ``audonnx.testing.create_model()``.
  Its default value is ``1``,
  whereas before ONNX runtime
  was selecting all available workers
* Added: ``session_options`` argument
  to ``audonnx.Model``,
  ``audonnx.load()``,
  and ``audonnx.testing.create_model()``
  to provide ONNX runtime options


Version 0.6.5 (2023-11-10)
--------------------------

* Fixed: publishing of documentation as Github pages


Version 0.6.4 (2023-11-10)
--------------------------

* Added: pre-processing step in the documentation
  on how to quantize an ONNX model
* Added: support for Python 3.11
* Removed: support for Python 3.7


Version 0.6.3 (2023-01-03)
--------------------------

* Added: support for Python 3.10
* Changed: split API documentation into sub-pages
  for each function


Version 0.6.2 (2022-07-14)
--------------------------

* Changed: require ``audobject>=0.7.2``


Version 0.6.1 (2022-06-27)
--------------------------

* Fixed: added missing ``onnx`` dependency to ``setup.cfg``


Version 0.6.0 (2022-06-23)
--------------------------

* Added: ``audonnx.Function``
* Added: ``audonnx.Model.labels()``
* Added: arguments
  ``concat``,
  ``outputs``,
  ``squeeze``
  to ``audonnx.Model.__call__()``
* Added: tests on Windows
* Added: ``audonnx.testing`` module
* Changed: optionally init ``audonnx.Model`` from proto object instead of ONNX file
* Changed: dynamic axis can be specified as ``None`` in ONNX graph
* Changed: support output nodes where last dimension is dynamic
* Deprecated: argument ``output_names`` of ``audonnx.Model.__call__()``


Version 0.5.2 (2022-04-01)
--------------------------

* Fixed: always replace dynamic axis names with ``-1``
  in input and output shapes of model nodes


Version 0.5.1 (2022-03-29)
--------------------------

* Added: argument ``auto_install`` to ``audonnx.load()``


Version 0.5.0 (2022-02-09)
--------------------------

* Added: argument ``device``
* Changed: use CPU by default
* Changed: require ``onnxruntime>=1.8.0``
* Removed:
  ``audonnx.Model.forward()``
  ``audonnx.Model.labels``,
  ``audonnx.Model.predict()``,
  ``audonnx.Model.transform``


Version 0.4.3 (2022-01-10)
--------------------------

* Fixed: publication of docs failed


Version 0.4.2 (2022-01-10)
--------------------------

* Fixed: publication of docs failed


Version 0.4.1 (2022-01-10)
--------------------------

* Fixed: author email address in Python package metadata


Version 0.4.0 (2022-01-10)
--------------------------

* Added: first public release
* Changed: switch to MIT license
* Changed: move repo to Github
* Fixed: remove ``audsp`` from docstring example
  as we no longer depend on it


Version 0.3.3 (2021-12-30)
--------------------------

* Changed: use Python 3.8 as default


Version 0.3.2 (2021-11-01)
--------------------------

* Changed: use ``audobject`` >=0.6.1


Version 0.3.1 (2021-10-05)
--------------------------

* Fixed: ``audonnx.load()`` try to load model from ONNX if YAML does not exist


Version 0.3.0 (2021-10-01)
--------------------------

* Changed: audobject >=0.5.0
* Changed: force ``.yaml`` extension when model is saved
* Fixed: if possible load model from ``.yaml`` in ``audonnx.load()``


Version 0.2.2 (2021-09-23)
--------------------------

* Fixed: link to ONNX runtime CUDA mapping table


Version 0.2.1 (2021-09-15)
--------------------------

* Fixed: loading of old models that contain a ``model.yaml`` file


Version 0.2.0 (2021-07-20)
--------------------------

* Added:
  ``audonnx.InputNode``,
  ``audonnx.Model.__call__()``,
  ``audonnx.Model.inputs``,
  ``audonnx.Model.outputs``,
  ``audonnx.OutputNode``
* Changed: reshape input to expected shape
* Changed: do not depend on existing models in tests and documentation
* Changed: support multiple input nodes
* Changed: make ``audonnx.Model`` serializable
* Deprecated:
  ``audonnx.Model.forward()``
  ``audonnx.Model.labels``,
  ``audonnx.Model.predict()``,
  ``audonnx.Model.transform``
* Removed:
  ``audonnx.Model.input_node``,
  ``audonnx.Model.input_shape``,
  ``audonnx.Model.input_type``,
  ``audonnx.Model.output_nodes``,
  ``audonnx.Model.output_shape``,
  ``audonnx.Model.output_type``,


Version 0.1.1 (2021-03-31)
--------------------------

* Changed: update documentation how to select specific GPU device


Version 0.1.0 (2021-03-25)
--------------------------

* Added: initial release


.. _Keep a Changelog:
    https://keepachangelog.com/en/1.0.0/
.. _Semantic Versioning:
    https://semver.org/spec/v2.0.0.html
