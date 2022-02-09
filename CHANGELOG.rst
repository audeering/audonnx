Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_,
and this project adheres to `Semantic Versioning`_.


Version 0.5.0 (2022-02-09)
--------------------------

* Added: argument ``device``
* Changed: use CPU by default
* Changed: require ``onnxruntime>=1.8.0``
* Removed:
  ``Model.forward()``
  ``Model.labels``,
  ``Model.predict()``,
  ``Model.transform``


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
  ``InputNode``,
  ``Model.__call__()``,
  ``Model.inputs``,
  ``Model.outputs``,
  ``OutputNode``
* Changed: reshape input to expected shape
* Changed: do not depend on existing models in tests and documentation
* Changed: support multiple input nodes
* Changed: make ``Model`` serializable
* Deprecated:
  ``Model.forward()``
  ``Model.labels``,
  ``Model.predict()``,
  ``Model.transform``
* Removed:
  ``Model.input_node``,
  ``Model.input_shape``,
  ``Model.input_type``,
  ``Model.output_nodes``,
  ``Model.output_shape``,
  ``Model.output_type``,


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
