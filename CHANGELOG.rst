Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_,
and this project adheres to `Semantic Versioning`_.


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
