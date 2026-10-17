kmodes documentation
====================

``kmodes`` provides two clustering estimators with a scikit-learn-style API:

* :class:`~kmodes.kmodes.KModes` groups rows containing categorical features.
  Its default cost counts the categories that differ from each cluster's mode.
* :class:`~kmodes.kprototypes.KPrototypes` groups rows containing both numerical
  and categorical features. Its default cost combines squared numerical
  distances with categorical mismatches, weighted by ``gamma``.

Start with the quickstarts, then consult the parameter guide for choices such
as ``init='Huang'`` and ``verbose=1``. The API reference is generated from the
estimator docstrings and lists every constructor parameter and public fitting
and prediction method.

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   parameters
   api
   contributing

Source code, issue reports, and the algorithm references are available in the
`GitHub repository <https://github.com/nicodv/kmodes>`_.
