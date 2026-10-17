Runnable quickstarts
====================

Each example is a complete script using a small, local dataset and a fixed
random seed. From a repository checkout, run::

    python examples/quickstart_kmodes.py
    python examples/quickstart_kprototypes.py

Categorical data with KModes
----------------------------

Every column is treated as categorical, including columns containing integers.
Use ``fit_predict`` to fit the model and obtain a cluster index for each row,
or ``fit`` followed by ``predict`` for new rows with the same feature order.
Cluster numbers are arbitrary identifiers, not ranks or class names.

.. literalinclude:: ../examples/quickstart_kmodes.py
   :language: python

Mixed data with KPrototypes
---------------------------

Use an object array to keep numerical values and category strings together.
``categorical=[1]`` selects the second column using zero-based indexing; the
other columns must contain numerical values. Pass the same categorical indices
to fitting and prediction. This is a method argument, not a constructor
parameter.

.. literalinclude:: ../examples/quickstart_kprototypes.py
   :language: python

The returned centroids contain numerical columns first, followed by categorical
columns, even if the input columns were interleaved. A mixed centroid array may
coerce numbers to strings; convert its numerical slice to floats when needed.

Before using your own data
--------------------------

Fill missing values and keep each column's types consistent. Sparse input is
not supported. Use KModes for entirely categorical input; for entirely numerical
input, use an algorithm such as scikit-learn's KMeans. For KPrototypes, consider
scaling numerical features so a large unit or range does not dominate the cost.
See :doc:`parameters` for initialization and weighting choices.
