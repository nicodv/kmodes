Choosing parameters
===================

KModes has eight constructor parameters; KPrototypes has ten, adding
``num_dissim`` and ``gamma``. You can supply any subset by name; omitted
parameters use their defaults. For example::

    from kmodes.kmodes import KModes
    from kmodes.kprototypes import KPrototypes

    km = KModes(n_clusters=4, init='Huang', n_init=5, verbose=1)
    kp = KPrototypes(n_clusters=4, init='Cao', n_init=5, verbose=0)

These are configuration examples; fitting requires at least four data rows.
See :doc:`quickstart` for complete runnable examples and :doc:`api` for all
parameter types, defaults, fitted attributes, and method arguments.

Initialization: ``init``
------------------------

The default is ``'Cao'`` for both estimators. The three named methods apply to
the categorical features:

.. list-table:: Named initialization methods
   :header-rows: 1
   :widths: 15 45 40

   * - Value
     - Categorical initialization
     - Repeated runs
   * - ``'Huang'``
     - Samples category frequencies and selects nearby observations.
     - Randomized; try several ``n_init`` runs.
   * - ``'Cao'``
     - Selects initial modes using category density and separation.
     - KModes sets ``n_init=1`` for the canonical spelling ``'Cao'``.
   * - ``'random'``
     - Samples rows for the initial categorical centroids, with replacement.
     - Randomized; try several ``n_init`` runs.

Named methods are case-insensitive, but use the spellings above: the KModes
constructor's ``n_init=1`` optimization specifically checks ``'Cao'``.
``'k-means++'`` and ``'auto'`` are not supported initialization values.

For KPrototypes, all three named methods initialize the numerical centroids
by drawing from normal distributions with the numerical features' means and
standard deviations. Thus ``'Cao'`` still has randomness and multiple runs can
help. If initialization produces empty clusters, KPrototypes retries and may
fall back to random categorical initialization before raising an error.

Custom centroids
~~~~~~~~~~~~~~~~

KModes also accepts an ndarray of shape ``(n_clusters, n_features)``.
KPrototypes accepts a **list of two ndarrays**:
``[numerical_centroids, categorical_centroids]``, with shapes
``(n_clusters, n_numerical_features)`` and
``(n_clusters, n_categorical_features)``. Each block follows the order of its
columns in the input. Both estimators set ``n_init=1`` for these custom inputs.

Categorical centroid entries must be **encoded integers**, not the original
category strings. The package maps each column's sorted unique values to
``0, 1, ...``. For example, ``'blue'`` maps to 0 and ``'red'`` to 1 below.
Encoding must match the data passed to ``fit``; a mapping from a different
dataset can silently select the wrong categories.

.. doctest::

    >>> import numpy as np
    >>> from kmodes.kmodes import KModes
    >>> from kmodes.kprototypes import KPrototypes
    >>> colors = np.array([['blue', 'small'], ['blue', 'medium'],
    ...                    ['red', 'large'], ['red', 'medium']])
    >>> # Second column encoding: large=0, medium=1, small=2.
    >>> initial_modes = np.array([[0, 2], [1, 0]])
    >>> km = KModes(n_clusters=2, init=initial_modes, n_init=5)
    >>> km.n_init
    1
    >>> labels = km.fit_predict(colors)
    >>> bool(labels[0] == labels[1] and labels[0] != labels[2])
    True
    >>> mixed = np.array([[1., 'blue'], [2., 'blue'],
    ...                   [9., 'red'], [10., 'red']], dtype=object)
    >>> initial = [np.array([[1.5], [9.5]]), np.array([[0], [1]])]
    >>> kp = KPrototypes(n_clusters=2, init=initial, n_init=5, gamma=1.)
    >>> kp.n_init
    1
    >>> labels = kp.fit_predict(mixed, categorical=[1])
    >>> bool(labels[0] == labels[1] and labels[0] != labels[2])
    True

Progress output: ``verbose``
----------------------------

``verbose=0`` (the default) is silent. Use ``verbose=1`` to print initialization
messages, iteration progress, moves, and costs. The implementation checks
whether ``verbose`` is truthy: other nonzero integers enable the same output;
``2`` or ``3`` do not select additional detail levels. ``False`` and ``True``
also work as off and on. This controls output, not the clustering objective.

Other common choices
--------------------

* ``n_clusters=8``: a positive integer no greater than the number of rows.
  Choose it for your application. If there are fewer distinct rows than
  requested clusters, the algorithm returns one centroid per distinct row.
* ``max_iter=100``: a nonnegative integer limit per run. ``0`` keeps the
  initialization and skips iterative refinement. Convergence can stop a run
  before the limit. When refinement is skipped, ``labels_`` is None; use
  ``predict`` or ``fit_predict`` to obtain labels.
* ``n_init=10``: a positive integer number of runs. The result with the lowest
  cost is retained; the initialization exceptions above can reduce this to 1.
* ``random_state=None``: use an integer seed (for example, ``42``) to make
  randomized runs reproducible for the same input and environment. You can also
  pass a NumPy ``RandomState`` instance; ``None`` uses NumPy's global state.
* ``n_jobs=1``: run initializations serially. Positive integers select worker
  counts, ``-1`` uses all CPUs, and ``-2`` uses all but one. ``0`` is invalid.
  Parallelism helps only when there are multiple runs and enough work per run.

For KPrototypes, ``gamma=None`` estimates a weight as half the mean population
standard deviation of the numerical columns. Supply a nonnegative float to
choose the weight yourself. The default per-row cost is::

    sum((numerical_row - numerical_centroid) ** 2)
    + gamma * number_of_categorical_mismatches

Larger ``gamma`` gives categories more influence; ``gamma=0`` removes their
contribution to the cost. The fitted ``model.gamma`` contains the estimated
value. Numerical scaling changes this balance.

Custom dissimilarity functions
------------------------------

``cat_dissim`` defaults to
:func:`~kmodes.util.dissim.matching_dissim`; KPrototypes also has
``num_dissim``, defaulting to
:func:`~kmodes.util.dissim.euclidean_dissim` (squared Euclidean distance).
Pass a callable, not a metric name string. It receives a centroid array and
one row and must return one distance per centroid. Categorical callables
receive encoded categories and must also accept keyword arguments ``X`` and
``membship`` (or ``**kwargs``).

Fitting arguments are separate
------------------------------

``X``, ``categorical``, and ``sample_weight`` belong to fitting methods, not
the constructor. In KPrototypes, select categorical columns with a list or
tuple of zero-based indices, for example ``categorical=[0, 2]``. A single
integer is supported for nonzero indices; use ``[0]`` for the first column
because the current implementation treats bare ``0`` as no selection.
All remaining columns are numerical. Pass the same indices to ``predict``.

``sample_weight=None`` gives every row weight 1. Otherwise supply a sequence
of nonnegative Python integers or floats, one per row, with at least
``n_clusters`` positive weights. Weights influence centroid updates and cost.
The ``y`` argument is ignored; these are unsupervised estimators.
