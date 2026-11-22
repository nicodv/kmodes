.. image:: https://img.shields.io/pypi/v/kmodes.svg
    :target: https://pypi.python.org/pypi/kmodes/
    :alt: Version
.. image:: https://anaconda.org/conda-forge/kmodes/badges/version.svg
    :target: https://anaconda.org/conda-forge/kmodes
    :alt: Conda forge page
.. image:: https://github.com/nicodv/kmodes/actions/workflows/python-package.yml/badge.svg?branch=master
    :target: https://github.com/nicodv/kmodes/actions/workflows/python-package.yml
    :alt: Build status
.. image:: https://coveralls.io/repos/nicodv/kmodes/badge.svg
    :target: https://coveralls.io/r/nicodv/kmodes
    :alt: Test coverage
.. image:: https://api.codacy.com/project/badge/Grade/cb19f1f1093a44fa845ebfdaf76975f6
   :alt: Codacy
   :target: https://app.codacy.com/app/nicodv/kmodes?utm_source=github.com&utm_medium=referral&utm_content=nicodv/kmodes&utm_campaign=Badge_Grade_Dashboard
.. image:: https://img.shields.io/pypi/dm/kmodes.svg
    :target: https://pypi.python.org/pypi/kmodes/
    :alt: Monthly downloads
.. image:: https://img.shields.io/pypi/pyversions/kmodes.svg
    :target: https://pypi.python.org/pypi/kmodes/
    :alt: Supported Python versions
.. image:: https://img.shields.io/pypi/l/kmodes.svg
    :target: https://github.com/nicodv/kmodes/blob/master/LICENSE
    :alt: License

kmodes
======

Description
-----------

Python implementations of the k-modes and k-prototypes clustering
algorithms. Relies on numpy for a lot of the heavy lifting.

k-modes is used for clustering categorical variables. It defines clusters
based on the number of matching categories between data points. (This is
in contrast to the more well-known k-means algorithm, which clusters
numerical data based on Euclidean distance.) The k-prototypes algorithm
combines k-modes and k-means and is able to cluster mixed numerical /
categorical data.

Implemented are:

- k-modes [HUANG97]_ [HUANG98]_
- k-modes with initialization based on density [CAO09]_
- k-prototypes [HUANG97]_

The code is modeled after the clustering algorithms in :code:`scikit-learn`
and has the same familiar interface.

I would love to have more people play around with this and give me
feedback on my implementation. If you come across any issues in running or
installing kmodes,
`please submit a bug report <https://github.com/nicodv/kmodes/issues>`_.

Enjoy!

Installation
------------

`kmodes` can be installed using `pip`:

.. code:: bash

    pip install kmodes

To upgrade to the latest version (recommended), run it like this:

.. code:: bash

    pip install --upgrade kmodes

`kmodes` can also conveniently be installed with `conda` from the `conda-forge` channel:

.. code:: bash

    conda install -c conda-forge kmodes

Alternatively, you can build the latest development version from source:

.. code:: bash

    git clone https://github.com/nicodv/kmodes.git
    cd kmodes
    python setup.py install

Usage
-----
.. code:: python

    import numpy as np
    from kmodes.kmodes import KModes

    # random categorical data
    data = np.random.choice(20, (100, 10))

    km = KModes(n_clusters=4, init='Huang', n_init=5, verbose=1)

    clusters = km.fit_predict(data)

    # Print the cluster centroids
    print(km.cluster_centroids_)

The examples directory showcases simple use cases of both k-modes
('soybean.py') and k-prototypes ('stocks.py').

Custom dissimilarity functions
------------------------------

Pass a callable as :code:`cat_dissim` to customize categorical dissimilarity
in :code:`KModes` or :code:`KPrototypes`. The callable receives a 2D array
:code:`a` and a 1D point :code:`b`, and must return a 1D array containing one
cost for each row of :code:`a`. Smaller costs mean more similar points.
The rows of :code:`a` may be observations during initialization or cluster
centroids during fitting and prediction; do not assume a fixed number of rows.

Categorical values are encoded as integers before the callable is invoked,
so it should not depend on the original category labels. It must also accept
the keyword arguments :code:`X` and :code:`membship`, which supply the encoded
data and cluster membership during fitting. They may be absent or
:code:`None` in other calls. Use :code:`**kwargs` if they are not needed.
For example, this gives mismatches in the first categorical column twice the
weight of mismatches in the second:

.. code:: python

    import numpy as np
    from kmodes.kmodes import KModes
    from kmodes.kprototypes import KPrototypes

    def weighted_matching(a, b, **kwargs):
        return np.sum((a != b) * np.array([2.0, 1.0]), axis=1)

    data = np.array([
        ['red', 'round'], ['red', 'square'], ['red', 'round'],
        ['blue', 'square'], ['blue', 'round'], ['blue', 'square'],
    ])
    km = KModes(n_clusters=2, cat_dissim=weighted_matching,
                init='Huang', n_init=1, random_state=0)
    labels = km.fit_predict(data)
    predictions = km.predict(data)

    mixed = np.column_stack(([0.0, 0.2, 0.1, 9.8, 10.0, 10.2], data))
    kp = KPrototypes(n_clusters=2, cat_dissim=weighted_matching,
                     init='Huang', n_init=1, random_state=0)
    mixed_labels = kp.fit_predict(mixed, categorical=[1, 2])
    mixed_predictions = kp.predict(mixed, categorical=[1, 2])

:code:`KPrototypes` also accepts :code:`num_dissim`. Its callable receives
only the numerical columns, with the same 2D/1D input and 1D output shapes.
The default is the sum of squared numerical differences, without a square
root. Its cost is combined with :code:`gamma * cat_dissim(...)`.
Changing dissimilarity does not change the centroid update rules: categorical
centroids still use modes, and numerical centroids still use means. Choose a
function compatible with these updates, such as the weighted mismatch above.

Parallel execution
------------------

The k-modes and k-prototypes implementations both offer support for
multiprocessing via the 
`joblib library <https://pythonhosted.org/joblib/generated/joblib.Parallel.html>`_,
similar to e.g. scikit-learn's implementation of k-means, using the
:code:`n_jobs` parameter. It generally does not make sense to set more jobs
than there are processor cores available on your system.

This potentially speeds up any execution with more than one initialization try,
:code:`n_init > 1`, which may be helpful to reduce the execution time for
larger problems. Note that it depends on your problem whether multiprocessing
actually helps, so be sure to try that out first. You can check out the
examples for some benchmarks.

FAQ
---

**Q: I'm seeing errors such as "TypeError: '<' not supported between instances of 'str' and 'float'"
when using the kprototypes algorithm.**

A: One or more of your numerical feature columns have string values in them. Make sure that all 
columns have consistent data types.

----

**Q: How does k-protypes know which of my features are numerical and which are categorical?**

A: You tell it which column indices are categorical using the :code:`categorical` argument. All others are assumed numerical. E.g., :code:`clusters = KPrototypes().fit_predict(X, categorical=[1, 2])`

----

**Q: I'm getting the following error, what gives? "ModuleNotFoundError: No module named 'kmodes.kmodes'; 'kmodes' is not a package".**

A: Make sure your working file is not called 'kmodes.py', because it might overrule the :code:`kmodes` package.

----

**Q: I'm getting the following error: "ValueError: Clustering algorithm could not initialize. Consider assigning the initial clusters manually."**

A: This is a feature, not a bug. :code:`kmodes` is telling you that it can't make sense of the data you are presenting it. At least, not with the parameters you are setting the algorithm with. It is up to you, the data scientist, to figure out why. Some hints to possible solutions:

- Run with fewer clusters as the data might not support a large number of clusters
- Explore and visualize your data, checking for weird distributions, outliers, etc.
- Clean and normalize the data
- Increase the ratio of rows to columns

----

**Q: I'm getting the following error: "ValueError: Input contains NaN, infinity, or a value too large for dtype('float64')."**

A: Following scikit-learn, the k-modes algorithm does not accept :code:`np.NaN` 
values in the :code:`X` matrix. Users are suggested to fill in the missing 
data in a way that makes sense for the problem at hand.

----

**Q: How would like your library to be cited?**

A: Something along these lines would do nicely:

.. code-block::

  @Misc{devos2015,
    author = {Nelis J. de Vos},
    title = {kmodes categorical clustering library},
    howpublished = {\url{https://github.com/nicodv/kmodes}},
    year = {2015--2024}
  }


References
----------

.. [HUANG97] Huang, Z.: Clustering large data sets with mixed numeric and
   categorical values, Proceedings of the First Pacific Asia Knowledge
   Discovery and Data Mining Conference, Singapore, pp. 21-34, 1997.

.. [HUANG98] Huang, Z.: Extensions to the k-modes algorithm for clustering
   large data sets with categorical values, Data Mining and Knowledge
   Discovery 2(3), pp. 283-304, 1998.

.. [CAO09] Cao, F., Liang, J, Bai, L.: A new initialization method for
   categorical data clustering, Expert Systems with Applications 36(7),
   pp. 10223-10228., 2009.
