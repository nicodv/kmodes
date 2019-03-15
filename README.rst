.. image:: https://img.shields.io/pypi/v/kmodes.svg
    :target: https://pypi.python.org/pypi/kmodes/
    :alt: Version
.. image:: https://travis-ci.org/nicodv/kmodes.svg?branch=master
    :target: https://travis-ci.org/nicodv/kmodes
    :alt: Test Status
.. image:: https://coveralls.io/repos/nicodv/kmodes/badge.svg
    :target: https://coveralls.io/r/nicodv/kmodes
    :alt: Test Coverage
.. image:: https://api.codacy.com/project/badge/Grade/cb19f1f1093a44fa845ebfdaf76975f6
   :alt: Codacy Badge
   :target: https://app.codacy.com/app/nicodv/kmodes?utm_source=github.com&utm_medium=referral&utm_content=nicodv/kmodes&utm_campaign=Badge_Grade_Dashboard
.. image:: https://requires.io/github/nicodv/kmodes/requirements.svg
     :target: https://requires.io/github/nicodv/kmodes/requirements/
     :alt: Requirements Status
.. image:: https://img.shields.io/pypi/pyversions/kmodes.svg
    :target: https://pypi.python.org/pypi/kmodes/
    :alt: Supported Python versions
.. image:: https://img.shields.io/github/stars/nicodv/kmodes.svg
    :target: https://github.com/nicodv/kmodes/
    :alt: Github stars
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

kmodes can be installed using pip:

.. code:: bash

    pip install kmodes

To upgrade to the latest version (recommended), run it like this:

.. code:: bash

    pip install --upgrade kmodes

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

Missing / unseen data
_____________________

The k-modes algorithm accepts :code:`np.NaN` values as missing values in
the :code:`X` matrix. However, users are strongly suggested to consider
filling in the missing data themselves in a way that makes sense for
the problem at hand. This is especially important in case of many missing
values.

The k-modes algorithm currently handles missing data as follows. When
fitting the model, :code:`np.NaN` values are encoded into their own
category (let's call it "unknown values"). When predicting, the model
treats any values in :code:`X` that (1) it has not seen before during
training, or (2) are missing, as being a member of the "unknown values"
category. Simply put, the algorithm treats any missing / unseen data as
matching with each other but mismatching with non-missing / seen data
when determining similarity between points.

The k-prototypes also accepts :code:`np.NaN` values as missing values for
the categorical variables, but does *not* accept missing values for the
numerical values. It is up to the user to come up with a way of
handling these missing data that is appropriate for the problem at hand.

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

Q: I'm seeing errors such as :code:`TypeError: '<' not supported between instances of 'str' and 'float'`
when using the :code:`kprototypes` algorithm.

A: One or more of your numerical feature columns have string values in them. Make sure that all 
columns have consistent data types.

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
