.. image:: https://img.shields.io/pypi/v/kmodes.svg
    :target: https://pypi.python.org/pypi/kmodes/
    :alt: Version
.. image:: https://img.shields.io/pypi/l/kmodes.svg
    :target: https://github.com/nicodv/kmodes/blob/master/LICENSE
    :alt: License
.. image:: https://travis-ci.org/nicodv/kmodes.svg?branch=master
    :target: https://travis-ci.org/nicodv/kmodes
    :alt: Test Status
.. image:: https://coveralls.io/repos/nicodv/kmodes/badge.svg
    :target: https://coveralls.io/r/nicodv/kmodes
    :alt: Test Coverage
.. image:: https://landscape.io/github/nicodv/kmodes/master/landscape.svg?style=flat
    :target: https://landscape.io/github/nicodv/kmodes/master
    :alt: Code Health

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
    from kmodes import kmodes
    
    # random categorical data
    data = np.random.choice(20, (100, 10))
    
    km = kmodes.KModes(n_clusters=4, init='Huang', n_init=5, verbose=1)

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
