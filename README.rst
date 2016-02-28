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

The code is modeled after the clustering algorithms in scikit-learn and has
the same familiar interface.

Simple usage examples of both k-modes ('soybean.py') and k-prototypes
('stocks.py') are included in the examples directory.

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
