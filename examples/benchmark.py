#!/usr/bin/env python

import timeit
import numpy as np
from kmodes import kmodes

n_clusters = 20
# no. of points
N = 1e5
# no. of dimensions
M = 10
# no. of times test is repeated
K = 3

data = np.random.randint(1, 1000, (N, M))


def huang():
    _ = kmodes.KModes(n_clusters=n_clusters, init='Huang', n_init=1)\
        .fit_predict(data)


def cao():
    _ = kmodes.KModes(n_clusters=n_clusters, init='Cao').fit_predict(data)


if __name__ == '__main__':

    for cm in ('huang', 'cao'):
        print(cm.capitalize() + ': {:.2} seconds'.format(
            timeit.timeit(cm + '()',
                          setup='from __main__ import ' + cm,
                          number=K)))
