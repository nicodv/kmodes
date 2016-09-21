#!/usr/bin/env python

import timeit
import numpy as np
from kmodes import kmodes

# number of clusters
K = 20
# no. of points
N = int(1e5)
# no. of dimensions
M = 10
# no. of times test is repeated
T = 3

data = np.random.randint(1, 1000, (N, M))


def huang():
    kmodes.KModes(n_clusters=K, init='Huang', n_init=1, verbose=2).fit_predict(data)


def cao():
    kmodes.KModes(n_clusters=K, init='Cao', verbose=2).fit_predict(data)


if __name__ == '__main__':

    for cm in ('huang', 'cao'):
        print(cm.capitalize() + ': {:.2} seconds'.format(
            timeit.timeit(cm + '()',
                          setup='from __main__ import ' + cm,
                          number=T)))
