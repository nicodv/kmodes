#!/usr/bin/env python

import timeit

import numpy as np

from kmodes.kprototypes import KPrototypes

# number of clusters
K = 20
# no. of points
N = int(1e5)
# no. of dimensions
M = 10
# no. of numerical dimensions
MN = 5
# no. of times test is repeated
T = 3

data = np.random.randint(1, 1000, (N, M))


def huang():
    KPrototypes(n_clusters=K, init='Huang', n_init=1, verbose=2)\
        .fit_predict(data, categorical=list(range(M - MN, M)))


def cao():
    KPrototypes(n_clusters=K, init='Cao', verbose=2)\
        .fit_predict(data, categorical=list(range(M - MN, M)))


if __name__ == '__main__':

    for cm in ('huang', 'cao'):
        print(cm.capitalize() + ': {:.2} seconds'.format(
            timeit.timeit(cm + '()',
                          setup='from __main__ import ' + cm,
                          number=T)))
