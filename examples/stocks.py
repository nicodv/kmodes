#!/usr/bin/env python

import numpy as np
from kmodes import kprototypes

# stocks with their market caps, sectors and countries
syms = np.genfromtxt('stocks.csv', dtype=str, delimiter=',')[:, 0]
X = np.genfromtxt('stocks.csv', dtype=object, delimiter=',')[:, 1:]

kproto = kprototypes.KPrototypes(n_clusters=4, init='Cao', verbose=2)
clusters = kproto.fit_predict(X, categorical=[1, 2])

for s, c in zip(syms, clusters):
    print("Symbol: {}, cluster:{}".format(s, c))
