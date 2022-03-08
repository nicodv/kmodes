#!/usr/bin/env python

import numpy as np
from kmodes.kprototypes import KPrototypes

# stocks with their market caps, sectors and countries
syms = np.genfromtxt('people.csv', dtype=str, delimiter=',')[:, 0]
X = np.genfromtxt('people.csv', dtype=object, delimiter=',')[:, 1:]
X[:, 0] = X[:, 0].astype(float)

weights = [1] * 4
weights[2] = 100
weights[3] = 100

kproto = KPrototypes(n_clusters=3, init='Cao', verbose=2, sample_weights=weights)
clusters = kproto.fit_predict(X, categorical=[2])

# Print cluster centroids of the trained model.
print(kproto.cluster_centroids_)
# Print training statistics
print(kproto.cost_)
print(kproto.n_iter_)

for s, c in zip(syms, clusters):
    print(f"Symbol: {s}, cluster:{c}")
