#!/usr/bin/env python

import pandas as pd
import numpy as np
from kmodes.kmodes import KModes

# reproduce results on small soybean data set
x = np.genfromtxt('soybean.csv', dtype=int, delimiter=',')[:, :-1]
y = np.genfromtxt('soybean.csv', dtype=str, delimiter=',', usecols=(35, ))
print(len(x))
k = 4
kmodes_huang = KModes(n_clusters=k, init='Huang', verbose=1)
kmodes_huang.fit(x)
print(f'k-modes (Huang) centroids:\n{kmodes_huang.cluster_centroids_}')

real_length = len(set(y))
print('Results tables:')
classTable = np.zeros((real_length, k))
for ii, _ in enumerate(y):
    classTable[int(y[ii][-1]) - 1, kmodes_huang.labels_[ii]] += 1

#     for ii, _ in enumerate(y):
#         classtable[int(y[ii][-1]) - 1, result.labels_[ii]] += 1
classTable = pd.DataFrame(classTable)
# classTable.to_csv('test2.csv')
print(classTable)

# # Print cluster centroids of the trained model.
# print('k-modes (Huang) centroids:')
# print(kmodes_huang.cluster_centroids_)
# # Print training statistics
# print(f'Final training cost: {kmodes_huang.cost_}')
# print(f'Training iterations: {kmodes_huang.n_iter_}')
#
# kmodes_cao = KModes(n_clusters=6, init='Cao', verbose=1)
# kmodes_cao.fit(x)
#
# # Print cluster centroids of the trained model.
# print('k-modes (Cao) centroids:')
# print(kmodes_cao.cluster_centroids_)
# # Print training statistics
# print(f'Final training cost: {kmodes_cao.cost_}')
# print(f'Training iterations: {kmodes_cao.n_iter_}')

#
# print('Results tables:')
# for result in (kmodes_huang, kmodes_cao):
#     classtable = np.zeros((6, 6), dtype=int)
#     for ii, _ in enumerate(y):
#         classtable[int(y[ii][-1]) - 1, result.labels_[ii]] += 1
#
#     print("\n")
#     print("    | Cl. 1 | Cl. 2 | Cl. 3 | Cl. 4 | Cl. 5 | Cl. 6 |")
#     print("----|-------|-------|-------|-------|-------|-------|")
#     for ii in range(6):
#         prargs = tuple([ii + 1] + list(classtable[ii, :]))
#         print(" D{0} |    {1:>2} |    {2:>2} |    {3:>2} |    {4:>2} |    {5:>2} |    {6:>2} |".format(*prargs))
