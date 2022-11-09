# -*- coding: UTF-8 -*-
"""
@Project ：kmodes
@File    ：Mushroom.py
@Author  ：Fangqi Nie
@Date    ：2022/10/30 16:18
@Contact :   kristinaNFQ@163.com
"""


import numpy as np
from kmodes.kmodes import KModes

# reproduce results on small soybean data set
x = np.genfromtxt('Encode_mushroom.csv', dtype=int, delimiter=',')[:, 1:]
y = np.genfromtxt('Encode_mushroom.csv', dtype=int, delimiter=',')[:, 0]
# print(len(x))
k = 3
kmodes_huang = KModes(n_clusters=k, init='Huang', verbose=1)
kmodes_huang.fit(x)
#
# # Print cluster centroids of the trained model.
# print('k-modes (Huang) centroids:')
# print(kmodes_huang.cluster_centroids_)
# # Print training statistics
# print(f'Final training cost: {kmodes_huang.cost_}')
# print(f'Training iterations: {kmodes_huang.n_iter_}')
#
# kmodes_cao = KModes(n_clusters=10, init='Cao', verbose=1)
# kmodes_cao.fit(x)

# Print cluster centroids of the trained model.
# print('k-modes (Cao) centroids:')
# print(kmodes_cao.cluster_centroids_)
# # Print training statistics
# print(f'Final training cost: {kmodes_cao.cost_}')
# print(f'Training iterations: {kmodes_cao.n_iter_}')
#
# print('Results tables:')
# for result in (kmodes_huang, kmodes_cao):
#     classtable = np.zeros((4, 4), dtype=int)
#     for ii, _ in enumerate(y):
#         classtable[int(y[ii][-1]) - 1, result.labels_[ii]] += 1
#
#     print("\n")
#     print("    | Cl. 1 | Cl. 2 | Cl. 3 | Cl. 4 |")
#     print("----|-------|-------|-------|-------|")
#     for ii in range(4):
#         prargs = tuple([ii + 1] + list(classtable[ii, :]))
#         print(" D{0} |    {1:>2} |    {2:>2} |    {3:>2} |    {4:>2} |".format(*prargs))
