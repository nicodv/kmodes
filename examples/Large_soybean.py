# -*- coding: UTF-8 -*-
"""
@Project ：kmodes
@File    ：Large_soybean.py
@Author  ：Fangqi Nie
@Date    ：2022/10/30 15:43
@Contact :   kristinaNFQ@163.com
"""

import numpy as np
import pandas as pd

from kmodes.kmodes import KModes

# reproduce results on small soybean data set
x = np.genfromtxt('Encode_soybean_large.csv', dtype=int, delimiter=',')[:, 1:]
y = np.genfromtxt('Encode_soybean_large.csv', dtype=int, delimiter=',')[:, 0]
# print(len(y))
k = 15
kmodes_huang = KModes(n_clusters=k, init='Huang', verbose=1)
kmodes_huang.fit(x)
# print(kmodes_huang.labels_)
#
# # Print cluster centroids of the trained model.
print(f'k-modes (Huang) centroids:\n{kmodes_huang.cluster_centroids_}')
# for i in range(k):
#     print(sum(kmodes_huang.membship[i]))

# # Print training statistics
# print(f'Final training cost: {kmodes_huang.cost_}')
# print(f'Training iterations: {kmodes_huang.n_iter_}')
#
# kmodes_cao = KModes(n_clusters=15, init='Cao', verbose=1)
# kmodes_cao.fit(x)
#
# # Print cluster centroids of the trained model.
# print('k-modes (Cao) centroids:')
# print(kmodes_cao.cluster_centroids_)
# # Print training statistics
# print(f'Final training cost: {kmodes_cao.cost_}')
# print(f'Training iterations: {kmodes_cao.n_iter_}')
#
# label:
# real:
real_length = len(set(y))
print('Results tables:')
classTable = np.zeros((real_length, k))
for ii, _ in enumerate(y):
    classTable[_, kmodes_huang.labels_[ii]] += 1

classTable = pd.DataFrame(classTable)
print(classTable)
# classTable.to_csv('test2.csv')
