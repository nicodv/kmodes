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
from kmodes.Estimate import NMI_sklearn, purity, ARI, AC

# reproduce results on small soybean data set
x = np.genfromtxt('Encode_mushroom.csv', dtype=int, delimiter=',')[:, 1:]
y = np.genfromtxt('Encode_mushroom.csv', dtype=int, delimiter=',')[:, 0]
# print(len(x))
k = 2
kmodes_huang = KModes(n_clusters=k, init='Huang', verbose=1)
kmodes_huang.fit(x)
number = 0
ARI_ = []
NMI_ = []
Purity_ = []
AC_ = []
for pre in kmodes_huang.all_labels:
    number += 1
    # print(f"number1 run: {number}\n")
    ri, ari, f_beta = ARI(kmodes_huang.labels_, y)
    # print(f"\nRI = {ri}\nARI = {ari}\nF_beta = {f_beta}\n")
    nmi = NMI_sklearn(kmodes_huang.labels_, y)
    # print(f"NMI = {nmi}\n")
    Purity = purity(kmodes_huang.labels_, y)
    # print(f"Purity = {Purity}")
    Acc = AC(kmodes_huang.labels_, y)
    ARI_.append(ari)
    NMI_.append(nmi)
    Purity_.append(Purity)
    AC_.append(Acc)
print("Sum of all run:\n")
print(f"ARI_std: {np.std(ARI_)}\nNMI_std: {np.std(NMI_)}\nPurity_std: {np.std(Purity_)}\nACC_std: {np.std(AC_)}")
print(f"ARI_mean: {np.mean(ARI_)}\nNMI_mean: {np.mean(NMI_)}\nPurity_mean: {np.mean(Purity_)}\nACC_mean: {np.mean(AC_)}")
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
