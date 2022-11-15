# -*- coding: UTF-8 -*-
"""
@Project ：kmodes
@File    ：Mushroom.py
@Author  ：Fangqi Nie
@Date    ：2022/10/30 16:18
@Contact :   kristinaNFQ@163.com
"""


import numpy as np
from kmodes.Estimate import NMI_sklearn, purity, ARI
from kmodes.kmodes import KModes
import pandas as pd

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
for pre in kmodes_huang.all_labels:
    number += 1
    # print(f"number1 run: {number}\n")
    ri, ari, f_beta = ARI(kmodes_huang.labels_, y)
    # print(f"\nRI = {ri}\nARI = {ari}\nF_beta = {f_beta}\n")
    nmi = NMI_sklearn(kmodes_huang.labels_, y)
    # print(f"NMI = {nmi}\n")
    Purity = purity(kmodes_huang.labels_, y)
    # print(f"Purity = {Purity}")
    ARI_.append(ari)
    NMI_.append(nmi)
    Purity_.append(Purity)
print("Sum of all run:\n")
print(f"ARI_std: {np.std(ARI_)}\nNMI_std: {np.std(NMI_)}\nPurity_std: {np.std(Purity_)}")
print(f"ARI_mean: {np.mean(ARI_)}\nNMI_mean: {np.mean(NMI_)}\nPurity_mean: {np.mean(Purity_)}")

real_length = len(set(y))
# print('Results tables:')
# classTable = np.zeros((real_length, k))
# for ii, _ in enumerate(y):
#     classTable[int(y[ii][-1]) - 1, kmodes_huang.labels_[ii]] += 1
# print(kmodes_huang.labels_)
#
# classTable = pd.DataFrame(classTable)
#
# print(classTable)
