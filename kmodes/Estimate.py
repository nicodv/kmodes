# -*- coding: UTF-8 -*-
"""
@Project ：kmodes
@File    ：Estimate.py
@Author  ：Fangqi Nie
@Date    ：2022/11/8 13:59
@Contact :   kristinaNFQ@163.com
"""

import math
import numpy as np
from sklearn import metrics
from sklearn.metrics import pair_confusion_matrix
from sklearn.metrics import accuracy_score
import collections


# 知道真实标签的指标
def NMI_SELF(A, B):
    # 样本点数
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    # 互信息计算
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A == idA)  # 输出满足条件的元素的下标
            idBOccur = np.where(B == idB)
            idABOccur = np.intersect1d(idAOccur, idBOccur)  # Find the intersection of two arrays.
            px = 1.0 * len(idAOccur[0]) / total
            py = 1.0 * len(idBOccur[0]) / total
            pxy = 1.0 * len(idABOccur) / total
            MI = MI + pxy * math.log(pxy / (px * py) + eps, 2)
    # 标准化互信息
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0 * len(np.where(A == idA)[0])
        Hx = Hx - (idAOccurCount / total) * math.log(idAOccurCount / total + eps, 2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0 * len(np.where(B == idB)[0])
        Hy = Hy - (idBOccurCount / total) * math.log(idBOccurCount / total + eps, 2)
    MIhat = 2.0 * MI / (Hx + Hy)
    return MIhat


def NMI_sklearn(predict, label):
    return metrics.normalized_mutual_info_score(predict, label)


def purity(labels_true, labels_pred):
    """
    Purity [0, 1] 越接近1表示聚类结果越好
    该值无法用于权衡聚类质量与簇个数之间的关系

    Parameters
    ----------
    labels_true
    labels_pred

    Returns 聚类纯度
    -------

    """
    clusters = np.unique(labels_pred)
    labels_true = np.reshape(labels_true, (-1, 1))
    labels_pred = np.reshape(labels_pred, (-1, 1))
    count = []
    for c in clusters:
        idx = np.where(labels_pred == c)[0]
        labels_tmp = labels_true[idx, :].reshape(-1)
        count.append(np.bincount(labels_tmp).max())
    return np.sum(count) / labels_true.shape[0]


def ARI(labels_true, labels_pred, beta=1.):
    """


    Parameters
    ----------
    labels_true
    labels_pred
    beta

    Returns
    -------

    """
    (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)
    ri = (tp + tn) / (tp + tn + fp + fn)
    ari = 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    p, r = tp / (tp + fp), tp / (tp + fn)
    f_beta = (1 + beta ** 2) * (p * r / ((beta ** 2) * p + r))
    return ri, ari, f_beta


def AC(labels_true, labels_pre):
    acc = accuracy_score(labels_true, labels_pre)
    return acc


if __name__ == '__main__':
    a = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    b = np.array([1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 1, 1, 3, 3, 3])
    # print(NMI_SELF(a, b))
    # print(metrics.normalized_mutual_info_score(a, b))  # 直接调用sklearn中的函数
    print(purity(a, b))
    ri, ari, f_beta = ARI(a, b)
    print(f"RI = {ri}\nARI = {ari}\nF_beta = {f_beta}")
    print(f"AC = {AC(a, b)}")
