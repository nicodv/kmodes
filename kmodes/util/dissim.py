"""
Dissimilarity measures for clustering
"""

import numpy as np
import collections


# 汉明距离
def matching_dissim(a, b, **_):
    """Simple matching dissimilarity function"""
    return np.sum(a != b, axis=1)


def jaccard_dissim_binary(a, b, **__):
    """Jaccard dissimilarity function for binary encoded variables"""
    if ((a == 0) | (a == 1)).all() and ((b == 0) | (b == 1)).all():
        numerator = np.sum(np.bitwise_and(a, b), axis=1)
        denominator = np.sum(np.bitwise_or(a, b), axis=1)
        if (denominator == 0).any(0):
            raise ValueError("Insufficient Number of data since union is 0")
        return 1 - numerator / denominator
    raise ValueError("Missing or non Binary values detected in Binary columns.")


def jaccard_dissim_label(a, b, **__):
    """Jaccard dissimilarity function for label encoded variables"""
    if np.isnan(a.astype('float64')).any() or np.isnan(b.astype('float64')).any():
        raise ValueError("Missing values detected in Numeric columns.")
    intersect_len = np.empty(len(a), dtype=int)
    union_len = np.empty(len(a), dtype=int)
    ii = 0
    for row in a:
        intersect_len[ii] = len(np.intersect1d(row, b))
        union_len[ii] = len(np.unique(row)) + len(np.unique(b)) - intersect_len[ii]
        ii += 1
    if (union_len == 0).any():
        raise ValueError("Insufficient Number of data since union is 0")
    return 1 - intersect_len / union_len


# 欧氏距离
def euclidean_dissim(a, b, **_):
    """Euclidean distance dissimilarity function"""
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("Missing values detected in numerical columns.")
    return np.sum((a - b) ** 2, axis=1)


def ng_dissim(a, b, X=None, membship=None):
    """Ng et al.'s dissimilarity measure, as presented in
    Michael K. Ng, Mark Junjie Li, Joshua Zhexue Huang, and Zengyou He, "On the
    Impact of Dissimilarity Measure in k-Modes Clustering Algorithm", IEEE
    Transactions on Pattern Analysis and Machine Intelligence, Vol. 29, No. 3,
    January, 2007

    This function can potentially speed up training convergence.

    Note that membship must be a rectangular array such that the
    len(membship) = len(a) and len(membship[i]) = X.shape[1]

    In case of missing membship, this function reverts to
    matching dissimilarity (e.g., when predicting).
    """
    # Without membership, revert to matching dissimilarity
    if membship is None:
        return matching_dissim(a, b)

    def calc_cjr(b, X, memj, idr):
        """Num objects w/ category value x_{i,r} for rth attr in jth cluster"""
        xcids = np.where(memj == 1)
        return float((np.take(X, xcids, axis=0)[0][:, idr] == b[idr]).sum(0))

    def calc_dissim(b, X, memj, idr):
        # Size of jth cluster
        cj = float(np.sum(memj))
        return (1.0 - (calc_cjr(b, X, memj, idr) / cj)) if cj != 0.0 else 0.0

    if len(membship) != a.shape[0] and len(membship[0]) != X.shape[1]:
        raise ValueError("'membship' must be a rectangular array where "
                         "the number of rows in 'membship' equals the "
                         "number of rows in 'a' and the number of "
                         "columns in 'membship' equals the number of rows in 'X'.")

    return np.array([np.array([calc_dissim(b, X, membship[idj], idr)
                               if b[idr] == t else 1.0
                               for idr, t in enumerate(val_a)]).sum(0)
                     for idj, val_a in enumerate(a)])


# 欧氏距离和汉明距离
def NC_HM_dissim(a, b, w1, w2, **_):
    """Euclidean distance dissimilarity function"""
    D1 = np.sum((a - b) ** 2, axis=1)
    # print(D1)
    # HM距离
    D2 = np.sum(a != b, axis=1)
    # print(D2)
    # if np.isnan(a).any() or np.isnan(b).any():
    #     raise ValueError("Missing values detected in numerical columns.")
    D = w1 * D1 + w2 * D2
    return D1, D2, D


def contextSUmatrix(data_final):
    # 传入的是已经编码好的数据，索引从0开始
    col = np.arange(data_final.shape[1])
    index = np.arange(data_final.shape[0])
    eps = 1.4e-45
    H_single = {}
    for k in col:
        data_count = collections.Counter(data_final[:, k])
        px = np.array(list(data_count.values())) / len(data_final[:, k])
        log_px = np.log2(px)
        hx = np.sum(-log_px * px)
        H_single[k] = hx
    # 计算矩阵SU
    SU = np.zeros((len(col), len(col)))
    for i in col:
        # i 和 j 都是属性名字
        for j in col:
            # 除去自己的
            if i == j:
                continue
            # data总和
            total = len(data_final[:, i])

            # A B属性的set
            A_ids = np.unique(data_final[:, i])
            A_ids = np.sort(A_ids)
            B_ids = np.unique(data_final[:, j])
            B_ids = np.sort(B_ids)

            # A B属性统计
            #         A_counts = np.array(list(data_final[i].value_counts()))
            A_counts = np.array(list(collections.Counter(data_final[:, i]).values()))
            B_counts = np.array(list(collections.Counter(data_final[:, j]).values()))
            #         B_counts = np.array(list(data_final[j].value_counts()))

            # A B 概率
            A_ids_pro = A_counts / total
            B_ids_pro = B_counts / total

            hxy = 0
            Iter = 0
            for idA in A_ids:
                sum_single = 0.0
                # 遍历B_id的每一个独一的值
                for idB in B_ids:
                    # 第属性i 中属于idA的索引
                    idAOccur = np.where(data_final[:, i] == idA)
                    # 第属性j 中属于idB的索引
                    idBOccur = np.where(data_final[:, j] == idB)
                    # 两者共同出现的次数
                    idABOccur = np.intersect1d(idAOccur, idBOccur)
                    pxy = 1.0 * len(idABOccur) / len(idAOccur[0]) + eps
                    log_pxy = - np.log2(pxy)
                    sum_single += pxy * log_pxy
                hxy += A_ids_pro[Iter] * sum_single
                Iter += 1
            SU[i][j] = hxy
    return SU


def Context_computer(SU):
    Context = []
    for i in range(SU.shape[0]):
        mean_su = np.mean(SU[i])
        context = np.where(SU[i] > mean_su)[0]
        Context.append(context)
    return Context


def Distance(Context, data_final):
    # 计算度量矩阵
    Dict_Distance = {}
    # 遍历每个属性对应的context
    for i in range(len(Context)):
        # i +1 是因为列名从1开始
        Y_ids = np.unique(data_final[i + 1])
        DistanceMatrix = np.zeros((len(Y_ids), len(Y_ids)))
        # 遍历属于当前context中的元素
        for X in Context[i]:
            X_ids = np.unique(data_final[X + 1])
            sum_pxy = []
            # 遍历当前属性对应的元素
            for idX in X_ids:
                idXOccur = np.where(data_final[X + 1] == idX)
                Pxy = []
                # 遍历当前属性对应的元素，建立DistMatrix
                for idY in Y_ids:
                    idYOccur = np.where(data_final[i + 1] == idY)
                    idYXOccur = np.intersect1d(idXOccur, idYOccur)
                    pxy = 1.0 * len(idYXOccur) / len(idXOccur[0])
                    Pxy.append(pxy)
                sum_pxy.append(np.array(Pxy))
            sum_pxy = np.array(sum_pxy)
            for col_x in range(sum_pxy.shape[1]):
                for col_y in range(sum_pxy.shape[1]):
                    DistanceMatrix[col_x][col_y] = DistanceMatrix[col_x][col_y] + np.sum(
                        np.square(sum_pxy[:, col_x] - sum_pxy[:, col_y]))
        # 开平方
        DistanceMatrix = np.sqrt(DistanceMatrix) / len(Context[i])
        Dict_Distance[i + 1] = DistanceMatrix

    return Dict_Distance


def Context_Dict(datafinal):
    SU = contextSUmatrix(datafinal)
    Context = Context_computer(SU)
    dict_distance = Distance(Context, datafinal)
    return dict_distance


def context_dissm(a, b, Dict_Distance, **_):
    """
    a : centroid
    b : curpoint
    """
    Distance = []
    for cur in a:
        distance = 0
        for col in range(len(cur)):
            distance += Dict_Distance[col + 1][cur[col], b[col]]
        Distance.append(distance)
    return np.array(Distance)
