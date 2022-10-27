"""
K-modes clustering for categorical data
"""

# pylint: disable=unused-argument,attribute-defined-outside-init

from collections import defaultdict

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array

from .util import get_max_value_key, encode_features, get_unique_rows, \
    decode_centroids, pandas_to_numpy
from .util.dissim import matching_dissim, ng_dissim, euclidean_dissim, NC_HM_dissim
from .util.init_methods import init_cao, init_huang

import plotly.graph_objects as go


class KModes(BaseEstimator, ClusterMixin):
    """k-modes clustering algorithm for categorical data.

    Parameters
    -----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    max_iter : int, default: 100
        Maximum number of iterations of the k-modes algorithm for a
        single run.

    cat_dissim : func, default: matching_dissim
        Dissimilarity function used by the k-modes algorithm for categorical variables.
        Defaults to the matching dissimilarity function.

    init : {'Huang', 'Cao', 'random' or an ndarray}, default: 'Cao'
        Method for initialization:
        'Huang': Method in Huang [1997, 1998]
        'Cao': Method in Cao et al. [2009]
        'random': choose 'n_clusters' observations (rows) at random from
        data for the initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        for the initial encoded centroids.

    n_init : int, default: 10
        Number of time the k-modes algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of cost.

    verbose : int, optional
        Verbosity mode.

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    n_jobs : int, default: 1
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    Attributes
    ----------
    cluster_centroids_ : array, [n_clusters, n_features]
        Categories of cluster centroids

    labels_ :
        Labels of each point

    cost_ : float
        Clustering cost, defined as the sum distance of all points to
        their respective cluster centroids.

    n_iter_ : int
        The number of iterations the algorithm ran for.

    epoch_costs_ :
        The cost of the algorithm at each epoch from start to completion.

    Notes
    -----
    See:
    Huang, Z.: Extensions to the k-modes algorithm for clustering large
    data sets with categorical values, Data Mining and Knowledge
    Discovery 2(3), 1998.

    """

    def __init__(self, n_clusters=8, max_iter=100, cat_dissim=matching_dissim,
                 init='Cao', n_init=10, verbose=0, random_state=None, n_jobs=1):
        # 要形成聚类的数量 以及要生成的中心点得数量
        self.n_clusters = n_clusters
        # kmodes在依次运行中最大迭代次数，单次运行的最大迭代次数
        self.max_iter = max_iter
        # kmodes算法对分类变量使用的异同函数
        self.cat_dissim = cat_dissim
        # 随机：从数据中随机选择ncluster观测值作为初始中心点
        self.init = init
        # 在不同的中心点种子下运行kmodes算法的次数，最终的结果将是ninit连续运行的最佳结果
        self.n_init = n_init
        # 初始化w
        # self.w1 = 1
        # self.w2 = 1
        # Verbosity
        self.verbose = verbose
        # 如果是int，random state 就是随机数发生器使用的种子， RandomState实例，则random_state 就是随机数生成器；如果是none，随机数发生器是RandomState
        self.random_state = random_state
        # 用于计算的作业数
        self.n_jobs = n_jobs
        if ((isinstance(self.init, str) and self.init == 'Cao') or
            hasattr(self.init, '__array__')) and self.n_init > 1:
            if self.verbose:
                print("Initialization method and algorithm are deterministic. "
                      "Setting n_init to 1.")
            self.n_init = 1

    def fit(self, X, y=None, sample_weight=None, **kwargs):
        """Compute k-modes clustering.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]

        sample_weight : sequence, default: None
        更新中心点的请示后，分配给每个单独数据点的权重，更新中心点时分配给每个数据点的权重
        The weight that is assigned to each individual data point when
        updating the centroids.
        """
        X = pandas_to_numpy(X)
        # random
        random_state = check_random_state(self.random_state)
        _validate_sample_weight(sample_weight, n_samples=X.shape[0],
                                n_clusters=self.n_clusters)

        self._enc_cluster_centroids, self._enc_map, self.labels_, self.cost_, \
        self.n_iter_, self.epoch_costs_ = k_modes(
            X,
            self.n_clusters,
            self.max_iter,
            self.cat_dissim,
            self.init,
            self.n_init,
            self.verbose,
            random_state,
            self.n_jobs,
            sample_weight
        )
        print(f"epoch_costs: {self.epoch_costs_}")
        self.plot(self.epoch_costs_)
        return self

    # 可视化紧密程度
    def plot(self, array):
        x = np.arange(len(array))
        y = array
        trace1 = go.Scatter(x=x, y=y, mode='lines+markers')  # 设置模式为折线图

        layout = go.Layout(
            title=' Compactness of Iteration ',
            xaxis=dict(title='Iteration'),
            yaxis=dict(title='Compactness'))
        fig = go.Figure(data=trace1, layout=layout)
        fig.show()

    def fit_predict(self, X, y=None, **kwargs):
        """Compute cluster centroids and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        """
        # 计算聚类中心点并预测每个样本的聚类指数
        return self.fit(X, **kwargs).predict(X, **kwargs)

    def predict(self, X, **kwargs):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """

        assert hasattr(self, '_enc_cluster_centroids'), "Model not yet fitted."

        if self.verbose and self.cat_dissim == ng_dissim:
            print("Ng's dissimilarity measure was used to train this model, "
                  "but now that it is predicting the model will fall back to "
                  "using simple matching dissimilarity.")

        X = pandas_to_numpy(X)
        # 检查数据是否符合规范
        X = check_array(X, dtype=None)
        # 给每一个属性编码
        X, _ = encode_features(X, enc_map=self._enc_map)
        return labels_cost(X, self._enc_cluster_centroids, self.cat_dissim)[0]

    @property
    def cluster_centroids_(self):
        if hasattr(self, '_enc_cluster_centroids'):
            return decode_centroids(self._enc_cluster_centroids, self._enc_map)
        raise AttributeError("'{}' object has no attribute 'cluster_centroids_' "
                             "because the model is not yet fitted.")


mu = 0.0001
w1 = 1
w2 = 1


def updateW(d1, d2):
    w1_ = 1 - mu * d1
    w2_ = 1 - mu * d2
    return w1_, w2_


def labels_cost(X, centroids, dissim, membship=None, sample_weight=None):
    """Calculate labels and cost function given a matrix of points and
    a list of centroids for the k-modes algorithm.
    """
    # 计算标签和损失函数
    X = check_array(X)

    n_points = X.shape[0]
    cost = 0.
    labels = np.empty(n_points, dtype=np.uint16)

    for ipoint, curpoint in enumerate(X):
        weight = sample_weight[ipoint] if sample_weight is not None else 1
        # 计算当前点与所有簇心之间的距离
        diss = dissim(centroids, curpoint, X=X, membship=membship)
        # 返回最小值的索引。也就是该点属于哪一个簇心
        clust = np.argmin(diss)
        # 给每个点贴上标签
        labels[ipoint] = clust
        # 计算每个簇中簇成员到簇心的和并相加 也就是紧密度 如果给每个权重人为赋值，则还需要乘以赋权
        cost += diss[clust] * weight
    # 返回根据D1和D2的距离，为了更新那个w
    return labels, cost


def k_modes(X, n_clusters, max_iter, dissim, init, n_init, verbose, random_state, n_jobs,
            sample_weight=None):
    """k-modes algorithm"""
    random_state = check_random_state(random_state)
    # 检查data是否为稀疏矩阵
    if sparse.issparse(X):
        raise TypeError("k-modes does not support sparse data.")

    X = check_array(X, dtype=None)

    # Convert the categorical values in X to integers for speed.
    # Based on the unique values in X, we can make a mapping to achieve this.
    # 将X中分类值转换为整数，以提高速度
    X, enc_map = encode_features(X)

    # 得到数据点数以及属性值数
    n_points, n_attrs = X.shape
    # 判断cluster是否小于数据点数量，如果小于则不能分这么多类
    assert n_clusters <= n_points, f"Cannot have more clusters ({n_clusters}) " \
                                   f"than data points ({n_points})."

    # Are there more n_clusters than unique rows? Then set the unique
    # rows as initial values and skip iteration.
    unique = get_unique_rows(X)
    # 计算有几个唯一的行
    n_unique = unique.shape[0]
    # 是否有比唯一行更多的聚类数，那么可以直接将唯一的行作为初始值并跳过迭代
    if n_unique <= n_clusters:
        max_iter = 0
        n_init = 1
        n_clusters = n_unique
        init = unique
    # 说明我是可以修改dissim
    # dissim = euclidean_dissim
    results = []
    seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
    if n_jobs == 1:
        for init_no in range(n_init):
            results.append(_k_modes_single(
                X, n_clusters, n_points, n_attrs, max_iter, dissim, init, init_no,
                verbose, seeds[init_no], sample_weight
            ))
    else:
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_k_modes_single)(X, n_clusters, n_points, n_attrs, max_iter,
                                     dissim, init, init_no, verbose, seed, sample_weight)
            for init_no, seed in enumerate(seeds))
    all_centroids, all_labels, all_costs, all_n_iters, all_epoch_costs = zip(*results)

    best = np.argmin(all_costs)
    if n_init > 1 and verbose:
        print(f"Best run was number {best + 1}")

    # 迭代n_init次数，找最好的
    return all_centroids[best], enc_map, all_labels[best], \
           all_costs[best], all_n_iters[best], all_epoch_costs[best]


def _k_modes_single(X, n_clusters, n_points, n_attrs, max_iter, dissim, init, init_no,
                    verbose, random_state, sample_weight=None):
    random_state = check_random_state(random_state)
    # 返回簇中心
    # _____ INIT _____
    # 这里强制设置dissim的原因是因为使用Cao的方法必须得基于欧几里得方法，不过后期可以考虑修改一下
    dissim = matching_dissim

    if verbose:
        print("Init: initializing centroids")
    if isinstance(init, str) and init.lower() == 'huang':
        centroids = init_huang(X, n_clusters, dissim, random_state)
    elif isinstance(init, str) and init.lower() == 'cao':
        # 返回基于Cao的方法得出的簇中心点
        centroids = init_cao(X, n_clusters, dissim)
    elif isinstance(init, str) and init.lower() == 'random':
        seeds = random_state.choice(range(n_points), n_clusters)
        centroids = X[seeds]
    elif hasattr(init, '__array__'):
        # Make sure init is a 2D array.
        if len(init.shape) == 1:
            init = np.atleast_2d(init).T
        assert init.shape[0] == n_clusters, \
            f"Wrong number of initial centroids in init ({init.shape[0]}, " \
            f"should be {n_clusters})."
        assert init.shape[1] == n_attrs, \
            f"Wrong number of attributes in init ({init.shape[1]}, " \
            f"should be {n_attrs})."
        centroids = np.asarray(init, dtype=np.uint16)
    else:
        raise NotImplementedError

    # 初始化簇
    if verbose:
        print("Init: initializing clusters")
    # 成员矩阵
    membship = np.zeros((n_clusters, n_points), dtype=np.bool_)
    # cl_attr_freq is a list of lists with dictionaries that contain the
    # frequencies of values per cluster and attribute.
    # 每个群组和属性的值得频率
    cl_attr_freq = [[defaultdict(int) for _ in range(n_attrs)]
                    for _ in range(n_clusters)]
    # enumerate 同时遍历索引和遍历元素 ipoint这里相当于索引， curpoint相当于元素
    for ipoint, curpoint in enumerate(X):
        weight = sample_weight[ipoint] if sample_weight is not None else 1
        # Initial assignment to clusters
        # 初始化分配到群组
        # 这样处理太高级了吧 求距离，找到相应最小的索引，然后根据memship矩阵一一填写数据点属于那一类
        # clust = np.argmin(dissim(centroids, curpoint, X=X, membship=membship))
        clust = np.argmin(dissim(centroids, curpoint, X=X, membship=membship))
        membship[clust, ipoint] = 1
        # Count attribute values per cluster.
        # 计算 每个data的属性值
        for iattr, curattr in enumerate(curpoint):
            cl_attr_freq[clust][iattr][curattr] += weight
    # Perform an initial centroid update.
    # 初始中心点更新
    for ik in range(n_clusters):
        for iattr in range(n_attrs):
            if sum(membship[ik]) == 0:
                # 如果是空的中心点， 则进行随机选择
                centroids[ik, iattr] = random_state.choice(X[:, iattr])
            else:
                # 更新簇中心，找到众数
                centroids[ik, iattr] = get_max_value_key(cl_attr_freq[ik][iattr])

    # _____ ITERATION _____
    # 开始迭代
    if verbose:
        print("Starting iterations...")
    itr = 0
    labels = None
    converged = False

    # 返回标签和损失函数（即紧密度）同时返回w1和w2
    _, cost = labels_cost(X, centroids, dissim, membship, sample_weight)
    # 记录 cost
    epoch_costs = [cost]
    # 在计算compactness中更新迭代w
    while itr < max_iter and not converged:
        itr += 1
        centroids, cl_attr_freq, membship, moves = _k_modes_iter(
            X,
            centroids,
            cl_attr_freq,
            membship,
            dissim,
            random_state,
            sample_weight
        )
        # All points seen in this iteration
        labels, ncost = labels_cost(X, centroids, dissim, membship, sample_weight)
        # 迭代停止条件，1. 点不再移动； 2. 损失函数变大
        converged = (moves == 0) or (ncost >= cost)
        epoch_costs.append(ncost)
        cost = ncost
        if verbose:
            print(f"Run {init_no + 1}, iteration: {itr}/{max_iter}, "
                  f"moves: {moves}, cost: {cost}")

    return centroids, labels, cost, itr, epoch_costs


def _k_modes_iter(X, centroids, cl_attr_freq, membship, dissim, random_state,
                  sample_weight):
    """
        :param X: 数据集
        :param centroids: 簇心
        :param cl_attr_freq: 三维矩阵 数据点、属性、 属性值
        :param membship: 成员矩阵 0 1构成的
        :param dissim: 距离计算公式
        :param random_state:
        :param sample_weight: 权重

        :return: centroids, cl_attr_freq, membship, moves

    Single iteration of k-modes clustering algorithm

    """
    # 单次迭代的 kmodes 聚类算法
    moves = 0

    for ipoint, curpoint in enumerate(X):
        weight = sample_weight[ipoint] if sample_weight is not None else 1

        # 计算该点与所有簇心的距离 # 开始计算不同w1和不同w2
        clust = np.argmin(dissim(centroids, curpoint, X=X, membship=membship))

        if membship[clust, ipoint]:
            # 成员本来就在应当的位置，则不用进行更新.
            continue

        # Move point, and update old/new cluster frequencies and centroids.
        # 移动点，并更新旧 / 新集群频率和中心点

        moves += 1
        # 之前的每个点属于的类
        old_clust = np.argwhere(membship[:, ipoint])[0][0]
        cl_attr_freq, membship, centroids = _move_point_cat(
            curpoint, ipoint, clust, old_clust, cl_attr_freq, membship, centroids,
            weight
        )

        # In case of an empty cluster, reinitialize with a random point
        # from the largest cluster.
        # 如果有一个空的cluster， 那么用一个随机点重新初始化，初始化于最大的群集
        if not membship[old_clust, :].any():
            from_clust = membship.sum(axis=1).argmax()
            choices = [ii for ii, ch in enumerate(membship[from_clust, :]) if ch]
            rindx = random_state.choice(choices)

            cl_attr_freq, membship, centroids = _move_point_cat(
                X[rindx], rindx, old_clust, from_clust, cl_attr_freq, membship,
                centroids, weight
            )

    return centroids, cl_attr_freq, membship, moves


def _move_point_cat(point, ipoint, to_clust, from_clust, cl_attr_freq,
                    membship, centroids, sample_weight):
    """

    :param point: 当前点
    :param ipoint: 编号
    :param to_clust: 现在该点属于的类
    :param from_clust: 之前该点属于的类
    :param cl_attr_freq: 三维矩阵 数据点、属性、 属性值
    :param membship: 成员矩阵 0 1构成的
    :param centroids: 簇心
    :param sample_weight: 权重

    :return: new cl_attr_freq, membship, centroids

    """
    # 在集群、分类属性之间移动点
    # 更新成员矩阵
    membship[to_clust, ipoint] = 1
    membship[from_clust, ipoint] = 0
    # 更新集群中属性的频率
    # Update frequencies of attributes in cluster.
    for iattr, curattr in enumerate(point):
        to_attr_counts = cl_attr_freq[to_clust][iattr]
        from_attr_counts = cl_attr_freq[from_clust][iattr]

        # Increment the attribute count for the new "to" cluster
        to_attr_counts[curattr] += sample_weight
        # 当前属性对应值的数量
        current_attribute_value_freq = to_attr_counts[curattr]
        # 当前簇中心的点对应值
        current_centroid_value = centroids[to_clust][iattr]
        current_centroid_freq = to_attr_counts[current_centroid_value]
        # 把值增加了新的模式，更新中心点
        if current_centroid_freq < current_attribute_value_freq:
            # We have incremented this value to the new mode. Update the centroid.
            centroids[to_clust][iattr] = curattr

        # Decrement the attribute count for the old "from" cluster
        # 减少旧的来自集群的属性计数
        from_attr_counts[curattr] -= sample_weight
        # 旧的簇心的值
        old_centroid_value = centroids[from_clust][iattr]
        # 刚从旧的中心点值中删除了一个计数，需要重新计算中心点，因为可能不再是最大的
        if old_centroid_value == curattr:
            # We have just removed a count from the old centroid value. We need to
            # recalculate the centroid as it may no longer be the maximum
            centroids[from_clust][iattr] = get_max_value_key(from_attr_counts)

    return cl_attr_freq, membship, centroids


def _validate_sample_weight(sample_weight, n_samples, n_clusters):
    if sample_weight is not None:
        if len(sample_weight) != n_samples:
            raise ValueError("sample_weight should be of equal size as samples.")
        if any(
                not isinstance(weight, int) and not isinstance(weight, float)
                for weight in sample_weight
        ):
            raise ValueError("sample_weight elements should either be int or floats.")
        if any(sample < 0 for sample in sample_weight):
            raise ValueError("sample_weight elements should be positive.")
        if sum([x > 0 for x in sample_weight]) < n_clusters:
            raise ValueError("Number of non-zero sample_weight elements should be "
                             "larger than the number of clusters.")
