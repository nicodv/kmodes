"""K-modes clustering"""

# Author: Nico de Vos <njdevos@gmail.com>
# License: MIT

from collections import defaultdict
import numpy as np


def _mode_from_dict(dic):
    """Fast method to get key for maximum value in dict."""
    v = list(dic.values())
    k = list(dic.keys())
    return k[v.index(max(v))]


def _matching_dissim(a, b):
    """Simple matching dissimilarity function"""
    return (np.atleast_2d(a) != b).sum(axis=1)


def _euclidean_dissim(a, b):
    """Euclidean distance dissimilarity function"""
    return np.sum((np.atleast_2d(a) - b) ** 2, axis=1)


def _init_huang(X, n_clusters):
    """Initialize n_clusters according to method by Huang [1997]."""
    nattrs = X.shape[1]
    centroids = np.empty((n_clusters, nattrs), dtype='object')
    # determine frequencies of attributes
    for iattr in range(nattrs):
        freq = defaultdict(int)
        for curattr in X[:, iattr]:
            freq[curattr] += 1
        # Sample centroids using the probabilities of attributes.
        # (I assume that's what's meant in the Huang [1998] paper; it works,
        # at least)
        # Note: sampling using population in static list with as many choices
        # as frequency counts. Since the counts are small integers,
        # memory consumption is low.
        choices = [chc for chc, wght in freq.items() for _ in range(wght)]
        centroids[:, iattr] = np.random.choice(choices, n_clusters)
    # The previously chosen centroids could result in empty clusters,
    # so set centroid to closest point in X.
    for ik in range(n_clusters):
        ndx = np.argsort(_matching_dissim(X, centroids[ik]))
        # We want the centroid to be unique.
        while np.all(X[ndx[0]] == centroids, axis=1).any():
            ndx = np.delete(ndx, 0)
        centroids[ik] = X[ndx[0]]

    return centroids


def _init_cao(X, n_clusters):
    """Initialize n_clusters according to method by Cao et al. [2009].

    Note: O(N * attr * n_clusters**2), so watch out with large n_clusters
    """
    npoints, nattrs = X.shape
    centroids = np.empty((n_clusters, nattrs), dtype='object')
    # Method is base don determining density of points.
    dens = np.zeros(npoints)
    for iattr in range(nattrs):
        freq = defaultdict(int)
        for val in X[:, iattr]:
            freq[val] += 1
        for ipoint in range(npoints):
            dens[ipoint] += freq[X[ipoint, iattr]] / float(nattrs)
    dens /= npoints

    # Choose initial centroids based on distance and density.
    centroids[0] = X[np.argmax(dens)]
    if n_clusters > 1:
        # For the remaining centroids, choose maximum dens * dissim to the
        # (already assigned) centroid with the lowest dens * dissim.
        for ik in range(1, n_clusters):
            dd = np.empty((ik, npoints))
            for ikk in range(ik):
                dd[ikk] = _matching_dissim(X, centroids[ikk]) * dens
            centroids[ik] = X[np.argmax(np.min(dd, axis=0))]

    return centroids


def _move_point_cat(point, ipoint, to_clust, from_clust,
                    cl_attr_freq, membership):
    """Move point between clusters, categorical attributes."""
    membership[to_clust, ipoint] = 1
    membership[from_clust, ipoint] = 0
    # Update frequencies of attributes in cluster.
    for iattr, curattr in enumerate(point):
        cl_attr_freq[to_clust][iattr][curattr] += 1
        cl_attr_freq[from_clust][iattr][curattr] -= 1
    return cl_attr_freq, membership


def _move_point_num(point, ipoint, to_clust, from_clust,
                    cl_attr_sum, membership):
    """Move point between clusters, numerical attributes."""
    membership[to_clust, ipoint] = 1
    membership[from_clust, ipoint] = 0
    # Update sum of attributes in cluster.
    for iattr, curattr in enumerate(point):
        cl_attr_sum[to_clust][iattr] += curattr
        cl_attr_sum[from_clust][iattr] -= curattr
    return cl_attr_sum, membership


def _labels_cost_kmodes(X, centroids):
    """Calculate labels and cost function given a matrix of points and
    a list of centroids for the k-modes algorithm.
    """

    npoints = X.shape[0]
    cost = 0.
    labels = np.empty(npoints, dtype='int64')
    for ipoint, curpoint in enumerate(X):
        diss = _matching_dissim(centroids, curpoint)
        clust = np.argmin(diss)
        labels[ipoint] = clust
        cost += diss[clust]

    return labels, cost


def _k_modes_iter(X, centroids, cl_attr_freq, membership):
    """Single iteration of k-modes clustering algorithm"""
    moves = 0
    for ipoint, curpoint in enumerate(X):
        clust = np.argmin(_matching_dissim(centroids, curpoint))
        if membership[clust, ipoint]:
            # Point is already in its right place.
            continue

        # Move point, and update old/new cluster frequencies and centroids.
        moves += 1
        old_clust = np.argwhere(membership[:, ipoint])[0][0]

        cl_attr_freq, membership = _move_point_cat(
            curpoint, ipoint, clust, old_clust, cl_attr_freq, membership)

        # Update new and old centroids by choosing mode of attribute.
        for iattr in range(len(curpoint)):
            for curc in (clust, old_clust):
                centroids[curc, iattr] = \
                    _mode_from_dict(cl_attr_freq[curc][iattr])

        # In case of an empty cluster, reinitialize with a random point
        # from the largest cluster.
        if sum(membership[old_clust, :]) == 0:
            from_clust = membership.sum(axis=1).argmax()
            choices = \
                [ii for ii, ch in enumerate(membership[from_clust, :]) if ch]
            rindx = np.random.choice(choices)

            cl_attr_freq, membership = _move_point_cat(
                X[rindx], rindx, old_clust, from_clust, cl_attr_freq,
                membership)

    return centroids, moves


def k_modes(X, n_clusters, init, n_init, max_iter, verbose):
    """k-modes algorithm"""

    # Convert to numpy array, if needed.
    X = np.asanyarray(X)
    npoints, nattrs = X.shape
    assert n_clusters < npoints, "More clusters than data points?"

    all_centroids = []
    all_labels = []
    all_costs = []
    for init_no in range(n_init):

        # _____ INIT _____
        if verbose:
            print("Init: initializing centroids")
        if init == 'Huang':
            centroids = _init_huang(X, n_clusters)
        elif init == 'Cao':
            centroids = _init_cao(X, n_clusters)
        elif init == 'random':
            seeds = np.random.choice(range(npoints), n_clusters)
            centroids = X[seeds]
        elif hasattr(init, '__array__'):
            centroids = init
        else:
            raise NotImplementedError

        if verbose:
            print("Init: initializing clusters")
        membership = np.zeros((n_clusters, npoints), dtype='int64')
        # cl_attr_freq is a list of lists with dictionaries that contain the
        # frequencies of values per cluster and attribute.
        cl_attr_freq = [[defaultdict(int) for _ in range(nattrs)]
                        for _ in range(n_clusters)]
        for ipoint, curpoint in enumerate(X):
            # Initial assignment to clusters
            clust = np.argmin(_matching_dissim(centroids, curpoint))
            membership[clust, ipoint] = 1
            # Count attribute values per cluster.
            for iattr, curattr in enumerate(curpoint):
                cl_attr_freq[clust][iattr][curattr] += 1
        # Perform an initial centroid update.
        for ik in range(n_clusters):
            for iattr in range(nattrs):
                centroids[ik, iattr] = _mode_from_dict(cl_attr_freq[ik][iattr])

        # _____ ITERATION _____
        if verbose:
            print("Starting iterations...")
        itr = 0
        converged = False
        cost = np.Inf
        while itr <= max_iter and not converged:
            itr += 1
            centroids, moves = \
                _k_modes_iter(X, centroids, cl_attr_freq, membership)
            # All points seen in this iteration
            labels, ncost = _labels_cost_kmodes(X, centroids)
            converged = (moves == 0) or (ncost >= cost)
            cost = ncost
            if verbose:
                print("Run {}, iteration: {}/{}, moves: {}, cost: {}"
                      .format(init_no + 1, itr, max_iter, moves, cost))

        # Store result of current run.
        all_centroids.append(centroids)
        all_labels.append(labels)
        all_costs.append(cost)

    best = np.argmin(all_costs)
    if n_init > 1 and verbose:
        print("Best run was number {}".format(best + 1))

    return all_centroids[best], all_labels[best], all_costs[best]


def _labels_cost_kprototypes(Xnum, Xcat, centroids, gamma):
    """Calculate labels and cost function given a matrix of points and
    a list of centroids for the k-prototypes algorithm.
    """

    npoints = Xnum.shape[0]
    cost = 0.
    labels = np.empty(npoints, dtype='int64')
    for ipoint in range(npoints):
        # Numerical cost = sum of Euclidean distances
        num_costs = _euclidean_dissim(centroids[0], Xnum[ipoint])
        cat_costs = _matching_dissim(centroids[1], Xcat[ipoint])
        # Gamma relates the categorical cost to the numerical cost.
        tot_costs = num_costs + gamma * cat_costs
        clust = np.argmin(tot_costs)
        labels[ipoint] = clust
        cost += tot_costs[clust]

    return labels, cost


def _k_prototypes_iter(Xnum, Xcat, centroids, cl_attr_sum, cl_attr_freq,
                       membership, gamma):
    """Single iteration of the k-prototypes algorithm"""
    moves = 0
    for ipoint in range(Xnum.shape[0]):
        clust = np.argmin(
            _euclidean_dissim(centroids[0], Xnum[ipoint]) +
            gamma * _matching_dissim(centroids[1], Xcat[ipoint]))
        if membership[clust, ipoint]:
            # Point is already in its right place.
            continue

        # Move point, and update old/new cluster frequencies and centroids.
        moves += 1
        old_clust = np.argwhere(membership[:, ipoint])[0][0]

        cl_attr_sum, membership = _move_point_num(
            Xnum[ipoint], ipoint, clust, old_clust, cl_attr_sum,
            membership)
        cl_attr_freq, membership = _move_point_cat(
            Xcat[ipoint], ipoint, clust, old_clust, cl_attr_freq,
            membership)

        # Update new and old centroids by choosing mean for numerical
        # and mode for categorical attributes.
        for iattr in range(len(Xnum[ipoint])):
            for curc in (clust, old_clust):
                if sum(membership[curc, :]):
                    centroids[0][curc, iattr] = \
                        cl_attr_sum[curc, iattr] / sum(membership[curc, :])
                else:
                    centroids[0][curc, iattr] = 0.
        for iattr in range(len(Xcat[ipoint])):
            for curc in (clust, old_clust):
                centroids[1][curc, iattr] = \
                    _mode_from_dict(cl_attr_freq[curc][iattr])

        # In case of an empty cluster, reinitialize with a random point
        # from largest cluster.
        if sum(membership[old_clust, :]) == 0:
            from_clust = membership.sum(axis=1).argmax()
            choices = \
                [ii for ii, ch in enumerate(membership[from_clust, :]) if ch]
            rindx = np.random.choice(choices)

            cl_attr_freq, membership = _move_point_num(
                Xnum[rindx], rindx, old_clust, from_clust, cl_attr_sum,
                membership)
            cl_attr_freq, membership = _move_point_cat(
                Xcat[rindx], rindx, old_clust, from_clust, cl_attr_freq,
                membership)

    return centroids, moves


def k_prototypes(X, n_clusters, gamma, init, n_init, max_iter, verbose):
    """k-prototypes algorithm"""

    assert len(X) == 2, "X should be a list of Xnum and Xcat arrays"
    # List where [0] = numerical part of centroid and
    # [1] = categorical part. Same for centroids.
    Xnum, Xcat = X
    # Convert to numpy arrays, if needed.
    Xnum = np.asanyarray(Xnum)
    Xcat = np.asanyarray(Xcat)
    nnumpoints, nnumattrs = Xnum.shape
    ncatpoints, ncatattrs = Xcat.shape
    assert nnumpoints == ncatpoints,\
        "Uneven number of numerical and categorical points"
    npoints = nnumpoints
    assert n_clusters < npoints, "More clusters than data points?"

    # Estimate a good value for gamma, which determines the weighing of
    # categorical values in clusters (see Huang [1997]).
    if gamma is None:
        gamma = 0.5 * Xnum.std()

    all_centroids = []
    all_labels = []
    all_costs = []
    for init_no in range(n_init):

        # For numerical part of initialization, we don't have a guarantee
        # that there is not an empty cluster, so we need to retry until
        # there is none.
        while True:
            # _____ INIT _____
            if verbose:
                print("Init: initializing centroids")
            if init == 'Huang':
                centroids = _init_huang(Xcat, n_clusters)
            elif init == 'Cao':
                centroids = _init_cao(Xcat, n_clusters)
            elif init == 'random':
                seeds = np.random.choice(range(npoints), n_clusters)
                centroids = Xcat[seeds]
            elif hasattr(init, '__array__'):
                centroids = init
            else:
                raise NotImplementedError

            # Numerical is initialized by drawing from normal distribution,
            # categorical following the k-modes methods.
            meanX = np.mean(Xnum, axis=0)
            stdX = np.std(Xnum, axis=0)
            centroids = [meanX + np.random.randn(n_clusters, nnumattrs) * stdX,
                         centroids]

            if verbose:
                print("Init: initializing clusters")
            membership = np.zeros((n_clusters, npoints), dtype='int64')
            # Keep track of the sum of attribute values per cluster so that we
            # can do k-means on the numerical attributes.
            cl_attr_sum = np.zeros((n_clusters, nnumattrs), dtype='float')
            # cl_attr_freq is a list of lists with dictionaries that contain
            # the frequencies of values per cluster and attribute.
            cl_attr_freq = [[defaultdict(int) for _ in range(ncatattrs)]
                            for _ in range(n_clusters)]
            for ipoint in range(npoints):
                # Initial assignment to clusters
                clust = np.argmin(
                    _euclidean_dissim(centroids[0], Xnum[ipoint]) +
                    gamma * _matching_dissim(centroids[1], Xcat[ipoint]))
                membership[clust, ipoint] = 1
                # Count attribute values per cluster.
                for iattr, curattr in enumerate(Xnum[ipoint]):
                    cl_attr_sum[clust, iattr] += curattr
                for iattr, curattr in enumerate(Xcat[ipoint]):
                    cl_attr_freq[clust][iattr][curattr] += 1

            # If no empty clusters, then consider initialization finalized.
            if membership.sum(axis=1).min() > 0:
                break

        # Perform an initial centroid update.
        for ik in range(n_clusters):
            for iattr in range(nnumattrs):
                centroids[0][ik, iattr] =  \
                    cl_attr_sum[ik, iattr] / sum(membership[ik, :])
            for iattr in range(ncatattrs):
                centroids[1][ik, iattr] = \
                    _mode_from_dict(cl_attr_freq[ik][iattr])

        # _____ ITERATION _____
        if verbose:
            print("Starting iterations...")
        itr = 0
        converged = False
        cost = np.Inf
        while itr <= max_iter and not converged:
            itr += 1
            centroids, moves = _k_prototypes_iter(
                Xnum, Xcat, centroids, cl_attr_sum, cl_attr_freq,
                membership, gamma)

            # All points seen in this iteration
            labels, ncost = \
                _labels_cost_kprototypes(Xnum, Xcat, centroids, gamma)
            converged = (moves == 0) or (ncost >= cost)
            cost = ncost
            if verbose:
                print("Run: {}, iteration: {}/{}, moves: {}, ncost: {}"
                      .format(init_no + 1, itr, max_iter, moves, ncost))

        # Store results of current run.
        all_centroids.append(centroids)
        all_labels.append(labels)
        all_costs.append(cost)

    best = np.argmin(all_costs)
    if n_init > 1 and verbose:
        print("Best run was number {}".format(best + 1))

    return all_centroids[best], all_labels[best], all_costs[best]


class KModes(object):

    """k-modes clustering algorithm for categorical data.

    Parameters
    -----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    max_iter : int, default: 300
        Maximum number of iterations of the k-modes algorithm for a
        single run.

    n_init : int, default: 10
        Number of time the k-modes algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of cost.

    init : {'Huang', 'Cao', 'random' or an ndarray}
        Method for initialization:
        'Huang': Method in Huang [1997, 1998]
        'Cao': Method in Cao et al. [2009]
        'random': choose k observations (rows) at random from data for
        the initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centroids.

    verbose : boolean, optional
        Verbosity mode.

    Attributes
    ----------
    cluster_centroids_ : array, [n_clusters, n_features]
        Categories of cluster centroids

    labels_ :
        Labels of each point

    cost_ : float
        Clustering cost, defined as the sum distance of all points to
        their respective cluster centroids.

    Notes
    -----
    See:
    Huang, Z.: Extensions to the k-modes algorithm for clustering large
    data sets with categorical values, Data Mining and Knowledge
    Discovery 2(3), 1998.

    """

    def __init__(self, n_clusters=8, init='Cao', n_init=10, max_iter=100,
                 verbose=0):

        if hasattr(init, '__array__'):
            n_clusters = init.shape[0]
            init = np.asarray(init, dtype=np.float64)

        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.verbose = verbose
        if (self.init == 'Cao' or hasattr(self.init, '__array__')) and \
                self.n_init > 1:
            if self.verbose:
                print("Initialization method and algorithm are deterministic. "
                      "Setting n_init to 1.")
            self.n_init = 1

        self.max_iter = max_iter

    def fit(self, X):
        """Compute k-modes clustering.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
        """

        self.cluster_centroids_, self.labels_, self.cost_ = \
            k_modes(X, self.n_clusters, self.init, self.n_init,
                    self.max_iter, self.verbose)
        return self

    def fit_predict(self, X, y=None):
        """Compute cluster centroids and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        """
        return self.fit(X).labels_

    def predict(self, X):
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
        assert hasattr(self, 'cluster_centroids_'), "Model not yet fitted."
        return self._labels_cost_kmodes(X, self.cluster_centroids_)[0]


class KPrototypes(KModes):
    """k-protoypes clustering algorithm for mixed numerical/categorical data.

    Parameters
    -----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    gamma : float, default: None
        Weighing factor that determines relative importance of numerical vs.
        categorical attributes (see discussion in Huang [1997]). By default,
        automatically calculated from data.

    max_iter : int, default: 300
        Maximum number of iterations of the k-modes algorithm for a
        single run.

    n_init : int, default: 10
        Number of time the k-modes algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of cost.

    init : {'Huang', 'Cao', 'random' or an ndarray}
        Method for initialization:
        'Huang': Method in Huang [1997, 1998]
        'Cao': Method in Cao et al. [2009]
        'random': choose k observations (rows) at random from data for
        the initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centroids.

    verbose : boolean, optional
        Verbosity mode.

    Attributes
    ----------
    cluster_centroids_ : array, [n_clusters, n_features]
        Categories of cluster centroids

    labels_ :
        Labels of each point

    cost_ : float
        Clustering cost, defined as the sum distance of all points to
        their respective cluster centroids.

    Notes
    -----
    See:
    Huang, Z.: Extensions to the k-modes algorithm for clustering large
    data sets with categorical values, Data Mining and Knowledge
    Discovery 2(3), 1998.

    """

    def __init__(self, n_clusters=8, gamma=None, init='Huang', n_init=10,
                 max_iter=100, verbose=0):

        super(KPrototypes, self).__init__(n_clusters, init, n_init, max_iter,
                                    verbose)

        self.gamma = gamma

    def fit(self, X):
        """Compute k-prototypes clustering.

        Parameters
        ----------
        X : list of array-like, shape=[[n_num_samples, n_features],
                                       [n_cat_samples, n_features]]
        """

        self.cluster_centroids_, self.labels_, self.cost_ = \
            k_prototypes(X, self.n_clusters, self.gamma, self.init,
                         self.n_init, self.max_iter, self.verbose)
        return self

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : list of array-like, shape=[[n_num_samples, n_features],
                                       [n_cat_samples, n_features]]

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        assert hasattr(self, 'cluster_centroids_'), "Model not yet fitted."
        return self._labels_cost_kprototypes(X, self.cluster_centroids_)[0]
