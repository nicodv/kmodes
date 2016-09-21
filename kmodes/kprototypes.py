"""
K-prototypes clustering for mixed categorical and numerical data
"""

# Author: 'Nico de Vos' <njdevos@gmail.com>
# License: MIT

# pylint: disable=super-on-old-class,unused-argument,attribute-defined-outside-init

from collections import defaultdict

import numpy as np
from scipy import sparse
from sklearn.utils.validation import check_array

from . import kmodes
from .util import get_max_value_key, encode_features, get_unique_rows
from .util.dissim import matching_dissim, euclidean_dissim

# Number of tries we give the initialization methods to find non-empty
# clusters before we switch to random initialization.
MAX_INIT_TRIES = 20
# Number of tries we give the initialization before we raise an
# initialization error.
RAISE_INIT_TRIES = 100


def move_point_num(point, ipoint, to_clust, from_clust, cl_attr_sum, membship):
    """Move point between clusters, numerical attributes."""
    membship[to_clust, ipoint] = 1
    membship[from_clust, ipoint] = 0
    # Update sum of attributes in cluster.
    for iattr, curattr in enumerate(point):
        cl_attr_sum[to_clust][iattr] += curattr
        cl_attr_sum[from_clust][iattr] -= curattr
    return cl_attr_sum, membship


def _split_num_cat(X, categorical):
    """Extract numerical and categorical columns.
    Convert to numpy arrays, if needed.

    :param X: Feature matrix
    :param categorical: Indices of categorical columns
    """
    Xnum = np.asanyarray(X[:, [ii for ii in range(X.shape[1])
                               if ii not in categorical]]).astype(np.float64)
    Xcat = np.asanyarray(X[:, categorical])
    return Xnum, Xcat


def _labels_cost(Xnum, Xcat, centroids, num_dissim, cat_dissim, gamma):
    """Calculate labels and cost function given a matrix of points and
    a list of centroids for the k-prototypes algorithm.
    """

    npoints = Xnum.shape[0]
    Xnum = check_array(Xnum)

    cost = 0.
    labels = np.empty(npoints, dtype=np.uint8)
    for ipoint in range(npoints):
        # Numerical cost = sum of Euclidean distances
        num_costs = num_dissim(centroids[0], Xnum[ipoint])
        cat_costs = cat_dissim(centroids[1], Xcat[ipoint])
        # Gamma relates the categorical cost to the numerical cost.
        tot_costs = num_costs + gamma * cat_costs
        clust = np.argmin(tot_costs)
        labels[ipoint] = clust
        cost += tot_costs[clust]

    return labels, cost


def _k_prototypes_iter(Xnum, Xcat, centroids, cl_attr_sum, cl_attr_freq,
                       membship, num_dissim, cat_dissim, gamma):
    """Single iteration of the k-prototypes algorithm"""
    moves = 0
    for ipoint in range(Xnum.shape[0]):
        clust = np.argmin(
            num_dissim(centroids[0], Xnum[ipoint]) +
            gamma * cat_dissim(centroids[1], Xcat[ipoint])
        )
        if membship[clust, ipoint]:
            # Point is already in its right place.
            continue

        # Move point, and update old/new cluster frequencies and centroids.
        moves += 1
        old_clust = np.argwhere(membship[:, ipoint])[0][0]

        cl_attr_sum, membship = move_point_num(
            Xnum[ipoint], ipoint, clust, old_clust, cl_attr_sum, membship
        )
        cl_attr_freq, membship, centroids[1] = kmodes.move_point_cat(
            Xcat[ipoint], ipoint, clust, old_clust,
            cl_attr_freq, membship, centroids[1]
        )

        # Update old and new centroids for numerical attributes using the mean
        # of all values
        for iattr in range(len(Xnum[ipoint])):
            for curc in (clust, old_clust):
                if sum(membship[curc, :]):
                    centroids[0][curc, iattr] = \
                        cl_attr_sum[curc, iattr] / sum(membship[curc, :])
                else:
                    centroids[0][curc, iattr] = 0.

        # In case of an empty cluster, reinitialize with a random point
        # from largest cluster.
        if sum(membship[old_clust, :]) == 0:
            from_clust = membship.sum(axis=1).argmax()
            choices = \
                [ii for ii, ch in enumerate(membship[from_clust, :]) if ch]
            rindx = np.random.choice(choices)

            cl_attr_sum, membship = move_point_num(
                Xnum[rindx], rindx, old_clust, from_clust, cl_attr_sum, membship
            )
            cl_attr_freq, membship, centroids[1] = kmodes.move_point_cat(
                Xcat[rindx], rindx, old_clust, from_clust,
                cl_attr_freq, membship, centroids[1]
            )

    return centroids, moves


def k_prototypes(X, categorical, n_clusters, max_iter, num_dissim, cat_dissim,
                 gamma, init, n_init, verbose):
    """k-prototypes algorithm"""

    if sparse.issparse(X):
        raise TypeError("k-prototypes does not support sparse data.")

    if categorical is None or not categorical:
        raise NotImplementedError(
            "No categorical data selected, effectively doing k-means. "
            "Present a list of categorical columns, or use scikit-learn's "
            "KMeans instead."
        )
    if isinstance(categorical, int):
        categorical = [categorical]
    assert len(categorical) != X.shape[1], \
        "All columns are categorical, use k-modes instead of k-prototypes."
    assert max(categorical) < X.shape[1], \
        "Categorical index larger than number of columns."

    ncatattrs = len(categorical)
    nnumattrs = X.shape[1] - ncatattrs
    npoints = X.shape[0]
    assert n_clusters <= npoints, "More clusters than data points?"

    Xnum, Xcat = _split_num_cat(X, categorical)
    Xnum, Xcat = check_array(Xnum), check_array(Xcat, dtype=None)

    # Convert the categorical values in Xcat to integers for speed.
    # Based on the unique values in Xcat, we can make a mapping to achieve this.
    Xcat, enc_map = encode_features(Xcat)

    # Are there more n_clusters than unique rows? Then set the unique
    # rows as initial values and skip iteration.
    unique = get_unique_rows(X)
    n_unique = unique.shape[0]
    if n_unique <= n_clusters:
        max_iter = 0
        n_init = 1
        n_clusters = n_unique
        init = list(_split_num_cat(unique, categorical))
        init[1], _ = encode_features(init[1], enc_map)

    # Estimate a good value for gamma, which determines the weighing of
    # categorical values in clusters (see Huang [1997]).
    if gamma is None:
        gamma = 0.5 * Xnum.std()

    all_centroids = []
    all_labels = []
    all_costs = []
    all_n_iters = []
    for init_no in range(n_init):

        # For numerical part of initialization, we don't have a guarantee
        # that there is not an empty cluster, so we need to retry until
        # there is none.
        init_tries = 0
        while True:
            init_tries += 1
            # _____ INIT _____
            if verbose:
                print("Init: initializing centroids")
            if isinstance(init, str) and init == 'Huang':
                centroids = kmodes.init_huang(Xcat, n_clusters, cat_dissim)
            elif isinstance(init, str) and init == 'Cao':
                centroids = kmodes.init_cao(Xcat, n_clusters, cat_dissim)
            elif isinstance(init, str) and init == 'random':
                seeds = np.random.choice(range(npoints), n_clusters)
                centroids = Xcat[seeds]
            elif isinstance(init, list):
                assert init[0].shape[0] == n_clusters, \
                    "Incorrect number of initial numerical centroids in init."
                assert init[0].shape[1] == nnumattrs, \
                    "Incorrect number of numerical attributes in init"
                assert init[1].shape[0] == n_clusters, \
                    "Incorrect number of initial categorical centroids in init."
                assert init[1].shape[1] == ncatattrs, \
                    "Incorrect number of categorical attributes in init"
                centroids = [np.asarray(init[0], dtype=np.float64),
                             np.asarray(init[1], dtype=np.uint8)]
            else:
                raise NotImplementedError("Initialization method not supported.")

            if not isinstance(init, list):
                # Numerical is initialized by drawing from normal distribution,
                # categorical following the k-modes methods.
                meanx = np.mean(Xnum, axis=0)
                stdx = np.std(Xnum, axis=0)
                centroids = [
                    meanx + np.random.randn(n_clusters, nnumattrs) * stdx,
                    centroids
                ]

            if verbose:
                print("Init: initializing clusters")
            membship = np.zeros((n_clusters, npoints), dtype=np.uint8)
            # Keep track of the sum of attribute values per cluster so that we
            # can do k-means on the numerical attributes.
            cl_attr_sum = np.zeros((n_clusters, nnumattrs), dtype=np.float64)
            # cl_attr_freq is a list of lists with dictionaries that contain
            # the frequencies of values per cluster and attribute.
            cl_attr_freq = [[defaultdict(int) for _ in range(ncatattrs)]
                            for _ in range(n_clusters)]
            for ipoint in range(npoints):
                # Initial assignment to clusters
                clust = np.argmin(
                    num_dissim(centroids[0], Xnum[ipoint]) +
                    gamma * cat_dissim(centroids[1], Xcat[ipoint])
                )
                membship[clust, ipoint] = 1
                # Count attribute values per cluster.
                for iattr, curattr in enumerate(Xnum[ipoint]):
                    cl_attr_sum[clust, iattr] += curattr
                for iattr, curattr in enumerate(Xcat[ipoint]):
                    cl_attr_freq[clust][iattr][curattr] += 1

            # If no empty clusters, then consider initialization finalized.
            if membship.sum(axis=1).min() > 0:
                break

            if init_tries == MAX_INIT_TRIES:
                # Could not get rid of empty clusters. Randomly
                # initialize instead.
                init = 'random'
            elif init_tries == RAISE_INIT_TRIES:
                raise ValueError(
                    "Clustering algorithm could not initialize. "
                    "Consider assigning the initial clusters manually."
                )

        # Perform an initial centroid update.
        for ik in range(n_clusters):
            for iattr in range(nnumattrs):
                centroids[0][ik, iattr] = \
                    cl_attr_sum[ik, iattr] / sum(membship[ik, :])
            for iattr in range(ncatattrs):
                centroids[1][ik, iattr] = \
                    get_max_value_key(cl_attr_freq[ik][iattr])

        # _____ ITERATION _____
        if verbose:
            print("Starting iterations...")
        itr = 0
        converged = False
        cost = np.Inf
        while itr <= max_iter and not converged:
            itr += 1
            centroids, moves = _k_prototypes_iter(Xnum, Xcat, centroids,
                                                  cl_attr_sum, cl_attr_freq,
                                                  membship, num_dissim, cat_dissim, gamma)

            # All points seen in this iteration
            labels, ncost = _labels_cost(Xnum, Xcat, centroids,
                                         num_dissim, cat_dissim, gamma)
            converged = (moves == 0) or (ncost >= cost)
            cost = ncost
            if verbose:
                print("Run: {}, iteration: {}/{}, moves: {}, ncost: {}"
                      .format(init_no + 1, itr, max_iter, moves, ncost))

        # Store results of current run.
        all_centroids.append(centroids)
        all_labels.append(labels)
        all_costs.append(cost)
        all_n_iters.append(itr)

    best = np.argmin(all_costs)
    if n_init > 1 and verbose:
        print("Best run was number {}".format(best + 1))

    # Note: return gamma in case it was automatically determined.
    return all_centroids[best], all_labels[best], all_costs[best], \
        all_n_iters[best], gamma, enc_map


class KPrototypes(kmodes.KModes):
    """k-protoypes clustering algorithm for mixed numerical/categorical data.

    Parameters
    -----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    max_iter : int, default: 300
        Maximum number of iterations of the k-modes algorithm for a
        single run.

    num_dissim : func, default: euclidian_dissim
        Dissimilarity function used by the algorithm for numerical variables.
        Defaults to the Euclidian dissimilarity function.

    cat_dissim : func, default: matching_dissim
        Dissimilarity function used by the algorithm for categorical variables.
        Defaults to the matching dissimilarity function.

    n_init : int, default: 10
        Number of time the k-modes algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of cost.

    init : {'Huang', 'Cao', 'random' or a list of ndarrays}
        Method for initialization:
        'Huang': Method in Huang [1997, 1998]
        'Cao': Method in Cao et al. [2009]
        'random': choose 'n_clusters' observations (rows) at random from
        data for the initial centroids.
        If a list of ndarrays is passed, it should be of length 2, with
        shapes (n_clusters, n_features) for numerical and categorical
        data respectively. These are the initial centroids.

    gamma : float, default: None
        Weighing factor that determines relative importance of numerical vs.
        categorical attributes (see discussion in Huang [1997]). By default,
        automatically calculated from data.

    verbose : integer, optional
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

    def __init__(self, n_clusters=8, max_iter=100, num_dissim=euclidean_dissim,
                 cat_dissim=matching_dissim, init='Huang', n_init=10, gamma=None,
                 verbose=0):

        super(KPrototypes, self).__init__(n_clusters, max_iter, cat_dissim,
                                          init, n_init, verbose)

        self.num_dissim = num_dissim
        self.gamma = gamma

    def fit(self, X, y=None, categorical=None):
        """Compute k-prototypes clustering.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
        categorical : Index of columns that contain categorical data
        """

        # If self.gamma is None, gamma will be automatically determined from
        # the data. The function below returns its value.
        self.cluster_centroids_, self.labels_, self.cost_, self.n_iter_, \
            self.gamma, self.enc_map_ = k_prototypes(X,
                                                     categorical,
                                                     self.n_clusters,
                                                     self.max_iter,
                                                     self.num_dissim,
                                                     self.cat_dissim,
                                                     self.gamma,
                                                     self.init,
                                                     self.n_init,
                                                     self.verbose)
        return self

    def predict(self, X, categorical=None):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            New data to predict.
        categorical : Index of columns that contain categorical data

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        assert hasattr(self, 'cluster_centroids_'), "Model not yet fitted."

        Xnum, Xcat = _split_num_cat(X, categorical)
        Xnum, Xcat = check_array(Xnum), check_array(Xcat, dtype=None)
        Xcat, _ = encode_features(Xcat, enc_map=self.enc_map_)
        return _labels_cost(Xnum, Xcat, self.cluster_centroids_,
                            self.num_dissim, self.cat_dissim, self.gamma)[0]
