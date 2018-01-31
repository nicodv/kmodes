"""
K-modes clustering for categorical data
"""

# pylint: disable=unused-argument,attribute-defined-outside-init

from collections import defaultdict

import numpy as np
from scipy import sparse
from matching.algorithms import extended_galeshapley
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array
from matching.algorithms import extended_galeshapley

from .util import get_max_value_key, encode_features, get_unique_rows, decode_centroids
from .util.dissim import matching_dissim, ng_dissim

def init_matching(X, n_clusters, dissim):
    """Initialise centroids according to Huang's method, where the random
    allocation of centroids to datapoints uses the extended Gale-Shapley
    algorithm to solve a capacitated matching game.

    Parameters
    ----------
    X : {array-like}, shape = (N, n_attrs)
        The dataset with N elements described by n_attrs attributes
    n_clusters : int
        The number of clusters to be found
    dissim : str
        The dissimilarity measure to be used

    Returns
    -------
    centroids : {array-like}, shape = (n_clusters, n_attrs)
        The initial centroids for the k-modes algorithm
    """
    n_attrs = X.shape[1]
    centroids = np.empty((n_clusters, n_attrs), dtype='object')

    # Follow Huang's method
    for i_attr in range(n_attrs):
        freq = defaultdict(int)
        for curr_attr in X[:, i_attr]:
            freq[curr_attr] += 1

        choices = [
            choice for choice, weight in freq.items() for _ in range(weight)
        ]
        choices = sorted(choices)
        centroids[:, i_attr] = np.random.choice(choices, n_clusters)

    # Set up preference dictionaries for suitors and reviewers, giving all
    # reviewers a capacity of 1.
    suitors = []
    suitor_pref_dict = {}
    reviewers = centroids
    reviewer_pref_dict = {}
    capacities = {tuple(r): 1 for r in reviewers}

    # Build our set of potential suitors
    for r in reviewers:
        sorted_idxs = np.argsort(dissim(X, r))
        suitors.append(X[sorted_idxs[:n_clusters]])

    suitors = np.concatenate(suitors)

    # Here we decide how to build the suitors' preference lists.
    for i in range(n_clusters ** 2):
        s = suitors[i]
        sorted_idxs = np.argsort(dissim(reviewers, s))
        suitor_pref_dict[tuple(s)] = reviewers[sorted_idxs].tolist()

    # Create preference lists for each reviewer.
    for r in reviewers:
        sorted_idxs = np.argsort(dissim(suitors, r))
        reviewer_pref_dict[tuple(r)] = suitors[sorted_idxs].tolist()

    solution = extended_galeshapley(suitor_pref_dict,
                                    reviewer_pref_dict,
                                    capacities)
    centroids = np.concatenate(list(solution.values()), axis=0).astype('object')

    return centroids


def init_huang(X, n_clusters, dissim):
    """Initialize centroids according to method by Huang [1997]."""
    n_attrs = X.shape[1]
    centroids = np.empty((n_clusters, n_attrs), dtype='object')
    # determine frequencies of attributes
    for iattr in range(n_attrs):
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
        # So that we are consistent between Python versions,
        # each with different dict ordering.
        choices = sorted(choices)
        centroids[:, iattr] = np.random.choice(choices, n_clusters)
    # The previously chosen centroids could result in empty clusters,
    # so set centroid to closest point in X.
    for ik in range(n_clusters):
        ndx = np.argsort(dissim(X, centroids[ik]))
        # We want the centroid to be unique.
        while np.all(X[ndx[0]] == centroids, axis=1).any():
            ndx = np.delete(ndx, 0)
        centroids[ik] = X[ndx[0]]

    return centroids


def init_cao(X, n_clusters, dissim):
    """Initialize centroids according to method by Cao et al. [2009].

    Note: O(N * attr * n_clusters**2), so watch out with large n_clusters
    """
    n_points, n_attrs = X.shape
    centroids = np.empty((n_clusters, n_attrs), dtype='object')
    # Method is based on determining density of points.
    dens = np.zeros(n_points)
    for iattr in range(n_attrs):
        freq = defaultdict(int)
        for val in X[:, iattr]:
            freq[val] += 1
        for ipoint in range(n_points):
            dens[ipoint] += freq[X[ipoint, iattr]] / float(n_attrs)
    dens /= n_points

    # Choose initial centroids based on distance and density.
    centroids[0] = X[np.argmax(dens)]
    if n_clusters > 1:
        # For the remaining centroids, choose maximum dens * dissim to the
        # (already assigned) centroid with the lowest dens * dissim.
        for ik in range(1, n_clusters):
            dd = np.empty((ik, n_points))
            for ikk in range(ik):
                dd[ikk] = dissim(X, centroids[ikk]) * dens
            centroids[ik] = X[np.argmax(np.min(dd, axis=0))]

    return centroids


def move_point_cat(point, ipoint, to_clust, from_clust, cl_attr_freq,
                   membship, centroids):
    """Move point between clusters, categorical attributes."""
    membship[to_clust, ipoint] = 1
    membship[from_clust, ipoint] = 0
    # Update frequencies of attributes in cluster.
    for iattr, curattr in enumerate(point):
        to_attr_counts = cl_attr_freq[to_clust][iattr]
        from_attr_counts = cl_attr_freq[from_clust][iattr]

        # Increment the attribute count for the new "to" cluster
        to_attr_counts[curattr] += 1

        current_attribute_value_freq = to_attr_counts[curattr]
        current_centroid_value = centroids[to_clust][iattr]
        current_centroid_freq = to_attr_counts[current_centroid_value]
        if current_centroid_freq < current_attribute_value_freq:
            # We have incremented this value to the new mode. Update the centroid.
            centroids[to_clust][iattr] = curattr

        # Decrement the attribute count for the old "from" cluster
        from_attr_counts[curattr] -= 1

        old_centroid_value = centroids[from_clust][iattr]
        if old_centroid_value == curattr:
            # We have just removed a count from the old centroid value. We need to
            # recalculate the centroid as it may no longer be the maximum
            centroids[from_clust][iattr] = get_max_value_key(from_attr_counts)

    return cl_attr_freq, membship, centroids


def _labels_cost(X, centroids, dissim, membship=None):
    """Calculate labels and cost function given a matrix of points and
    a list of centroids for the k-modes algorithm.
    """

    X = check_array(X)

    n_points = X.shape[0]
    cost = 0.
    labels = np.empty(n_points, dtype=np.uint8)
    for ipoint, curpoint in enumerate(X):
        diss = dissim(centroids, curpoint, X=X, membship=membship)
        clust = np.argmin(diss)
        labels[ipoint] = clust
        cost += diss[clust]

    return labels, cost


def _k_modes_iter(X, centroids, cl_attr_freq, membship, dissim):
    """Single iteration of k-modes clustering algorithm"""
    moves = 0
    for ipoint, curpoint in enumerate(X):
        clust = np.argmin(dissim(centroids, curpoint, X=X, membship=membship))
        if membship[clust, ipoint]:
            # Point is already in its right place.
            continue

        # Move point, and update old/new cluster frequencies and centroids.
        moves += 1
        old_clust = np.argwhere(membship[:, ipoint])[0][0]

        cl_attr_freq, membship, centroids = move_point_cat(
            curpoint, ipoint, clust, old_clust, cl_attr_freq, membship, centroids
        )

        # In case of an empty cluster, reinitialize with a random point
        # from the largest cluster.
        if not membship[old_clust, :].any():
            from_clust = membship.sum(axis=1).argmax()
            choices = [ii for ii, ch in enumerate(membship[from_clust, :]) if ch]
            rindx = np.random.choice(choices)

            cl_attr_freq, membship, centroids = move_point_cat(
                X[rindx], rindx, old_clust, from_clust, cl_attr_freq, membship, centroids
            )

    return centroids, moves


def k_modes(X, n_clusters, max_iter, dissim, init, n_init, verbose):
    """k-modes algorithm"""

    if sparse.issparse(X):
        raise TypeError("k-modes does not support sparse data.")

    # Convert pandas objects to numpy arrays.
    if 'pandas' in str(X.__class__):
        X = X.values

    X = check_array(X, dtype=None)

    # Convert the categorical values in X to integers for speed.
    # Based on the unique values in X, we can make a mapping to achieve this.
    X, enc_map = encode_features(X)

    n_points, n_attrs = X.shape
    assert n_clusters <= n_points, "Cannot have more clusters ({}) " \
                                   "than data points ({}).".format(n_clusters, n_points)

    # Are there more n_clusters than unique rows? Then set the unique
    # rows as initial values and skip iteration.
    unique = get_unique_rows(X)
    n_unique = unique.shape[0]
    if n_unique <= n_clusters:
        max_iter = 0
        n_init = 1
        n_clusters = n_unique
        init = unique

    all_centroids = []
    all_labels = []
    all_costs = []
    all_n_iters = []
    for init_no in range(n_init):

        # _____ INIT _____
        if verbose:
            print("Init: initializing centroids")
        if isinstance(init, str) and init.lower() == 'matching':
            centroids = init_matching(X, n_clusters, dissim)
        elif isinstance(init, str) and init.lower() == 'huang':
            centroids = init_huang(X, n_clusters, dissim)
        elif isinstance(init, str) and init.lower() == 'cao':
            centroids = init_cao(X, n_clusters, dissim)
        elif isinstance(init, str) and init.lower() == 'random':
            seeds = np.random.choice(range(n_points), n_clusters)
            centroids = X[seeds]
        elif hasattr(init, '__array__'):
            # Make sure init is a 2D array.
            if len(init.shape) == 1:
                init = np.atleast_2d(init).T
            assert init.shape[0] == n_clusters, \
                "Wrong number of initial centroids in init ({}, should be {})."\
                .format(init.shape[0], n_clusters)
            assert init.shape[1] == n_attrs, \
                "Wrong number of attributes in init ({}, should be {})."\
                .format(init.shape[1], n_attrs)
            centroids = np.asarray(init, dtype=np.uint8)
        else:
            raise NotImplementedError

        if verbose:
            print("Init: initializing clusters")
        membship = np.zeros((n_clusters, n_points), dtype=np.uint8)
        # cl_attr_freq is a list of lists with dictionaries that contain the
        # frequencies of values per cluster and attribute.
        cl_attr_freq = [[defaultdict(int) for _ in range(n_attrs)]
                        for _ in range(n_clusters)]
        for ipoint, curpoint in enumerate(X):
            # Initial assignment to clusters
            clust = np.argmin(dissim(centroids, curpoint, X=X, membship=membship))
            membship[clust, ipoint] = 1
            # Count attribute values per cluster.
            for iattr, curattr in enumerate(curpoint):
                cl_attr_freq[clust][iattr][curattr] += 1
        # Perform an initial centroid update.
        for ik in range(n_clusters):
            for iattr in range(n_attrs):
                if sum(membship[ik]) == 0:
                    # Empty centroid, choose randomly
                    centroids[ik, iattr] = np.random.choice(X[:, iattr])
                else:
                    centroids[ik, iattr] = get_max_value_key(cl_attr_freq[ik][iattr])

        # _____ ITERATION _____
        if verbose:
            print("Starting iterations...")
        itr = 0
        converged = False
        cost = np.Inf
        while itr <= max_iter and not converged:
            itr += 1
            centroids, moves = _k_modes_iter(X, centroids, cl_attr_freq, membship, dissim)
            # All points seen in this iteration
            labels, ncost = _labels_cost(X, centroids, dissim, membship)
            converged = (moves == 0) or (ncost >= cost)
            cost = ncost
            if verbose:
                print("Run {}, iteration: {}/{}, moves: {}, cost: {}"
                      .format(init_no + 1, itr, max_iter, moves, cost))

        # Store result of current run.
        all_centroids.append(centroids)
        all_labels.append(labels)
        all_costs.append(cost)
        all_n_iters.append(itr)

    best = np.argmin(all_costs)
    if n_init > 1 and verbose:
        print("Best run was number {}".format(best + 1))

    return all_centroids[best], enc_map, all_labels[best], \
        all_costs[best], all_n_iters[best]


class KModes(BaseEstimator, ClusterMixin):

    """k-modes clustering algorithm for categorical data.

    Parameters
    -----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    max_iter : int, default: 300
        Maximum number of iterations of the k-modes algorithm for a
        single run.

    cat_dissim : func, default: matching_dissim
        Dissimilarity function used by the k-modes algorithm for categorical variables.
        Defaults to the matching dissimilarity function.

    init : {'Huang', 'Cao', 'random', 'matching' or an ndarray}, default: 'Cao'
        Method for initialization:
        'Huang': Method in Huang [1997, 1998]
        'Cao': Method in Cao et al. [2009]
        'random': choose 'n_clusters' observations (rows) at random from
        data for the initial centroids.
        'matching': Follow Huang's method and assign centroids to points in the
        set by considering the situation as a capacitated matching game to be
        solved with an extension of the Gale-Shapley algorithm.
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centroids.

    n_init : int, default: 10
        Number of time the k-modes algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of cost.

    verbose : int, optional
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

    n_iter_ : int
        The number of iterations the algorithm ran for.

    Notes
    -----
    See:
    Huang, Z.: Extensions to the k-modes algorithm for clustering large
    data sets with categorical values, Data Mining and Knowledge
    Discovery 2(3), 1998.

    """

    def __init__(self, n_clusters=8, max_iter=100, cat_dissim=matching_dissim,
                 init='Cao', n_init=1, verbose=0):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.cat_dissim = cat_dissim
        self.init = init
        self.n_init = n_init
        self.verbose = verbose
        if ((isinstance(self.init, str) and self.init == 'Cao') or
                hasattr(self.init, '__array__')) and self.n_init > 1:
            if self.verbose:
                print("Initialization method and algorithm are deterministic. "
                      "Setting n_init to 1.")
            self.n_init = 1

    def fit(self, X, y=None, **kwargs):
        """Compute k-modes clustering.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
        """

        self._enc_cluster_centroids, self._enc_map, self.labels_,\
            self.cost_, self.n_iter_ = k_modes(X,
                                               self.n_clusters,
                                               self.max_iter,
                                               self.cat_dissim,
                                               self.init,
                                               self.n_init,
                                               self.verbose)
        return self

    def fit_predict(self, X, y=None, **kwargs):
        """Compute cluster centroids and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        """
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

        X = check_array(X, dtype=None)
        X, _ = encode_features(X, enc_map=self._enc_map)
        return _labels_cost(X, self._enc_cluster_centroids, self.cat_dissim)[0]

    @property
    def cluster_centroids_(self):
        if hasattr(self, '_enc_cluster_centroids'):
            return decode_centroids(self._enc_cluster_centroids, self._enc_map)
        else:
            raise AttributeError("'{}' object has no attribute 'cluster_centroids_' "
                                 "because the model is not yet fitted.")
