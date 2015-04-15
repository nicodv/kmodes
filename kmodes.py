#!/usr/bin/env python

__author__ = 'Nico de Vos'
__email__ = 'njdevos@gmail.com'
__license__ = 'MIT'
__version__ = '0.8'

import random
import numpy as np
from collections import defaultdict


# noinspection PyNoneFunctionAssignment,PyTypeChecker,PyUnresolvedReferences
class KModes(object):

    def __init__(self, k):
        """k-modes clustering algorithm for categorical data.
        See:
        Huang, Z.: Extensions to the k-modes algorithm for clustering large data sets with
        categorical values, Data Mining and Knowledge Discovery 2(3), 1998.

        Inputs:     k           = number of clusters
        Attributes: clusters    = cluster numbers [no. points]
                    centroids   = centroids [k * no. attributes]
                    membership  = membership matrix [k * no. points]
                    cost        = clustering cost, defined as the sum distance of
                                  all points to their respective clusters

        """
        assert k > 1, "Choose at least 2 clusters."
        self.k = k

        # generalized form with alpha. alpha > 1 for fuzzy k-modes
        self.alpha = 1

        # init some variables
        self.membership = self.clusters = self.centroids = self.cost = None

    def cluster(self, x, pre_runs=10, pre_pctl=20, *args, **kwargs):
        """Shell around _perform_clustering method that tries to ensure a good clustering
        result by choosing one that has a relatively low clustering cost compared to the
        costs of a number of pre-runs. (Huang [1998] states that clustering cost can be
        used to judge the clustering quality.)

        """

        if pre_runs and 'init_method' in kwargs and kwargs['init_method'] == 'Cao':
            print("Initialization method and algorithm are deterministic. Disabling preruns...")
            pre_runs = None

        if pre_runs:
            precosts = np.empty(pre_runs)
            for pr in range(pre_runs):
                self._perform_clustering(x, *args, verbose=0, **kwargs)
                precosts[pr] = self.cost
                print("Prerun {0} / {1}, Cost = {2}".format(pr + 1, pre_runs, precosts[pr]))
            goodcost = np.percentile(precosts, pre_pctl)
        else:
            goodcost = np.inf

        while True:
            self._perform_clustering(x, *args, verbose=1, **kwargs)
            if self.cost <= goodcost:
                break

    def _perform_clustering(self, x, init_method='Huang', max_iters=100, verbose=1):
        """Inputs:  x           = data points [no. points * no. attributes]
                    init_method = initialization method ('Huang' for the one described in
                                  Huang [1998], 'Cao' for the one in Cao et al. [2009])
                    max_iters   = maximum no. of iterations
                    verbose     = 0 for no and 1 for normal algorithm progress information,
                                  2 for internal algorithm details

        """
        # convert to numpy array, if needed
        x = np.asanyarray(x)
        npoints, nattrs = x.shape
        assert self.k < npoints, "More clusters than data points?"

        self.initMethod = init_method

        # ----------------------
        #    INIT
        # ----------------------
        if verbose:
            print("Init: initializing centroids")
        self.init_centroids(x)

        if verbose:
            print("Init: initializing clusters")
        self.membership = np.zeros((self.k, npoints), dtype='int64')
        # self._clustAttrFreq is a list of lists with dictionaries that contain the
        # frequencies of values per cluster and attribute
        self._clustAttrFreq = [[defaultdict(int) for _ in range(nattrs)] for _ in range(self.k)]
        for ipoint, curpoint in enumerate(x):
            # initial assigns to clusters
            cluster = np.argmin(self.get_dissim(self.centroids, curpoint))
            self.membership[cluster, ipoint] = 1
            # count attribute values per cluster
            for iattr, curattr in enumerate(curpoint):
                self._clustAttrFreq[cluster][iattr][curattr] += 1
        # perform an initial centroid update
        for ik in range(self.k):
            for iattr in range(nattrs):
                self.centroids[ik, iattr] = self.get_mode(self._clustAttrFreq[ik][iattr])

        # ----------------------
        #    ITERATION
        # ----------------------
        if verbose:
            print("Starting iterations...")
        itr = 0
        converged = False
        while itr <= max_iters and not converged:
            itr += 1
            moves = 0
            for ipoint, curpoint in enumerate(x):
                cluster = np.argmin(self.get_dissim(self.centroids, curpoint))
                # if necessary: move point, and update old/new cluster frequencies and centroids
                if not self.membership[cluster, ipoint]:
                    moves += 1
                    oldcluster = np.argwhere(self.membership[:, ipoint])[0][0]
                    self._add_point_to_cluster(curpoint, ipoint, cluster)
                    self._remove_point_from_cluster(curpoint, ipoint, oldcluster)
                    # update new and old centroids by choosing most likely attribute
                    for iattr, curattr in enumerate(curpoint):
                        for curc in (cluster, oldcluster):
                            self.centroids[curc, iattr] = self.get_mode(
                                self._clustAttrFreq[curc][iattr])
                    if verbose == 2:
                        print("Move from cluster {0} to {1}".format(oldcluster, cluster))

                        # in case of an empty cluster, reinitialize with a random point
                        # that is not a centroid
                        if sum(self.membership[oldcluster, :]) == 0:
                            while True:
                                rindx = np.random.randint(npoints)
                                if not np.all(x[rindx] == self.centroids).any():
                                    break
                            self._add_point_to_cluster(x[rindx], rindx, oldcluster)
                            fromcluster = np.argwhere(self.membership[:, rindx])[0][0]
                            self._remove_point_from_cluster(x[rindx], rindx, fromcluster)

            # all points seen in this iteration
            converged = (moves == 0)
            if verbose:
                print("Iteration: {0}/{1}, moves: {2}".format(itr, max_iters, moves))

        self.calculate_clustering_cost(x)
        self.clusters = np.array([np.argwhere(self.membership[:, pt])[0] for pt in range(npoints)])

    def init_centroids(self, x):
        assert self.initMethod in ('Huang', 'Cao')
        npoints, nattrs = x.shape
        self.centroids = np.empty((self.k, nattrs))
        if self.initMethod == 'Huang':
            # determine frequencies of attributes
            for iattr in range(nattrs):
                freq = defaultdict(int)
                for curattr in x[:, iattr]:
                    freq[curattr] += 1
                # sample centroids using the probabilities of attributes
                # (I assume that's what's meant in the Huang [1998] paper; it works, at least)
                # note: sampling using population in static list with as many choices as
                # frequency counts this works well since (1) we re-use the list k times here,
                # and (2) the counts are small integers so memory consumption is low
                choices = [chc for chc, wght in freq.items() for _ in range(wght)]
                for ik in range(self.k):
                    self.centroids[ik, iattr] = random.choice(choices)
            # the previously chosen centroids could result in empty clusters,
            # so set centroid to closest point in x
            for ik in range(self.k):
                ndx = np.argsort(self.get_dissim(x, self.centroids[ik]))
                # and we want the centroid to be unique
                while np.all(x[ndx[0]] == self.centroids, axis=1).any():
                    ndx = np.delete(ndx, 0)
                self.centroids[ik] = x[ndx[0]]

        elif self.initMethod == 'Cao':
            # Note: O(N * at * k**2), so watch out with k
            # determine densities points
            dens = np.zeros(npoints)
            for iattr in range(nattrs):
                freq = defaultdict(int)
                for val in x[:, iattr]:
                    freq[val] += 1
                for ipoint in range(npoints):
                    dens[ipoint] += freq[x[ipoint, iattr]] / float(nattrs)
            dens /= npoints

            # choose centroids based on distance and density
            self.centroids[0] = x[np.argmax(dens)]
            dissim = self.get_dissim(x, self.centroids[0])
            self.centroids[1] = x[np.argmax(dissim * dens)]
            # for the reamining centroids, choose max dens * dissim to the (already assigned)
            # centroid with the lowest dens * dissim
            for ik in range(2, self.k):
                dd = np.empty((ik, npoints))
                for ikk in range(ik):
                    dd[ikk] = self.get_dissim(x, self.centroids[ikk]) * dens
                self.centroids[ik] = x[np.argmax(np.min(dd, axis=0))]

        return

    def _add_point_to_cluster(self, point, ipoint, cluster):
        self.membership[cluster, ipoint] = 1
        # update frequencies of attributes in cluster
        for iattr, curattr in enumerate(point):
            self._clustAttrFreq[cluster][iattr][curattr] += 1
        return

    def _remove_point_from_cluster(self, point, ipoint, cluster):
        self.membership[cluster, ipoint] = 0
        # update frequencies of attributes in cluster
        for iattr, curattr in enumerate(point):
            self._clustAttrFreq[cluster][iattr][curattr] -= 1
        return

    @staticmethod
    def get_dissim(a, b):
        # simple matching dissimilarity
        return (a != b).sum(axis=1)

    @staticmethod
    def get_mode(dic):
        # Fast method (supposedly) to get key for maximum value in dict.
        v = list(dic.values())
        k = list(dic.keys())
        if len(v) == 0:
            pass
        return k[v.index(max(v))]

    def calculate_clustering_cost(self, x):
        self.cost = 0
        for ipoint, curpoint in enumerate(x):
            self.cost += np.sum(self.get_dissim(self.centroids, curpoint) *
                                (self.membership[:, ipoint] ** self.alpha))
        return


# noinspection PyUnresolvedReferences,PyTypeChecker,PyMethodOverriding
class KPrototypes(KModes):

    def __init__(self, k):
        """k-protoypes clustering algorithm for mixed numeric and categorical data.
        Huang, Z.: Clustering large data sets with mixed numeric and categorical values,
        Proceedings of the First Pacific Asia Knowledge Discovery and Data Mining Conference,
        Singapore, pp. 21-34, 1997.

        Inputs:     k           = number of clusters
        Attributes: clusters    = cluster numbers [no. points]
                    centroids   = centroids, two lists (num. and cat.) with [k * no. attributes]
                    membership  = membership matrix [k * no. points]
                    cost        = clustering cost, defined as the sum distance of
                                  all points to their respective clusters
                    gamma       = weighing factor that determines relative importance of
                                  num./cat. attributes (see discussion in Huang [1997])

        """
        super(KPrototypes, self).__init__(k)

        self.gamma = None

    def _perform_clustering(self, x, gamma=None, init_method='Huang', max_iters=100, verbose=1):
        """Inputs:  xnum        = numeric data points [no. points * no. numeric attributes]
                    xcat        = categorical data points [no. points * no. numeric attributes]
                    gamma       = weighing factor that determines relative importance of
                                  num./cat. attributes (see discussion in Huang [1997])
                    initMethod  = initialization method ('Huang' for the one described in
                                  Huang [1998], 'Cao' for the one in Cao et al. [2009])
                    max_iters   = maximum no. of iterations
                    verbose     = 0 for no and 1 for normal algorithm progress information,
                                  2 for internal algorithm details

        """
        # convert to numpy arrays, if needed
        xnum, xcat = x[0], x[1]
        xnum = np.asanyarray(xnum)
        xcat = np.asanyarray(xcat)
        nnumpoints, nnumattrs = xnum.shape
        ncatpoints, ncatattrs = xcat.shape
        assert nnumpoints == ncatpoints, "More numerical points than categorical?"
        npoints = nnumpoints
        assert self.k < npoints, "More clusters than data points?"

        self.initMethod = init_method

        # estimate a good value for gamma, which determines the weighing of
        # categorical values in clusters (see Huang [1997])
        if gamma is None:
            gamma = 0.5 * xnum.std()
        self.gamma = gamma

        # ----------------------
        #    INIT
        # ----------------------
        if verbose:
            print("Init: initializing centroids")
        # list where [0] = numerical part of centroid and [1] = categorical part
        self.init_centroids(xcat)
        self.centroids = [np.mean(xnum, axis=0) + np.random.randn(self.k, nnumattrs) *
                          np.std(xnum, axis=0), self.centroids]

        if verbose:
            print("Init: initializing clusters")
        self.membership = np.zeros((self.k, npoints), dtype='int64')
        # keep track of the sum of attribute values per cluster
        self._clustAttrSum = np.zeros((self.k, nnumattrs), dtype='float')
        # self._clustAttrFreq is a list of lists with dictionaries that contain
        # the frequencies of values per cluster and attribute
        self._clustAttrFreq = [[defaultdict(int) for _ in range(ncatattrs)] for _ in range(self.k)]
        for ipoint in range(npoints):
            # initial assigns to clusters
            cluster = np.argmin(self.get_dissim_num(self.centroids[0], xnum[ipoint]) +
                                self.gamma * self.get_dissim(self.centroids[1], xcat[ipoint]))
            self.membership[cluster, ipoint] = 1
            # count attribute values per cluster
            for iattr, curattr in enumerate(xnum[ipoint]):
                self._clustAttrSum[cluster, iattr] += curattr
            for iattr, curattr in enumerate(xcat[ipoint]):
                self._clustAttrFreq[cluster][iattr][curattr] += 1
        for ik in range(self.k):
            # in case of an empty cluster, reinitialize with a random point
            # that is not a centroid
            if sum(self.membership[ik, :]) == 0:
                while True:
                    rindex = np.random.randint(npoints)
                    if not np.all(np.vstack((np.all(xnum[rindex] == self.centroids[0], axis=1),
                                             np.all(xcat[rindex] == self.centroids[1], axis=1))),
                                  axis=0).any():
                        break
                self._add_point_to_cluster(xnum[rindex], xcat[rindex], rindex, ik)
                fromcluster = np.argwhere(self.membership[:, rindex])[0][0]
                self._remove_point_from_cluster(xnum[rindex], xcat[rindex], rindex, fromcluster)
        # perform an initial centroid update
        for ik in range(self.k):
            for iattr in range(nnumattrs):
                # TODO: occasionally "invalid value encountered in double_scalars" in following line
                self.centroids[0][ik, iattr] = \
                    self._clustAttrSum[ik, iattr] / sum(self.membership[ik, :])
            for iattr in range(ncatattrs):
                self.centroids[1][ik, iattr] = self.get_mode(self._clustAttrFreq[ik][iattr])

        # ----------------------
        #    ITERATION
        # ----------------------
        if verbose:
            print("Starting iterations...")
        itr = 0
        converged = False
        while itr <= max_iters and not converged:
            itr += 1
            moves = 0
            for ipoint in range(npoints):
                cluster = np.argmin(self.get_dissim_num(self.centroids[0], xnum[ipoint]) +
                                    self.gamma * self.get_dissim(self.centroids[1], xcat[ipoint]))
                # if necessary: move point, and update old/new cluster frequencies and centroids
                if not self.membership[cluster, ipoint]:
                    moves += 1
                    oldcluster = np.argwhere(self.membership[:, ipoint])[0][0]
                    self._add_point_to_cluster(xnum[ipoint], xcat[ipoint], ipoint, cluster)
                    self._remove_point_from_cluster(xnum[ipoint], xcat[ipoint], ipoint, oldcluster)
                    # update new and old centroids by choosing mean for numerical and
                    # most likely for categorical attributes
                    for iattr in range(len(xnum[ipoint])):
                        for curc in (cluster, oldcluster):
                            if sum(self.membership[curc, :]):
                                self.centroids[0][curc, iattr] = \
                                    self._clustAttrSum[curc, iattr] / sum(self.membership[curc, :])
                            else:
                                self.centroids[0][curc, iattr] = 0
                    for iattr in range(len(xcat[ipoint])):
                        for curc in (cluster, oldcluster):
                            self.centroids[1][curc, iattr] = \
                                self.get_mode(self._clustAttrFreq[curc][iattr])
                    if verbose == 2:
                        print("Move from cluster {0} to {1}".format(oldcluster, cluster))

                    # in case of an empty cluster, reinitialize with a random point
                    # that is not a centroid
                    if sum(self.membership[oldcluster, :]) == 0:
                        while True:
                            rindex = np.random.randint(npoints)
                            if not np.all(np.vstack((
                                    np.all(xnum[rindex] == self.centroids[0], axis=1),
                                    np.all(xcat[rindex] == self.centroids[1], axis=1))),
                                    axis=0).any():
                                break
                        self._add_point_to_cluster(xnum[rindex], xcat[rindex], rindex, oldcluster)
                        fromcluster = np.argwhere(self.membership[:, rindex])[0][0]
                        self._remove_point_from_cluster(
                            xnum[rindex], xcat[rindex], rindex, fromcluster)

            # all points seen in this iteration
            converged = (moves == 0)
            if verbose:
                print("Iteration: {0}/{1}, moves: {2}".format(itr, max_iters, moves))

        self.calculate_clustering_cost(xnum, xcat)
        self.clusters = np.array([np.argwhere(self.membership[:, pt])[0] for pt in range(npoints)])

    def _add_point_to_cluster(self, point_num, point_cat, ipoint, cluster):
        self.membership[cluster, ipoint] = 1
        # update sums of attributes in cluster
        for iattr, curattr in enumerate(point_num):
            self._clustAttrSum[cluster][iattr] += curattr
        # update frequencies of attributes in cluster
        for iattr, curattr in enumerate(point_cat):
            self._clustAttrFreq[cluster][iattr][curattr] += 1
        return

    def _remove_point_from_cluster(self, point_num, point_cat, ipoint, cluster):
        self.membership[cluster, ipoint] = 0
        # update sums of attributes in cluster
        for iattr, curattr in enumerate(point_num):
            self._clustAttrSum[cluster][iattr] -= curattr
        # update frequencies of attributes in cluster
        for iattr, curattr in enumerate(point_cat):
            self._clustAttrFreq[cluster][iattr][curattr] -= 1
        return

    @staticmethod
    def get_dissim_num(anum, b):
        # Euclidean distance
        return np.sum((anum - b) ** 2, axis=1)

    def calculate_clustering_cost(self, xnum, xcat):
        ncost = 0
        ccost = 0
        for ipoint, curpoint in enumerate(xnum):
            ncost += np.sum(self.get_dissim_num(self.centroids[0], curpoint) *
                            (self.membership[:, ipoint] ** self.alpha))
        for ipoint, curpoint in enumerate(xcat):
            ccost += np.sum(self.get_dissim(self.centroids[1], curpoint) *
                            (self.membership[:, ipoint] ** self.alpha))
        self.cost = ncost + self.gamma * ccost
        if np.isnan(self.cost):
            pass
        return


# noinspection PyNoneFunctionAssignment,PyTypeChecker
class FuzzyKModes(KModes):

    def __init__(self, k, alpha=1.5):
        """Fuzzy k-modes clustering algorithm for categorical data.
        Uses traditional, hard centroids, following Huang, Z., Ng, M.K.:
        A fuzzy k-modes algorithm for clustering categorical data,
        IEEE Transactions on Fuzzy Systems 7(4), 1999.

        Inputs:     k           = number of clusters
                    alpha       = alpha coefficient
        Attributes: clusters    = cluster numbers with max. membership [no. points]
                    membership  = membership matrix [k * no. points]
                    centroids   = centroids [k * no. attributes]
                    cost        = clustering cost

        """
        super(FuzzyKModes, self).__init__(k)

        assert alpha > 1, "alpha should be > 1 (alpha = 1 equals regular k-modes)."
        self.alpha = alpha

        self.omega = None

    def _perform_clustering(self, x, init_method='Huang', max_iters=200, tol=1e-5,
                            cost_inter=1, verbose=1):
        """Inputs:  x           = data points [no. points * no. attributes]
                    initMethod  = initialization method ('Huang' for the one described in
                                  Huang [1998], 'Cao' for the one in Cao et al. [2009]).
                    max_iters   = maximum no. of iterations
                    tol         = tolerance for termination criterion
                    cost_inter  = frequency with which to check the total cost
                                  (for speeding things up, since it is computationally expensive)
                    verbose     = 0 for no and 1 for normal algorithm progress information,
                                  2 for internal algorithm details

        """

        # convert to numpy array, if needed
        x = np.asanyarray(x)
        npoints, nattrs = x.shape
        assert self.k < npoints, "More clusters than data points?"

        self.initMethod = init_method

        # ----------------------
        #    INIT
        # ----------------------
        if verbose:
            print("Init: initializing centroids")
        self.init_centroids(x)

        # store for all attributes which points have a certain attribute value
        self._domAttrPoints = [defaultdict(list) for _ in range(nattrs)]
        for ipoint, curpoint in enumerate(x):
            for iattr, curattr in enumerate(curpoint):
                self._domAttrPoints[iattr][curattr].append(ipoint)

        # ----------------------
        #    ITERATION
        # ----------------------
        if verbose:
            print("Starting iterations...")
        itr = 0
        converged = False
        lastcost = np.inf
        while itr <= max_iters and not converged:
            self.update_membership(x)
            self.update_centroids()

            # computationally expensive, only check every N steps
            if itr % cost_inter == 0:
                self.calculate_clustering_cost(x)
                converged = self.cost >= lastcost * (1 - tol)
                lastcost = self.cost
                if verbose:
                    print("Iteration: {0}/{1}, cost: {2}".format(itr, max_iters, self.cost))
            itr += 1

        self.clusters = np.array([int(np.argmax(self.membership[:, pt])) for pt in range(npoints)])

    def update_membership(self, x, threshold=1e-3):
        npoints = x.shape[0]
        self.membership = np.empty((self.k, npoints))
        for ipoint, curpoint in enumerate(x):
            dissim = self.get_dissim(self.centroids, curpoint)
            if np.any(dissim <= threshold):
                self.membership[:, ipoint] = np.where(dissim <= threshold, 1, threshold)
            else:
                for ik in range(len(self.centroids)):
                    factor = 1. / (self.alpha - 1)
                    # noinspection PyTypeChecker
                    self.membership[ik, ipoint] = 1 / np.sum((float(dissim[ik]) / dissim) ** factor)
        return

    def update_centroids(self):
        self.centroids = np.empty((self.k, len(self._domAttrPoints)))
        for ik in range(self.k):
            for iattr in range(len(self._domAttrPoints)):
                # return attribute that maximizes the sum of the memberships
                v = list(self._domAttrPoints[iattr].values())
                k = list(self._domAttrPoints[iattr].keys())
                memvar = [sum(self.membership[ik, x] ** self.alpha) for x in v]
                # noinspection PyTypeChecker
                self.centroids[ik, iattr] = k[np.argmax(memvar)]
        return


# noinspection PyTypeChecker,PyNoneFunctionAssignment,PyUnresolvedReferences
class FuzzyCentroidsKModes(KModes):

    def __init__(self, k, alpha=1.5):
        """Fuzzy k-modes clustering algorithm for categorical data.
        Uses fuzzy centroids, following and Kim, D.-W., Lee, K.H., Lee, D.:
        Fuzzy clustering of categorical data using fuzzy centroids, Pattern
        Recognition Letters 25, 1262-1271, 2004.

        Inputs:     k           = number of clusters
                    alpha       = alpha coefficient
        Attributes: clusters    = cluster numbers with max. membership [no. points]
                    membership  = membership matrix [k * no. points]
                    omega       = fuzzy centroids [dicts with element values as keys,
                                  element memberships as values, inside lists for
                                  attributes inside list for centroids]
                    cost        = clustering cost

        """
        super(FuzzyCentroidsKModes, self).__init__(k)

        assert k > 1, "Choose at least 2 clusters."
        self.k = k

        assert alpha > 1, "alpha should be > 1 (alpha = 1 equals regular k-modes)."
        self.alpha = alpha

    def _perform_clustering(self, x, max_iters=100, tol=1e-5, cost_inter=1, verbose=1):
        """Inputs:  x           = data points [no. points * no. attributes]
                    max_iters   = maximum no. of iterations
                    tol         = tolerance for termination criterion
                    cost_inter  = frequency with which to check the total cost
                                  (for speeding things up, since it is computationally expensive)
                    verbose     = 0 for no and 1 for normal algorithm progress information,
                                  2 for internal algorithm details

        """

        # convert to numpy array, if needed
        x = np.asanyarray(x)
        npoints, nattrs = x.shape
        assert self.k < npoints, "More clusters than data points?"

        # ----------------------
        #    INIT
        # ----------------------
        if verbose:
            print("Init: initializing centroids")
        # count all attributes
        freqattrs = [defaultdict(int) for _ in range(nattrs)]
        for curpoint in x:
            for iattr, curattr in enumerate(curpoint):
                freqattrs[iattr][curattr] += 1

        # omega = fuzzy set (as dict) for each attribute per cluster
        self.omega = [[{} for _ in range(nattrs)] for _ in range(self.k)]
        for ik in range(self.k):
            for iattr in range(nattrs):
                # a bit unclear form the paper, but this is how they do it in their code
                # give a random attribute 1.0 membership and the rest 0.0
                randint = np.random.randint(len(freqattrs[iattr]))
                for iVal, curVal in enumerate(freqattrs[iattr]):
                    self.omega[ik][iattr][curVal] = float(iVal == randint)

        # ----------------------
        #    ITERATION
        # ----------------------
        if verbose:
            print("Starting iterations...")
        itr = 0
        converged = False
        lastcost = np.inf
        while itr <= max_iters and not converged:
            # O(k*N*at*no. of unique values)
            self.update_membership(x)

            # O(k*N*at)
            self.update_centroids(x)

            # computationally expensive, only check every N steps
            if itr % cost_inter == 0:
                self.calculate_clustering_cost(x)
                converged = self.cost >= lastcost * (1 - tol)
                lastcost = self.cost
                if verbose:
                    print("Iteration: {0}/{1}, cost: {2}".format(itr, max_iters, self.cost))
            itr += 1

        self.clusters = np.array([int(np.argmax(self.membership[:, pt])) for pt in range(npoints)])

    def update_membership(self, x, threshold=1e-3):
        # Eq. 20 from Kim et al. [2004]
        npoints = x.shape[0]
        self.membership = np.empty((self.k, npoints))
        for ipoint, curpoint in enumerate(x):
            dissim = self.get_fuzzy_dissim(curpoint)
            if np.any(dissim <= threshold):
                self.membership[:, ipoint] = np.where(dissim <= threshold, 1, threshold)
            else:
                # NOTE: squaring the distances is not mentioned in the paper, but it is
                # in the code of Kim et al.; seems to improve performance
                dissim **= 2
                for ik in range(len(self.omega)):
                    factor = 1. / (self.alpha - 1)
                    self.membership[ik, ipoint] = 1 / np.sum((float(dissim[ik]) / dissim) ** factor)
        return

    def update_centroids(self, x):
        # noinspection PyAttributeOutsideInit
        self.omega = [[defaultdict(float) for _ in range(x.shape[1])] for _ in range(self.k)]
        for ik in range(self.k):
            for iattr in range(x.shape[1]):
                for ipoint, curpoint in enumerate(x[:, iattr]):
                    self.omega[ik][iattr][curpoint] += self.membership[ik, ipoint] ** self.alpha
                # normalize so that sum omegas is 1, analogous to k-means
                # (see e.g. Yang et al. [2008] who explain better than the original paper)
                sumomg = sum(self.omega[ik][iattr].values())
                for key in self.omega[ik][iattr].keys():
                    self.omega[ik][iattr][key] /= sumomg
        return

    def get_fuzzy_dissim(self, x):
        # TODO: slow, could it be faster?
        # dissimilarity = sums of all omegas for non-matching attributes
        # see Eqs. 13-15 of Kim et al. [2004]
        dissim = np.zeros(len(self.omega))
        for ik in range(len(self.omega)):
            for iattr, curattr in enumerate(self.omega[ik]):
                nonmatch = [v for k, v in curattr.items() if k != x[iattr]]
                # dissim[ik] += sum(nonmatch)
                # following the code of Kim et al., seems to work better
                dissim[ik] += sum(nonmatch) / np.sqrt(np.sum(np.array(list(curattr.values())) ** 2))
        return dissim

    def calculate_clustering_cost(self, x):
        self.cost = 0
        for ipoint, curpoint in enumerate(x):
            self.cost += np.sum(self.get_fuzzy_dissim(curpoint) *
                                (self.membership[:, ipoint] ** self.alpha))
        return


# noinspection PyUnresolvedReferences,PyTypeChecker
def soybean_test():
    # reproduce results on small soybean data set
    x = np.genfromtxt('./test.csv', dtype=str, delimiter=',')[:, :-1]
    y = np.genfromtxt('./test.csv', dtype=int, delimiter=',', usecols=5)

    # drop columns with single value
    # x = x[:, np.std(x, axis=0) > 0.]

    kmodes_huang = KModes(4)
    kmodes_huang.cluster(x, init_method='Huang')
    kmodes_cao = KModes(4)
    kmodes_cao.cluster(x, init_method='Cao')
    kproto = KPrototypes(4)
    kproto.cluster([np.random.randn(x.shape[0], 3), x], init_method='Huang')
    # fkmodes = FuzzyKModes(4, alpha=1.1)
    # fkmodes.cluster(x)
    # ffkmodes = FuzzyCentroidsKModes(4, alpha=1.8)
    # ffkmodes.cluster(x)

    for result in (kmodes_huang, kmodes_cao, kproto):
        classtable = np.zeros((4, 4), dtype=int)
        for ii, _ in enumerate(y):
            classtable[int(y[ii][-1]) - 1, result.clusters[ii]] += 1

        print("\n")
        print("    | Cl. 1 | Cl. 2 | Cl. 3 | Cl. 4 |")
        print("----|-------|-------|-------|-------|")
        for ii in range(4):
            prargs = tuple([ii + 1] + list(classtable[ii, :]))
            print(" D{0} |    {1:>2} |    {2:>2} |    {3:>2} |    {4:>2} |".format(*prargs))


if __name__ == "__main__":
    soybean_test()
