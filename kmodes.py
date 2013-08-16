#!/usr/bin/env python

__author__  = 'Nico de Vos'
__email__   = 'njdevos@gmail.com'
__license__ = 'MIT'
__version__ = '0.7'

import random
import numpy as np
from collections import defaultdict


class KModes(object):
    
    def __init__(self, k):
        '''k-modes clustering algorithm for categorical data.
        See:
        Huang, Z.: Extensions to the k-modes algorithm for clustering large data sets with
        categorical values, Data Mining and Knowledge Discovery 2(3), 1998.
        
        Inputs:     k           = number of clusters
        Attributes: clusters    = cluster numbers [no. points]
                    centroids   = centroids [k * no. attributes]
                    membership  = membership matrix [k * no. points]
                    cost        = clustering cost, defined as the sum distance of
                                  all points to their respective clusters
        
        '''
        assert k > 1, "Choose at least 2 clusters."
        self.k = k
        
        # generalized form with alpha. alpha > 1 for fuzzy k-modes
        self.alpha = 1
    
    def cluster(self, X, preRuns=10, prePctl=20, *args, **kwargs):
        '''Shell around _perform_clustering method that tries to ensure a good clustering
        result by choosing one that has a relatively low clustering cost compared to the
        costs of a number of pre-runs. (Huang [1998] states that clustering cost can be
        used to judge the clustering quality.)
        
        '''
        
        if preRuns and kwargs.has_key('initMethod') and kwargs['initMethod'] == 'Cao':
            print("Initialization method and algorithm are deterministic. Disabling preruns...")
            preRuns = None
        
        if preRuns:
            preCosts = np.empty(preRuns)
            for pr in range(preRuns):
                self._perform_clustering(X, *args, verbose=0, **kwargs)
                preCosts[pr] = self.cost
                print("Prerun {0} / {1}, Cost = {2}".format(pr+1, preRuns, preCosts[pr]))
            goodCost = np.percentile(preCosts, prePctl)
        else:
            goodCost = np.inf
        
        while True:
            self._perform_clustering(X, *args, verbose=1, **kwargs)
            if self.cost <= goodCost:
                break
    
    def _perform_clustering(self, X, initMethod='Huang', maxIters=100, verbose=1):
        '''Inputs:  X           = data points [no. points * no. attributes]
                    initMethod  = initialization method ('Huang' for the one described in
                                  Huang [1998], 'Cao' for the one in Cao et al. [2009])
                    maxIters    = maximum no. of iterations
                    verbose     = 0 for no and 1 for normal algorithm progress information,
                                  2 for internal algorithm details
        
        '''
        # convert to numpy array, if needed
        X = np.asanyarray(X)
        nPoints, nAttrs = X.shape
        assert self.k < nPoints, "More clusters than data points?"
        
        self.initMethod = initMethod
        
        # ----------------------
        #    INIT
        # ----------------------
        if verbose:
            print("Init: initializing centroids")
        self.init_centroids(X)
        
        if verbose:
            print("Init: initializing clusters")
        self.membership = np.zeros((self.k, nPoints), dtype='int64')
        # self._clustAttrFreq is a list of lists with dictionaries that contain the
        # frequencies of values per cluster and attribute
        self._clustAttrFreq = [[defaultdict(int) for _ in range(nAttrs)] for _ in range(self.k)]
        for iPoint, curPoint in enumerate(X):
            # initial assigns to clusters
            cluster = np.argmin(self.get_dissim(self.centroids, curPoint))
            self.membership[cluster,iPoint] = 1
            # count attribute values per cluster
            for iAttr, curAttr in enumerate(curPoint):
                self._clustAttrFreq[cluster][iAttr][curAttr] += 1
        # perform an initial centroid update
        for ik in range(self.k):
            for iAttr in range(nAttrs):
                self.centroids[ik,iAttr] = self.get_mode(self._clustAttrFreq[ik][iAttr])
        
        # ----------------------
        #    ITERATION
        # ----------------------
        if verbose:
            print("Starting iterations...")
        itr = 0
        converged = False
        while itr <= maxIters and not converged:
            itr += 1
            moves = 0
            for iPoint, curPoint in enumerate(X):
                cluster = np.argmin(self.get_dissim(self.centroids, curPoint))
                # if necessary: move point, and update old/new cluster frequencies and centroids
                if not self.membership[cluster, iPoint]:
                    moves += 1
                    oldCluster = np.argwhere(self.membership[:,iPoint])[0][0]
                    self._add_point_to_cluster(curPoint, iPoint, cluster)
                    self._remove_point_from_cluster(curPoint, iPoint, oldCluster)
                    # update new and old centroids by choosing most likely attribute
                    for iAttr, curAttr in enumerate(curPoint):
                        for curc in (cluster, oldCluster):
                            self.centroids[curc, iAttr] = self.get_mode(self._clustAttrFreq[curc][iAttr])
                    if verbose == 2:
                        print("Move from cluster {0} to {1}".format(oldCluster, cluster))
                        
                        # in case of an empty cluster, reinitialize with a random point
                        # that is not a centroid
                        if sum(self.membership[oldCluster,:]) == 0:
                            while True:
                                rIndx = np.random.randint(nPoints)
                                if not all(X[rIndx] == self.centroids).any():
                                    break
                            self._add_point_to_cluster(X[rIndx], rIndx, oldCluster)
                            fromCluster = np.argwhere(self.membership[:,rIndx])[0][0]
                            self._remove_point_from_cluster(X[rIndx], rIndx, fromCluster)
            
            # all points seen in this iteration
            converged = (moves == 0)
            if verbose:
                print("Iteration: {0}/{1}, moves: {2}".format(itr, maxIters, moves))
        
        self.calculate_clustering_cost(X)
        self.clusters = np.array([np.argwhere(self.membership[:,pt])[0] \
                        for pt in range(nPoints)])
    
    def init_centroids(self, X):
        assert self.initMethod in ('Huang', 'Cao')
        nPoints, nAttrs = X.shape
        self.centroids = np.empty((self.k, nAttrs))
        if self.initMethod == 'Huang':
            # determine frequencies of attributes
            for iAttr in range(nAttrs):
                freq = defaultdict(int)
                for curAttr in X[:,iAttr]:
                    freq[curAttr] += 1
                # sample centroids using the probabilities of attributes
                # (I assume that's what's meant in the Huang [1998] paper; it works, at least)
                # note: sampling using population in static list with as many choices as
                # frequency counts this works well since (1) we re-use the list k times here,
                # and (2) the counts are small integers so memory consumption is low
                choices = [chc for chc, wght in freq.items() for _ in range(wght)]
                for ik in range(self.k):
                    self.centroids[ik, iAttr] = random.choice(choices)
            # the previously chosen centroids could result in empty clusters,
            # so set centroid to closest point in X
            for ik in range(self.k):
                ndx = np.argsort(self.get_dissim(X, self.centroids[ik]))
                # and we want the centroid to be unique
                while np.all(X[ndx[0]] == self.centroids, axis=1).any():
                    ndx = np.delete(ndx, 0)
                self.centroids[ik] = X[ndx[0]]
        
        elif self.initMethod == 'Cao':
            # Note: O(N * at * k**2), so watch out with k
            # determine densities points
            dens = np.zeros(nPoints)
            for iAttr in range(nAttrs):
                freq = defaultdict(int)
                for val in X[:,iAttr]:
                    freq[val] += 1
                for iPoint in range(nPoints):
                    dens[iPoint] += freq[X[iPoint,iAttr]] / float(nAttrs)
            dens /= nPoints
            
            # choose centroids based on distance and density
            self.centroids[0] = X[np.argmax(dens)]
            dissim = self.get_dissim(X, self.centroids[0])
            self.centroids[1] = X[np.argmax(dissim * dens)]
            # for the reamining centroids, choose max dens * dissim to the (already assigned)
            # centroid with the lowest dens * dissim
            for ik in range(2,self.k):
                dd = np.empty((ik, nPoints))
                for ikk in range(ik):
                    dd[ikk] = self.get_dissim(X, self.centroids[ikk]) * dens
                self.centroids[ik] = X[np.argmax(np.min(dd, axis=0))]
        
        return
    
    def _add_point_to_cluster(self, point, iPoint, cluster):
        self.membership[cluster,iPoint] = 1
        # update frequencies of attributes in cluster
        for iAttr, curAttr in enumerate(point):
            self._clustAttrFreq[cluster][iAttr][curAttr] += 1
        return
    
    def _remove_point_from_cluster(self, point, iPoint, cluster):
        self.membership[cluster,iPoint] = 0
        # update frequencies of attributes in cluster
        for iAttr, curAttr in enumerate(point):
            self._clustAttrFreq[cluster][iAttr][curAttr] -= 1
        return
    
    @staticmethod
    def get_dissim(A, b):
        # simple matching dissimilarity
        return (A != b).sum(axis=1)
    
    @staticmethod
    def get_mode(dic):
        # Fast method (supposedly) to get key for maximum value in dict.
        v = list(dic.values())
        k = list(dic.keys())
        if len(v) == 0:
            pass
        return k[v.index(max(v))]
    
    def calculate_clustering_cost(self, X):
        self.cost = 0
        for iPoint, curPoint in enumerate(X):
            self.cost += np.sum( self.get_dissim(self.centroids, curPoint) * \
                         (self.membership[:,iPoint] ** self.alpha) )
        return

###################################################################################################

class KPrototypes(KModes):
    
    def __init__(self, k):
        '''k-protoypes clustering algorithm for mixed numeric and categorical data.
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
        
        '''
        super(KPrototypes, self).__init__(k)
    
    def _perform_clustering(self, X, gamma=None, initMethod='Huang', maxIters=100, verbose=1):
        '''Inputs:  Xnum        = numeric data points [no. points * no. numeric attributes]
                    Xcat        = categorical data points [no. points * no. numeric attributes]
                    gamma       = weighing factor that determines relative importance of
                                  num./cat. attributes (see discussion in Huang [1997])
                    initMethod  = initialization method ('Huang' for the one described in
                                  Huang [1998], 'Cao' for the one in Cao et al. [2009])
                    maxIters    = maximum no. of iterations
                    verbose     = 0 for no and 1 for normal algorithm progress information,
                                  2 for internal algorithm details
        
        '''
        # convert to numpy arrays, if needed
        Xnum, Xcat = X[0], X[1]
        Xnum = np.asanyarray(Xnum)
        Xcat = np.asanyarray(Xcat)
        nNumPoints, nNumAttrs = Xnum.shape
        nCatPoints, nCatAttrs = Xcat.shape
        assert nNumPoints == nCatPoints, "More numerical points than categorical?"
        nPoints = nNumPoints
        assert self.k < nPoints, "More clusters than data points?"
        
        self.initMethod = initMethod
        
        # estimate a good value for gamma, which determines the weighing of
        # categorical values in clusters (see Huang [1997])
        if gamma is None:
            gamma = 0.5 * np.std(Xnum)
        self.gamma = gamma
        
        # ----------------------
        #    INIT
        # ----------------------
        if verbose:
            print("Init: initializing centroids")
        # list where [0] = numerical part of centroid and [1] = categorical part
        self.init_centroids(Xcat)
        self.centroids = [np.mean(Xnum, axis=0) + np.random.randn(self.k, nNumAttrs) * \
                         np.std(Xnum, axis=0), self.centroids]
        
        if verbose:
            print("Init: initializing clusters")
        self.membership = np.zeros((self.k, nPoints), dtype='int64')
        # keep track of the sum of attribute values per cluster
        self._clustAttrSum = np.zeros((self.k, nNumAttrs), dtype='float')
        # self._clustAttrFreq is a list of lists with dictionaries that contain
        # the frequencies of values per cluster and attribute
        self._clustAttrFreq = [[defaultdict(int) for _ in range(nCatAttrs)] for _ in range(self.k)]
        for iPoint in range(nPoints):
            # initial assigns to clusters
            cluster = np.argmin(self.get_dissim_num(self.centroids[0], Xnum[iPoint]) + \
                      self.gamma * self.get_dissim(self.centroids[1], Xcat[iPoint]))
            self.membership[cluster,iPoint] = 1
            # count attribute values per cluster
            for iAttr, curAttr in enumerate(Xnum[iPoint]):
                self._clustAttrSum[cluster,iAttr] += curAttr
            for iAttr, curAttr in enumerate(Xcat[iPoint]):
                self._clustAttrFreq[cluster][iAttr][curAttr] += 1
        for ik in range(self.k):
            # in case of an empty cluster, reinitialize with a random point
            # that is not a centroid
            if sum(self.membership[ik,:]) == 0:
                while True:
                    rIndx = np.random.randint(nPoints)
                    if not np.all(np.vstack((np.all(Xnum[rIndx] == self.centroids[0], axis=1), \
                                             np.all(Xcat[rIndx] == self.centroids[1], axis=1))), \
                                             axis=0).any():
                        break
                self._add_point_to_cluster(Xnum[rIndx], Xcat[rIndx], rIndx, ik)
                fromCluster = np.argwhere(self.membership[:,rIndx])[0][0]
                self._remove_point_from_cluster(Xnum[rIndx], Xcat[rIndx], rIndx, fromCluster)
        # perform an initial centroid update
        for ik in range(self.k):
            for iAttr in range(nNumAttrs):
                self.centroids[0][ik,iAttr] = self._clustAttrSum[ik,iAttr] / sum(self.membership[ik,:])
            for iAttr in range(nCatAttrs):
                self.centroids[1][ik,iAttr] = self.get_mode(self._clustAttrFreq[ik][iAttr])
        
        # ----------------------
        #    ITERATION
        # ----------------------
        if verbose:
            print("Starting iterations...")
        itr = 0
        converged = False
        while itr <= maxIters and not converged:
            itr += 1
            moves = 0
            for iPoint in range(nPoints):
                cluster = np.argmin(self.get_dissim_num(self.centroids[0], Xnum[iPoint]) + \
                          self.gamma * self.get_dissim(self.centroids[1], Xcat[iPoint]))
                # if necessary: move point, and update old/new cluster frequencies and centroids
                if not self.membership[cluster, iPoint]:
                    moves += 1
                    oldCluster = np.argwhere(self.membership[:,iPoint])[0][0]
                    self._add_point_to_cluster(Xnum[iPoint], Xcat[iPoint], iPoint, cluster)
                    self._remove_point_from_cluster(Xnum[iPoint], Xcat[iPoint], iPoint, oldCluster)
                    # update new and old centroids by choosing mean for numerical and
                    # most likely for categorical attributes
                    for iAttr in range(len(Xnum[iPoint])):
                        for curc in (cluster, oldCluster):
                            if sum(self.membership[curc,:]):
                                self.centroids[0][curc, iAttr] = self._clustAttrSum[ik,iAttr] / \
                                                                 sum(self.membership[curc,:])
                            else:
                                self.centroids[0][curc, iAttr] = 0
                    for iAttr in range(len(Xcat[iPoint])):
                        for curc in (cluster, oldCluster):
                            self.centroids[1][curc, iAttr] = \
                                self.get_mode(self._clustAttrFreq[curc][iAttr])
                    if verbose == 2:
                        print("Move from cluster {0} to {1}".format(oldCluster, cluster))
                    
                    # in case of an empty cluster, reinitialize with a random point
                    # that is not a centroid
                    if sum(self.membership[oldCluster,:]) == 0:
                        while True:
                            rIndx = np.random.randint(nPoints)
                            if not all(Xnum[rIndx] == self.centroids[0]).any() and \
                               not all(Xcat[rIndx] == self.centroids[1]).any():
                                break
                        self._add_point_to_cluster(Xnum[rIndx], Xcat[rIndx], rIndx, oldCluster)
                        fromCluster = np.argwhere(self.membership[:,rIndx])[0][0]
                        self._remove_point_from_cluster(Xnum[rIndx], Xcat[rIndx], rIndx, fromCluster)
            
            # all points seen in this iteration
            converged = (moves == 0)
            if verbose:
                print("Iteration: {0}/{1}, moves: {2}".format(itr, maxIters, moves))
        
        self.calculate_clustering_cost(Xnum, Xcat)
        self.clusters = np.array([np.argwhere(self.membership[:,pt])[0] for pt in range(nPoints)])
    
    def _add_point_to_cluster(self, pointNum, pointCat, iPoint, cluster):
        self.membership[cluster,iPoint] = 1
        # update sums of attributes in cluster
        for iAttr, curAttr in enumerate(pointNum):
            self._clustAttrSum[cluster][iAttr] += curAttr
        # update frequencies of attributes in cluster
        for iAttr, curAttr in enumerate(pointCat):
            self._clustAttrFreq[cluster][iAttr][curAttr] += 1
        return
    
    def _remove_point_from_cluster(self, pointNum, pointCat, iPoint, cluster):
        self.membership[cluster,iPoint] = 0
        # update sums of attributes in cluster
        for iAttr, curAttr in enumerate(pointNum):
            self._clustAttrSum[cluster][iAttr] -= curAttr
        # update frequencies of attributes in cluster
        for iAttr, curAttr in enumerate(pointCat):
            self._clustAttrFreq[cluster][iAttr][curAttr] -= 1
        return
    
    @staticmethod
    def get_dissim_num(Anum, b):
        # Euclidian distance
        return np.sum((Anum - b)**2, axis=1)
    
    def calculate_clustering_cost(self, Xnum, Xcat):
        ncost = 0
        ccost = 0
        for iPoint, curPoint in enumerate(Xnum):
            ncost += np.sum( self.get_dissim_num(self.centroids[0], curPoint) * \
                     (self.membership[:,iPoint] ** self.alpha) )
        for iPoint, curPoint in enumerate(Xcat):
            ccost += np.sum( self.get_dissim(self.centroids[1], curPoint) * \
                     (self.membership[:,iPoint] ** self.alpha) )
        self.cost = ncost + self.gamma * ccost
        if np.isnan(self.cost):
            pass
        return

###################################################################################################

class FuzzyKModes(KModes):
    
    def __init__(self, k, alpha=1.5):
        '''Fuzzy k-modes clustering algorithm for categorical data.
        Uses traditional, hard centroids, following Huang, Z., Ng, M.K.:
        A fuzzy k-modes algorithm for clustering categorical data, 
        IEEE Transactions on Fuzzy Systems 7(4), 1999.
        
        Inputs:     k           = number of clusters
                    alpha       = alpha coefficient
        Attributes: clusters    = cluster numbers with max. membership [no. points]
                    membership  = membership matrix [k * no. points]
                    centroids   = centroids [k * no. attributes]
                    cost        = clustering cost
        
        '''
        super(FuzzyKModes, self).__init__(k)
        
        assert alpha > 1, "alpha should be > 1 (alpha = 1 equals regular k-modes)."
        self.alpha = alpha
        
    def _perform_clustering(self, X, initMethod='Huang', maxIters=200, tol=1e-5, \
                            costInter=1, verbose=1):
        '''Inputs:  X           = data points [no. points * no. attributes]
                    initMethod  = initialization method ('Huang' for the one described in
                                  Huang [1998], 'Cao' for the one in Cao et al. [2009]).
                    maxIters    = maximum no. of iterations
                    tol         = tolerance for termination criterion
                    costInter   = frequency with which to check the total cost
                                  (for speeding things up, since it is computationally expensive)
                    verbose     = 0 for no and 1 for normal algorithm progress information,
                                  2 for internal algorithm details
        
        '''
        
        # convert to numpy array, if needed
        X = np.asanyarray(X)
        nPoints, nAttrs = X.shape
        assert self.k < nPoints, "More clusters than data points?"
        
        self.initMethod = initMethod
        
        # ----------------------
        #    INIT
        # ----------------------
        if verbose:
            print("Init: initializing centroids")
        self.init_centroids(X)
        
        # store for all attributes which points have a certain attribute value
        self._domAttrPoints = [defaultdict(list) for _ in range(nAttrs)]
        for iPoint, curPoint in enumerate(X):
            for iAttr, curAttr in enumerate(curPoint):
                self._domAttrPoints[iAttr][curAttr].append(iPoint)
        
        # ----------------------
        #    ITERATION
        # ----------------------
        if verbose:
            print("Starting iterations...")
        itr = 0
        converged = False
        lastCost = np.inf
        while itr <= maxIters and not converged:
            self.update_membership(X)
            self.update_centroids()
            
            # computationally expensive, only check every N steps
            if itr % costInter == 0:
                self.calculate_clustering_cost(X)
                converged = self.cost >= lastCost * (1-tol)
                lastCost = self.cost
                if verbose:
                    print("Iteration: {0}/{1}, cost: {2}".format(itr, maxIters, self.cost))
            itr += 1
        
        self.clusters = np.array([int(np.argmax(self.membership[:,pt])) for pt in range(nPoints)])
    
    def update_membership(self, X, treshold=1e-3):
        nPoints = X.shape[0]
        self.membership = np.empty((self.k, nPoints))
        for iPoint, curPoint in enumerate(X):
            dissim = self.get_dissim(self.centroids, curPoint)
            if np.any(dissim <= treshold):
                self.membership[:,iPoint] = np.where(dissim <= treshold, 1, treshold)
            else:
                for ik in range(len(self.centroids)):
                    factor = 1. / (self.alpha - 1)
                    self.membership[ik,iPoint] = 1 / np.sum( (float(dissim[ik]) / dissim)**factor )
        return
    
    def update_centroids(self):
        self.centroids = np.empty((self.k, len(self._domAttrPoints)))
        for ik in range(self.k):
            for iAttr in range(len(self._domAttrPoints)):
                # return attribute that maximizes the sum of the memberships
                v = list(self._domAttrPoints[iAttr].values())
                k = list(self._domAttrPoints[iAttr].keys())
                memvar = [sum(self.membership[ik,x]**self.alpha) for x in v]
                self.centroids[ik, iAttr] = k[np.argmax(memvar)]
        return

###################################################################################################

class FuzzyCentroidsKModes(KModes):
    
    def __init__(self, k, alpha=1.5):
        '''Fuzzy k-modes clustering algorithm for categorical data.
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
        
        '''
        assert k > 1, "Choose at least 2 clusters."
        self.k = k
        
        assert alpha > 1, "alpha should be > 1 (alpha = 1 equals regular k-modes)."
        self.alpha = alpha
    
    def _perform_clustering(self, X, maxIters=100, tol=1e-5, costInter=1, verbose=1):
        '''Inputs:  X           = data points [no. points * no. attributes]
                    maxIters    = maximum no. of iterations
                    tol         = tolerance for termination criterion
                    costInter   = frequency with which to check the total cost
                                  (for speeding things up, since it is computationally expensive)
                    verbose     = 0 for no and 1 for normal algorithm progress information,
                                  2 for internal algorithm details
        
        '''
        
        # convert to numpy array, if needed
        X = np.asanyarray(X)
        nPoints, nAttrs = X.shape
        assert self.k < nPoints, "More clusters than data points?"
        
        # ----------------------
        #    INIT
        # ----------------------
        if verbose:
            print("Init: initializing centroids")
        # count all attributes
        freqAttrs = [defaultdict(int) for _ in range(nAttrs)]
        for curPoint in X:
            for iAttr, curAttr in enumerate(curPoint):
                freqAttrs[iAttr][curAttr] += 1
        
        # omega = fuzzy set (as dict) for each attribute per cluster
        self.omega = [[{} for _ in range(nAttrs)] for _ in range(self.k)]
        for ik in range(self.k):
            for iAttr in range(nAttrs):
                # a bit unclear form the paper, but this is how they do it in their code
                # give a random attribute 1.0 membership and the rest 0.0
                randInt = np.random.randint(len(freqAttrs[iAttr]))
                for iVal, curVal in enumerate(freqAttrs[iAttr]):
                    self.omega[ik][iAttr][curVal] = float(iVal == randInt)
        
        # ----------------------
        #    ITERATION
        # ----------------------
        if verbose:
            print("Starting iterations...")
        itr = 0
        converged = False
        lastCost = np.inf
        while itr <= maxIters and not converged:
            # O(k*N*at*no. of unique values)
            self.update_membership(X)
            
            # O(k*N*at)
            self.update_centroids(X)
            
            # computationally expensive, only check every N steps
            if itr % costInter == 0:
                self.calculate_clustering_cost(X)
                converged = self.cost >= lastCost * (1-tol)
                lastCost = self.cost
                if verbose:
                    print("Iteration: {0}/{1}, cost: {2}".format(itr, maxIters, self.cost))
            itr += 1
        
        self.clusters = np.array([int(np.argmax(self.membership[:,pt])) for pt in range(nPoints)])
    
    def update_membership(self, X, treshold=1e-3):
        # Eq. 20 from Kim et al. [2004]
        nPoints = X.shape[0]
        self.membership = np.empty((self.k, nPoints))
        for iPoint, curPoint in enumerate(X):
            dissim = self.get_fuzzy_dissim(curPoint)
            if np.any(dissim <= treshold):
                self.membership[:,iPoint] = np.where(dissim <= treshold, 1, treshold)
            else:
                # NOTE: squaring the distances is not mentioned in the paper, but it is
                # in the code of Kim et al.; seems to improve performance
                dissim = dissim ** 2
                for ik, curc in enumerate(self.omega):
                    factor = 1. / (self.alpha - 1)
                    self.membership[ik,iPoint] = 1 / np.sum( (float(dissim[ik]) / dissim)**factor )
        return
    
    def update_centroids(self, X):
        self.omega = [[defaultdict(float) for _ in range(X.shape[1])] for _ in range(self.k)]
        for ik in range(self.k):
            for iAttr in range(X.shape[1]):
                for iPoint, curPoint in enumerate(X[:,iAttr]):
                    self.omega[ik][iAttr][curPoint] += self.membership[ik,iPoint] ** self.alpha
                # normalize so that sum omegas is 1, analogous to k-means
                # (see e.g. Yang et al. [2008] who explain better than the original paper)
                sumOmg = sum(self.omega[ik][iAttr].values())
                for key in self.omega[ik][iAttr].keys():
                    self.omega[ik][iAttr][key] /= sumOmg
        return
    
    def get_fuzzy_dissim(self, x):
        # dissimilarity = sums of all omegas for non-matching attributes
        # see Eqs. 13-15 of Kim et al. [2004]
        dissim = np.zeros(len(self.omega))
        for ik in range(len(self.omega)):
            for iAttr in range(len(self.omega[ik])):
                attrValues = np.array(self.omega[ik][iAttr].items())
                nonMatch = [v for k, v in attrValues if k != x[iAttr]]
                # dissim[ik] += sum(nonMatch)
                # following the code of Kim et al., seems to work better
                dissim[ik] += sum(nonMatch) / np.sqrt(np.sum(attrValues ** 2))
        return dissim
    
    def calculate_clustering_cost(self, X):
        self.cost = 0
        for iPoint, curPoint in enumerate(X):
            self.cost += np.sum( self.get_fuzzy_dissim(curPoint) * \
                         (self.membership[:,iPoint] ** self.alpha) )
        return


def soybean_test():
    # reproduce results on small soybean data set
    X = np.genfromtxt('./soybean.csv', dtype='int64', delimiter=',')[:,:-1]
    y = np.genfromtxt('./soybean.csv', dtype='unicode', delimiter=',', usecols=35)
    
    # drop columns with single value
    X = X[:,np.std(X, axis=0) > 0.]
    
    kmodes_huang = KModes(4)
    kmodes_huang.cluster(X, initMethod='Huang')
    kmodes_cao = KModes(4)
    kmodes_cao.cluster(X, initMethod='Cao')
    kproto = KPrototypes(4)
    kproto.cluster([np.random.randn(X.shape[0], 3), X], initMethod='Huang')
    fkmodes = FuzzyKModes(4, alpha=1.1)
    fkmodes.cluster(X)
    ffkmodes = FuzzyCentroidsKModes(4, alpha=1.8)
    ffkmodes.cluster(X)
    
    for result in (kmodes_huang, kmodes_cao, kproto, fkmodes, ffkmodes):
        classtable = np.zeros((4,4), dtype='int64')
        for ii,_ in enumerate(y):
            classtable[int(y[ii][-1])-1,result.clusters[ii]] += 1
        
        print("\n")
        print("    | Cl. 1 | Cl. 2 | Cl. 3 | Cl. 4 |")
        print("----|-------|-------|-------|-------|")
        for ii in range(4):
            prargs = tuple([ii+1] + list(classtable[ii,:]))
            print(" D{0} |    {1:>2} |    {2:>2} |    {3:>2} |    {4:>2} |".format(*prargs))


if __name__ == "__main__":
    soybean_test()
