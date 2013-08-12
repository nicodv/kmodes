#!/usr/bin/env python

'''
Implementation of the k-modes clustering algorithm and several of its variations.
'''
__author__  = 'Nico de Vos'
__email__   = 'njdevos@gmail.com'
__license__ = 'MIT'
__version__ = '0.5'

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
    
    def cluster(self, X, initMethod='Huang', maxIters=100, verbose=1):
        '''Inputs:  X           = data points [no. points * no. attributes]
                    initMethod  = initialization method ('Huang' for the one described in
                                  Huang [1998], 'Cao' for the one in Cao et al. [2009])
                    maxIters    = maximum no. of iterations
        '''
        # convert to numpy array, if needed
        X = np.asanyarray(X)
        nPoints, nAttrs = X.shape
        assert self.k < nPoints, "More clusters than data points?"
        
        self.initMethod = initMethod
        
        # ----------------------
        #    INIT
        # ----------------------
        print("Init: initializing centroids")
        centroids = self.init_centroids(X)
        
        print("Init: initializing clusters")
        membership = np.zeros((self.k, nPoints), dtype='int64')
        # clustFreq is a list of lists with dictionaries that contain the
        # frequencies of values per cluster and attribute
        clustAttrFreq = [[defaultdict(int) for _ in range(nAttrs)] for _ in range(self.k)]
        for iPoint, curPoint in enumerate(X):
            # initial assigns to clusters
            cluster = np.argmin(self.get_dissim(centroids, curPoint))
            member[cluster,iPoint] = 1
            # count attribute values per cluster
            for iAttr, curAttr in enumerate(curPoint):
                clustAttrFreq[cluster][iAttr][curAttr] += 1
        # perform an initial centroid update
        for ik in range(self.k):
            for iAttr in range(nAttrs):
                centroids[ik,iAttr] = choose_mode(clustAttrFreq[ik][iAttr])
        
        # ----------------------
        #    ITERATION
        # ----------------------
        print("Starting iterations...")
        itr = 0
        converged = False
        while itr <= maxIters and not converged:
            itr += 1
            moves = 0
            for iPoint, curPoint in enumerate(X):
                cluster = np.argmin(self.get_dissim(centroids, curPoint))
                # if necessary: move point, and update old/new cluster frequencies and centroids
                if not member[cluster, iPoint]:
                    moves += 1
                    oldcluster = np.argwhere(member[:,iPoint])
                    membership[oldcluster,iPoint] = 0
                    membership[cluster,iPoint] = 1
                    for iAttr, curAttr in enumerate(curPoint):
                        # update frequencies of attributes in clusters
                        clustFreq[cluster][iAttr][curAttr] += 1
                        clustFreq[oldcluster][iAttr][curAttr] -= 1
                        # update new and old centroids by choosing most likely attribute
                        for curc in (cluster, oldcluster):
                            cent[curc, iAttr] = key_for_max_value(clustFreq[curc][iAttr])
                    if verbose == 2:
                        print("Move from cluster {0} to {1}".format(oldcluster, cluster))
            
            # all points seen in this iteration
            converged = (moves == 0)
            if verbose:
                print("Iteration: {0}/{1}, moves: {2}".format(itr, maxIters, moves))
        
        self.cost       = self.clustering_cost(X, centroids, membership)
        self.centroids  = centroids
        self.clusters   = np.array([int(np.argwhere(membership[:,point])) for point in range(nPoints)])
        self.membership = membership
    
    def init_centroids(self, X):
        assert self.initMethod in ('Huang', 'Cao')
        nPoints, nAttrs = X.shape
        centroids = np.empty((self.k, nAttrs))
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
                    centroids[ik, iAttr] = random.choice(choices)
            # the previously chosen centroids could result in empty clusters,
            # so set centroid to closest point in X
            for ik in range(self.k):
                dissim = self.get_dissim(X, centroids[ik])
                ndx = dissim.argsort()
                # and we want the centroid to be unique
                while np.all(X[ndx[0]] == centroids, axis=1).any():
                    ndx = np.delete(ndx, 0)
                centroids[ik] = X[ndx[0]]
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
            centroids[0] = X[np.argmax(dens)]
            dissim = self.get_dissim(X, centroids[0])
            centroids[1] = X[np.argmax(dissim * dens)]
            # for the reamining centroids, choose max dens * dissim to the (already assigned)
            # centroid with the lowest dens * dissim
            for ik in range(2,self.k):
                dd = np.empty((ik, nPoints))
                for ikk in range(ik):
                    dd[ikk] = self.get_dissim(X, centroids[ikk]) * dens
                centroids[ik] = X[np.argmax(np.min(dd, axis=0))]
        
        return centroids
    
    @staticmethod
    def get_dissim(A, b):
        # simple matching dissimilarity
        return (A != b).sum(axis=1)
    
    def clustering_cost(self, X, centroids, membership):
        cost = 0
        for iPoint, curPoint in enumerate(X):
            cost += np.sum( self.get_dissim(centroids, curPoint) * (member[:,iPoint] ** self.alpha) )
        return cost

####################################################################################################

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
    
    def cluster(self, Xnum, Xcat, gamma=None, initMethod='Huang', maxIters=100, verbose=1):
        '''Inputs:  Xnum        = numeric data points [no. points * no. numeric attributes]
                    Xcat        = categorical data points [no. points * no. numeric attributes]
                    gamma       = weighing factor that determines relative importance of
                                  num./cat. attributes (see discussion in Huang [1997])
                    initMethod  = initialization method ('Huang' for the one described in
                                  Huang [1998], 'Cao' for the one in Cao et al. [2009])
                    maxIters    = maximum no. of iterations
        '''
        # convert to numpy arrays, if needed
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
        print("Init: initializing centroids")
        # list where [0] = numerical part of centroid and [1] = categorical part
        centroids = [np.mean(Xnum, axis=0) + np.random.randn(self.k, nNumAttrs) * \
                    np.std(Xnum, axis=0), self.init_centroids(Xcat)]
        
        print("Init: initializing clusters")
        membership = np.zeros((self.k, nPoints), dtype='int64')
        # keep track of the sum of attribute values per cluster
        clustSum = np.zeros((self.k, nNumAttrs), dtype='float')
        # clustFreq is a list of lists with dictionaries that contain the
        # frequencies of values per cluster and attribute
        clustFreq = [[defaultdict(int) for _ in range(nCatAttrs)] for _ in range(self.k)]
        for iPoint in range(nPoints):
            # initial assigns to clusters
            cluster = np.argmin(self.get_dissim_num(centroids[0], Xnum[iPoint]) + \
                      self.gamma * self.get_dissim(centroids[1], Xcat[iPoint]))
            member[cluster,iPoint] = 1
            # count attribute values per cluster
            for iNumAttr, curNumAttr in enumerate(Xnum[iPoint]):
                clustSum[cluster,iNumAttr] += curNumAttr
            for iCatAttr, curCatAttr in enumerate(Xcat[iPoint]):
                clustFreq[cluster][iCatAttr][curCatAttr] += 1
        # perform an initial centroid update
        for ik in range(self.k):
            for iNumAttr in range(nNumAttrs):
                cent[0][ik,iNumAttr] = clustSum[ik,iNumAttr] / sum(member[ik,:])
            for iCatAttr in range(nCatAttrs):
                cent[1][ik,iCatAttr] = key_for_max_value(clustFreq[ik][iCatAttr])
        
        # ----------------------
        #    ITERATION
        # ----------------------
        print("Starting iterations...")
        itr = 0
        converged = False
        while itr <= maxIters and not converged:
            itr += 1
            moves = 0
            for iPoint in range(nPoints):
                cluster = np.argmin(self.get_dissim_num(centroids[0], Xnum[iPoint]) + \
                          self.gamma * self.get_dissim(centroids[1], Xcat[iPoint]))
                # if necessary: move point, and update old/new cluster frequencies and centroids
                if not member[cluster, iPoint]:
                    moves += 1
                    oldcluster = np.argwhere(membership[:,iPoint])
                    membership[oldcluster,iPoint] = 0
                    membership[cluster,iPoint] = 1
                    for iNumAttr, curNumAttr in enumerate(Xnum[iPoint]):
                        clustSum[cluster,iNumAttr] += curNumAttr
                        clustSum[oldcluster,iNumAttr] -= curNumAttr
                        for curc in (cluster, oldcluster):
                            cent[0][curc, iNumAttr] = clustSum[ik,iNumAttr] / sum(member[curc,:])
                    for iCatAttr, curCatAttr in enumerate(Xcat[iPoint]):
                        # update frequencies of attributes in clusters
                        clustFreq[cluster][iCatAttr][curCatAttr] += 1
                        clustFreq[oldcluster][iCatAttr][curCatAttr] -= 1
                        # update new and old centroids by choosing most likely attribute
                        for curc in (cluster, oldcluster):
                            centroids[1][curc, iCatAttr] = key_for_max_value(clustFreq[curc][iCatAttr])
                    if verbose == 2:
                        print("Move from cluster {0} to {1}".format(oldcluster, cluster))
            
            # all points seen in this iteration
            converged = (moves == 0)
            if verbose:
                print("Iteration: {0}/{1}, moves: {2}".format(itr, maxIters, moves))
        
        self.cost       = self.clustering_cost(Xnum, Xcat, centroids, membership, gamma)
        self.centroids  = centroids
        self.clusters   = np.array([int(np.argwhere(membership[:,point])) for point in range(nPoints)])
        self.membership = membership
    
    @staticmethod
    def get_dissim_num(Anum, b):
        # Euclidian distance
        return np.sum((Anum - b)**2, axis=1)
    
    def clustering_cost(self, Xnum, Xcat, centroids, membership):
        ncost = 0
        ccost = 0
        for iPoint, curPoint in enumerate(Xnum):
            ncost += np.sum( self.get_dissim_num(centroids[0], curPoint) * (membership[:,iPoint] ** self.alpha) )
        for iPoint, curPoint in enumerate(Xcat):
            ccost += np.sum( self.get_dissim(centroids[1], curPoint) * (membership[:,iPoint] ** self.alpha) )
        return ncost + self.gamma * ccost

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
        
    def cluster(self, X, initMethod='Huang', maxIters=200, tol=1e-5, costInter=1, verbose=1):
        '''Inputs:  X           = data points [no. points * no. attributes]
                    initMethod  = initialization method ('Huang' for the one described in
                                  Huang [1998], 'Cao' for the one in Cao et al. [2009]).
                    maxIters    = maximum no. of iterations
                    tol         = tolerance for termination criterion
                    costInter   = frequency with which to check the total cost
                                  (for speeding things up, since it is computationally expensive)
        
        '''
        
        # convert to numpy array, if needed
        X = np.asanyarray(X)
        nPoints, nAttrs = X.shape
        assert self.k < nPoints, "More clusters than data points?"
        
        self.initMethod = initMethod
        
        # ----------------------
        #    INIT
        # ----------------------
        print("Init: initializing centroids")
        centroids = self.init_centroids(X)
        
        # store for all attributes which points have a certain attribute value
        domAttrPoints = [defaultdict(list) for _ in range(nAttrs)]
        for iPoint, curPoint in enumerate(X):
            for iAttr, curAttr in enumerate(curPoint):
                domAttrPoints[iAttr][curAttr].append(iPoint)
        
        # ----------------------
        #    ITERATION
        # ----------------------
        print("Starting iterations...")
        itr = 0
        converged = False
        lastCost = np.inf
        while itr <= maxIters and not converged:
            membership = self.update_membership(centroids, X)
            for ik in range(self.k):
                centroids[ik] = self.update_centroid(domAttrPoints, membership[ik])
            
            # computationally expensive, only check every N steps
            if itr % costInter == 0:
                cost = self.clustering_cost(X, centroids, membership)
                converged = cost >= lastCost * (1-tol)
                lastCost = cost
                if verbose:
                    print("Iteration: {0}/{1}, cost: {2}".format(itr, maxIters, cost))
            itr += 1
        
        self.cost       = cost
        self.centroids  = centroids
        self.clusters   = np.array([int(np.argmax(membership[:,point])) for point in range(nPoints)])
        self.membership = membership
    
    def update_membership(self, centroids, X, treshold=1e-3):
        nPoints = X.shape[0]
        membership = np.empty((self.k, nPoints))
        for iPoint, curPoint in enumerate(X):
            dissim = self.get_dissim(centroids, curPoint)
            if np.any(dissim <= treshold):
                membership[:,iPoint] = np.where(dissim <= treshold, 1, treshold)
            else:
                for ik in range(len(centroids)):
                    factor = 1. / (self.alpha - 1)
                    membership[ik,iPoint] = 1 / np.sum( (float(dissim[ik]) / dissim)**factor )
        return membership
    
    def update_centroid(self, domAttrPoints, membership):
        centroids = []
        for iAttr in range(len(domAttrPoints)):
            # return attribute that maximizes the sum of the memberships
            v = list(domAttrPoints[iAttr].values())
            k = list(domAttrPoints[iAttr.keys())
            memvar = [sum(membership[x]**self.alpha) for x in v]
            centroids.append(k[np.argmax(memvar)])
        return np.array(centroids)

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
        super(FuzzyCentroidsKModes, self).__init__(k)
        
        assert alpha > 1, "alpha should be > 1 (alpha = 1 equals regular k-modes)."
        self.alpha = alpha
    
    def cluster(self, X, maxIters=100, tol=1e-5, costInter=1, verbose=1):
        '''Inputs:  X           = data points [no. points * no. attributes]
                    maxIters    = maximum no. of iterations
                    tol         = tolerance for termination criterion
                    costInter   = frequency with which to check the total cost
                                  (for speeding things up, since it is computationally expensive)
        
        '''
        
        # convert to numpy array, if needed
        X = np.asanyarray(X)
        nPoints, nAttrs = X.shape
        assert self.k < nPoints, "More clusters than data points?"
        
        # ----------------------
        #    INIT
        # ----------------------
        # count all attributes
        freqAttrs = [defaultdict(int) for _ in range(nAttrs)]
        for curPoint in X:
            for iAttr, curAttr in enumerate(curPoint):
                freqAttrs[iAttr][curAttr] += 1
        
        # omega = fuzzy set (as dict) for each attribute per cluster
        omega = [[{} for _ in range(at)] for _ in range(self.k)]
        for ik in range(self.k):
            for iAttr in range(nAttrs):
                # a bit unclear form the paper, but this is how they do it in their code
                # give a random attribute 1.0 membership and the rest 0.0
                randInt = np.random.randint(len(freqAt[iAttr]))
                for iVal, curVal in enumerate(freqAttrs[iAttr]):
                    omega[ik][iAttr][curVal] = float(iVal == randInt)
        
        # ----------------------
        #    ITERATION
        # ----------------------
        print("Starting iterations...")
        itr = 0
        converged = False
        lastCost = np.inf
        while itr <= maxIters and not converged:
            # O(k*N*at*no. of unique values)
            membership = self.update_membership(X, omega)
            
            # O(k*N*at)
            omega = self.update_centroids(X, membership)
            
            # computationally expensive, only check every N steps
            if itr % costInter == 0:
                cost = self.clustering_cost(X, membership, omega)
                converged = cost >= lastCost * (1-tol)
                lastCost = cost
                if verbose:
                    print("Iteration: {0}/{1}, cost: {2}".format(itr, maxIters, cost))
            itr += 1
        
        self.cost       = cost
        self.omega      = omega
        self.clusters   = np.array([int(np.argmax(membership[:,point])) for point in range(nPoints)])
        self.membership = membership
    
    def update_membership(self, X, omega, treshold=1e-3):
        # Eq. 20 from Kim et al. [2004]
        nPoints = X.shape[0]
        member = np.empty((self.k, nPoints))
        for iPoint, curPoint in enumerate(X):
            dissim = self.get_fuzzy_dissim(omega, curPoint)
            if np.any(dissim <= treshold):
                membership[:,iPoint] = np.where(dissim <= treshold, 1, treshold)
            else:
                # NOTE: squaring the distances is not in the paper, but in the code of Kim et al.;
                # supposedly improves performance
                dissim = dissim ** 2
                for ik, curc in enumerate(omega):
                    factor = 1. / (self.alpha - 1)
                    membership[ik,iPoint] = 1 / np.sum( (float(dissim[ik]) / dissim)**factor )
        return membership
    
    def update_centroids(self, X, member):
        omega = [[defaultdict(float) for _ in range(X.shape[1])] for _ in range(self.k)]
        for ik in range(self.k):
            for iAttr in range(X.shape[1]):
                for iPoint, curPoint in enumerate(X[:,iAttr]):
                    omega[ik][iAttr][curPoint] += membership[ik,iPoint] ** self.alpha
                # normalize so that sum omegas is 1, analogous to k-means
                # (see Yang et al. [2008] who explain better than the original paper)
                # NOTE: in their code, Kim et al. use a power of 2 instead of alpha
                # in the normalization step; probably does not make a big difference
                sumOmg = sum(omega[ik][iAttr].values())
                for key in omega[ik][iAttr].keys():
                    omega[ik][iAttr][key] /= sumOmg
        return omega
    
    @staticmethod
    def get_fuzzy_dissim(omega, x):
        # dissimilarity = sums of all omegas for non-matching attributes
        # see Eqs. 13-15 of Kim et al. [2004]
        dissim = np.zeros(len(omega))
        for ik in range(len(omega)):
            for iAttr in range(len(omega[ik])):
                nonMatch = [v for k, v in omega[ik][iAttr].items() if k != x[iAttr]]
                dissim[ik] += sum(nonMatch)
                # this is how the code of Kim et al. does it
                # dissim[ik] += sum(nonMatch) / np.sqrt(omega[ik][iat].items() ** 2)
        return dissim
    
    def clustering_cost(self, X, membership, omega):
        cost = 0
        for iPoint, curPoint in enumerate(X):
            cost += np.sum( self.get_fuzzy_dissim(omega, curPoint) * (membership[:,iPoint] ** self.alpha) )
        return cost


def key_for_max_value(d):
    # Fast method (supposedly) to get key for maximum value in dict.
    v = list(d.values())
    k = list(d.keys())
    return k[v.index(max(v))]

def opt_kmodes(k, X, preRuns=10, goodPctl=20, **kwargs):
    '''Shell around k-modes algorithm that tries to ensure a good clustering result
    by choosing one that has a relatively low clustering cost compared to the
    costs of a number of pre-runs. (Huang [1998] states that clustering cost can be
    used to judge the clustering quality.)
    
    Returns a (good) KModes class instantiation.
    
    '''
    
    if kwargs['initMethod'] == 'Cao' and kwargs['centUpd'] == 'mode':
        print("Hint: Cao initialization method + mode updates = deterministic. \
                No opt_kmodes necessary, run kmodes method directly instead.")
    
    preCosts = []
    print("Starting preruns...")
    for _ in range(preRuns):
        kmodes = KModes(k)
        kmodes.cluster(X, verbose=0, **kwargs)
        preCosts.append(kmodes.cost)
        print("Cost = {0}".format(kmodes.cost))
    
    while True:
        kmodes = KModes(k)
        kmodes.cluster(X, verbose=1, **kwargs)
        if kmodes.cost <= np.percentile(preCosts, goodPctl):
            print("Found a good clustering.")
            print("Cost = {0}".format(kmodes.cost))
            break
    
    return kmodes

if __name__ == "__main__":
    # reproduce results on small soybean data set
    X = np.genfromtxt('./soybean.csv', dtype='int64', delimiter=',')[:,:-1]
    y = np.genfromtxt('./soybean.csv', dtype='unicode', delimiter=',', usecols=35)
    
    # drop columns with single value
    X = X[:,np.std(X, axis=0) > 0.]
    
    kmodes_huang = KModes(4)
    kmodes_huang.cluster(X, init='Huang')
    kmodes_cao = KModes(4)
    kmodes_cao.cluster(X, init='Cao')
    kproto = KPrototypes(4)
    kproto.cluster(np.random.randn(X.shape[0], 3), X, init='Huang')
    fkmodes = FuzzyKModes(4, alpha=1.1                              )
    fkmodes.cluster(X)
    # TODO: Kim et al. [2004] report best results with alpha=1.8,
    # but I find about 1.01 to 1.3. higher than that: very poor results
    # what's going on?
    # alpha 1.05 --> high factor --> differences in distance to centroids
    # are blown up, forcing convergence
    ffkmodes = FuzzyClustersKModes(4, alpha=1.1)
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
    
