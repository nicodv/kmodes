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
        
        Inputs:     k       = number of clusters
        Attributes: Xclust  = cluster numbers [no. points]
                    cent    = centroids [k * no. attributes]
                    cost    = clustering cost, defined as the sum distance of
                              all points to their respective clusters
        
        '''
        assert k > 1, "Choose at least 2 clusters."
        self.k = k
        
        # generalized form with alpha. alpha > 1 for fuzzy k-modes
        self.alpha = 1
    
    def cluster(self, X, init='Huang', maxIters=100, verbose=1):
        '''Inputs:  X           = data points [no. points * no. attributes]
                    init        = initialization method ('Huang' for the one described in
                                  Huang [1998], 'Cao' for the one in Cao et al. [2009])
                    maxIters    = maximum no. of iterations
        '''
        # convert to numpy array, if needed
        X = np.asanyarray(X)
        N, at = X.shape
        assert self.k < N, "More clusters than data points?"
        
        self.init = init
        
        # ----------------------
        #    INIT
        # ----------------------
        print("Init: initializing centroids")
        cent = self.init_centroids(X)
        
        print("Init: initializing clusters")
        member = np.zeros((self.k, N), dtype='int64')
        # clustFreq is a list of lists with dictionaries that contain the
        # frequencies of values per cluster and attribute
        clustFreq = [[defaultdict(int) for _ in range(at)] for _ in range(self.k)]
        for iN, curx in enumerate(X):
            # initial assigns to clusters
            dissim = self.get_dissim(cent, curx)
            cluster = np.argmin(dissim)
            member[cluster,iN] = 1
            # count attribute values per cluster
            for iat, val in enumerate(curx):
                clustFreq[cluster][iat][val] += 1
        # perform an initial centroid update
        for ik in range(self.k):
            for iat in range(at):
                cent[ik,iat] = key_for_max_value(clustFreq[ik][iat])
        
        # ----------------------
        #    ITERATION
        # ----------------------
        print("Starting iterations...")
        itr = 0
        converged = False
        while itr <= maxIters and not converged:
            itr += 1
            moves = 0
            for iN, curx in enumerate(X):
                dissim = self.get_dissim(cent, curx)
                cluster = np.argmin(dissim)
                # if necessary: move point, and update old/new cluster frequencies and centroids
                if not member[cluster, iN]:
                    moves += 1
                    oldcluster = np.argwhere(member[:,iN])
                    member[oldcluster,iN] = 0
                    member[cluster,iN] = 1
                    for iat, val in enumerate(curx):
                        # update frequencies of attributes in clusters
                        clustFreq[cluster][iat][val] += 1
                        clustFreq[oldcluster][iat][val] -= 1
                        # update new and old centroids by choosing most likely attribute
                        for curc in (cluster, oldcluster):
                            cent[curc, iat] = key_for_max_value(clustFreq[curc][iat])
                    if verbose == 2:
                        print("Move from cluster {0} to {1}".format(oldcluster, cluster))
            
            # all points seen in this iteration
            converged = (moves == 0)
            if verbose:
                print("Iteration: {0}/{1}, moves: {2}".format(itr, maxIters, moves))
        
        self.cost = self.clustering_cost(X, cent, member)
        self.cent = cent
        self.Xclust = np.array([int(np.argwhere(member[:,x])) for x in range(N)])
        self.member = member
    
    def init_centroids(self, X):
        assert self.init in ('Huang', 'Cao')
        N, at = X.shape
        cent = np.empty((self.k, at))
        if self.init == 'Huang':
            # determine frequencies of attributes
            for iat in range(at):
                freq = defaultdict(int)
                for val in X[:,iat]:
                    freq[val] += 1
                # sample centroids using the probabilities of attributes
                # (I assume that's what's meant in the Huang [1998] paper; it works, at least)
                # note: sampling using population in static list with as many choices as
                # frequency counts this works well since (1) we re-use the list k times here,
                # and (2) the counts are small integers so memory consumption is low
                choices = [chc for chc, wght in freq.items() for _ in range(wght)]
                for ik in range(self.k):
                    cent[ik, iat] = random.choice(choices)
            # the previously chosen centroids could result in empty clusters,
            # so set centroid to closest point in X
            for ik in range(self.k):
                dissim = self.get_dissim(X, cent[ik])
                ndx = dissim.argsort()
                # and we want the centroid to be unique
                while np.all(X[ndx[0]] == cent, axis=1).any():
                    ndx = np.delete(ndx, 0)
                cent[ik] = X[ndx[0]]
        elif self.init == 'Cao':
            # Note: O(N * at * k**2), so watch out with k
            # determine densities points
            dens = np.zeros(N)
            for iat in range(at):
                freq = defaultdict(int)
                for val in X[:,iat]:
                    freq[val] += 1
                for iN in range(N):
                    dens[iN] += freq[X[iN,iat]] / float(at)
            dens /= N
            
            # choose centroids based on distance and density
            cent[0] = X[np.argmax(dens)]
            dissim = self.get_dissim(X, cent[0])
            cent[1] = X[np.argmax(dissim * dens)]
            # for the reamining centroids, choose max dens * dissim to the (already assigned)
            # centroid with the lowest dens * dissim
            for ic in range(2,self.k):
                dd = np.empty((ic, N))
                for icp in range(ic):
                    dd[icp] = self.get_dissim(X, cent[icp]) * dens
                cent[ic] = X[np.argmax(np.min(dd, axis=0))]
        
        return cent
    
    def get_dissim(self, A, b):
        # TODO: add other dissimilarity measures?
        # simple matching dissimilarity
        return (A != b).sum(axis=1)
    
    def clustering_cost(self, X, cent, member):
        cost = 0
        for iN, curx in enumerate(X):
            cost += np.sum( self.get_dissim(cent, curx) * (member[:,iN] ** self.alpha) )
        return cost

####################################################################################################

class KPrototypes(KModes):
    
    def __init__(self, k):
        '''k-protoypes clustering algorithm for mixed numeric and categorical data.
        See:
        Huang, Z.: Clustering large data sets with mixed numeric and categorical values,
        Proceedings of the First Pacific Asia Knowledge Discovery and Data Mining Conference,
        Singapore: World Scientific, pp. 21–34, 1997.
        
        Inputs:     k       = number of clusters
        Attributes: Xclust  = cluster numbers [no. points]
                    cent    = centroids, two lists (num. and cat.) with [k * no. attributes]
                    cost    = clustering cost, defined as the sum distance of
                              all points to their respective clusters
        
        '''
        super(KPrototypes, self).__init__(k)
    
    def cluster(self, Xnum, Xcat, gamma=None, init='Huang', maxIters=100, verbose=1):
        '''Inputs:  Xnum        = numeric data points [no. points * no. numeric attributes]
                    Xcat        = categorical data points [no. points * no. numeric attributes]
                    gamma       = weighing factor that determines relative importance of
                                  num./cat. attributes (see discussion in Huang [1997])
                    init        = initialization method ('Huang' for the one described in
                                  Huang [1998], 'Cao' for the one in Cao et al. [2009])
                    maxIters    = maximum no. of iterations
        '''
        # convert to numpy arrays, if needed
        Xnum = np.asanyarray(Xnum)
        Xcat = np.asanyarray(Xcat)
        Nnum, atnum = Xnum.shape
        Ncat, atcat = Xcat.shape
        assert Nnum == Ncat, "More numerical points than categorical?"
        N = Nnum
        assert self.k < Nnum, "More clusters than data points?"
        
        self.init = init
        
        # estimate a good value for gamma, which determines the weighing of
        # categorical values in clusters (see Huang [1997])
        if gamma is None:
            gamma = 0.5 * np.std(Xnum)
        
        # ----------------------
        #    INIT
        # ----------------------
        print("Init: initializing centroids")
        cent = [np.mean(Xnum, axis=1) + np.random.randn((k, at)) * np.std(Xnum, axis=1), 
                self.init_centroids(Xcat)]
        
        print("Init: initializing clusters")
        member = np.zeros((self.k, N), dtype='int64')
        # keep track of the sum of attribute values per cluster
        clustSum = np.zeros((self.k, at), dtype='float')
        # clustFreq is a list of lists with dictionaries that contain the
        # frequencies of values per cluster and attribute
        clustFreq = [[defaultdict(int) for _ in range(at)] for _ in range(self.k)]
        for iN in range(N):
            # initial assigns to clusters
            dissim = self.get_dissim_num(cent[0], Xnum[iN]) + \
                     gamma * self.get_dissim_cat(cent[1], Xcat[iN])
            cluster = np.argmin(dissim)
            member[cluster,iN] = 1
            # count attribute values per cluster
            for iat, val in enumerate(Xnum[iN]):
                clustSum[cluster,iat] += val
            for iat, val in enumerate(Xcat[iN]):
                clustFreq[cluster][iat][val] += 1
        # perform an initial centroid update
        for ik in range(self.k):
            for iat in range(at):
                cent[0][ik,iat] = clustSum[ik,iat] / sum(member[ik,:])
                cent[1][ik,iat] = key_for_max_value(clustFreq[ik][iat])
        
        # ----------------------
        #    ITERATION
        # ----------------------
        print("Starting iterations...")
        itr = 0
        converged = False
        while itr <= maxIters and not converged:
            itr += 1
            moves = 0
            for iN in range(N):
                dissim = self.get_dissim_num(cent[0], Xnum[iN]) + \
                         gamma * self.get_dissim_cat(cent[1], Xcat[iN])
                cluster = np.argmin(dissim)
                # if necessary: move point, and update old/new cluster frequencies and centroids
                if not member[cluster, iN]:
                    moves += 1
                    oldcluster = np.argwhere(member[:,iN])
                    member[oldcluster,iN] = 0
                    member[cluster,iN] = 1
                    for iat, val in enumerate(Xnum[iN]):
                        clustSum[cluster,iat] += val
                        clustSum[oldcluster,iat] -= val
                        for curc in (cluster, oldcluster):
                            cent[0][curc, iat] = clustSum[ik,iat] / sum(member[curc,:])
                    for iat, val in enumerate(Xcat[iN]):
                        # update frequencies of attributes in clusters
                        clustFreq[cluster][iat][val] += 1
                        clustFreq[oldcluster][iat][val] -= 1
                        # update new and old centroids by choosing most likely attribute
                        for curc in (cluster, oldcluster):
                            cent[1][curc, iat] = key_for_max_value(clustFreq[curc][iat])
                    if verbose == 2:
                        print("Move from cluster {0} to {1}".format(oldcluster, cluster))
            
            # all points seen in this iteration
            converged = (moves == 0)
            if verbose:
                print("Iteration: {0}/{1}, moves: {2}".format(itr, maxIters, moves))
        
        self.cost = self.clustering_cost(X, cent, member)
        self.cent = cent
        self.Xclust = np.array([int(np.argwhere(member[:,x])) for x in range(N)])
        self.member = member
    
    def get_dissim_num(self, Anum, b):
        # Euclidian distance
        return np.sum((Anum - b)**2, axis=1)
    
    def get_dissim_cat(self, Acat, b):
        # simple matching dissimilarity
        return  np.sum(Acat != b,axis=1)
    
    def clustering_cost(self, X, cent, member, gamma):
        cost = 0
        for iN, curx in enumerate(Xnum):
            ncost += np.sum( self.get_dissim_num(cent, curx) * (member[:,iN] ** self.alpha) )
        for iN, curx in enumerate(Xcat):
            ncost += np.sum( self.get_dissim_cat(cent, curx) * (member[:,iN] ** self.alpha) )
        return ncost + gamma * ccost

###################################################################################################

class FuzzyKModes(KModes):
    
    def __init__(self, k, alpha=1.5):
        '''Fuzzy k-modes clustering algorithm for categorical data.
        Uses traditional, hard centroids, following Huang, Z., Ng, M.K.:
        A fuzzy k-modes algorithm for clustering categorical data, 
        IEEE Transactions on Fuzzy Systems 7(4), 1999.
        
        Inputs:     k       = number of clusters
                    alpha   = alpha coefficient
        Attributes: Xclust  = cluster numbers with max. membership [no. points]
                    member  = membership matrix [k * no. points]
                    cent    = centroids [k * no. attributes]
                    cost    = clustering cost
        
        '''
        super(FuzzyKModes, self).__init__(k)
        
        assert alpha > 1, "alpha should be > 1 (alpha = 1 equals regular k-modes)."
        self.alpha = alpha
        
    def cluster(self, X, init='Huang', maxIters=200, tol=0.001, costInter=1, verbose=1):
        '''Inputs:  X           = data points [no. points * no. attributes]
                    init        = initialization method ('Huang' for the one described in
                                  Huang [1998], 'Cao' for the one in Cao et al. [2009]).
                    maxIters    = maximum no. of iterations
                    tol         = tolerance for termination criterion
                    costInter   = frequency with which to check the total cost
                                  (for speeding things up, since it is computationally expensive)
        
        '''
        
        # convert to numpy array, if needed
        X = np.asanyarray(X)
        N, at = X.shape
        assert self.k < N, "More clusters than data points?"
        
        self.init = init
    
        # ----------------------
        #    INIT
        # ----------------------
        print("Init: initializing centroids")
        cent = self.init_centroids(X)
        
        # store for all attributes which points have a certain attribute value
        domAtX = [defaultdict(list) for _ in range(at)]
        for iN, curx in enumerate(X):
            for iat, val in enumerate(curx):
                domAtX[iat][val].append(iN)
        
        # ----------------------
        #    ITERATION
        # ----------------------
        print("Starting iterations...")
        itr = 0
        converged = False
        lastCost = np.inf
        while itr <= maxIters and not converged:
            member = self.update_membership(cent, X)
            for ik in range(self.k):
                cent[ik] = self.update_centroid(domAtX, member[ik])
            
            # computationally expensive, only check every N steps
            if itr % costInter == 0:
                cost = self.clustering_cost(X, cent, member)
                converged = cost >= lastCost * (1-tol)
                lastCost = cost
                if verbose:
                    print("Iteration: {0}/{1}, cost: {2}".format(itr, maxIters, cost))
            itr += 1
        
        self.cost = cost
        self.cent = cent
        self.Xclust = np.array([int(np.argmax(member[:,x])) for x in range(N)])
        self.member = member
    
    def update_membership(self, cent, X):
        N = X.shape[0]
        member = np.empty((self.k, N))
        for iN, curx in enumerate(X):
            dissim = self.get_dissim(cent, curx)
            if np.any(dissim == 0):
                member[:,iN] = np.where(dissim == 0, 1, 0)
            else:
                for ik, curc in enumerate(cent):
                    factor = 1. / (self.alpha - 1)
                    member[ik,iN] = 1 / np.sum( (float(dissim[ik]) / dissim)**factor )
        return member
    
    def update_centroid(self, domAtX, member):
        cent = []
        for iat in range(len(domAtX)):
            # return attribute that maximizes the sum of the memberships
            v = list(domAtX[iat].values())
            k = list(domAtX[iat].keys())
            memvar = [sum(member[x]**self.alpha) for x in v]
            cent.append(k[np.argmax(memvar)])
        return np.array(cent)

###################################################################################################

class FuzzyFuzzyKModes(KModes):
    
    def __init__(self, k, alpha=1.5):
        '''Fuzzy k-modes clustering algorithm for categorical data.
        Uses fuzzy centroids, following and Kim, D.-W., Lee, K.H., Lee, D.:
        Fuzzy clustering of categorical data using fuzzy centroids, Pattern
        Recognition Letters 25, 1262-1271, 2004.
        
        Inputs:     k       = number of clusters
                    alpha   = alpha coefficient
        Attributes: Xclust  = cluster numbers with max. membership [no. points]
                    member  = membership matrix [k * no. points]
                    omega   = fuzzy centroids [dicts with element values as keys,
                              element memberships as values, inside lists for 
                              attributes inside list for centroids]
                    cost    = clustering cost
        
        '''
        super(FuzzyFuzzyKModes, self).__init__(k)
        
        assert alpha > 1, "alpha should be > 1 (alpha = 1 equals regular k-modes)."
        self.alpha = alpha
    
    def cluster(self, X, maxIters=200, tol = 0.001, costInter=1, verbose=1):
        '''Inputs:  X           = data points [no. points * no. attributes]
                    maxIters    = maximum no. of iterations
                    tol         = tolerance for termination criterion
                    costInter   = frequency with which to check the total cost
                                  (for speeding things up, since it is computationally expensive)
        
        '''
        
        # convert to numpy array, if needed
        X = np.asanyarray(X)
        N, at = X.shape
        assert self.k < N, "More clusters than data points?"
        
        # ----------------------
        #    INIT
        # ----------------------
        # count all attributes
        freqAt = [defaultdict(int) for _ in range(at)]
        for curx in X:
            for iat, val in enumerate(curx):
                freqAt[iat][val] += 1
        
        # omega = fuzzy set (as dict) for each attribute per cluster
        omega = [[{} for _ in range(at)] for _ in range(self.k)]
        for ik in range(self.k):
            for iat in range(at):
                rands = np.random.rand(len(freqAt[iat]))
                # to make sum omegas = 1 per attribute
                rands = rands / sum(rands)
                for ival, val in enumerate(freqAt[iat]):
                    omega[ik][iat][val] = rands[ival]
        
        # ----------------------
        #    ITERATION
        # ----------------------
        print("Starting iterations...")
        itr = 0
        converged = False
        lastCost = np.inf
        while itr <= maxIters and not converged:
            # O(k*N*at*no. of unique values)
            member = self.update_membership(X, omega)
            
            # O(k*N*at)
            omega = self.update_centroids(X, member)
            
            # computationally expensive, only check every N steps
            if itr % costInter == 0:
                cost = self.clustering_cost(X, member, omega)
                converged = cost >= lastCost * (1-tol)
                lastCost = cost
                if verbose:
                    print("Iteration: {0}/{1}, cost: {2}".format(itr, maxIters, cost))
            itr += 1
        
        self.cost = cost
        self.omega = omega
        self.Xclust = np.array([int(np.argmax(member[:,x])) for x in range(N)])
        self.member = member
    
    def update_membership(self, X, omega):
        # Eq. 20 from Kim et al. [2004]
        N = X.shape[0]
        member = np.empty((self.k, N))
        for iN, curx in enumerate(X):
            dissim = self.get_fuzzy_dissim(omega, curx)
            for ik, curc in enumerate(omega):
                factor = 1. / (self.alpha - 1)
                member[ik,iN] = 1 / np.sum( (float(dissim[ik]) / dissim)**factor )
        return member
    
    def update_centroids(self, X, member):
        omega = [[defaultdict(float) for _ in range(X.shape[1])] for _ in range(self.k)]
        for ik in range(self.k):
            for iat in range(X.shape[1]):
                for iN, curx in enumerate(X[:,iat]):
                    omega[ik][iat][curx] += member[ik,iN] ** self.alpha
                # normalize (see Yang et al. [2008] for clearer explanation)
                sumomg = sum(omega[ik][iat].values())
                for k in omega[ik][iat].keys():
                    omega[ik][iat][k] /= sumomg
        return omega
    
    def get_fuzzy_dissim(self, omega, x):
        # dissimilarity = sums of all omegas for non-matching attributes
        # see Eqs. 13-15 of Kim et al. [2004]
        dissim = np.zeros(len(omega))
        for ik in range(len(omega)):
            for iat in range(len(omega[ik])):
                nonMatch = [v for k, v in omega[ik][iat].items() if k != x[iat]]
                dissim[ik] += sum(nonMatch)
        return dissim
    
    def clustering_cost(self, X, member, omega):
        cost = 0
        for iN, curx in enumerate(X):
            cost += np.sum( self.get_fuzzy_dissim(omega, curx) * (member[:,iN] ** self.alpha) )
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
    
    if kwargs['init'] == 'Cao' and kwargs['centUpd'] == 'mode':
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
    
    #kmodes_h = opt_kmodes(4, X, preRuns=10, goodPctl=20, init='Huang', maxIters=100)
    kmodes_huang = KModes(4)
    kmodes_huang.cluster(X, init='Huang')
    kmodes_cao = KModes(4)
    kmodes_cao.cluster(X, init='Cao')
    kproto = KPrototypes(4)
    kproto.cluster(np.random.randn((X.shape[0], 3)), X, init='Huang')
    fkmodes = FuzzyKModes(4, alpha=1.1)
    fkmodes.cluster(X)
    ffkmodes = FuzzyFuzzyKModes(4, alpha=1.8)
    ffkmodes.cluster(X)
    
    for result in (kmodes_huang, kmodes_cao, fkmodes, ffkmodes):
        classtable = np.zeros((4,4), dtype='int64')
        for ii,_ in enumerate(y):
            classtable[int(y[ii][-1])-1,result.Xclust[ii]] += 1
        
        print("    | Cl. 1 | Cl. 2 | Cl. 3 | Cl. 4 |")
        print("----|-------|-------|-------|-------|")
        for ii in range(4):
            prargs = tuple([ii+1] + list(classtable[ii,:]))
            print(" D{0} |    {1:>2} |    {2:>2} |    {3:>2} |    {4:>2} |".format(*prargs))
    
