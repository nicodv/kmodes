"""
Dissimilarity measures for clustering
"""

# Author: 'Nico de Vos' <njdevos@gmail.com>
# License: MIT

import numpy as np

def matching_dissim(a, b, **kwargs):
    """Simple matching dissimilarity function"""
    return np.sum(a != b, axis=1)


def euclidean_dissim(a, b, **kwargs):
    """Euclidean distance dissimilarity function"""
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("Missing values detected in numerical columns.")
    return np.sum((a - b) ** 2, axis=1)


# Implementation of Ng et al.'s dissimilarity measure
# 	Michael K. Ng, Mark Junjie Li, Joshua Zhexue Huang, and Zengyou He, "On the
# 	Impact of Dissimilarity Measure in k-Modes Clustering Algorithm", IEEE
# 	Transactions on Pattern Analysis and Machine Intelligence, Vol. 29, No. 3,
# 	January, 2007
# Author: 'Ben Andow' <beandow@ncsu.edu>
def ng_dissim(a, b, **kwargs):
    X = kwargs['X']
    membship = kwargs['membship']
    return np.array([ np.array([calc_dissim(b, X, membship[idj], idr, idj) if b[idr] == t else 1.0 for idr,t in enumerate(val_a) ]).sum(0) for idj,val_a in enumerate(a) ])

# Num objects w/ category value x_{i,r} for rth attr in jth cluster
def calcCJR(b, X, memj, idr):
    xcids = np.where(np.in1d(memj.ravel(), [1]).reshape(memj.shape))
    return float((np.take(X, xcids, axis=0)[0][:, idr] == b[idr]).sum(0))

def calc_dissim(b, X, memj, idr, idj):
    CJ = float(np.sum(memj))           # Size of jth cluster
    return (1.0 - (calcCJR(b, X, memj, idr) / CJ)) if CJ != 0.0 else 0.0

