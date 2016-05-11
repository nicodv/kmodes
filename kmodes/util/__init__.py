"""
Generic utilities for clustering
"""

# Author: 'Nico de Vos' <njdevos@gmail.com>
# License: MIT

import numpy as np


def get_max_value_key(dic):
    """Fast method to get key for maximum value in dict."""
    v = list(dic.values())
    k = list(dic.keys())
    return k[v.index(max(v))]


def encode_features(X, enc_map=None):
    """Converts categorical values in each column of X to integers in the range
    [0, n_unique_values_in_column - 1], if X is not already of integer type.
    Unknown values get a value of -1.

    If mapping is not provided, it is calculated based on the valus in X.
    """
    if np.issubdtype(X.dtype, np.integer):
        # Already integer type. do nothing.
        return X, enc_map

    if enc_map is None:
        fit = True
        # We will calculate enc_map, so initialize the list of column mappings.
        enc_map = []
    else:
        fit = False

    Xenc = np.zeros(X.shape).astype('int')
    for ii in range(X.shape[1]):
        if fit:
            enc_map.append({val: jj for jj, val in enumerate(np.unique(X[:, ii]))})
        # Unknown categories when predicting all get a value of -1.
        Xenc[:, ii] = np.array([enc_map[ii].get(x, -1) for x in X[:, ii]])

    return Xenc, enc_map
