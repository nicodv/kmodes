"""
Dissimilarity measures for clustering
"""

import numpy as np

def matching_dissim(a: np.array, b: np.array, weights: list = False, **_):
    """
    simple matching dissimilarity function as expressed at
    https://dergipark.org.tr/tr/download/article-file/714584.

    Parameters
    ----------
    - a (numpy.ndarray): The first input array.
    - b (numpy.ndarray): The second input array.
    - weights (list, optional): A list of weights to apply to the dissimilarity calculation. Default is False.
    - **_ (dict): Additional keyword arguments. (Unused in the function)

    Returns
    -------
    - numpy.ndarray: An array containing the distance Manhattan dissimilarity between corresponding elements of `a` and `b`.

    Note:
    - This distance counts the number of categorical variables matches between both vectors
    - If `weights` is provided, it is used to weigh the dissimilarity for each element in the arrays.
      The length of `weights` should match the number of elements in the arrays.
    """  # noqa: D205, D401
    if weights:
        distance = np.sum((a != b).astype(int) * weights, axis=1)
    else:
        distance = np.sum(a != b, axis=1)
    return distance

def jaccard_dissim_binary(a, b, **__):
    """Jaccard dissimilarity function for binary encoded variables"""
    if ((a == 0) | (a == 1)).all() and ((b == 0) | (b == 1)).all():
        numerator = np.sum(np.bitwise_and(a, b), axis=1)
        denominator = np.sum(np.bitwise_or(a, b), axis=1)
        if (denominator == 0).any(0):
            raise ValueError("Insufficient Number of data since union is 0")
        return 1 - numerator / denominator
    raise ValueError("Missing or non Binary values detected in Binary columns.")


def jaccard_dissim_label(a, b, **__):
    """Jaccard dissimilarity function for label encoded variables"""
    if np.isnan(a.astype('float64')).any() or np.isnan(b.astype('float64')).any():
        raise ValueError("Missing values detected in Numeric columns.")
    intersect_len = np.empty(len(a), dtype=int)
    union_len = np.empty(len(a), dtype=int)
    ii = 0
    for row in a:
        intersect_len[ii] = len(np.intersect1d(row, b))
        union_len[ii] = len(np.unique(row)) + len(np.unique(b)) - intersect_len[ii]
        ii += 1
    if (union_len == 0).any():
        raise ValueError("Insufficient Number of data since union is 0")
    return 1 - intersect_len / union_len



def euclidean_dissim(a: np.array, b: np.array, weights: list = None, **_):
    """
    Calculate the Euclidean squared distance (no square root) dissimilarity between two arrays.

    Parameters
    ----------
    - a (numpy.ndarray): The first input array.
    - b (numpy.ndarray): The second input array.
    - weights (list, optional): A list of weights to apply to the dissimilarity calculation. Default is None.
    - **_ (dict): Additional keyword arguments. (Unused in the function)

    Returns
    -------
    - numpy.ndarray: An array containing the Euclidean squared distance dissimilarity between corresponding
      elements of `a` and `b`.

    Raises
    ------
    - ValueError: If missing values (NaN) are detected in numerical columns of either input array.

    Note:
    - The Euclidean distance dissimilarity is computed as the square root of the sum of squared differences
    between the two arrays.
    - If `weights` is provided, it is used to weigh the dissimilarity for each element in the arrays.
      The length of `weights` should match the number of elements in the arrays.

    """
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("Missing values detected in numerical columns.")
    elif weights:
        distance = np.sum(((a - b) ** 2) * weights, axis=1)
    else:
        distance = np.sum((a - b) ** 2, axis=1)
    return distance


def ng_dissim(a, b, X=None, membship=None, **_):
    """Ng et al.'s dissimilarity measure, as presented in
    Michael K. Ng, Mark Junjie Li, Joshua Zhexue Huang, and Zengyou He, "On the
    Impact of Dissimilarity Measure in k-Modes Clustering Algorithm", IEEE
    Transactions on Pattern Analysis and Machine Intelligence, Vol. 29, No. 3,
    January, 2007

    This function can potentially speed up training convergence.

    Note that membship must be a rectangular array such that the
    len(membship) = len(a) and len(membship[i]) = X.shape[1]

    In case of missing membship, this function reverts back to
    matching dissimilarity (e.g., when predicting).
    """
    # Without membership, revert to matching dissimilarity
    if membship is None:
        return matching_dissim(a, b)

    def calc_cjr(b, X, memj, idr):
        """Num objects w/ category value x_{i,r} for rth attr in jth cluster"""
        xcids = np.where(memj == 1)
        return float((np.take(X, xcids, axis=0)[0][:, idr] == b[idr]).sum(0))

    def calc_dissim(b, X, memj, idr):
        # Size of jth cluster
        cj = float(np.sum(memj))
        return (1.0 - (calc_cjr(b, X, memj, idr) / cj)) if cj != 0.0 else 0.0

    if len(membship) != a.shape[0] and len(membship[0]) != X.shape[1]:
        raise ValueError("'membship' must be a rectangular array where "
                         "the number of rows in 'membship' equals the "
                         "number of rows in 'a' and the number of "
                         "columns in 'membship' equals the number of rows in 'X'.")

    return np.array([np.array([calc_dissim(b, X, membship[idj], idr)
                               if b[idr] == t else 1.0
                               for idr, t in enumerate(val_a)]).sum(0)
                     for idj, val_a in enumerate(a)])
