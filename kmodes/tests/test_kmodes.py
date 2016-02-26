"""
Tests for k-modes clustering algorithm
"""

import pickle

import numpy as np
from sklearn.utils.testing import assert_equal, assert_raises, assert_array_equal

from kmodes import kmodes


def test_get_max_value_key():
    max_key = kmodes.get_max_value_key({'a': 3, 'b': 10, 'c': -1, 'd': 9.9})
    assert_equal('b', max_key)


def test_matching_dissim():
    a = np.array([0, 1, 2, 0, 1, 2])
    b = np.array([0, 1, 2, 0, 1, 0])
    assert_raises(AssertionError, kmodes.matching_dissim, a, b)

    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    assert_equal(1, kmodes.matching_dissim(a, b))

    a = np.array([['a', 'b', 'c', 'd']])
    b = np.array([['a', 'b', 'c', 'd'], ['d', 'c', 'b', 'a']])
    assert_array_equal((0, 4), kmodes.matching_dissim(a, b))


def test_pickle():
    obj = kmodes.KModes()
    s = pickle.dumps(obj)
    assert_equal(type(pickle.loads(s)), obj.__class__)
