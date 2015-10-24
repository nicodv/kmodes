"""
Tests for k-prototypes clustering algorithm
"""

import pickle

from sklearn.utils.testing import assert_equal

from kmodes.kprototypes import KPrototypes


def test_pickle():
    obj = KPrototypes()
    s = pickle.dumps(obj)
    assert_equal(type(pickle.loads(s)), obj.__class__)
