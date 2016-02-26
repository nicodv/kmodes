"""
Tests for k-prototypes clustering algorithm
"""

import pickle

from sklearn.utils.testing import assert_equal

from kmodes import kprototypes


def test_pickle():
    obj = kprototypes.KPrototypes()
    s = pickle.dumps(obj)
    assert_equal(type(pickle.loads(s)), obj.__class__)
