"""
Tests for k-modes clustering algorithm
"""

import pickle

from sklearn.utils.testing import assert_equal

from kmodes.kmodes import KModes, get_max_value_key


def test_mode_from_dict():
    max_key = get_max_value_key({'a': 3, 'b': 10, 'c': -1, 'd': 9.9})
    assert_equal('b', max_key)


def test_pickle():
    obj = KModes()
    s = pickle.dumps(obj)
    assert_equal(type(pickle.loads(s)), obj.__class__)
