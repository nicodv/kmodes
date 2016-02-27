"""
Tests for k-prototypes clustering algorithm
"""

import pickle
import unittest

import numpy as np
from sklearn.utils.testing import assert_equal

from kmodes import kprototypes

STOCKS = np.array([
    [738.5, 'tech', 'USA'],
    [369.5, 'nrg', 'USA'],
    [368.2, 'tech', 'USA'],
    [346.7, 'tech', 'USA'],
    [343.5, 'fin', 'USA'],
    [282.4, 'fin', 'USA'],
    [282.1, 'tel', 'CN'],
    [279.7, 'cons', 'USA'],
    [257.2, 'cons', 'USA'],
    [205.2, 'tel', 'USA'],
    [192.1, 'tech', 'USA'],
    [195.7, 'nrg', 'NL']
])


class TestKProtoTypes(unittest.TestCase):

    def test_pickle(self):
        obj = kprototypes.KPrototypes()
        s = pickle.dumps(obj)
        assert_equal(type(pickle.loads(s)), obj.__class__)

    def test_kprotoypes_huang_stocks(self):
        np.random.seed(42)
        kproto_huang = kprototypes.KPrototypes(n_clusters=4, init='Huang', verbose=0)
        result = kproto_huang.fit_predict(STOCKS, categorical=[1, 2])
        expected = np.array([3, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
        np.testing.assert_array_equal(result, expected)

    def test_kprotoypes_cao_stocks(self):
        np.random.seed(42)
        kproto_cao = kprototypes.KPrototypes(n_clusters=4, init='Cao', verbose=0)
        result = kproto_cao.fit_predict(STOCKS, categorical=[1, 2])
        expected = np.array([2, 3, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(result, expected)


if __name__ == '__main__':
    unittest.main()
