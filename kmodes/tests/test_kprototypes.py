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

    def test_kprotoypes_stocks(self):
        # Number/index of categoricals does not make sense
        kproto = kprototypes.KPrototypes(n_clusters=4, init='Cao', verbose=2)
        with self.assertRaises(AssertionError):
            kproto.fit_predict(STOCKS, categorical=[1, 3])
        with self.assertRaises(AssertionError):
            kproto.fit_predict(STOCKS, categorical=[0, 1, 2])

    def test_kprotoypes_huang_stocks(self):
        np.random.seed(42)
        kproto_huang = kprototypes.KPrototypes(n_clusters=4, n_init=1, init='Huang', verbose=2)
        result = kproto_huang.fit_predict(STOCKS, categorical=[1, 2])
        expected = np.array([0, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1])
        np.testing.assert_array_equal(result, expected)
        self.assertTrue(result.dtype == np.dtype(np.uint8))

    def test_kprotoypes_cao_stocks(self):
        np.random.seed(42)
        kproto_cao = kprototypes.KPrototypes(n_clusters=4, init='Cao', verbose=2)
        result = kproto_cao.fit_predict(STOCKS, categorical=[1, 2])
        expected = np.array([2, 3, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(result, expected)
        self.assertTrue(result.dtype == np.dtype(np.uint8))

    def test_kprotoypes_random_stocks(self):
        kproto_random = kprototypes.KPrototypes(n_clusters=4, init='random', verbose=2)
        result = kproto_random.fit(STOCKS, categorical=[1, 2])
        self.assertIsInstance(result, kprototypes.KPrototypes)

    def test_kprotoypes_init_stocks(self):
        # Wrong order
        init_vals = [
            np.array([[6, 2],
                      [5, 2],
                      [4, 2],
                      [3, 2]]),
            np.array([[382.27919457],
                      [350.76963718],
                      [13.31595618],
                      [540.50533708]])
        ]
        kproto_init = kprototypes.KPrototypes(n_clusters=4, init=init_vals, verbose=2)
        with self.assertRaises(AssertionError):
            kproto_init.fit_predict(STOCKS, categorical=[1, 2])

        init_vals = [
            np.array([[0.],
                      [0.],
                      [0.],
                      [0.]]),
            np.array([[6, 2],
                      [5, 2],
                      [4, 2],
                      [3, 2]])
        ]
        np.random.seed(42)
        kproto_init = kprototypes.KPrototypes(n_clusters=4, init=init_vals, verbose=2)
        result = kproto_init.fit_predict(STOCKS, categorical=[1, 2])
        expected = np.array([0, 2, 2, 2, 2, 3, 3, 3, 3, 1, 1, 1])
        np.testing.assert_array_equal(result, expected)
        self.assertTrue(result.dtype == np.dtype(np.uint8))


if __name__ == '__main__':
    unittest.main()
