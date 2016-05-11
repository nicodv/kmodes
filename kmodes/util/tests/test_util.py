"""
Tests for dissimilarity measures
"""

import unittest

import numpy as np
from sklearn.utils.testing import assert_equal, assert_array_equal

from kmodes.util import get_max_value_key, encode_features


STOCKS_CAT = np.array([
    ['tech', 'USA'],
    ['nrg', 'USA'],
    ['tech', 'USA'],
    ['tech', 'USA'],
    ['fin', 'USA'],
    ['fin', 'USA'],
    ['tel', 'CN'],
    ['cons', 'USA'],
    ['cons', 'USA'],
    ['tel', 'USA'],
    ['tech', 'USA'],
    ['nrg', 'NL']
])


class TestDissimilarityMeasures(unittest.TestCase):

    def test_get_max_value_key(self):
        max_key = get_max_value_key({'a': 3, 'b': 10, 'c': -1, 'd': 9.9})
        assert_equal('b', max_key)

    def test_encode_features(self):
        X_enc, enc_map = encode_features(STOCKS_CAT)
        expected_X = np.array([[3, 2],
                               [2, 2],
                               [3, 2],
                               [3, 2],
                               [1, 2],
                               [1, 2],
                               [4, 0],
                               [0, 2],
                               [0, 2],
                               [4, 2],
                               [3, 2],
                               [2, 1]])
        assert_array_equal(X_enc, expected_X)
        self.assertEqual(enc_map,
                         [{'cons': 0, 'fin': 1, 'nrg': 2, 'tech': 3, 'tel': 4},
                          {'CN': 0, 'NL': 1, 'USA': 2}])
