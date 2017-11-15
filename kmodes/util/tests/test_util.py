"""
Tests for utils
"""

import unittest

import numpy as np
from sklearn.utils.testing import assert_equal, assert_array_equal

from kmodes.util import get_max_value_key, encode_features, get_unique_rows, \
    decode_centroids


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

SPOTTY_CAT = np.array([
    [0, 0],
    [0, 0],
    [0, 0],
    [1, 0],
    [np.NaN, 0],
    [9, 0],
    [9, 1],
    [np.NaN, 0],
    [8, 0],
    [8, 0],
    [0, 0],
    [0, 2]
])


class TestUtils(unittest.TestCase):

    def test_get_max_value_key(self):
        max_key = get_max_value_key({'a': 3, 'b': 10, 'c': -1, 'd': 9.9})
        assert_equal('b', max_key)

        # Make sure minimum key is consistently selected for equal values.
        max_key = get_max_value_key({'d': 10, 'c': 10, 'b': 10, 'a': 10})
        assert_equal('a', max_key)

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

    def test_missing_encode_features(self):
        X_enc, enc_map = encode_features(SPOTTY_CAT)
        expected_X = np.array([[0, 0],
                               [0, 0],
                               [0, 0],
                               [1, 0],
                               [-1, 0],
                               [3, 0],
                               [3, 1],
                               [-1, 0],
                               [2, 0],
                               [2, 0],
                               [0, 0],
                               [0, 2]])
        assert_array_equal(X_enc, expected_X)
        self.assertEqual(enc_map,
                         [{0.: 0, 1.: 1, 8.: 2, 9.: 3},
                          {0.: 0, 1.: 1, 2.: 2}])

    def test_get_unique_rows(self):
        result = get_unique_rows(STOCKS_CAT)
        expected = np.array([
            ['tel', 'USA'],
            ['tel', 'CN'],
            ['nrg', 'USA'],
            ['nrg', 'NL'],
            ['tech', 'USA'],
            ['cons', 'USA'],
            ['fin', 'USA'],
        ])
        # Check if each row is found exactly 1 time.
        for row in expected:
            mask = result == row
            matches = np.where(np.all(mask, axis=1))
            self.assertEqual(len(matches), 1)

    def test_decode_centroids(self):
        enc = np.array([[3, 2],
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
        mapping = [{'cons': 0, 'fin': 1, 'nrg': 2, 'tech': 3, 'tel': 4},
                   {'CN': 0, 'NL': 1, 'USA': 2}]
        res = decode_centroids(enc, mapping)
        assert_array_equal(res, STOCKS_CAT)
