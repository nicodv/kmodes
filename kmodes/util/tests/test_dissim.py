"""
Tests for dissimilarity measures
"""

import unittest

import numpy as np
from sklearn.utils.testing import assert_equal, assert_array_equal

from kmodes.util.dissim import matching_dissim, euclidean_dissim


class TestDissimilarityMeasures(unittest.TestCase):

    def test_matching_dissim(self):
        a = np.array([[0, 1, 2, 0, 1, 2]])
        b = np.array([[0, 1, 2, 0, 1, 0]])
        assert_equal(1, matching_dissim(a, b))

        a = np.array([[np.NaN, 1, 2, 0, 1, 2]])
        b = np.array([[0, 1, 2, 0, 1, 0]])
        assert_equal(2, matching_dissim(a, b))

        a = np.array([['a', 'b', 'c', 'd']])
        b = np.array([['a', 'b', 'c', 'd'], ['d', 'c', 'b', 'a']])
        assert_array_equal(np.array([0, 4]), matching_dissim(a, b))

    def test_euclidian_dissim(self):
        a = np.array([[0., 1., 2., 0., 1., 2.]])
        b = np.array([[3., 1., 3., 0., 1., 0.]])
        assert_equal(14., euclidean_dissim(a, b))

        a = np.array([[np.NaN, 1., 2., 0., 1., 2.]])
        b = np.array([[3., 1., 3., 0., 1., 0.]])
        with self.assertRaises(ValueError):
            euclidean_dissim(a, b)
