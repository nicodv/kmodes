"""
Tests for dissimilarity measures
"""

import unittest

import numpy as np
from sklearn.utils.testing import assert_equal, assert_array_equal

from kmodes.util.dissim import matching_dissim, euclidean_dissim, ng_dissim
from kmodes.util.dissim import jaccard_dissim_binary, jaccard_dissim_label


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

    def test_jaccard_dissim_binary(self):
        a = np.array([[0, 1, 1, 0, 1, 1]])
        b = np.array([[0, 1, 1, 0, 1, 0]])
        assert_equal(0.25, jaccard_dissim_binary(a, b))

        a = np.array([[0, 1, 1, 0, 1, 1]])
        b = np.array([[0, np.NaN, 1, 0, 1, 0]])
        with self.assertRaises(ValueError):
            jaccard_dissim_binary(a, b)

        # test where values are non binary but also not having np.NaN
        a = np.array([[0, 1, 2, 0, 1, 2]])
        b = np.array([[0, 1, 2, 0, 1, 0]])
        with self.assertRaises(ValueError):
            jaccard_dissim_binary(a, b)

        # test for dissimilarity = 0: sets are the same
        a = np.array([[1, 1, 0, 1, 1, 0]])
        b = np.array([[1, 1, 0, 1, 1, 0]])
        assert_equal(0, jaccard_dissim_binary(a, b))

        # test for dissimilarity = 1: sets are completely different
        a = np.array([[0, 0, 1, 0, 0, 1]])
        b = np.array([[1, 1, 0, 1, 1, 0]])
        assert_equal(1, jaccard_dissim_binary(a, b))

    def test_jaccard_dissim_label(self):
        a = np.array([[0, 1, 2, 0, 1, 2]])
        b = np.array([[0, 1, 2, 0, 3, 0]])
        assert_equal(0.25, jaccard_dissim_label(a, b))

        a = np.array([[np.NaN, 1, 2, 0, 1, 2]])
        b = np.array([[0, 1, 2, 0, 1, 0]])
        with self.assertRaises(ValueError):
            jaccard_dissim_label(a, b)

        # test for dissimilarity = 0: sets are the same
        a = np.array([[1, 2, 0, 3, 1, 0]])
        b = np.array([[1, 2, 0, 3, 1, 0]])
        assert_equal(0, jaccard_dissim_label(a, b))

        # test for dissimilarity = 1: sets are completely different
        a = np.array([[1, 2, 0, 3, 1, 0]])
        b = np.array([[5, 4, 6, 7, 8, 9]])
        assert_equal(1, jaccard_dissim_label(a, b))

    def test_euclidian_dissim(self):
        a = np.array([[0., 1., 2., 0., 1., 2.]])
        b = np.array([[3., 1., 3., 0., 1., 0.]])
        assert_equal(14., euclidean_dissim(a, b))

        a = np.array([[np.NaN, 1., 2., 0., 1., 2.]])
        b = np.array([[3., 1., 3., 0., 1., 0.]])
        with self.assertRaises(ValueError):
            euclidean_dissim(a, b)

    def test_ng_dissim(self):
        X = np.array([[0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 1]])
        centroids = X
        membship = np.array([[1, 0], [0, 1]])

        assert_array_equal(np.array([0., 1.]),
                           ng_dissim(centroids, X[0], X=X, membship=membship))
        assert_array_equal(np.array([1., 0.]),
                           ng_dissim(centroids, X[1], X=X, membship=membship))

        # Unit test for initialization (i.e., same as matching_dissim)
        membship = np.array([[0, 0], [0, 0]])
        mdiss_00 = matching_dissim(np.array([X[0]]), np.array([X[0]]))[0]
        mdiss_01 = matching_dissim(np.array([X[0]]), np.array([X[1]]))[0]
        mdiss_11 = matching_dissim(np.array([X[1]]), np.array([X[1]]))[0]

        assert_array_equal(np.array([mdiss_00, mdiss_01]),
                           ng_dissim(centroids, X[0], X=X, membship=membship))
        assert_array_equal(np.array([mdiss_01, mdiss_11]),
                           ng_dissim(centroids, X[1], X=X, membship=membship))

        # Unit test for NaN
        X = np.array([[np.NaN, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 1]])
        centroids = X
        membship = np.array([[1, 0], [0, 1]])

        assert_array_equal(np.array([1., 2.]),
                           ng_dissim(centroids, X[0], X=X, membship=membship))
        assert_array_equal(np.array([2., 0.]),
                           ng_dissim(centroids, X[1], X=X, membship=membship))

        # Unit test for initialization with NaN(i.e., same as matching_dissim)
        membship = np.array([[0, 0], [0, 0]])
        mdiss_00 = matching_dissim(np.array([X[0]]), np.array([X[0]]))[0]
        mdiss_01 = matching_dissim(np.array([X[0]]), np.array([X[1]]))[0]
        mdiss_11 = matching_dissim(np.array([X[1]]), np.array([X[1]]))[0]

        assert_array_equal(np.array([mdiss_00, mdiss_01]),
                           ng_dissim(centroids, X[0], X=X, membship=membship))
        assert_array_equal(np.array([mdiss_01, mdiss_11]),
                           ng_dissim(centroids, X[1], X=X, membship=membship))

        X = np.array([['a', 'b', 'c', 'd'], ['a', 'b', 'e', 'd'], ['d', 'c', 'b', 'a']])
        centroids = np.array([['a', 'b', 'c', 'd'], ['d', 'c', 'b', 'a']])
        membship = np.array([[1, 1, 0], [0, 0, 1]])

        assert_array_equal(np.array([0.5, 4.]),
                           ng_dissim(centroids, X[0], X=X, membship=membship))
        assert_array_equal(np.array([1., 4.]),
                           ng_dissim(centroids, X[1], X=X, membship=membship))
        assert_array_equal(np.array([4., 0.]),
                           ng_dissim(centroids, X[2], X=X, membship=membship))

        # Unit test for initialization (i.e., same as matching_dissim)
        membship = np.array([[0, 0, 0], [0, 0, 0]])
        mdiss_00 = matching_dissim(np.array([X[0]]), np.array([X[0]]))[0]
        mdiss_01 = matching_dissim(np.array([X[0]]), np.array([X[1]]))[0]
        mdiss_11 = matching_dissim(np.array([X[1]]), np.array([X[1]]))[0]
        mdiss_02 = matching_dissim(np.array([X[0]]), np.array([X[2]]))[0]
        mdiss_12 = matching_dissim(np.array([X[0]]), np.array([X[2]]))[0]
        mdiss_22 = matching_dissim(np.array([X[2]]), np.array([X[2]]))[0]

        assert_array_equal(np.array([mdiss_00, mdiss_02]),
                           ng_dissim(centroids, X[0], X=X, membship=membship))
        assert_array_equal(np.array([mdiss_01, mdiss_12]),
                           ng_dissim(centroids, X[1], X=X, membship=membship))
        assert_array_equal(np.array([mdiss_12, mdiss_22]),
                           ng_dissim(centroids, X[2], X=X, membship=membship))
