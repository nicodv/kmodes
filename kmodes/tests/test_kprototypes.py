"""
Tests for k-prototypes clustering algorithm
"""

import pickle
import unittest

from sklearn.utils.testing import assert_equal

from kmodes import kprototypes


class TestKProtoTypes(unittest.TestCase):

    def test_pickle(self):
        obj = kprototypes.KPrototypes()
        s = pickle.dumps(obj)
        assert_equal(type(pickle.loads(s)), obj.__class__)


if __name__ == '__main__':
    unittest.main()
