"""
Tests for k-modes clustering algorithm
"""

import pickle
import unittest

import numpy as np
from sklearn.utils.testing import assert_equal

from kmodes.kmodes import KModes
from kmodes.util.dissim import ng_dissim, jaccard_dissim_binary, jaccard_dissim_label


SOYBEAN = np.array([
    [4, 0, 2, 1, 1, 1, 0, 1, 0, 2, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 3, 1, 1, 1, 0, 0, 0, 0,
     4, 0, 0, 0, 0, 0, 0, 'D1'],
    [5, 0, 2, 1, 0, 3, 1, 1, 1, 2, 1, 1, 0, 2, 2, 0, 0, 0, 1, 1, 3, 0, 1, 1, 0, 0, 0, 0,
     4, 0, 0, 0, 0, 0, 0, 'D1'],
    [3, 0, 2, 1, 0, 2, 0, 2, 1, 1, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 3, 0, 1, 1, 0, 0, 0, 0,
     4, 0, 0, 0, 0, 0, 0, 'D1'],
    [6, 0, 2, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 2, 2, 0, 0, 0, 1, 1, 3, 1, 1, 1, 0, 0, 0, 0,
     4, 0, 0, 0, 0, 0, 0, 'D1'],
    [4, 0, 2, 1, 0, 3, 0, 2, 0, 2, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 3, 1, 1, 1, 0, 0, 0, 0,
     4, 0, 0, 0, 0, 0, 0, 'D1'],
    [5, 0, 2, 1, 0, 2, 0, 1, 1, 0, 1, 1, 0, 2, 2, 0, 0, 0, 1, 1, 3, 1, 1, 1, 0, 0, 0, 0,
     4, 0, 0, 0, 0, 0, 0, 'D1'],
    [3, 0, 2, 1, 0, 2, 1, 1, 0, 1, 1, 1, 0, 2, 2, 0, 0, 0, 1, 1, 3, 0, 1, 1, 0, 0, 0, 0,
     4, 0, 0, 0, 0, 0, 0, 'D1'],
    [3, 0, 2, 1, 0, 1, 0, 2, 1, 2, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 3, 0, 1, 1, 0, 0, 0, 0,
     4, 0, 0, 0, 0, 0, 0, 'D1'],
    [6, 0, 2, 1, 0, 3, 0, 1, 1, 1, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 3, 1, 1, 1, 0, 0, 0, 0,
     4, 0, 0, 0, 0, 0, 0, 'D1'],
    [6, 0, 2, 1, 0, 1, 0, 1, 0, 2, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 3, 1, 1, 1, 0, 0, 0, 0,
     4, 0, 0, 0, 0, 0, 0, 'D1'],
    [6, 0, 0, 2, 1, 0, 2, 1, 0, 0, 1, 1, 0, 2, 2, 0, 0, 0, 1, 1, 0, 3, 0, 0, 0, 2, 1, 0,
     4, 0, 0, 0, 0, 0, 0, 'D2'],
    [4, 0, 0, 1, 0, 2, 3, 1, 1, 1, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 2, 1, 0,
     4, 0, 0, 0, 0, 0, 0, 'D2'],
    [5, 0, 0, 2, 0, 3, 2, 1, 0, 2, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 2, 1, 0,
     4, 0, 0, 0, 0, 0, 0, 'D2'],
    [6, 0, 0, 1, 1, 3, 3, 1, 1, 0, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 2, 1, 0,
     4, 0, 0, 0, 0, 0, 0, 'D2'],
    [3, 0, 0, 2, 1, 0, 2, 1, 0, 1, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 2, 1, 0,
     4, 0, 0, 0, 0, 0, 0, 'D2'],
    [4, 0, 0, 1, 1, 1, 3, 1, 1, 1, 1, 1, 0, 2, 2, 0, 0, 0, 1, 1, 0, 3, 0, 0, 0, 2, 1, 0,
     4, 0, 0, 0, 0, 0, 0, 'D2'],
    [3, 0, 0, 1, 0, 1, 2, 1, 0, 0, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 2, 1, 0,
     4, 0, 0, 0, 0, 0, 0, 'D2'],
    [5, 0, 0, 2, 1, 2, 2, 1, 0, 2, 1, 1, 0, 2, 2, 0, 0, 0, 1, 1, 0, 3, 0, 0, 0, 2, 1, 0,
     4, 0, 0, 0, 0, 0, 0, 'D2'],
    [6, 0, 0, 2, 0, 1, 3, 1, 1, 0, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 2, 1, 0,
     4, 0, 0, 0, 0, 0, 0, 'D2'],
    [5, 0, 0, 2, 1, 3, 3, 1, 1, 2, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 2, 1, 0,
     4, 0, 0, 0, 0, 0, 0, 'D2'],
    [0, 1, 2, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 2, 2, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 3,
     4, 0, 0, 0, 0, 0, 0, 'D3'],
    [2, 1, 2, 0, 0, 3, 1, 2, 0, 1, 1, 0, 0, 2, 2, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 3,
     4, 0, 0, 0, 0, 0, 0, 'D3'],
    [2, 1, 2, 0, 0, 2, 1, 1, 0, 2, 1, 0, 0, 2, 2, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 3,
     4, 0, 0, 0, 0, 0, 0, 'D3'],
    [0, 1, 2, 0, 0, 0, 1, 1, 1, 2, 1, 0, 0, 2, 2, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 3,
     4, 0, 0, 0, 0, 0, 0, 'D3'],
    [0, 1, 2, 0, 0, 2, 1, 1, 1, 1, 1, 0, 0, 2, 2, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 3,
     4, 0, 0, 0, 0, 0, 0, 'D3'],
    [4, 0, 2, 0, 1, 0, 1, 2, 0, 2, 1, 1, 0, 2, 2, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 3,
     4, 0, 0, 0, 0, 0, 0, 'D3'],
    [2, 1, 2, 0, 0, 3, 1, 2, 0, 2, 1, 0, 0, 2, 2, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 3,
     4, 0, 0, 0, 0, 0, 0, 'D3'],
    [0, 1, 2, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 2, 2, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 3,
     4, 0, 0, 0, 0, 0, 1, 'D3'],
    [3, 0, 2, 0, 1, 3, 1, 2, 0, 1, 1, 0, 0, 2, 2, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 3,
     4, 0, 0, 0, 0, 0, 0, 'D3'],
    [0, 1, 2, 0, 0, 1, 1, 2, 1, 2, 1, 0, 0, 2, 2, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 3,
     4, 0, 0, 0, 0, 0, 0, 'D3'],
    [2, 1, 2, 1, 1, 3, 1, 2, 1, 2, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 2, 2, 0, 1, 0, 0, 0, 3,
     4, 0, 0, 0, 0, 0, 1, 'D4'],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 0, 0, 3,
     4, 0, 0, 0, 0, 0, 1, 'D4'],
    [3, 1, 2, 0, 0, 1, 1, 2, 1, 0, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 2, 2, 0, 0, 0, 0, 0, 3,
     4, 0, 0, 0, 0, 0, 1, 'D4'],
    [2, 1, 2, 1, 1, 1, 1, 2, 0, 2, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 1, 2, 0, 1, 0, 0, 0, 3,
     4, 0, 0, 0, 0, 0, 1, 'D4'],
    [1, 1, 2, 0, 0, 3, 1, 1, 1, 2, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 2, 2, 0, 0, 0, 0, 0, 3,
     4, 0, 0, 0, 0, 0, 1, 'D4'],
    [1, 1, 2, 1, 0, 0, 1, 2, 1, 1, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 2, 2, 0, 0, 0, 0, 0, 3,
     4, 0, 0, 0, 0, 0, 1, 'D4'],
    [0, 1, 2, 1, 0, 3, 1, 1, 0, 0, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 0, 0, 3,
     4, 0, 0, 0, 0, 0, 1, 'D4'],
    [2, 1, 2, 0, 0, 1, 1, 2, 0, 0, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 0, 0, 3,
     4, 0, 0, 0, 0, 0, 1, 'D4'],
    [3, 1, 2, 0, 0, 2, 1, 2, 1, 1, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 2, 2, 0, 0, 0, 0, 0, 3,
     4, 0, 0, 0, 0, 0, 1, 'D4'],
    [3, 1, 1, 0, 0, 2, 1, 2, 1, 2, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 2, 2, 0, 0, 0, 0, 0, 3,
     4, 0, 0, 0, 0, 0, 1, 'D4'],
    [0, 1, 2, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 1, 2, 0, 1, 0, 0, 0, 3,
     4, 0, 0, 0, 0, 0, 1, 'D4'],
    [1, 1, 2, 1, 1, 3, 1, 2, 0, 1, 1, 1, 0, 2, 2, 0, 0, 0, 1, 1, 1, 2, 0, 1, 0, 0, 0, 3,
     4, 0, 0, 0, 0, 0, 1, 'D4'],
    [1, 1, 2, 0, 0, 0, 1, 2, 1, 0, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 2, 2, 0, 0, 0, 0, 0, 3,
     4, 0, 0, 0, 0, 0, 1, 'D4'],
    [1, 1, 2, 1, 1, 2, 3, 1, 1, 1, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 2, 2, 0, 1, 0, 0, 0, 3,
     4, 0, 0, 0, 0, 0, 1, 'D4'],
    [2, 1, 1, 0, 0, 3, 1, 2, 0, 2, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 0, 0, 3,
     4, 0, 0, 0, 0, 0, 1, 'D4'],
    [0, 1, 1, 1, 1, 2, 1, 2, 1, 0, 1, 1, 0, 2, 2, 0, 0, 0, 1, 1, 2, 2, 0, 1, 0, 0, 0, 3,
     4, 0, 0, 0, 0, 0, 1, 'D4'],
    [0, 1, 2, 1, 0, 3, 1, 1, 0, 2, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 0, 0, 3,
     4, 0, 0, 0, 0, 0, 1, 'D4'],
])
# Drop target column
SOYBEAN = SOYBEAN[:, :35]

SOYBEAN2 = np.array([
    [4, 0, 2, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 3, 0, 1, 1, 0, 0, 0, 0,
     4, 0, 0, 0, 0, 0, 0, 'D1'],
    [7, 0, 0, 2, 1, 0, 2, 1, 0, 0, 1, 1, 0, 2, 2, 0, 0, 0, 1, 1, 0, 3, 0, 0, 0, 2, 1, 0,
     4, 0, 0, 0, 0, 0, 0, 'D2'],
    [0, 1, 2, 0, 0, 1, 1, 1, 1, 2, 1, 0, 0, 2, 2, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 3,
     4, 0, 0, 0, 0, 0, 0, 'D3'],
    [2, 1, 2, 1, 1, 3, 1, 2, 1, 1, 1, 1, 0, 2, 2, 0, 0, 0, 1, 1, 1, 2, 0, 1, 0, 0, 0, 3,
     4, 0, 0, 0, 0, 0, 1, 'D4'],
])
# Drop target column
SOYBEAN2 = SOYBEAN2[:, :35]

# test data with categorical variables that have been label encoded
TEST_DATA = np.array([
    [2, 22, 14, 45,  2,  0,  1,  2,  5],
    [2, 13, 13, 19,  2,  0,  1,  2,  5],
    [3, 25,  4,  3,  0,  1,  2,  0,  4],
    [2, 13, 15, 18,  0,  1,  2,  2,  3],
    [3, 10,  4, 42,  0,  2,  1,  1,  2],
    [2, 16, 21, 14,  0,  1,  2,  2,  2],
    [2, 16, 19, 37,  0,  2,  1,  2,  2],
    [2, 20,  9, 34,  0,  1,  2,  3,  5],
    [2, 14, 21, 44,  0,  1,  2,  3,  2],
    [2, 26,  5, 30,  0,  1,  2,  3,  3],
    [3, 18, 17, 41,  3,  3,  3,  2,  0],
    [2, 20,  1, 27,  3,  3,  3,  2,  0],
    [3,  6,  8, 19,  0,  1,  2,  1,  2],
    [2, 13,  8, 41,  3,  3,  3,  2,  0],
    [2, 18, 17, 41,  3,  3,  3,  2,  0],
    [2, 16, 19, 42,  0,  1,  2,  2,  5],
    [7,  7,  5, 43,  0,  2,  1,  2,  2],
    [2, 18, 17, 41,  3,  3,  3,  2,  0],
    [3,  3,  5, 12,  3,  3,  3,  2,  0],
    [2, 18, 17, 41,  3,  3,  3,  2,  0],
    [7, 15, 19, 17,  0,  1,  2,  2,  2],
    [1,  1, 15, 24,  0,  1,  2,  2,  2],
    [2, 18, 17, 41,  3,  3,  3,  2,  0],
    [2,  5,  7,  9,  0,  1,  2,  3,  5],
    [2, 24,  6, 10,  0,  2,  1,  2,  2],
    [2, 13, 16, 29,  0,  2,  1,  2,  2],
    [3,  6,  8,  1,  0,  1,  2,  2,  5],
    [2, 16, 15, 34,  0,  1,  2,  2,  1],
    [0, 24, 14, 12,  3,  3,  3,  2,  0],
    [3,  8, 21, 13,  3,  3,  3,  2,  0],
    [2, 17, 15, 42,  3,  3,  3,  2,  0],
    [2, 25, 18, 16,  3,  3,  3,  2,  0],
    [2,  3, 15, 42,  3,  3,  3,  2,  0],
    [6, 13, 15, 22,  3,  3,  3,  2,  0],
    [3,  8, 18, 24,  1,  0,  2,  2,  5],
    [7, 20, 15, 26,  1,  0,  2,  2,  1],
    [2, 20,  7, 35,  0,  1,  2,  2,  5],
    [2, 16, 12, 28,  0,  1,  2,  2,  5],
    [2, 16,  5, 39,  0,  1,  2,  2,  2],
    [3,  6, 11,  8,  0,  1,  2,  2,  2],
    [7,  6, 15, 44,  1,  0,  2,  2,  4],
    [2, 18, 17, 41,  3,  3,  3,  2,  0],
    [2, 18, 17, 41,  3,  3,  3,  2,  0],
    [2, 16,  7,  6,  3,  3,  3,  2,  0],
    [1, 13,  2, 46,  3,  3,  3,  2,  0],
    [0, 14,  5, 41,  3,  3,  3,  2,  0],
    [2, 24, 19,  0,  3,  3,  3,  2,  0],
    [2, 14,  3, 35,  3,  3,  3,  2,  0],
    [6, 19,  7,  5,  0,  2,  1,  2,  2],
    [5,  6, 11, 44,  3,  3,  3,  2,  0],
    [7, 16, 21, 21,  3,  3,  3,  2,  0],
    [2, 19,  7, 44,  3,  3,  3,  2,  0],
    [2, 24, 18, 33,  1,  0,  2,  1,  4],
    [2, 16,  8, 44,  0,  2,  1,  2,  1],
    [3,  2,  5, 15,  0,  1,  2,  2,  2],
    [2, 18, 17, 41,  3,  3,  3,  2,  0],
    [2,  4, 15, 47,  0,  1,  2,  2,  2],
    [7, 13, 15, 25,  0,  1,  2,  2,  1],
    [1, 19, 10, 15,  3,  3,  3,  2,  0],
    [2, 13,  5, 44,  0,  1,  2,  1,  2],
    [5, 11, 18, 20,  3,  3,  3,  2,  0],
    [7,  9,  5, 40,  0,  1,  2,  1,  4],
    [3,  6, 16, 38,  3,  3,  3,  2,  0],
    [2, 24, 22, 12,  0,  1,  2,  2,  3],
    [5, 18, 17, 41,  3,  3,  3,  2,  0],
    [2, 18, 17, 41,  3,  3,  3,  2,  0],
    [2, 16, 15, 23,  0,  1,  2,  2,  5],
    [2, 13,  0, 25,  1,  0,  2,  2,  2],
    [2, 23, 15, 36,  3,  3,  3,  2,  0],
    [2, 25, 10,  2,  1,  0,  2,  2,  5],
    [2, 21,  7,  4,  1,  0,  2,  2,  1],
    [1, 18, 17, 41,  3,  3,  3,  2,  0],
    [2, 18, 17, 41,  3,  3,  3,  2,  0],
    [6,  9,  1,  0,  3,  3,  3,  2,  0],
    [1,  7, 20, 47,  3,  3,  3,  2,  0],
    [2, 25, 10,  7,  0,  1,  2,  2,  2],
    [7,  0,  4, 32,  1,  2,  0,  2,  5],
    [1, 12, 12, 15,  0,  1,  2,  3,  3],
    [2, 26, 15, 25,  0,  1,  2,  0,  5],
    [2, 20, 15, 19,  0,  1,  2,  2,  1],
    [4,  6,  9, 11,  2,  0,  1,  1,  4],
    [2, 13, 15, 42,  0,  2,  1,  2,  2],
    [3,  5, 21, 31,  0,  1,  2,  3,  5],
    [2, 13, 19, 33,  0,  2,  1,  2,  2],
    [1, 11, 10,  0,  0,  2,  1,  0,  2]
])

TEST_DATA_PREDICT = np.array([
    [2, 22, 14, 45,  2,  0,  1,  2,  5],
    [7, 13, 13, 19,  2,  0,  1,  2,  5],
    [5, 18, 19, 33,  0,  2,  1,  2,  2],
    [1, 11, 10,  0,  0,  2,  1,  0,  2]
])


def assert_cluster_splits_equal(array1, array2):

    def find_splits(x):
        return np.where(np.hstack((np.array([1]), np.diff(x))))[0]

    np.testing.assert_array_equal(find_splits(array1), find_splits(array2))


class TestKModes(unittest.TestCase):

    def test_pickle(self):
        obj = KModes()
        s = pickle.dumps(obj)
        assert_equal(type(pickle.loads(s)), obj.__class__)

    def test_kmodes_huang_soybean(self):
        kmodes_huang = KModes(n_clusters=4, n_init=2, init='Huang', verbose=2,
                              random_state=42)
        result = kmodes_huang.fit_predict(SOYBEAN)
        expected = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2,
                             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        assert_cluster_splits_equal(result, expected)
        self.assertTrue(result.dtype == np.dtype(np.uint16))

    def test_kmodes_huang_soybean_parallel(self):
        kmodes_huang = KModes(n_clusters=4, n_init=4, init='Huang', verbose=2,
                              random_state=42, n_jobs=4)
        result = kmodes_huang.fit_predict(SOYBEAN)
        expected = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2,
                             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        assert_cluster_splits_equal(result, expected)
        self.assertTrue(result.dtype == np.dtype(np.uint16))

    def test_kmodes_cao_soybean(self):
        kmodes_cao = KModes(n_clusters=4, init='Cao', verbose=2)
        result = kmodes_cao.fit_predict(SOYBEAN)
        expected = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        assert_cluster_splits_equal(result, expected)
        self.assertTrue(result.dtype == np.dtype(np.uint16))

    def test_kmodes_predict_soybean(self):
        kmodes_cao = KModes(n_clusters=4, init='Cao', verbose=2)
        kmodes_cao = kmodes_cao.fit(SOYBEAN)
        result = kmodes_cao.predict(SOYBEAN2)
        expected = np.array([2, 1, 3, 0])
        assert_cluster_splits_equal(result, expected)
        self.assertTrue(result.dtype == np.dtype(np.uint16))

    def test_kmodes_predict_unfitted(self):
        kmodes_cao = KModes(n_clusters=4, init='Cao', verbose=2)
        with self.assertRaises(AssertionError):
            kmodes_cao.predict(SOYBEAN)
        with self.assertRaises(AttributeError):
            kmodes_cao.cluster_centroids_

    def test_kmodes_random_soybean(self):
        kmodes_random = KModes(n_clusters=4, init='random', verbose=2,
                               random_state=42)
        result = kmodes_random.fit(SOYBEAN)
        self.assertIsInstance(result, KModes)

    def test_kmodes_init_soybean(self):
        init_vals = np.array(
            [[0, 1, 2, 1, 0, 3, 1, 1, 0, 2, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 1, 2,
              0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 1],
             [4, 0, 0, 1, 1, 1, 3, 1, 1, 1, 1, 1, 0, 2, 2, 0, 0, 0, 1, 1, 0, 3,
              0, 0, 0, 2, 1, 0, 4, 0, 0, 0, 0, 0, 0],
             [3, 0, 2, 1, 0, 2, 0, 2, 1, 1, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 3, 0,
              1, 1, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0],
             [3, 0, 2, 0, 1, 3, 1, 2, 0, 1, 1, 0, 0, 2, 2, 0, 0, 0, 1, 1, 1, 1,
              0, 1, 1, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0]])
        kmodes_init = KModes(n_clusters=4, init=init_vals, verbose=2)
        result = kmodes_init.fit_predict(SOYBEAN)
        expected = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        assert_cluster_splits_equal(result, expected)

        # 5 initial centroids, 4 n_clusters
        init_vals = np.array(
            [[0, 1],
             [4, 0],
             [4, 0],
             [3, 0],
             [3, 0]])
        kmodes_init = KModes(n_clusters=4, init=init_vals, verbose=2)
        with self.assertRaises(AssertionError):
            kmodes_init.fit(SOYBEAN)

        # wrong number of attributes
        init_vals = np.array(
            [0, 1, 2, 3])
        kmodes_init = KModes(n_clusters=4, init=init_vals, verbose=2)
        with self.assertRaises(AssertionError):
            kmodes_init.fit(SOYBEAN)

    def test_kmodes_empty_init_cluster_soybean(self):
        # Check if the clustering does not crash in case of an empty cluster.
        init_vals = np.array(
            [[0, 1, 2, 1, 0, 3, 1, 1, 0, 2, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 1, 2,
              0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 1],
             [4, 0, 0, 1, 1, 1, 3, 1, 1, 1, 1, 1, 0, 2, 2, 0, 0, 0, 1, 1, 0, 3,
              0, 0, 0, 2, 1, 0, 4, 0, 0, 0, 0, 0, 0],
             [3, 0, 2, 1, 0, 2, 0, 2, 1, 1, 1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 3, 0,
              1, 1, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0],
             [3, 0, 2, 0, 1, 3, 1, 2, 0, 1, 1, 0, 0, 2, 2, 0, 0, 0, 1, 1, 1, 1,
              0, 1, 1, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0]])
        kmodes_init = KModes(n_clusters=4, init=init_vals, verbose=2)
        result = kmodes_init.fit(SOYBEAN)
        self.assertIsInstance(result, KModes)

    def test_kmodes_empty_init_cluster_edge_case(self):
        # Edge case from: https://github.com/nicodv/kmodes/issues/106,
        # due to negative values in all-integer data.
        init_vals = np.array([
            [14, 0, 16, 0, -1, -1, 158, 115],
            [2, 0, 3, 3, 127, 105, 295, 197],
            [10, 2, 12, 3, 136, 20, 77, 42],
            [2, 0, 3, 4, 127, 55, 150, 63],
            [1, 0, 21, 5, 39, -1, 124, 90],
            [17, 2, 12, 3, 22, 175, 242, 164],
            [5, 1, 7, -1, -1, -1, 69, 38],
            [3, 3, 6, -1, -1, -1, 267, 175],
            [1, 0, 21, 4, 71, -1, 276, 196],
            [11, 2, 12, 5, -1, -1, 209, 148],
            [2, 0, 3, 5, 127, 105, 375, 263],
            [2, 0, 3, 4, 28, 105, 16, 8],
            [13, 2, 12, -1, -1, -1, 263, 187],
            [6, 2, 6, 4, 21, 20, 370, 256],
            [10, 2, 12, 3, 136, 137, 59, 31]
        ])
        data = np.hstack((init_vals, init_vals))
        kmodes_init = KModes(n_clusters=15, init='Huang', verbose=2)
        kmodes_init.fit_predict(data)
        kmodes_init.cluster_centroids_

    def test_kmodes_unknowninit_soybean(self):
        with self.assertRaises(NotImplementedError):
            KModes(n_clusters=4, init='nonsense', verbose=2).fit(SOYBEAN)

    def test_kmodes_nunique_nclusters(self):
        data = np.array([
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 2],
            [0, 2],
            [0, 2]
        ])
        kmodes_cao = KModes(n_clusters=6, init='Cao', verbose=2,
                            random_state=42)
        result = kmodes_cao.fit_predict(data, categorical=[1])
        expected = np.array([0, 0, 0, 1, 1, 1])
        assert_cluster_splits_equal(result, expected)
        np.testing.assert_array_equal(kmodes_cao.cluster_centroids_,
                                      np.array([[0, 2],
                                                [0, 1]]))

    def test_kmodes_huang_soybean_ng(self):
        kmodes_huang = KModes(n_clusters=4, n_init=2, init='Huang', verbose=2,
                              cat_dissim=ng_dissim, random_state=42)
        result = kmodes_huang.fit_predict(SOYBEAN)
        expected = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
                             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        assert_cluster_splits_equal(result, expected)
        self.assertTrue(result.dtype == np.dtype(np.uint16))

    def test_kmodes_cao_soybean_ng(self):
        kmodes_cao = KModes(n_clusters=4, init='Cao', verbose=2,
                            cat_dissim=ng_dissim)
        result = kmodes_cao.fit_predict(SOYBEAN)
        expected = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        assert_cluster_splits_equal(result, expected)
        self.assertTrue(result.dtype == np.dtype(np.uint16))

    def test_kmodes_predict_soybean_ng(self):
        kmodes_cao = KModes(n_clusters=4, init='Cao', verbose=2,
                            cat_dissim=ng_dissim)
        kmodes_cao = kmodes_cao.fit(SOYBEAN)
        result = kmodes_cao.predict(SOYBEAN2)
        expected = np.array([2, 1, 3, 0])
        assert_cluster_splits_equal(result, expected)
        self.assertTrue(result.dtype == np.dtype(np.uint16))

    def test_kmodes_nunique_nclusters_ng(self):
        data = np.array([
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 2],
            [0, 2],
            [0, 2]
        ])
        kmodes_cao = KModes(n_clusters=6, init='Cao', verbose=2,
                            cat_dissim=ng_dissim, random_state=42)
        result = kmodes_cao.fit_predict(data, categorical=[1])
        expected = np.array([0, 0, 0, 1, 1, 1])
        assert_cluster_splits_equal(result, expected)
        np.testing.assert_array_equal(kmodes_cao.cluster_centroids_,
                                      np.array([[0, 2],
                                                [0, 1]]))

    def test_kmodes_huang_soybean_jaccard_dissim_binary(self):
        kmodes_huang = KModes(n_clusters=4, n_init=2, init='Huang', verbose=2,
                              cat_dissim=jaccard_dissim_binary, random_state=42)
        # binary encoded variables are required
        bin_variables = SOYBEAN.astype(bool).astype(int)
        result = kmodes_huang.fit_predict(bin_variables)
        expected = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 3, 1, 1, 3, 3, 1, 1, 1, 1, 3, 1, 1, 3, 1, 3, 3, 1, 3,
                             3, 3, 1, 1, 3, 1, 3, 1, 1])
        assert_cluster_splits_equal(result, expected)
        self.assertTrue(result.dtype == np.dtype(np.uint16))

    def test_kmodes_cao_soybean_jaccard_dissim_binary(self):
        kmodes_Cao = KModes(n_clusters=4, n_init=2, init='Cao', verbose=2,
                            cat_dissim=jaccard_dissim_binary, random_state=42)
        # binary encoded variables are required
        bin_variables = SOYBEAN.astype(bool).astype(int)
        result = kmodes_Cao.fit_predict(bin_variables)
        expected = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0])

        assert_cluster_splits_equal(result, expected)
        self.assertTrue(result.dtype == np.dtype(np.uint16))

    def test_kmodes_predict_soybean_jaccard_dissim_binary(self):
        kmodes_huang = KModes(n_clusters=4, n_init=2, init='Huang', verbose=2,
                              cat_dissim=jaccard_dissim_binary, random_state=42)
        # binary encoded variables are required
        bin_variables = SOYBEAN.astype(bool).astype(int)
        kmodes_huang = kmodes_huang.fit(bin_variables)
        # binary encoded variables required for prediction as well
        bin_variables_pred = SOYBEAN2.astype(bool).astype(int)
        result = kmodes_huang.fit_predict(bin_variables_pred)
        expected = np.array([0, 1, 2, 3])
        assert_cluster_splits_equal(result, expected)
        self.assertTrue(result.dtype == np.dtype(np.uint16))

    def test_kmodes_huang_soybean_jaccard_dissim_label(self):
        kmodes_huang = KModes(n_clusters=4, n_init=2, init='Huang', verbose=2,
                              cat_dissim=jaccard_dissim_label, random_state=42)
        result = kmodes_huang.fit_predict(TEST_DATA)
        expected = np.array([3, 3, 2, 1, 1, 3, 3, 3, 3, 3, 0, 2, 2, 0, 0, 3, 3, 0, 0,
                             0, 2, 2, 0, 3, 2, 3, 2, 2, 0, 1, 1, 0, 1, 1, 0, 2, 3, 3,
                             3, 2, 2, 0, 0, 2, 1, 0, 0, 0, 2, 3, 0, 0, 2, 3, 2, 0, 2,
                             2, 2, 3, 0, 3, 2, 2, 0, 0, 3, 2, 1, 3, 2, 0, 0, 2, 2, 2,
                             3, 2, 2, 2, 2, 1, 3, 2, 2])
        assert_cluster_splits_equal(result, expected)
        self.assertTrue(result.dtype == np.dtype(np.uint16))

    def test_kmodes_cao_soybean_jaccard_dissim_label(self):
        kmodes_huang = KModes(n_clusters=4, n_init=2, init='Cao', verbose=2,
                              cat_dissim=jaccard_dissim_label, random_state=42)
        result = kmodes_huang.fit_predict(TEST_DATA)
        expected = np.array([3, 3, 1, 0, 0, 1, 1, 3, 2, 3, 0, 3, 2, 0, 0, 3, 3, 0,
                             0, 0, 1, 1, 0, 2, 1, 1, 2, 1, 0, 0, 0, 0, 0, 2, 0, 1,
                             3, 1, 1, 2, 2, 0, 0, 2, 0, 0, 0, 0, 3, 2, 2, 2, 0, 1,
                             1, 0, 1, 1, 1, 3, 0, 3, 2, 0, 0, 0, 1, 1, 0, 1, 1, 0,
                             0, 2, 2, 1, 3, 1, 1, 3, 1, 1, 3, 3, 1])

        assert_cluster_splits_equal(result, expected)
        self.assertTrue(result.dtype == np.dtype(np.uint16))

    def test_kmodes_predict_soybean_jaccard_dissim_label(self):
        kmodes_huang = KModes(n_clusters=4, n_init=2, init='Huang', verbose=2,
                              cat_dissim=jaccard_dissim_label, random_state=42)
        kmodes_huang = kmodes_huang.fit(TEST_DATA)
        result = kmodes_huang.fit_predict(TEST_DATA_PREDICT)
        expected = np.array([1, 0, 1, 2])
        assert_cluster_splits_equal(result, expected)
        self.assertTrue(result.dtype == np.dtype(np.uint16))

    def test_kmodes_ninit(self):
        kmodes = KModes(n_init=10, init='Huang')
        self.assertEqual(kmodes.n_init, 10)
        kmodes = KModes(n_init=10)
        self.assertEqual(kmodes.n_init, 1)
        kmodes = KModes(n_init=10, init=np.array([1, 1]))
        self.assertEqual(kmodes.n_init, 1)

    def test_kmodes_epoch_costs(self):
        kmodes = KModes(n_clusters=4, init='Cao', random_state=42)
        kmodes.fit(SOYBEAN)
        self.assertEqual(kmodes.epoch_costs_, [206.0, 204.0, 199.0, 199.0])
