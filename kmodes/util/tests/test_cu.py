import numpy as np
from kmodes.util.category_utility import category_utiity

def test_cu_example():
    x = np.array([
        ['Red','Short', 'True', 0],
        ['Red','Long', 'False', 0],
        ['Blue','Medium', 'True', 1],
        ['Green','Medium', 'True', 1],
        ['Green','Medium', 'False', 1],
    ])
    print x
    category_utiity(x)

test_cu_example()