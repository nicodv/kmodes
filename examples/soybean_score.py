import numpy as np
from kmodes import kmodes
from kmodes.util import category_utility

x = np.genfromtxt('soybean.csv', dtype=int, delimiter=',')[:, :-1]
kmodes_huang = kmodes.KModes(n_clusters=4, init='Huang', verbose=1)
kmodes_huang.fit(x)
print type(x)
print x.shape
labels = kmodes_huang.labels_
x = np.hstack([x, kmodes_huang.labels_.reshape(47,1)])
print x.shape
print x
utility = category_utility.category_utiity(x)