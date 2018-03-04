import numpy as np
from sklearn import datasets

from kmodes.kprototypes import KPrototypes

iris = datasets.load_iris()

data = np.c_[iris['data'], iris['target']]

kp = KPrototypes(n_clusters=3, init='Huang', n_init=1, verbose=True)
kp.fit_predict(data, categorical=[4])

print(kp.cluster_centroids_)
print(kp.labels_)
