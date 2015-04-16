
import numpy as np
from kmodes import kmodes

# stocks with their market caps, sectors and countries
syms = np.genfromtxt('proto.csv', dtype=str, delimiter=',')[:, 0]
xnum = np.genfromtxt('proto.csv', dtype=float, delimiter=',')[:, 1]
xnum = np.atleast_2d(xnum).T
xcat = np.genfromtxt('proto.csv', dtype=str, delimiter=',')[:, 2:]

kproto = kmodes.KPrototypes(n_clusters=4, init='Cao', verbose=2)
clusters = kproto.fit_predict([xnum, xcat])

for s, c in zip(syms, clusters):
    print("Symbol: {}, cluster:{}".format(s, c[0]))
