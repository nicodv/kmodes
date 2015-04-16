
import numpy as np
from kmodes import kmodes

# reproduce results on small soybean data set
x = np.genfromtxt('soybean.csv', dtype=int, delimiter=',')[:, :-1]
y = np.genfromtxt('soybean.csv', dtype=str, delimiter=',', usecols=35)

kmodes_huang = kmodes.KModes(n_clusters=4, init='Huang', verbose=0)
kmodes_huang.fit_predict(x)
kmodes_cao = kmodes.KModes(n_clusters=4, init='Cao', verbose=1)
kmodes_cao.fit_predict(x)

for result in (kmodes_huang, kmodes_cao):
    classtable = np.zeros((4, 4), dtype=int)
    for ii, _ in enumerate(y):
        classtable[int(y[ii][-1]) - 1, result.labels_[ii]] += 1

    print("\n")
    print("    | Cl. 1 | Cl. 2 | Cl. 3 | Cl. 4 |")
    print("----|-------|-------|-------|-------|")
    for ii in range(4):
        prargs = tuple([ii + 1] + list(classtable[ii, :]))
        print(" D{0} |    {1:>2} |    {2:>2} |    {3:>2} |    {4:>2} |"
              .format(*prargs))
