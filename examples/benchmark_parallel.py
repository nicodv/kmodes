import time

import numpy as np

from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes


# number of clusters
K = 20
# no. of points for K-Prototypes
N_kproto = int(1e4)
# no. of points for K-Modes
# We set this higher than K-Prototypes, because K-Modes converges much faster
# and we want to reduce the influence of overhead during the benchmark.
N_kmodes = int(1e5)
# no. of dimensions
M = 10
# no. of numerical dimensions
MN = 5

data = np.random.randint(1, 1000, (max(N_kproto, N_kmodes), M))


def kprototypes():
    # Draw a seed, so both jobs converge in an equal amount of iterations
    seed = np.random.randint(np.iinfo(np.int32).max)

    start = time.time()
    KPrototypes(n_clusters=K, init='Huang', n_init=4, verbose=2,
                random_state=seed) \
        .fit(data[:N_kproto, :], categorical=list(range(M - MN, M)))
    single = time.time() - start
    print('Finished 4 runs on 1 thread in {:.2f} seconds'.format(single))

    np.random.seed(seed)
    start = time.time()
    KPrototypes(n_clusters=K, init='Huang', n_init=4, n_jobs=4, verbose=2,
                random_state=seed) \
        .fit(data[:N_kproto, :], categorical=list(range(M - MN, M)))
    multi = time.time() - start
    print('Finished 4 runs on 4 threads in {:.2f} seconds'.format(multi))

    return single, multi


def kmodes():
    # Draw a seed, so both jobs converge in an equal amount of iterations
    seed = np.random.randint(np.iinfo(np.int32).max)

    start = time.time()
    KModes(n_clusters=K, init='Huang', n_init=4, verbose=2,
           random_state=seed).fit(data[:N_kmodes, :])
    single = time.time() - start
    print('Finished 4 runs on 1 thread in {:.2f} seconds'.format(single))

    start = time.time()
    KModes(n_clusters=K, init='Huang', n_init=4, n_jobs=4, verbose=2,
           random_state=seed).fit(data[:N_kmodes, :])
    multi = time.time() - start
    print('Finished 4 runs on 4 threads in {:.2f} seconds'.format(multi))

    return single, multi


if __name__ == '__main__':
    print('Starting K-Prototypes on 1 and on 4 threads for {} clusters with {}'
          ' points of {} features'.format(K, N_kproto, M))
    res_kproto = kprototypes()
    print('Starting K-Modes on 1 and on 4 threads for {} clusters with {}'
          ' points of {} features'.format(K, N_kmodes, M))
    res_kmodes = kmodes()
    print()
    print('K-Protoypes took {:.2f} s for 1 thread and {:.2f} s for 4 threads:'
          ' a {:.1f}x speed-up'.format(res_kproto[0], res_kproto[1],
                                       res_kproto[0] / res_kproto[1]))
    print('K-Modes took {:.2f} s for 1 thread and {:.2f} s for 4 threads:'
          ' a {:.1f}x speed-up'.format(res_kmodes[0], res_kmodes[1],
                                       res_kmodes[0] / res_kmodes[1]))


