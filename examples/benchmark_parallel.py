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
# max no. of cores to run on
C = 4

data = np.random.randint(1, 1000, (max(N_kproto, N_kmodes), M))


def _kprototypes(k, n_init, n_jobs, seed):
    KPrototypes(n_clusters=k, init='Huang', n_init=n_init, n_jobs=n_jobs,
                random_state=seed) \
        .fit(data[:N_kproto, :], categorical=list(range(M - MN, M)))


def _kmodes(k, n_init, n_jobs, seed):
    KModes(n_clusters=k, init='Huang', n_init=n_init, n_jobs=n_jobs,
           random_state=seed) \
        .fit(data[:N_kmodes, :])


def run(task, stop):
    # Draw a seed, so both jobs converge in an equal amount of iterations
    seed = np.random.randint(np.iinfo(np.int32).max)
    baseline = 0

    for n_jobs in range(1, stop + 1):
        print('Starting runs on {} core(s)'.format(n_jobs))
        t_start = time.time()
        task(K, stop, n_jobs, seed)
        runtime = time.time() - t_start

        if n_jobs == 1:
            baseline = runtime
            print('Finished {} runs on 1 core in {:.2f} seconds'.format(stop, runtime))
        else:
            print('Finished {} runs on {} cores in {:.2f} seconds, a {:.1f}x '
                  'speed-up'.format(stop, n_jobs, runtime, baseline / runtime))


if __name__ == '__main__':
    print(f"Running K-Prototypes on 1 to {C} cores for {C} initialization tries "
          f"of {K} clusters with {N_kproto} points of {M} features")
    run(_kprototypes, C)

    print(f"\nRunning K-Modes on 1 to {C} cores for {C} initialization tries "
          f"of {K} clusters with {N_kmodes} points of {M} features")
    run(_kmodes, C)
