"""Cluster categorical rows: python examples/quickstart_kmodes.py."""

import numpy as np

from kmodes.kmodes import KModes


def main():
    data = np.array([
        ['red', 'small'],
        ['red', 'small'],
        ['red', 'medium'],
        ['blue', 'large'],
        ['blue', 'large'],
        ['blue', 'medium'],
    ])
    model = KModes(n_clusters=2, init='Huang', n_init=5,
                   verbose=0, random_state=42)
    labels = model.fit_predict(data)
    predictions = model.predict(np.array([['red', 'small'], ['blue', 'large']]))

    # Check the grouping without depending on the arbitrary cluster numbers.
    assert np.all(labels[:3] == labels[0])
    assert np.all(labels[3:] == labels[3])
    assert labels[0] != labels[3]
    np.testing.assert_array_equal(predictions, labels[[0, 3]])
    assert model.cluster_centroids_.shape == (2, 2)
    assert model.cost_ == 2

    print('Labels:', labels)
    print('Centroids:', model.cluster_centroids_)
    print('Cost:', model.cost_)
    print('New row predictions:', predictions)


if __name__ == '__main__':
    main()
