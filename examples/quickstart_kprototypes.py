"""Cluster mixed rows: python examples/quickstart_kprototypes.py."""

import numpy as np

from kmodes.kprototypes import KPrototypes


def main():
    # Columns: a numerical measurement, a category, another measurement.
    data = np.array([
        [1.0, 'red', 10.0],
        [1.2, 'red', 11.0],
        [0.8, 'red', 9.0],
        [8.0, 'blue', 80.0],
        [8.2, 'blue', 81.0],
        [7.8, 'blue', 79.0],
    ], dtype=object)
    categorical = [1]
    model = KPrototypes(n_clusters=2, init='Cao', n_init=5,
                        gamma=None, verbose=0, random_state=42)
    labels = model.fit_predict(data, categorical=categorical)
    new_rows = np.array([[1.1, 'red', 10.0], [8.1, 'blue', 80.0]], dtype=object)
    predictions = model.predict(new_rows, categorical=categorical)

    assert np.all(labels[:3] == labels[0])
    assert np.all(labels[3:] == labels[3])
    assert labels[0] != labels[3]
    np.testing.assert_array_equal(predictions, labels[[0, 3]])
    # Centroid columns are numerical first, categorical last.
    centers = model.cluster_centroids_
    order = np.argsort(centers[:, 0].astype(float))
    np.testing.assert_allclose(centers[order, :2].astype(float), [[1, 10], [8, 80]])
    np.testing.assert_array_equal(centers[order, 2], ['red', 'blue'])
    np.testing.assert_allclose(model.cost_, 4.16)
    assert model.gamma > 0

    print('Labels:', labels)
    print('Centroids (numerical, then categorical):', centers)
    print('Cost:', model.cost_)
    print('Gamma:', model.gamma)
    print('New row predictions:', predictions)


if __name__ == '__main__':
    main()
