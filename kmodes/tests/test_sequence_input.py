"""Sequence inputs follow the documented array-like estimator interface."""

from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes


def _data_and_model(kind):
    if kind == 'kmodes':
        data = [['a', 'x'], ['a', 'y'], ['b', 'z'],
                ['b', 'x'], ['a', 'x'], ['b', 'z']]
        model = KModes(n_clusters=2, n_init=1, random_state=7)
        kwargs = {}
    else:
        data = [[0., 'a'], [1., 'a'], [10., 'b'],
                [11., 'b'], [.25, 'a'], [10.5, 'b']]
        model = KPrototypes(n_clusters=2, n_init=1, random_state=7)
        kwargs = {'categorical': [1]}
    return data, model, kwargs


@pytest.mark.parametrize('kind', ['kmodes', 'kprototypes'])
@pytest.mark.parametrize('container', [list, tuple])
@pytest.mark.parametrize('method', ['fit', 'fit_predict', 'predict'])
def test_sequence_input(kind, container, method):
    data, model, kwargs = _data_and_model(kind)
    _, reference, _ = _data_and_model(kind)
    array = np.asarray(data)
    sequence = container(container(row) for row in data)
    original = deepcopy(sequence)
    reference.fit(array, **kwargs)

    if method == 'predict':
        model.fit(array, **kwargs)
        result = model.predict(sequence, **kwargs)
    elif method == 'fit_predict':
        result = model.fit_predict(sequence, **kwargs)
    else:
        assert model.fit(sequence, **kwargs) is model
        result = model.labels_

    assert_array_equal(result, reference.predict(array, **kwargs))
    assert_array_equal(model.cluster_centroids_, reference.cluster_centroids_)
    assert model.cost_ == reference.cost_
    assert sequence == original


@pytest.mark.parametrize('kind', ['kmodes', 'kprototypes'])
@pytest.mark.parametrize('container', [list, tuple])
def test_sequence_input_with_sample_weight(kind, container):
    data, model, kwargs = _data_and_model(kind)
    _, reference, _ = _data_and_model(kind)
    weights = [1, 2, 1, 2, 1, 2]
    sequence = container(container(row) for row in data)
    original = deepcopy(sequence)
    reference.fit(np.asarray(data), sample_weight=weights, **kwargs)

    model.fit(sequence, sample_weight=weights, **kwargs)

    assert_array_equal(model.labels_, reference.labels_)
    assert_array_equal(model.cluster_centroids_, reference.cluster_centroids_)
    assert model.cost_ == reference.cost_
    assert sequence == original
