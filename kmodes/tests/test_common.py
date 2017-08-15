"""
General sklearn tests for the estimators in kmodes.
"""

from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes

from sklearn.utils.testing import assert_false
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import _named_check

from sklearn.utils.estimator_checks import (
    _yield_all_checks,
    check_parameters_default_constructible)

all_estimators = lambda: (('kmodes', KModes), ('kprototypes', KPrototypes))


def test_all_estimator_no_base_class():
    # test that all_estimators doesn't find abstract classes.
    for name, Estimator in all_estimators():
        msg = ("Base estimators such as {0} should not be included"
               " in all_estimators").format(name)
        assert_false(name.lower().startswith('base'), msg=msg)


def test_all_estimators():
    # Test that estimators are default-constructible, cloneable
    # and have working repr.
    estimators = all_estimators()

    # Meta sanity-check to make sure that the estimator introspection runs
    # properly
    assert_greater(len(estimators), 0)

    for name, Estimator in estimators:
        # some can just not be sensibly default constructed
        yield (_named_check(check_parameters_default_constructible, name),
               name, Estimator)


def test_non_meta_estimators():
    # input validation etc for non-meta estimators
    estimators = all_estimators()
    for name, Estimator in estimators:
        estimator = Estimator()
        if name == 'kmodes':
            for check in _yield_all_checks(name, Estimator):
                # Skip these
                if check.__name__ not in ('check_clustering',
                                          'check_dtype_object'):
                    yield _named_check(check, name), name, estimator
        elif name == 'kprototypes':
            for check in _yield_all_checks(name, Estimator):
                # Only do these
                if check.__name__ in ('check_estimator_sparse_data',
                                      'check_clusterer_compute_labels_predict',
                                      'check_estimators_partial_fit_n_features'):
                    yield _named_check(check, name), name, estimator
