"""
General sklearn tests for the estimators in kmodes.
"""

from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes

from sklearn.utils.testing import assert_false
from sklearn.utils.testing import assert_greater

from sklearn.utils.estimator_checks import (
    check_parameters_default_constructible,
    check_estimator_sparse_data,
    check_estimators_dtypes,
    check_estimators_empty_data_messages,
    check_estimators_nan_inf,
    check_estimators_overwrite_params,
    check_fit_score_takes_y,
    check_pipeline_consistency)

all_estimators = lambda: (('kmodes', KModes), ('kprototypes', KPrototypes))


def test_all_estimator_no_base_class():
    # test that all_estimators doesn't find abstract classes.
    for name, Estimator in all_estimators():
        msg = ("Base estimators such as {0} should not be included"
               " in all_estimators").format(name)
        assert_false(name.lower().startswith('base'), msg=msg)


def test_all_estimators():
    # Test that estimators are default-constructible, clonable
    # and have working repr.
    estimators = all_estimators()

    # Meta sanity-check to make sure that the estimator introspection runs
    # properly
    assert_greater(len(estimators), 0)

    for name, Estimator in estimators:
        # some can just not be sensibly default constructed
        yield check_parameters_default_constructible, name, Estimator


def test_non_meta_estimators():
    # input validation etc for non-meta estimators
    estimators = all_estimators()
    for name, Estimator in estimators:
        if name != 'kprototypes':
            yield check_estimators_dtypes, name, Estimator
            yield check_fit_score_takes_y, name, Estimator

            # Check that all estimator yield informative messages when
            # trained on empty datasets
            yield check_estimators_empty_data_messages, name, Estimator

            yield check_pipeline_consistency, name, Estimator

            if name not in ['Imputer']:
                # Test that all estimators check their input for NaN's and infs
                yield check_estimators_nan_inf, name, Estimator

            yield check_estimators_overwrite_params, name, Estimator
        yield check_estimator_sparse_data, name, Estimator
