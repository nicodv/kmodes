"""
General sklearn tests for the estimators in kmodes.
"""
from sklearn.utils.testing import assert_greater
from sklearn.utils.estimator_checks import (
    _yield_all_checks,
    check_parameters_default_constructible
)

from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
from kmodes.util.testing import _named_check

all_estimators = lambda: (('kmodes', KModes), ('kprototypes', KPrototypes))

KMODES_INCLUDE_CHECKS = (
    'check_estimators_dtypes',
    'check_fit_score_takes_y',
    'check_sample_weights_pandas_series',
    'check_sample_weights_list',
    'check_sample_weights_invariance',
    'check_estimators_fit_returns_self',
    'check_complex_data',
    'check_estimators_empty_data_messages',
    'check_pipeline_consistency',
    'check_estimators_nan_inf',
    'check_estimators_overwrite_params',
    'check_estimator_sparse_data',
    'check_estimators_overwrite_params',
    'check_estimators_pickle',
    'check_fit2d_predict1d',
    'check_methods_subset_invariance',
    'check_fit2d_1sample',
    'check_fit2d_1feature',
    'check_fit1d',
    'check_get_params_invariance',
    'check_set_params',
    'check_dict_unchanged',
    'check_dont_overwrite_parameters',
    'check_fit_idempotent',
    'check_clusterer_compute_labels_predict',
    'check_estimators_partial_fit_n_features',
    'check_non_transformer_estimators_n_iter',
)

KPROTOTYPES_INCLUDE_CHECKS = (
    'check_sample_weights_pandas_series',
    'check_sample_weights_list',
    'check_sample_weights_invariance',
    'check_estimator_sparse_data',
    'check_get_params_invariance',
    'check_set_params',
    'check_clusterer_compute_labels_predict',
    'check_estimators_partial_fit_n_features',
)


def test_all_estimator_no_base_class():
    # test that all_estimators doesn't find abstract classes.
    for name, Estimator in all_estimators():
        msg = ("Base estimators such as {0} should not be included"
               " in all_estimators").format(name)
        assert not name.lower().startswith('base'), msg


def test_all_estimators():
    estimators = all_estimators()

    # Meta sanity-check to make sure that the estimator introspection runs
    # properly
    assert_greater(len(estimators), 0)

    for name, Estimator in estimators:
        # some can just not be sensibly default constructed
        yield (_named_check(check_parameters_default_constructible, name),
               name, Estimator)


def test_non_meta_estimators():
    for name, Estimator in all_estimators():
        if name == 'kmodes':
            relevant_checks = KMODES_INCLUDE_CHECKS
        elif name == 'kprototypes':
            relevant_checks = KPROTOTYPES_INCLUDE_CHECKS
        else:
            raise NotImplementedError
        estimator = Estimator()
        for check in _yield_all_checks(name, estimator):
            if hasattr(check, '__name__') and check.__name__ in relevant_checks:
                yield _named_check(check, name), name, estimator
