
from nose.tools import assert_equal
from kmodes.kmodes import get_max_value_key


def test_mode_from_dict():
    max_key = get_max_value_key({'a': 3, 'b': 10, 'c': -1, 'd': 9.9})
    assert_equal('b', max_key)
