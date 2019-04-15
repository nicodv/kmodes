import inspect


class _named_check(object):  # noqa
    """Wraps a check to show a useful description
    Parameters
    ----------
    check : function
        Must have ``__name__`` and ``__call__``
    arg_text : str
        A summary of arguments to the check
    """
    # Setting the description on the function itself can give incorrect results
    # in failing tests
    def __init__(self, check, arg_text):
        self.check = check
        self.description = ("{0[1]}.{0[3]}:{1.__name__}({2})".format(
            inspect.stack()[1], check, arg_text))

    def __call__(self, *args, **kwargs):
        return self.check(*args, **kwargs)
