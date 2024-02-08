from collections.abc import Hashable
from functools import partial
import os

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"

import warnings
import json
import hashlib
import sys


def deprecated(func):
    """
    Mark a function as deprecated.
    It will result in a warning being emmitted when the function is used.
    """

    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning) # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning, stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning) # reset filter
        return func(*args, **kwargs)

    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func


def persist(original_func):
    """
    Cache returned value of function in a function. Useful when calling functions
    recursively, especially in dynamic programming where lots of returned values
    can be reused.
    """

    cache = {}

    def new_func(*args):
        h = hashlib.sha1(str(original_func.__name__) + str(args)).hexdigest()
        if h not in cache:
            cache[h] = original_func(*args)
        return cache[h]

    return new_func


def dummy(func):
    """ Dummy decorator. """
    return func


class memoized(object):
    """ Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).

    memoized is similar to persist, but if applied to
    class methods, persist will cache on a per-class basis, while memoized
    will cache on a per-object basis.

    Taken from: https://wiki.python.org/moin/PythonDecoratorLibrary#Memoize
    """

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, Hashable):
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self):
        """ Return the function's docstring. """
        return self.func.__doc__

    def __get__(self, obj, objtype):
        """ Support instance methods. """
        return partial(self.__call__, obj)


def persist_to_file(file_name):
    """
    Cache (or persist) returned value of function in a json file .
    """

    def decorator(original_func):

        if not os.path.exists(file_name):
            cache = {}
        else:
            try:
                cache = json.load(open(file_name, 'rt'))
            except (IOError, ValueError):
                sys.exit(1)

        def new_func(*args):
            h = hashlib.sha1(str(original_func.__name__) + str(args)).hexdigest()
            if h not in cache:
                cache[h] = original_func(*args)
                file_dir = os.path.dirname(file_name)
                os.makedirs(file_dir, exist_ok=True)
                json.dump(cache, open(file_name, 'wt'))
            return cache[h]

        return new_func

    return decorator


def persist_to_dir(dir_name):
    """
    Cache (or persist) returned value of function in a directory of files.
    """

    def decorator(original_func):

        def new_func(*args):
            h = hashlib.sha1(str(original_func.__name__) + str(args)).hexdigest()
            file_name = os.path.join(dir_name, h)
            if not os.path.exists(file_name):
                os.makedirs(dir_name, exist_ok=True)
                res = original_func(*args)
                json.dump(res, open(file_name, 'wt'))
            else:
                res = json.load(open(file_name, 'rt'))
            return res

        return new_func

    return decorator


def override(interface_class):
    def overrider(method):
        assert(method.__name__ in dir(interface_class)), \
            f"{method.__name__} does not override any method in {interface_class.__name__}"
        return method
    return overrider


class change_repr(object):
    def __init__(self, functor):
        self.functor = functor
        #  lets copy some key attributes from the original function
        self.__name__ = functor.__name__
        self.__doc__ = functor.__doc__

    def __call__(self, *args, **kwargs):
        return self.functor(*args, **kwargs)

    def __repr__(self):
        return self.functor.__name__
