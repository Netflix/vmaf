import os

__copyright__ = "Copyright 2016-2017, Netflix, Inc."
__license__ = "Apache, Version 2.0"

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
        warnings.simplefilter('always', DeprecationWarning) #turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning, stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning) #reset filter
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
                json.dump(cache, open(file_name, 'wt'))
            return cache[h]

        return new_func

    return decorator