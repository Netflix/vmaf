__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import sys
import os

def get_stdout_logger():
    import logging
    logger = logging.getLogger()
    handler = logging.StreamHandler(stream=sys.stdout)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger

def close_logger(logger):
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

def get_file_name_without_extension(path):
    """

    >>> get_file_name_without_extension('yuv/src01_hrc01.yuv')
    'src01_hrc01'
    >>> get_file_name_without_extension('yuv/src01_hrc01')
    'src01_hrc01'
    >>> get_file_name_without_extension('abc/xyz/src01_hrc01.yuv')
    'src01_hrc01'

    """
    return os.path.splitext(path.split("/")[-1])[0]

def get_file_name_with_extension(path):
    """

    >>> get_file_name_with_extension('yuv/src01_hrc01.yuv')
    'src01_hrc01.yuv'
    >>> get_file_name_with_extension('src01_hrc01.yuv')
    'src01_hrc01.yuv'
    >>> get_file_name_with_extension('abc/xyz/src01_hrc01.yuv')
    'src01_hrc01.yuv'

    """
    return path.split("/")[-1]

def get_dir_without_last_slash(path):
    """

    >>> get_dir_without_last_slash('abc/src01_hrc01.yuv')
    'abc'
    >>> get_dir_without_last_slash('src01_hrc01.yuv')
    ''
    >>> get_dir_without_last_slash('abc/xyz/src01_hrc01.yuv')
    'abc/xyz'

    """
    return "/".join(path.split("/")[:-1])

def make_parent_dirs_if_nonexist(path):
    dst_dir = get_dir_without_last_slash(path)
    # create dir if not exist yet
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)

def delete_dir_if_exists(dir):
    if os.path.isdir(dir):
        os.rmdir(dir)

def get_unique_str_from_recursive_dict(d):
    """
    String representation with sorted keys and values
    >>> get_unique_str_from_recursive_dict({'a':1, 'b':2, 'c':{'x':'0', 'y':'1'}})
    '{"a": 1, "b": 2, "c": {"x": "0", "y": "1"}}'
    >>> get_unique_str_from_recursive_dict({'a':1, 'c':2, 'b':{'y':'1', 'x':'0', }})
    '{"a": 1, "b": {"x": "0", "y": "1"}, "c": 2}'
    """
    from collections import OrderedDict
    import json
    def to_ordered_dict_recursively(d):
        if isinstance(d, dict):
            return OrderedDict(map(
                                    lambda (k,v): (to_ordered_dict_recursively(k),
                                                   to_ordered_dict_recursively(v)),
                                    sorted(d.items())
                                   )
                              )
        else:
            return d
    return json.dumps(to_ordered_dict_recursively(d))

def indices(a, func):
    """
    Get indices of elements in an array which satisfies func
    >>> indices([1, 2, 3, 4], lambda x: x>2)
    [2, 3]
    >>> indices([1, 2, 3, 4], lambda x: x==2.5)
    []
    >>> indices([1, 2, 3, 4], lambda x: x>1 and x<=3)
    [1, 2]
    >>> indices([1, 2, 3, 4], lambda x: x in [2, 4])
    [1, 3]
    """
    return [i for (i, val) in enumerate(a) if func(val)]

def import_python_file(filepath):
    """
    Import a python file as a module.
    :param filepath:
    :return:
    """
    import imp
    filename = get_file_name_without_extension(filepath)
    ret = imp.load_source(filename, filepath)

    return ret

def make_absolute_path(path, current_dir):
    '''

    >>> make_absolute_path('abc/cde.fg', '/xyz/')
    '/xyz/abc/cde.fg'
    >>> make_absolute_path('/abc/cde.fg', '/xyz/')
    '/abc/cde.fg'

    '''
    if path[0] == '/':
        return path
    else:
        return current_dir + path

if __name__ == '__main__':
    import doctest
    doctest.testmod()


